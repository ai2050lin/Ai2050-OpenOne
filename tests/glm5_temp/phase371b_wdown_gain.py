"""
Phase 371b: W_down Gain Analysis — Why does scattered MLP activation produce 1D output?
================================================================================

Phase 371 found:
- DS7B L4 MLP writes 98.4% of PC1 (norm=115.6)
- But gate/up neurons are highly scattered (top-10 concentration < 0.03%)
- W_down is near-full-rank (Phase 369)

This test answers: How does W_down transform scattered activation into 1D output?

Key analysis:
1. Compute W_down's singular value structure — is there a dominant direction?
2. Project the MLP input (gate_times_up) through W_down and see which singular
   mode gets amplified most
3. Compare W_down's top singular vector direction with the PC1 direction
4. For Qwen3/GLM4, check if their W_down has similar structure

Models: deepseek7b, qwen3, glm4
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

TEST_PAIRS = TEST_PAIRS_42 + EXTRA_PAIRS

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


def load_mlp_weights(model, model_name, layer_idx):
    """Load MLP weight matrices, handling meta device and different architectures."""
    import glob
    from safetensors import safe_open
    
    layers = get_layers(model)
    mlp = layers[layer_idx].mlp
    
    # Try standard SwiGLU
    gate_proj = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    gate_up_proj = getattr(mlp, "gate_up_proj", None)  # GLM4
    
    W_gate = W_up = W_down = None
    
    if gate_proj is not None and up_proj is not None and down_proj is not None:
        # Check meta device
        try:
            W_gate = gate_proj.weight.detach().cpu().float().numpy()
            W_up = up_proj.weight.detach().cpu().float().numpy()
            W_down = down_proj.weight.detach().cpu().float().numpy()
        except:
            pass
    
    if W_gate is None:
        # Load from safetensors
        model_path = MODEL_CONFIGS[model_name]["path"]
        for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                for key in sf.keys():
                    if f"layers.{layer_idx}.mlp.gate_proj.weight" in key:
                        W_gate = sf.get_tensor(key).float().numpy()
                    elif f"layers.{layer_idx}.mlp.up_proj.weight" in key:
                        W_up = sf.get_tensor(key).float().numpy()
                    elif f"layers.{layer_idx}.mlp.down_proj.weight" in key:
                        W_down = sf.get_tensor(key).float().numpy()
                    elif f"layers.{layer_idx}.mlp.gate_up_proj.weight" in key:
                        full_w = sf.get_tensor(key).float().numpy()
                        half = full_w.shape[0] // 2
                        W_gate = full_w[:half]
                        W_up = full_w[half:]
                    elif f"layers.{layer_idx}.mlp.dense_h_to_4h.weight" in key:
                        full_w = sf.get_tensor(key).float().numpy()
                        half = full_w.shape[0] // 2
                        W_gate = full_w[:half]
                        W_up = full_w[half:]
                    elif f"layers.{layer_idx}.mlp.dense_4h_to_h.weight" in key:
                        W_down = sf.get_tensor(key).float().numpy()
            if W_gate is not None and W_down is not None:
                break
    
    return W_gate, W_up, W_down


def analyze_wdown_gain(model_name, model, n_layers, d_model):
    """
    Analyze W_down's gain structure and its relationship to the PC1 direction.
    
    Key questions:
    1. Does W_down have a dominant singular direction that aligns with PC1?
    2. How much of the MLP output variance is captured by W_down's top singular mode?
    3. Compare the gain distribution across layers
    """
    log("\n" + "="*60)
    log("W_down Gain Structure Analysis")
    log("="*60)
    
    if model_name == "deepseek7b":
        target_layers = [3, 4, 5, 6, 12, 18, 24]
    elif model_name == "qwen3":
        target_layers = [3, 4, 5, 8, 16, 28]
    else:
        target_layers = [3, 4, 5, 10, 20, 30]
    
    results = {}
    
    for l in target_layers:
        W_gate, W_up, W_down = load_mlp_weights(model, model_name, l)
        if W_down is None:
            log(f"  L{l}: Could not load W_down")
            continue
        
        d_model_local = W_down.shape[0]
        d_ff = W_down.shape[1]
        
        # SVD of W_down
        U_d, S_d, Vt_d = np.linalg.svd(W_down, full_matrices=False)
        
        # Singular value distribution
        total_sv_energy = np.sum(S_d**2)
        sv_explained = (S_d**2) / total_sv_energy
        top1_sv = sv_explained[0]
        top5_sv = np.sum(sv_explained[:5])
        top10_sv = np.sum(sv_explained[:10])
        top20_sv = np.sum(sv_explained[:20])
        
        # Effective rank of W_down
        eff_rank_95 = int(np.searchsorted(np.cumsum(sv_explained), 0.95) + 1)
        eff_rank_99 = int(np.searchsorted(np.cumsum(sv_explained), 0.99) + 1)
        
        # Gain: how much does W_down amplify in its top singular direction vs average?
        # gain_top1 = S_d[0] / mean(S_d)
        # gain_ratio = S_d[0] / S_d[-1] (condition number indicator)
        mean_sv = np.mean(S_d)
        gain_top1_over_mean = S_d[0] / mean_sv
        condition_number = S_d[0] / (S_d[-1] + 1e-10)
        
        # W_down's top output direction (left singular vector u1)
        u1_down = U_d[:, 0]  # (d_model,) — the direction W_down amplifies most
        
        # Now we need to compare u1_down with the actual PC1 direction
        # PC1 direction is from the Δh_output PCA at this layer
        # We need to collect Δh for this comparison
        # For now, store u1_down and compare later
        
        results[str(l)] = {
            "d_ff": int(d_ff),
            "d_model": int(d_model_local),
            "top1_sv_explained": float(top1_sv),
            "top5_sv_explained": float(top5_sv),
            "top10_sv_explained": float(top10_sv),
            "top20_sv_explained": float(top20_sv),
            "eff_rank_95": eff_rank_95,
            "eff_rank_99": eff_rank_99,
            "gain_top1_over_mean": float(gain_top1_over_mean),
            "condition_number": float(condition_number),
            "u1_down_norm": float(np.linalg.norm(u1_down)),
        }
        
        log(f"  L{l}: W_down {W_down.shape}, top1_sv={top1_sv:.4f}, top5={top5_sv:.4f}, "
            f"top10={top10_sv:.4f}, eff_rank_95={eff_rank_95}, gain={gain_top1_over_mean:.2f}, "
            f"cond={condition_number:.1f}")
        
        # Save u1 for later comparison
        results[str(l)]["u1_down"] = u1_down.tolist()
    
    return results


def analyze_wdown_pc1_alignment(model, tokenizer, device, model_name, n_layers, d_model):
    """
    Compare W_down's top output direction (u1) with the actual PC1 of Δh.
    
    If u1_down aligns with PC1 of Δh, it means W_down's highest-gain direction
    is what creates the 1D collapse.
    """
    log("\n" + "="*60)
    log("W_down Top Direction vs Δh PC1 Alignment")
    log("="*60)
    
    n_pairs = len(TEST_PAIRS)
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    
    if model_name == "deepseek7b":
        target_layers = [3, 4, 5, 6, 12, 18, 24]
    elif model_name == "qwen3":
        target_layers = [3, 4, 5, 8, 16, 28]
    else:
        target_layers = [3, 4, 5, 10, 20, 30]
    
    results = {}
    
    for l in target_layers:
        # Load W_down u1
        W_gate, W_up, W_down = load_mlp_weights(model, model_name, l)
        if W_down is None:
            continue
        
        U_d, S_d, Vt_d = np.linalg.svd(W_down, full_matrices=False)
        u1_down = U_d[:, 0]  # W_down's top output direction
        
        # Collect Δh for this layer
        h_clean_list = []
        h_corrupt_list = []
        
        for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
            if pidx % 20 == 0:
                log(f"  L{l}: pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
            corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
            
            with torch.no_grad():
                clean_out = model(input_ids=clean_inputs["input_ids"].to(input_device),
                                  attention_mask=clean_inputs["attention_mask"].to(input_device),
                                  output_hidden_states=True)
                corrupt_out = model(input_ids=corrupt_inputs["input_ids"].to(input_device),
                                    attention_mask=corrupt_inputs["attention_mask"].to(input_device),
                                    output_hidden_states=True)
            
            last_pos_c = clean_inputs["input_ids"].shape[1] - 1
            last_pos_r = corrupt_inputs["input_ids"].shape[1] - 1
            
            h_clean_list.append(clean_out.hidden_states[l+1][0, last_pos_c].detach().cpu().float().numpy())
            h_corrupt_list.append(corrupt_out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy())
            
            del clean_out, corrupt_out
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        h_clean = np.array(h_clean_list)  # (n_pairs, d_model)
        h_corrupt = np.array(h_corrupt_list)
        dh = h_clean - h_corrupt
        
        # PCA on Δh
        M_centered = dh - dh.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        except:
            continue
        
        total_var = np.sum(S**2)
        if total_var < 1e-10:
            continue
        
        explained = (S**2) / total_var
        pc1_dir = Vt[0]  # (d_model,)
        
        # Alignment: cos(u1_down, pc1)
        cos_u1_pc1 = np.dot(u1_down, pc1_dir)
        
        # Also check top-5 u directions
        cos_top5 = []
        for k in range(min(5, U_d.shape[1])):
            cos_k = np.dot(U_d[:, k], pc1_dir)
            cos_top5.append(float(cos_k))
        
        # How much of Δh's PC1 projection is explained by W_down's top mode?
        # Project each Δh onto u1_down and pc1_dir
        proj_u1 = dh @ u1_down  # (n_pairs,)
        proj_pc1 = dh @ pc1_dir  # (n_pairs,)
        
        # Correlation between these projections
        if np.std(proj_u1) > 1e-10 and np.std(proj_pc1) > 1e-10:
            corr_u1_pc1 = np.corrcoef(proj_u1, proj_pc1)[0, 1]
        else:
            corr_u1_pc1 = 0.0
        
        # Norm ratio: how much of Δh norm is in u1_down direction?
        norm_in_u1 = np.mean(np.abs(proj_u1))
        norm_total = np.mean(np.linalg.norm(dh, axis=1))
        frac_in_u1 = norm_in_u1 / (norm_total + 1e-10)
        
        # Check: what fraction of Δh's PC1 variance is in W_down's top-5 output modes?
        frac_in_top5_modes = []
        for k in range(min(5, U_d.shape[1])):
            proj_k = dh @ U_d[:, k]
            frac_k = np.mean(np.abs(proj_k)) / (norm_total + 1e-10)
            frac_in_top5_modes.append(float(frac_k))
        
        results[str(l)] = {
            "pc1_explained": float(explained[0]),
            "cos_u1_down_pc1": float(cos_u1_pc1),
            "cos_top5_down_pc1": cos_top5,
            "corr_proj_u1_pc1": float(corr_u1_pc1),
            "frac_dh_norm_in_u1": float(frac_in_u1),
            "frac_dh_norm_in_top5_modes": frac_in_top5_modes,
        }
        
        log(f"  L{l}: PC1={explained[0]:.3f}, cos(u1↓,PC1)={cos_u1_pc1:.3f}, "
            f"corr={corr_u1_pc1:.3f}, frac_u1={frac_in_u1:.3f}")
        log(f"    cos(top5↓,PC1)={[f'{c:.3f}' for c in cos_top5]}")
        log(f"    frac_top5_modes={[f'{f:.3f}' for f in frac_in_top5_modes]}")
    
    return results


def run_model(model_name):
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    
    log(f"\n{'='*60}")
    log(f"Phase 371b: {model_name}")
    log(f"{'='*60}")
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    log(f"  Model loaded in {time.time()-t0:.1f}s")
    
    # Part 1: W_down gain structure
    log("\n--- Part 1: W_down gain structure ---")
    gain_results = analyze_wdown_gain(model_name, model, n_layers, d_model)
    
    # Part 2: W_down u1 vs PC1 alignment
    log("\n--- Part 2: W_down u1 vs Δh PC1 alignment ---")
    align_results = analyze_wdown_pc1_alignment(model, tokenizer, device, model_name, n_layers, d_model)
    
    # Combine
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": len(TEST_PAIRS),
        "wdown_gain": gain_results,
        "wdown_pc1_alignment": align_results,
    }
    
    os.makedirs("results/phase371_l5_source", exist_ok=True)
    out_path = f"results/phase371_l5_source/{model_name}_phase371b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"  Results saved to {out_path}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  Model released")
    
    return all_results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    
    if model_name == "all":
        for name in ["qwen3", "deepseek7b", "glm4"]:
            try:
                run_model(name)
            except Exception as e:
                log(f"!!! {name} failed: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_model(model_name)
    
    log("\nPhase 371b complete!")


if __name__ == "__main__":
    main()
