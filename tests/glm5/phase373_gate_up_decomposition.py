"""
Phase 373: Gate vs Up Decomposition — Why is gate*up Δ 1D in DS7B?
===================================================================

Phase 372 found that DS7B L4's Δ(gate*up) is already 1D (PC1=0.99) in d_ff space,
while MLP input Δ is high-dimensional (PC1=0.10). The compression from high-dim
to 1D happens inside the MLP.

This test decomposes: gate*up = SiLU(W_gate @ x) * (W_up @ x)
to determine which component creates the 1D structure.

Key analyses:
1. gate Δ structure: Is SiLU(W_gate @ x_clean) - SiLU(W_gate @ x_corrupt) 1D?
2. up Δ structure: Is W_up @ x_clean - W_up @ x_corrupt 1D?
3. Linearized gate Δ: Is W_gate @ (x_clean - x_corrupt) 1D? (before SiLU)
4. Interaction: Does the product gate*up create 1D from two high-dim components?
5. Cross-model comparison

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


def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


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
    import glob
    from safetensors import safe_open
    
    layers = get_layers(model)
    mlp = layers[layer_idx].mlp
    
    gate_proj = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    
    W_gate = W_up = W_down = None
    
    if gate_proj is not None and up_proj is not None and down_proj is not None:
        try:
            W_gate = gate_proj.weight.detach().cpu().float().numpy()
            W_up = up_proj.weight.detach().cpu().float().numpy()
            W_down = down_proj.weight.detach().cpu().float().numpy()
        except:
            pass
    
    if W_gate is None:
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


def get_mlp_activation_fn(model_name):
    if model_name == "glm4":
        return "gelu"
    else:
        return "silu"


def compute_pca_explained(X):
    """Compute PCA explained variance ratios. Returns (explained, eff_rank, pc1_dir)."""
    M = X - X.mean(axis=0, keepdims=True)
    try:
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
    except:
        return None, None, None
    total_var = np.sum(S**2)
    if total_var < 1e-10:
        return None, None, None
    explained = (S**2) / total_var
    eff_rank = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)
    return explained, eff_rank, Vt[0]


def analyze_gate_up_decomposition(model, tokenizer, device, model_name, 
                                   target_layers, n_layers, d_model):
    """
    Decompose gate*up into gate and up components to find source of 1D structure.
    """
    log("\n--- Gate vs Up Decomposition ---")
    
    n_pairs = len(TEST_PAIRS)
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    act_fn = get_mlp_activation_fn(model_name)
    
    results = {}
    
    for l in target_layers:
        log(f"\n  Layer {l}:")
        
        W_gate, W_up, W_down = load_mlp_weights(model, model_name, l)
        if W_gate is None or W_down is None:
            log(f"    Could not load weights, skipping")
            continue
        
        d_ff = W_gate.shape[0]
        
        # Collect MLP inputs
        mlp_in_clean_list = []
        mlp_in_corrupt_list = []
        
        mlp = layers[l].mlp
        
        for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
            if pidx % 20 == 0:
                log(f"    Pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
            corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
            
            captured = {}
            
            def mlp_input_hook(module, input, output=None):
                captured["mlp_input"] = input[0].detach().cpu().float()
            
            h_mlp_in = mlp.register_forward_pre_hook(mlp_input_hook)
            
            with torch.no_grad():
                clean_out = model(
                    input_ids=clean_inputs["input_ids"].to(input_device),
                    attention_mask=clean_inputs["attention_mask"].to(input_device),
                    output_hidden_states=True)
            
            last_pos_c = clean_inputs["input_ids"].shape[1] - 1
            mlp_in_clean = captured["mlp_input"][0, last_pos_c].numpy()
            mlp_in_clean_list.append(mlp_in_clean)
            
            captured.clear()
            
            with torch.no_grad():
                corrupt_out = model(
                    input_ids=corrupt_inputs["input_ids"].to(input_device),
                    attention_mask=corrupt_inputs["attention_mask"].to(input_device),
                    output_hidden_states=True)
            
            last_pos_r = corrupt_inputs["input_ids"].shape[1] - 1
            mlp_in_corrupt = captured["mlp_input"][0, last_pos_r].numpy()
            mlp_in_corrupt_list.append(mlp_in_corrupt)
            
            h_mlp_in.remove()
            del clean_out, corrupt_out
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        mlp_in_clean = np.array(mlp_in_clean_list)    # (n_pairs, d_model)
        mlp_in_corrupt = np.array(mlp_in_corrupt_list)
        
        # ===== Compute intermediate activations =====
        
        # Linear projections (before activation)
        gate_linear_clean = mlp_in_clean @ W_gate.T    # (n_pairs, d_ff)
        gate_linear_corrupt = mlp_in_corrupt @ W_gate.T
        up_clean = mlp_in_clean @ W_up.T               # (n_pairs, d_ff)
        up_corrupt = mlp_in_corrupt @ W_up.T
        
        # After activation
        if act_fn == "silu":
            gate_act_clean = _silu(gate_linear_clean)
            gate_act_corrupt = _silu(gate_linear_corrupt)
        else:
            gate_act_clean = _gelu(gate_linear_clean)
            gate_act_corrupt = _gelu(gate_linear_corrupt)
        
        # gate*up
        gate_up_clean = gate_act_clean * up_clean
        gate_up_corrupt = gate_act_corrupt * up_corrupt
        
        # Differences
        d_mlp_in = mlp_in_clean - mlp_in_corrupt               # (n_pairs, d_model)
        d_gate_linear = gate_linear_clean - gate_linear_corrupt  # (n_pairs, d_ff) — linear gate diff
        d_gate_act = gate_act_clean - gate_act_corrupt           # (n_pairs, d_ff) — after SiLU/GeLU
        d_up = up_clean - up_corrupt                             # (n_pairs, d_ff) — up diff
        d_gate_up = gate_up_clean - gate_up_corrupt             # (n_pairs, d_ff) — gate*up diff
        
        # ===== Analysis 1: PCA structure of each component =====
        log(f"\n    === Analysis 1: PCA of each component's Δ ===")
        
        pca_mlp_in, rank_in, _ = compute_pca_explained(d_mlp_in)
        pca_gate_lin, rank_gl, pc1_gl = compute_pca_explained(d_gate_linear)
        pca_gate_act, rank_ga, pc1_ga = compute_pca_explained(d_gate_act)
        pca_up, rank_up, pc1_up = compute_pca_explained(d_up)
        pca_gu, rank_gu, pc1_gu = compute_pca_explained(d_gate_up)
        
        log(f"    Δ(MLP input)   PC1={pca_mlp_in[0]:.4f}, PC5={pca_mlp_in[4]:.4f}, eff_rank={rank_in}")
        log(f"    Δ(gate_linear) PC1={pca_gate_lin[0]:.4f}, PC5={pca_gate_lin[4]:.4f}, eff_rank={rank_gl}")
        log(f"    Δ(gate_act)    PC1={pca_gate_act[0]:.4f}, PC5={pca_gate_act[4]:.4f}, eff_rank={rank_ga}")
        log(f"    Δ(up)          PC1={pca_up[0]:.4f}, PC5={pca_up[4]:.4f}, eff_rank={rank_up}")
        log(f"    Δ(gate*up)     PC1={pca_gu[0]:.4f}, PC5={pca_gu[4]:.4f}, eff_rank={rank_gu}")
        
        # ===== Analysis 2: Norm analysis =====
        log(f"\n    === Analysis 2: Norm analysis ===")
        
        norms = {
            "d_mlp_in": np.mean(np.linalg.norm(d_mlp_in, axis=1)),
            "d_gate_linear": np.mean(np.linalg.norm(d_gate_linear, axis=1)),
            "d_gate_act": np.mean(np.linalg.norm(d_gate_act, axis=1)),
            "d_up": np.mean(np.linalg.norm(d_up, axis=1)),
            "d_gate_up": np.mean(np.linalg.norm(d_gate_up, axis=1)),
        }
        
        log(f"    Δ(MLP input) norm:   {norms['d_mlp_in']:.4f}")
        log(f"    Δ(gate_linear) norm: {norms['d_gate_linear']:.4f}")
        log(f"    Δ(gate_act) norm:    {norms['d_gate_act']:.4f}")
        log(f"    Δ(up) norm:          {norms['d_up']:.4f}")
        log(f"    Δ(gate*up) norm:     {norms['d_gate_up']:.4f}")
        
        # ===== Analysis 3: Cross-alignment of PC1 directions =====
        log(f"\n    === Analysis 3: Cross-alignment ===")
        
        # cos between different component PC1s
        cos_gl_ga = np.dot(pc1_gl, pc1_ga) if pc1_gl is not None and pc1_ga is not None else None
        cos_gl_up = np.dot(pc1_gl, pc1_up) if pc1_gl is not None and pc1_up is not None else None
        cos_ga_up = np.dot(pc1_ga, pc1_up) if pc1_ga is not None and pc1_up is not None else None
        cos_ga_gu = np.dot(pc1_ga, pc1_gu) if pc1_ga is not None and pc1_gu is not None else None
        cos_up_gu = np.dot(pc1_up, pc1_gu) if pc1_up is not None and pc1_gu is not None else None
        cos_gl_gu = np.dot(pc1_gl, pc1_gu) if pc1_gl is not None and pc1_gu is not None else None
        
        log(f"    cos(gate_lin PC1, gate_act PC1) = {cos_gl_ga:.4f}" if cos_gl_ga else "")
        log(f"    cos(gate_lin PC1, up PC1) = {cos_gl_up:.4f}" if cos_gl_up else "")
        log(f"    cos(gate_act PC1, up PC1) = {cos_ga_up:.4f}" if cos_ga_up else "")
        log(f"    cos(gate_act PC1, gate*up PC1) = {cos_ga_gu:.4f}" if cos_ga_gu else "")
        log(f"    cos(up PC1, gate*up PC1) = {cos_up_gu:.4f}" if cos_up_gu else "")
        log(f"    cos(gate_lin PC1, gate*up PC1) = {cos_gl_gu:.4f}" if cos_gl_gu else "")
        
        # ===== Analysis 4: Contribution decomposition =====
        log(f"\n    === Analysis 4: Contribution decomposition ===")
        
        # gate*up_clean - gate*up_corrupt = gate_act_clean * up_clean - gate_act_corrupt * up_corrupt
        # = gate_act_clean * (up_clean - up_corrupt) + (gate_act_clean - gate_act_corrupt) * up_corrupt
        #   + (gate_act_clean - gate_act_corrupt) * (up_clean - up_corrupt)
        # ≈ gate_act_clean * Δ(up) + Δ(gate_act) * up_corrupt + Δ(gate_act) * Δ(up)
        
        gate_contribution = d_gate_act * up_corrupt        # gate change * up_base
        up_contribution = gate_act_clean * d_up            # gate_base * up change
        interaction = d_gate_act * d_up                    # gate change * up change
        
        gate_contrib_norm = np.mean(np.linalg.norm(gate_contribution, axis=1))
        up_contrib_norm = np.mean(np.linalg.norm(up_contribution, axis=1))
        interaction_norm = np.mean(np.linalg.norm(interaction, axis=1))
        total_norm = gate_contrib_norm + up_contrib_norm + interaction_norm
        
        if total_norm > 1e-10:
            frac_gate = gate_contrib_norm / total_norm
            frac_up = up_contrib_norm / total_norm
            frac_interact = interaction_norm / total_norm
        else:
            frac_gate = frac_up = frac_interact = 0
        
        log(f"    gate_change * up_base: norm={gate_contrib_norm:.4f}, frac={frac_gate:.4f}")
        log(f"    gate_base * up_change: norm={up_contrib_norm:.4f}, frac={frac_up:.4f}")
        log(f"    gate_change * up_change: norm={interaction_norm:.4f}, frac={frac_interact:.4f}")
        
        # PCA of each contribution component
        pca_gate_contrib, _, pc1_gc = compute_pca_explained(gate_contribution)
        pca_up_contrib, _, pc1_uc = compute_pca_explained(up_contribution)
        pca_interact, _, pc1_ic = compute_pca_explained(interaction)
        
        log(f"    gate_change*up_base PC1={pca_gate_contrib[0]:.4f}" if pca_gate_contrib is not None else "")
        log(f"    gate_base*up_change PC1={pca_up_contrib[0]:.4f}" if pca_up_contrib is not None else "")
        log(f"    interaction PC1={pca_interact[0]:.4f}" if pca_interact is not None else "")
        
        # Alignment of contribution PC1s with gate*up PC1
        cos_gc_gu = np.dot(pc1_gc, pc1_gu) if pc1_gc is not None and pc1_gu is not None else None
        cos_uc_gu = np.dot(pc1_uc, pc1_gu) if pc1_uc is not None and pc1_gu is not None else None
        
        log(f"    cos(gate_contrib PC1, gate*up PC1) = {cos_gc_gu:.4f}" if cos_gc_gu else "")
        log(f"    cos(up_contrib PC1, gate*up PC1) = {cos_uc_gu:.4f}" if cos_uc_gu else "")
        
        # ===== Analysis 5: SiLU/GeLU effect — does activation function create 1D? =====
        log(f"\n    === Analysis 5: Activation function effect ===")
        
        # Compare linearized gate diff vs actual gate diff
        # If SiLU is approximately linear, d_gate_act ≈ SiLU'(gate_linear) * d_gate_linear
        # The Jacobian of SiLU is: sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x)) = sigmoid(x)*(1 + x*(1-sigmoid(x)))
        
        if act_fn == "silu":
            sig = 1.0 / (1.0 + np.exp(-np.clip(gate_linear_clean, -50, 50)))
            silu_jacobian = sig * (1 + gate_linear_clean * (1 - sig))  # (n_pairs, d_ff)
        else:
            # GeLU approximate Jacobian
            x = gate_linear_clean
            gelu_jacobian = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3))) + \
                           0.5 * x * (1 - np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3))**2) * \
                           np.sqrt(2/np.pi) * (1 + 0.134145 * x**2)
            silu_jacobian = gelu_jacobian
        
        # Linearized gate diff: SiLU'(gate_clean) * d_gate_linear
        d_gate_linearized = silu_jacobian * d_gate_linear
        
        pca_lin, _, pc1_lin = compute_pca_explained(d_gate_linearized)
        
        log(f"    Δ(gate_linear) PC1 = {pca_gate_lin[0]:.4f}" if pca_gate_lin is not None else "")
        log(f"    Δ(gate_linearized) PC1 = {pca_lin[0]:.4f}" if pca_lin is not None else "")
        log(f"    Δ(gate_act) PC1 = {pca_gate_act[0]:.4f}" if pca_gate_act is not None else "")
        
        # Does the Jacobian (gate saturation) create the 1D structure?
        # Check: how selective is the Jacobian? Are certain neurons much more "active"?
        jac_mean = np.mean(silu_jacobian, axis=0)  # (d_ff,) — average Jacobian per neuron
        jac_std = np.std(silu_jacobian, axis=0)
        jac_selectivity = jac_std / (jac_mean + 1e-10)  # coefficient of variation
        
        log(f"    Jacobian mean={np.mean(jac_mean):.4f}, std={np.std(jac_mean):.4f}")
        log(f"    Jacobian selectivity: mean CV={np.mean(jac_selectivity):.4f}, "
            f"max CV={np.max(jac_selectivity):.4f}")
        
        # Which neurons have the highest Jacobian?
        top_neurons = np.argsort(jac_mean)[::-1][:20]
        log(f"    Top-20 Jacobian neurons: {top_neurons[:10].tolist()}")
        log(f"    Top-20 Jacobian values: {[f'{jac_mean[i]:.4f}' for i in top_neurons[:10]]}")
        
        # Does the linearized gate diff have similar PC1 to actual?
        cos_lin_ga = np.dot(pc1_lin, pc1_ga) if pc1_lin is not None and pc1_ga is not None else None
        log(f"    cos(linearized PC1, gate_act PC1) = {cos_lin_ga:.4f}" if cos_lin_ga else "")
        
        # ===== Analysis 6: Gate saturation and selective amplification =====
        log(f"\n    === Analysis 6: Gate saturation analysis ===")
        
        # For DS7B's SiLU: if gate_linear_clean is very positive, SiLU ≈ gate_linear (unsaturated)
        # If gate_linear_clean is near 0, SiLU ≈ 0 (saturated off)
        # If gate_linear_clean is very negative, SiLU ≈ 0 (saturated off)
        
        # Compute gate activation rate
        gate_active_clean = np.mean(np.abs(gate_act_clean) > 0.01 * np.max(np.abs(gate_act_clean)))
        gate_active_corrupt = np.mean(np.abs(gate_act_corrupt) > 0.01 * np.max(np.abs(gate_act_corrupt)))
        
        log(f"    Gate active rate (clean): {gate_active_clean:.4f}")
        log(f"    Gate active rate (corrupt): {gate_active_corrupt:.4f}")
        
        # Compute gate*up energy distribution
        gu_energy = np.mean(gate_up_clean**2, axis=0)  # (d_ff,)
        total_energy = np.sum(gu_energy)
        if total_energy > 1e-10:
            gu_concentration = np.sort(gu_energy / total_energy)[::-1]
            top1_conc = gu_concentration[0]
            top10_conc = np.sum(gu_concentration[:10])
            top100_conc = np.sum(gu_concentration[:100])
        else:
            top1_conc = top10_conc = top100_conc = 0
        
        log(f"    gate*up energy top1: {top1_conc:.6f}, top10: {top10_conc:.6f}, top100: {top100_conc:.6f}")
        
        # ===== Store results =====
        layer_result = {
            "d_ff": int(d_ff),
            "d_model": int(W_gate.shape[1]),
            "n_pairs": n_pairs,
            # Analysis 1: PCA
            "pca_mlp_input": {"pc1": float(pca_mlp_in[0]), "pc5": float(pca_mlp_in[4]), "eff_rank": rank_in} if pca_mlp_in is not None else None,
            "pca_gate_linear": {"pc1": float(pca_gate_lin[0]), "pc5": float(pca_gate_lin[4]), "eff_rank": rank_gl} if pca_gate_lin is not None else None,
            "pca_gate_act": {"pc1": float(pca_gate_act[0]), "pc5": float(pca_gate_act[4]), "eff_rank": rank_ga} if pca_gate_act is not None else None,
            "pca_up": {"pc1": float(pca_up[0]), "pc5": float(pca_up[4]), "eff_rank": rank_up} if pca_up is not None else None,
            "pca_gate_up": {"pc1": float(pca_gu[0]), "pc5": float(pca_gu[4]), "eff_rank": rank_gu} if pca_gu is not None else None,
            "pca_gate_linearized": {"pc1": float(pca_lin[0]), "pc5": float(pca_lin[4])} if pca_lin is not None else None,
            # Analysis 2: Norms
            "norms": {k: float(v) for k, v in norms.items()},
            # Analysis 3: Cross-alignment
            "cos_gate_lin_gate_act": float(cos_gl_ga) if cos_gl_ga else None,
            "cos_gate_lin_up": float(cos_gl_up) if cos_gl_up else None,
            "cos_gate_act_up": float(cos_ga_up) if cos_ga_up else None,
            "cos_gate_act_gate_up": float(cos_ga_gu) if cos_ga_gu else None,
            "cos_up_gate_up": float(cos_up_gu) if cos_up_gu else None,
            "cos_gate_lin_gate_up": float(cos_gl_gu) if cos_gl_gu else None,
            "cos_linearized_gate_act": float(cos_lin_ga) if cos_lin_ga else None,
            # Analysis 4: Contribution decomposition
            "frac_gate_contrib": float(frac_gate),
            "frac_up_contrib": float(frac_up),
            "frac_interaction": float(frac_interact),
            "pca_gate_contrib_pc1": float(pca_gate_contrib[0]) if pca_gate_contrib is not None else None,
            "pca_up_contrib_pc1": float(pca_up_contrib[0]) if pca_up_contrib is not None else None,
            "pca_interact_pc1": float(pca_interact[0]) if pca_interact is not None else None,
            "cos_gate_contrib_pc1_gu": float(cos_gc_gu) if cos_gc_gu else None,
            "cos_up_contrib_pc1_gu": float(cos_uc_gu) if cos_uc_gu else None,
            # Analysis 5-6
            "jacobian_mean": float(np.mean(jac_mean)),
            "jacobian_std": float(np.std(jac_mean)),
            "jacobian_cv_mean": float(np.mean(jac_selectivity)),
            "gate_active_rate_clean": float(gate_active_clean),
            "gate_active_rate_corrupt": float(gate_active_corrupt),
            "gate_up_top1_concentration": float(top1_conc),
            "gate_up_top10_concentration": float(top10_conc),
            "gate_up_top100_concentration": float(top100_conc),
        }
        
        results[str(l)] = layer_result
        log(f"    Layer {l} done")
    
    return results


def run_model(model_name):
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    
    log(f"\n{'='*60}")
    log(f"Phase 373: {model_name}")
    log(f"{'='*60}")
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    log(f"  Model loaded in {time.time()-t0:.1f}s")
    
    if model_name == "deepseek7b":
        target_layers = [3, 4, 5, 6, 8, 12, 18, 24]
    elif model_name == "qwen3":
        target_layers = [3, 4, 5, 8, 16, 28]
    else:
        target_layers = [3, 4, 5, 10, 20, 30]
    
    results = analyze_gate_up_decomposition(
        model, tokenizer, device, model_name, target_layers, n_layers, d_model)
    
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": len(TEST_PAIRS),
        "phase": "373",
        "target_layers": target_layers,
        "layer_results": results,
    }
    
    os.makedirs("results/phase373_gate_up_decomp", exist_ok=True)
    out_path = f"results/phase373_gate_up_decomp/{model_name}_phase373.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"  Results saved to {out_path}")
    
    # Print summary
    log(f"\n{'='*60}")
    log(f"Phase 373 Summary: {model_name}")
    log(f"{'='*60}")
    log(f"{'Layer':>5} | {'in PC1':>7} | {'gL PC1':>7} | {'gA PC1':>7} | {'up PC1':>7} | "
        f"{'gU PC1':>7} | {'gLin PC1':>7} | {'frac_g':>7} | {'frac_u':>7} | {'frac_i':>7}")
    log("-" * 100)
    
    for l in sorted(results.keys(), key=int):
        r = results[l]
        pca_in = r.get("pca_mlp_input", {}) or {}
        pca_gl = r.get("pca_gate_linear", {}) or {}
        pca_ga = r.get("pca_gate_act", {}) or {}
        pca_up = r.get("pca_up", {}) or {}
        pca_gu = r.get("pca_gate_up", {}) or {}
        pca_lin = r.get("pca_gate_linearized", {}) or {}
        
        log(f"  L{int(l):>3} | {pca_in.get('pc1',0):>7.3f} | {pca_gl.get('pc1',0):>7.3f} | "
            f"{pca_ga.get('pc1',0):>7.3f} | {pca_up.get('pc1',0):>7.3f} | "
            f"{pca_gu.get('pc1',0):>7.3f} | {pca_lin.get('pc1',0):>7.3f} | "
            f"{r.get('frac_gate_contrib',0):>7.3f} | {r.get('frac_up_contrib',0):>7.3f} | "
            f"{r.get('frac_interaction',0):>7.3f}")
    
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
    
    log("\nPhase 373 complete!")


if __name__ == "__main__":
    main()
