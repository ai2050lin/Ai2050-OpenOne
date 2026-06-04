"""
Phase 372: Pre-W_down Activation Structure — Completing the Causal Chain
========================================================================

Phase 371b found: DS7B's W_down u1 aligns with Δh PC1 (cos=0.77)
Key question: Are the gate*up activations concentrated along W_down's v1 
(top right singular vector) in DS7B but not in Qwen3/GLM4?

If gate*up activations have a strong component along v1_down:
  W_down @ gate_up ≈ S[0] * (v1·gate_up) * u1
  Since u1 aligns with PC1 → output is 1D

This completes the causal chain:
  input → LN → W_gate/W_up → gate*up → W_down → Δh
                                     ↓           ↓
                              concentrated    focused to
                              along v1_down   u1_down (≈PC1)

Analysis:
1. Compute gate*up activations for binding vs non-binding
2. Project gate*up onto W_down's right singular vectors (v1, v2, ...)
3. Check if binding signal is concentrated in v1 direction
4. Compare the "v1-concentration" across DS7B/Qwen3/GLM4
5. Also analyze W_gate/W_up: do they project input into a subspace
   aligned with v1_down?

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
    """Load MLP weight matrices."""
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
    """Return the activation function for the MLP."""
    if model_name == "glm4":
        return "gelu"  # GLM4 uses GeLU with fused gate_up_proj
    else:
        return "silu"  # DS7B and Qwen3 use SwiGLU (SiLU)


def collect_mlp_intermediate_activations(model, tokenizer, device, model_name, 
                                          target_layers, n_layers, d_model):
    """
    Collect MLP intermediate activations for binding vs non-binding pairs.
    
    For each pair and each layer, we collect:
    1. MLP input (after layer norm)
    2. gate activation: SiLU/gelu(W_gate @ x) or gate portion of gate_up
    3. up activation: W_up @ x
    4. gate_times_up (pre-W_down activation)
    5. MLP output (after W_down)
    
    We use hooks to capture intermediate activations.
    """
    log("\n--- Collecting MLP intermediate activations ---")
    
    n_pairs = len(TEST_PAIRS)
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    act_fn = get_mlp_activation_fn(model_name)
    
    results = {}
    
    for l in target_layers:
        log(f"\n  Layer {l}:")
        
        # Load weights
        W_gate, W_up, W_down = load_mlp_weights(model, model_name, l)
        if W_gate is None or W_down is None:
            log(f"    Could not load weights, skipping")
            continue
        
        # SVD of W_down to get v1 (right singular vector)
        U_d, S_d, Vt_d = np.linalg.svd(W_down, full_matrices=False)
        # v1_down = Vt_d[0] — the direction in d_ff that W_down amplifies most
        # u1_down = U_d[:, 0] — the direction in d_model that results from v1
        
        # Store top-k right singular vectors
        n_sv = min(20, Vt_d.shape[0])
        V_top = Vt_d[:n_sv]  # (n_sv, d_ff) — top right singular vectors
        
        # Collect activations using hooks
        gate_up_clean_list = []
        gate_up_corrupt_list = []
        mlp_input_clean_list = []
        mlp_input_corrupt_list = []
        mlp_output_clean_list = []
        mlp_output_corrupt_list = []
        h_clean_list = []
        h_corrupt_list = []
        
        mlp = layers[l].mlp
        
        for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
            if pidx % 20 == 0:
                log(f"    Pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
            corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
            
            # Use hooks to capture intermediate activations
            captured = {}
            
            def make_hook(name):
                def hook_fn(module, input, output):
                    captured[name] = output.detach().cpu().float()
                return hook_fn
            
            # Register hooks on gate/up/down projections
            hook_handles = []
            
            # Hook on the activation function output (gate_times_up)
            # For SwiGLU: mlp.act_fn outputs the gated activation
            # For GeLU (GLM4): gate_up_proj output is split into gate and up
            gate_proj = getattr(mlp, "gate_proj", None)
            up_proj = getattr(mlp, "up_proj", None)
            down_proj = getattr(mlp, "down_proj", None)
            gate_up_proj = getattr(mlp, "gate_up_proj", None)
            
            # Strategy: compute gate*up manually from weights + input
            # We'll hook the MLP input (after LN) and compute the rest offline
            
            # Hook the LayerNorm output (MLP input)
            ln_post_attn = getattr(layers[l], "post_attention_layernorm", None)
            if ln_post_attn is None:
                ln_post_attn = getattr(layers[l], "input_layernorm", None)
            # For models with separate LN for MLP
            ln_mlp = getattr(layers[l], "post_attention_layernorm", None)
            if ln_mlp is None:
                ln_mlp = getattr(layers[l], "ln2", None)
            
            # We need a different approach: use forward hook on the MLP module itself
            # to get its input, then compute intermediates from weights
            
            def mlp_input_hook(module, input, output=None):
                # For pre-hook style: input is the MLP input
                captured["mlp_input"] = input[0].detach().cpu().float()
            
            def mlp_output_hook(module, input, output):
                captured["mlp_output"] = output.detach().cpu().float()
            
            h_mlp_in = mlp.register_forward_pre_hook(mlp_input_hook)
            h_mlp_out = mlp.register_forward_hook(mlp_output_hook)
            hook_handles.extend([h_mlp_in, h_mlp_out])
            
            # Forward pass - clean
            with torch.no_grad():
                clean_out = model(
                    input_ids=clean_inputs["input_ids"].to(input_device),
                    attention_mask=clean_inputs["attention_mask"].to(input_device),
                    output_hidden_states=True)
            
            last_pos_c = clean_inputs["input_ids"].shape[1] - 1
            
            # Get MLP input and output from hooks
            mlp_in_clean = captured["mlp_input"][0, last_pos_c].numpy()  # (d_model,)
            mlp_out_clean = captured["mlp_output"][0, last_pos_c].numpy()  # (d_model,)
            h_c = clean_out.hidden_states[l+1][0, last_pos_c].detach().cpu().float().numpy()
            
            # Compute gate*up activation offline
            if act_fn == "silu":
                # SwiGLU: gate = SiLU(W_gate @ x), up = W_up @ x, gate_times_up = gate * up
                gate_act_clean = _silu(W_gate @ mlp_in_clean)
                up_act_clean = W_up @ mlp_in_clean
                gate_times_up_clean = gate_act_clean * up_act_clean
            else:
                # GeLU (GLM4): gate = GeLU(W_gate @ x), up = W_up @ x
                gate_act_clean = _gelu(W_gate @ mlp_in_clean)
                up_act_clean = W_up @ mlp_in_clean
                gate_times_up_clean = gate_act_clean * up_act_clean
            
            gate_up_clean_list.append(gate_times_up_clean)
            mlp_input_clean_list.append(mlp_in_clean)
            mlp_output_clean_list.append(mlp_out_clean)
            h_clean_list.append(h_c)
            
            # Clear captured
            captured.clear()
            
            # Forward pass - corrupt
            with torch.no_grad():
                corrupt_out = model(
                    input_ids=corrupt_inputs["input_ids"].to(input_device),
                    attention_mask=corrupt_inputs["attention_mask"].to(input_device),
                    output_hidden_states=True)
            
            last_pos_r = corrupt_inputs["input_ids"].shape[1] - 1
            
            mlp_in_corrupt = captured["mlp_input"][0, last_pos_r].numpy()
            mlp_out_corrupt = captured["mlp_output"][0, last_pos_r].numpy()
            h_r = corrupt_out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy()
            
            if act_fn == "silu":
                gate_act_corrupt = _silu(W_gate @ mlp_in_corrupt)
                up_act_corrupt = W_up @ mlp_in_corrupt
                gate_times_up_corrupt = gate_act_corrupt * up_act_corrupt
            else:
                gate_act_corrupt = _gelu(W_gate @ mlp_in_corrupt)
                up_act_corrupt = W_up @ mlp_in_corrupt
                gate_times_up_corrupt = gate_act_corrupt * up_act_corrupt
            
            gate_up_corrupt_list.append(gate_times_up_corrupt)
            mlp_input_corrupt_list.append(mlp_in_corrupt)
            mlp_output_corrupt_list.append(mlp_out_corrupt)
            h_corrupt_list.append(h_r)
            
            # Remove hooks
            for h in hook_handles:
                h.remove()
            
            del clean_out, corrupt_out
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        # Stack into arrays
        gate_up_clean = np.array(gate_up_clean_list)    # (n_pairs, d_ff)
        gate_up_corrupt = np.array(gate_up_corrupt_list)
        mlp_in_clean = np.array(mlp_input_clean_list)    # (n_pairs, d_model)
        mlp_in_corrupt = np.array(mlp_input_corrupt_list)
        mlp_out_clean = np.array(mlp_output_clean_list)  # (n_pairs, d_model)
        mlp_out_corrupt = np.array(mlp_output_corrupt_list)
        h_clean = np.array(h_clean_list)
        h_corrupt = np.array(h_corrupt_list)
        
        # Compute differences
        d_gate_up = gate_up_clean - gate_up_corrupt   # (n_pairs, d_ff) — Δ(gate*up)
        d_mlp_in = mlp_in_clean - mlp_in_corrupt      # (n_pairs, d_model) — Δ(MLP input)
        d_mlp_out = mlp_out_clean - mlp_out_corrupt    # (n_pairs, d_model) — Δ(MLP output)
        dh = h_clean - h_corrupt                        # (n_pairs, d_model) — Δh
        
        # ===== Analysis 1: Δ(gate*up) projection onto W_down right singular vectors =====
        log(f"\n    === Analysis 1: Δ(gate*up) projection onto W_down v_k ===")
        
        # Project Δ(gate*up) onto top-k right singular vectors
        proj_vk = d_gate_up @ V_top.T  # (n_pairs, n_sv) — projections onto v1, v2, ...
        
        # Compute the fraction of Δ(gate*up) norm in each v_k direction
        dgu_norms = np.linalg.norm(d_gate_up, axis=1)  # (n_pairs,)
        mean_dgu_norm = np.mean(dgu_norms)
        
        frac_in_vk = []
        for k in range(n_sv):
            proj_k_norm = np.mean(np.abs(proj_vk[:, k]))
            frac_k = proj_k_norm / (mean_dgu_norm + 1e-10)
            frac_in_vk.append(float(frac_k))
        
        log(f"    Δ(gate*up) norm={mean_dgu_norm:.2f}")
        log(f"    frac in v1={frac_in_vk[0]:.4f}, v2={frac_in_vk[1]:.4f}, "
            f"v3={frac_in_vk[2]:.4f}, v5={frac_in_vk[4]:.4f}, v10={frac_in_vk[9]:.4f}")
        
        # Compute cumulative fraction in top-k v directions
        cum_frac = []
        for k in [1, 5, 10, 20]:
            # Fraction of Δ(gate*up) squared norm in top-k v directions
            proj_topk = proj_vk[:, :k]  # (n_pairs, k)
            energy_topk = np.sum(proj_topk**2, axis=1)
            energy_total = np.sum(d_gate_up**2, axis=1)
            cum_frac_k = np.mean(energy_topk / (energy_total + 1e-10))
            cum_frac.append(float(cum_frac_k))
        
        log(f"    cum energy in top-1={cum_frac[0]:.4f}, top-5={cum_frac[1]:.4f}, "
            f"top-10={cum_frac[2]:.4f}, top-20={cum_frac[3]:.4f}")
        
        # ===== Analysis 2: Δ(gate*up) PCA — independent structure =====
        log(f"\n    === Analysis 2: Δ(gate*up) PCA structure ===")
        
        M_centered = d_gate_up - d_gate_up.mean(axis=0, keepdims=True)
        try:
            U_gu, S_gu, Vt_gu = np.linalg.svd(M_centered, full_matrices=False)
        except:
            U_gu = S_gu = Vt_gu = None
        
        if S_gu is not None and np.sum(S_gu**2) > 1e-10:
            total_var_gu = np.sum(S_gu**2)
            explained_gu = (S_gu**2) / total_var_gu
            
            log(f"    Δ(gate*up) PC1={explained_gu[0]:.4f}, PC2={explained_gu[1]:.4f}, "
                f"PC5={explained_gu[4]:.4f}, eff_rank={int(np.searchsorted(np.cumsum(explained_gu), 0.95)+1)}")
            
            # Check alignment of Δ(gate*up) PC1 with W_down v1
            pc1_gate_up = Vt_gu[0]  # (d_ff,)
            cos_pc1_v1 = np.dot(pc1_gate_up, V_top[0])
            log(f"    cos(Δ(gate*up) PC1, v1_down)={cos_pc1_v1:.4f}")
            
            # Check top-5 alignments
            cos_top5 = [float(np.dot(Vt_gu[k], V_top[0])) for k in range(min(5, Vt_gu.shape[0]))]
            log(f"    cos(Δ(gate*up) top5 PC, v1_down)={[f'{c:.4f}' for c in cos_top5]}")
        else:
            explained_gu = None
            cos_pc1_v1 = None
        
        # ===== Analysis 3: Δ(MLP input) → W_gate/W_up alignment =====
        log(f"\n    === Analysis 3: Δ(MLP input) structure ===")
        
        M_in_centered = d_mlp_in - d_mlp_in.mean(axis=0, keepdims=True)
        try:
            U_in, S_in, Vt_in = np.linalg.svd(M_in_centered, full_matrices=False)
        except:
            U_in = S_in = Vt_in = None
        
        if S_in is not None and np.sum(S_in**2) > 1e-10:
            total_var_in = np.sum(S_in**2)
            explained_in = (S_in**2) / total_var_in
            
            log(f"    Δ(MLP input) PC1={explained_in[0]:.4f}, PC2={explained_in[1]:.4f}, "
                f"eff_rank={int(np.searchsorted(np.cumsum(explained_in), 0.95)+1)}")
            
            # Does W_gate project Δ(MLP input) into a subspace aligned with v1_down?
            # W_gate: (d_ff, d_model), v1_down: (d_ff,)
            # W_gate.T @ v1_down = (d_model,) — the input direction that gate maps to v1_down
            wgate_to_v1 = W_gate.T @ V_top[0]  # (d_model,)
            wup_to_v1 = W_up.T @ V_top[0]      # (d_model,)
            
            # How much does Δ(MLP input) project onto these directions?
            proj_gate_v1 = d_mlp_in @ wgate_to_v1
            proj_up_v1 = d_mlp_in @ wup_to_v1
            dmi_norms = np.linalg.norm(d_mlp_in, axis=1)
            
            frac_gate_v1 = np.mean(np.abs(proj_gate_v1)) / (np.mean(dmi_norms) + 1e-10)
            frac_up_v1 = np.mean(np.abs(proj_up_v1)) / (np.mean(dmi_norms) + 1e-10)
            
            log(f"    Δ(MLP input) frac projected to W_gate→v1: {frac_gate_v1:.4f}")
            log(f"    Δ(MLP input) frac projected to W_up→v1: {frac_up_v1:.4f}")
        else:
            explained_in = None
            frac_gate_v1 = frac_up_v1 = None
        
        # ===== Analysis 4: Full causal chain verification =====
        log(f"\n    === Analysis 4: Causal chain verification ===")
        
        # Verify: W_down @ Δ(gate*up) ≈ Δ(MLP output)
        reconstructed_dmlp = (W_down @ d_gate_up.T).T  # (n_pairs, d_model)
        recon_error = np.mean(np.linalg.norm(reconstructed_dmlp - d_mlp_out, axis=1)) / \
                      (np.mean(np.linalg.norm(d_mlp_out, axis=1)) + 1e-10)
        log(f"    Recon: W_down @ Δ(gate*up) vs Δ(MLP output): error={recon_error:.4f}")
        
        # How much of Δ(MLP output) is captured by S[0]*(v1·Δ(gate*up))*u1?
        proj_v1 = d_gate_up @ V_top[0]  # (n_pairs,) — Δ(gate*up) projected onto v1
        mlp_out_from_v1 = np.outer(proj_v1 * S_d[0], U_d[:, 0])  # (n_pairs, d_model)
        
        frac_from_v1_mode = np.mean(np.linalg.norm(mlp_out_from_v1, axis=1)) / \
                           (np.mean(np.linalg.norm(d_mlp_out, axis=1)) + 1e-10)
        
        log(f"    Fraction of Δ(MLP output) from v1 mode alone: {frac_from_v1_mode:.4f}")
        
        # Compare with Δh PC1
        M_dh_centered = dh - dh.mean(axis=0, keepdims=True)
        try:
            U_dh, S_dh, Vt_dh = np.linalg.svd(M_dh_centered, full_matrices=False)
            pc1_dh = Vt_dh[0]
            total_var_dh = np.sum(S_dh**2)
            explained_dh = (S_dh**2) / total_var_dh
            
            # How much of Δh is captured by the v1→u1 mode?
            proj_dh_on_u1 = dh @ U_d[:, 0]
            frac_dh_from_v1_mode = np.mean(np.abs(proj_dh_on_u1)) / \
                                   (np.mean(np.linalg.norm(dh, axis=1)) + 1e-10)
            
            cos_u1_pc1 = np.dot(U_d[:, 0], pc1_dh)
            
            log(f"    Δh PC1={explained_dh[0]:.4f}, cos(u1↓, PC1)={cos_u1_pc1:.4f}")
            log(f"    Δh frac in u1_down direction: {frac_dh_from_v1_mode:.4f}")
        except:
            explained_dh = None
            cos_u1_pc1 = None
            frac_dh_from_v1_mode = None
        
        # ===== Analysis 5: gate*up PCA for clean activations (not just Δ) =====
        log(f"\n    === Analysis 5: Clean gate*up activation structure ===")
        
        M_gu_clean_centered = gate_up_clean - gate_up_clean.mean(axis=0, keepdims=True)
        try:
            _, S_gu_clean, Vt_gu_clean = np.linalg.svd(M_gu_clean_centered, full_matrices=False)
            total_var_clean = np.sum(S_gu_clean**2)
            explained_clean = (S_gu_clean**2) / total_var_clean
            
            log(f"    Clean gate*up PC1={explained_clean[0]:.4f}, PC5={explained_clean[4]:.4f}, "
                f"eff_rank={int(np.searchsorted(np.cumsum(explained_clean), 0.95)+1)}")
            
            cos_clean_v1 = np.dot(Vt_gu_clean[0], V_top[0])
            log(f"    cos(clean gate*up PC1, v1_down)={cos_clean_v1:.4f}")
        except:
            explained_clean = None
            cos_clean_v1 = None
        
        # ===== Store results =====
        layer_result = {
            "d_ff": int(W_down.shape[1]),
            "d_model": int(W_down.shape[0]),
            "n_pairs": n_pairs,
            # Analysis 1
            "dgu_mean_norm": float(mean_dgu_norm),
            "dgu_frac_in_vk": frac_in_vk[:20],
            "dgu_cum_energy_in_top_k": {
                "1": cum_frac[0], "5": cum_frac[1], "10": cum_frac[2], "20": cum_frac[3]
            },
            # Analysis 2
            "dgu_pc1_explained": float(explained_gu[0]) if explained_gu is not None else None,
            "dgu_pc5_explained": float(explained_gu[4]) if explained_gu is not None else None,
            "cos_dgu_pc1_v1_down": float(cos_pc1_v1) if cos_pc1_v1 is not None else None,
            # Analysis 3
            "dmlp_input_pc1_explained": float(explained_in[0]) if explained_in is not None else None,
            "frac_dmlp_input_to_wgate_v1": float(frac_gate_v1) if frac_gate_v1 is not None else None,
            "frac_dmlp_input_to_wup_v1": float(frac_up_v1) if frac_up_v1 is not None else None,
            # Analysis 4
            "recon_error_wdown_dgu": float(recon_error),
            "frac_dmlp_out_from_v1_mode": float(frac_from_v1_mode),
            "dh_pc1_explained": float(explained_dh[0]) if explained_dh is not None else None,
            "cos_u1_down_pc1": float(cos_u1_pc1) if cos_u1_pc1 is not None else None,
            "frac_dh_in_u1_down": float(frac_dh_from_v1_mode) if frac_dh_from_v1_mode is not None else None,
            # Analysis 5
            "clean_gu_pc1_explained": float(explained_clean[0]) if explained_clean is not None else None,
            "cos_clean_gu_pc1_v1_down": float(cos_clean_v1) if cos_clean_v1 is not None else None,
            # Singular values
            "wdown_top5_sv": [float(s) for s in S_d[:5]],
            "wdown_top1_gain": float(S_d[0] / np.mean(S_d)),
        }
        
        results[str(l)] = layer_result
        log(f"    Layer {l} done")
    
    return results


def _silu(x):
    """SiLU/Swish activation: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def _gelu(x):
    """GeLU activation (approximate)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def run_model(model_name):
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    
    log(f"\n{'='*60}")
    log(f"Phase 372: {model_name}")
    log(f"{'='*60}")
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    log(f"  Model loaded in {time.time()-t0:.1f}s")
    
    # Select target layers based on model
    if model_name == "deepseek7b":
        target_layers = [3, 4, 5, 6, 8, 12, 18, 24]
    elif model_name == "qwen3":
        target_layers = [3, 4, 5, 8, 16, 28]
    else:
        target_layers = [3, 4, 5, 10, 20, 30]
    
    # Main analysis
    results = collect_mlp_intermediate_activations(
        model, tokenizer, device, model_name, target_layers, n_layers, d_model)
    
    # Save results
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": len(TEST_PAIRS),
        "phase": "372",
        "target_layers": target_layers,
        "layer_results": results,
    }
    
    os.makedirs("results/phase372_pre_wdown", exist_ok=True)
    out_path = f"results/phase372_pre_wdown/{model_name}_phase372.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"  Results saved to {out_path}")
    
    # Print summary
    log(f"\n{'='*60}")
    log(f"Phase 372 Summary: {model_name}")
    log(f"{'='*60}")
    log(f"{'Layer':>5} | {'dgu frac v1':>12} | {'dgu cum1':>8} | {'dgu PC1':>8} | "
        f"{'cos(pc1,v1)':>12} | {'v1→mlp frac':>12} | {'cos(u1,PC1)':>12}")
    log("-" * 90)
    
    for l in sorted(results.keys(), key=int):
        r = results[l]
        log(f"  L{int(l):>3} | {r.get('dgu_frac_in_vk',[0]*20)[0]:>12.4f} | "
            f"{r.get('dgu_cum_energy_in_top_k',{}).get('1',0):>8.4f} | "
            f"{r.get('dgu_pc1_explained',0) or 0:>8.4f} | "
            f"{r.get('cos_dgu_pc1_v1_down',0) or 0:>12.4f} | "
            f"{r.get('frac_dmlp_out_from_v1_mode',0):>12.4f} | "
            f"{r.get('cos_u1_down_pc1',0) or 0:>12.4f}")
    
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
    
    log("\nPhase 372 complete!")


if __name__ == "__main__":
    main()
