"""
Phase 371: DS7B L4→L5 Norm Explosion Source Decomposition
=========================================================

Phase 370c pinpointed the 1D collapse at DS7B L4→L5:
- L4: norm=15.7, PC1=0.10, PC1/residual=0.19
- L5: norm=118.9, PC1=0.99, PC1/residual=3.99 (7.58x norm explosion)
- L4→L5 PC1 direction cos=0.251 (near-orthogonal rotation)

This test decomposes the L5 residual update into component contributions
to determine WHO writes the massive PC1.

Architecture of a transformer layer:
  h_after_attn = h_input + attn_out(h_after_input_norm)
  h_after_mlp  = h_after_attn + mlp_out(h_after_attn_norm)

For SwiGLU MLP: mlp_out = W_down(Swish(W_gate @ x) ⊙ (W_up @ x))

Decomposition:
  Δh_L5 = h_clean_L5 - h_corrupt_L5
  Δh_L4 = h_clean_L4 - h_corrupt_L4

  Δh_attn = Δ(attn_out at L5)  = attn_out_clean - attn_out_corrupt
  Δh_mlp  = Δ(mlp_out at L5)   = mlp_out_clean - mlp_out_corrupt

  Total change: Δh_L5 ≈ Δh_L4 + Δh_attn + Δh_mlp
  (This is exact in linear approximation)

  PC1 at L5: project each component onto L5's PC1 direction
  → Who contributes the most to PC1 norm?

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
    for attr in ["lm_head", "embed_out"]:
        if hasattr(model, attr):
            w = getattr(model, attr)
            if hasattr(w, "weight"):
                weight = w.weight
                if not weight.is_meta:
                    return weight.detach().cpu().float().numpy()
    import glob
    from safetensors import safe_open
    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            for key in sf.keys():
                if 'lm_head' in key or 'embed_out' in key:
                    return sf.get_tensor(key).float().numpy()
    return None


def collect_component_data(model, tokenizer, device, model_name, n_layers, d_model):
    """
    Collect residual stream states and component outputs at key layers.
    
    For each pair, capture at the COLLAPSE layer and adjacent layers:
    - DS7B: L4 (pre-collapse), L5 (collapse), L6 (post-collapse)
    - Qwen3: L4, L8, L16
    - GLM4: L4, L10, L20
    
    At each target layer, capture:
    1. h_input (before the layer's input norm)
    2. h_after_attn (after attention + residual add)
    3. h_after_mlp (after MLP + residual add = layer output)
    4. attn_out (raw attention output, before residual add)
    5. mlp_out (raw MLP output, before residual add)
    """
    n_pairs = len(TEST_PAIRS)
    cfg = MODEL_CONFIGS[model_name]
    
    # Determine target layers based on model
    if model_name == "deepseek7b":
        target_layers = [3, 4, 5, 6, 7, 8]
    elif model_name == "qwen3":
        target_layers = [3, 4, 5, 6, 7, 8, 16]
    else:  # glm4
        target_layers = [3, 4, 5, 6, 7, 8, 20]
    
    layers = get_layers(model)
    
    # Data containers: per-layer, per-pair
    data = {}
    for l in target_layers:
        data[l] = {
            "h_input_clean": np.zeros((n_pairs, d_model), dtype=np.float32),
            "h_input_corrupt": np.zeros((n_pairs, d_model), dtype=np.float32),
            "h_output_clean": np.zeros((n_pairs, d_model), dtype=np.float32),
            "h_output_corrupt": np.zeros((n_pairs, d_model), dtype=np.float32),
            "h_after_attn_clean": np.zeros((n_pairs, d_model), dtype=np.float32),
            "h_after_attn_corrupt": np.zeros((n_pairs, d_model), dtype=np.float32),
            "attn_out_clean": np.zeros((n_pairs, d_model), dtype=np.float32),
            "attn_out_corrupt": np.zeros((n_pairs, d_model), dtype=np.float32),
            "mlp_out_clean": np.zeros((n_pairs, d_model), dtype=np.float32),
            "mlp_out_corrupt": np.zeros((n_pairs, d_model), dtype=np.float32),
        }
    
    input_device = next(model.parameters()).device
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        if pidx % 5 == 0:
            log(f"  Pair {pidx+1}/{n_pairs}: {obj}-{target}/{competitor}")
        
        clean_prompt = TEMPLATE.format(obj=obj, attr=target)
        corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
        
        clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
        corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
        
        clean_ids = clean_inputs["input_ids"].to(input_device)
        clean_mask = clean_inputs["attention_mask"].to(input_device)
        corrupt_ids = corrupt_inputs["input_ids"].to(input_device)
        corrupt_mask = corrupt_inputs["attention_mask"].to(input_device)
        
        for prompt_type, ids, mask in [("clean", clean_ids, clean_mask), 
                                         ("corrupt", corrupt_ids, corrupt_mask)]:
            # Register hooks on target layers
            captured = {}
            hooks = []
            
            for l in target_layers:
                layer = layers[l]
                
                # Hook 1: Capture input to the layer (from residual stream)
                # This is the hidden state before input_layernorm
                # We capture it from the module's forward pre-hook
                
                # Hook 2: Capture attention output (before residual add)
                # For Llama-style: self_attn output is the result of o_proj
                sa = layer.self_attn
                
                def make_attn_hook(layer_idx):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[f"L{layer_idx}_attn_out"] = output[0].detach()
                        else:
                            captured[f"L{layer_idx}_attn_out"] = output.detach()
                    return hook
                
                hooks.append(sa.register_forward_hook(make_attn_hook(l)))
                
                # Hook 3: Capture MLP output (before residual add)
                mlp = layer.mlp
                
                def make_mlp_hook(layer_idx):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[f"L{layer_idx}_mlp_out"] = output[0].detach()
                        else:
                            captured[f"L{layer_idx}_mlp_out"] = output.detach()
                    return hook
                
                hooks.append(mlp.register_forward_hook(make_mlp_hook(l)))
            
            # Forward pass
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            
            # Remove hooks
            for h in hooks:
                h.remove()
            
            # Extract data
            last_pos = ids.shape[1] - 1
            hs = out.hidden_states  # tuple of (1, seq_len, d_model)
            
            for l in target_layers:
                # h_input = hidden state at layer l (input to this layer)
                h_in = hs[l][0, last_pos].detach().cpu().float().numpy()
                # h_output = hidden state at layer l+1 (output of this layer)
                h_out = hs[l + 1][0, last_pos].detach().cpu().float().numpy()
                
                # Attention output
                attn_key = f"L{l}_attn_out"
                if attn_key in captured:
                    attn_out = captured[attn_key][0, last_pos].detach().cpu().float().numpy()
                else:
                    attn_out = h_out - h_in  # fallback
                
                # MLP output
                mlp_key = f"L{l}_mlp_out"
                if mlp_key in captured:
                    mlp_out = captured[mlp_key][0, last_pos].detach().cpu().float().numpy()
                else:
                    # fallback: mlp_out = h_output - h_after_attn
                    # h_after_attn = h_input + attn_out
                    mlp_out = h_out - h_in - attn_out
                
                # h_after_attn = h_input + attn_out
                h_after_attn = h_in + attn_out
                
                suffix = "clean" if prompt_type == "clean" else "corrupt"
                data[l][f"h_input_{suffix}"][pidx] = h_in
                data[l][f"h_output_{suffix}"][pidx] = h_out
                data[l][f"h_after_attn_{suffix}"][pidx] = h_after_attn
                data[l][f"attn_out_{suffix}"][pidx] = attn_out
                data[l][f"mlp_out_{suffix}"][pidx] = mlp_out
            
            del out, captured
            torch.cuda.empty_cache()
    
    return data, target_layers


def analyze_component_contributions(data, target_layers, model_name, d_model):
    """
    Decompose Δh at each layer into attention and MLP contributions,
    then project onto the layer's PC1 direction.
    
    Key analysis:
    1. Compute Δh = h_clean - h_corrupt at each layer output
    2. Decompose: Δh ≈ Δh_input + Δattn_out + Δmlp_out
       - Δh_input: change carried from previous layer
       - Δattn_out: new change written by attention
       - Δmlp_out: new change written by MLP
    3. Find PC1 direction at each layer
    4. Project each component onto PC1
    5. Determine: who writes the PC1 norm explosion?
    """
    log("\n" + "="*60)
    log("Component Contribution Analysis")
    log("="*60)
    
    n_pairs = data[target_layers[0]]["h_output_clean"].shape[0]
    results = {}
    
    for l in target_layers:
        d = data[l]
        
        # Δh at layer output
        dh_output = d["h_output_clean"] - d["h_output_corrupt"]  # (n_pairs, d_model)
        
        # Δh at layer input
        dh_input = d["h_input_clean"] - d["h_input_corrupt"]
        
        # Δattn_out and Δmlp_out
        d_attn = d["attn_out_clean"] - d["attn_out_corrupt"]
        d_mlp = d["mlp_out_clean"] - d["mlp_out_corrupt"]
        
        # Verify decomposition: dh_output ≈ dh_input + d_attn + d_mlp
        reconstruction = dh_input + d_attn + d_mlp
        recon_error = np.mean(np.linalg.norm(dh_output - reconstruction, axis=1)) / (np.mean(np.linalg.norm(dh_output, axis=1)) + 1e-10)
        
        # PCA on Δh_output to find PC1
        M_centered = dh_output - dh_output.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        except:
            results[str(l)] = {"skip": True}
            continue
        
        total_var = np.sum(S**2)
        if total_var < 1e-10:
            results[str(l)] = {"skip": "zero_var"}
            continue
        
        explained = (S**2) / total_var
        pc1_dir = Vt[0]  # (d_model,)
        eff_rank = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)
        
        # Project each component onto PC1
        # For each pair: projection = dot(component, pc1_dir)
        pc1_proj_output = dh_output @ pc1_dir  # (n_pairs,)
        pc1_proj_input = dh_input @ pc1_dir
        pc1_proj_attn = d_attn @ pc1_dir
        pc1_proj_mlp = d_mlp @ pc1_dir
        
        # Norms of each component
        norm_output = np.linalg.norm(dh_output, axis=1)
        norm_input = np.linalg.norm(dh_input, axis=1)
        norm_attn = np.linalg.norm(d_attn, axis=1)
        norm_mlp = np.linalg.norm(d_mlp, axis=1)
        
        # PC1 component norms
        pc1_norm_output = np.abs(pc1_proj_output)
        pc1_norm_input = np.abs(pc1_proj_input)
        pc1_norm_attn = np.abs(pc1_proj_attn)
        pc1_norm_mlp = np.abs(pc1_proj_mlp)
        
        # Contribution ratio: how much of PC1 comes from each source
        # Use absolute values since signs can cancel
        total_pc1_written = np.abs(pc1_proj_attn) + np.abs(pc1_proj_mlp) + 1e-10
        attn_pc1_frac = np.mean(np.abs(pc1_proj_attn)) / np.mean(total_pc1_written)
        mlp_pc1_frac = np.mean(np.abs(pc1_proj_mlp)) / np.mean(total_pc1_written)
        
        # Signed contribution (important for understanding additive effects)
        mean_pc1_output = np.mean(pc1_proj_output)
        mean_pc1_input = np.mean(pc1_proj_input)
        mean_pc1_attn = np.mean(pc1_proj_attn)
        mean_pc1_mlp = np.mean(pc1_proj_mlp)
        
        # Also analyze: how much of the total Δh norm is from each component
        total_norm_written = np.mean(norm_attn) + np.mean(norm_mlp) + 1e-10
        attn_norm_frac = np.mean(norm_attn) / total_norm_written
        mlp_norm_frac = np.mean(norm_mlp) / total_norm_written
        
        # PC1 direction alignment: cos(angle) between component's mean direction and PC1
        mean_attn_dir = d_attn.mean(axis=0)
        mean_mlp_dir = d_mlp.mean(axis=0)
        
        cos_attn_pc1 = np.dot(mean_attn_dir, pc1_dir) / (np.linalg.norm(mean_attn_dir) + 1e-10)
        cos_mlp_pc1 = np.dot(mean_mlp_dir, pc1_dir) / (np.linalg.norm(mean_mlp_dir) + 1e-10)
        
        # Norm explosion from input to output
        norm_explosion = np.mean(norm_output) / (np.mean(norm_input) + 1e-10)
        pc1_norm_explosion = np.mean(pc1_norm_output) / (np.mean(pc1_norm_input) + 1e-10)
        
        results[str(l)] = {
            "pc1_explained": float(explained[0]),
            "eff_rank": eff_rank,
            "recon_error": float(recon_error),
            "mean_norm": {
                "output": float(np.mean(norm_output)),
                "input": float(np.mean(norm_input)),
                "attn": float(np.mean(norm_attn)),
                "mlp": float(np.mean(norm_mlp)),
            },
            "mean_pc1_projection": {
                "output": float(mean_pc1_output),
                "input": float(mean_pc1_input),
                "attn": float(mean_pc1_attn),
                "mlp": float(mean_pc1_mlp),
            },
            "pc1_contribution_frac": {
                "attn": float(attn_pc1_frac),
                "mlp": float(mlp_pc1_frac),
            },
            "norm_contribution_frac": {
                "attn": float(attn_norm_frac),
                "mlp": float(mlp_norm_frac),
            },
            "direction_alignment_with_pc1": {
                "attn": float(cos_attn_pc1),
                "mlp": float(cos_mlp_pc1),
            },
            "norm_explosion_factor": float(norm_explosion),
            "pc1_norm_explosion_factor": float(pc1_norm_explosion),
        }
        
        log(f"  L{l}: PC1={explained[0]:.3f}, rank={eff_rank}, recon_err={recon_error:.4f}")
        log(f"    Norm: out={np.mean(norm_output):.1f}, in={np.mean(norm_input):.1f}, "
            f"attn={np.mean(norm_attn):.1f}, mlp={np.mean(norm_mlp):.1f}")
        log(f"    PC1 proj: out={mean_pc1_output:.2f}, in={mean_pc1_input:.2f}, "
            f"attn={mean_pc1_attn:.2f}, mlp={mean_pc1_mlp:.2f}")
        log(f"    PC1 contrib: attn={attn_pc1_frac:.3f}, mlp={mlp_pc1_frac:.3f}")
        log(f"    Dir align: attn→PC1={cos_attn_pc1:.3f}, mlp→PC1={cos_mlp_pc1:.3f}")
        log(f"    Norm explosion: total={norm_explosion:.2f}x, PC1={pc1_norm_explosion:.2f}x")
    
    return results


def analyze_mlp_internals(model, tokenizer, device, model_name, n_layers, d_model):
    """
    For DS7B: decompose MLP output at the collapse layer (L5) into gate/up contributions.
    
    SwiGLU: mlp_out = W_down(Swish(W_gate @ x) ⊙ (W_up @ x))
    
    We capture:
    - gate_pre: input to W_gate (after input_layernorm for MLP)
    - up_pre: same as gate_pre
    - gate_post: after W_gate (pre-Swish)
    - up_post: after W_up
    - gate_activated: Swish(gate_post)
    - gate_times_up: gate_activated ⊙ up_post
    - mlp_out: W_down(gate_times_up)
    
    Then decompose Δmlp_out into contributions from Δgate_activated and Δup_post.
    """
    log("\n" + "="*60)
    log("MLP Internal Decomposition (SwiGLU)")
    log("="*60)
    
    if model_name not in ["deepseek7b", "qwen3", "glm4"]:
        log("  Skipping: not a SwiGLU model")
        return {}
    
    n_pairs = len(TEST_PAIRS)
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    
    # Target: the collapse layer
    if model_name == "deepseek7b":
        target_layers = [4, 5, 6]
    elif model_name == "qwen3":
        target_layers = [4, 5, 8]
    else:
        target_layers = [4, 5, 10]
    
    results = {}
    
    for target_l in target_layers:
        log(f"  Analyzing MLP internals at L{target_l}...")
        
        layer = layers[target_l]
        mlp = layer.mlp
        
        # Find gate_proj, up_proj, down_proj (different naming per architecture)
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        gate_up_proj = getattr(mlp, "gate_up_proj", None)  # GLM4: merged gate+up
        is_swiglu = True
        mlp_type = type(mlp).__name__
        
        if gate_proj is not None and up_proj is not None and down_proj is not None:
            # Standard SwiGLU (DS7B, Qwen3)
            pass
        elif gate_up_proj is not None and down_proj is not None:
            # GLM4: GlmMLP with merged gate_up_proj
            is_swiglu = False
            log(f"  L{target_l}: Detected GlmMLP (gate_up_proj + down_proj)")
            import glob
            from safetensors import safe_open
            model_path = MODEL_CONFIGS[model_name]["path"]
            W_gate = None
            W_up = None
            W_down = None
            for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    for key in sf.keys():
                        if f"layers.{target_l}.mlp.gate_up_proj.weight" in key:
                            full_w = sf.get_tensor(key).float().numpy()
                            half = full_w.shape[0] // 2
                            W_gate = full_w[:half]
                            W_up = full_w[half:]
                        elif f"layers.{target_l}.mlp.down_proj.weight" in key:
                            W_down = sf.get_tensor(key).float().numpy()
                if W_gate is not None and W_down is not None:
                    break
            if W_gate is None:
                full_w = gate_up_proj.weight.detach().cpu().float().numpy()
                half = full_w.shape[0] // 2
                W_gate = full_w[:half]
                W_up = full_w[half:]
                W_down = down_proj.weight.detach().cpu().float().numpy()
        elif gate_proj is None or up_proj is None or down_proj is None:
            # Try dense_h_to_4h / dense_4h_to_h (GPT-NeoX style)
            dense_h_to_4h = getattr(mlp, "dense_h_to_4h", None)
            dense_4h_to_h = getattr(mlp, "dense_4h_to_h", None)
            if dense_h_to_4h is not None and dense_4h_to_h is not None:
                is_swiglu = False
                import glob
                from safetensors import safe_open
                model_path = MODEL_CONFIGS[model_name]["path"]
                W_gate = None
                W_up = None
                W_down = None
                for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
                    with safe_open(sf_file, framework='pt', device='cpu') as sf:
                        for key in sf.keys():
                            if f"layers.{target_l}.mlp.dense_h_to_4h.weight" in key:
                                full_w = sf.get_tensor(key).float().numpy()
                                half = full_w.shape[0] // 2
                                W_gate = full_w[:half]
                                W_up = full_w[half:]
                            elif f"layers.{target_l}.mlp.dense_4h_to_h.weight" in key:
                                W_down = sf.get_tensor(key).float().numpy()
                    if W_gate is not None and W_down is not None:
                        break
                if W_gate is None:
                    W_h_to_4h = dense_h_to_4h.weight.detach().cpu().float().numpy()
                    W_4h_to_h = dense_4h_to_h.weight.detach().cpu().float().numpy()
                    half = W_h_to_4h.shape[0] // 2
                    W_gate = W_h_to_4h[:half]
                    W_up = W_h_to_4h[half:]
                    W_down = W_4h_to_h
            else:
                log(f"  L{target_l}: Unknown MLP type ({mlp_type}), skipping MLP internals")
                continue
        
        if not is_swiglu and W_gate is not None:
            # W_gate, W_up, W_down already set from GLM4 or dense_h_to_4h
            pass
        elif gate_proj is None:
            log(f"  L{target_l}: No gate_proj found, skipping MLP internals")
            continue
        
        # Get weight matrices
        if is_swiglu:
            W_gate = gate_proj.weight.detach().cpu().float().numpy()  # (d_ff, d_model)
            W_up = up_proj.weight.detach().cpu().float().numpy()      # (d_ff, d_model)
            W_down = down_proj.weight.detach().cpu().float().numpy()  # (d_model, d_ff)
        # else: W_gate, W_up, W_down already set above for GLM4
        
        # Check if weights are on meta device (offloaded)
        if W_gate.shape[0] < 10:
            log(f"  L{target_l}: W_gate seems invalid (shape={W_gate.shape}), loading from safetensors")
            import glob
            from safetensors import safe_open
            model_path = MODEL_CONFIGS[model_name]["path"]
            for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    for key in sf.keys():
                        if f"layers.{target_l}.mlp.gate_proj.weight" in key:
                            W_gate = sf.get_tensor(key).float().numpy()
                        elif f"layers.{target_l}.mlp.up_proj.weight" in key:
                            W_up = sf.get_tensor(key).float().numpy()
                        elif f"layers.{target_l}.mlp.down_proj.weight" in key:
                            W_down = sf.get_tensor(key).float().numpy()
                        elif f"layers.{target_l}.mlp.dense_h_to_4h.weight" in key:
                            full_w = sf.get_tensor(key).float().numpy()
                            half = full_w.shape[0] // 2
                            W_gate = full_w[:half]
                            W_up = full_w[half:]
                        elif f"layers.{target_l}.mlp.dense_4h_to_h.weight" in key:
                            W_down = sf.get_tensor(key).float().numpy()
        
        d_ff = W_gate.shape[0]
        log(f"  L{target_l}: W_gate={W_gate.shape}, W_up={W_up.shape}, W_down={W_down.shape}")
        
        # We need to capture the MLP input (post-attention, post-layernorm for MLP)
        # The MLP input goes through the post-attention layernorm first
        # For Llama-style: layer.post_attention_layernorm
        
        # Instead of hooking inside MLP, we'll reconstruct from captured hidden states
        # We already have h_after_attn from collect_component_data
        # MLP input = post_attention_layernorm(h_after_attn)
        
        # Get post-attention layernorm weight
        ln_w = None
        for attr in ["post_attention_layernorm", "ln_2", "input_layernorm"]:
            ln_obj = getattr(layer, attr, None)
            if ln_obj is not None and hasattr(ln_obj, "weight"):
                w = ln_obj.weight
                if not w.is_meta:
                    ln_w = w.detach().cpu().float().numpy()
                    break
        
        if ln_w is None:
            # Load from safetensors
            import glob
            from safetensors import safe_open
            for attr in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
                key = f"model.layers.{target_l}.{attr}.weight"
                model_path = MODEL_CONFIGS[model_name]["path"]
                for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
                    with safe_open(sf_file, framework='pt', device='cpu') as sf:
                        if key in sf.keys():
                            ln_w = sf.get_tensor(key).float().numpy()
                            break
                if ln_w is not None:
                    break
        
        if ln_w is None:
            ln_w = np.ones(d_model, dtype=np.float32)
            log(f"  L{target_l}: No LN weight found, using ones")
        
        # Now do a mini forward pass for each pair
        # Capture hidden states at the target layer
        
        # We need to run the model and capture h_after_attn at the target layer
        captured_data = {
            "h_after_attn_clean": [],
            "h_after_attn_corrupt": [],
        }
        
        # Hook to capture h_after_attn (residual after attention, before MLP LN)
        for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
            if pidx % 10 == 0:
                log(f"    MLP pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            # Capture h_after_attn via hook
            for prompt_type, prompt in [("clean", clean_prompt), ("corrupt", corrupt_prompt)]:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                ids = inputs["input_ids"].to(input_device)
                mask = inputs["attention_mask"].to(input_device)
                
                h_after_attn_captured = {}
                
                def make_hook(layer_idx):
                    def hook(module, input, output):
                        # This is the self_attn hook - output is attn_out
                        # h_after_attn = h_input + attn_out
                        # We capture attn_out and compute h_after_attn from hidden states
                        if isinstance(output, tuple):
                            h_after_attn_captured["attn_out"] = output[0].detach()
                        else:
                            h_after_attn_captured["attn_out"] = output.detach()
                    return hook
                
                hook = layers[target_l].self_attn.register_forward_hook(make_hook(target_l))
                
                with torch.no_grad():
                    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                
                hook.remove()
                
                last_pos = ids.shape[1] - 1
                # h_after_attn = hs[l+1] is wrong; need h after attn residual add
                # Actually: hs[l] is input to layer l, hs[l+1] is output of layer l
                # h_after_attn is between: h_input + attn_out
                # We can compute it as: h_after_attn = hs[l] + attn_out
                h_input = out.hidden_states[target_l][0, last_pos].detach().cpu().float().numpy()
                
                if "attn_out" in h_after_attn_captured:
                    attn_out = h_after_attn_captured["attn_out"][0, last_pos].detach().cpu().float().numpy()
                    h_after_attn = h_input + attn_out
                else:
                    # Fallback: approximate
                    h_after_attn = h_input  # will be approximate
                
                captured_data[f"h_after_attn_{prompt_type}"].append(h_after_attn)
                
                del out, h_after_attn_captured
                torch.cuda.empty_cache()
        
        # Now compute MLP internals
        h_after_attn_clean = np.array(captured_data["h_after_attn_clean"])  # (n_pairs, d_model)
        h_after_attn_corrupt = np.array(captured_data["h_after_attn_corrupt"])
        
        # Apply post-attention layernorm
        def rms_norm(x, w, eps=1e-6):
            variance = np.mean(x ** 2, axis=-1, keepdims=True)
            x_normed = x / np.sqrt(variance + eps)
            return x_normed * w
        
        mlp_input_clean = rms_norm(h_after_attn_clean, ln_w)  # (n_pairs, d_model)
        mlp_input_corrupt = rms_norm(h_after_attn_corrupt, ln_w)
        d_mlp_input = mlp_input_clean - mlp_input_corrupt
        
        # Gate and Up projections
        gate_pre_act_clean = mlp_input_clean @ W_gate.T  # (n_pairs, d_ff)
        gate_pre_act_corrupt = mlp_input_corrupt @ W_gate.T
        up_act_clean = mlp_input_clean @ W_up.T  # (n_pairs, d_ff)
        up_act_corrupt = mlp_input_corrupt @ W_up.T
        
        # Swish activation (for SwiGLU) or GeLU (for GLM4)
        if model_name == "glm4":
            # GLM4 uses GeLU
            def activation_fn(x):
                # Approximate GeLU
                return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        else:
            activation_fn = lambda x: x / (1 + np.exp(-x))  # Swish
        
        gate_act_clean = activation_fn(gate_pre_act_clean)
        gate_act_corrupt = activation_fn(gate_pre_act_corrupt)
        
        # Gate × Up
        gate_times_up_clean = gate_act_clean * up_act_clean
        gate_times_up_corrupt = gate_act_corrupt * up_act_corrupt
        
        # Down projection
        mlp_out_clean = gate_times_up_clean @ W_down.T  # (n_pairs, d_model)
        mlp_out_corrupt = gate_times_up_corrupt @ W_down.T
        
        # Δmlp_out decomposition
        d_mlp_out = mlp_out_clean - mlp_out_corrupt
        
        # Decompose: which part of SwiGLU drives the change?
        # gate_times_up = gate_act ⊙ up_act
        # Δ(gate_times_up) ≈ Δgate_act ⊙ up_clean + gate_clean ⊙ Δup_act + Δgate_act ⊙ Δup_act
        d_gate_act = gate_act_clean - gate_act_corrupt
        d_up_act = up_act_clean - up_act_corrupt
        
        # Contribution from gate change
        gate_change_contribution = d_gate_act * up_act_clean
        # Contribution from up change
        up_change_contribution = gate_act_clean * d_up_act
        # Interaction term
        interaction = d_gate_act * d_up_act
        
        # Project through W_down
        gate_change_mlp = gate_change_contribution @ W_down.T
        up_change_mlp = up_change_contribution @ W_down.T
        interaction_mlp = interaction @ W_down.T
        
        # Norms
        norm_d_mlp_out = np.mean(np.linalg.norm(d_mlp_out, axis=1))
        norm_gate_change = np.mean(np.linalg.norm(gate_change_mlp, axis=1))
        norm_up_change = np.mean(np.linalg.norm(up_change_mlp, axis=1))
        norm_interaction = np.mean(np.linalg.norm(interaction_mlp, axis=1))
        
        total_change = norm_gate_change + norm_up_change + norm_interaction + 1e-10
        
        # Sparsity analysis: how many gate neurons are active?
        gate_sparsity_clean = np.mean(gate_act_clean > 0.01)
        gate_sparsity_corrupt = np.mean(gate_act_corrupt > 0.01)
        
        # Top contributing neurons: which gate/up neurons drive the most MLP output change?
        # For each pair, find the top-k gate neurons where |Δ(gate_act)| is largest
        d_gate_abs = np.abs(d_gate_act)  # (n_pairs, d_ff)
        top10_gate_neurons = np.argsort(np.mean(d_gate_abs, axis=0))[::-1][:10]
        
        # Check: are these top neurons the same ones with high gate activation?
        mean_gate_act = np.mean(gate_act_clean, axis=0)  # (d_ff,)
        top10_active_neurons = np.argsort(mean_gate_act)[::-1][:10]
        
        overlap = len(set(top10_gate_neurons) & set(top10_active_neurons))
        
        # Neuron concentration: top-10 gate change neurons account for what fraction of total gate change?
        total_gate_change = np.sum(d_gate_abs)
        top10_gate_change = np.sum(np.mean(d_gate_abs, axis=0)[top10_gate_neurons])
        concentration_top10 = top10_gate_change / (total_gate_change + 1e-10)
        
        # Also compute: top-1, top-5, top-20 concentration
        top1_conc = np.sum(np.mean(d_gate_abs, axis=0)[np.argsort(np.mean(d_gate_abs, axis=0))[::-1][:1]]) / (total_gate_change + 1e-10)
        top5_conc = np.sum(np.mean(d_gate_abs, axis=0)[np.argsort(np.mean(d_gate_abs, axis=0))[::-1][:5]]) / (total_gate_change + 1e-10)
        top20_conc = np.sum(np.mean(d_gate_abs, axis=0)[np.argsort(np.mean(d_gate_abs, axis=0))[::-1][:20]]) / (total_gate_change + 1e-10)
        
        results[str(target_l)] = {
            "d_ff": int(d_ff),
            "gate_up_decomposition": {
                "gate_change_norm": float(norm_gate_change),
                "up_change_norm": float(norm_up_change),
                "interaction_norm": float(norm_interaction),
                "total_mlp_out_change_norm": float(norm_d_mlp_out),
                "gate_change_frac": float(norm_gate_change / total_change),
                "up_change_frac": float(norm_up_change / total_change),
                "interaction_frac": float(norm_interaction / total_change),
            },
            "gate_sparsity": {
                "clean_frac_active": float(gate_sparsity_clean),
                "corrupt_frac_active": float(gate_sparsity_corrupt),
            },
            "neuron_concentration": {
                "top1": float(top1_conc),
                "top5": float(top5_conc),
                "top10": float(concentration_top10),
                "top20": float(top20_conc),
            },
            "top10_change_vs_active_overlap": int(overlap),
        }
        
        log(f"  L{target_l}: d_ff={d_ff}")
        log(f"    Gate/Up decomposition: gate={norm_gate_change:.2f}, up={norm_up_change:.2f}, "
            f"interaction={norm_interaction:.2f}")
        log(f"    Fractions: gate={norm_gate_change/total_change:.3f}, "
            f"up={norm_up_change/total_change:.3f}, inter={norm_interaction/total_change:.3f}")
        log(f"    Neuron concentration: top1={top1_conc:.4f}, top5={top5_conc:.4f}, "
            f"top10={concentration_top10:.4f}, top20={top20_conc:.4f}")
        log(f"    Gate sparsity: clean={gate_sparsity_clean:.3f}, corrupt={gate_sparsity_corrupt:.3f}")
    
    return results


def run_model(model_name):
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    
    log(f"\n{'='*60}")
    log(f"Phase 371: {model_name}")
    log(f"{'='*60}")
    
    # Load model
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    log(f"  Model loaded in {time.time()-t0:.1f}s")
    
    # Part 1: Collect component data
    log("\n--- Part 1: Collecting component data ---")
    t0 = time.time()
    data, target_layers = collect_component_data(model, tokenizer, device, model_name, n_layers, d_model)
    log(f"  Data collection done in {time.time()-t0:.1f}s")
    
    # Part 2: Analyze component contributions
    log("\n--- Part 2: Component contribution analysis ---")
    comp_results = analyze_component_contributions(data, target_layers, model_name, d_model)
    
    # Part 3: MLP internals
    log("\n--- Part 3: MLP internal decomposition ---")
    mlp_results = analyze_mlp_internals(model, tokenizer, device, model_name, n_layers, d_model)
    
    # Combine results
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": len(TEST_PAIRS),
        "target_layers": target_layers,
        "component_contributions": comp_results,
        "mlp_internals": mlp_results,
    }
    
    # Save
    os.makedirs("results/phase371_l5_source", exist_ok=True)
    out_path = f"results/phase371_l5_source/{model_name}_phase371.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"  Results saved to {out_path}")
    
    # Release model
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
    
    log("\nPhase 371 complete!")


if __name__ == "__main__":
    main()
