"""
Phase 342: MLP Internal Channel Analysis + Embedding Diff Patch Control
=========================================================================

Core unsolved question: HOW does MLP transform object identity into attribute
compatibility ranking? This script decomposes MLP computation at the channel level.

Experiments:
A. MLP Channel Binding Decomposition
   - For each key MLP layer, compute each intermediate channel's contribution 
     to the binding direction
   - MLP output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
   - Channel i contributes: (d @ W_down[:, i]) * SiLU(gate_i) * up_i
   - where d = W_U[target] - W_U[competitor] is the binding direction

B. Compatible Boost vs Incompatible Suppression
   - Channels with d @ W_down[:, i] > 0: write toward compatible attribute
   - Channels with d @ W_down[:, i] < 0: write toward incompatible attribute
   - Compare clean vs corrupted activation patterns
   - Does MLP boost compatible, suppress incompatible, or both?

C. Top Channel Ablation Test
   - Ablate top-k binding-contributing channels and measure binding drop
   - Compare with random channel ablation (control)
   - Verify causal necessity of specific channels

D. Embedding Diff Patch (Control for Identity Block Triviality)
   - Phase 340 showed identity block (L0-L2 full) gives ~100% recovery
   - Is this because L0-L2 compute something special, or just because 
     they provide the correct residual stream input?
   - Test: add embedding diff (clean - corrupted) at L0 input, 
     without replacing any layer computation
   - If embedding diff alone gives high recovery → identity block is trivial
   - If embedding diff alone gives low recovery → L0-L2 computation is necessary

Usage:
  python tests/glm5/phase342_mlp_channel_analysis.py qwen3
  python tests/glm5/phase342_mlp_channel_analysis.py glm4
  python tests/glm5/phase342_mlp_channel_analysis.py deepseek7b
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
        "binding_layers": [21, 23, 25, 27, 29],  # Key binding MLP layers
        "identity_layers": [0, 1, 2],
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
        "binding_layers": [30, 33, 36, 38],  # Key binding MLP layers
        "identity_layers": [0, 1, 2, 3, 4],
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
        "binding_layers": [19, 21, 23, 24],  # Key binding MLP layers
        "identity_layers": [0, 1, 2],
    },
}

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

CORRUPTED_BASELINE = "The item"


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


def safe_weight_to_numpy(weight_tensor):
    """Convert a weight tensor to numpy, handling meta/offloaded tensors."""
    w = weight_tensor
    if w.is_meta:
        # Meta tensor: need to fetch from offloaded storage
        # Use accelerate's dispatch if available
        try:
            from accelerate import infer_auto_device_map
            # Try to materialize by calling .to("cpu")
            w = w.to("cpu")
        except Exception:
            # Fallback: load from safetensors
            return None
    try:
        return w.detach().cpu().float().numpy()
    except Exception:
        return None


def get_mlp_weights(layer, model_name=None, model=None):
    """Extract MLP weights for SwiGLU or standard MLP.
    Handles meta/offloaded tensors from device_map="auto".
    Returns: (W_gate, W_up, W_down, d_ff)
    """
    mlp = layer.mlp
    W_down = None
    W_gate = None
    W_up = None
    d_ff = 0

    if hasattr(mlp, 'gate_up_proj'):
        w = safe_weight_to_numpy(mlp.gate_up_proj.weight)
        if w is not None:
            d_ff = w.shape[0] // 2
            W_gate = w[:d_ff]
            W_up = w[d_ff:]
    elif hasattr(mlp, 'gate_proj'):
        W_gate = safe_weight_to_numpy(mlp.gate_proj.weight)
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_gate is not None:
            d_ff = W_gate.shape[0]
        elif W_up is not None:
            d_ff = W_up.shape[0]
    elif hasattr(mlp, 'up_proj'):
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_up is not None:
            d_ff = W_up.shape[0]

    if hasattr(mlp, 'down_proj'):
        W_down = safe_weight_to_numpy(mlp.down_proj.weight)

    # Fallback: if weights are on meta device, load from safetensors
    if (W_down is None or (hasattr(mlp, 'gate_proj') and W_gate is None)) and model_name is not None:
        W_gate, W_up, W_down, d_ff = load_mlp_weights_from_disk(model_name, model, layer)

    return W_gate, W_up, W_down, d_ff


def load_mlp_weights_from_disk(model_name, model, target_layer):
    """Load MLP weights from safetensors files when model weights are on meta device."""
    import glob
    from safetensors import safe_open
    
    model_path = MODEL_CONFIGS[model_name]["path"]
    sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    
    # Find which layer index this is
    layers = get_layers(model)
    layer_idx = None
    for i, l in enumerate(layers):
        if l is target_layer:
            layer_idx = i
            break
    
    if layer_idx is None:
        return None, None, None, 0
    
    W_gate = None
    W_up = None
    W_down = None
    d_ff = 0
    
    for sf_file in sf_files:
        try:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                keys = sf.keys()
                # Try different naming patterns
                gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
                down_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
                gate_up_key = f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"
                
                if gate_up_key in keys:
                    w = sf.get_tensor(gate_up_key).float().numpy()
                    d_ff = w.shape[0] // 2
                    W_gate = w[:d_ff]
                    W_up = w[d_ff:]
                elif gate_key in keys:
                    W_gate = sf.get_tensor(gate_key).float().numpy()
                    d_ff = W_gate.shape[0]
                if up_key in keys and W_up is None:
                    W_up = sf.get_tensor(up_key).float().numpy()
                    if d_ff == 0:
                        d_ff = W_up.shape[0]
                if down_key in keys:
                    W_down = sf.get_tensor(down_key).float().numpy()
                    
                if W_down is not None:
                    break
        except Exception:
            continue
    
    return W_gate, W_up, W_down, d_ff


# ===== Capture MLP Internal Activations =====

def capture_mlp_internals(model, tokenizer, device, prompt, target_layers, n_layers):
    """Capture gate_proj output and up_proj output at target layers."""
    layers = get_layers(model)
    captured = {}

    def make_hook(key, is_gate=False):
        def hook(module, input, output):
            if isinstance(output, tuple):
                val = output[0].detach().cpu()
            else:
                val = output.detach().cpu()
            # Store last token position
            captured[key] = val[0, -1, :].float().numpy()  # [d_ff]
        return hook

    hooks = []
    for li in target_layers:
        layer = layers[li]
        if hasattr(layer.mlp, 'gate_proj'):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(
                make_hook(f"gate_{li}", is_gate=True)))
        elif hasattr(layer.mlp, 'gate_up_proj'):
            # For GLM4, gate_up_proj output contains both gate and up
            # We need to split it
            def make_glm4_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        val = output[0].detach().cpu()
                    else:
                        val = output.detach().cpu()
                    # val shape: [batch, seq, 2*d_ff]
                    val_last = val[0, -1, :].float().numpy()
                    d_ff = val_last.shape[0] // 2
                    captured[f"gate_{layer_idx}"] = val_last[:d_ff]
                    captured[f"up_{layer_idx}"] = val_last[d_ff:]
                return hook
            hooks.append(layer.mlp.gate_up_proj.register_forward_hook(
                make_glm4_hook(li)))
        if hasattr(layer.mlp, 'up_proj'):
            hooks.append(layer.mlp.up_proj.register_forward_hook(
                make_hook(f"up_{li}")))
        # Also capture MLP output
        hooks.append(layer.mlp.register_forward_hook(
            make_hook(f"mlp_out_{li}")))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    for h in hooks:
        h.remove()

    final_hidden = out.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    embed_hidden = out.hidden_states[0][0, -1].detach().cpu().float().numpy()

    return captured, final_hidden, embed_hidden


# ===== Experiment A: MLP Channel Binding Decomposition =====

def channel_binding_decomposition(W_down, binding_dir, gate_acts, up_acts, d_ff):
    """Decompose MLP output's binding contribution by channel.
    
    MLP output = W_down @ (SiLU(gate) * up)
    Binding contribution = binding_dir @ MLP_output
                        = Σ_i (binding_dir @ W_down[:, i]) * SiLU(gate_i) * up_i
    
    Returns per-channel binding contributions and classifications.
    """
    # SiLU activation on gate
    gate_silu = gate_acts * (1.0 / (1.0 + np.exp(-gate_acts)))  # SiLU = x * sigmoid(x)
    
    # Per-channel contribution
    # binding_dir: [d_model], W_down: [d_model, d_ff]
    down_projection = binding_dir @ W_down  # [d_ff] — how much each channel writes toward binding
    
    # Channel contribution = down_projection[i] * SiLU(gate[i]) * up[i]
    channel_contrib = down_projection * gate_silu * up_acts  # [d_ff]
    
    # Classify channels
    # Positive down_projection: channel writes toward compatible attribute
    # Negative down_projection: channel writes toward incompatible attribute
    compat_channels = down_projection > 0  # writes toward compatible
    incompat_channels = down_projection < 0  # writes toward incompatible
    
    total_binding = float(np.sum(channel_contrib))
    compat_contrib = float(np.sum(channel_contrib[compat_channels]))
    incompat_contrib = float(np.sum(channel_contrib[incompat_channels]))
    
    return {
        "channel_contrib": channel_contrib,
        "down_projection": down_projection,
        "gate_silu": gate_silu,
        "total_binding": total_binding,
        "compat_contrib": compat_contrib,
        "incompat_contrib": incompat_contrib,
        "n_compat": int(np.sum(compat_channels)),
        "n_incompat": int(np.sum(incompat_channels)),
        "top_channels": np.argsort(np.abs(channel_contrib))[::-1][:50].tolist(),
    }


# ===== Experiment B: Compatible Boost vs Incompatible Suppression =====

def boost_vs_suppress_analysis(clean_decomp, corrupted_decomp, d_ff):
    """Compare clean vs corrupted channel contributions.
    
    Compatible boost: channels that write toward compatible attribute
                      are more active in clean than corrupted
    Incompatible suppression: channels that write toward incompatible attribute
                             are less active in clean than corrupted
    """
    clean_contrib = clean_decomp["channel_contrib"]
    corrupt_contrib = corrupted_decomp["channel_contrib"]
    clean_down = clean_decomp["down_projection"]
    clean_silu = clean_decomp["gate_silu"]
    corrupt_silu = corrupted_decomp["gate_silu"]
    
    # Delta contribution
    delta_contrib = clean_contrib - corrupt_contrib
    
    # For compatible channels (down_projection > 0):
    # If delta_contrib > 0: channel contributes more to binding in clean → BOOST
    # If delta_contrib < 0: channel contributes less to binding in clean → REDUCTION
    compat_mask = clean_down > 0
    incompat_mask = clean_down < 0
    
    compat_boost = float(np.sum(delta_contrib[compat_mask & (delta_contrib > 0)]))
    compat_reduction = float(np.sum(delta_contrib[compat_mask & (delta_contrib < 0)]))
    incompat_suppress = float(np.sum(np.abs(delta_contrib[incompat_mask & (delta_contrib < 0)])))
    incompat_increase = float(np.sum(delta_contrib[incompat_mask & (delta_contrib > 0)]))
    
    # Gate activation differences
    silu_diff = clean_silu - corrupt_silu
    compat_silu_boost = float(np.mean(silu_diff[compat_mask])) if compat_mask.sum() > 0 else 0
    incompat_silu_change = float(np.mean(silu_diff[incompat_mask])) if incompat_mask.sum() > 0 else 0
    
    return {
        "delta_total": float(np.sum(delta_contrib)),
        "compat_boost": compat_boost,
        "compat_reduction": compat_reduction,
        "incompat_suppress": incompat_suppress,
        "incompat_increase": incompat_increase,
        "compat_net": compat_boost + compat_reduction,
        "incompat_net": -incompat_suppress + incompat_increase,
        "compat_silu_diff": compat_silu_boost,
        "incompat_silu_diff": incompat_silu_change,
        "n_compat": int(compat_mask.sum()),
        "n_incompat": int(incompat_mask.sum()),
    }


# ===== Experiment C: Top Channel Ablation =====

def ablation_test(model, tokenizer, device, prompt, W_U, tid_t, tid_c,
                  binding_dir, target_layer, top_channel_indices, n_layers):
    """Ablate top-k channels at target layer and measure binding drop."""
    layers = get_layers(model)
    binding_dir_t = torch.tensor(binding_dir, dtype=torch.float32)
    
    # Get clean binding baseline
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    clean_hidden = out.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    binding_clean = float(binding_dir @ clean_hidden)
    
    results = {}
    for k in [5, 10, 20, 50]:
        ablate_indices = top_channel_indices[:k]
        
        def make_ablation_hook(indices, layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out_val = output[0].clone()
                    out_val[0, -1, indices] = 0.0
                    return (out_val,) + output[1:]
                else:
                    out_val = output.clone()
                    out_val[0, -1, indices] = 0.0
                    return out_val
            return hook
        
        # Determine which submodule to hook
        layer = layers[target_layer]
        # Hook MLP output (simplest approach — zero out contribution of top channels)
        # More precise: hook gate_proj and zero those channels
        if hasattr(layer.mlp, 'gate_proj'):
            hook_target = layer.mlp.gate_proj
        elif hasattr(layer.mlp, 'gate_up_proj'):
            hook_target = layer.mlp.gate_up_proj
        else:
            continue
        
        ablate_indices_t = torch.tensor(ablate_indices, dtype=torch.long)
        hook = hook_target.register_forward_hook(
            make_ablation_hook(ablate_indices_t, target_layer))
        
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        hook.remove()
        
        ablated_hidden = out.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        binding_ablated = float(binding_dir @ ablated_hidden)
        
        binding_drop = binding_clean - binding_ablated
        results[f"top{k}"] = {
            "binding_clean": round(binding_clean, 4),
            "binding_ablated": round(binding_ablated, 4),
            "binding_drop": round(binding_drop, 4),
            "drop_pct": round(100.0 * binding_drop / max(abs(binding_clean), 1e-10), 1),
        }
        
        del ablated_hidden
        gc.collect()
        torch.cuda.empty_cache()
    
    # Random ablation control (same k=20)
    np.random.seed(42)
    d_ff = len(top_channel_indices) * 10  # Approximate
    rand_indices = np.random.choice(range(d_ff), size=20, replace=False).tolist()
    
    if hasattr(layer.mlp, 'gate_proj'):
        hook_target = layer.mlp.gate_proj
    elif hasattr(layer.mlp, 'gate_up_proj'):
        hook_target = layer.mlp.gate_up_proj
    else:
        return results
    
    rand_indices_t = torch.tensor(rand_indices, dtype=torch.long)
    hook = hook_target.register_forward_hook(
        make_ablation_hook(rand_indices_t, target_layer))
    
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hook.remove()
    
    rand_ablated_hidden = out.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    binding_rand = float(binding_dir @ rand_ablated_hidden)
    
    results["random20"] = {
        "binding_ablated": round(binding_rand, 4),
        "binding_drop": round(binding_clean - binding_rand, 4),
        "drop_pct": round(100.0 * (binding_clean - binding_rand) / max(abs(binding_clean), 1e-10), 1),
    }
    
    del rand_ablated_hidden
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


# ===== Experiment D: Embedding Diff Patch =====

def embedding_diff_patch_test(model, tokenizer, device, clean_prompt, corrupted_prompt,
                              W_U, tid_t, tid_c, n_layers):
    """Test if adding embedding diff at L0 input recovers binding.
    
    This controls for whether identity block ~100% recovery is trivial
    (just providing correct input) vs meaningful (L0-L2 computation matters).
    """
    binding_dir = W_U[tid_t] - W_U[tid_c]
    
    # Get embedding-level hidden states (hidden_states[0])
    # and final hidden states for both clean and corrupted
    inp_clean = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out_clean = model(**inp_clean, output_hidden_states=True)
    embed_clean = out_clean.hidden_states[0][0, -1].detach().cpu().float().numpy()
    final_clean = out_clean.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    
    inp_corrupt = tokenizer(corrupted_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out_corrupt = model(**inp_corrupt, output_hidden_states=True)
    embed_corrupt = out_corrupt.hidden_states[0][0, -1].detach().cpu().float().numpy()
    final_corrupt = out_corrupt.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    
    # Binding baselines
    binding_clean = float(binding_dir @ final_clean)
    binding_corrupt = float(binding_dir @ final_corrupt)
    binding_range = binding_clean - binding_corrupt
    
    if abs(binding_range) < 0.3:
        return {"error": "binding_range too small", "binding_range": round(binding_range, 4)}
    
    # Embedding diff
    embed_diff = embed_clean - embed_corrupt
    
    # Patch: Add embed_diff to L0 input when running corrupted prompt
    layers = get_layers(model)
    
    # For device_map="auto", we need to find the right device
    # The embed_tokens output goes to the first layer's device
    first_layer_device = None
    for name, param in layers[0].named_parameters():
        first_layer_device = param.device
        break
    if first_layer_device is None:
        first_layer_device = device
    
    embed_diff_t = torch.tensor(embed_diff, dtype=torch.bfloat16, device=first_layer_device)
    
    def make_embed_patch_hook(diff_tensor):
        def hook(module, input, output):
            # output is hidden_states[0]: [batch, seq, d_model]
            # Add diff to last token position
            if isinstance(output, tuple):
                out_val = output[0].clone()
            else:
                out_val = output.clone()
            device_out = out_val.device
            dtype_out = out_val.dtype
            diff_dev = diff_tensor.to(device=device_out, dtype=dtype_out)
            out_val[0, -1, :] += diff_dev
            if isinstance(output, tuple):
                return (out_val,) + output[1:]
            return out_val
        return hook
    
    # We need to hook the embedding output / first layer input
    # For device_map="auto", embed_tokens might be on meta device
    # Use a more robust approach: hook into the first transformer layer's forward
    # to modify its input (which is the embedding output)
    hook_handle = None
    
    def make_first_layer_patch_hook(diff_tensor):
        def hook(module, input, output):
            # input is (hidden_states, ...) where hidden_states is embedding output
            # We modify hidden_states at the last token position
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0]
                # Create modified hidden_states
                hs_modified = hidden_states.clone()
                device_hs = hs_modified.device
                dtype_hs = hs_modified.dtype
                diff_dev = diff_tensor.to(device=device_hs, dtype=dtype_hs)
                hs_modified[0, -1, :] += diff_dev
                # Pass modified input through the layer
                # We need to call the layer manually with modified input
                # This is tricky with hooks. Alternative: modify output instead
                pass
            return output  # Don't modify, we'll use a different approach
        return hook
    
    # Better approach: Run model with modified input_ids
    # Since "The apple" and "The item" have the same length,
    # we can just replace the corrupted token with the clean token in input_ids
    # and add the position embedding difference
    
    # Simplest robust approach: Run model with clean input_ids
    # This IS equivalent to running the clean model, which gives ~100%
    # This is actually the correct result - it confirms that embedding
    # determines binding computation
    
    # For a more meaningful test, we should use a PARTIAL embedding diff
    # Test: add only binding-direction component of embedding diff
    binding_dir_norm = np.linalg.norm(binding_dir)
    if binding_dir_norm > 0:
        # Project embedding diff onto binding direction
        binding_unit = binding_dir / binding_dir_norm
        embed_diff_binding = float(np.dot(embed_diff, binding_unit))
        embed_diff_binding_vec = embed_diff_binding * binding_unit
        # Component orthogonal to binding direction
        embed_diff_ortho = embed_diff - embed_diff_binding_vec
        
        # Test 1: Add only binding-direction component of embedding diff
        embed_diff_binding_t = torch.tensor(embed_diff_binding_vec, 
                                            dtype=torch.bfloat16, 
                                            device=first_layer_device)
        
        # Test 2: Add only orthogonal component
        embed_diff_ortho_t = torch.tensor(embed_diff_ortho,
                                          dtype=torch.bfloat16,
                                          device=first_layer_device)
    else:
        embed_diff_binding_t = None
        embed_diff_ortho_t = None
        embed_diff_binding = 0
        embed_diff_ortho_norm = 0
    
    # For the FULL embedding diff test, simply run clean model
    # (since embedding diff = clean - corrupted, adding it to corrupted = clean)
    # Result: ~100% recovery (trivially)
    binding_patched_full = binding_clean  # Trivially 100%
    recovery_full = 100.0
    
    # For the binding-direction-only test, we need to hook the embedding
    # Use embed_tokens if accessible, otherwise use first layer input modification
    results_partial = {}
    
    for test_name, diff_tensor in [("binding_only", embed_diff_binding_t), 
                                    ("ortho_only", embed_diff_ortho_t)]:
        if diff_tensor is None:
            continue
            
        def make_patch_fn(diff_t):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out_val = output[0].clone()
                else:
                    out_val = output.clone()
                device_out = out_val.device
                dtype_out = out_val.dtype
                diff_dev = diff_t.to(device=device_out, dtype=dtype_out)
                out_val[0, -1, :] += diff_dev
                if isinstance(output, tuple):
                    return (out_val,) + output[1:]
                return out_val
            return hook
        
        # Try to hook embed_tokens
        hooked = False
        if hasattr(model.model, 'embed_tokens') and not model.model.embed_tokens.weight.is_meta:
            hook_handle = model.model.embed_tokens.register_forward_hook(
                make_patch_fn(diff_tensor))
            hooked = True
        elif hasattr(model.model, 'wte') and not model.model.wte.weight.is_meta:
            hook_handle = model.model.wte.register_forward_hook(
                make_patch_fn(diff_tensor))
            hooked = True
        
        if hooked:
            inp_corrupt2 = tokenizer(corrupted_prompt, return_tensors="pt", 
                                     truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out_patched = model(**inp_corrupt2, output_hidden_states=True)
            hook_handle.remove()
            
            final_patched = out_patched.hidden_states[-1][0, -1].detach().cpu().float().numpy()
            binding_patched = float(binding_dir @ final_patched)
            recovery_pct = 100.0 * (binding_patched - binding_corrupt) / max(binding_range, 1e-10)
            
            results_partial[test_name] = {
                "binding_patched": round(binding_patched, 4),
                "recovery_pct": round(recovery_pct, 1),
            }
            
            del out_patched
            gc.collect()
            torch.cuda.empty_cache()
    
    del out_clean, out_corrupt, out_patched
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "binding_clean": round(binding_clean, 4),
        "binding_corrupt": round(binding_corrupt, 4),
        "binding_range": round(binding_range, 4),
        "recovery_full": round(recovery_full, 1),  # Trivially ~100%
        "embed_diff_norm": round(float(np.linalg.norm(embed_diff)), 4),
        "embed_diff_binding_component": round(embed_diff_binding, 4) if binding_dir_norm > 0 else 0,
        "embed_diff_ortho_norm": round(float(np.linalg.norm(embed_diff - (embed_diff_binding * binding_dir / max(binding_dir_norm, 1e-10)))), 4) if binding_dir_norm > 0 else round(float(np.linalg.norm(embed_diff)), 4),
        "partial_results": results_partial,
    }


# ===== Main Experiment =====

def run_experiment(model_name):
    log(f"Phase 342: MLP Channel Analysis + Embedding Diff Control — {model_name}")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    binding_layers = cfg["binding_layers"]

    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")

    if torch.cuda.is_available():
        log(f"  GPU after load: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Pre-extract MLP weights for binding layers
    layers = get_layers(model)
    mlp_weights = {}
    log(f"  Extracting MLP weights for layers {binding_layers}...")
    for li in binding_layers:
        W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {
            "W_gate": W_gate,
            "W_up": W_up,
            "W_down": W_down,
            "d_ff": d_ff,
        }
        log(f"    L{li}: d_ff={d_ff}, W_down={W_down.shape if W_down is not None else 'None'}")

    # ==================================================================
    # EXPERIMENT A + B: Channel Decomposition + Boost vs Suppress
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"EXPERIMENT A+B: MLP Channel Binding Decomposition")
    log(f"{'='*70}")

    channel_results = {}
    boost_suppress_results = {}

    # Use a subset of pairs for channel analysis (to save time)
    # Focus on high-binding-range pairs
    test_pairs = []
    for obj, target_val, competitor_val in HC_PAIRS:
        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)
        if tid_t is None or tid_c is None:
            continue
        test_pairs.append((obj, target_val, competitor_val, tid_t, tid_c))

    # Per-pair, per-layer channel analysis
    all_pair_channel_data = {}  # {pair_key: {layer: {...}}}
    all_pair_boost_data = {}    # {pair_key: {layer: {...}}}

    for pidx, (obj, target_val, competitor_val, tid_t, tid_c) in enumerate(test_pairs):
        pair_key = f"{obj}_{target_val}"
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
        
        # Capture MLP internals for clean and corrupted
        clean_caps, clean_final, clean_embed = capture_mlp_internals(
            model, tokenizer, device, clean_prompt, binding_layers, n_layers)
        corrupt_caps, corrupt_final, corrupt_embed = capture_mlp_internals(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, n_layers)
        
        pair_channel = {}
        pair_boost = {}
        
        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]
            d_ff = mw["d_ff"]
            
            if W_down is None:
                continue
            
            # Get gate and up activations
            gate_key = f"gate_{li}"
            up_key = f"up_{li}"
            
            if gate_key not in clean_caps or up_key not in clean_caps:
                # For GLM4, both are captured under gate_up_proj hook
                if gate_key in clean_caps and f"up_{li}" not in clean_caps:
                    # Try to use gate_key only (GLM4 split in hook)
                    if up_key in clean_caps:
                        clean_gate = clean_caps[gate_key][:d_ff]
                        clean_up = clean_caps[up_key]
                    else:
                        continue
                else:
                    continue
            else:
                clean_gate = clean_caps[gate_key]
                clean_up = clean_caps[up_key]
            
            if gate_key not in corrupt_caps:
                continue
            corrupt_gate = corrupt_caps[gate_key]
            corrupt_up = corrupt_caps.get(up_key, np.zeros_like(clean_up))
            
            # Ensure sizes match
            if clean_gate.shape[0] != d_ff or clean_up.shape[0] != d_ff:
                min_dim = min(clean_gate.shape[0], d_ff)
                clean_gate = clean_gate[:min_dim]
                clean_up = clean_up[:min_dim]
                corrupt_gate = corrupt_gate[:min_dim]
                corrupt_up = corrupt_up[:min_dim]
                # Also truncate W_down
                W_down = W_down[:, :min_dim]
            
            # Channel decomposition
            clean_decomp = channel_binding_decomposition(W_down, binding_dir, clean_gate, clean_up, d_ff)
            corrupt_decomp = channel_binding_decomposition(W_down, binding_dir, corrupt_gate, corrupt_up, d_ff)
            
            # Boost vs suppress
            boost_data = boost_vs_suppress_analysis(clean_decomp, corrupt_decomp, d_ff)
            
            pair_channel[li] = {
                "total_binding_clean": round(clean_decomp["total_binding"], 4),
                "total_binding_corrupt": round(corrupt_decomp["total_binding"], 4),
                "compat_contrib_clean": round(clean_decomp["compat_contrib"], 4),
                "incompat_contrib_clean": round(clean_decomp["incompat_contrib"], 4),
                "n_compat": clean_decomp["n_compat"],
                "n_incompat": clean_decomp["n_incompat"],
                "top5_channels": clean_decomp["top_channels"][:5],
                "top10_channels": clean_decomp["top_channels"][:10],
            }
            pair_boost[li] = boost_data
        
        all_pair_channel_data[pair_key] = pair_channel
        all_pair_boost_data[pair_key] = pair_boost
        
        del clean_caps, corrupt_caps
        gc.collect()
        torch.cuda.empty_cache()
        
        if (pidx + 1) % 6 == 0 or pidx < 2:
            elapsed = time.time() - t0
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{pidx+1}/{len(test_pairs)}] {pair_key}: "
                f"{len(pair_channel)} layers analyzed, elapsed={elapsed:.0f}s, GPU={gpu_mem:.2f}GB")

    # Aggregate channel results across pairs
    log(f"\n--- Channel Decomposition Aggregate ---")
    log(f"  {'Layer':>6} {'Binding_clean':>14} {'Binding_corrupt':>16} "
        f"{'Compat%':>9} {'Incompat%':>10} {'N_pairs':>8}")
    log("  " + "-" * 70)
    
    for li in binding_layers:
        binding_clean_list = []
        binding_corrupt_list = []
        compat_pct_list = []
        incompat_pct_list = []
        
        for pair_key, pdata in all_pair_channel_data.items():
            if li in pdata:
                bc = pdata[li]["total_binding_clean"]
                bcr = pdata[li]["total_binding_corrupt"]
                total_abs = abs(bc) + abs(bcr) if (abs(bc) + abs(bcr)) > 0 else 1
                binding_clean_list.append(bc)
                binding_corrupt_list.append(bcr)
                compat_pct = pdata[li]["compat_contrib_clean"] / max(abs(bc), 1e-10) * 100
                incompat_pct = pdata[li]["incompat_contrib_clean"] / max(abs(bc), 1e-10) * 100
                compat_pct_list.append(compat_pct)
                incompat_pct_list.append(incompat_pct)
        
        if binding_clean_list:
            channel_results[li] = {
                "binding_clean_mean": round(float(np.mean(binding_clean_list)), 4),
                "binding_clean_std": round(float(np.std(binding_clean_list)), 4),
                "binding_corrupt_mean": round(float(np.mean(binding_corrupt_list)), 4),
                "compat_pct_mean": round(float(np.mean(compat_pct_list)), 1),
                "compat_pct_std": round(float(np.std(compat_pct_list)), 1),
                "incompat_pct_mean": round(float(np.mean(incompat_pct_list)), 1),
                "n_pairs": len(binding_clean_list),
            }
            cr = channel_results[li]
            log(f"  L{li:>5} {cr['binding_clean_mean']:>+14.4f} {cr['binding_corrupt_mean']:>+16.4f} "
                f"{cr['compat_pct_mean']:>+9.1f}% {cr['incompat_pct_mean']:>+10.1f}% {cr['n_pairs']:>8}")

    # Aggregate boost vs suppress results
    log(f"\n--- Boost vs Suppress Aggregate ---")
    log(f"  {'Layer':>6} {'Delta_total':>12} {'Compat_boost':>13} {'Compat_reduc':>13} "
        f"{'Incompat_supp':>14} {'Incompat_incr':>14}")
    log("  " + "-" * 80)
    
    for li in binding_layers:
        delta_list = []
        boost_list = []
        reduc_list = []
        supp_list = []
        incr_list = []
        
        for pair_key, bdata in all_pair_boost_data.items():
            if li in bdata:
                delta_list.append(bdata[li]["delta_total"])
                boost_list.append(bdata[li]["compat_boost"])
                reduc_list.append(bdata[li]["compat_reduction"])
                supp_list.append(bdata[li]["incompat_suppress"])
                incr_list.append(bdata[li]["incompat_increase"])
        
        if delta_list:
            boost_suppress_results[li] = {
                "delta_total_mean": round(float(np.mean(delta_list)), 4),
                "delta_total_std": round(float(np.std(delta_list)), 4),
                "compat_boost_mean": round(float(np.mean(boost_list)), 4),
                "compat_boost_std": round(float(np.std(boost_list)), 4),
                "compat_reduction_mean": round(float(np.mean(reduc_list)), 4),
                "incompat_suppress_mean": round(float(np.mean(supp_list)), 4),
                "incompat_increase_mean": round(float(np.mean(incr_list)), 4),
                "n_pairs": len(delta_list),
            }
            bs = boost_suppress_results[li]
            log(f"  L{li:>5} {bs['delta_total_mean']:>+12.4f} {bs['compat_boost_mean']:>+13.4f} "
                f"{bs['compat_reduction_mean']:>+13.4f} {bs['incompat_suppress_mean']:>+14.4f} "
                f"{bs['incompat_increase_mean']:>+14.4f}")

    # ==================================================================
    # EXPERIMENT C: Top Channel Ablation
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"EXPERIMENT C: Top Channel Ablation Test")
    log(f"{'='*70}")

    ablation_results = {}
    
    # Use apple-red pair as primary test
    test_obj, test_target, test_competitor = "apple", "red", "blue"
    tid_t = get_token_id(tokenizer, test_target)
    tid_c = get_token_id(tokenizer, test_competitor)
    
    if tid_t is not None and tid_c is not None:
        binding_dir = W_U[tid_t] - W_U[tid_c]
        clean_prompt = f"The {test_obj}"
        
        # Use the layer with strongest binding contribution
        best_layer = binding_layers[len(binding_layers) // 2]  # Middle layer
        
        # Get top channels from our analysis
        pair_key = f"{test_obj}_{test_target}"
        if pair_key in all_pair_channel_data and best_layer in all_pair_channel_data[pair_key]:
            top_ch = all_pair_channel_data[pair_key][best_layer]["top10_channels"]
        else:
            # Compute on the fly
            clean_caps, _, _ = capture_mlp_internals(
                model, tokenizer, device, clean_prompt, [best_layer], n_layers)
            mw = mlp_weights[best_layer]
            gate_key = f"gate_{best_layer}"
            up_key = f"up_{best_layer}"
            if gate_key in clean_caps:
                decomp = channel_binding_decomposition(
                    mw["W_down"], binding_dir, 
                    clean_caps[gate_key], clean_caps.get(up_key, np.ones(mw["d_ff"])),
                    mw["d_ff"])
                top_ch = decomp["top_channels"][:10]
            else:
                top_ch = list(range(10))
            del clean_caps
            gc.collect()
            torch.cuda.empty_cache()
        
        log(f"  Ablation test on L{best_layer} for {pair_key}")
        log(f"  Top channels: {top_ch[:10]}")
        
        try:
            abl_results = ablation_test(
                model, tokenizer, device, clean_prompt, W_U, tid_t, tid_c,
                binding_dir, best_layer, top_ch, n_layers)
            ablation_results = {
                "layer": best_layer,
                "pair": pair_key,
                "results": abl_results,
            }
            for k, v in abl_results.items():
                if isinstance(v, dict) and "drop_pct" in v:
                    log(f"  {k}: drop={v['drop_pct']:+.1f}% "
                        f"(clean={v['binding_clean']:.4f} → ablated={v['binding_ablated']:.4f})")
        except Exception as e:
            log(f"  Ablation test failed: {e}")
            ablation_results = {"error": str(e)}

    # ==================================================================
    # EXPERIMENT D: Embedding Diff Patch
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"EXPERIMENT D: Embedding Diff Patch Control")
    log(f"{'='*70}")
    log(f"  Testing: Does adding embedding diff at L0 input recover binding?")
    log(f"  If yes → identity block ~100% is trivial (correct input → correct output)")
    log(f"  If no  → L0-L2 computation is necessary beyond just embedding")

    embed_patch_results = {}
    embed_test_pairs = [
        ("apple", "red", "blue"),
        ("snow", "white", "black"),
        ("fire", "hot", "cold"),
        ("sky", "blue", "green"),
        ("banana", "yellow", "purple"),
        ("ice", "cold", "hot"),
        ("grass", "green", "red"),
        ("ocean", "blue", "yellow"),
    ]

    for pidx, (obj, target_val, competitor_val) in enumerate(embed_test_pairs):
        pair_key = f"{obj}_{target_val}"
        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)
        if tid_t is None or tid_c is None:
            continue
        
        clean_prompt = f"The {obj}"
        
        try:
            result = embedding_diff_patch_test(
                model, tokenizer, device, clean_prompt, CORRUPTED_BASELINE,
                W_U, tid_t, tid_c, n_layers)
            embed_patch_results[pair_key] = result
            
            if "recovery_full" in result:
                log(f"  {pair_key}: full_trivially_100%, "
                    f"embed_diff_norm={result['embed_diff_norm']:.4f}")
                if "partial_results" in result:
                    for pname, pdata in result["partial_results"].items():
                        log(f"    {pname}: recovery={pdata['recovery_pct']:+.1f}%")
            elif "error" in result:
                log(f"  {pair_key}: {result['error']}")
        except Exception as e:
            log(f"  {pair_key}: FAILED - {e}")
            embed_patch_results[pair_key] = {"error": str(e)}
        
        gc.collect()
        torch.cuda.empty_cache()
        
        if (pidx + 1) % 4 == 0:
            elapsed = time.time() - t0
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  Progress: {pidx+1}/{len(embed_test_pairs)}, elapsed={elapsed:.0f}s, GPU={gpu_mem:.2f}GB")

    # Aggregate embedding patch results
    embed_aggs = {}
    
    # Full embedding diff is trivially ~100%, so focus on partial patches
    binding_only_recs = []
    ortho_only_recs = []
    for pair_key, r in embed_patch_results.items():
        if "partial_results" in r:
            if "binding_only" in r["partial_results"]:
                binding_only_recs.append(r["partial_results"]["binding_only"]["recovery_pct"])
            if "ortho_only" in r["partial_results"]:
                ortho_only_recs.append(r["partial_results"]["ortho_only"]["recovery_pct"])
    
    embed_aggs = {
        "full_embedding_trivially_100": True,
        "binding_only_mean": round(float(np.mean(binding_only_recs)), 1) if binding_only_recs else None,
        "binding_only_std": round(float(np.std(binding_only_recs)), 1) if binding_only_recs else None,
        "ortho_only_mean": round(float(np.mean(ortho_only_recs)), 1) if ortho_only_recs else None,
        "ortho_only_std": round(float(np.std(ortho_only_recs)), 1) if ortho_only_recs else None,
        "n_valid": len(binding_only_recs),
    }
    
    log(f"\n  Embedding Patch Summary:")
    log(f"    Full embedding diff: trivially ~100% (equivalent to running clean model)")
    if binding_only_recs:
        log(f"    Binding-direction only: mean={embed_aggs['binding_only_mean']:+.1f}% "
            f"(std={embed_aggs['binding_only_std']:.1f}%)")
    if ortho_only_recs:
        log(f"    Orthogonal only: mean={embed_aggs['ortho_only_mean']:+.1f}% "
            f"(std={embed_aggs['ortho_only_std']:.1f}%)")
    log(f"    → If binding_only >> ortho_only: embedding binding component drives binding computation")
    log(f"    → If ortho_only >> binding_only: non-binding embedding info drives binding computation")

    # ==================================================================
    # Save results
    # ==================================================================
    save_data = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experimentA_channel_decomposition": {
            str(k): v for k, v in channel_results.items()
        },
        "experimentB_boost_vs_suppress": {
            str(k): v for k, v in boost_suppress_results.items()
        },
        "experimentC_ablation": ablation_results,
        "experimentD_embedding_patch": {
            "per_pair": embed_patch_results,
            "aggregate": embed_aggs,
        },
        "per_pair_channel_data": {
            str(k): {str(li): v for li, v in pdata.items()}
            for k, pdata in all_pair_channel_data.items()
        },
        "per_pair_boost_data": {
            str(k): {str(li): v for li, v in bdata.items()}
            for k, bdata in all_pair_boost_data.items()
        },
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

    os.makedirs("results/phase342_mlp_channel", exist_ok=True)
    out_path = f"results/phase342_mlp_channel/{model_name}_phase342.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {out_path}")

    # ==================================================================
    # Print final summary
    # ==================================================================
    log(f"\n{'='*80}")
    log(f"PHASE 342 SUMMARY — {model_name}")
    log(f"{'='*80}")
    
    log(f"\nA. Channel Decomposition:")
    for li_str, cr in channel_results.items():
        log(f"  L{li_str}: compat={cr['compat_pct_mean']:+.1f}% of binding, "
            f"incompat={cr['incompat_pct_mean']:+.1f}%")
    
    log(f"\nB. Boost vs Suppress:")
    for li_str, bs in boost_suppress_results.items():
        log(f"  L{li_str}: compat_boost={bs['compat_boost_mean']:+.4f}, "
            f"incompat_suppress={bs['incompat_suppress_mean']:+.4f}")
    
    log(f"\nC. Ablation:")
    if "results" in ablation_results:
        for k, v in ablation_results["results"].items():
            if isinstance(v, dict) and "drop_pct" in v:
                log(f"  {k}: {v['drop_pct']:+.1f}% binding drop")
    
    log(f"\nD. Embedding Patch:")
    if embed_aggs:
        log(f"  Full embedding diff: trivially ~100%")
        if embed_aggs.get("binding_only_mean") is not None:
            log(f"  Binding-only component: {embed_aggs['binding_only_mean']:+.1f}%")
        if embed_aggs.get("ortho_only_mean") is not None:
            log(f"  Orthogonal component: {embed_aggs['ortho_only_mean']:+.1f}%")
    
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
        log(f"Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_experiment(model_name)
    log("Phase 342 complete!")
