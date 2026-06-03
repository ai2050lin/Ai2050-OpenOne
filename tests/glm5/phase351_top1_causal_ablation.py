"""
Phase 351: Top 1% Channel Causal Ablation + Overlap Structure
==============================================================

Goals:
1. Causal test: Does ablating Top 1% channels destroy binding?
2. Overlap: Are Top 1% |cproj| channels also Top |Δact| channels? Cross-pair overlap?
3. Boost vs Suppress: Do Top 1% channels mainly boost compatible or suppress incompatible?

Method:
- Zero ablation of Top 1% |cproj| channels
- Zero ablation of Top 1% |Δact| channels
- Zero ablation of random matched channels (control)
- Measure: HC binding, NI suppression, rank metric
- Overlap: Jaccard between Top 1% cproj, Top 1% Δact, Top 1% contribution
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
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("rain", "wet", "dry"), ("desert", "hot", "cold"), ("moon", "white", "black"),
    ("smoke", "gray", "red"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("night", "dark", "bright"), ("steel", "gray", "gold"), ("ivory", "white", "black"),
]

CORRUPTED_BASELINE = "The item"


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
                if dk in keys and W_down is None:
                    W_down = sf.get_tensor(dk).float().numpy()
                if W_down is not None:
                    break
        except:
            continue
    return W_gate, W_up, W_down, d_ff


def get_mlp_weights(layer, model_name=None, model=None):
    mlp = layer.mlp
    W_gate = W_up = W_down = None; d_ff = 0
    if hasattr(mlp, 'gate_up_proj'):
        w = safe_weight_to_numpy(mlp.gate_up_proj.weight)
        if w is not None:
            d_ff = w.shape[0] // 2; W_gate, W_up = w[:d_ff], w[d_ff:]
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
    if W_down is None and model_name is not None:
        layers = get_layers(model)
        for i, l in enumerate(layers):
            if l is layer:
                W_gate, W_up, W_down, d_ff = get_mlp_weights_from_disk(model_name, i)
                break
    return W_gate, W_up, W_down, d_ff


def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


def get_logit_diff(model, tokenizer, device, prompt, target_id, competitor_id):
    """Get logit difference (target - competitor) for a prompt."""
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    return float(logits[target_id] - logits[competitor_id]), logits


def capture_mlp_internals(model, tokenizer, device, prompt, target_layers):
    """Capture MLP gate/up activations and logits without patching."""
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
                    captured[f"gate_{idx}"] = v[:d]; captured[f"up_{idx}"] = v[d:]
                return hook
            hooks.append(layer.mlp.gate_up_proj.register_forward_hook(make_glm4_hook(li)))
        if hasattr(layer.mlp, 'up_proj'):
            hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook(f"up_{li}")))
    
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    
    for h in hooks:
        h.remove()
    
    return captured, logits


def run_model_with_channel_ablation(model, tokenizer, device, prompt, target_layers,
                                     channels_to_zero_by_layer):
    """
    Run model with specific MLP intermediate channels zeroed out.
    
    This patches the down_proj input (the SiLU(gate)*up vector) by zeroing
    specified channels before they pass through down_proj.
    
    channels_to_zero_by_layer: {layer_idx: set of channel indices to zero}
    
    Returns logits after ablation.
    """
    layers = get_layers(model)
    
    def make_down_proj_input_hook(channels_to_zero):
        """Hook on down_proj that zeros specified channels in the input tensor."""
        def hook(module, args, kwargs, output):
            # args[0] is the input tensor [batch, seq, d_ff]
            # We need to modify the input, not the output
            # But with register_forward_hook we get output, not input
            # Use register_forward_pre_hook instead
            pass
        return hook
    
    # Use pre-hook on down_proj to zero channels in input
    hooks = []
    
    for li, channels_to_zero in channels_to_zero_by_layer.items():
        if li >= len(layers) or not channels_to_zero:
            continue
        
        mlp = layers[li].mlp
        ch_list = sorted(channels_to_zero)
        ch_tensor = torch.tensor(ch_list, dtype=torch.long)
        
        def make_pre_hook(ch_indices):
            def pre_hook(module, args):
                # args is a tuple of input tensors
                # First arg is the intermediate activation [batch, seq, d_ff]
                inp = args[0]
                if inp.dim() == 3 and inp.shape[-1] > max(ch_indices):
                    modified = inp.clone()
                    modified[:, :, ch_indices] = 0.0
                    return (modified,) + args[1:]
                return args
            return pre_hook
        
        # Hook on down_proj input
        hooks.append(mlp.down_proj.register_forward_pre_hook(make_pre_hook(ch_tensor)))
    
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    
    for h in hooks:
        h.remove()
    
    return logits


def run_ablation_experiment(model, tokenizer, device, model_name, W_U,
                            obj, target, competitor, binding_layers, ablation_type, 
                            ablation_channels_by_layer):
    """
    Run true causal ablation by patching MLP intermediate activations during forward pass.
    
    ablation_type: "none" (baseline), "top1_cproj", "top1_dact", "random"
    ablation_channels_by_layer: {layer_idx: set of channel indices to zero}
    
    Returns dict with logit metrics and attribution metrics.
    """
    cfg = MODEL_CONFIGS[model_name]
    tid_t = get_token_id(tokenizer, target)
    tid_c = get_token_id(tokenizer, competitor)
    if tid_t is None or tid_c is None:
        return None
    
    direction = W_U[tid_t] - W_U[tid_c]
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-10:
        return None
    direction_normed = direction / dir_norm
    
    clean_prompt = f"The {obj}"
    layers = get_layers(model)
    
    # Run clean and corrupt WITHOUT ablation to get baseline logits
    clean_caps, clean_logits = capture_mlp_internals(
        model, tokenizer, device, clean_prompt, binding_layers)
    corrupt_caps, corrupt_logits = capture_mlp_internals(
        model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)
    
    clean_diff_base = float(clean_logits[tid_t] - clean_logits[tid_c])
    corrupt_diff_base = float(corrupt_logits[tid_t] - corrupt_logits[tid_c])
    binding_effect_base = clean_diff_base - corrupt_diff_base
    
    # Attribution-based binding (from MLP weights and activations)
    total_binding_attn = 0.0
    total_binding_ablated_attn = 0.0
    per_layer_results = {}
    
    for li in binding_layers:
        W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        if W_down is None:
            continue
        gk = f"gate_{li}"; uk = f"up_{li}"
        if gk not in clean_caps or gk not in corrupt_caps:
            continue
        
        cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
        cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
        cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
        
        min_d = min(d_ff, W_down.shape[1], cg.shape[0])
        Wd = W_down[:, :min_d]
        gsc = silu_np(cg[:min_d]); gsr = silu_np(crg[:min_d])
        uc = cu[:min_d]; ur = cru[:min_d]
        dact = gsc * uc - gsr * ur
        channel_proj = Wd.T @ direction_normed
        
        full_binding = float(np.sum(channel_proj * dact))
        
        channels_to_zero = ablation_channels_by_layer.get(li, set())
        if channels_to_zero:
            mask = np.ones(min_d, dtype=bool)
            for ch in channels_to_zero:
                if ch < min_d:
                    mask[ch] = False
            ablated_binding = float(np.sum(channel_proj[mask] * dact[mask]))
        else:
            ablated_binding = full_binding
        
        total_binding_attn += full_binding
        total_binding_ablated_attn += ablated_binding
        
        per_layer_results[li] = {
            "full_binding": full_binding,
            "ablated_binding": ablated_binding,
            "binding_lost": full_binding - ablated_binding,
            "frac_lost": (full_binding - ablated_binding) / max(abs(full_binding), 1e-10),
        }
    
    # Now run TRUE causal ablation: forward pass with channels zeroed
    if ablation_type != "none" and ablation_channels_by_layer:
        # Clean with ablation
        ablated_clean_logits = run_model_with_channel_ablation(
            model, tokenizer, device, clean_prompt, binding_layers,
            ablation_channels_by_layer)
        clean_diff_ablated = float(ablated_clean_logits[tid_t] - ablated_clean_logits[tid_c])
        
        # Corrupt with ablation
        ablated_corrupt_logits = run_model_with_channel_ablation(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers,
            ablation_channels_by_layer)
        corrupt_diff_ablated = float(ablated_corrupt_logits[tid_t] - ablated_corrupt_logits[tid_c])
        
        binding_effect_ablated = clean_diff_ablated - corrupt_diff_ablated
    else:
        clean_diff_ablated = clean_diff_base
        corrupt_diff_ablated = corrupt_diff_base
        binding_effect_ablated = binding_effect_base
    
    del clean_caps, corrupt_caps, clean_logits, corrupt_logits
    gc.collect(); torch.cuda.empty_cache()
    
    return {
        "obj": obj, "target": target, "competitor": competitor,
        # True causal metrics (from logits)
        "clean_diff_base": clean_diff_base,
        "corrupt_diff_base": corrupt_diff_base,
        "binding_effect_base": binding_effect_base,
        "clean_diff_ablated": clean_diff_ablated,
        "corrupt_diff_ablated": corrupt_diff_ablated,
        "binding_effect_ablated": binding_effect_ablated,
        "binding_effect_lost": binding_effect_base - binding_effect_ablated,
        "frac_binding_effect_lost": (binding_effect_base - binding_effect_ablated) / max(abs(binding_effect_base), 1e-10),
        # Attribution metrics (from MLP weights)
        "total_binding_attn": total_binding_attn,
        "total_binding_ablated_attn": total_binding_ablated_attn,
        "frac_attn_lost": (total_binding_attn - total_binding_ablated_attn) / max(abs(total_binding_attn), 1e-10),
        "per_layer": per_layer_results,
    }


def run_overlap_analysis(model, tokenizer, device, model_name, W_U, binding_layers):
    """
    Part 2: Overlap analysis between Top 1% cproj, Top 1% Δact, Top 1% contribution.
    Also cross-pair overlap.
    """
    cfg = MODEL_CONFIGS[model_name]
    layers = get_layers(model)
    
    # Per-layer overlap across all pairs
    layer_overlap = {}
    # Cross-pair channel sets
    layer_cproj_top1 = defaultdict(lambda: defaultdict(set))  # li -> pair -> set
    layer_dact_top1 = defaultdict(lambda: defaultdict(set))
    layer_contrib_top1 = defaultdict(lambda: defaultdict(set))
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue
        
        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            continue
        direction_normed = direction / dir_norm
        
        clean_prompt = f"The {obj}"
        clean_caps, _ = capture_mlp_internals(
            model, tokenizer, device, clean_prompt, binding_layers)
        corrupt_caps, _ = capture_mlp_internals(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)
        
        for li in binding_layers:
            W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
            if W_down is None:
                continue
            
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue
            
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
            
            min_d = min(d_ff, W_down.shape[1], cg.shape[0])
            Wd = W_down[:, :min_d]
            
            gsc = silu_np(cg[:min_d]); gsr = silu_np(crg[:min_d])
            uc = cu[:min_d]; ur = cru[:min_d]
            
            dact = gsc * uc - gsr * ur
            channel_proj = Wd.T @ direction_normed
            contribution = channel_proj * dact
            
            n_top1 = max(1, min_d // 100)
            
            # Top 1% by |cproj|
            top_cproj = set(np.argsort(np.abs(channel_proj))[-n_top1:])
            # Top 1% by |Δact|
            top_dact = set(np.argsort(np.abs(dact))[-n_top1:])
            # Top 1% by |contribution|
            top_contrib = set(np.argsort(np.abs(contribution))[-n_top1:])
            
            pair_key = f"{obj}-{target}"
            layer_cproj_top1[li][pair_key] = top_cproj
            layer_dact_top1[li][pair_key] = top_dact
            layer_contrib_top1[li][pair_key] = top_contrib
        
        del clean_caps, corrupt_caps
        gc.collect(); torch.cuda.empty_cache()
        
        if (pidx + 1) % 10 == 0:
            log(f"  [Overlap] {pidx+1}/{len(TEST_PAIRS)} pairs processed")
    
    # Compute overlaps
    overlap_results = {}
    for li in binding_layers:
        pairs = list(layer_cproj_top1[li].keys())
        if len(pairs) < 2:
            continue
        
        # Within-pair overlap: Jaccard between Top 1% cproj ∩ Top 1% Δact etc.
        cproj_dact_jaccard = []
        cproj_contrib_jaccard = []
        dact_contrib_jaccard = []
        
        for pk in pairs:
            s1 = layer_cproj_top1[li][pk]
            s2 = layer_dact_top1[li][pk]
            s3 = layer_contrib_top1[li][pk]
            
            if len(s1) > 0 and len(s2) > 0:
                j12 = len(s1 & s2) / len(s1 | s2)
                cproj_dact_jaccard.append(j12)
            if len(s1) > 0 and len(s3) > 0:
                j13 = len(s1 & s3) / len(s1 | s3)
                cproj_contrib_jaccard.append(j13)
            if len(s2) > 0 and len(s3) > 0:
                j23 = len(s2 & s3) / len(s2 | s3)
                dact_contrib_jaccard.append(j23)
        
        # Cross-pair overlap: Jaccard between same-type Top 1% across pairs
        cross_cproj_jaccard = []
        cross_dact_jaccard = []
        cross_contrib_jaccard = []
        
        for i in range(len(pairs)):
            for j in range(i+1, min(i+6, len(pairs))):  # Sample pairs to limit computation
                pk1, pk2 = pairs[i], pairs[j]
                
                s1c = layer_cproj_top1[li][pk1]
                s2c = layer_cproj_top1[li][pk2]
                if len(s1c) > 0 and len(s2c) > 0:
                    cross_cproj_jaccard.append(len(s1c & s2c) / len(s1c | s2c))
                
                s1d = layer_dact_top1[li][pk1]
                s2d = layer_dact_top1[li][pk2]
                if len(s1d) > 0 and len(s2d) > 0:
                    cross_dact_jaccard.append(len(s1d & s2d) / len(s1d | s2d))
                
                s1v = layer_contrib_top1[li][pk1]
                s2v = layer_contrib_top1[li][pk2]
                if len(s1v) > 0 and len(s2v) > 0:
                    cross_contrib_jaccard.append(len(s1v & s2v) / len(s1v | s2v))
        
        # Random baseline: expected Jaccard for sets of size n_top1 from d_ff elements
        # Expected Jaccard ≈ n_top1 / (2*d_ff - n_top1) for random sets
        d_ff = MODEL_CONFIGS[model_name]["d_ff"]
        n_top1_est = max(1, d_ff // 100)
        random_jaccard = n_top1_est / (2 * d_ff - n_top1_est)
        
        overlap_results[li] = {
            "within_cproj_dact": float(np.mean(cproj_dact_jaccard)) if cproj_dact_jaccard else 0,
            "within_cproj_contrib": float(np.mean(cproj_contrib_jaccard)) if cproj_contrib_jaccard else 0,
            "within_dact_contrib": float(np.mean(dact_contrib_jaccard)) if dact_contrib_jaccard else 0,
            "cross_cproj": float(np.mean(cross_cproj_jaccard)) if cross_cproj_jaccard else 0,
            "cross_dact": float(np.mean(cross_dact_jaccard)) if cross_dact_jaccard else 0,
            "cross_contrib": float(np.mean(cross_contrib_jaccard)) if cross_contrib_jaccard else 0,
            "random_baseline": float(random_jaccard),
            "cross_vs_random_cproj": float(np.mean(cross_cproj_jaccard) / random_jaccard) if cross_cproj_jaccard else 0,
            "cross_vs_random_dact": float(np.mean(cross_dact_jaccard) / random_jaccard) if cross_dact_jaccard else 0,
            "cross_vs_random_contrib": float(np.mean(cross_contrib_jaccard) / random_jaccard) if cross_contrib_jaccard else 0,
        }
    
    return overlap_results


def run_boost_suppress_analysis(model, tokenizer, device, model_name, W_U, binding_layers):
    """
    Part 3: Boost vs Suppress decomposition.
    Do Top 1% channels mainly boost compatible or suppress incompatible?
    """
    cfg = MODEL_CONFIGS[model_name]
    layers = get_layers(model)
    
    band_metrics = defaultdict(lambda: {
        "target_boost": 0.0, "competitor_suppress": 0.0,
        "target_gross": 0.0, "competitor_gross": 0.0,
        "n_pairs": 0,
    })
    
    BANDS = [
        ("Top 1%", 0.00, 0.01),
        ("1-10%", 0.01, 0.10),
        ("10-30%", 0.10, 0.30),
        ("30-60%", 0.30, 0.60),
        ("Bottom 40%", 0.60, 1.00),
    ]
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue
        
        dir_t = W_U[tid_t].copy()
        dir_c = W_U[tid_c].copy()
        dir_diff = dir_t - dir_c
        dir_norm = np.linalg.norm(dir_diff)
        if dir_norm < 1e-10:
            continue
        dir_diff_n = dir_diff / dir_norm
        
        # Also compute individual directions (normalized)
        norm_t = np.linalg.norm(dir_t)
        norm_c = np.linalg.norm(dir_c)
        dir_t_n = dir_t / max(norm_t, 1e-10)
        dir_c_n = dir_c / max(norm_c, 1e-10)
        
        clean_prompt = f"The {obj}"
        clean_caps, _ = capture_mlp_internals(
            model, tokenizer, device, clean_prompt, binding_layers)
        corrupt_caps, _ = capture_mlp_internals(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)
        
        for li in binding_layers:
            W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
            if W_down is None:
                continue
            
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue
            
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
            
            min_d = min(d_ff, W_down.shape[1], cg.shape[0])
            Wd = W_down[:, :min_d]
            
            gsc = silu_np(cg[:min_d]); gsr = silu_np(crg[:min_d])
            uc = cu[:min_d]; ur = cru[:min_d]
            
            dact = gsc * uc - gsr * ur
            channel_proj_diff = Wd.T @ dir_diff_n  # projection on diff direction
            channel_proj_t = Wd.T @ dir_t_n  # projection on target direction
            channel_proj_c = Wd.T @ dir_c_n  # projection on competitor direction
            
            abs_cproj = np.abs(channel_proj_diff)
            sorted_indices = np.argsort(abs_cproj)[::-1]
            rank = np.zeros(min_d, dtype=int)
            rank[sorted_indices] = np.arange(min_d)
            frac_rank = rank / max(min_d - 1, 1)
            
            for band_name, lo, hi in BANDS:
                mask = (frac_rank >= lo) & (frac_rank < hi)
                if not np.any(mask):
                    continue
                
                bd = dact[mask]
                bp_t = channel_proj_t[mask]
                bp_c = channel_proj_c[mask]
                
                # Target boost: contribution to target logit direction
                target_contrib = np.sum(bp_t * bd)
                # Competitor contribution: contribution to competitor logit direction  
                competitor_contrib = np.sum(bp_c * bd)
                
                # Net binding = target boost - competitor suppression
                # If target_contrib > 0 → boosts target
                # If competitor_contrib < 0 → suppresses competitor
                target_boost = max(0, float(target_contrib))
                competitor_suppress = max(0, float(-competitor_contrib))
                
                band_metrics[band_name]["target_boost"] += target_boost
                band_metrics[band_name]["competitor_suppress"] += competitor_suppress
                band_metrics[band_name]["target_gross"] += float(np.sum(np.abs(bp_t * bd)))
                band_metrics[band_name]["competitor_gross"] += float(np.sum(np.abs(bp_c * bd)))
                band_metrics[band_name]["n_pairs"] += 1
        
        del clean_caps, corrupt_caps
        gc.collect(); torch.cuda.empty_cache()
        
        if (pidx + 1) % 10 == 0:
            log(f"  [Boost/Suppress] {pidx+1}/{len(TEST_PAIRS)} pairs processed")
    
    # Normalize
    results = {}
    for band_name in ["Top 1%", "1-10%", "10-30%", "30-60%", "Bottom 40%"]:
        m = band_metrics[band_name]
        n = max(m["n_pairs"], 1)
        total_signal = m["target_boost"] + m["competitor_suppress"]
        results[band_name] = {
            "target_boost": float(m["target_boost"] / n),
            "competitor_suppress": float(m["competitor_suppress"] / n),
            "boost_frac": float(m["target_boost"] / max(total_signal, 1e-10)),
            "suppress_frac": float(m["competitor_suppress"] / max(total_signal, 1e-10)),
            "target_gross": float(m["target_gross"] / n),
            "competitor_gross": float(m["competitor_gross"] / n),
            "n_pairs": m["n_pairs"],
        }
    
    return results


def run_experiment(model_name):
    log(f"Phase 351: Top 1% Causal Ablation + Overlap + Boost/Suppress ({model_name})")
    log("=" * 70)
    t0 = time.time()
    cfg = MODEL_CONFIGS[model_name]
    binding_layers = cfg["binding_layers"]

    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    d_model = W_U.shape[1]
    layers = get_layers(model)
    
    # Pre-load all MLP weights
    mlp_weights = {}
    for li in binding_layers:
        W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_gate": W_gate, "W_up": W_up, "W_down": W_down, "d_ff": d_ff}
    log(f"  MLP weights loaded for {len(binding_layers)} layers")

    # ============================================================
    # PART 1: Top 1% Channel Causal Ablation
    # ============================================================
    log(f"\n{'='*70}")
    log(f"PART 1: Causal Ablation of Top 1% Channels")
    log(f"{'='*70}")
    
    # First, identify Top 1% channels per layer (using apple-red pair as reference)
    # Then test ablation across all pairs
    
    # Identify Top 1% |cproj| channels per layer (aggregate across pairs)
    top1_cproj_channels = {}  # li -> set of channel indices
    top1_dact_channels = {}
    random_channels = {}
    
    # Use first 10 pairs to identify channels
    ref_pairs = TEST_PAIRS[:10]
    channel_counts_cproj = defaultdict(lambda: defaultdict(int))
    channel_counts_dact = defaultdict(lambda: defaultdict(int))
    
    log(f"  Identifying Top 1% channels from {len(ref_pairs)} reference pairs...")
    for pidx, (obj, target, competitor) in enumerate(ref_pairs):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue
        
        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            continue
        direction_normed = direction / dir_norm
        
        clean_caps, _ = capture_mlp_internals(
            model, tokenizer, device, f"The {obj}", binding_layers)
        corrupt_caps, _ = capture_mlp_internals(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)
        
        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None:
                continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue
            
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
            
            min_d = min(d_ff, W_down.shape[1], cg.shape[0])
            Wd = W_down[:, :min_d]
            gsc = silu_np(cg[:min_d]); gsr = silu_np(crg[:min_d])
            uc = cu[:min_d]; ur = cru[:min_d]
            dact = gsc * uc - gsr * ur
            
            channel_proj = Wd.T @ direction_normed
            n_top1 = max(1, min_d // 100)
            
            # Top 1% by |cproj|
            top_cproj_idx = np.argsort(np.abs(channel_proj))[-n_top1:]
            for ch in top_cproj_idx:
                channel_counts_cproj[li][int(ch)] += 1
            
            # Top 1% by |Δact|
            top_dact_idx = np.argsort(np.abs(dact))[-n_top1:]
            for ch in top_dact_idx:
                channel_counts_dact[li][int(ch)] += 1
        
        del clean_caps, corrupt_caps
        gc.collect(); torch.cuda.empty_cache()
    
    # Select channels that appear in >= 50% of reference pairs
    n_ref = len(ref_pairs)
    for li in binding_layers:
        d_ff = mlp_weights[li]["d_ff"]
        n_top1 = max(1, d_ff // 100)
        
        # Top 1% cproj: channels appearing in >= 30% of pairs
        top1_cproj_channels[li] = set(
            ch for ch, cnt in channel_counts_cproj[li].items()
            if cnt >= n_ref * 0.3
        )
        if len(top1_cproj_channels[li]) == 0:
            # Fallback: take top channels by total count
            sorted_ch = sorted(channel_counts_cproj[li].items(), key=lambda x: -x[1])
            top1_cproj_channels[li] = set(ch for ch, _ in sorted_ch[:n_top1])
        
        # Top 1% Δact: channels appearing in >= 30% of pairs
        top1_dact_channels[li] = set(
            ch for ch, cnt in channel_counts_dact[li].items()
            if cnt >= n_ref * 0.3
        )
        if len(top1_dact_channels[li]) == 0:
            sorted_ch = sorted(channel_counts_dact[li].items(), key=lambda x: -x[1])
            top1_dact_channels[li] = set(ch for ch, _ in sorted_ch[:n_top1])
        
        # Random control: same number of channels, randomly selected
        np.random.seed(42)
        all_channels = set(range(d_ff))
        n_random = len(top1_cproj_channels[li])
        random_channels[li] = set(np.random.choice(list(all_channels), size=min(n_random, len(all_channels)), replace=False))
        
        log(f"  Layer {li}: Top1_cproj={len(top1_cproj_channels[li])} ch, "
            f"Top1_dact={len(top1_dact_channels[li])} ch, Random={len(random_channels[li])} ch")
    
    # Run ablation test on all 30 pairs
    log(f"\n  Running ablation on {len(TEST_PAIRS)} pairs...")
    
    ablation_results = {
        "top1_cproj": [], "top1_dact": [], "random": [], "baseline": []
    }
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        # Baseline: no ablation
        result_base = run_ablation_experiment(
            model, tokenizer, device, model_name, W_U,
            obj, target, competitor, binding_layers, "none", {})
        if result_base is None:
            continue
        
        # Top 1% cproj ablation
        result_cproj = run_ablation_experiment(
            model, tokenizer, device, model_name, W_U,
            obj, target, competitor, binding_layers, "top1_cproj", 
            {li: top1_cproj_channels[li] for li in binding_layers})
        
        # Top 1% Δact ablation
        result_dact = run_ablation_experiment(
            model, tokenizer, device, model_name, W_U,
            obj, target, competitor, binding_layers, "top1_dact",
            {li: top1_dact_channels[li] for li in binding_layers})
        
        # Random ablation
        result_random = run_ablation_experiment(
            model, tokenizer, device, model_name, W_U,
            obj, target, competitor, binding_layers, "random",
            {li: random_channels[li] for li in binding_layers})
        
        ablation_results["baseline"].append(result_base)
        ablation_results["top1_cproj"].append(result_cproj)
        ablation_results["top1_dact"].append(result_dact)
        ablation_results["random"].append(result_random)
        
        if (pidx + 1) % 10 == 0:
            log(f"  [{pidx+1}/{len(TEST_PAIRS)}] elapsed={time.time()-t0:.0f}s")
    
    # Summarize ablation results
    log(f"\n  --- Ablation Summary (True Causal) ---")
    log(f"  {'Type':<15} {'BaseBindEff':>12} {'AblatedBindEff':>15} {'FracLost(logit)':>16} {'FracLost(attn)':>16}")
    log(f"  {'-'*75}")
    
    ablation_summary = {}
    for atype in ["baseline", "top1_cproj", "top1_dact", "random"]:
        results_list = ablation_results[atype]
        if not results_list:
            continue
        
        mean_base_eff = np.mean([r["binding_effect_base"] for r in results_list])
        mean_ablated_eff = np.mean([r["binding_effect_ablated"] for r in results_list])
        mean_frac_logit = np.mean([r["frac_binding_effect_lost"] for r in results_list])
        mean_frac_attn = np.mean([r["frac_attn_lost"] for r in results_list])
        
        ablation_summary[atype] = {
            "mean_base_binding_effect": float(mean_base_eff),
            "mean_ablated_binding_effect": float(mean_ablated_eff),
            "mean_frac_logit_lost": float(mean_frac_logit),
            "mean_frac_attn_lost": float(mean_frac_attn),
            "n_pairs": len(results_list),
        }
        
        log(f"  {atype:<15} {mean_base_eff:>+12.4f} {mean_ablated_eff:>+15.4f} "
            f"{mean_frac_logit:>16.4f} {mean_frac_attn:>16.4f}")
    
    # ============================================================
    # PART 2: Overlap Structure
    # ============================================================
    log(f"\n{'='*70}")
    log(f"PART 2: Top 1% Overlap Structure")
    log(f"{'='*70}")
    
    overlap_results = run_overlap_analysis(model, tokenizer, device, model_name, W_U, binding_layers)
    
    for li, ov in overlap_results.items():
        log(f"\n  Layer {li}:")
        log(f"    Within-pair overlap:")
        log(f"      cproj ∩ dact:   Jaccard={ov['within_cproj_dact']:.4f}")
        log(f"      cproj ∩ contrib: Jaccard={ov['within_cproj_contrib']:.4f}")
        log(f"      dact ∩ contrib:  Jaccard={ov['within_dact_contrib']:.4f}")
        log(f"    Cross-pair overlap:")
        log(f"      cproj:  Jaccard={ov['cross_cproj']:.4f} ({ov['cross_vs_random_cproj']:.1f}x random)")
        log(f"      dact:   Jaccard={ov['cross_dact']:.4f} ({ov['cross_vs_random_dact']:.1f}x random)")
        log(f"      contrib: Jaccard={ov['cross_contrib']:.4f} ({ov['cross_vs_random_contrib']:.1f}x random)")
        log(f"    Random baseline: {ov['random_baseline']:.6f}")
    
    # ============================================================
    # PART 3: Boost vs Suppress
    # ============================================================
    log(f"\n{'='*70}")
    log(f"PART 3: Boost vs Suppress Decomposition")
    log(f"{'='*70}")
    
    boost_suppress = run_boost_suppress_analysis(model, tokenizer, device, model_name, W_U, binding_layers)
    
    log(f"\n  {'Band':<15} {'TargetBoost':>12} {'CompetSuppress':>14} {'Boost%':>8} {'Suppress%':>10}")
    log(f"  {'-'*62}")
    for band_name in ["Top 1%", "1-10%", "10-30%", "30-60%", "Bottom 40%"]:
        bs = boost_suppress[band_name]
        log(f"  {band_name:<15} {bs['target_boost']:>+12.4f} {bs['competitor_suppress']:>+14.4f} "
            f"{bs['boost_frac']:>8.3f} {bs['suppress_frac']:>10.3f}")
    
    # ============================================================
    # Save results
    # ============================================================
    all_results = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": len(ablation_results["baseline"]),
        "ablation_summary": ablation_summary,
        "overlap": {str(k): v for k, v in overlap_results.items()},
        "boost_suppress": boost_suppress,
    }
    
    os.makedirs("results/phase351_top1_causal_ablation", exist_ok=True)
    out_path = f"results/phase351_top1_causal_ablation/{model_name}_phase351.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    log(f"\n  Saved to {out_path}")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    log(f"Phase 351 complete for {model_name} in {time.time()-t0:.0f}s")
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
