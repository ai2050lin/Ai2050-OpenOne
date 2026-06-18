"""
Phase 516: Validated Trajectory Subspace Causal Patching
========================================================

Phase 515核心问题：Hook干预全部delta=0，导致U_trajectory因果性无法验证。
Phase 516目标：
1. 修复中间层干预管线（诊断hook/pre_hook/inputs_embeds）
2. 验证d_traj在多层的因果效应（用inputs_embeds作为backup）
3. 证明d_traj干预可以改善semantic hit
4. RMSNorm分解解释hs[-1]负效应

干预策略：
- inputs_embeds: 修改embedding层（已知有效，作为baseline）
- pre_hook: 修改中间层输入（需要验证是否有效）
- 直接用model.model.norm+lm_head: 修改最终hidden state（验证用）

用法:
  python tests/glm5/phase516_validated_causal_patching.py qwen3 --test-objects 10
  python tests/glm5/phase516_validated_causal_patching.py glm4 --test-objects 5
  python tests/glm5/phase516_validated_causal_patching.py deepseek7b --test-objects 5
"""
import sys, os, gc, time, argparse, json, math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import numpy as np
import torch
from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS

# ============== Configuration ==============
FRUIT_OBJECTS = ["apple", "banana", "orange", "grape", "strawberry",
                 "mango", "pear", "cherry", "watermelon", "pineapple",
                 "peach", "lemon", "lime", "coconut", "kiwi"]
EMOTION_OBJECTS = ["happiness", "sadness", "anger", "fear", "surprise",
                   "disgust", "pride", "shame", "love", "jealousy"]
COLOR_OBJECTS = ["red", "blue", "green", "yellow", "purple",
                 "orange", "pink", "brown", "gray", "white"]
ACTION_OBJECTS = ["running", "jumping", "eating", "writing", "speaking",
                  "swimming", "dancing", "sleeping", "thinking", "singing"]

FRUIT_TEMPLATES = [
    "belongs to the category of",
    "is classified as a type of",
    "is a kind of",
]
EMOTION_TEMPLATES = [
    "belongs to the category of",
    "is classified as a type of",
    "is a kind of",
]
COLOR_TEMPLATES = [
    "belongs to the category of",
    "is classified as a type of",
    "is a kind of",
]
ACTION_NATURAL_TEMPLATES = [
    "The person is",
    "This activity is called",
    "The action is",
]

CATEGORY_WORDS = {
    "fruit": "fruit",
    "emotion": "emotion",
    "color": "color",
    "action_natural": "action",
}

ACTION_VERBS = ["run", "jump", "eat", "write", "speak", "swim", "dance", "sleep", "think", "sing"]


def get_test_config(category, test_objects_n):
    configs = {
        "fruit": (FRUIT_OBJECTS[:test_objects_n], FRUIT_TEMPLATES, "fruit"),
        "emotion": (EMOTION_OBJECTS[:test_objects_n], EMOTION_TEMPLATES, "emotion"),
        "color": (COLOR_OBJECTS[:test_objects_n], COLOR_TEMPLATES, "color"),
        "action_natural": (ACTION_OBJECTS[:test_objects_n], ACTION_NATURAL_TEMPLATES, "action"),
    }
    return configs[category]


def load_model_bf16_auto(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    
    print(f"[P516] Loading {model_name} (bf16 + device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    
    input_device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[P516] {model_name} loaded: class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, input_device


def safe_encode(tokenizer, text, device, max_length=64):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device)}


def compute_cat_logit(logits, tokenizer, category_word):
    cat_ids = tokenizer.encode(category_word, add_special_tokens=False)
    return float(logits[cat_ids[0]]) if cat_ids else None


def classify_trajectory(text, category_word, category_type="fruit"):
    text_lower = text.lower().strip()
    cat_lower = category_word.lower()
    cat_present = cat_lower in text_lower
    
    if category_type == "action":
        for verb in ACTION_VERBS:
            if verb in text_lower:
                return "semantic_answer"
    
    phrases = [f"a {cat_lower}", f"an {cat_lower}", f"the {cat_lower}",
               f"type of {cat_lower}", f"kind of {cat_lower}",
               f"is a {cat_lower}", f"is an {cat_lower}", f"as a {cat_lower}"]
    has_phrase = any(p in text_lower for p in phrases)
    
    if cat_present and has_phrase:
        return "semantic_answer"
    elif cat_present:
        return "lexical"
    return "miss"


# ============== Intervention Methods ==============

def intervene_inputs_embeds(model, tokenizer, input_device, prompt, direction, alpha):
    """Intervention at embedding layer via inputs_embeds modification"""
    enc = safe_encode(tokenizer, prompt, input_device)
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(enc["input_ids"]).detach().clone()
    d = torch.tensor(direction, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    inputs_embeds[0, -1, :] += d * alpha
    with torch.no_grad():
        out = model(inputs_embeds=inputs_embeds, attention_mask=enc["attention_mask"])
    return out.logits[:, -1, :].float().cpu().numpy().flatten()


def intervene_pre_hook(model, tokenizer, input_device, prompt, direction, alpha, target_layer):
    """Intervention using register_forward_pre_hook on target_layer"""
    enc = safe_encode(tokenizer, prompt, input_device)
    layers = get_layers(model)
    layer_device = next(layers[target_layer].parameters()).device
    layer_dtype = next(layers[target_layer].parameters()).dtype
    d_tensor = torch.tensor(direction, dtype=layer_dtype, device=layer_device) * alpha
    
    fired = [False]
    def pre_hook_fn(module, args):
        if isinstance(args, tuple) and len(args) > 0:
            hs = args[0]
            modified = hs.clone()
            modified[:, -1, :] += d_tensor
            fired[0] = True
            return (modified,) + args[1:]
        return args
    
    hook = layers[target_layer].register_forward_pre_hook(pre_hook_fn)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    hook.remove()
    return out.logits[:, -1, :].float().cpu().numpy().flatten(), fired[0]


def intervene_hook(model, tokenizer, input_device, prompt, direction, alpha, target_layer):
    """Intervention using register_forward_hook on target_layer"""
    enc = safe_encode(tokenizer, prompt, input_device)
    layers = get_layers(model)
    layer_device = next(layers[target_layer].parameters()).device
    layer_dtype = next(layers[target_layer].parameters()).dtype
    d_tensor = torch.tensor(direction, dtype=layer_dtype, device=layer_device) * alpha
    
    fired = [False]
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            modified = output[0].clone()
            modified[:, -1, :] += d_tensor
            fired[0] = True
            return (modified,) + output[1:]
        return output
    
    hook = layers[target_layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    hook.remove()
    return out.logits[:, -1, :].float().cpu().numpy().flatten(), fired[0]


def get_logits_from_hidden(model, hidden_states):
    """Get logits from hidden states, handling tied embeddings"""
    import torch.nn.functional as F
    h_normed = model.model.norm(hidden_states)
    # For tied embeddings (lm_head is Embedding, not Linear), use F.linear
    if isinstance(model.lm_head, torch.nn.Embedding):
        return F.linear(h_normed, model.lm_head.weight)
    else:
        return model.lm_head(h_normed)


# ============== Safe weight loading (handles meta device from device_map=auto) ==============

_WEIGHT_CACHE = {}

def get_norm_weight_safe(model, model_name):
    """Get RMSNorm weight as numpy, loading from safetensors if on meta device"""
    cache_key = f"{model_name}_norm"
    if cache_key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[cache_key]
    
    w = model.model.norm.weight
    if not w.is_meta:
        g = w.detach().float().cpu().numpy()
    else:
        # Load from safetensors
        import glob, os
        from safetensors import safe_open
        cfg = MODEL_CONFIGS[model_name]
        sf_files = glob.glob(os.path.join(cfg["path"], '*.safetensors'))
        g = None
        possible_keys = ['model.norm.weight', 'model.model.norm.weight']
        for sf_file in sf_files:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                keys = list(sf.keys())
                for pk in possible_keys:
                    if pk in keys:
                        g = sf.get_tensor(pk).float().numpy()
                        print(f"  [norm_weight] Loaded from {os.path.basename(sf_file)}, key={pk}, shape={g.shape}")
                        break
                if g is None:
                    # Fuzzy match: ends with norm.weight, not a layer norm
                    for k in keys:
                        if k.endswith('norm.weight') and 'layers' not in k and 'attention' not in k and 'post' not in k and 'input' not in k:
                            g = sf.get_tensor(k).float().numpy()
                            print(f"  [norm_weight] Loaded from {os.path.basename(sf_file)}, key={k}, shape={g.shape}")
                            break
                if g is not None:
                    break
        if g is None:
            raise ValueError(f"Cannot load norm weight for {model_name}")
    
    _WEIGHT_CACHE[cache_key] = g
    return g


def get_W_U_cached(model, model_name):
    """Get W_U with caching"""
    cache_key = f"{model_name}_WU"
    if cache_key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[cache_key]
    W_U = get_W_U(model, model_name)
    _WEIGHT_CACHE[cache_key] = W_U
    return W_U


def intervene_final_hs(model, tokenizer, input_device, prompt, direction, alpha, model_name=None):
    """Intervention by modifying final hidden state — numpy-based (handles meta device)"""
    enc = safe_encode(tokenizer, prompt, input_device)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
    # Get final hidden state at last position as numpy
    h_final = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()  # [d_model]
    d = np.array(direction, dtype=np.float32)
    h_mod = h_final + d * alpha
    
    # Compute logits in numpy: RMSNorm + W_U
    g = get_norm_weight_safe(model, model_name)
    W_U = get_W_U_cached(model, model_name)
    eps = getattr(model.config, 'rms_norm_eps', 1e-6)
    rms = np.sqrt(np.mean(h_mod**2) + eps)
    h_normed = g * h_mod / rms
    logits = h_normed @ W_U.T  # [vocab_size]
    return logits


# ============== Exp1: Intervention Pipeline Diagnostic ==============

def exp1_pipeline_diagnostic(model, tokenizer, input_device, model_name):
    print("\n" + "="*60)
    print("Exp1: Intervention Pipeline Diagnostic")
    print("="*60)
    
    info = get_model_info(model, model_name)
    prompt = "An apple belongs to the category of"
    enc = safe_encode(tokenizer, prompt, input_device)
    
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
    baseline_logits = out.logits[:, -1, :].float().cpu().numpy().flatten()
    
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    fruit_logit_b = float(baseline_logits[fruit_ids[0]])
    
    d_model = info.d_model
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    
    # Construct test directions
    np.random.seed(42)
    random_dir = np.random.randn(d_model).astype(np.float32)
    random_dir = random_dir / np.linalg.norm(random_dir) * 5.0
    
    top_ids = np.argsort(baseline_logits)[-5:][::-1]
    comp_id = top_ids[0] if top_ids[0] != fruit_ids[0] else top_ids[1]
    q_c = (W_U[fruit_ids[0]] - W_U[comp_id])
    q_c_scaled = q_c / (np.linalg.norm(q_c) + 1e-8) * 5.0
    
    test_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    test_alphas = [0.1, 1.0, 10.0]
    
    results = {}
    working_counts = {}
    
    method_tests = [
        ("inputs_embeds", lambda d, a, l: intervene_inputs_embeds(model, tokenizer, input_device, prompt, d, a)),
        ("pre_hook", lambda d, a, l: intervene_pre_hook(model, tokenizer, input_device, prompt, d, a, l)),
        ("hook", lambda d, a, l: intervene_hook(model, tokenizer, input_device, prompt, d, a, l)),
        ("final_hs", lambda d, a, l: intervene_final_hs(model, tokenizer, input_device, prompt, d, a, model_name)),
    ]
    
    for method_name, method_fn in method_tests:
        print(f"\n--- Method: {method_name} ---")
        working = 0
        method_res = []
        
        for layer_idx in test_layers:
            for alpha in test_alphas:
                for dir_name, direction in [("random", random_dir), ("q_c", q_c_scaled)]:
                    try:
                        result = method_fn(direction, alpha, layer_idx)
                        if isinstance(result, tuple):
                            mod_logits, hook_fired = result
                        else:
                            mod_logits = result
                            hook_fired = True
                        
                        fruit_logit_mod = float(mod_logits[fruit_ids[0]])
                        delta = fruit_logit_mod - fruit_logit_b
                        worked = abs(delta) > 0.01
                        
                        if worked:
                            working += 1
                        
                        method_res.append({
                            "layer": layer_idx, "alpha": alpha, "direction": dir_name,
                            "delta": round(delta, 4), "worked": worked,
                            "hook_fired": hook_fired if isinstance(result, tuple) else None,
                        })
                        
                        if worked or alpha == test_alphas[-1]:
                            fired_str = f" fired={hook_fired}" if isinstance(result, tuple) else ""
                            print(f"  L{layer_idx} α={alpha} {dir_name}: Δ={delta:+.4f}{' ✓' if worked else ' ✗'}{fired_str}")
                    
                    except Exception as e:
                        err = str(e)[:60]
                        print(f"  L{layer_idx} α={alpha} {dir_name}: ERR {err}")
                        method_res.append({"layer": layer_idx, "alpha": alpha, "direction": dir_name,
                                          "error": err})
        
        results[method_name] = method_res
        working_counts[method_name] = working
    
    best = max(working_counts, key=working_counts.get) if any(v > 0 for v in working_counts.values()) else "inputs_embeds"
    print(f"\n  Working counts: {working_counts}")
    print(f"  Best method: {best}")
    
    return results, best


# ============== Exp2: d_traj Multi-Layer Causal Validation ==============

def exp2_d_traj_causal(model, tokenizer, input_device, model_name, best_method,
                       categories, test_objects_n):
    print("\n" + "="*60)
    print("Exp2: d_traj Multi-Layer Causal Validation")
    print("="*60)
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    probe_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    
    all_results = {}
    
    for category in categories:
        print(f"\n--- Category: {category} ---")
        objects, templates, cat_word = get_test_config(category, test_objects_n)
        cat_ids = tokenizer.encode(cat_word, add_special_tokens=False)
        
        # Phase A: Collect success/failure hidden states
        success_hs = {l: [] for l in range(n_layers + 1)}
        fail_hs = {l: [] for l in range(n_layers + 1)}
        success_cat_logits = []
        fail_cat_logits = []
        
        for obj in objects:
            for tmpl in templates:
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" and tmpl[0] != "T" else f"A {obj} {tmpl}"
                if category == "action_natural":
                    prompt = f"{obj.replace('ing', '')}. {tmpl}"
                
                enc = safe_encode(tokenizer, prompt, input_device)
                gen_kwargs = dict(max_new_tokens=8, do_sample=False, repetition_penalty=1.2)
                with torch.no_grad():
                    gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                             **gen_kwargs)
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                continuation = gen_text[len(prompt):].strip()
                quality = classify_trajectory(continuation, cat_word, category)
                
                with torch.no_grad():
                    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                output_hidden_states=True)
                logits = out.logits[:, -1, :].float().cpu().numpy().flatten()
                cat_logit = float(logits[cat_ids[0]]) if cat_ids else None
                hs = [h[:, -1, :].detach().float().cpu().numpy().flatten() for h in out.hidden_states]
                
                is_success = quality in ["semantic_answer", "natural_phrase"]
                if is_success:
                    for l in range(n_layers + 1):
                        success_hs[l].append(hs[l])
                    success_cat_logits.append(cat_logit)
                else:
                    for l in range(n_layers + 1):
                        fail_hs[l].append(hs[l])
                    fail_cat_logits.append(cat_logit)
        
        n_suc = len(success_cat_logits)
        n_fail = len(fail_cat_logits)
        suc_cat_mean = np.mean(success_cat_logits) if success_cat_logits else None
        fail_cat_mean = np.mean(fail_cat_logits) if fail_cat_logits else None
        
        print(f"  Success: {n_suc}, Fail: {n_fail}")
        if suc_cat_mean and fail_cat_mean:
            print(f"  Suc cat_logit: {suc_cat_mean:.2f}, Fail cat_logit: {fail_cat_mean:.2f}")
        
        # Phase B: Compute d_traj and test intervention at embedding layer
        d_traj_results = {}
        
        if n_suc >= 2 and n_fail >= 2:
            # Compute d_traj at MULTIPLE layers (not just embedding)
            # Note: h[0] = embedding output (just token embeddings, no context)
            #       h[1] = after layer 0 (has some context)
            #       h[n_layers] = final hidden state (full context)
            # Use h[n_layers//2] and h[n_layers] as primary d_traj sources
            d_traj_layers = {}
            for l in [n_layers//4, n_layers//2, 3*n_layers//4, n_layers]:
                if l in success_hs and l in fail_hs and len(success_hs[l]) >= 2 and len(fail_hs[l]) >= 2:
                    suc_mean = np.mean(success_hs[l], axis=0)
                    fail_mean = np.mean(fail_hs[l], axis=0)
                    d_traj_layers[l] = suc_mean - fail_mean
            
            # Use the layer with highest d_traj norm as primary
            if d_traj_layers:
                best_layer = max(d_traj_layers, key=lambda l: np.linalg.norm(d_traj_layers[l]))
                d_traj = d_traj_layers[best_layer]
                d_traj_norm = np.linalg.norm(d_traj)
                d_traj_scaled = d_traj / (d_traj_norm + 1e-8) * 5.0
                
                print(f"  d_traj at layer {best_layer}: norm={d_traj_norm:.2f}")
                for l, d in sorted(d_traj_layers.items()):
                    print(f"    L{l}: norm={np.linalg.norm(d):.2f}")
            else:
                print(f"  No layers with sufficient data for d_traj")
                d_traj_scaled = None
                best_layer = None
                d_traj_norm = 0
            
            # Test d_traj intervention using pre_hook (known to work at all layers)
            test_prompts = []
            for obj in objects[:5]:
                tmpl = templates[0]
                p = f"An {obj} {tmpl}" if obj[0] in "aeiou" and tmpl[0] != "T" else f"A {obj} {tmpl}"
                if category == "action_natural":
                    p = f"{obj.replace('ing', '')}. {tmpl}"
                test_prompts.append(p)
            
            intervention_data = []
            if d_traj_scaled is not None:
                # pre_hook target: best_layer is hidden_state index (0..n_layers)
                # transformer layers are 0..n_layers-1, so clamp
                pre_hook_layer = min(best_layer, n_layers - 1)
                for alpha in [0.5, 1.0, 2.0]:
                    for action in ["add", "remove"]:
                        deltas = []
                        for tp in test_prompts:
                            try:
                                direction = d_traj_scaled if action == "add" else -d_traj_scaled
                                # Use pre_hook at pre_hook_layer (clamped to valid range)
                                mod_logits, _ = intervene_pre_hook(
                                    model, tokenizer, input_device, tp, direction, alpha, pre_hook_layer)
                                mod_cat = float(mod_logits[cat_ids[0]]) if cat_ids else None
                                
                                enc = safe_encode(tokenizer, tp, input_device)
                                with torch.no_grad():
                                    base_out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                                base_cat = float(base_out.logits[0, -1, cat_ids[0]]) if cat_ids else None
                                
                                deltas.append(mod_cat - base_cat if mod_cat is not None and base_cat is not None else None)
                            except Exception as e:
                                deltas.append(None)
                        
                        mean_delta = np.mean([d for d in deltas if d is not None]) if any(d is not None for d in deltas) else None
                        intervention_data.append({
                            "alpha": alpha, "action": action,
                            "intervention_layer": pre_hook_layer,
                            "d_traj_source_layer": best_layer,
                            "mean_cat_logit_delta": round(float(mean_delta), 4) if mean_delta is not None else None,
                            "n_valid": sum(1 for d in deltas if d is not None),
                        })
                        print(f"  {action} α={alpha} at L{pre_hook_layer}: mean Δcat={mean_delta:.4f}" if mean_delta is not None else f"  {action} α={alpha}: no valid data")
            
            # Also compute d_traj at final layer for final_hs intervention test
            d_traj_final = None
            d_traj_final_norm = 0
            if n_layers in success_hs and n_layers in fail_hs and len(success_hs[n_layers]) >= 2 and len(fail_hs[n_layers]) >= 2:
                suc_final = np.mean(success_hs[n_layers], axis=0)
                fail_final = np.mean(fail_hs[n_layers], axis=0)
                d_traj_final = suc_final - fail_final
                d_traj_final_norm = np.linalg.norm(d_traj_final)
            
            # Test d_traj_final intervention at final hidden state
            final_intervention = []
            if d_traj_final is not None:
                d_traj_final_scaled = d_traj_final / (d_traj_final_norm + 1e-8) * 5.0
                for alpha in [0.5, 1.0, 2.0]:
                    for action in ["add", "remove"]:
                        deltas = []
                        for tp in test_prompts:
                            try:
                                direction = d_traj_final_scaled if action == "add" else -d_traj_final_scaled
                                mod_logits = intervene_final_hs(
                                    model, tokenizer, input_device, tp, direction, alpha, model_name)
                                mod_cat = float(mod_logits[cat_ids[0]]) if cat_ids else None
                                
                                enc = safe_encode(tokenizer, tp, input_device)
                                with torch.no_grad():
                                    base_out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                                base_cat = float(base_out.logits[0, -1, cat_ids[0]]) if cat_ids else None
                                
                                deltas.append(mod_cat - base_cat if mod_cat is not None and base_cat is not None else None)
                            except Exception as e:
                                deltas.append(None)
                        
                        mean_delta = np.mean([d for d in deltas if d is not None]) if any(d is not None for d in deltas) else None
                        final_intervention.append({
                            "alpha": alpha, "action": action,
                            "mean_cat_logit_delta": round(float(mean_delta), 4) if mean_delta is not None else None,
                        })
            
            d_traj_results = {
                "d_traj_best_layer": best_layer,
                "d_traj_norm": round(float(d_traj_norm), 4),
                "d_traj_final_norm": round(float(d_traj_final_norm), 4),
                "d_traj_layer_norms": {l: round(float(np.linalg.norm(d)), 4) for l, d in d_traj_layers.items()},
                "pre_hook_intervention": intervention_data,
                "final_hs_intervention": final_intervention,
            }
        else:
            d_traj_results = {"error": f"insufficient data: suc={n_suc}, fail={n_fail}"}
        
        all_results[category] = {
            "n_success": n_suc, "n_failure": n_fail,
            "suc_cat_logit_mean": round(float(suc_cat_mean), 4) if suc_cat_mean else None,
            "fail_cat_logit_mean": round(float(fail_cat_mean), 4) if fail_cat_mean else None,
            "d_traj_analysis": d_traj_results,
        }
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return all_results


# ============== Exp3: d_traj Generation Test ==============

def exp3_d_traj_generation(model, tokenizer, input_device, model_name, categories, test_objects_n):
    print("\n" + "="*60)
    print("Exp3: d_traj Generation Test")
    print("="*60)
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    all_results = {}
    
    for category in categories:
        print(f"\n--- Category: {category} ---")
        objects, templates, cat_word = get_test_config(category, test_objects_n)
        cat_ids = tokenizer.encode(cat_word, add_special_tokens=False)
        
        # Compute d_traj at middle layer (not embedding, which has no context)
        success_hs_mid = []
        fail_hs_mid = []
        mid_layer = n_layers // 2
        
        for obj in objects:
            for tmpl in templates:
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" and tmpl[0] != "T" else f"A {obj} {tmpl}"
                if category == "action_natural":
                    prompt = f"{obj.replace('ing', '')}. {tmpl}"
                
                enc = safe_encode(tokenizer, prompt, input_device)
                gen_kwargs = dict(max_new_tokens=8, do_sample=False, repetition_penalty=1.2)
                with torch.no_grad():
                    gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                             **gen_kwargs)
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                continuation = gen_text[len(prompt):].strip()
                quality = classify_trajectory(continuation, cat_word, category)
                
                with torch.no_grad():
                    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                output_hidden_states=True)
                h_mid = out.hidden_states[mid_layer][:, -1, :].detach().float().cpu().numpy().flatten()
                
                if quality in ["semantic_answer", "natural_phrase"]:
                    success_hs_mid.append(h_mid)
                else:
                    fail_hs_mid.append(h_mid)
        
        if len(success_hs_mid) < 2 or len(fail_hs_mid) < 2:
            print(f"  Insufficient data: suc={len(success_hs_mid)}, fail={len(fail_hs_mid)}")
            all_results[category] = {"error": "insufficient data"}
            continue
        
        d_traj = np.mean(success_hs_mid, axis=0) - np.mean(fail_hs_mid, axis=0)
        d_norm = np.linalg.norm(d_traj)
        d_scaled = d_traj / (d_norm + 1e-8) * 5.0
        
        # Generation test with multiple alphas
        gen_results = []
        test_alphas = [0.0, 0.5, 1.0, 2.0]
        
        for obj in objects[:5]:
            for tmpl in templates[:1]:
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" and tmpl[0] != "T" else f"A {obj} {tmpl}"
                if category == "action_natural":
                    prompt = f"{obj.replace('ing', '')}. {tmpl}"
                
                for alpha in test_alphas:
                    try:
                        if alpha == 0.0:
                            enc = safe_encode(tokenizer, prompt, input_device)
                            gen_kwargs = dict(max_new_tokens=8, do_sample=False, repetition_penalty=1.2)
                            with torch.no_grad():
                                gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                                         **gen_kwargs)
                            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                        else:
                            enc = safe_encode(tokenizer, prompt, input_device)
                            embed_layer = model.get_input_embeddings()
                            inputs_embeds = embed_layer(enc["input_ids"]).detach().clone()
                            d = torch.tensor(d_scaled * alpha, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                            inputs_embeds[0, -1, :] += d
                            gen_kwargs = dict(max_new_tokens=8, do_sample=False, repetition_penalty=1.2)
                            with torch.no_grad():
                                gen_ids = model.generate(inputs_embeds=inputs_embeds,
                                                         attention_mask=enc["attention_mask"],
                                                         **gen_kwargs)
                            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                        
                        continuation = gen_text[len(prompt):].strip()
                        quality = classify_trajectory(continuation, cat_word, category)
                        gen_results.append({"object": obj, "alpha": alpha, "quality": quality,
                                           "continuation": continuation[:60]})
                    except Exception as e:
                        gen_results.append({"object": obj, "alpha": alpha, "error": str(e)[:60]})
        
        # Summarize
        quality_summary = {}
        for alpha in test_alphas:
            alpha_res = [r for r in gen_results if r.get("alpha") == alpha and "quality" in r]
            if alpha_res:
                qs = [r["quality"] for r in alpha_res]
                quality_summary[str(alpha)] = {
                    "n": len(qs),
                    "semantic": sum(1 for q in qs if q == "semantic_answer"),
                    "lexical": sum(1 for q in qs if q == "lexical"),
                    "miss": sum(1 for q in qs if q == "miss"),
                }
        
        print(f"  d_traj norm: {d_norm:.2f}")
        for alpha, qs in sorted(quality_summary.items()):
            print(f"    α={alpha}: sem={qs['semantic']}/{qs['n']}, lex={qs['lexical']}/{qs['n']}, miss={qs['miss']}/{qs['n']}")
        
        all_results[category] = {
            "d_traj_norm": round(float(d_norm), 4),
            "d_traj_layer": mid_layer,
            "n_success": len(success_hs_mid), "n_failure": len(fail_hs_mid),
            "quality_summary": quality_summary,
            "gen_results": gen_results,
        }
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return all_results


# ============== Exp4: RMSNorm Decomposition ==============

def exp4_rmsnorm_decomposition(model, tokenizer, input_device, model_name):
    print("\n" + "="*60)
    print("Exp4: RMSNorm Decomposition")
    print("="*60)
    
    info = get_model_info(model, model_name)
    d_model = info.d_model
    W_U = get_W_U(model, model_name)
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    prompt = "An apple belongs to the category of"
    enc = safe_encode(tokenizer, prompt, input_device)
    
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
    
    # Get final hidden state (before norm) — use [0, -1, :] to remove batch dim
    h_final = out.hidden_states[-1][0, -1, :].detach()  # [d_model] on model device, model dtype
    h_np = h_final.float().cpu().numpy()
    
    # Get RMSNorm weight
    g = get_norm_weight_safe(model, model_name)  # [d_model]
    
    # RMS
    rms_h = np.sqrt(np.mean(h_np**2) + 1e-8)
    
    # Target token
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    w_fruit = W_U[fruit_ids[0]]  # [d_model]
    
    # D_c = sum(g * h * w_fruit) / rms(h)
    D_c = np.sum(g * h_np * w_fruit) / rms_h
    
    # Actual logit
    actual_logit = float(out.logits[0, -1, fruit_ids[0]].float().cpu().item())
    
    print(f"  Baseline: rms(h)={rms_h:.4f}, D_c(analytical)={D_c:.4f}, actual_logit={actual_logit:.4f}")
    
    # Test with q_c direction
    top_ids = np.argsort(out.logits[0, -1, :].float().cpu().numpy())[-5:][::-1]
    comp_id = top_ids[0] if top_ids[0] != fruit_ids[0] else top_ids[1]
    q_c = (w_fruit - W_U[comp_id])
    q_c_scaled = q_c / (np.linalg.norm(q_c) + 1e-8) * 5.0
    
    results = []
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
        delta_h = q_c_scaled * alpha
        h_mod = h_np + delta_h
        
        # Analytical decomposition
        rms_mod = np.sqrt(np.mean(h_mod**2) + 1e-8)
        D_c_mod = np.sum(g * h_mod * w_fruit) / rms_mod
        
        # Numerator effect: <g * delta_h, w_fruit> / rms(h)
        numerator_effect = np.sum(g * delta_h * w_fruit) / rms_h
        
        # Denominator effect (from RMS change): approximately D_c * (1/rms_ratio - 1)
        rms_ratio = rms_mod / rms_h
        denominator_effect = D_c * (1.0 / rms_ratio - 1.0)
        
        delta_D_c = D_c_mod - D_c
        
        # Actual effect (numpy-based RMSNorm + W_U — avoids device/tied-embeddings issues)
        rms_eps = getattr(model.config, 'rms_norm_eps', 1e-6)
        rms_mod_np = np.sqrt(np.mean(h_mod**2) + rms_eps)
        h_mod_normed = g * h_mod / rms_mod_np
        actual_mod = float(np.dot(h_mod_normed, w_fruit))
        actual_delta = actual_mod - actual_logit
        
        results.append({
            "alpha": alpha,
            "rms_mod": round(float(rms_mod), 4),
            "rms_ratio": round(float(rms_ratio), 6),
            "D_c_mod": round(float(D_c_mod), 4),
            "delta_D_c": round(float(delta_D_c), 4),
            "numerator_effect": round(float(numerator_effect), 4),
            "denominator_effect": round(float(denominator_effect), 4),
            "actual_delta": round(float(actual_delta), 4) if actual_delta is not None else None,
        })
        
        print(f"  α={alpha}: num={numerator_effect:+.4f}, den={denominator_effect:+.4f}, "
              f"analytical_Δ={delta_D_c:+.4f}, actual_Δ={actual_delta:+.4f}")
    
    return {
        "baseline_rms": round(float(rms_h), 4),
        "baseline_D_c": round(float(D_c), 4),
        "baseline_logit": round(float(actual_logit), 4),
        "decomposition": results,
    }


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test-objects", type=int, default=5)
    parser.add_argument("--categories", nargs="+", default=["fruit", "color", "emotion"])
    args = parser.parse_args()
    
    t_start = time.time()
    
    model, tokenizer, input_device = load_model_bf16_auto(args.model)
    info = get_model_info(model, args.model)
    print(f"  n_layers={info.n_layers}, d_model={info.d_model}")
    
    results = {"model": args.model, "model_info": {
        "n_layers": info.n_layers, "d_model": info.d_model}}
    
    # Exp1
    exp1_res, best_method = exp1_pipeline_diagnostic(model, tokenizer, input_device, args.model)
    results["exp1_pipeline"] = exp1_res
    results["best_method"] = best_method
    
    # Exp2
    exp2_res = exp2_d_traj_causal(model, tokenizer, input_device, args.model, best_method,
                                   args.categories, args.test_objects)
    results["exp2_d_traj_causal"] = exp2_res
    
    # Exp3
    exp3_res = exp3_d_traj_generation(model, tokenizer, input_device, args.model,
                                       args.categories, args.test_objects)
    results["exp3_generation"] = exp3_res
    
    # Exp4 (with try/except to save partial results on failure)
    try:
        exp4_res = exp4_rmsnorm_decomposition(model, tokenizer, input_device, args.model)
        results["exp4_rmsnorm"] = exp4_res
    except Exception as e:
        print(f"\nExp4 failed: {e}")
        results["exp4_rmsnorm"] = {"error": str(e)}
    
    # Save
    os.makedirs("results/glm5_phase516_validated_causal", exist_ok=True)
    out_path = f"results/glm5_phase516_validated_causal/phase516_{args.model}_validated_causal.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")
    print(f"Best method: {best_method}")


if __name__ == "__main__":
    main()
