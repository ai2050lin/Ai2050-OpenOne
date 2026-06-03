"""
Phase 363: Per-layer C2R Damage Scan + Multi-position Patch + h_in Decomposition
================================================================================

Core question: Which layers CREATE binding? (incremental contribution)

Part 1: Per-layer C2R damage scan (PRIMARY)
  - At sampled layers (every 3-4 layers + core layers):
    C2R-attn: replace clean attn_out with corrupt → measure binding damage
    C2R-mlp:  replace clean mlp_out with corrupt → measure binding damage
  - Also R2C at core layers for Phase 361 comparison
  - This directly identifies "binding creation layers" vs "carrying layers"

Part 2: Multi-position h_in patch
  - At core layers, patch h_in at:
    a. Last token only (standard approach)
    b. Object token(s) only
    c. All tokens
  - Checks if binding is localized to last token or distributed

Part 3: h_in component decomposition
  - At core layers, decompose h_in clean-corrupt diff into:
    a. binding-parallel component (along W_U[target] - W_U[competitor] direction)
    b. orthogonal component (everything else)
  - Patch each separately to see which recovers binding

Unified notation:
  C2R: effect = -Δgap / |base_gap| (positive = binding damaged)
  R2C: effect = +Δgap / |base_gap| (positive = binding rescued)
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Model Configs =====
MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
        "sample_layers": sorted(set(list(range(0, 36, 3)) + [1, 2, 23, 27, 35])),
        "core_layers": [23, 27],
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
        "sample_layers": sorted(set(list(range(0, 40, 4)) + [1, 2, 36, 38, 39])),
        "core_layers": [36, 38],
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
        "sample_layers": sorted(set(list(range(0, 28, 3)) + [1, 2, 19, 21, 27])),
        "core_layers": [19, 21],
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


# ===== Model Loading =====

def load_model_bf16(model_name):
    """BF16 + device_map=auto + flash attention"""
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


def compute_effect(delta_gap, base_gap, direction):
    abs_base = abs(base_gap)
    if abs_base < 1e-10:
        return 0.0
    if direction == "c2r":
        return -delta_gap / abs_base
    else:
        return delta_gap / abs_base


def find_object_positions(tokenizer, prompt, obj_word):
    """Find token positions of the object word in the prompt."""
    input_ids = tokenizer.encode(prompt)
    positions = []
    for i, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid]).strip().lower()
        if obj_word.lower() in decoded and decoded != '':
            positions.append(i)
    # If no match found, try position 1 as fallback (after "The")
    if not positions:
        positions = [1] if len(input_ids) > 1 else [0]
    return positions


# ===== Activation Capture =====

def capture_attn_mlp_activations(model, tokenizer, device, prompt, target_layers):
    """Capture attn_out and mlp_out at target layers for last token."""
    layers = get_layers(model)
    captured = {}

    def make_fwd_hook_last(key):
        def hook(module, input, output):
            val = output[0] if isinstance(output, tuple) else output
            captured[key] = val[0, -1, :].detach().cpu().float().numpy()
        return hook

    hooks = []
    for li in target_layers:
        layer = layers[li]
        hooks.append(layer.self_attn.register_forward_hook(
            make_fwd_hook_last(f"attn_out_{li}")))
        hooks.append(layer.mlp.register_forward_hook(
            make_fwd_hook_last(f"mlp_out_{li}")))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()

    for h in hooks:
        h.remove()

    return captured, logits


def capture_h_in_all_positions(model, tokenizer, device, prompt, target_layers):
    """Capture h_in at ALL token positions (not just last token)."""
    layers = get_layers(model)
    captured = {}

    def make_pre_hook_all(key):
        def pre_hook(module, args):
            inp = args[0]
            captured[key] = inp[0].detach().cpu().float().numpy()  # [seq_len, d_model]
        return pre_hook

    hooks = []
    for li in target_layers:
        hooks.append(layers[li].register_forward_pre_hook(
            make_pre_hook_all(f"h_in_{li}")))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    seq_len = inp["input_ids"].shape[1]

    for h in hooks:
        h.remove()

    return captured, logits, seq_len


# ===== Patch Hooks =====

def _make_output_patch_hook_last(replacement):
    """Replace last token of module output."""
    def hook(module, input, output):
        val = output[0] if isinstance(output, tuple) else output
        modified = val.clone()
        rep_t = torch.tensor(replacement, dtype=modified.dtype, device=modified.device)
        modified[0, -1, :] = rep_t
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook


def _make_input_patch_hook_positions(replacement_dict):
    """
    Replace h_in at specific token positions.
    replacement_dict: {position_idx: numpy_array[d_model]}
    """
    def pre_hook(module, args):
        hidden_states = args[0]
        modified = hidden_states.clone()
        for pos, rep in replacement_dict.items():
            rep_t = torch.tensor(rep, dtype=modified.dtype, device=modified.device)
            modified[0, pos, :] = rep_t
        return (modified,) + args[1:]
    return pre_hook


def _make_input_patch_hook_all(replacement_full):
    """Replace h_in at ALL token positions with replacement_full [seq_len, d_model]."""
    def pre_hook(module, args):
        hidden_states = args[0]
        modified = hidden_states.clone()
        rep_t = torch.tensor(replacement_full, dtype=modified.dtype, device=modified.device)
        modified[0, :, :] = rep_t
        return (modified,) + args[1:]
    return pre_hook


def _make_input_patch_hook_component(replacement_component, original_h_in):
    """
    Replace only a specific component of h_in at last token.
    replacement_component: numpy array [d_model] — the component to replace
    original_h_in: numpy array [d_model] — the original (corrupt) h_in at last token
    
    The patched h_in = original_h_in + replacement_component
    (replacement_component is the clean-corrupt diff projected onto some direction)
    """
    def pre_hook(module, args):
        hidden_states = args[0]
        modified = hidden_states.clone()
        # At last token: original + component
        orig_t = torch.tensor(original_h_in, dtype=modified.dtype, device=modified.device)
        comp_t = torch.tensor(replacement_component, dtype=modified.dtype, device=modified.device)
        modified[0, -1, :] = orig_t + comp_t
        return (modified,) + args[1:]
    return pre_hook


# ===== Part 1: Per-layer C2R Damage Scan =====

def run_part1(model, tokenizer, device, model_name):
    """Per-layer C2R damage scan: measure incremental contribution of each layer."""
    cfg = MODEL_CONFIGS[model_name]
    sample_layers = cfg["sample_layers"]
    core_layers = cfg["core_layers"]
    all_test_layers = sorted(set(sample_layers + core_layers))
    
    log(f"\n  Part 1: C2R damage scan at {len(all_test_layers)} layers")
    
    results = {L: {"c2r_attn": [], "c2r_mlp": [], "r2c_attn": [], "r2c_mlp": []}
               for L in all_test_layers}
    
    t0 = time.time()
    n_test = len(TEST_PAIRS)
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue
        
        clean_prompt = f"The {obj}"
        
        # Capture activations
        clean_acts, clean_logits = capture_attn_mlp_activations(
            model, tokenizer, device, clean_prompt, all_test_layers)
        corrupt_acts, corrupt_logits = capture_attn_mlp_activations(
            model, tokenizer, device, CORRUPTED_BASELINE, all_test_layers)
        
        clean_target = float(clean_logits[tid_t])
        clean_compet = float(clean_logits[tid_c])
        corrupt_target = float(corrupt_logits[tid_t])
        corrupt_compet = float(corrupt_logits[tid_c])
        clean_gap = clean_target - clean_compet
        corrupt_gap = corrupt_target - corrupt_compet
        base_gap = clean_gap - corrupt_gap
        
        if abs(base_gap) < 1e-10:
            del clean_acts, corrupt_acts
            gc.collect(); torch.cuda.empty_cache()
            continue
        
        for li in all_test_layers:
            clean_attn = clean_acts.get(f"attn_out_{li}")
            clean_mlp = clean_acts.get(f"mlp_out_{li}")
            corrupt_attn = corrupt_acts.get(f"attn_out_{li}")
            corrupt_mlp = corrupt_acts.get(f"mlp_out_{li}")
            
            if clean_attn is None or corrupt_attn is None:
                continue
            
            # --- C2R-attn: replace clean attn_out with corrupt ---
            hook = get_layers(model)[li].self_attn.register_forward_hook(
                _make_output_patch_hook_last(corrupt_attn))
            inp = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                pout = model(**inp, output_hidden_states=False)
            p_logits = pout.logits[0, -1].float().cpu().numpy()
            hook.remove()
            pt, pc = float(p_logits[tid_t]), float(p_logits[tid_c])
            delta_gap = (pt - pc) - clean_gap
            effect = compute_effect(delta_gap, base_gap, "c2r")
            results[li]["c2r_attn"].append({"effect": effect, "delta_gap": delta_gap})
            gc.collect(); torch.cuda.empty_cache()
            
            # --- C2R-mlp: replace clean mlp_out with corrupt ---
            hook = get_layers(model)[li].mlp.register_forward_hook(
                _make_output_patch_hook_last(corrupt_mlp))
            inp = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                pout = model(**inp, output_hidden_states=False)
            p_logits = pout.logits[0, -1].float().cpu().numpy()
            hook.remove()
            pt, pc = float(p_logits[tid_t]), float(p_logits[tid_c])
            delta_gap = (pt - pc) - clean_gap
            effect = compute_effect(delta_gap, base_gap, "c2r")
            results[li]["c2r_mlp"].append({"effect": effect, "delta_gap": delta_gap})
            gc.collect(); torch.cuda.empty_cache()
            
            # --- R2C only at core layers ---
            if li in core_layers:
                # R2C-attn: replace corrupt attn_out with clean
                hook = get_layers(model)[li].self_attn.register_forward_hook(
                    _make_output_patch_hook_last(clean_attn))
                inp = tokenizer(CORRUPTED_BASELINE, return_tensors="pt", truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    pout = model(**inp, output_hidden_states=False)
                p_logits = pout.logits[0, -1].float().cpu().numpy()
                hook.remove()
                pt, pc = float(p_logits[tid_t]), float(p_logits[tid_c])
                delta_gap = (pt - pc) - corrupt_gap
                effect = compute_effect(delta_gap, base_gap, "r2c")
                results[li]["r2c_attn"].append({"effect": effect, "delta_gap": delta_gap})
                gc.collect(); torch.cuda.empty_cache()
                
                # R2C-mlp: replace corrupt mlp_out with clean
                hook = get_layers(model)[li].mlp.register_forward_hook(
                    _make_output_patch_hook_last(clean_mlp))
                inp = tokenizer(CORRUPTED_BASELINE, return_tensors="pt", truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    pout = model(**inp, output_hidden_states=False)
                p_logits = pout.logits[0, -1].float().cpu().numpy()
                hook.remove()
                pt, pc = float(p_logits[tid_t]), float(p_logits[tid_c])
                delta_gap = (pt - pc) - corrupt_gap
                effect = compute_effect(delta_gap, base_gap, "r2c")
                results[li]["r2c_mlp"].append({"effect": effect, "delta_gap": delta_gap})
                gc.collect(); torch.cuda.empty_cache()
        
        del clean_acts, corrupt_acts
        gc.collect(); torch.cuda.empty_cache()
        
        if (pidx + 1) % 5 == 0:
            elapsed = time.time() - t0
            gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  Part1 [{pidx+1}/{n_test}] elapsed={elapsed:.0f}s GPU={gpu_gb:.1f}GB")
    
    return results, all_test_layers


# ===== Part 2: Multi-position h_in Patch =====

def run_part2(model, tokenizer, device, model_name):
    """Multi-position h_in patch at core layers."""
    cfg = MODEL_CONFIGS[model_name]
    core_layers = cfg["core_layers"]
    
    log(f"\n  Part 2: Multi-position h_in patch at layers {core_layers}")
    
    positions_to_test = ["last_token", "object_token", "all_tokens"]
    results = {L: {pos: [] for pos in positions_to_test} for L in core_layers}
    
    t0 = time.time()
    n_test = len(TEST_PAIRS)
    layers_obj = get_layers(model)
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue
        
        clean_prompt = f"The {obj}"
        
        # Find object token positions in clean and corrupt prompts
        obj_positions_clean = find_object_positions(tokenizer, clean_prompt, obj)
        obj_positions_corrupt = find_object_positions(tokenizer, CORRUPTED_BASELINE, "item")
        
        # Capture h_in at ALL positions for clean and corrupt
        clean_h_all, clean_logits, clean_seq_len = capture_h_in_all_positions(
            model, tokenizer, device, clean_prompt, core_layers)
        corrupt_h_all, corrupt_logits, corrupt_seq_len = capture_h_in_all_positions(
            model, tokenizer, device, CORRUPTED_BASELINE, core_layers)
        
        clean_target = float(clean_logits[tid_t])
        clean_compet = float(clean_logits[tid_c])
        corrupt_target = float(corrupt_logits[tid_t])
        corrupt_compet = float(corrupt_logits[tid_c])
        clean_gap = clean_target - clean_compet
        corrupt_gap = corrupt_target - corrupt_compet
        base_gap = clean_gap - corrupt_gap
        
        if abs(base_gap) < 1e-10:
            del clean_h_all, corrupt_h_all
            gc.collect(); torch.cuda.empty_cache()
            continue
        
        for li in core_layers:
            clean_h_li = clean_h_all.get(f"h_in_{li}")  # [seq_len, d_model]
            corrupt_h_li = corrupt_h_all.get(f"h_in_{li}")
            
            if clean_h_li is None or corrupt_h_li is None:
                continue
            
            for pos_type in positions_to_test:
                if pos_type == "last_token":
                    # Patch last token only (standard approach)
                    # R2C: corrupt prompt, replace last token h_in with clean
                    last_pos_corrupt = corrupt_seq_len - 1
                    repl = {last_pos_corrupt: clean_h_li[-1]}
                    
                    hook = layers_obj[li].register_forward_pre_hook(
                        _make_input_patch_hook_positions(repl))
                    inp = tokenizer(CORRUPTED_BASELINE, return_tensors="pt",
                                    truncation=True, max_length=128).to(device)
                    with torch.no_grad():
                        pout = model(**inp, output_hidden_states=False)
                    p_logits = pout.logits[0, -1].float().cpu().numpy()
                    hook.remove()
                    
                elif pos_type == "object_token":
                    # Patch object token positions only
                    # For corrupt prompt: patch positions that correspond to "item"
                    repl = {}
                    for p in obj_positions_corrupt:
                        if p < corrupt_h_li.shape[0]:
                            # Use clean h_in at corresponding position
                            # Map corrupt position to clean position (same offset)
                            clean_p = min(p, clean_h_li.shape[0] - 1)
                            repl[p] = clean_h_li[clean_p]
                    
                    if not repl:
                        continue
                    
                    hook = layers_obj[li].register_forward_pre_hook(
                        _make_input_patch_hook_positions(repl))
                    inp = tokenizer(CORRUPTED_BASELINE, return_tensors="pt",
                                    truncation=True, max_length=128).to(device)
                    with torch.no_grad():
                        pout = model(**inp, output_hidden_states=False)
                    p_logits = pout.logits[0, -1].float().cpu().numpy()
                    hook.remove()
                    
                elif pos_type == "all_tokens":
                    # Patch all token positions
                    # Use clean h_in for positions that exist in both
                    min_len = min(clean_h_li.shape[0], corrupt_h_li.shape[0])
                    full_repl = np.zeros_like(corrupt_h_li)
                    full_repl[:min_len] = clean_h_li[:min_len]
                    # For positions beyond clean length, keep corrupt
                    if corrupt_h_li.shape[0] > min_len:
                        full_repl[min_len:] = corrupt_h_li[min_len:]
                    
                    hook = layers_obj[li].register_forward_pre_hook(
                        _make_input_patch_hook_all(full_repl))
                    inp = tokenizer(CORRUPTED_BASELINE, return_tensors="pt",
                                    truncation=True, max_length=128).to(device)
                    with torch.no_grad():
                        pout = model(**inp, output_hidden_states=False)
                    p_logits = pout.logits[0, -1].float().cpu().numpy()
                    hook.remove()
                
                pt, pc = float(p_logits[tid_t]), float(p_logits[tid_c])
                p_gap = pt - pc
                delta_gap = p_gap - corrupt_gap
                effect = compute_effect(delta_gap, base_gap, "r2c")
                results[li][pos_type].append({"effect": effect, "delta_gap": delta_gap})
                
                gc.collect(); torch.cuda.empty_cache()
        
        del clean_h_all, corrupt_h_all
        gc.collect(); torch.cuda.empty_cache()
        
        if (pidx + 1) % 5 == 0:
            elapsed = time.time() - t0
            gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  Part2 [{pidx+1}/{n_test}] elapsed={elapsed:.0f}s GPU={gpu_gb:.1f}GB")
    
    return results, core_layers, positions_to_test


# ===== Part 3: h_in Component Decomposition =====

def run_part3(model, tokenizer, device, model_name, W_U):
    """Decompose h_in diff into binding-parallel and orthogonal components."""
    cfg = MODEL_CONFIGS[model_name]
    core_layers = cfg["core_layers"]
    d_model = cfg["d_model"]
    
    log(f"\n  Part 3: h_in decomposition at layers {core_layers}")
    
    components = ["full_diff", "binding_parallel", "orthogonal"]
    results = {L: {comp: [] for comp in components} for L in core_layers}
    
    t0 = time.time()
    n_test = len(TEST_PAIRS)
    layers_obj = get_layers(model)
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue
        
        clean_prompt = f"The {obj}"
        
        # Capture h_in at last token for clean and corrupt
        clean_h_all, clean_logits, _ = capture_h_in_all_positions(
            model, tokenizer, device, clean_prompt, core_layers)
        corrupt_h_all, corrupt_logits, _ = capture_h_in_all_positions(
            model, tokenizer, device, CORRUPTED_BASELINE, core_layers)
        
        clean_target = float(clean_logits[tid_t])
        clean_compet = float(clean_logits[tid_c])
        corrupt_target = float(corrupt_logits[tid_t])
        corrupt_compet = float(corrupt_logits[tid_c])
        clean_gap = clean_target - clean_compet
        corrupt_gap = corrupt_target - corrupt_compet
        base_gap = clean_gap - corrupt_gap
        
        if abs(base_gap) < 1e-10:
            del clean_h_all, corrupt_h_all
            gc.collect(); torch.cuda.empty_cache()
            continue
        
        # Compute binding direction in d_model space
        # binding_dir = W_U[target_id] - W_U[competitor_id]
        binding_dir = W_U[tid_t] - W_U[tid_c]  # [d_model]
        binding_dir_norm = np.linalg.norm(binding_dir)
        if binding_dir_norm > 1e-10:
            binding_dir_unit = binding_dir / binding_dir_norm
        else:
            binding_dir_unit = np.zeros(d_model)
        
        for li in core_layers:
            clean_h_last = clean_h_all.get(f"h_in_{li}")  # [seq_len, d_model]
            corrupt_h_last = corrupt_h_all.get(f"h_in_{li}")
            
            if clean_h_last is None or corrupt_h_last is None:
                continue
            
            # h_in diff at last token
            h_diff = clean_h_last[-1] - corrupt_h_last[-1]  # [d_model]
            corrupt_h_last_token = corrupt_h_last[-1]  # [d_model]
            
            # Decompose h_diff
            # binding_parallel = (h_diff · binding_dir_unit) * binding_dir_unit
            parallel_coeff = np.dot(h_diff, binding_dir_unit)
            h_diff_parallel = parallel_coeff * binding_dir_unit
            h_diff_orthogonal = h_diff - h_diff_parallel
            
            for comp in components:
                if comp == "full_diff":
                    patch_component = h_diff
                elif comp == "binding_parallel":
                    patch_component = h_diff_parallel
                elif comp == "orthogonal":
                    patch_component = h_diff_orthogonal
                else:
                    continue
                
                # R2C: corrupt prompt, add component to corrupt h_in at last token
                hook = layers_obj[li].register_forward_pre_hook(
                    _make_input_patch_hook_component(patch_component, corrupt_h_last_token))
                inp = tokenizer(CORRUPTED_BASELINE, return_tensors="pt",
                                truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    pout = model(**inp, output_hidden_states=False)
                p_logits = pout.logits[0, -1].float().cpu().numpy()
                hook.remove()
                
                pt, pc = float(p_logits[tid_t]), float(p_logits[tid_c])
                p_gap = pt - pc
                delta_gap = p_gap - corrupt_gap
                effect = compute_effect(delta_gap, base_gap, "r2c")
                
                # Also store decomposition info
                results[li][comp].append({
                    "effect": effect,
                    "delta_gap": delta_gap,
                    "parallel_coeff": float(parallel_coeff),
                    "h_diff_norm": float(np.linalg.norm(h_diff)),
                    "parallel_norm": float(np.linalg.norm(h_diff_parallel)),
                    "orthogonal_norm": float(np.linalg.norm(h_diff_orthogonal)),
                    "parallel_fraction": float(np.linalg.norm(h_diff_parallel)**2 / 
                                               max(np.linalg.norm(h_diff)**2, 1e-20)),
                })
                
                gc.collect(); torch.cuda.empty_cache()
        
        del clean_h_all, corrupt_h_all
        gc.collect(); torch.cuda.empty_cache()
        
        if (pidx + 1) % 5 == 0:
            elapsed = time.time() - t0
            gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  Part3 [{pidx+1}/{n_test}] elapsed={elapsed:.0f}s GPU={gpu_gb:.1f}GB")
    
    return results, core_layers, components


# ===== Main Experiment =====

def run_experiment(model_name):
    log(f"Phase 363: Per-layer C2R Scan + Multi-position + Decomposition ({model_name})")
    log("=" * 70)
    t0 = time.time()
    cfg = MODEL_CONFIGS[model_name]
    
    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    
    n_test = len(TEST_PAIRS)
    log(f"  Model: {model_name}, n_layers={cfg['n_layers']}, "
        f"sample_layers={cfg['sample_layers']}, core_layers={cfg['core_layers']}")
    
    # ===== Part 1: Per-layer C2R damage scan =====
    part1_results, part1_layers = run_part1(model, tokenizer, device, model_name)
    
    # ===== Part 2: Multi-position h_in patch =====
    part2_results, part2_layers, part2_positions = run_part2(model, tokenizer, device, model_name)
    
    # ===== Part 3: h_in decomposition =====
    part3_results, part3_layers, part3_components = run_part3(model, tokenizer, device, model_name, W_U)
    
    # ================================================================
    # Summary
    # ================================================================
    log(f"\n  {'='*100}")
    log(f"  Phase 363 Summary: {model_name}")
    log(f"  {'='*100}")
    
    # --- Part 1: C2R damage scan ---
    log(f"\n  === Part 1: Per-layer C2R Damage Scan ===")
    log(f"  {'Layer':>6} {'C2R-attn':>12} {'C2R-mlp':>12} {'attn+mlp':>12} {'attn>mlp':>10} {'n':>4}")
    log(f"  {'-'*60}")
    
    part1_summary = {}
    peak_attn_layer = None
    peak_mlp_layer = None
    peak_attn_effect = -999
    peak_mlp_effect = -999
    
    for li in sorted(part1_layers):
        attn_vals = part1_results[li]["c2r_attn"]
        mlp_vals = part1_results[li]["c2r_mlp"]
        n = len(attn_vals)
        if n == 0:
            continue
        
        attn_mean = float(np.mean([v["effect"] for v in attn_vals]))
        mlp_mean = float(np.mean([v["effect"] for v in mlp_vals]))
        combined = attn_mean + mlp_mean
        dominant = "attn" if attn_mean > mlp_mean else "MLP"
        
        part1_summary[li] = {
            "c2r_attn_mean": attn_mean,
            "c2r_mlp_mean": mlp_mean,
            "c2r_combined": combined,
            "n": n,
        }
        
        if attn_mean > peak_attn_effect:
            peak_attn_effect = attn_mean
            peak_attn_layer = li
        if mlp_mean > peak_mlp_effect:
            peak_mlp_effect = mlp_mean
            peak_mlp_layer = li
        
        log(f"  L{li:>4} {attn_mean:>+12.4f} {mlp_mean:>+12.4f} {combined:>+12.4f} {dominant:>10} {n:>4}")
    
    log(f"\n  Peak C2R-attn layer: L{peak_attn_layer} (effect={peak_attn_effect:+.4f})")
    log(f"  Peak C2R-mlp layer: L{peak_mlp_layer} (effect={peak_mlp_effect:+.4f})")
    
    # R2C at core layers
    log(f"\n  --- R2C at Core Layers ---")
    log(f"  {'Layer':>6} {'R2C-attn':>12} {'R2C-mlp':>12} {'n':>4}")
    log(f"  {'-'*40}")
    for li in cfg["core_layers"]:
        attn_vals = part1_results[li]["r2c_attn"]
        mlp_vals = part1_results[li]["r2c_mlp"]
        n = len(attn_vals)
        if n == 0:
            continue
        attn_mean = float(np.mean([v["effect"] for v in attn_vals]))
        mlp_mean = float(np.mean([v["effect"] for v in mlp_vals]))
        log(f"  L{li:>4} {attn_mean:>+12.4f} {mlp_mean:>+12.4f} {n:>4}")
    
    # Bootstrap CI for key layers
    log(f"\n  --- Bootstrap 95% CI for C2R damage (key layers) ---")
    np.random.seed(42)
    n_bootstrap = 1000
    key_layers_for_ci = sorted(set([0, 1, 2] + 
                                    [peak_attn_layer, peak_mlp_layer] + 
                                    cfg["core_layers"]))
    key_layers_for_ci = [L for L in key_layers_for_ci if L in part1_results 
                         and len(part1_results[L]["c2r_attn"]) >= 5]
    
    for li in key_layers_for_ci:
        for cond in ["c2r_attn", "c2r_mlp"]:
            vals = part1_results[li][cond]
            if len(vals) < 5:
                continue
            effects = np.array([v["effect"] for v in vals])
            boot_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(effects, size=len(effects), replace=True)
                boot_means.append(float(np.mean(sample)))
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))
            mean_eff = float(np.mean(effects))
            log(f"  L{li:>4} {cond:>10}: {mean_eff:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    
    # --- Part 2: Multi-position ---
    log(f"\n  === Part 2: Multi-position h_in Patch (R2C) ===")
    log(f"  {'Layer':>6}", end="")
    for pos in part2_positions:
        log(f" {pos:>14}", end="")
    log(f" {'n':>4}")
    log(f"  {'-'*65}")
    
    part2_summary = {}
    for li in part2_layers:
        row_str = f"  L{li:>4}"
        n = 0
        for pos in part2_positions:
            vals = part2_results[li][pos]
            n = len(vals)
            if n == 0:
                row_str += f" {'N/A':>14}"
                continue
            mean_eff = float(np.mean([v["effect"] for v in vals]))
            row_str += f" {mean_eff:>+14.4f}"
            part2_summary[f"L{li}_{pos}"] = mean_eff
        row_str += f" {n:>4}"
        log(row_str)
    
    # --- Part 3: h_in Decomposition ---
    log(f"\n  === Part 3: h_in Component Decomposition (R2C) ===")
    log(f"  {'Layer':>6}", end="")
    for comp in part3_components:
        log(f" {comp:>16}", end="")
    log(f" {'par_frac':>10} {'n':>4}")
    log(f"  {'-'*80}")
    
    part3_summary = {}
    for li in part3_layers:
        row_str = f"  L{li:>4}"
        n = 0
        par_frac_mean = 0
        for comp in part3_components:
            vals = part3_results[li][comp]
            n = len(vals)
            if n == 0:
                row_str += f" {'N/A':>16}"
                continue
            mean_eff = float(np.mean([v["effect"] for v in vals]))
            row_str += f" {mean_eff:>+16.4f}"
            part3_summary[f"L{li}_{comp}"] = mean_eff
            if comp == "full_diff":
                par_frac_mean = float(np.mean([v["parallel_fraction"] for v in vals]))
        row_str += f" {par_frac_mean:>10.4f} {n:>4}"
        log(row_str)
    
    # Decomposition details
    log(f"\n  --- Decomposition Norms ---")
    log(f"  {'Layer':>6} {'h_diff_norm':>12} {'par_norm':>12} {'orth_norm':>12} {'par_frac':>10}")
    log(f"  {'-'*60}")
    for li in part3_layers:
        vals = part3_results[li]["full_diff"]
        if len(vals) == 0:
            continue
        h_diff_norm = float(np.mean([v["h_diff_norm"] for v in vals]))
        par_norm = float(np.mean([v["parallel_norm"] for v in vals]))
        orth_norm = float(np.mean([v["orthogonal_norm"] for v in vals]))
        par_frac = float(np.mean([v["parallel_fraction"] for v in vals]))
        log(f"  L{li:>4} {h_diff_norm:>12.4f} {par_norm:>12.4f} {orth_norm:>12.4f} {par_frac:>10.4f}")
    
    # ================================================================
    # Save
    # ================================================================
    output = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "363",
        "unified_notation": {
            "C2R_effect": "-Δgap / |base_gap| (positive = binding damaged)",
            "R2C_effect": "+Δgap / |base_gap| (positive = binding rescued)",
            "base_gap": "clean_gap - corrupt_gap",
        },
        "sample_layers": cfg["sample_layers"],
        "core_layers": cfg["core_layers"],
        "n_pairs": n_test,
        "part1_summary": {str(k): v for k, v in part1_summary.items()},
        "part2_summary": part2_summary,
        "part3_summary": part3_summary,
        "part1_per_pair": {},
        "part2_per_pair": {},
        "part3_per_pair": {},
    }
    
    for li in part1_layers:
        output["part1_per_pair"][str(li)] = {
            "c2r_attn_effects": [v["effect"] for v in part1_results[li]["c2r_attn"]],
            "c2r_mlp_effects": [v["effect"] for v in part1_results[li]["c2r_mlp"]],
            "r2c_attn_effects": [v["effect"] for v in part1_results[li]["r2c_attn"]],
            "r2c_mlp_effects": [v["effect"] for v in part1_results[li]["r2c_mlp"]],
        }
    
    for li in part2_layers:
        output["part2_per_pair"][str(li)] = {}
        for pos in part2_positions:
            output["part2_per_pair"][str(li)][pos] = {
                "effects": [v["effect"] for v in part2_results[li][pos]],
            }
    
    for li in part3_layers:
        output["part3_per_pair"][str(li)] = {}
        for comp in part3_components:
            output["part3_per_pair"][str(li)][comp] = {
                "effects": [v["effect"] for v in part3_results[li][comp]],
                "parallel_coeffs": [v["parallel_coeff"] for v in part3_results[li][comp]],
                "h_diff_norms": [v["h_diff_norm"] for v in part3_results[li][comp]],
                "parallel_norms": [v["parallel_norm"] for v in part3_results[li][comp]],
                "orthogonal_norms": [v["orthogonal_norm"] for v in part3_results[li][comp]],
                "parallel_fractions": [v["parallel_fraction"] for v in part3_results[li][comp]],
            }
    
    os.makedirs("results/phase363_per_layer_c2r_scan", exist_ok=True)
    out_path = f"results/phase363_per_layer_c2r_scan/{model_name}_phase363.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    log(f"\n  Saved to {out_path}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Phase 363 complete for {model_name} in {time.time()-t0:.0f}s")
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
