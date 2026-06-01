"""
Phase 320: Causal Verification of Relation Activation Directions
=================================================================

Phase 319b proved attribute/function directions have geometric stability
(pair_cos=0.55-0.60), but did NOT prove they have causal efficacy.

This test answers: Do attribute/function activation directions have
causal power when injected into neutral sentences?

Three experiments:
  Part A: Attribute direction causal injection
    - Extract d_attr from "the {N} is {A}" vs "the {N} is just an object"
    - Also extract d_color from "the {N} is red" vs "the {N} is green"
      (pure attribute axis, no object baseline confound)
    - Inject into neutral: "The {N'} is" and measure logit change
    - Test cross-object transfer: apple→red direction into banana→?

  Part B: Function direction causal injection
    - Extract d_func from "people use the {T} to {V}" vs "people use the {T}"
    - Inject into "People use the {T'} to" and measure logit change
    - Test cross-tool transfer: knife→cut direction into scissors→?

  Part C: Negation sub-type decomposition
    - Compare not/never/barely/un-/not bad/did not try/tried not to
    - Test if they form sub-clusters or a single subspace

Usage:
  python tests/glm5/phase320_causal_verification.py qwen3
  python tests/glm5/phase320_causal_verification.py glm4
  python tests/glm5/phase320_causal_verification.py deepseek7b
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model, get_W_U

RESULT_DIR = Path("results/phase320_causal")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except:
            pass


# =====================================================================
# STIMULI
# =====================================================================

# Part A: Attribute causal injection
# Source pairs: extract d_attr from these
ATTRIBUTE_SOURCES = [
    # (noun, attribute, attribute_value)
    ("apple", "color", "red"),
    ("sky", "color", "blue"),
    ("grass", "color", "green"),
    ("banana", "color", "yellow"),
    ("orange", "color", "orange"),
    ("ice", "temperature", "cold"),
    ("fire", "temperature", "hot"),
    ("summer", "temperature", "warm"),
    ("winter", "temperature", "cold"),
    ("lava", "temperature", "hot"),
    ("silk", "texture", "smooth"),
    ("sandpaper", "texture", "rough"),
    ("cotton", "texture", "soft"),
    ("stone", "texture", "hard"),
    ("velvet", "texture", "smooth"),
    ("lemon", "taste", "sour"),
    ("honey", "taste", "sweet"),
    ("chili", "taste", "spicy"),
    ("coffee", "taste", "bitter"),
    ("salt", "taste", "salty"),
]

# Target objects for injection (different from source)
ATTRIBUTE_TARGETS = [
    ("strawberry", "color", "red"),
    ("ocean", "color", "blue"),
    ("emerald", "color", "green"),
    ("sun", "color", "yellow"),
    ("copper", "color", "orange"),
    ("snow", "temperature", "cold"),
    ("stove", "temperature", "hot"),
    ("spring", "temperature", "warm"),
    ("frost", "temperature", "cold"),
    ("desert", "temperature", "hot"),
    ("satin", "texture", "smooth"),
    ("concrete", "texture", "rough"),
    ("wool", "texture", "soft"),
    ("diamond", "texture", "hard"),
    ("leather", "texture", "smooth"),
    ("vinegar", "taste", "sour"),
    ("sugar", "taste", "sweet"),
    ("pepper", "taste", "spicy"),
    ("cocoa", "taste", "bitter"),
    ("seawater", "taste", "salty"),
]

# Same-attribute-value pairs for d_color (pure attribute axis)
SAME_ATTRIBUTE_PAIRS = [
    # (noun, value1, value2) → d = h("the {noun} is {value2}") - h("the {noun} is {value1}")
    ("apple", "red", "green"),
    ("sky", "blue", "gray"),
    ("fire", "hot", "warm"),
    ("ice", "cold", "cool"),
    ("lemon", "sour", "sweet"),
    ("road", "rough", "smooth"),
    ("water", "cold", "hot"),
    ("cake", "sweet", "bitter"),
    ("metal", "hard", "soft"),
    ("wind", "strong", "gentle"),
    ("room", "bright", "dark"),
    ("soup", "hot", "cold"),
    ("wood", "hard", "soft"),
    ("cloth", "smooth", "rough"),
    ("tea", "hot", "cold"),
]


# Part B: Function causal injection
FUNCTION_SOURCES = [
    # (tool, action)
    ("knife", "cut"), ("scissors", "snip"), ("pen", "write"),
    ("pencil", "draw"), ("car", "drive"), ("bicycle", "ride"),
    ("phone", "call"), ("key", "unlock"), ("cup", "drink"),
    ("lamp", "illuminate"), ("clock", "measure"), ("umbrella", "protect"),
    ("camera", "capture"), ("brush", "paint"), ("hammer", "nail"),
    ("saw", "cut"), ("needle", "sew"), ("shovel", "dig"),
    ("thermometer", "measure"), ("compass", "navigate"),
]

# Same-function different tools (for cross-tool transfer)
SAME_FUNCTION_GROUPS = [
    # (tool1, tool2, shared_action)
    ("knife", "scissors", "cut"),
    ("pen", "pencil", "write"),
    ("car", "bus", "drive"),
    ("lamp", "flashlight", "illuminate"),
    ("camera", "microphone", "record"),
    ("clock", "watch", "measure"),
    ("umbrella", "coat", "protect"),
    ("hammer", "mallet", "strike"),
    ("telescope", "microscope", "observe"),
    ("key", "password", "unlock"),
]


# Part C: Negation sub-type decomposition
NEGATION_ITEMS = [
    # (positive, negation_type, negative_sentence)
    # Type: not, never, barely, un-, double_neg, scope_neg
]

# Build negation items programmatically
NEGATION_ADJECTIVES = [
    "happy", "good", "great", "clean", "safe", "warm",
    "fast", "strong", "bright", "quiet", "easy", "soft",
    "rich", "young", "healthy", "comfortable", "clear", "simple",
    "smooth", "fresh",
]

NEGATION_TYPES = {
    "not": lambda adj: (f"very {adj}", f"not {adj}"),
    "never": lambda adj: (f"very {adj}", f"never {adj}"),
    "barely": lambda adj: (f"very {adj}", f"barely {adj}"),
    "morphological": lambda adj: (f"very {adj}", f"un{adj}") if adj[0] in "bhcmnprstw" else None,
    "double_neg": lambda adj: (f"not {adj}", f"not un{adj}") if adj[0] in "bhcmnprstw" else None,
    "scope_neg": lambda adj: (f"tried to be {adj}", f"did not try to be {adj}"),
}


# =====================================================================
# MODEL LOADING (same as phase319b)
# =====================================================================

def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    attn_impl = "flash_attention_2"
    log(f"Loading {model_name} (bf16 + device_map=auto + {attn_impl})...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation=attn_impl,
        )
        log(f"  Loaded with {attn_impl}")
    except Exception as e:
        log(f"  flash_attention_2 failed ({e}), falling back to sdpa")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation="sdpa",
            )
            log(f"  Loaded with sdpa")
        except Exception as e2:
            log(f"  sdpa failed ({e2}), falling back to eager")
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation="eager",
            )
            log(f"  Loaded with eager")

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Model: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        log(f"  Layer allocation: GPU={gpu_count}, CPU={cpu_count} components")

    return model, tokenizer, device


# =====================================================================
# REPRESENTATION EXTRACTION
# =====================================================================

def get_target_layers(n_layers):
    if n_layers >= 36:
        return [6, 12, 18, 24, n_layers - 2]
    elif n_layers >= 28:
        return [4, 8, 12, 16, n_layers - 2]
    else:
        return [2, 4, 8, 12, n_layers - 2]


def extract_last_token_rep(model, tokenizer, device, sentences, target_layers, label=""):
    """Extract last-token representations for a list of sentences."""
    layers_list = get_layers(model)
    cache = {}
    captured = {}

    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[li] = output[0].detach().float().cpu()
            else:
                captured[li] = output.detach().float().cpu()
        return hook_fn

    hooks = [layers_list[li].register_forward_hook(make_hook(li)) for li in target_layers]

    try:
        for idx, sent in enumerate(sentences):
            inp = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128).to(device)
            captured.clear()
            with torch.no_grad():
                model(**inp)

            for li in target_layers:
                if li in captured:
                    cache[(sent, li)] = captured[li][0, -1].numpy()

            if (idx + 1) % 50 == 0 or idx == len(sentences) - 1:
                log(f"    {label} Extracted {idx+1}/{len(sentences)}, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

            if (idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    return cache


def get_logits_for_prompt(model, tokenizer, device, prompt, top_k=20):
    """Get top-k logits for next token prediction."""
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1].float().cpu().numpy()
    top_k_ids = np.argsort(logits)[-top_k:][::-1]
    top_k_tokens = [(tokenizer.decode([i]).strip().lower(), float(logits[i])) for i in top_k_ids]
    return logits, top_k_tokens


def inject_direction_and_get_logits(model, tokenizer, device, prompt, direction, layer_idx, alpha, top_k=20):
    """
    Inject direction at a specific layer's output and get logits.
    
    This modifies the residual stream at layer_idx by adding alpha * direction.
    """
    layers_list = get_layers(model)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    injected_logits = None
    
    def hook_fn(module, input, output):
        nonlocal injected_logits
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        # Add direction to last token position
        d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
        hidden_modified = hidden.clone()
        hidden_modified[0, -1, :] += (alpha * d_tensor).to(hidden.dtype)
        
        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        return hidden_modified
    
    hook = layers_list[layer_idx].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            out = model(**inp)
        injected_logits = out.logits[0, -1].float().cpu().numpy()
    finally:
        hook.remove()
    
    top_k_ids = np.argsort(injected_logits)[-top_k:][::-1]
    top_k_tokens = [(tokenizer.decode([i]).strip().lower(), float(injected_logits[i])) for i in top_k_ids]
    return injected_logits, top_k_tokens


# =====================================================================
# PART A: ATTRIBUTE CAUSAL INJECTION
# =====================================================================

def run_attribute_causal(model, tokenizer, device, info, cache, target_layers):
    """
    Test whether attribute activation directions have causal efficacy.
    
    Design:
    1. Extract d_attr from source: h("the apple is red") - h("the apple is just an object")
    2. Extract d_color (pure axis): h("the apple is green") - h("the apple is red")
    3. Inject d_attr into neutral sentence "The banana is" at various layers
    4. Measure: does logit for "yellow"/"sweet"/color words increase?
    5. Compare d_attr vs d_color injection effects
    """
    log("\n" + "="*60)
    log("Part A: Attribute Direction Causal Injection")
    log("="*60)
    
    W_U = get_W_U(model, info.name)
    results = {}
    deepest_li = target_layers[-1]
    
    # --- A1: Extract source directions (from cache) ---
    log("  A1: Extracting source attribute directions...")
    attr_directions = {}  # (noun, attr_type, value) -> {layer: direction}
    
    for noun, attr_type, value in ATTRIBUTE_SOURCES:
        sent_pos = f"the {noun} is {value}"
        sent_neg = f"the {noun} is just an object"
        
        for li in target_layers:
            k_pos = (sent_pos, li)
            k_neg = (sent_neg, li)
            if k_pos in cache and k_neg in cache:
                d = cache[k_pos] - cache[k_neg]
                key = (noun, attr_type, value)
                if key not in attr_directions:
                    attr_directions[key] = {}
                attr_directions[key][li] = d
    
    log(f"  Extracted {len(attr_directions)} attribute directions")
    
    # --- A2: Extract pure color/attribute axis directions ---
    log("  A2: Extracting pure attribute axis directions (same-object, different values)...")
    axis_directions = {}  # (noun, value1, value2) -> {layer: direction}
    
    for noun, val1, val2 in SAME_ATTRIBUTE_PAIRS:
        sent1 = f"the {noun} is {val1}"
        sent2 = f"the {noun} is {val2}"
        
        for li in target_layers:
            k1 = (sent1, li)
            k2 = (sent2, li)
            if k1 in cache and k2 in cache:
                d = cache[k2] - cache[k1]
                key = (noun, val1, val2)
                if key not in axis_directions:
                    axis_directions[key] = {}
                axis_directions[key][li] = d
    
    log(f"  Extracted {len(axis_directions)} axis directions")
    
    # --- A3: Causal injection test ---
    log("  A3: Causal injection — attribute directions into neutral sentences...")
    
    # Test injection at deepest layer
    injection_results = []
    alphas = [0.5, 1.0, 2.0, 4.0]
    
    # For each source direction, inject into a DIFFERENT target object
    # and measure whether the expected attribute value increases in logits
    
    # Build attribute-type to expected-value mapping for targets
    attr_type_values = {
        "color": ["red", "blue", "green", "yellow", "orange", "white", "black", "brown", "pink", "purple"],
        "temperature": ["hot", "cold", "warm", "cool", "freezing", "boiling"],
        "texture": ["smooth", "rough", "soft", "hard", "silky", "coarse"],
        "taste": ["sweet", "sour", "bitter", "salty", "spicy", "savory"],
    }
    
    # Sample: use 10 source-target pairs per attribute type
    test_pairs = []
    for attr_type in ["color", "temperature", "texture", "taste"]:
        sources = [(n, at, v) for n, at, v in ATTRIBUTE_SOURCES if at == attr_type]
        targets = [(n, at, v) for n, at, v in ATTRIBUTE_TARGETS if at == attr_type]
        for i in range(min(5, len(sources), len(targets))):
            test_pairs.append((sources[i], targets[i]))
    
    log(f"  Testing {len(test_pairs)} source→target pairs at layer L{deepest_li}")
    
    # Get baseline logits for target sentences
    for src_key, tgt_key in test_pairs:
        src_noun, src_attr, src_val = src_key
        tgt_noun, tgt_attr, tgt_val = tgt_key
        
        if src_key not in attr_directions or deepest_li not in attr_directions[src_key]:
            continue
        
        d_attr = attr_directions[src_key][deepest_li]
        d_norm = np.linalg.norm(d_attr)
        if d_norm < 1e-10:
            continue
        d_attr_unit = d_attr / d_norm
        
        # Target prompt
        target_prompt = f"The {tgt_noun} is"
        
        # Get baseline logits
        baseline_logits, baseline_top = get_logits_for_prompt(model, tokenizer, device, target_prompt, top_k=30)
        
        # Get expected value token IDs
        expected_values = attr_type_values.get(tgt_attr, [tgt_val])
        
        for alpha in alphas:
            # Inject d_attr direction
            inj_logits, inj_top = inject_direction_and_get_logits(
                model, tokenizer, device, target_prompt,
                d_attr_unit, deepest_li, alpha, top_k=30
            )
            
            # Measure logit change for expected values
            logit_changes = {}
            for val in expected_values:
                val_ids = tokenizer.encode(val, add_special_tokens=False)
                if val_ids:
                    vid = val_ids[0]
                    delta = float(inj_logits[vid] - baseline_logits[vid])
                    logit_changes[val] = delta
            
            # Measure logit change for the specific expected value
            tgt_val_ids = tokenizer.encode(tgt_val, add_special_tokens=False)
            tgt_val_delta = 0.0
            if tgt_val_ids:
                tgt_val_delta = float(inj_logits[tgt_val_ids[0]] - baseline_logits[tgt_val_ids[0]])
            
            # Measure overall top-k shift
            baseline_top_set = set(t for t, _ in baseline_top[:10])
            inj_top_set = set(t for t, _ in inj_top[:10])
            new_in_top10 = inj_top_set - baseline_top_set
            
            injection_results.append({
                "source": f"{src_noun}→{src_val}",
                "target": f"{tgt_noun}→{tgt_attr}",
                "attr_type": tgt_attr,
                "alpha": alpha,
                "target_value_delta": round(tgt_val_delta, 3),
                "expected_value_deltas": {k: round(v, 3) for k, v in logit_changes.items()},
                "new_in_top10": list(new_in_top10),
                "baseline_top5": [(t, round(s, 2)) for t, s in baseline_top[:5]],
                "injected_top5": [(t, round(s, 2)) for t, s in inj_top[:5]],
            })
    
    # --- A4: Pure axis injection ---
    log("  A4: Pure attribute axis injection (same-object different-value)...")
    
    axis_injection_results = []
    for noun, val1, val2 in SAME_ATTRIBUTE_PAIRS:
        key = (noun, val1, val2)
        if key not in axis_directions or deepest_li not in axis_directions[key]:
            continue
        
        d_axis = axis_directions[key][deepest_li]
        d_norm = np.linalg.norm(d_axis)
        if d_norm < 1e-10:
            continue
        d_axis_unit = d_axis / d_norm
        
        # Inject into SAME object with neutral prompt
        target_prompt = f"the {noun} is"
        
        baseline_logits, baseline_top = get_logits_for_prompt(model, tokenizer, device, target_prompt, top_k=30)
        
        for alpha in [1.0, 2.0, 4.0]:
            inj_logits, inj_top = inject_direction_and_get_logits(
                model, tokenizer, device, target_prompt,
                d_axis_unit, deepest_li, alpha, top_k=30
            )
            
            # Check val2 (target direction) and val1 (source direction)
            val2_ids = tokenizer.encode(val2, add_special_tokens=False)
            val1_ids = tokenizer.encode(val1, add_special_tokens=False)
            
            delta_val2 = float(inj_logits[val2_ids[0]] - baseline_logits[val2_ids[0]]) if val2_ids else 0
            delta_val1 = float(inj_logits[val1_ids[0]] - baseline_logits[val1_ids[0]]) if val1_ids else 0
            
            axis_injection_results.append({
                "source": f"{noun}: {val1}→{val2}",
                "alpha": alpha,
                "delta_target_value": round(delta_val2, 3),
                "delta_source_value": round(delta_val1, 3),
                "ratio": round(delta_val2 / max(abs(delta_val1), 0.001), 2) if abs(delta_val1) > 0.001 else "N/A",
            })
    
    # --- A5: Cross-object transfer for same attribute type ---
    log("  A5: Cross-object transfer — color direction from apple→red into banana...")
    
    transfer_results = []
    # Use first 5 color sources, inject into first 5 color targets
    color_sources = [(n, at, v) for n, at, v in ATTRIBUTE_SOURCES if at == "color"][:5]
    color_targets = [(n, at, v) for n, at, v in ATTRIBUTE_TARGETS if at == "color"][:5]
    
    for src_key in color_sources:
        if src_key not in attr_directions or deepest_li not in attr_directions[src_key]:
            continue
        d_src = attr_directions[src_key][deepest_li]
        d_norm = np.linalg.norm(d_src)
        if d_norm < 1e-10:
            continue
        d_src_unit = d_src / d_norm
        
        for tgt_key in color_targets:
            tgt_noun, _, tgt_val = tgt_key
            target_prompt = f"The {tgt_noun} is"
            baseline_logits, _ = get_logits_for_prompt(model, tokenizer, device, target_prompt, top_k=30)
            
            inj_logits, inj_top = inject_direction_and_get_logits(
                model, tokenizer, device, target_prompt,
                d_src_unit, deepest_li, 2.0, top_k=30
            )
            
            tgt_val_ids = tokenizer.encode(tgt_val, add_special_tokens=False)
            delta = float(inj_logits[tgt_val_ids[0]] - baseline_logits[tgt_val_ids[0]]) if tgt_val_ids else 0
            
            # Also check if any color word appears in top5
            color_words = set(attr_type_values["color"])
            inj_top5_words = set(t for t, _ in inj_top[:5])
            color_in_top5 = inj_top5_words & color_words
            
            transfer_results.append({
                "source": f"{src_key[0]}→{src_key[2]}",
                "target": f"{tgt_noun}→{tgt_val}",
                "delta_expected": round(delta, 3),
                "color_in_top5": list(color_in_top5),
            })
    
    results["attribute_injection"] = injection_results
    results["axis_injection"] = axis_injection_results
    results["attribute_transfer"] = transfer_results
    
    # Summary
    log("\n  --- Attribute Causal Summary ---")
    if injection_results:
        for alpha in alphas:
            alpha_results = [r for r in injection_results if r["alpha"] == alpha]
            if alpha_results:
                mean_delta = np.mean([r["target_value_delta"] for r in alpha_results])
                pos_frac = np.mean([1 if r["target_value_delta"] > 0 else 0 for r in alpha_results])
                log(f"  d_attr injection alpha={alpha}: mean_delta={mean_delta:.3f}, frac_positive={pos_frac:.2f}")
    
    if axis_injection_results:
        for alpha in [1.0, 2.0, 4.0]:
            alpha_results = [r for r in axis_injection_results if r["alpha"] == alpha]
            if alpha_results:
                mean_tgt = np.mean([r["delta_target_value"] for r in alpha_results])
                mean_src = np.mean([r["delta_source_value"] for r in alpha_results])
                pos_frac = np.mean([1 if r["delta_target_value"] > 0 else 0 for r in alpha_results])
                log(f"  d_axis injection alpha={alpha}: mean_target_delta={mean_tgt:.3f}, mean_source_delta={mean_src:.3f}, frac_positive={pos_frac:.2f}")
    
    if transfer_results:
        mean_transfer = np.mean([r["delta_expected"] for r in transfer_results])
        pos_frac = np.mean([1 if r["delta_expected"] > 0 else 0 for r in transfer_results])
        frac_color_top5 = np.mean([1 if r["color_in_top5"] else 0 for r in transfer_results])
        log(f"  Cross-object transfer: mean_delta={mean_transfer:.3f}, frac_positive={pos_frac:.2f}, frac_color_top5={frac_color_top5:.2f}")
    
    return results


# =====================================================================
# PART B: FUNCTION CAUSAL INJECTION
# =====================================================================

def run_function_causal(model, tokenizer, device, info, cache, target_layers):
    """
    Test whether function activation directions have causal efficacy.
    """
    log("\n" + "="*60)
    log("Part B: Function Direction Causal Injection")
    log("="*60)
    
    results = {}
    deepest_li = target_layers[-1]
    
    # --- B1: Extract source function directions ---
    log("  B1: Extracting source function directions...")
    func_directions = {}  # (tool, action) -> {layer: direction}
    
    for tool, action in FUNCTION_SOURCES:
        sent_pos = f"people use the {tool} to {action}"
        sent_neg = f"people use the {tool}"
        
        for li in target_layers:
            k_pos = (sent_pos, li)
            k_neg = (sent_neg, li)
            if k_pos in cache and k_neg in cache:
                d = cache[k_pos] - cache[k_neg]
                key = (tool, action)
                if key not in func_directions:
                    func_directions[key] = {}
                func_directions[key][li] = d
    
    log(f"  Extracted {len(func_directions)} function directions")
    
    # --- B2: Same-function cross-tool transfer ---
    log("  B2: Cross-tool transfer for same function...")
    
    transfer_results = []
    action_words = ["cut", "write", "drive", "measure", "protect", "illuminate",
                    "record", "paint", "strike", "observe", "sew", "dig", "navigate"]
    
    for tool1, tool2, shared_action in SAME_FUNCTION_GROUPS:
        src_key = (tool1, shared_action)
        if src_key not in func_directions or deepest_li not in func_directions[src_key]:
            # Try alternative action names
            continue
        
        d_src = func_directions[src_key][deepest_li]
        d_norm = np.linalg.norm(d_src)
        if d_norm < 1e-10:
            continue
        d_src_unit = d_src / d_norm
        
        # Inject into tool2's neutral prompt
        target_prompt = f"People use the {tool2} to"
        baseline_logits, baseline_top = get_logits_for_prompt(model, tokenizer, device, target_prompt, top_k=30)
        
        for alpha in [1.0, 2.0]:
            inj_logits, inj_top = inject_direction_and_get_logits(
                model, tokenizer, device, target_prompt,
                d_src_unit, deepest_li, alpha, top_k=30
            )
            
            # Check if shared_action appears in logits
            action_ids = tokenizer.encode(shared_action, add_special_tokens=False)
            delta = float(inj_logits[action_ids[0]] - baseline_logits[action_ids[0]]) if action_ids else 0
            
            # Check top5 overlap with action words
            inj_top5_words = set(t for t, _ in inj_top[:5])
            action_in_top5 = inj_top5_words & set(action_words)
            
            baseline_top5_words = set(t for t, _ in baseline_top[:5])
            inj_top5_words_str = [t for t, _ in inj_top[:5]]
            baseline_top5_words_str = [t for t, _ in baseline_top[:5]]
            
            transfer_results.append({
                "source": f"{tool1}→{shared_action}",
                "target_tool": tool2,
                "alpha": alpha,
                "delta_shared_action": round(delta, 3),
                "action_in_top5": list(action_in_top5),
                "baseline_top5": baseline_top5_words_str,
                "injected_top5": inj_top5_words_str,
            })
    
    # --- B3: Direct function injection (same tool) ---
    log("  B3: Direct function injection (same tool)...")
    direct_results = []
    
    for tool, action in FUNCTION_SOURCES[:10]:  # First 10 for speed
        src_key = (tool, action)
        if src_key not in func_directions or deepest_li not in func_directions[src_key]:
            continue
        
        d_src = func_directions[src_key][deepest_li]
        d_norm = np.linalg.norm(d_src)
        if d_norm < 1e-10:
            continue
        d_src_unit = d_src / d_norm
        
        target_prompt = f"People use the {tool} to"
        baseline_logits, _ = get_logits_for_prompt(model, tokenizer, device, target_prompt, top_k=30)
        
        for alpha in [1.0, 2.0]:
            inj_logits, inj_top = inject_direction_and_get_logits(
                model, tokenizer, device, target_prompt,
                d_src_unit, deepest_li, alpha, top_k=30
            )
            
            action_ids = tokenizer.encode(action, add_special_tokens=False)
            delta = float(inj_logits[action_ids[0]] - baseline_logits[action_ids[0]]) if action_ids else 0
            
            direct_results.append({
                "tool": tool,
                "action": action,
                "alpha": alpha,
                "delta_action": round(delta, 3),
            })
    
    results["function_transfer"] = transfer_results
    results["function_direct"] = direct_results
    
    # Summary
    log("\n  --- Function Causal Summary ---")
    if direct_results:
        for alpha in [1.0, 2.0]:
            alpha_res = [r for r in direct_results if r["alpha"] == alpha]
            if alpha_res:
                mean_delta = np.mean([r["delta_action"] for r in alpha_res])
                pos_frac = np.mean([1 if r["delta_action"] > 0 else 0 for r in alpha_res])
                log(f"  Direct injection alpha={alpha}: mean_delta={mean_delta:.3f}, frac_positive={pos_frac:.2f}")
    
    if transfer_results:
        for alpha in [1.0, 2.0]:
            alpha_res = [r for r in transfer_results if r["alpha"] == alpha]
            if alpha_res:
                mean_delta = np.mean([r["delta_shared_action"] for r in alpha_res])
                pos_frac = np.mean([1 if r["delta_shared_action"] > 0 else 0 for r in alpha_res])
                frac_action_top5 = np.mean([1 if r["action_in_top5"] else 0 for r in alpha_res])
                log(f"  Cross-tool transfer alpha={alpha}: mean_delta={mean_delta:.3f}, frac_positive={pos_frac:.2f}, frac_action_top5={frac_action_top5:.2f}")
    
    return results


# =====================================================================
# PART C: NEGATION SUB-TYPE DECOMPOSITION
# =====================================================================

def run_negation_decomposition(model, tokenizer, device, info, cache, target_layers):
    """
    Decompose negation into sub-types and test their subspace structure.
    """
    log("\n" + "="*60)
    log("Part C: Negation Sub-Type Decomposition")
    log("="*60)
    
    results = {}
    deepest_li = target_layers[-1]
    
    # Build negation sentences
    neg_data = {}  # neg_type -> [(pos_sent, neg_sent)]
    
    for adj in NEGATION_ADJECTIVES:
        for neg_type, neg_fn in NEGATION_TYPES.items():
            pair = neg_fn(adj)
            if pair is None:
                continue
            pos_sent, neg_sent = pair
            if neg_type not in neg_data:
                neg_data[neg_type] = []
            neg_data[neg_type].append((pos_sent, neg_sent, adj))
    
    log(f"  Negation types: {list(neg_data.keys())}")
    for nt, pairs in neg_data.items():
        log(f"    {nt}: {len(pairs)} pairs")
    
    # C1: Extract directions for each negation type
    log("  C1: Extracting negation type directions...")
    neg_directions = {}  # neg_type -> [direction_vectors]
    
    for neg_type, pairs in neg_data.items():
        dirs = []
        for pos_sent, neg_sent, adj in pairs:
            k_pos = (pos_sent, deepest_li)
            k_neg = (neg_sent, deepest_li)
            if k_pos in cache and k_neg in cache:
                d = cache[k_neg] - cache[k_pos]
                dirs.append(d)
        neg_directions[neg_type] = dirs
        log(f"    {neg_type}: {len(dirs)} directions at L{deepest_li}")
    
    # C2: Pairwise cosine within and between negation types
    log("  C2: Within-type and cross-type cosine similarity...")
    
    neg_types = sorted(neg_directions.keys())
    cross_type_cosines = {}
    
    for i, nt1 in enumerate(neg_types):
        for j, nt2 in enumerate(neg_types):
            if i > j:
                continue
            D1 = neg_directions[nt1]
            D2 = neg_directions[nt2]
            
            if not D1 or not D2:
                continue
            
            D1_arr = np.array(D1)
            D2_arr = np.array(D2)
            
            # Normalize
            D1_norms = np.linalg.norm(D1_arr, axis=1, keepdims=True)
            D2_norms = np.linalg.norm(D2_arr, axis=1, keepdims=True)
            D1_norms = np.maximum(D1_norms, 1e-10)
            D2_norms = np.maximum(D2_norms, 1e-10)
            D1_unit = D1_arr / D1_norms
            D2_unit = D2_arr / D2_norms
            
            if i == j:
                # Within-type pairwise cosine
                cos_mat = D1_unit @ D1_unit.T
                n = cos_mat.shape[0]
                mask = ~np.eye(n, dtype=bool)
                cos_vals = cos_mat[mask]
                cross_type_cosines[f"{nt1}_within"] = {
                    "mean": float(np.mean(cos_vals)),
                    "median": float(np.median(cos_vals)),
                    "std": float(np.std(cos_vals)),
                }
            else:
                # Cross-type: mean of all pairwise cosines
                cos_mat = D1_unit @ D2_unit.T
                cos_vals = cos_mat.flatten()
                cross_type_cosines[f"{nt1}_vs_{nt2}"] = {
                    "mean": float(np.mean(cos_vals)),
                    "median": float(np.median(cos_vals)),
                }
    
    # C3: Subspace analysis for each negation type
    log("  C3: Subspace analysis per negation type...")
    subspace_results = {}
    
    for nt, dirs in neg_directions.items():
        if len(dirs) < 3:
            continue
        D = np.array(dirs)
        D_norms = np.linalg.norm(D, axis=1, keepdims=True)
        D_norms = np.maximum(D_norms, 1e-10)
        D_unit = D / D_norms
        
        # PCA
        D_c = D_unit - D_unit.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(D_c, full_matrices=False)
        eig = (S ** 2) / (len(dirs) - 1)
        total = eig.sum()
        if total > 1e-20:
            ratio = eig / total
            cum = np.cumsum(ratio)
            dim_at_80 = int(np.searchsorted(cum, 0.80) + 1)
        else:
            dim_at_80 = -1
        
        # Pairwise cosine
        cos_mat = D_unit @ D_unit.T
        n = cos_mat.shape[0]
        mask = ~np.eye(n, dtype=bool)
        pair_cos = cos_mat[mask]
        
        subspace_results[nt] = {
            "n_pairs": len(dirs),
            "dim_at_80": dim_at_80,
            "pair_cos_mean": float(np.mean(pair_cos)),
            "pair_cos_median": float(np.median(pair_cos)),
            "norm_mean": float(np.mean(np.linalg.norm(D, axis=1))),
        }
        log(f"    {nt}: n={len(dirs)}, dim@80={dim_at_80}, pair_cos={np.mean(pair_cos):.3f}")
    
    # C4: Causal test — inject negation direction and check polarity shift
    log("  C4: Negation causal injection...")
    
    neg_causal_results = []
    test_adjs = NEGATION_ADJECTIVES[:5]  # First 5 for speed
    
    for neg_type in ["not", "never", "barely"]:
        dirs = neg_directions.get(neg_type, [])
        if not dirs:
            continue
        
        # Mean direction
        mean_dir = np.mean(dirs, axis=0)
        mean_norm = np.linalg.norm(mean_dir)
        if mean_norm < 1e-10:
            continue
        mean_dir_unit = mean_dir / mean_norm
        
        for adj in test_adjs:
            prompt = f"very {adj}"
            baseline_logits, baseline_top = get_logits_for_prompt(
                model, tokenizer, device, prompt, top_k=30
            )
            
            for alpha in [2.0, 4.0]:
                inj_logits, inj_top = inject_direction_and_get_logits(
                    model, tokenizer, device, prompt,
                    mean_dir_unit, deepest_li, alpha, top_k=30
                )
                
                # Check if negation words (not, never, barely, un-) appear more
                neg_words = ["not", "never", "barely", "no", "un"]
                neg_deltas = {}
                for nw in neg_words:
                    nw_ids = tokenizer.encode(nw, add_special_tokens=False)
                    if nw_ids:
                        neg_deltas[nw] = round(float(inj_logits[nw_ids[0]] - baseline_logits[nw_ids[0]]), 3)
                
                # Check if positive words decrease
                adj_ids = tokenizer.encode(adj, add_special_tokens=False)
                adj_delta = float(inj_logits[adj_ids[0]] - baseline_logits[adj_ids[0]]) if adj_ids else 0
                
                neg_causal_results.append({
                    "neg_type": neg_type,
                    "adjective": adj,
                    "alpha": alpha,
                    "negation_deltas": neg_deltas,
                    "adjective_delta": round(adj_delta, 3),
                })
    
    results["cross_type_cosines"] = cross_type_cosines
    results["subspace_results"] = subspace_results
    results["negation_causal"] = neg_causal_results
    
    # Summary
    log("\n  --- Negation Decomposition Summary ---")
    for key, val in cross_type_cosines.items():
        if "within" in key:
            log(f"  {key}: mean_cos={val['mean']:.3f}")
        else:
            log(f"  {key}: mean_cos={val['mean']:.3f}")
    
    if neg_causal_results:
        for neg_type in ["not", "never", "barely"]:
            type_res = [r for r in neg_causal_results if r["neg_type"] == neg_type]
            if type_res:
                mean_neg_delta = np.mean([max(r["negation_deltas"].values()) if r["negation_deltas"] else 0 for r in type_res])
                mean_adj_delta = np.mean([r["adjective_delta"] for r in type_res])
                log(f"  {neg_type} causal: mean_max_neg_delta={mean_neg_delta:.3f}, mean_adj_delta={mean_adj_delta:.3f}")
    
    return results


# =====================================================================
# COLLECT ALL UNIQUE SENTENCES
# =====================================================================

def collect_all_sentences():
    """Collect all unique sentences needed for the experiment."""
    sentences = set()
    
    # Part A: Attribute
    for noun, attr_type, value in ATTRIBUTE_SOURCES:
        sentences.add(f"the {noun} is {value}")
        sentences.add(f"the {noun} is just an object")
    
    for noun, attr_type, value in ATTRIBUTE_TARGETS:
        sentences.add(f"the {noun} is {value}")
        sentences.add(f"the {noun} is just an object")
    
    for noun, val1, val2 in SAME_ATTRIBUTE_PAIRS:
        sentences.add(f"the {noun} is {val1}")
        sentences.add(f"the {noun} is {val2}")
    
    # Part B: Function
    for tool, action in FUNCTION_SOURCES:
        sentences.add(f"people use the {tool} to {action}")
        sentences.add(f"people use the {tool}")
    
    for tool1, tool2, _ in SAME_FUNCTION_GROUPS:
        sentences.add(f"people use the {tool1}")
        sentences.add(f"people use the {tool2}")
    
    # Part C: Negation
    for adj in NEGATION_ADJECTIVES:
        for neg_type, neg_fn in NEGATION_TYPES.items():
            pair = neg_fn(adj)
            if pair is None:
                continue
            pos_sent, neg_sent = pair
            sentences.add(pos_sent)
            sentences.add(neg_sent)
    
    return sorted(sentences)


# =====================================================================
# MAIN
# =====================================================================

def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase320_{model_name}.log")
    
    log(f"=== Phase 320: Causal Verification for {model_name} ===")
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}, class={info.model_class}")
    t_load = time.time() - t0
    log(f"  Load time: {t_load:.1f}s")
    
    target_layers = get_target_layers(info.n_layers)
    log(f"  Target layers: {target_layers}")
    
    # Collect all sentences
    all_sentences = collect_all_sentences()
    log(f"  Total unique sentences: {len(all_sentences)}")
    
    # Extract representations
    t0 = time.time()
    log("Extracting representations...")
    cache = extract_last_token_rep(model, tokenizer, device, all_sentences, target_layers, label="All")
    t_extract = time.time() - t0
    log(f"  Extraction time: {t_extract:.1f}s")
    
    # Run all three parts
    all_results = {}
    
    all_results["attribute"] = run_attribute_causal(
        model, tokenizer, device, info, cache, target_layers
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    all_results["function"] = run_function_causal(
        model, tokenizer, device, info, cache, target_layers
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    all_results["negation"] = run_negation_decomposition(
        model, tokenizer, device, info, cache, target_layers
    )
    
    # Save
    output = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "target_layers": target_layers,
        "extraction_time_s": round(t_extract, 1),
        "results": all_results,
    }
    
    out_path = RESULT_DIR / f"{model_name}_phase320.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")
    
    # Print summary
    print_overall_summary(output)
    
    # Cleanup
    del cache
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released.")
    
    return output


def print_overall_summary(results):
    log("\n" + "=" * 70)
    log(f"PHASE 320 OVERALL SUMMARY - {results['model']}")
    log("=" * 70)
    
    # Attribute summary
    attr = results["results"].get("attribute", {})
    inj = attr.get("attribute_injection", [])
    axis = attr.get("axis_injection", [])
    transfer = attr.get("attribute_transfer", [])
    
    log("\n--- Attribute Causal Injection ---")
    if inj:
        for alpha in [0.5, 1.0, 2.0, 4.0]:
            ar = [r for r in inj if r["alpha"] == alpha]
            if ar:
                mean_d = np.mean([r["target_value_delta"] for r in ar])
                pos = np.mean([1 if r["target_value_delta"] > 0 else 0 for r in ar])
                log(f"  d_attr alpha={alpha}: mean_delta={mean_d:.3f}, positive={pos:.0%}")
    
    if axis:
        for alpha in [1.0, 2.0, 4.0]:
            ar = [r for r in axis if r["alpha"] == alpha]
            if ar:
                mean_tgt = np.mean([r["delta_target_value"] for r in ar])
                mean_src = np.mean([r["delta_source_value"] for r in ar])
                pos = np.mean([1 if r["delta_target_value"] > 0 else 0 for r in ar])
                log(f"  d_axis alpha={alpha}: target_delta={mean_tgt:.3f}, source_delta={mean_src:.3f}, positive={pos:.0%}")
    
    if transfer:
        mean_d = np.mean([r["delta_expected"] for r in transfer])
        pos = np.mean([1 if r["delta_expected"] > 0 else 0 for r in transfer])
        log(f"  Cross-object transfer: mean_delta={mean_d:.3f}, positive={pos:.0%}")
    
    # Function summary
    func = results["results"].get("function", {})
    direct = func.get("function_direct", [])
    ftransfer = func.get("function_transfer", [])
    
    log("\n--- Function Causal Injection ---")
    if direct:
        for alpha in [1.0, 2.0]:
            ar = [r for r in direct if r["alpha"] == alpha]
            if ar:
                mean_d = np.mean([r["delta_action"] for r in ar])
                pos = np.mean([1 if r["delta_action"] > 0 else 0 for r in ar])
                log(f"  Direct alpha={alpha}: mean_delta={mean_d:.3f}, positive={pos:.0%}")
    
    if ftransfer:
        for alpha in [1.0, 2.0]:
            ar = [r for r in ftransfer if r["alpha"] == alpha]
            if ar:
                mean_d = np.mean([r["delta_shared_action"] for r in ar])
                pos = np.mean([1 if r["delta_shared_action"] > 0 else 0 for r in ar])
                log(f"  Transfer alpha={alpha}: mean_delta={mean_d:.3f}, positive={pos:.0%}")
    
    # Negation summary
    neg = results["results"].get("negation", {})
    sub = neg.get("subspace_results", {})
    cosines = neg.get("cross_type_cosines", {})
    neg_causal = neg.get("negation_causal", [])
    
    log("\n--- Negation Decomposition ---")
    for nt, data in sub.items():
        log(f"  {nt}: dim@80={data['dim_at_80']}, pair_cos={data['pair_cos_mean']:.3f}, n={data['n_pairs']}")
    
    for key, val in cosines.items():
        if "vs" in key:
            log(f"  Cross-type {key}: mean_cos={val['mean']:.3f}")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for mn in ["qwen3", "glm4", "deepseek7b"]:
            log(f"\n{'#'*70}")
            log(f"# Starting {mn}")
            log(f"{'#'*70}")
            try:
                run_model(mn)
            except Exception as e:
                log(f"ERROR running {mn}: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(10)
    else:
        run_model(model_name)
