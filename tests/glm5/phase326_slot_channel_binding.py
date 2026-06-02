"""
Phase 326: Slot Verification + Sensory Channel + Binding Initial Test
=====================================================================
Three sub-tests in one script:
1. Slot output pattern: does slot push generic property words UP while pushing specific values DOWN?
2. Sensory channel: "looks red" vs "feels rough" vs "is hot" — is effect from attribute type or sensory verb?
3. Object-attribute binding: does apple+color direction prefer "red" over "blue"?

Usage:
  python tests/glm5/phase326_slot_channel_binding.py qwen3
  python tests/glm5/phase326_slot_channel_binding.py glm4
  python tests/glm5/phase326_slot_channel_binding.py deepseek7b
"""
import sys, os, time, json, torch, numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# --- Model Configs ---
MODEL_CONFIGS = {
    "qwen3": {"path": "", "n_layers": 36, "d_model": 2560, "opt_layer": 0},
    "glm4": {"path": "", "n_layers": 40, "d_model": 4096, "opt_layer": 3},
    "deepseek7b": {"path": "", "n_layers": 28, "d_model": 3584, "opt_layer": 6},
}
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from model_utils import MODEL_CONFIGS as _MU
    for k in MODEL_CONFIGS:
        if k in _MU:
            MODEL_CONFIGS[k]["path"] = _MU[k]["path"]
except ImportError:
    pass

# ============================================================
# TEST 1: Slot Output Pattern
# Does slot direction push generic property words up?
# ============================================================
# Objects for testing
SLOT_TEST_OBJECTS = [
    "apple", "banana", "snow", "knife", "stone", "silk",
    "tea", "lemon", "ball", "ice",
]

# Three output categories
SPECIFIC_VALUE_WORDS = ["red", "blue", "green", "yellow", "sweet", "sour", "hot", "cold", "rough", "smooth", "round", "flat"]
TYPE_WORDS = ["color", "taste", "temperature", "texture", "shape", "size"]
GENERIC_PROPERTY_WORDS = ["property", "feature", "quality", "characteristic", "attribute", "trait", "aspect", "nature"]

# Slot templates
SLOT_TEMPLATES = [
    "{obj} has some feature",
    "{obj} has a property",
    "{obj} has some quality",
]
BASE_TEMPLATE = "{obj} is an object"

# ============================================================
# TEST 2: Sensory Channel Comparison
# Is the effect from attribute type or sensory verb?
# ============================================================
# For color: visual "looks red" vs state "is red"
# For texture: tactile "feels rough" vs state "is rough"
# For temperature: tactile "feels hot" vs state "is hot"

CHANNEL_PAIRS = {
    "color": {
        "objects": ["apple", "sky", "grass", "snow", "banana", "cherry", "ocean", "leaf"],
        "values": ["red", "blue", "green", "white", "yellow", "red", "blue", "green"],
        "visual": "{obj} looks {val}",   # sensory channel
        "state": "{obj} is {val}",        # bare state
        "type": "{obj} has a color",
    },
    "texture": {
        "objects": ["stone", "silk", "sandpaper", "velvet", "glass", "bark", "cotton", "concrete"],
        "values": ["rough", "smooth", "coarse", "soft", "smooth", "rough", "soft", "rough"],
        "tactile": "{obj} feels {val}",   # sensory channel
        "state": "{obj} is {val}",        # bare state
        "type": "{obj} has a surface feel",
    },
    "temperature": {
        "objects": ["tea", "ice", "soup", "snow", "stove", "freezer", "coffee", "oven"],
        "values": ["hot", "cold", "hot", "cold", "hot", "cold", "hot", "cold"],
        "tactile": "{obj} feels {val}",       # sensory channel
        "contact": "{obj} is {val} to touch",  # contact channel
        "state": "{obj} is {val}",             # bare state
        "type": "{obj} has a temperature quality",
    },
}

# ============================================================
# TEST 3: Object-Attribute Binding
# Does color direction prefer compatible values?
# ============================================================
BINDING_PAIRS = [
    # (object, compatible_value, incompatible_value, attribute_type)
    ("apple", "red", "blue", "color"),
    ("banana", "yellow", "blue", "color"),
    ("snow", "white", "black", "color"),
    ("sky", "blue", "green", "color"),
    ("grass", "green", "red", "color"),
    ("knife", "sharp", "soft", "texture"),
    ("stone", "rough", "sweet", "texture"),
    ("silk", "smooth", "rough", "texture"),
    ("tea", "hot", "cold", "temperature"),
    ("ice", "cold", "hot", "temperature"),
]


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=impl,
            )
            log(f"  Loaded {model_name} with attn_impl={impl}")
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Model: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def get_layers(model):
    if hasattr(model, 'model'):
        inner = model.model
    else:
        inner = model
    if hasattr(inner, 'layers'):
        return list(inner.layers)
    elif hasattr(inner, 'encoder') and hasattr(inner.encoder, 'layer'):
        return list(inner.encoder.layer)
    return []


def extract_rep(model, tokenizer, device, sentence, target_layer):
    layers_list = get_layers(model)
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['rep'] = output[0].detach().float().cpu()
        else:
            captured['rep'] = output.detach().float().cpu()
    hook = layers_list[target_layer].register_forward_hook(hook_fn)
    inp = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
    try:
        with torch.no_grad():
            model(**inp)
        return captured['rep'][0, -1].numpy()
    finally:
        hook.remove()


def inject_and_get_logits(model, tokenizer, device, prompt, direction, layer_idx, alpha):
    layers_list = get_layers(model)
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
        hidden_modified = hidden.clone()
        hidden_modified[0, -1, :] += (alpha * d_tensor).to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        return hidden_modified
    hook = layers_list[layer_idx].register_forward_hook(hook_fn)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    try:
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1].float().cpu().numpy()
    finally:
        hook.remove()
    return logits


def get_baseline_logits(model, tokenizer, device, prompt):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1].float().cpu().numpy()


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


def get_logits_for_words(logits, tokenizer, words):
    """Get logit values for a list of words."""
    result = {}
    for w in words:
        tid = get_token_id(tokenizer, w)
        if tid is not None:
            result[w] = float(logits[tid])
    return result


# ============================================================
# TEST 1: Slot Output Pattern
# ============================================================
def test_slot_output(model, tokenizer, device, opt_layer, alpha):
    log("\n" + "="*60)
    log("TEST 1: Slot Output Pattern")
    log("="*60)
    
    all_words = SPECIFIC_VALUE_WORDS + TYPE_WORDS + GENERIC_PROPERTY_WORDS
    results = {"per_object": [], "aggregated": {}}
    
    cat_deltas = {"specific": [], "type": [], "generic": []}
    
    for obj in SLOT_TEST_OBJECTS:
        base = BASE_TEMPLATE.format(obj=obj)
        baseline_logits = get_baseline_logits(model, tokenizer, device, base)
        
        # Average slot direction across templates
        slot_dirs = []
        for tpl in SLOT_TEMPLATES:
            test = tpl.format(obj=obj)
            h_base = extract_rep(model, tokenizer, device, base, opt_layer)
            h_test = extract_rep(model, tokenizer, device, test, opt_layer)
            direction = h_test - h_base
            dir_norm = direction / (np.linalg.norm(direction) + 1e-8)
            slot_dirs.append(dir_norm)
        
        avg_dir = np.mean(slot_dirs, axis=0)
        avg_dir = avg_dir / (np.linalg.norm(avg_dir) + 1e-8)
        
        # Inject slot direction
        patched_logits = inject_and_get_logits(model, tokenizer, device, base, avg_dir, opt_layer, alpha)
        
        # Compute deltas for each category
        obj_result = {"object": obj}
        for cat_name, cat_words in [("specific", SPECIFIC_VALUE_WORDS), 
                                     ("type", TYPE_WORDS), 
                                     ("generic", GENERIC_PROPERTY_WORDS)]:
            deltas = []
            for w in cat_words:
                tid = get_token_id(tokenizer, w)
                if tid is not None:
                    d = float(patched_logits[tid] - baseline_logits[tid])
                    deltas.append(d)
            mean_d = float(np.mean(deltas)) if deltas else 0.0
            cat_deltas[cat_name].append(mean_d)
            obj_result[cat_name] = round(mean_d, 4)
        
        results["per_object"].append(obj_result)
        log(f"  {obj}: specific={obj_result['specific']:.4f}, type={obj_result['type']:.4f}, generic={obj_result['generic']:.4f}")
    
    # Aggregate
    for cat in ["specific", "type", "generic"]:
        vals = cat_deltas[cat]
        results["aggregated"][cat] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "positive_rate": round(sum(1 for v in vals if v > 0) / max(len(vals), 1), 4),
        }
    
    log(f"\n  AGGREGATED slot direction effect:")
    log(f"    Specific values: mean={results['aggregated']['specific']['mean']:.4f}, "
        f"pos_rate={results['aggregated']['specific']['positive_rate']:.2f}")
    log(f"    Type words:      mean={results['aggregated']['type']['mean']:.4f}, "
        f"pos_rate={results['aggregated']['type']['positive_rate']:.2f}")
    log(f"    Generic property: mean={results['aggregated']['generic']['mean']:.4f}, "
        f"pos_rate={results['aggregated']['generic']['positive_rate']:.2f}")
    
    return results


# ============================================================
# TEST 2: Sensory Channel Comparison
# ============================================================
def test_sensory_channel(model, tokenizer, device, opt_layer, alpha):
    log("\n" + "="*60)
    log("TEST 2: Sensory Channel Comparison")
    log("="*60)
    
    results = {}
    
    for attr_type, cfg in CHANNEL_PAIRS.items():
        log(f"\n  --- {attr_type} ---")
        objects = cfg["objects"]
        values = cfg["values"]
        
        # Determine channels to test
        channels = {}
        for key in cfg:
            if key not in ["objects", "values"] and key != "type":
                channels[key] = cfg[key]
        # Always include type
        channels["type"] = cfg.get("type", f"{{obj}} has a {attr_type}")
        
        # Get cluster words
        if attr_type == "color":
            cluster = ["red", "blue", "green", "white", "black", "yellow", "orange", "purple", "pink", "brown"]
        elif attr_type == "texture":
            cluster = ["rough", "smooth", "coarse", "soft", "hard", "bumpy", "silky", "grainy"]
        elif attr_type == "temperature":
            cluster = ["hot", "cold", "warm", "cool", "freezing", "boiling", "icy", "scalding"]
        else:
            cluster = []
        
        channel_stats = {}
        
        for ch_name, ch_template in channels.items():
            tgt_deltas = []
            clst_deltas = []
            
            for i, (obj, val) in enumerate(zip(objects, values)):
                base = BASE_TEMPLATE.format(obj=obj)
                test = ch_template.format(obj=obj, val=val)
                
                # Extract direction
                h_base = extract_rep(model, tokenizer, device, base, opt_layer)
                h_test = extract_rep(model, tokenizer, device, test, opt_layer)
                direction = h_test - h_base
                dir_norm = direction / (np.linalg.norm(direction) + 1e-8)
                
                # Inject
                baseline_logits = get_baseline_logits(model, tokenizer, device, base)
                patched_logits = inject_and_get_logits(model, tokenizer, device, base, dir_norm, opt_layer, alpha)
                
                # Target delta
                tid = get_token_id(tokenizer, val)
                if tid is not None:
                    tgt_deltas.append(float(patched_logits[tid] - baseline_logits[tid]))
                
                # Cluster delta
                clst_d = []
                for w in cluster:
                    wid = get_token_id(tokenizer, w)
                    if wid is not None:
                        clst_d.append(float(patched_logits[wid] - baseline_logits[wid]))
                if clst_d:
                    clst_deltas.append(float(np.mean(clst_d)))
            
            channel_stats[ch_name] = {
                "tgt_mean": round(float(np.mean(tgt_deltas)), 4) if tgt_deltas else 0,
                "cluster_mean": round(float(np.mean(clst_deltas)), 4) if clst_deltas else 0,
                "n": len(tgt_deltas),
            }
            log(f"    {ch_name}: tgt={channel_stats[ch_name]['tgt_mean']:.4f}, "
                f"cluster={channel_stats[ch_name]['cluster_mean']:.4f}")
        
        # Compare channels
        results[attr_type] = channel_stats
    
    return results


# ============================================================
# TEST 3: Object-Attribute Binding
# ============================================================
def test_binding(model, tokenizer, device, opt_layer, alpha):
    log("\n" + "="*60)
    log("TEST 3: Object-Attribute Binding")
    log("="*60)
    
    results = {"per_pair": [], "aggregated": {}}
    
    # We test 3 direction types: type direction, value direction (compatible), value direction (incompatible)
    binding_scores = {"type": [], "compat_value": [], "incompat_value": []}
    
    for obj, compat_val, incompat_val, attr_type in BINDING_PAIRS:
        base = BASE_TEMPLATE.format(obj=obj)
        baseline_logits = get_baseline_logits(model, tokenizer, device, base)
        
        # Get token IDs
        compat_tid = get_token_id(tokenizer, compat_val)
        incompat_tid = get_token_id(tokenizer, incompat_val)
        
        pair_result = {"object": obj, "compat": compat_val, "incompat": incompat_val, "type": attr_type}
        
        # Test 1: type direction
        if attr_type == "color":
            type_test = f"{obj} has a color"
        elif attr_type == "texture":
            type_test = f"{obj} has a surface feel"
        elif attr_type == "temperature":
            type_test = f"{obj} has a temperature quality"
        else:
            type_test = f"{obj} has a {attr_type}"
        
        h_base = extract_rep(model, tokenizer, device, base, opt_layer)
        h_type = extract_rep(model, tokenizer, device, type_test, opt_layer)
        type_dir = h_type - h_base
        type_dir_norm = type_dir / (np.linalg.norm(type_dir) + 1e-8)
        
        patched_logits = inject_and_get_logits(model, tokenizer, device, base, type_dir_norm, opt_layer, alpha)
        
        compat_delta_type = float(patched_logits[compat_tid] - baseline_logits[compat_tid]) if compat_tid else 0
        incompat_delta_type = float(patched_logits[incompat_tid] - baseline_logits[incompat_tid]) if incompat_tid else 0
        binding_type = compat_delta_type - incompat_delta_type
        
        binding_scores["type"].append(binding_type)
        pair_result["type_dir"] = {
            "compat_delta": round(compat_delta_type, 4),
            "incompat_delta": round(incompat_delta_type, 4),
            "binding": round(binding_type, 4),
        }
        
        # Test 2: compatible value direction
        if attr_type == "color":
            compat_test = f"{obj} looks {compat_val}"
        elif attr_type == "texture":
            compat_test = f"{obj} feels {compat_val}"
        elif attr_type == "temperature":
            compat_test = f"{obj} is {compat_val} to touch"
        else:
            compat_test = f"{obj} is {compat_val}"
        
        h_compat = extract_rep(model, tokenizer, device, compat_test, opt_layer)
        compat_dir = h_compat - h_base
        compat_dir_norm = compat_dir / (np.linalg.norm(compat_dir) + 1e-8)
        
        patched_logits = inject_and_get_logits(model, tokenizer, device, base, compat_dir_norm, opt_layer, alpha)
        
        compat_delta_cv = float(patched_logits[compat_tid] - baseline_logits[compat_tid]) if compat_tid else 0
        incompat_delta_cv = float(patched_logits[incompat_tid] - baseline_logits[incompat_tid]) if incompat_tid else 0
        binding_cv = compat_delta_cv - incompat_delta_cv
        
        binding_scores["compat_value"].append(binding_cv)
        pair_result["compat_value_dir"] = {
            "compat_delta": round(compat_delta_cv, 4),
            "incompat_delta": round(incompat_delta_cv, 4),
            "binding": round(binding_cv, 4),
        }
        
        # Test 3: incompatible value direction
        if attr_type == "color":
            incompat_test = f"{obj} looks {incompat_val}"
        elif attr_type == "texture":
            incompat_test = f"{obj} feels {incompat_val}"
        elif attr_type == "temperature":
            incompat_test = f"{obj} is {incompat_val} to touch"
        else:
            incompat_test = f"{obj} is {incompat_val}"
        
        h_incompat = extract_rep(model, tokenizer, device, incompat_test, opt_layer)
        incompat_dir = h_incompat - h_base
        incompat_dir_norm = incompat_dir / (np.linalg.norm(incompat_dir) + 1e-8)
        
        patched_logits = inject_and_get_logits(model, tokenizer, device, base, incompat_dir_norm, opt_layer, alpha)
        
        compat_delta_iv = float(patched_logits[compat_tid] - baseline_logits[compat_tid]) if compat_tid else 0
        incompat_delta_iv = float(patched_logits[incompat_tid] - baseline_logits[incompat_tid]) if incompat_tid else 0
        binding_iv = compat_delta_iv - incompat_delta_iv
        
        binding_scores["incompat_value"].append(binding_iv)
        pair_result["incompat_value_dir"] = {
            "compat_delta": round(compat_delta_iv, 4),
            "incompat_delta": round(incompat_delta_iv, 4),
            "binding": round(binding_iv, 4),
        }
        
        results["per_pair"].append(pair_result)
        log(f"  {obj}-{compat_val}/{incompat_val}: "
            f"type_binding={binding_type:.4f}, compat_binding={binding_cv:.4f}, incompat_binding={binding_iv:.4f}")
    
    # Aggregate
    for direction in ["type", "compat_value", "incompat_value"]:
        vals = binding_scores[direction]
        results["aggregated"][direction] = {
            "mean_binding": round(float(np.mean(vals)), 4),
            "positive_rate": round(sum(1 for v in vals if v > 0) / max(len(vals), 1), 4),
        }
    
    log(f"\n  AGGREGATED binding scores:")
    for d in ["type", "compat_value", "incompat_value"]:
        log(f"    {d}: mean_binding={results['aggregated'][d]['mean_binding']:.4f}, "
            f"pos_rate={results['aggregated'][d]['positive_rate']:.2f}")
    
    return results


# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python phase326_slot_channel_binding.py <model_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    t0 = time.time()
    log(f"=== Phase 326: Slot+Channel+Binding for {model_name} ===")
    
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    opt_layer = cfg["opt_layer"]
    alpha = 2.0
    
    log(f"  n_layers={cfg['n_layers']}, d_model={cfg['d_model']}, opt_layer={opt_layer}, alpha={alpha}")
    
    # Test 1: Slot output pattern
    slot_results = test_slot_output(model, tokenizer, device, opt_layer, alpha)
    
    # Test 2: Sensory channel comparison
    channel_results = test_sensory_channel(model, tokenizer, device, opt_layer, alpha)
    
    # Test 3: Object-attribute binding
    binding_results = test_binding(model, tokenizer, device, opt_layer, alpha)
    
    # Save
    out = {
        "model": model_name,
        "n_layers": cfg["n_layers"],
        "d_model": cfg["d_model"],
        "opt_layer": opt_layer,
        "alpha": alpha,
        "test1_slot_output": slot_results,
        "test2_sensory_channel": channel_results,
        "test3_binding": binding_results,
    }
    
    os.makedirs("results/phase326_slot_channel_binding", exist_ok=True)
    out_path = f"results/phase326_slot_channel_binding/{model_name}_phase326.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"\n  Saved to {out_path}")
    
    # Release
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"Done. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
