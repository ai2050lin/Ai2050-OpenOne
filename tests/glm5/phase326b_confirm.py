"""
Phase 326b: Confirmation of Phase 326 Key Findings
1. GLM4 slot pushes generic property words (+0.61) while suppressing specific values (-0.81) — confirm with 20 objects
2. GLM4 type direction has weak binding (+0.64) — confirm with 20 binding pairs
3. GLM4 temperature "to touch" effect (3.54) — confirm with more pairs
"""
import sys, os, time, json, torch, numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

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

# --- Test 1: Slot output with 20 objects ---
SLOT_OBJECTS = [
    "apple", "banana", "snow", "knife", "stone", "silk", "tea", "lemon",
    "ball", "ice", "rose", "cotton", "steel", "wood", "glass", "sand",
    "fur", "clay", "honey", "salt",
]

SPECIFIC_VALUES = ["red", "blue", "green", "yellow", "sweet", "sour", "hot", "cold", "rough", "smooth", "round", "flat", "sharp", "soft", "hard"]
TYPE_WORDS = ["color", "taste", "temperature", "texture", "shape", "size"]
GENERIC_PROPERTY = ["property", "feature", "quality", "characteristic", "attribute", "trait", "aspect", "nature"]

SLOT_TEMPLATES = ["{obj} has some feature", "{obj} has a property", "{obj} has some quality"]
BASE = "{obj} is an object"

# --- Test 2: Binding with 20 pairs ---
BINDING_PAIRS = [
    ("apple", "red", "blue", "color"),
    ("banana", "yellow", "blue", "color"),
    ("snow", "white", "black", "color"),
    ("sky", "blue", "green", "color"),
    ("grass", "green", "red", "color"),
    ("cherry", "red", "white", "color"),
    ("orange", "orange", "purple", "color"),
    ("leaf", "green", "blue", "color"),
    ("coal", "black", "white", "color"),
    ("sun", "yellow", "black", "color"),
    ("knife", "sharp", "soft", "texture"),
    ("stone", "rough", "sweet", "texture"),
    ("silk", "smooth", "rough", "texture"),
    ("sandpaper", "coarse", "smooth", "texture"),
    ("velvet", "soft", "hard", "texture"),
    ("tea", "hot", "cold", "temperature"),
    ("ice", "cold", "hot", "temperature"),
    ("stove", "hot", "cold", "temperature"),
    ("snow", "cold", "hot", "temperature"),
    ("oven", "hot", "cold", "temperature"),
]

# --- Test 3: Temperature "to touch" vs other channels ---
TEMP_OBJECTS = [
    ("tea", "hot"), ("ice", "cold"), ("soup", "hot"), ("snow", "cold"),
    ("stove", "hot"), ("freezer", "cold"), ("coffee", "hot"), ("fridge", "cold"),
    ("oven", "hot"), ("glacier", "cold"), ("magma", "hot"), ("arctic", "cold"),
    ("fireplace", "hot"), ("winter", "cold"), ("desert", "hot"), ("tundra", "cold"),
    ("volcano", "hot"), ("iceberg", "cold"), ("sauna", "hot"), ("frost", "cold"),
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
        captured['rep'] = (output[0] if isinstance(output, tuple) else output).detach().float().cpu()
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
        return (hidden_modified,) + output[1:] if isinstance(output, tuple) else hidden_modified
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase326b_confirm.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    t0 = time.time()
    log(f"=== Phase 326b: Confirmation for {model_name} ===")

    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    opt_layer = cfg["opt_layer"]
    alpha = 2.0

    log(f"  n_layers={cfg['n_layers']}, d_model={cfg['d_model']}, opt_layer={opt_layer}, alpha={alpha}")

    results = {}

    # ============================================================
    # TEST 1: Slot output with 20 objects
    # ============================================================
    log("\n=== TEST 1: Slot Output Pattern (20 objects) ===")
    cat_deltas = {"specific": [], "type": [], "generic": []}

    for i, obj in enumerate(SLOT_OBJECTS):
        if (i+1) % 5 == 0:
            log(f"  slot object {i+1}/20 done")

        base = BASE.format(obj=obj)
        baseline_logits = get_baseline_logits(model, tokenizer, device, base)

        slot_dirs = []
        for tpl in SLOT_TEMPLATES:
            test = tpl.format(obj=obj)
            h_b = extract_rep(model, tokenizer, device, base, opt_layer)
            h_t = extract_rep(model, tokenizer, device, test, opt_layer)
            d = h_t - h_b
            d = d / (np.linalg.norm(d) + 1e-8)
            slot_dirs.append(d)
        avg_dir = np.mean(slot_dirs, axis=0)
        avg_dir = avg_dir / (np.linalg.norm(avg_dir) + 1e-8)

        patched_logits = inject_and_get_logits(model, tokenizer, device, base, avg_dir, opt_layer, alpha)

        for cat_name, cat_words in [("specific", SPECIFIC_VALUES), ("type", TYPE_WORDS), ("generic", GENERIC_PROPERTY)]:
            deltas = []
            for w in cat_words:
                tid = get_token_id(tokenizer, w)
                if tid is not None:
                    deltas.append(float(patched_logits[tid] - baseline_logits[tid]))
            cat_deltas[cat_name].append(float(np.mean(deltas)) if deltas else 0.0)

    slot_agg = {}
    for cat in ["specific", "type", "generic"]:
        vals = cat_deltas[cat]
        slot_agg[cat] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "positive_rate": round(sum(1 for v in vals if v > 0) / max(len(vals), 1), 4),
        }

    log(f"  Slot effect (20 objects):")
    for cat in ["specific", "type", "generic"]:
        log(f"    {cat}: mean={slot_agg[cat]['mean']:.4f}, pos_rate={slot_agg[cat]['positive_rate']:.2f}")

    results["slot_output"] = slot_agg

    # ============================================================
    # TEST 2: Binding with 20 pairs
    # ============================================================
    log("\n=== TEST 2: Binding (20 pairs) ===")
    binding_scores = {"type": [], "compat_value": []}

    for i, (obj, compat, incompat, attr_type) in enumerate(BINDING_PAIRS):
        if (i+1) % 5 == 0:
            log(f"  binding pair {i+1}/20 done")

        base = BASE.format(obj=obj)
        baseline_logits = get_baseline_logits(model, tokenizer, device, base)

        compat_tid = get_token_id(tokenizer, compat)
        incompat_tid = get_token_id(tokenizer, incompat)

        # Type direction
        if attr_type == "color":
            type_test = f"{obj} has a color"
        elif attr_type == "texture":
            type_test = f"{obj} has a surface feel"
        else:
            type_test = f"{obj} has a temperature quality"

        h_b = extract_rep(model, tokenizer, device, base, opt_layer)
        h_t = extract_rep(model, tokenizer, device, type_test, opt_layer)
        type_dir = (h_t - h_b)
        type_dir = type_dir / (np.linalg.norm(type_dir) + 1e-8)

        patched_logits = inject_and_get_logits(model, tokenizer, device, base, type_dir, opt_layer, alpha)
        compat_d = float(patched_logits[compat_tid] - baseline_logits[compat_tid]) if compat_tid else 0
        incompat_d = float(patched_logits[incompat_tid] - baseline_logits[incompat_tid]) if incompat_tid else 0
        binding_scores["type"].append(compat_d - incompat_d)

        # Compatible value direction
        if attr_type == "color":
            compat_test = f"{obj} looks {compat}"
        elif attr_type == "texture":
            compat_test = f"{obj} feels {compat}"
        else:
            compat_test = f"{obj} is {compat} to touch"

        h_c = extract_rep(model, tokenizer, device, compat_test, opt_layer)
        compat_dir = (h_c - h_b)
        compat_dir = compat_dir / (np.linalg.norm(compat_dir) + 1e-8)

        patched_logits = inject_and_get_logits(model, tokenizer, device, base, compat_dir, opt_layer, alpha)
        compat_d2 = float(patched_logits[compat_tid] - baseline_logits[compat_tid]) if compat_tid else 0
        incompat_d2 = float(patched_logits[incompat_tid] - baseline_logits[incompat_tid]) if incompat_tid else 0
        binding_scores["compat_value"].append(compat_d2 - incompat_d2)

    binding_agg = {}
    for direction in ["type", "compat_value"]:
        vals = binding_scores[direction]
        binding_agg[direction] = {
            "mean_binding": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "positive_rate": round(sum(1 for v in vals if v > 0) / max(len(vals), 1), 4),
        }

    log(f"  Binding (20 pairs):")
    for d in ["type", "compat_value"]:
        log(f"    {d}: mean_binding={binding_agg[d]['mean_binding']:.4f}, "
            f"pos_rate={binding_agg[d]['positive_rate']:.2f}")

    results["binding"] = binding_agg

    # ============================================================
    # TEST 3: Temperature "to touch" with 20 pairs
    # ============================================================
    log("\n=== TEST 3: Temperature Channel (20 pairs) ===")
    channel_stats = {"contact": [], "state": [], "type": []}

    for i, (obj, val) in enumerate(TEMP_OBJECTS):
        if (i+1) % 5 == 0:
            log(f"  temperature pair {i+1}/20 done")

        base = BASE.format(obj=obj)
        baseline_logits = get_baseline_logits(model, tokenizer, device, base)
        tgt_tid = get_token_id(tokenizer, val)

        channels = {
            "contact": f"{obj} is {val} to touch",
            "state": f"{obj} is {val}",
            "type": f"{obj} has a temperature quality",
        }

        for ch_name, ch_test in channels.items():
            h_b = extract_rep(model, tokenizer, device, base, opt_layer)
            h_t = extract_rep(model, tokenizer, device, ch_test, opt_layer)
            d = (h_t - h_b)
            d = d / (np.linalg.norm(d) + 1e-8)

            patched_logits = inject_and_get_logits(model, tokenizer, device, base, d, opt_layer, alpha)
            if tgt_tid is not None:
                tgt_delta = float(patched_logits[tgt_tid] - baseline_logits[tgt_tid])
                channel_stats[ch_name].append(tgt_delta)

    temp_agg = {}
    for ch in ["contact", "state", "type"]:
        vals = channel_stats[ch]
        temp_agg[ch] = {
            "tgt_mean": round(float(np.mean(vals)), 4) if vals else 0,
            "std": round(float(np.std(vals)), 4) if vals else 0,
            "n": len(vals),
        }

    log(f"  Temperature channels (20 pairs):")
    for ch in ["contact", "state", "type"]:
        log(f"    {ch}: tgt_mean={temp_agg[ch]['tgt_mean']:.4f} (n={temp_agg[ch]['n']})")

    results["temperature_channel"] = temp_agg

    # Save
    out = {
        "model": model_name,
        "n_layers": cfg["n_layers"],
        "d_model": cfg["d_model"],
        "opt_layer": opt_layer,
        "alpha": alpha,
        "results": results,
    }

    os.makedirs("results/phase326b_confirm", exist_ok=True)
    out_path = f"results/phase326b_confirm/{model_name}_phase326b.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"\n  Saved to {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"Done. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
