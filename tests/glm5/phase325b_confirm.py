"""
Phase 325b: Confirmation of Key Phase 325 Findings (20 pairs per type)
1. GLM4 texture: type 0.84→2.51 with specialized template
2. DS7B color: value -0.09→+0.50 with specialized template
3. GLM4 temperature: slot -1.06→+0.57 with specialized template
4. GLM4 shape: type 0.21→0.81 with specialized template
"""
import sys, os, time, json, torch, numpy as np
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

MODEL_CONFIGS = {
    "qwen3": {"path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c", "n_layers": 36, "d_model": 2560, "opt_layer": 0},
    "glm4": {"path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf", "n_layers": 40, "d_model": 4096, "opt_layer": 3},
    "deepseek7b": {"path": "D:/develop/model/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/3c056b0c44b4c8b0c9c5b0e3c5a5d5b5c5e5f5g5", "n_layers": 28, "d_model": 3584, "opt_layer": 6},
}
# Try to get real path from model_utils
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from model_utils import MODEL_CONFIGS as _MU_CONFIGS
    for k in MODEL_CONFIGS:
        if k in _MU_CONFIGS:
            MODEL_CONFIGS[k]["path"] = _MU_CONFIGS[k]["path"]
except ImportError:
    pass

# Extended pairs (20 per type)
EXTENDED_PAIRS = {
    "texture": [
        ("stone", "rough"), ("silk", "smooth"), ("sandpaper", "coarse"),
        ("velvet", "soft"), ("glass", "smooth"), ("bark", "rough"),
        ("cotton", "soft"), ("concrete", "rough"), ("marble", "smooth"),
        ("leather", "smooth"), ("metal", "smooth"), ("wood", "rough"),
        ("rubber", "soft"), ("ceramic", "smooth"), ("linen", "rough"),
        ("satin", "smooth"), ("chalk", "rough"), ("ice", "smooth"),
        ("carpet", "soft"), ("brick", "rough"),
    ],
    "temperature": [
        ("tea", "hot"), ("ice", "cold"), ("soup", "hot"),
        ("snow", "cold"), ("stove", "hot"), ("freezer", "cold"),
        ("coffee", "hot"), ("fridge", "cold"), ("oven", "hot"),
        ("glacier", "cold"), ("magma", "hot"), ("arctic", "cold"),
        ("fireplace", "hot"), ("winter", "cold"), ("desert", "hot"),
        ("tundra", "cold"), ("volcano", "hot"), ("iceberg", "cold"),
        ("sauna", "hot"), ("frost", "cold"),
    ],
    "shape": [
        ("ball", "round"), ("box", "square"), ("pyramid", "triangular"),
        ("cylinder", "cylindrical"), ("plate", "flat"), ("sphere", "spherical"),
        ("cube", "cubic"), ("cone", "conical"), ("ring", "circular"),
        ("diamond", "diamond-shaped"), ("egg", "oval"), ("tube", "tubular"),
        ("wheel", "round"), ("brick", "rectangular"), ("crystal", "angular"),
        ("coin", "circular"), ("planet", "spherical"), ("prism", "triangular"),
        ("arch", "arched"), ("spiral", "spiral"),
    ],
    "color": [
        ("apple", "red"), ("sky", "blue"), ("grass", "green"),
        ("snow", "white"), ("night", "black"), ("banana", "yellow"),
        ("orange", "orange"), ("grape", "purple"), ("cherry", "red"),
        ("ocean", "blue"), ("leaf", "green"), ("cloud", "white"),
        ("coal", "black"), ("sun", "yellow"), ("carrot", "orange"),
        ("lavender", "purple"), ("strawberry", "red"), ("sapphire", "blue"),
        ("emerald", "green"), ("ivory", "white"),
    ],
}

# Specialized templates only (confirmed better in Phase 325)
SPEC_TEMPLATES = {
    "texture": {
        "slot": "{obj} has a surface quality",
        "type": "{obj} has a surface feel",
        "value": "{obj} feels {val}",
    },
    "temperature": {
        "slot": "{obj} has a thermal state",
        "type": "{obj} has a temperature quality",
        "value": "{obj} is {val} to touch",
    },
    "shape": {
        "slot": "{obj} has a geometric form",
        "type": "{obj} has a geometric shape",
        "value": "{obj} has a {val} shape",
    },
    "color": {
        "slot": "{obj} has some visual feature",
        "type": "{obj} has a color",
        "value": "{obj} looks {val}",
    },
}

BASE = "{obj} is an object"

CLUSTERS = {
    "texture": ["rough", "smooth", "coarse", "soft", "hard", "bumpy", "silky", "grainy", "fuzzy", "polished"],
    "temperature": ["hot", "cold", "warm", "cool", "freezing", "boiling", "lukewarm", "icy", "scalding", "chilly"],
    "shape": ["round", "square", "triangular", "flat", "spherical", "cubic", "conical", "circular", "oval", "rectangular"],
    "color": ["red", "blue", "green", "white", "black", "yellow", "orange", "purple", "pink", "brown"],
}

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
    layers = []
    if hasattr(model, 'model'):
        inner = model.model
    else:
        inner = model
    if hasattr(inner, 'layers'):
        layers = list(inner.layers)
    elif hasattr(inner, 'encoder') and hasattr(inner.encoder, 'layer'):
        layers = list(inner.encoder.layer)
    return layers


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


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase325b_confirm.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    t0 = time.time()
    log(f"=== Phase 325b: Confirmation for {model_name} ===")

    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    opt_layer = cfg["opt_layer"]
    alpha = 2.0

    log(f"  n_layers={cfg['n_layers']}, d_model={cfg['d_model']}, opt_layer={opt_layer}, alpha={alpha}")

    results = {}

    for attr_type in ["texture", "temperature", "shape", "color"]:
        log(f"\n  --- {attr_type} (20 pairs) ---")
        pairs = EXTENDED_PAIRS[attr_type]
        templates = SPEC_TEMPLATES[attr_type]
        cluster_words = CLUSTERS[attr_type]

        # Get cluster token IDs
        cluster_ids = [get_token_id(tokenizer, w) for w in cluster_words]
        cluster_ids = [i for i in cluster_ids if i is not None]

        stats = {"slot_tgt": [], "type_tgt": [], "value_tgt": [],
                 "slot_clst": [], "type_clst": [], "value_clst": []}

        for i, (obj, val) in enumerate(pairs):
            if (i + 1) % 5 == 0:
                log(f"    {attr_type} pair {i+1}/{len(pairs)} done")

            base = BASE.format(obj=obj)
            tgt_id = get_token_id(tokenizer, val)

            # Get baseline logits
            baseline_logits = get_baseline_logits(model, tokenizer, device, base)

            for level in ["slot", "type", "value"]:
                test = templates[level].format(obj=obj, val=val)

                # Extract direction
                h_base = extract_rep(model, tokenizer, device, base, opt_layer)
                h_test = extract_rep(model, tokenizer, device, test, opt_layer)
                direction = h_test - h_base
                dir_norm = direction / (np.linalg.norm(direction) + 1e-8)

                # Inject and get logits
                patched_logits = inject_and_get_logits(
                    model, tokenizer, device, base, dir_norm, opt_layer, alpha
                )

                # Target word delta
                if tgt_id is not None:
                    tgt_delta = float(patched_logits[tgt_id] - baseline_logits[tgt_id])
                else:
                    tgt_delta = 0.0
                stats[f"{level}_tgt"].append(tgt_delta)

                # Cluster delta
                clst_deltas = [float(patched_logits[cid] - baseline_logits[cid]) for cid in cluster_ids]
                stats[f"{level}_clst"].append(float(np.mean(clst_deltas)))

        # Aggregate
        agg = {}
        for level in ["slot", "type", "value"]:
            gt = stats[f"{level}_tgt"]
            gc = stats[f"{level}_clst"]
            agg[level] = {
                "tgt_mean": round(float(np.mean(gt)), 4),
                "cluster_mean": round(float(np.mean(gc)), 4),
                "tgt_negative_rate": round(sum(1 for x in gt if x < 0) / max(len(gt), 1), 4),
                "n_pairs": len(gt),
            }

        best_level = max(["slot", "type", "value"], key=lambda l: agg[l]["tgt_mean"])
        if agg[best_level]["tgt_mean"] <= 0:
            best_level = "none"

        results[attr_type] = {
            "aggregated": agg,
            "best_level": best_level,
        }

        log(f"  {attr_type} Specialized (20 pairs): slot={agg['slot']['tgt_mean']:.4f}, "
            f"type={agg['type']['tgt_mean']:.4f}, value={agg['value']['tgt_mean']:.4f} | best={best_level}")

    # Save
    out = {
        "model": model_name,
        "n_layers": cfg["n_layers"],
        "d_model": cfg["d_model"],
        "opt_layer": opt_layer,
        "alpha": alpha,
        "n_pairs_per_type": 20,
        "results": results,
    }

    os.makedirs("results/phase325b_confirm", exist_ok=True)
    out_path = f"results/phase325b_confirm/{model_name}_phase325b.json"
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
