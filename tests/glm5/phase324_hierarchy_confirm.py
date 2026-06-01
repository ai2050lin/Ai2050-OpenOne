"""
Phase 324: Multi-Template Confirmation of Attribute Hierarchy (Slot/Type/Value)
===============================================================================

Goal: Confirm that the slot→type→value decomposition is NOT a template artifact,
but reflects genuine hierarchical encoding in the model.

Design:
1. Multiple parallel templates for each level (slot/type/value) to control for
   abstractness, naturalness, word frequency, syntactic structure
2. Extended attribute types: color(30), taste(30), temperature(30), texture(30), shape(30), size(30)
3. Cross-template consistency check: does slot/type/value separation persist?
4. Cluster readout: does slot push ALL attribute clusters, type push SAME-TYPE clusters,
   value push SPECIFIC word?
5. Object-attribute binding test: does "apple+color" push apple-compatible colors more?

Usage:
  python tests/glm5/phase324_hierarchy_confirm.py qwen3
  python tests/glm5/phase324_hierarchy_confirm.py glm4
  python tests/glm5/phase324_hierarchy_confirm.py deepseek7b
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model, get_W_U

RESULT_DIR = Path("results/phase324_hierarchy")
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
# Multi-Template Design for Slot/Type/Value
# =====================================================================
# Key insight: each "level" has MULTIPLE templates with different wording
# but same semantic level. If directions from different templates are
# consistent (high cosine), the structure is real, not template artifact.

MULTI_TEMPLATES = {
    "color": {
        "slot": [
            "The {obj} has some feature",
            "The {obj} has a property",
            "The {obj} has some quality",
            "The {obj} has a characteristic",
        ],
        "type": [
            "The {obj} has a color",
            "The {obj} has a certain color",
            "The {obj} has some color",
            "The color of the {obj} is",
        ],
        "value": [
            "The {obj} is {val}",
            "The {obj} has the color {val}",
            "The {obj} appears {val}",
            "The {obj} looks {val}",
        ],
        "baseline": [
            "The {obj} is something",
            "The {obj} is just an object",
        ],
    },
    "taste": {
        "slot": [
            "The {obj} has some feature",
            "The {obj} has a property",
            "The {obj} has some quality",
            "The {obj} has a characteristic",
        ],
        "type": [
            "The {obj} has a taste",
            "The {obj} has a certain taste",
            "The {obj} has some taste",
            "The flavor of the {obj} is",
        ],
        "value": [
            "The {obj} is {val}",
            "The {obj} has the taste {val}",
            "The {obj} tastes {val}",
            "The {obj} flavor is {val}",
        ],
        "baseline": [
            "The {obj} is something",
            "The {obj} is just an object",
        ],
    },
    "temperature": {
        "slot": [
            "The {obj} has some feature",
            "The {obj} has a property",
            "The {obj} has some quality",
            "The {obj} has a characteristic",
        ],
        "type": [
            "The {obj} has a temperature",
            "The {obj} has a certain temperature",
            "The {obj} has some temperature",
            "The temperature of the {obj} is",
        ],
        "value": [
            "The {obj} is {val}",
            "The {obj} has the temperature {val}",
            "The {obj} feels {val}",
            "The {obj} temperature is {val}",
        ],
        "baseline": [
            "The {obj} is something",
            "The {obj} is just an object",
        ],
    },
    "texture": {
        "slot": [
            "The {obj} has some feature",
            "The {obj} has a property",
            "The {obj} has some quality",
            "The {obj} has a characteristic",
        ],
        "type": [
            "The {obj} has a texture",
            "The {obj} has a certain texture",
            "The {obj} has some texture",
            "The texture of the {obj} is",
        ],
        "value": [
            "The {obj} is {val}",
            "The {obj} has the texture {val}",
            "The {obj} feels {val}",
            "The {obj} texture is {val}",
        ],
        "baseline": [
            "The {obj} is something",
            "The {obj} is just an object",
        ],
    },
    "shape": {
        "slot": [
            "The {obj} has some feature",
            "The {obj} has a property",
            "The {obj} has some quality",
            "The {obj} has a characteristic",
        ],
        "type": [
            "The {obj} has a shape",
            "The {obj} has a certain shape",
            "The {obj} has some shape",
            "The shape of the {obj} is",
        ],
        "value": [
            "The {obj} is {val}",
            "The {obj} has the shape {val}",
            "The {obj} appears {val}",
            "The {obj} shape is {val}",
        ],
        "baseline": [
            "The {obj} is something",
            "The {obj} is just an object",
        ],
    },
    "size": {
        "slot": [
            "The {obj} has some feature",
            "The {obj} has a property",
            "The {obj} has some quality",
            "The {obj} has a characteristic",
        ],
        "type": [
            "The {obj} has a size",
            "The {obj} has a certain size",
            "The {obj} has some size",
            "The size of the {obj} is",
        ],
        "value": [
            "The {obj} is {val}",
            "The {obj} has the size {val}",
            "The {obj} appears {val}",
            "The {obj} size is {val}",
        ],
        "baseline": [
            "The {obj} is something",
            "The {obj} is just an object",
        ],
    },
}

# Object-attribute pairs: 30 pairs per attribute type (balanced)
OBJ_ATTR_PAIRS = {
    "color": [
        ("apple", "red"), ("sky", "blue"), ("grass", "green"), ("sun", "yellow"),
        ("snow", "white"), ("night", "black"), ("orange", "orange"), ("grape", "purple"),
        ("rose", "red"), ("ocean", "blue"), ("leaf", "green"), ("gold", "yellow"),
        ("cloud", "white"), ("coal", "black"), ("carrot", "orange"), ("plum", "purple"),
        ("cherry", "red"), ("sapphire", "blue"), ("emerald", "green"), ("lemon", "yellow"),
        ("ivory", "white"), ("raven", "black"), ("marigold", "orange"), ("lavender", "purple"),
        ("strawberry", "red"), ("turquoise", "blue"), ("mint", "green"), ("banana", "yellow"),
        ("pearl", "white"), ("obsidian", "black"),
    ],
    "taste": [
        ("lemon", "sour"), ("honey", "sweet"), ("coffee", "bitter"), ("salt", "salty"),
        ("chili", "spicy"), ("vinegar", "sour"), ("candy", "sweet"), ("dark chocolate", "bitter"),
        ("soy sauce", "salty"), ("pepper", "spicy"), ("grapefruit", "sour"), ("sugar", "sweet"),
        ("espresso", "bitter"), ("seawater", "salty"), ("ginger", "spicy"), ("lime", "sour"),
        ("maple syrup", "sweet"), ("kale", "bitter"), ("pretzel", "salty"), ("wasabi", "spicy"),
        ("tamarind", "sour"), ("caramel", "sweet"), ("coffee bean", "bitter"), ("bacon", "salty"),
        ("jalapeno", "spicy"), ("yogurt", "sour"), ("vanilla", "sweet"), ("olive", "bitter"),
        ("cheese", "salty"), ("cinnamon", "spicy"),
    ],
    "temperature": [
        ("fire", "hot"), ("ice", "cold"), ("blanket", "warm"), ("breeze", "cool"),
        ("lava", "hot"), ("frost", "cold"), ("heater", "warm"), ("refrigerator", "cool"),
        ("stove", "hot"), ("glacier", "cold"), ("sunlight", "warm"), ("shade", "cool"),
        ("desert", "hot"), ("arctic", "cold"), ("tea", "warm"), ("spring", "cool"),
        ("oven", "hot"), ("snowflake", "cold"), ("campfire", "warm"), ("evening", "cool"),
        ("volcano", "hot"), ("iceberg", "cold"), ("cocoa", "warm"), ("waterfall", "cool"),
        ("summer", "hot"), ("winter", "cold"), ("autumn", "warm"), ("morning", "cool"),
        ("sauna", "hot"), ("freezer", "cold"),
    ],
    "texture": [
        ("silk", "smooth"), ("sandpaper", "rough"), ("pillow", "soft"), ("diamond", "hard"),
        ("glass", "smooth"), ("bark", "rough"), ("cotton", "soft"), ("rock", "hard"),
        ("velvet", "smooth"), ("concrete", "rough"), ("feather", "soft"), ("steel", "hard"),
        ("marble", "smooth"), ("gravel", "rough"), ("wool", "soft"), ("iron", "hard"),
        ("ice", "smooth"), ("brick", "rough"), ("sponge", "soft"), ("bone", "hard"),
        ("porcelain", "smooth"), ("asphalt", "rough"), ("fur", "soft"), ("shell", "hard"),
        ("polish", "smooth"), ("rust", "rough"), ("cashmere", "soft"), ("granite", "hard"),
        ("ceramic", "smooth"), ("sand", "rough"),
    ],
    "shape": [
        ("ball", "round"), ("box", "square"), ("needle", "thin"), ("mountain", "tall"),
        ("wheel", "round"), ("tile", "square"), ("wire", "thin"), ("tower", "tall"),
        ("coin", "round"), ("window", "square"), ("thread", "thin"), ("building", "tall"),
        ("planet", "round"), ("tablet", "square"), ("hair", "thin"), ("tree", "tall"),
        ("orange", "round"), ("frame", "square"), ("ribbon", "thin"), ("pillar", "tall"),
        ("globe", "round"), ("screen", "square"), ("spider web", "thin"), ("skyscraper", "tall"),
        ("moon", "round"), ("card", "square"), ("string", "thin"), ("flagpole", "tall"),
        ("bubble", "round"), ("canvas", "square"),
    ],
    "size": [
        ("elephant", "large"), ("ant", "small"), ("whale", "huge"), ("grain", "tiny"),
        ("mountain", "large"), ("pebble", "small"), ("building", "huge"), ("speck", "tiny"),
        ("ship", "large"), ("insect", "small"), ("planet", "huge"), ("dust", "tiny"),
        ("tree", "large"), ("seed", "small"), ("continent", "huge"), ("bacterium", "tiny"),
        ("car", "large"), ("button", "small"), ("galaxy", "huge"), ("atom", "tiny"),
        ("house", "large"), ("pin", "small"), ("ocean", "huge"), ("pixel", "tiny"),
        ("horse", "large"), ("coin", "small"), ("sun", "huge"), ("cell", "tiny"),
        ("piano", "large"), ("bead", "small"),
    ],
}

# Word clusters for readout analysis
WORD_CLUSTERS = {
    "color": ["red", "blue", "green", "yellow", "white", "black", "orange", "purple", "pink", "brown"],
    "taste": ["sweet", "sour", "bitter", "salty", "spicy", "savory", "tangy", "umami"],
    "temperature": ["hot", "cold", "warm", "cool", "freezing", "boiling", "lukewarm", "frigid"],
    "texture": ["smooth", "rough", "soft", "hard", "sharp", "fluffy", "slick", "bumpy"],
    "shape": ["round", "square", "thin", "tall", "flat", "curved", "wide", "narrow"],
    "size": ["large", "small", "huge", "tiny", "big", "little", "massive", "minute"],
    "object": ["apple", "table", "car", "house", "book", "water", "idea", "music"],
    "action": ["run", "eat", "think", "make", "see", "walk", "write", "build"],
    "negation": ["not", "no", "never", "neither", "nothing", "none", "without", "lack"],
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


def extract_rep_at_layer(model, tokenizer, device, sentence, target_layer):
    """Extract representation at a single layer."""
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


def inject_direction_at_layer(model, tokenizer, device, prompt, direction, layer_idx, alpha):
    """Inject direction at a specific layer and return logits."""
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


def get_cluster_token_ids(tokenizer, cluster_words):
    ids = []
    for w in cluster_words:
        tok_ids = tokenizer.encode(w, add_special_tokens=False)
        if tok_ids:
            ids.append((w, tok_ids[0]))
    return ids


def compute_cluster_mean(logits, cluster_ids):
    if not cluster_ids:
        return 0.0
    return float(np.mean([float(logits[tid]) for _, tid in cluster_ids]))


def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase324_{model_name}.log")

    log(f"=== Phase 324: Multi-Template Hierarchy Confirmation for {model_name} ===")

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}")

    # Select optimal layer per model
    if model_name == "glm4":
        test_layers = [2, 3, 4]
        opt_layer = 3
    elif model_name == "qwen3":
        test_layers = [0, 1, 2]
        opt_layer = 0
    else:
        test_layers = [5, 6, 7]
        opt_layer = 6

    alpha = 2.0
    results = {}

    # ===================================================================
    # Test 1: Cross-Template Consistency
    # For each level (slot/type/value), extract directions from different
    # templates and measure their pairwise cosine similarity.
    # If directions from different templates for the SAME level have high
    # cosine, the structure is real and not template artifact.
    # ===================================================================
    log("\n" + "="*60)
    log("Test 1: Cross-Template Consistency")
    log("="*60)

    # Use first 8 pairs per attribute type for efficiency
    consistency_results = []
    n_pairs_per_type = 8

    for attr_type in ["color", "taste", "temperature", "texture", "shape", "size"]:
        log(f"\n  Attribute type: {attr_type}")
        pairs = OBJ_ATTR_PAIRS[attr_type][:n_pairs_per_type]
        templates = MULTI_TEMPLATES[attr_type]

        # Collect directions per level per template
        level_directions = defaultdict(lambda: defaultdict(list))  # level -> template_idx -> [directions]

        for pair_idx, (noun, val) in enumerate(pairs):
            try:
                val_ids = tokenizer.encode(val, add_special_tokens=False)
                if not val_ids:
                    continue

                for li in [opt_layer]:
                    # Extract baseline direction (average of two baselines)
                    base_dirs = []
                    for base_tmpl in templates["baseline"]:
                        sent = base_tmpl.format(obj=noun, val=val)
                        h = extract_rep_at_layer(model, tokenizer, device, sent, li)
                        base_dirs.append(h)
                    h_baseline = np.mean(base_dirs, axis=0)

                    # Extract slot/type/value directions for each template
                    for level in ["slot", "type", "value"]:
                        for tmpl_idx, tmpl in enumerate(templates[level]):
                            sent = tmpl.format(obj=noun, val=val)
                            h = extract_rep_at_layer(model, tokenizer, device, sent, li)
                            d = h - h_baseline
                            norm = np.linalg.norm(d)
                            if norm > 1e-10:
                                level_directions[level][tmpl_idx].append(d / norm)

                if pair_idx % 4 == 0:
                    log(f"    {attr_type} pair {pair_idx+1}/{n_pairs_per_type} done")

            except Exception as e:
                log(f"    Error on pair {noun}→{val}: {str(e)[:60]}")
                continue

        # Compute cross-template cosine for each level
        level_consistency = {}
        for level in ["slot", "type", "value"]:
            tmpl_dirs = level_directions[level]
            tmpl_indices = sorted(tmpl_dirs.keys())
            if len(tmpl_indices) < 2:
                continue

            # Average direction per template
            avg_dirs = {}
            for ti in tmpl_indices:
                all_dirs = np.array(tmpl_dirs[ti])
                avg_dir = np.mean(all_dirs, axis=0)
                norm = np.linalg.norm(avg_dir)
                if norm > 1e-10:
                    avg_dirs[ti] = avg_dir / norm

            # Pairwise cosine between templates
            cosines = []
            dir_list = list(avg_dirs.values())
            for i in range(len(dir_list)):
                for j in range(i+1, len(dir_list)):
                    cos_val = float(np.dot(dir_list[i], dir_list[j]))
                    cosines.append(cos_val)

            if cosines:
                level_consistency[level] = {
                    "mean_cos": round(float(np.mean(cosines)), 4),
                    "min_cos": round(float(np.min(cosines)), 4),
                    "max_cos": round(float(np.max(cosines)), 4),
                    "n_templates": len(tmpl_indices),
                    "n_pairs": len(cosines),
                }

        # Cross-level cosine (slot vs type, slot vs value, type vs value)
        cross_level_cos = {}
        level_avg = {}
        for level in ["slot", "type", "value"]:
            if level in level_directions:
                all_dirs = []
                for ti in level_directions[level]:
                    all_dirs.extend(level_directions[level][ti])
                if all_dirs:
                    avg = np.mean(np.array(all_dirs), axis=0)
                    norm = np.linalg.norm(avg)
                    if norm > 1e-10:
                        level_avg[level] = avg / norm

        for l1, l2 in [("slot", "type"), ("slot", "value"), ("type", "value")]:
            if l1 in level_avg and l2 in level_avg:
                cos_val = float(np.dot(level_avg[l1], level_avg[l2]))
                cross_level_cos[f"cos({l1},{l2})"] = round(cos_val, 4)

        entry = {
            "attr_type": attr_type,
            "layer": opt_layer,
            "level_consistency": level_consistency,
            "cross_level_cos": cross_level_cos,
        }
        consistency_results.append(entry)

        log(f"    Consistency:")
        for level, stats in level_consistency.items():
            log(f"      {level}: mean_cos={stats['mean_cos']:.4f}, "
                f"min={stats['min_cos']:.4f}, max={stats['max_cos']:.4f}")
        for k, v in cross_level_cos.items():
            log(f"      {k}={v:.4f}")

        torch.cuda.empty_cache()

    results["consistency"] = consistency_results

    # ===================================================================
    # Test 2: Cluster Readout Pattern
    # Inject slot/type/value directions and check which word clusters move.
    # - Slot should push ALL attribute clusters (opens attribute space)
    # - Type should push SAME-TYPE cluster (constrains to specific type)
    # - Value should push SPECIFIC word (selects exact value)
    # ===================================================================
    log("\n" + "="*60)
    log("Test 2: Cluster Readout Pattern — Slot/Type/Value")
    log("="*60)

    cluster_readout_results = []
    n_test_pairs = 6  # per attribute type

    for attr_type in ["color", "taste", "temperature", "texture", "shape", "size"]:
        log(f"\n  Cluster readout: {attr_type}")
        pairs = OBJ_ATTR_PAIRS[attr_type][:n_test_pairs]
        templates = MULTI_TEMPLATES[attr_type]

        # Get cluster token IDs
        cluster_ids = {}
        for c_name, c_words in WORD_CLUSTERS.items():
            cluster_ids[c_name] = get_cluster_token_ids(tokenizer, c_words)

        for pair_idx, (noun, val) in enumerate(pairs):
            val_ids = tokenizer.encode(val, add_special_tokens=False)
            if not val_ids:
                continue
            tgt_id = val_ids[0]

            for li in [opt_layer]:
                # Get baseline
                target_prompt = f"The {noun} is"
                baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
                baseline_logit = float(baseline_logits[tgt_id])
                baseline_clusters = {cn: compute_cluster_mean(baseline_logits, cids)
                                    for cn, cids in cluster_ids.items()}

                # Extract directions (using template index 0 for each level)
                base_sent = templates["baseline"][0].format(obj=noun, val=val)
                h_baseline = extract_rep_at_layer(model, tokenizer, device, base_sent, li)

                for level in ["slot", "type", "value"]:
                    tmpl = templates[level][0]  # Use first template
                    sent = tmpl.format(obj=noun, val=val)
                    h_level = extract_rep_at_layer(model, tokenizer, device, sent, li)
                    d = h_level - h_baseline
                    norm = np.linalg.norm(d)
                    if norm < 1e-10:
                        continue
                    d_unit = d / norm

                    # Inject and measure cluster deltas
                    inj_logits = inject_direction_at_layer(
                        model, tokenizer, device, target_prompt, d_unit, li, alpha)

                    tgt_delta = float(inj_logits[tgt_id] - baseline_logit)
                    cluster_deltas = {}
                    for cn, cids in cluster_ids.items():
                        inj_mean = compute_cluster_mean(inj_logits, cids)
                        cluster_deltas[cn] = round(inj_mean - baseline_clusters[cn], 4)

                    entry = {
                        "attr_type": attr_type,
                        "pair": f"{noun}→{val}",
                        "layer": li,
                        "level": level,
                        "tgt_delta": round(tgt_delta, 4),
                        "cluster_deltas": cluster_deltas,
                    }
                    cluster_readout_results.append(entry)

            if pair_idx % 3 == 0:
                log(f"    {attr_type} pair {pair_idx+1}/{n_test_pairs} done")
            torch.cuda.empty_cache()

    results["cluster_readout"] = cluster_readout_results

    # Aggregate: average cluster deltas per level per attr_type
    log("\n  Cluster Readout Aggregation:")
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in cluster_readout_results:
        at = r["attr_type"]
        level = r["level"]
        for cn, delta in r["cluster_deltas"].items():
            agg[at][level][cn].append(delta)

    for at in sorted(agg.keys()):
        log(f"    {at}:")
        for level in ["slot", "type", "value"]:
            if level in agg[at]:
                parts = []
                for cn in ["color", "taste", "temperature", "texture", "shape", "size", "object"]:
                    if cn in agg[at][level]:
                        mean_d = np.mean(agg[at][level][cn])
                        parts.append(f"{cn}={mean_d:.3f}")
                if parts:
                    log(f"      {level}: {', '.join(parts)}")

    # ===================================================================
    # Test 3: Slot Suppresses Specific Values — Confirmation
    # Phase 323b found slot direction suppresses specific attribute words.
    # Verify with more templates and more attribute types.
    # ===================================================================
    log("\n" + "="*60)
    log("Test 3: Slot Direction Effect on Specific Values")
    log("="*60)

    slot_effect_results = []
    n_slot_pairs = 10  # per attribute type

    for attr_type in ["color", "taste", "temperature", "texture", "shape", "size"]:
        pairs = OBJ_ATTR_PAIRS[attr_type][:n_slot_pairs]
        templates = MULTI_TEMPLATES[attr_type]

        slot_tgt_deltas = []
        slot_cluster_deltas = []
        type_tgt_deltas = []
        type_cluster_deltas = []
        value_tgt_deltas = []
        value_cluster_deltas = []

        for noun, val in pairs:
            val_ids = tokenizer.encode(val, add_special_tokens=False)
            if not val_ids:
                continue
            tgt_id = val_ids[0]
            target_cluster = WORD_CLUSTERS.get(attr_type, [])
            tgt_cluster_ids = get_cluster_token_ids(tokenizer, target_cluster)

            for li in [opt_layer]:
                target_prompt = f"The {noun} is"
                baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
                baseline_logit = float(baseline_logits[tgt_id])
                baseline_cluster = compute_cluster_mean(baseline_logits, tgt_cluster_ids)

                base_sent = templates["baseline"][0].format(obj=noun, val=val)
                h_baseline = extract_rep_at_layer(model, tokenizer, device, base_sent, li)

                for level, level_tgt, level_cluster in [
                    ("slot", slot_tgt_deltas, slot_cluster_deltas),
                    ("type", type_tgt_deltas, type_cluster_deltas),
                    ("value", value_tgt_deltas, value_cluster_deltas),
                ]:
                    tmpl = templates[level][0]
                    sent = tmpl.format(obj=noun, val=val)
                    h_level = extract_rep_at_layer(model, tokenizer, device, sent, li)
                    d = h_level - h_baseline
                    norm = np.linalg.norm(d)
                    if norm < 1e-10:
                        continue
                    d_unit = d / norm

                    inj_logits = inject_direction_at_layer(
                        model, tokenizer, device, target_prompt, d_unit, li, alpha)
                    level_tgt.append(float(inj_logits[tgt_id] - baseline_logit))
                    level_cluster.append(
                        compute_cluster_mean(inj_logits, tgt_cluster_ids) - baseline_cluster)

            torch.cuda.empty_cache()

        entry = {
            "attr_type": attr_type,
            "layer": opt_layer,
            "slot": {"tgt_mean": round(float(np.mean(slot_tgt_deltas)), 4) if slot_tgt_deltas else 0,
                     "cluster_mean": round(float(np.mean(slot_cluster_deltas)), 4) if slot_cluster_deltas else 0,
                     "tgt_negative_rate": round(sum(1 for x in slot_tgt_deltas if x < 0) / max(len(slot_tgt_deltas), 1), 4)},
            "type": {"tgt_mean": round(float(np.mean(type_tgt_deltas)), 4) if type_tgt_deltas else 0,
                     "cluster_mean": round(float(np.mean(type_cluster_deltas)), 4) if type_cluster_deltas else 0,
                     "tgt_negative_rate": round(sum(1 for x in type_tgt_deltas if x < 0) / max(len(type_tgt_deltas), 1), 4)},
            "value": {"tgt_mean": round(float(np.mean(value_tgt_deltas)), 4) if value_tgt_deltas else 0,
                      "cluster_mean": round(float(np.mean(value_cluster_deltas)), 4) if value_cluster_deltas else 0,
                      "tgt_negative_rate": round(sum(1 for x in value_tgt_deltas if x < 0) / max(len(value_tgt_deltas), 1), 4)},
        }
        slot_effect_results.append(entry)

        log(f"    {attr_type}: slot_tgt={entry['slot']['tgt_mean']:.4f}(neg={entry['slot']['tgt_negative_rate']:.0%}), "
            f"type_tgt={entry['type']['tgt_mean']:.4f}(neg={entry['type']['tgt_negative_rate']:.0%}), "
            f"value_tgt={entry['value']['tgt_mean']:.4f}(neg={entry['value']['tgt_negative_rate']:.0%})")

    results["slot_effect"] = slot_effect_results

    # ===================================================================
    # Test 4: Object-Attribute Binding
    # Does "apple+color" push apple-compatible colors (red/green) more than
    # incompatible colors (blue/purple)?
    # ===================================================================
    log("\n" + "="*60)
    log("Test 4: Object-Attribute Binding")
    log("="*60)

    binding_results = []
    binding_pairs = [
        ("apple", "red", "color", ["red", "green", "yellow"], ["blue", "purple", "black"]),
        ("banana", "yellow", "color", ["yellow", "green"], ["red", "blue", "black"]),
        ("snow", "white", "color", ["white", "blue"], ["red", "green", "orange"]),
        ("lemon", "sour", "taste", ["sour", "bitter"], ["sweet", "spicy"]),
        ("honey", "sweet", "taste", ["sweet", "savory"], ["sour", "bitter"]),
        ("fire", "hot", "temperature", ["hot", "warm"], ["cold", "cool", "freezing"]),
        ("ice", "cold", "temperature", ["cold", "cool"], ["hot", "warm", "boiling"]),
        ("silk", "smooth", "texture", ["smooth", "soft"], ["rough", "hard"]),
    ]

    for noun, val, attr_type, compat_words, incompat_words in binding_pairs:
        val_ids = tokenizer.encode(val, add_special_tokens=False)
        if not val_ids:
            continue
        tgt_id = val_ids[0]

        compat_ids = get_cluster_token_ids(tokenizer, compat_words)
        incompat_ids = get_cluster_token_ids(tokenizer, incompat_words)

        templates = MULTI_TEMPLATES[attr_type]

        for li in [opt_layer]:
            target_prompt = f"The {noun} is"
            baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
            baseline_compat = compute_cluster_mean(baseline_logits, compat_ids)
            baseline_incompat = compute_cluster_mean(baseline_logits, incompat_ids)

            base_sent = templates["baseline"][0].format(obj=noun, val=val)
            h_baseline = extract_rep_at_layer(model, tokenizer, device, base_sent, li)

            for level in ["slot", "type", "value"]:
                tmpl = templates[level][0]
                sent = tmpl.format(obj=noun, val=val)
                h_level = extract_rep_at_layer(model, tokenizer, device, sent, li)
                d = h_level - h_baseline
                norm = np.linalg.norm(d)
                if norm < 1e-10:
                    continue
                d_unit = d / norm

                inj_logits = inject_direction_at_layer(
                    model, tokenizer, device, target_prompt, d_unit, li, alpha)

                compat_delta = compute_cluster_mean(inj_logits, compat_ids) - baseline_compat
                incompat_delta = compute_cluster_mean(inj_logits, incompat_ids) - baseline_incompat
                binding_score = compat_delta - incompat_delta  # positive = binding exists

                entry = {
                    "pair": f"{noun}→{val}",
                    "attr_type": attr_type,
                    "level": level,
                    "compat_delta": round(compat_delta, 4),
                    "incompat_delta": round(incompat_delta, 4),
                    "binding_score": round(binding_score, 4),
                }
                binding_results.append(entry)

        torch.cuda.empty_cache()

    # Aggregate binding by level
    log("\n  Binding Aggregation by Level:")
    level_binding = defaultdict(list)
    for r in binding_results:
        level_binding[r["level"]].append(r["binding_score"])

    for level in ["slot", "type", "value"]:
        if level in level_binding:
            scores = level_binding[level]
            log(f"    {level}: binding_score mean={np.mean(scores):.4f}, "
                f"positive_rate={sum(1 for s in scores if s > 0)/max(len(scores),1):.0%}")

    results["binding"] = binding_results

    # ===================================================================
    # Save
    # ===================================================================
    output = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "opt_layer": opt_layer,
        "alpha": alpha,
        "results": results,
    }

    out_path = RESULT_DIR / f"{model_name}_phase324.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # Summary
    log("\n" + "="*60)
    log(f"PHASE 324 SUMMARY - {model_name}")
    log("="*60)

    log("\n  Test 1: Cross-Template Consistency")
    for cr in consistency_results:
        at = cr["attr_type"]
        for level, stats in cr["level_consistency"].items():
            log(f"    {at}/{level}: mean_cos={stats['mean_cos']:.4f}")
        for k, v in cr["cross_level_cos"].items():
            log(f"    {at}/{k}={v:.4f}")

    log("\n  Test 3: Slot Effect")
    for se in slot_effect_results:
        log(f"    {se['attr_type']}: slot_tgt={se['slot']['tgt_mean']:.4f}, "
            f"type_tgt={se['type']['tgt_mean']:.4f}, value_tgt={se['value']['tgt_mean']:.4f}")

    log("\n  Test 4: Binding")
    for level in ["slot", "type", "value"]:
        if level in level_binding:
            log(f"    {level}: binding={np.mean(level_binding[level]):.4f}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released. Total time: {time.time()-t0:.1f}s")

    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        for mn in ["qwen3", "glm4", "deepseek7b"]:
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
