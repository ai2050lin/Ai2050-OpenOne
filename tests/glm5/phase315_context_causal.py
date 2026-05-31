"""
Phase 315: Context Activation + Relation-Level Causal Test
==========================================================
Solves two key gaps from Phase 314:
  A) Attribute/function relations not preserved in static context → test with task context
  B) Mantel correlation is not causal → do relation-level causal patching

Part A: Context Activation Test
  - Compare concept pairs in 3 conditions:
    1. Static (no task): "the apple was fresh"
    2. Attribute context: "The apple is usually ___"
    3. Function context: "You can cut it with a ___"
  - Measure cosine distance between related vs random pairs in each condition
  - If distance decreases in context → relation is conditionally activated

Part B: Relation-Level Causal Test
  - For each relation type, extract "relation direction" (avg delta between related pairs)
  - Inject relation direction at mid-layer and measure effect on target tokens
  - Compare: same_class injection → fruit output? negation injection → not output?
  - This tests whether relation structure has causal efficacy

Usage:
  python tests/glm5/phase315_context_causal.py qwen3
  python tests/glm5/phase315_context_causal.py glm4
  python tests/glm5/phase315_context_causal.py deepseek7b
"""
import sys, os, gc, time, json, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase315_context_causal")
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
# PART A: CONTEXT ACTIVATION STIMULI
# =====================================================================

# Concept pairs for each relation type, with context templates
CONTEXT_PAIRS = {
    "attribute": {
        "pairs": [
            ("apple", "red"), ("apple", "sweet"), ("banana", "yellow"), ("banana", "sweet"),
            ("knife", "sharp"), ("ice", "cold"), ("fire", "hot"), ("dog", "loyal"),
            ("sky", "blue"), ("snow", "white"), ("lemon", "sour"), ("stone", "hard"),
            ("feather", "light_weight"), ("silk", "smooth"), ("honey", "sticky"),
        ],
        "contexts": {
            "static": "the {obj} was there",
            "attribute_probe": "The {obj} is usually",
            "attribute_fill": "The {obj} has the quality of being",
        },
        "target_attr": "The {obj} is usually {attr}",
    },
    "function": {
        "pairs": [
            ("knife", "cut"), ("key", "open"), ("car", "drive"), ("hammer", "hit"),
            ("pen", "write"), ("cup", "drink"), ("phone", "call"), ("door", "enter"),
            ("book", "read"), ("shovel", "dig"), ("boat", "sail"), ("lamp", "illuminate"),
            ("lock", "secure"), ("bridge", "cross"), ("knife", "slice"),
        ],
        "contexts": {
            "static": "the {obj} was there",
            "function_probe": "You use a {obj} to",
            "function_fill": "A {obj} is designed for",
        },
        "target_func": "You use a {obj} to {func}",
    },
    "same_class": {
        "pairs": [
            ("apple", "banana"), ("dog", "cat"), ("knife", "hammer"),
            ("car", "bus"), ("river", "lake"), ("table", "chair"),
            ("rose", "lily"), ("eagle", "hawk"), ("sword", "shield"),
        ],
        "contexts": {
            "static": "the {obj1} was there",
            "category_probe": "The {obj1} and the {obj2} are both",
        },
    },
    "negation": {
        "pairs": [
            ("happy", "not_happy"), ("possible", "not_possible"),
            ("open", "not_open"), ("clean", "not_clean"),
            ("safe", "not_safe"), ("good", "not_good"),
            ("fair", "not_fair"), ("clear", "not_clear"),
        ],
        "contexts": {
            "static_pos": "they felt {pos}",
            "static_neg": "they were {neg}",
            "negation_probe": "It is {neg} that things are {pos}",
        },
    },
}

# Random baseline pairs (cross-category, no expected relation)
RANDOM_PAIRS = [
    ("apple", "hammer"), ("dog", "key"), ("knife", "river"),
    ("car", "happy"), ("table", "cold"), ("fire", "cat"),
    ("silk", "bus"), ("book", "lemon"), ("lamp", "eagle"),
    ("door", "sweet"), ("stone", "shovel"), ("snow", "sword"),
    ("feather", "bridge"), ("honey", "chair"), ("phone", "sky"),
    ("rose", "pen"), ("lock", "hawk"), ("boat", "sour"),
    ("cup", "lily"), ("shovel", "smooth"),
]


# =====================================================================
# PART B: CAUSAL RELATION DIRECTIONS
# =====================================================================

# For causal injection, we need sentence pairs that differ by a specific relation
CAUSAL_STIMULI = {
    "same_class": {
        "base": "the fruit was very",
        "related": ["the apple was very", "the banana was very", "the pear was very"],
        "unrelated": ["the hammer was very", "the knife was very", "the car was very"],
        "target_tokens": ["fresh", "sweet", "ripe", "juicy"],
    },
    "hypernym": {
        "base": "it was a kind of",
        "related": ["the apple is a kind of", "the dog is a kind of", "the knife is a kind of"],
        "unrelated": ["the running is a kind of", "the red is a kind of", "the cutting is a kind of"],
        "target_tokens": ["fruit", "animal", "tool"],
    },
    "negation": {
        "base": "they were very happy about it",
        "related": ["they were not happy about it", "they were never happy about it"],
        "unrelated": ["they were very sad about it", "they were quite angry about it"],
        "target_tokens": ["not", "never", "unhappy"],
    },
    "antonym": {
        "base": "the room was very bright",
        "related": ["the room was very dark", "the room was very dim"],
        "unrelated": ["the room was very large", "the room was very quiet"],
        "target_tokens": ["dark", "dim", "shadow"],
    },
    "attribute": {
        "base": "the apple was very",
        "related": ["the red apple was very", "the sweet apple was very", "the ripe apple was very"],
        "unrelated": ["the heavy apple was very", "the distant apple was very"],
        "target_tokens": ["red", "sweet", "ripe", "colorful"],
    },
    "function": {
        "base": "the knife was very",
        "related": ["the knife for cutting was very", "the sharp knife was very"],
        "unrelated": ["the old knife was very", "the lost knife was very"],
        "target_tokens": ["sharp", "useful", "cutting"],
    },
}


def load_model_bf16(model_name):
    """Load model with BF16 + device_map=auto + SDPA"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]

    log(f"Loading {model_name} (bf16 + device_map=auto + sdpa)...")
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
    device = next(model.parameters()).device

    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"Model loaded: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        log(f"Layer allocation: GPU={gpu_count}, CPU={cpu_count} components")

    return model, tokenizer, device


def get_rep_at_layer(model, tokenizer, device, sentence, layer_idx):
    """Get last-token representation at a specific layer."""
    layers = get_layers(model)
    layer = layers[layer_idx]

    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured["h"] = output[0].detach()
        else:
            captured["h"] = output.detach()

    handle = layer.register_forward_hook(hook_fn)

    inp = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inp)

    handle.remove()

    if "h" in captured:
        return captured["h"][0, -1].cpu().float().numpy()
    return None


# =====================================================================
# PART A: CONTEXT ACTIVATION ANALYSIS
# =====================================================================

def run_context_activation(model, tokenizer, device, model_info):
    """Test whether attribute/function relations are conditionally activated by context."""
    log("=" * 60)
    log("PART A: Context Activation Test")
    log("=" * 60)

    n_layers = model_info.n_layers
    # Sample layers - ensure deep layers included
    if n_layers >= 36:
        test_layers = [2, 6, 12, 18, 24, 30, n_layers - 4, n_layers - 2]
    elif n_layers >= 24:
        test_layers = [2, 6, 12, 18, n_layers - 4, n_layers - 2]
    else:
        test_layers = [2, 4, 8, 12, 16, n_layers - 2]
    log(f"Test layers: {test_layers}")

    results = {}

    for rel_type, config in CONTEXT_PAIRS.items():
        log(f"\n--- Relation type: {rel_type} ---")
        pairs = config["pairs"]
        contexts = config["contexts"]

        rel_results = {"pairs": pairs, "layers": {}}

        for li in test_layers:
            layer_data = {"contexts": {}}

            for ctx_name, ctx_template in contexts.items():
                pair_dists = []

                for pair in pairs:
                    obj1, obj2 = pair

                    # Generate sentences for each concept in the pair
                    try:
                        sent1 = ctx_template.format(obj=obj1, obj1=obj1, obj2=obj2,
                                                     pos=obj1, neg=obj2, attr=obj2, func=obj2)
                        sent2 = ctx_template.format(obj=obj2, obj1=obj2, obj2=obj1,
                                                     pos=obj2, neg=obj1, attr=obj1, func=obj1)
                    except (KeyError, IndexError):
                        # Try simpler substitution
                        sent1 = ctx_template.replace("{obj}", obj1).replace("{obj1}", obj1).replace("{obj2}", obj2).replace("{pos}", obj1).replace("{neg}", obj2).replace("{attr}", obj2).replace("{func}", obj2)
                        sent2 = ctx_template.replace("{obj}", obj2).replace("{obj1}", obj2).replace("{obj2}", obj1).replace("{pos}", obj2).replace("{neg}", obj1).replace("{attr}", obj1).replace("{func}", obj1)

                    h1 = get_rep_at_layer(model, tokenizer, device, sent1, li)
                    h2 = get_rep_at_layer(model, tokenizer, device, sent2, li)

                    if h1 is not None and h2 is not None:
                        n1 = np.linalg.norm(h1)
                        n2 = np.linalg.norm(h2)
                        if n1 > 1e-10 and n2 > 1e-10:
                            cos_sim = np.dot(h1, h2) / (n1 * n2)
                            cos_dist = 1.0 - cos_sim
                            pair_dists.append(cos_dist)

                if pair_dists:
                    layer_data["contexts"][ctx_name] = {
                        "mean_dist": float(np.mean(pair_dists)),
                        "std_dist": float(np.std(pair_dists)),
                        "n_pairs": len(pair_dists),
                    }

            # Random baseline for this layer
            random_dists = []
            for obj1, obj2 in RANDOM_PAIRS:
                sent1 = f"the {obj1} was there"
                sent2 = f"the {obj2} was there"
                h1 = get_rep_at_layer(model, tokenizer, device, sent1, li)
                h2 = get_rep_at_layer(model, tokenizer, device, sent2, li)
                if h1 is not None and h2 is not None:
                    n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_sim = np.dot(h1, h2) / (n1 * n2)
                        random_dists.append(1.0 - cos_sim)

            if random_dists:
                layer_data["random_baseline"] = {
                    "mean_dist": float(np.mean(random_dists)),
                    "std_dist": float(np.std(random_dists)),
                    "n_pairs": len(random_dists),
                }

            # Compute activation ratio for each context vs random
            random_mean = layer_data.get("random_baseline", {}).get("mean_dist", 1.0)
            for ctx_name, ctx_data in layer_data["contexts"].items():
                if random_mean > 1e-10:
                    ctx_data["ratio_vs_random"] = random_mean / ctx_data["mean_dist"] if ctx_data["mean_dist"] > 1e-10 else 0
                else:
                    ctx_data["ratio_vs_random"] = 0

            rel_results["layers"][str(li)] = layer_data

            # Log
            parts = [f"L{li}:"]
            for ctx_name, ctx_data in layer_data["contexts"].items():
                ratio = ctx_data.get("ratio_vs_random", 0)
                parts.append(f"{ctx_name}={ctx_data['mean_dist']:.3f}(r={ratio:.2f})")
            random_info = layer_data.get("random_baseline", {})
            if random_info:
                parts.append(f"random={random_info['mean_dist']:.3f}")
            log("  " + " ".join(parts))

        results[rel_type] = rel_results

    return results


# =====================================================================
# PART B: RELATION-LEVEL CAUSAL TEST
# =====================================================================

def run_causal_relation(model, tokenizer, device, model_info):
    """Test whether relation directions have causal efficacy."""
    log("=" * 60)
    log("PART B: Relation-Level Causal Test")
    log("=" * 60)

    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Use mid-layer for injection (where relation structure is strongest per Phase 314)
    if n_layers >= 36:
        inject_layer = 12
        read_layers = [6, 12, 18, 24, n_layers - 2]
    elif n_layers >= 24:
        inject_layer = 8
        read_layers = [4, 8, 12, 16, n_layers - 2]
    else:
        inject_layer = 8
        read_layers = [4, 8, 12, 16, n_layers - 2]

    log(f"Inject layer: {inject_layer}, Read layers: {read_layers}")

    results = {}

    for rel_type, config in CAUSAL_STIMULI.items():
        log(f"\n--- Causal test: {rel_type} ---")

        # Step 1: Extract relation direction
        # relation_dir = mean(h_related - h_base) - mean(h_unrelated - h_base)
        base_sent = config["base"]
        related_sents = config["related"]
        unrelated_sents = config["unrelated"]
        target_tokens = config["target_tokens"]

        log(f"  Base: '{base_sent}'")
        log(f"  Related: {related_sents}")
        log(f"  Unrelated: {unrelated_sents}")

        # Get representations at inject layer
        h_base = get_rep_at_layer(model, tokenizer, device, base_sent, inject_layer)
        if h_base is None:
            log(f"  WARNING: Could not get base representation, skipping")
            continue

        related_vecs = []
        for s in related_sents:
            h = get_rep_at_layer(model, tokenizer, device, s, inject_layer)
            if h is not None:
                related_vecs.append(h - h_base)

        unrelated_vecs = []
        for s in unrelated_sents:
            h = get_rep_at_layer(model, tokenizer, device, s, inject_layer)
            if h is not None:
                unrelated_vecs.append(h - h_base)

        if len(related_vecs) < 2 or len(unrelated_vecs) < 2:
            log(f"  WARNING: Too few vectors (rel={len(related_vecs)}, unrel={len(unrelated_vecs)}), skipping")
            continue

        # Relation direction = mean(related deltas) - mean(unrelated deltas)
        mean_related_delta = np.mean(related_vecs, axis=0)
        mean_unrelated_delta = np.mean(unrelated_vecs, axis=0)
        relation_dir = mean_related_delta - mean_unrelated_delta

        # Also compute: pure related delta, pure unrelated delta
        pure_related_dir = mean_related_delta
        pure_unrelated_dir = mean_unrelated_delta

        dir_norm = np.linalg.norm(relation_dir)
        related_norm = np.linalg.norm(pure_related_dir)
        unrelated_norm = np.linalg.norm(pure_unrelated_dir)

        log(f"  Relation dir norm={dir_norm:.6f}, related_norm={related_norm:.6f}, unrelated_norm={unrelated_norm:.6f}")

        if dir_norm < 1e-10:
            log(f"  WARNING: Relation direction near zero, skipping")
            continue

        # Step 2: Inject direction at inject_layer and measure output effects
        # Use natural scale injection (match the norm of the relation direction)
        # Also test with unit direction for comparison

        # Get W_U for token-level measurement
        from model_utils import get_W_U
        W_U = get_W_U(model, model_info.name)

        # Token IDs for target tokens
        target_ids = {}
        for tok in target_tokens:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                target_ids[tok] = ids[0]

        random_ids = np.random.choice(W_U.shape[0], size=min(20, W_U.shape[0]), replace=False)

        # Test injection at multiple scales
        test_directions = {
            "relation_dir": relation_dir,
            "related_delta": pure_related_dir,
            "unrelated_delta": pure_unrelated_dir,
            # Random baseline
            "random": np.random.randn(d_model).astype(np.float32),
        }

        # Normalize: use natural scale and unit scale
        injection_results = {}

        for dir_name, direction in test_directions.items():
            dir_norm_val = np.linalg.norm(direction)
            if dir_norm_val < 1e-10:
                continue

            # Natural scale injection (1x and 0.5x)
            for scale in [0.5, 1.0]:
                scale_label = f"{scale}x"
                scaled_dir = direction * scale
                scaled_norm = np.linalg.norm(scaled_dir)

                # W_U gain
                wu_proj = W_U @ scaled_dir  # [vocab_size]
                wu_gain = float(np.linalg.norm(wu_proj) / scaled_norm) if scaled_norm > 1e-10 else 0

                # Target token logit effects
                target_effects = {}
                for tok, tok_id in target_ids.items():
                    target_effects[tok] = float(wu_proj[tok_id])

                # Random token baseline
                random_effects = [float(wu_proj[rid]) for rid in random_ids]
                random_mean = float(np.mean(np.abs(random_effects)))
                random_max = float(np.max(np.abs(random_effects)))

                key = f"{dir_name}_{scale_label}"
                injection_results[key] = {
                    "dir_norm": float(dir_norm_val),
                    "scaled_norm": float(scaled_norm),
                    "wu_gain": wu_gain,
                    "target_effects": target_effects,
                    "random_mean_abs_effect": random_mean,
                    "random_max_abs_effect": random_max,
                    "target_vs_random_ratio": {tok: abs(v) / random_mean if random_mean > 1e-10 else 0
                                               for tok, v in target_effects.items()},
                }

        # Step 3: Hook-based causal injection test
        # Inject direction at inject_layer, read output at read_layers
        log(f"  Running hook-based causal injection at L{inject_layer}...")

        hook_results = {}
        layers = get_layers(model)

        for dir_name, direction in [("relation_dir", relation_dir), ("random", test_directions["random"])]:
            dir_norm_val = np.linalg.norm(direction)
            if dir_norm_val < 1e-10:
                continue

            # Use 1x natural scale for relation_dir, match norm for random
            if dir_name == "random":
                inject_vec = direction / np.linalg.norm(direction) * dir_norm_val
            else:
                inject_vec = direction

            for read_li in read_layers:
                # Capture representations at read layer
                captured_clean = {}
                captured_inject = {}

                def make_hook(capture_dict, name):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            capture_dict[name] = output[0].detach()
                        else:
                            capture_dict[name] = output.detach()
                    return hook_fn

                # Clean run
                read_layer = layers[read_li]
                h_read = read_layer.register_forward_hook(make_hook(captured_clean, "h"))

                inp = tokenizer(base_sent, return_tensors="pt").to(device)
                with torch.no_grad():
                    model(**inp)
                h_read.remove()

                # Injected run: inject direction at inject_layer
                inject_layer_obj = layers[inject_layer]
                inject_vec_tensor = torch.tensor(inject_vec, dtype=torch.bfloat16, device=device)

                captured_inject_at_inject = {}
                def inject_hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].clone()
                        h[0, -1, :] += inject_vec_tensor.to(h.dtype)
                        captured_inject_at_inject["h"] = h
                        return (h,) + output[1:]
                    else:
                        h = output.clone()
                        h[0, -1, :] += inject_vec_tensor.to(h.dtype)
                        captured_inject_at_inject["h"] = h
                        return h

                h_inject = inject_layer_obj.register_forward_hook(inject_hook)
                h_read2 = read_layer.register_forward_hook(make_hook(captured_inject, "h"))

                with torch.no_grad():
                    model(**inp)

                h_inject.remove()
                h_read2.remove()

                # Compute effect at read layer
                if "h" in captured_clean and "h" in captured_inject:
                    h_clean = captured_clean["h"][0, -1].cpu().float().numpy()
                    h_injected = captured_inject["h"][0, -1].cpu().float().numpy()
                    delta_h = h_injected - h_clean

                    # W_U projection of delta
                    delta_logits = W_U @ delta_h  # [vocab_size]

                    # Target token effects
                    target_deltas = {}
                    for tok, tok_id in target_ids.items():
                        target_deltas[tok] = float(delta_logits[tok_id])

                    # Random baseline
                    random_deltas = [float(delta_logits[rid]) for rid in random_ids]
                    random_abs_mean = float(np.mean(np.abs(random_deltas)))

                    key = f"{dir_name}_L{read_li}"
                    hook_results[key] = {
                        "delta_h_norm": float(np.linalg.norm(delta_h)),
                        "target_logit_deltas": target_deltas,
                        "random_abs_mean": random_abs_mean,
                        "target_vs_random": {tok: abs(v) / random_abs_mean if random_abs_mean > 1e-10 else 0
                                             for tok, v in target_deltas.items()},
                    }

        rel_causal = {
            "inject_layer": inject_layer,
            "read_layers": read_layers,
            "relation_dir_norm": float(dir_norm),
            "related_delta_norm": float(related_norm),
            "unrelated_delta_norm": float(unrelated_norm),
            "injection_WU_results": injection_results,
            "hook_causal_results": hook_results,
        }

        # Log summary
        log(f"  W_U injection results:")
        for key, data in injection_results.items():
            top_target = max(data["target_vs_random_ratio"].items(), key=lambda x: x[1]) if data["target_vs_random_ratio"] else ("N/A", 0)
            log(f"    {key}: wu_gain={data['wu_gain']:.3f}, top_target={top_target[0]}({top_target[1]:.2f}x random)")

        log(f"  Hook causal results:")
        for key, data in hook_results.items():
            top_target = max(data["target_vs_random"].items(), key=lambda x: x[1]) if data["target_vs_random"] else ("N/A", 0)
            log(f"    {key}: delta_norm={data['delta_h_norm']:.4f}, top_target={top_target[0]}({top_target[1]:.2f}x random)")

        results[rel_type] = rel_causal

    return results


# =====================================================================
# MAIN
# =====================================================================

def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase315_{model_name}.log")

    log(f"=== Phase 315: Context Activation + Causal Relation for {model_name} ===")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"Model: n_layers={info.n_layers}, d_model={info.d_model}")

    # Part A: Context Activation
    t0 = time.time()
    context_results = run_context_activation(model, tokenizer, device, info)
    t_ctx = time.time() - t0
    log(f"Part A completed in {t_ctx:.1f}s")

    # Part B: Causal Relation
    t0 = time.time()
    causal_results = run_causal_relation(model, tokenizer, device, info)
    t_causal = time.time() - t0
    log(f"Part B completed in {t_causal:.1f}s")

    # Save results
    all_results = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "part_a_context_activation": context_results,
        "part_b_causal_relation": causal_results,
        "timing": {
            "context_test_s": round(t_ctx, 1),
            "causal_test_s": round(t_causal, 1),
        }
    }

    out_path = RESULT_DIR / f"{model_name}_context_causal.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # Print summary
    log("\n" + "=" * 70)
    log(f"PHASE 315 SUMMARY - {model_name}")
    log("=" * 70)

    log("\nPart A: Context Activation (ratio_vs_random > 1 means relation preserved)")
    for rel_type, rel_data in context_results.items():
        log(f"\n  {rel_type}:")
        # Find best layer and context
        best_ratio = 0
        best_config = ""
        for li_str, li_data in rel_data.get("layers", {}).items():
            for ctx_name, ctx_data in li_data.get("contexts", {}).items():
                ratio = ctx_data.get("ratio_vs_random", 0)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_config = f"L{li_str}/{ctx_name}"
        log(f"    Best: {best_config} = {best_ratio:.2f}x")

        # Show mid-layer detail
        mid_layer = str(info.n_layers // 3)
        if mid_layer in rel_data.get("layers", {}):
            ml = rel_data["layers"][mid_layer]
            for ctx_name, ctx_data in ml.get("contexts", {}).items():
                ratio = ctx_data.get("ratio_vs_random", 0)
                log(f"    L{mid_layer} {ctx_name}: dist={ctx_data['mean_dist']:.3f}, ratio={ratio:.2f}")

    log("\nPart B: Causal Relation (target_vs_random > 1 means causal efficacy)")
    for rel_type, rel_data in causal_results.items():
        log(f"\n  {rel_type}:")
        log(f"    relation_dir_norm={rel_data['relation_dir_norm']:.6f}")

        # Best hook causal result
        best_causal = 0
        best_causal_config = ""
        for key, data in rel_data.get("hook_causal_results", {}).items():
            if key.startswith("relation_dir"):
                for tok, ratio in data.get("target_vs_random", {}).items():
                    if ratio > best_causal:
                        best_causal = ratio
                        best_causal_config = f"{key}/{tok}"
        log(f"    Best causal: {best_causal_config} = {best_causal:.2f}x random")

    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released.")


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
                import traceback
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(10)
    else:
        run_model(model_name)
