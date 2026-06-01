"""
Phase 317: Attribute Fix + Context×Causal Interaction + Negation Expansion
===========================================================================

Three tests addressing Phase 315-R2 hard issues:

Test 1: Attribute Activation with PARALLEL Templates (50 pairs, 6 layers)
  - FIX: Use parallel templates for both obj and attr words
  - Templates: static, attribute_probe, attribute_fill
  - 50 real pairs + 50 random baseline pairs

Test 2: Context × Causal Interaction (CRITICAL NEW TEST)
  - Same attribute direction injected into different contexts
  - Compare causal efficacy: attribute context vs static context
  - If ratio > 1 → context gating opens relation pathways (proves conditional encoding)

Test 3: Expanded Negation with Controls (25 pairs)
  - 10 regular negation, 5 double negation, 5 weak negation, 5 more regular
  - Norm-matched random direction controls
  - Antonym direction comparison
  - Distinguish logical negation from semantic polarity

Usage:
  python tests/glm5/phase317_comprehensive.py qwen3
  python tests/glm5/phase317_comprehensive.py glm4
  python tests/glm5/phase317_comprehensive.py deepseek7b
"""
import sys, os, gc, time, json, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model, get_W_U

RESULT_DIR = Path("results/phase317_comprehensive")
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

# Test 1: 50 attribute pairs with parallel templates
ATTR_PAIRS = [
    # Color
    ("apple", "red"), ("banana", "yellow"), ("sky", "blue"), ("grass", "green"),
    ("snow", "white"), ("night", "dark"), ("fire", "orange"), ("sunset", "golden"),
    ("lemon", "yellow"), ("rose", "red"),
    # Temperature
    ("ice", "cold"), ("fire", "hot"), ("summer", "warm"), ("winter", "cold"),
    ("desert", "hot"), ("arctic", "freezing"),
    # Texture
    ("silk", "smooth"), ("sandpaper", "rough"), ("glass", "smooth"), ("stone", "hard"),
    ("cotton", "soft"), ("steel", "hard"), ("feather", "light"), ("lead", "heavy"),
    # Taste
    ("lemon", "sour"), ("honey", "sweet"), ("salt", "salty"), ("chili", "spicy"),
    ("coffee", "bitter"), ("sugar", "sweet"),
    # Size
    ("mountain", "large"), ("ant", "small"), ("elephant", "huge"), ("mouse", "tiny"),
    # Shape
    ("ball", "round"), ("sword", "sharp"), ("plate", "flat"), ("needle", "pointed"),
    # Speed
    ("cheetah", "fast"), ("turtle", "slow"), ("rocket", "fast"), ("snail", "slow"),
    # Sound
    ("thunder", "loud"), ("whisper", "quiet"), ("drum", "loud"), ("breeze", "gentle"),
    # More
    ("diamond", "hard"), ("cloud", "soft"), ("iron", "strong"), ("thread", "thin"),
    ("forest", "dense"), ("lake", "calm"),
]

RANDOM_PAIRS = [
    ("apple", "hammer"), ("dog", "key"), ("knife", "river"), ("car", "happy"),
    ("table", "cold"), ("fire", "cat"), ("silk", "bus"), ("book", "lemon"),
    ("lamp", "eagle"), ("door", "sweet"), ("stone", "shovel"), ("snow", "sword"),
    ("feather", "bridge"), ("honey", "chair"), ("phone", "sky"), ("rose", "pen"),
    ("lock", "hawk"), ("boat", "sour"), ("cup", "lily"), ("shovel", "smooth"),
    ("diamond", "turtle"), ("cloud", "needle"), ("iron", "sunset"), ("thread", "arctic"),
    ("forest", "plate"), ("desert", "ball"), ("storm", "baby"), ("lake", "salt"),
    ("cheetah", "glass"), ("rocket", "sandpaper"), ("drum", "winter"), ("breeze", "steel"),
    ("mountain", "silk"), ("ant", "thunder"), ("elephant", "whisper"), ("mouse", "diamond"),
    ("ball", "cotton"), ("sword", "honey"), ("plate", "cheetah"), ("needle", "lake"),
    ("turtle", "coffee"), ("snail", "iron"), ("sugar", "sword"), ("salt", "cloud"),
    ("summer", "needle"), ("winter", "sugar"), ("sandpaper", "breeze"), ("glass", "storm"),
]

# PARALLEL templates — both words in the pair go through the SAME template
ATTR_TEMPLATES = {
    "static": "the {w} was there",
    "attribute_probe": "The {w} is usually",
    "attribute_fill": "The {w} has the quality of being",
}


# Test 2: Context × Causal interaction pairs
CONTEXT_CAUSAL_PAIRS = [
    ("apple", "red", ["red", "sweet", "ripe", "green", "yellow"]),
    ("knife", "sharp", ["sharp", "cutting", "dull", "metal", "blade"]),
    ("ice", "cold", ["cold", "freezing", "frozen", "cool", "chilly"]),
    ("fire", "hot", ["hot", "burning", "warm", "fiery", "blazing"]),
    ("silk", "smooth", ["smooth", "soft", "silky", "slippery", "delicate"]),
    ("stone", "hard", ["hard", "solid", "rocky", "tough", "firm"]),
    ("honey", "sweet", ["sweet", "sticky", "delicious", "sugary", "golden"]),
    ("lemon", "sour", ["sour", "bitter", "tart", "acidic", "citrus"]),
    ("thunder", "loud", ["loud", "noisy", "powerful", "booming", "deafening"]),
    ("cotton", "soft", ["soft", "fluffy", "gentle", "comfortable", "light"]),
    ("cheetah", "fast", ["fast", "quick", "swift", "speedy", "rapid"]),
    ("elephant", "huge", ["huge", "large", "massive", "big", "enormous"]),
    ("diamond", "hard", ["hard", "precious", "brilliant", "shiny", "valuable"]),
    ("snow", "white", ["white", "cold", "pure", "clean", "frozen"]),
    ("night", "dark", ["dark", "black", "shadowy", "dim", "gloomy"]),
]


# Test 3: Expanded negation pairs
NEGATION_PAIRS = {
    "regular": [
        {"pos": "they were very happy about it", "neg": "they were not happy about it",
         "pos_tokens": ["happy", "glad", "pleased"], "neg_tokens": ["not", "unhappy", "disappointed"]},
        {"pos": "the place was very safe", "neg": "the place was not safe",
         "pos_tokens": ["safe", "secure", "protected"], "neg_tokens": ["not", "dangerous", "unsafe"]},
        {"pos": "the result was very good", "neg": "the result was not good",
         "pos_tokens": ["good", "great", "excellent"], "neg_tokens": ["not", "bad", "poor"]},
        {"pos": "the room was very clean", "neg": "the room was not clean",
         "pos_tokens": ["clean", "tidy", "neat"], "neg_tokens": ["not", "dirty", "messy"]},
        {"pos": "the task was very possible", "neg": "the task was not possible",
         "pos_tokens": ["possible", "feasible", "achievable"], "neg_tokens": ["not", "impossible", "unfeasible"]},
        {"pos": "the room was very bright", "neg": "the room was not bright",
         "pos_tokens": ["bright", "light", "illuminated"], "neg_tokens": ["not", "dark", "dim"]},
        {"pos": "the man was very strong", "neg": "the man was not strong",
         "pos_tokens": ["strong", "powerful", "mighty"], "neg_tokens": ["not", "weak", "feeble"]},
        {"pos": "the weather was very warm", "neg": "the weather was not warm",
         "pos_tokens": ["warm", "mild", "pleasant"], "neg_tokens": ["not", "cold", "chilly"]},
        {"pos": "the door was very open", "neg": "the door was not open",
         "pos_tokens": ["open", "accessible", "unlocked"], "neg_tokens": ["not", "closed", "shut"]},
        {"pos": "the situation was very fair", "neg": "the situation was not fair",
         "pos_tokens": ["fair", "just", "equitable"], "neg_tokens": ["not", "unfair", "biased"]},
    ],
    "double_negation": [
        {"pos": "the result was very bad", "neg": "the result was not bad",
         "pos_tokens": ["bad", "poor", "terrible"], "neg_tokens": ["not", "okay", "acceptable"]},
        {"pos": "the task was very impossible", "neg": "the task was not impossible",
         "pos_tokens": ["impossible", "unfeasible", "hopeless"], "neg_tokens": ["not", "possible", "feasible"]},
        {"pos": "the answer was very wrong", "neg": "the answer was not wrong",
         "pos_tokens": ["wrong", "incorrect", "false"], "neg_tokens": ["not", "correct", "right"]},
        {"pos": "the outcome was very unlikely", "neg": "the outcome was not unlikely",
         "pos_tokens": ["unlikely", "improbable", "doubtful"], "neg_tokens": ["not", "likely", "probable"]},
        {"pos": "the person was very unhappy", "neg": "the person was not unhappy",
         "pos_tokens": ["unhappy", "sad", "miserable"], "neg_tokens": ["not", "happy", "okay"]},
    ],
    "weak_negation": [
        {"pos": "the movie was very great", "neg": "the movie was not great",
         "pos_tokens": ["great", "excellent", "amazing"], "neg_tokens": ["not", "okay", "mediocre"]},
        {"pos": "the food was very terrible", "neg": "the food was not terrible",
         "pos_tokens": ["terrible", "awful", "horrible"], "neg_tokens": ["not", "okay", "decent"]},
        {"pos": "the plan was very perfect", "neg": "the plan was not perfect",
         "pos_tokens": ["perfect", "ideal", "flawless"], "neg_tokens": ["not", "okay", "imperfect"]},
        {"pos": "the design was very horrible", "neg": "the design was not horrible",
         "pos_tokens": ["horrible", "terrible", "awful"], "neg_tokens": ["not", "okay", "decent"]},
        {"pos": "the idea was very amazing", "neg": "the idea was not amazing",
         "pos_tokens": ["amazing", "wonderful", "incredible"], "neg_tokens": ["not", "okay", "ordinary"]},
    ],
}


# =====================================================================
# MODEL LOADING
# =====================================================================

def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]

    # Try flash_attention_2 first, fall back to sdpa
    attn_impl = "flash_attention_2"
    log(f"Loading {model_name} (bf16 + device_map=auto + {attn_impl})...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=attn_impl,
        )
        log(f"  Loaded with {attn_impl}")
    except Exception as e:
        log(f"  flash_attention_2 failed ({e}), falling back to sdpa")
        attn_impl = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=attn_impl,
        )
        log(f"  Loaded with {attn_impl}")

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
# UTILITY: Multi-layer representation extraction
# =====================================================================

def get_reps_at_layers(model, tokenizer, device, sentence, layer_indices):
    """Get last-token representations at multiple layers in ONE forward pass."""
    layers = get_layers(model)
    captured = {}

    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[li] = output[0].detach()
            else:
                captured[li] = output.detach()
        return hook_fn

    hooks = [layers[li].register_forward_hook(make_hook(li)) for li in layer_indices]

    inp = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        model(**inp)

    for h in hooks:
        h.remove()

    result = {}
    for li in layer_indices:
        if li in captured:
            result[li] = captured[li][0, -1].cpu().float().numpy()
    return result


def inject_and_read(model, tokenizer, device, sentence, inject_layer, inject_vec,
                    read_layers, W_U_np, target_token_ids, random_token_ids):
    """Inject direction at inject_layer, read effects at read_layers.
    Returns dict of results per read layer."""
    layers = get_layers(model)

    # Step 1: Clean run - capture at all read layers
    captured_clean = {}
    def make_clean_hook(rl):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured_clean[rl] = output[0].detach()
            else:
                captured_clean[rl] = output.detach()
        return hook_fn

    hooks_clean = [layers[rl].register_forward_hook(make_clean_hook(rl)) for rl in read_layers]

    inp = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        model(**inp)
    for h in hooks_clean:
        h.remove()

    # Step 2: Injected run
    captured_inject = {}
    inject_tensor = torch.tensor(inject_vec, dtype=torch.bfloat16, device=device)

    def inject_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            h[0, -1, :] += inject_tensor.to(h.dtype)
            return (h,) + output[1:]
        else:
            h = output.clone()
            h[0, -1, :] += inject_tensor.to(h.dtype)
            return h

    def make_inject_read_hook(rl):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured_inject[rl] = output[0].detach()
            else:
                captured_inject[rl] = output.detach()
        return hook_fn

    h_inject = layers[inject_layer].register_forward_hook(inject_hook)
    hooks_read = [layers[rl].register_forward_hook(make_inject_read_hook(rl)) for rl in read_layers]

    with torch.no_grad():
        model(**inp)

    h_inject.remove()
    for h in hooks_read:
        h.remove()

    # Step 3: Compute effects
    results = {}
    for rl in read_layers:
        if rl in captured_clean and rl in captured_inject:
            h_clean = captured_clean[rl][0, -1].cpu().float().numpy()
            h_injected = captured_inject[rl][0, -1].cpu().float().numpy()
            delta_h = h_injected - h_clean

            # W_U projection
            delta_logits = W_U_np @ delta_h

            # Target token effects
            target_effects = {}
            for tok, tid in target_token_ids.items():
                target_effects[tok] = float(delta_logits[tid])

            # Random baseline
            random_effects = [float(delta_logits[rid]) for rid in random_token_ids]
            random_abs_mean = float(np.mean(np.abs(random_effects)))

            # Ratio vs random
            target_vs_random = {}
            for tok, v in target_effects.items():
                target_vs_random[tok] = abs(v) / random_abs_mean if random_abs_mean > 1e-10 else 0

            results[f"L{rl}"] = {
                "delta_h_norm": float(np.linalg.norm(delta_h)),
                "target_effects": target_effects,
                "random_abs_mean": random_abs_mean,
                "target_vs_random": target_vs_random,
            }

    return results


# =====================================================================
# TEST 1: ATTRIBUTE ACTIVATION WITH PARALLEL TEMPLATES
# =====================================================================

def test1_attribute_parallel(model, tokenizer, device, model_info):
    log("=" * 60)
    log("TEST 1: Attribute Activation with PARALLEL Templates (50 pairs)")
    log("=" * 60)

    n_layers = model_info.n_layers
    # Ensure deep layers included
    if n_layers >= 36:
        test_layers = [2, 6, 12, 18, 24, n_layers - 2]
    elif n_layers >= 24:
        test_layers = [2, 6, 12, 18, n_layers - 2]
    else:
        test_layers = [2, 4, 8, 12, 16, n_layers - 2]
    log(f"Test layers: {test_layers}")

    results = {}
    n_total = len(ATTR_PAIRS) * 3 * 2 + len(RANDOM_PAIRS)  # 3 templates, 2 words per pair
    n_done = 0

    for tmpl_name, tmpl in ATTR_TEMPLATES.items():
        tmpl_results = {}

        for li in test_layers:
            # Real attribute pairs
            real_dists = []
            for obj, attr in ATTR_PAIRS:
                sent1 = tmpl.replace("{w}", obj)
                sent2 = tmpl.replace("{w}", attr)
                reps = get_reps_at_layers(model, tokenizer, device, sent1, [li])
                reps2 = get_reps_at_layers(model, tokenizer, device, sent2, [li])
                if li in reps and li in reps2:
                    h1, h2 = reps[li], reps2[li]
                    n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        real_dists.append(1.0 - np.dot(h1, h2) / (n1 * n2))
                n_done += 2
                if n_done % 50 == 0:
                    log(f"  [{tmpl_name}] L{li}: {n_done}/{n_total} done")

            # Random baseline pairs
            random_dists = []
            for obj, attr in RANDOM_PAIRS:
                sent1 = tmpl.replace("{w}", obj)
                sent2 = tmpl.replace("{w}", attr)
                reps = get_reps_at_layers(model, tokenizer, device, sent1, [li])
                reps2 = get_reps_at_layers(model, tokenizer, device, sent2, [li])
                if li in reps and li in reps2:
                    h1, h2 = reps[li], reps2[li]
                    n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        random_dists.append(1.0 - np.dot(h1, h2) / (n1 * n2))
                n_done += 2

            real_mean = np.mean(real_dists) if real_dists else 0
            random_mean = np.mean(random_dists) if random_dists else 0
            ratio = random_mean / real_mean if real_mean > 1e-10 else 0

            tmpl_results[str(li)] = {
                "real_mean_dist": float(real_mean),
                "real_std_dist": float(np.std(real_dists)) if real_dists else 0,
                "random_mean_dist": float(random_mean),
                "random_std_dist": float(np.std(random_dists)) if random_dists else 0,
                "ratio_vs_random": float(ratio),
                "n_real": len(real_dists),
                "n_random": len(random_dists),
            }

            log(f"  L{li} {tmpl_name}: real={real_mean:.4f}, random={random_mean:.4f}, ratio={ratio:.2f}")

        results[tmpl_name] = tmpl_results

        # Memory cleanup
        torch.cuda.empty_cache()

    return results


# =====================================================================
# TEST 2: CONTEXT × CAUSAL INTERACTION
# =====================================================================

def test2_context_causal(model, tokenizer, device, model_info):
    log("=" * 60)
    log("TEST 2: Context × Causal Interaction (CRITICAL)")
    log("=" * 60)

    n_layers = model_info.n_layers
    d_model = model_info.d_model

    if n_layers >= 36:
        inject_layer = 12
        read_layers = [12, 18, 24, n_layers - 2]
    elif n_layers >= 24:
        inject_layer = 8
        read_layers = [8, 12, 16, n_layers - 2]
    else:
        inject_layer = 8
        read_layers = [8, 12, 16, n_layers - 2]

    log(f"Inject layer: {inject_layer}, Read layers: {read_layers}")

    W_U = get_W_U(model, model_info.name)

    results = {}

    for pair_idx, (obj, attr, target_toks) in enumerate(CONTEXT_CAUSAL_PAIRS):
        log(f"\n  Pair {pair_idx+1}/{len(CONTEXT_CAUSAL_PAIRS)}: ({obj}, {attr})")

        # Step 1: Extract attribute direction
        # Direction = "the {attr} {obj} was very" - "the {obj} was very" at inject_layer
        sent_with_attr = f"the {attr} {obj} was very"
        sent_without = f"the {obj} was very"

        reps_with = get_reps_at_layers(model, tokenizer, device, sent_with_attr, [inject_layer])
        reps_without = get_reps_at_layers(model, tokenizer, device, sent_without, [inject_layer])

        if inject_layer not in reps_with or inject_layer not in reps_without:
            log(f"  WARNING: Could not extract representations, skipping")
            continue

        attr_dir = reps_with[inject_layer] - reps_without[inject_layer]
        attr_dir_norm = np.linalg.norm(attr_dir)
        log(f"  attr_dir_norm={attr_dir_norm:.4f}")

        if attr_dir_norm < 1e-10:
            continue

        # Norm-matched random direction
        np.random.seed(pair_idx * 100 + 42)
        random_dir = np.random.randn(d_model).astype(np.float32)
        random_dir = random_dir / np.linalg.norm(random_dir) * attr_dir_norm

        # Token IDs for targets
        target_ids = {}
        for tok in target_toks:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                target_ids[tok] = ids[0]
        # "not" token
        not_ids = tokenizer.encode("not", add_special_tokens=False)
        if not_ids:
            target_ids["not"] = not_ids[0]

        random_ids = np.random.choice(W_U.shape[0], size=min(30, W_U.shape[0]), replace=False)

        # Step 2: Inject into STATIC context
        static_sent = f"the {obj} was there"
        log(f"  Injecting into STATIC: '{static_sent}'")
        static_attr_results = inject_and_read(
            model, tokenizer, device, static_sent, inject_layer, attr_dir,
            read_layers, W_U, target_ids, random_ids
        )
        static_random_results = inject_and_read(
            model, tokenizer, device, static_sent, inject_layer, random_dir,
            read_layers, W_U, target_ids, random_ids
        )

        # Step 3: Inject into ATTRIBUTE context
        attr_sent = f"The {obj} is usually"
        log(f"  Injecting into ATTRIBUTE: '{attr_sent}'")
        attr_attr_results = inject_and_read(
            model, tokenizer, device, attr_sent, inject_layer, attr_dir,
            read_layers, W_U, target_ids, random_ids
        )
        attr_random_results = inject_and_read(
            model, tokenizer, device, attr_sent, inject_layer, random_dir,
            read_layers, W_U, target_ids, random_ids
        )

        # Step 4: Compute interaction ratio
        pair_result = {
            "obj": obj, "attr": attr,
            "attr_dir_norm": float(attr_dir_norm),
            "static_context": static_attr_results,
            "static_random": static_random_results,
            "attribute_context": attr_attr_results,
            "attribute_random": attr_random_results,
            "interaction_ratios": {},
        }

        # For each read layer, compute ratio: (effect in attr ctx) / (effect in static ctx)
        for rl_key in static_attr_results:
            if rl_key in attr_attr_results:
                static_top = max(static_attr_results[rl_key].get("target_vs_random", {}).values(), default=0)
                attr_top = max(attr_attr_results[rl_key].get("target_vs_random", {}).values(), default=0)
                interaction_ratio = attr_top / static_top if static_top > 0.01 else 0

                # Also for random direction
                static_r_top = max(static_random_results.get(rl_key, {}).get("target_vs_random", {}).values(), default=0)
                attr_r_top = max(attr_random_results.get(rl_key, {}).get("target_vs_random", {}).values(), default=0)
                random_interaction_ratio = attr_r_top / static_r_top if static_r_top > 0.01 else 0

                pair_result["interaction_ratios"][rl_key] = {
                    "attr_dir_ratio": float(interaction_ratio),
                    "random_dir_ratio": float(random_interaction_ratio),
                    "static_attr_top": float(static_top),
                    "attribute_attr_top": float(attr_top),
                }

                log(f"    {rl_key}: attr_dir_interaction={interaction_ratio:.2f}, "
                    f"random_interaction={random_interaction_ratio:.2f}, "
                    f"static_top={static_top:.2f}, attr_top={attr_top:.2f}")

        results[f"{obj}_{attr}"] = pair_result

        # Memory cleanup
        torch.cuda.empty_cache()

    return results


# =====================================================================
# TEST 3: EXPANDED NEGATION WITH CONTROLS
# =====================================================================

def test3_negation_expanded(model, tokenizer, device, model_info):
    log("=" * 60)
    log("TEST 3: Expanded Negation with Controls (25 pairs)")
    log("=" * 60)

    n_layers = model_info.n_layers
    d_model = model_info.d_model

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

    W_U = get_W_U(model, model_info.name)
    results = {}
    pair_idx = 0

    for neg_type, pairs in NEGATION_PAIRS.items():
        log(f"\n--- Negation type: {neg_type} ---")
        type_results = {}

        for pidx, pair_data in enumerate(pairs):
            pos_sent = pair_data["pos"]
            neg_sent = pair_data["neg"]
            pos_tokens = pair_data["pos_tokens"]
            neg_tokens = pair_data["neg_tokens"]

            log(f"  Pair {pidx+1}/{len(pairs)}: '{pos_sent}' vs '{neg_sent}'")

            # Extract negation direction at inject layer
            reps_pos = get_reps_at_layers(model, tokenizer, device, pos_sent, [inject_layer])
            reps_neg = get_reps_at_layers(model, tokenizer, device, neg_sent, [inject_layer])

            if inject_layer not in reps_pos or inject_layer not in reps_neg:
                log(f"  WARNING: Could not extract reps, skipping")
                continue

            h_pos = reps_pos[inject_layer]
            h_neg = reps_neg[inject_layer]
            neg_dir = h_neg - h_pos
            neg_dir_norm = np.linalg.norm(neg_dir)

            # Unit direction
            neg_dir_unit = neg_dir / neg_dir_norm if neg_dir_norm > 1e-10 else neg_dir

            # Norm-matched random direction
            np.random.seed(pair_idx * 1000 + pidx * 100 + 42)
            random_dir = np.random.randn(d_model).astype(np.float32)
            random_dir = random_dir / np.linalg.norm(random_dir) * neg_dir_norm

            # Antonym direction: for "happy", antonym direction = h("sad") - h("happy")
            # Extract from simple sentences
            pos_word = pos_tokens[0]  # e.g., "happy"
            neg_word = neg_tokens[1] if len(neg_tokens) > 1 else neg_tokens[0]  # e.g., "unhappy"
            if neg_word == "not" and len(neg_tokens) > 1:
                neg_word = neg_tokens[1]

            antonym_dir = None
            antonym_dir_norm = 0
            sent_pos_word = f"they were very {pos_word}"
            sent_neg_word = f"they were very {neg_word}"
            reps_pw = get_reps_at_layers(model, tokenizer, device, sent_pos_word, [inject_layer])
            reps_nw = get_reps_at_layers(model, tokenizer, device, sent_neg_word, [inject_layer])
            if inject_layer in reps_pw and inject_layer in reps_nw:
                antonym_dir = reps_nw[inject_layer] - reps_pw[inject_layer]
                antonym_dir_norm = np.linalg.norm(antonym_dir)
                # Scale to same norm as neg_dir for fair comparison
                if antonym_dir_norm > 1e-10:
                    antonym_dir = antonym_dir / antonym_dir_norm * neg_dir_norm

            log(f"  neg_dir_norm={neg_dir_norm:.4f}, antonym_dir_norm={antonym_dir_norm:.4f}")

            if neg_dir_norm < 1e-10:
                continue

            # Token IDs
            all_tokens = list(set(pos_tokens + neg_tokens))
            target_ids = {}
            for tok in all_tokens:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                if ids:
                    target_ids[tok] = ids[0]

            random_ids = np.random.choice(W_U.shape[0], size=min(30, W_U.shape[0]), replace=False)

            # Test three directions: neg_dir, random_dir, antonym_dir
            pair_result = {
                "pos": pos_sent, "neg": neg_sent,
                "neg_type": neg_type,
                "neg_dir_norm": float(neg_dir_norm),
                "antonym_dir_norm": float(antonym_dir_norm),
                "neg_dir_causal": {},
                "random_dir_causal": {},
                "antonym_dir_causal": {},
            }

            # Inject negation direction
            neg_causal = inject_and_read(
                model, tokenizer, device, pos_sent, inject_layer, neg_dir,
                read_layers, W_U, target_ids, random_ids
            )
            pair_result["neg_dir_causal"] = neg_causal

            # Inject random direction (norm-matched)
            random_causal = inject_and_read(
                model, tokenizer, device, pos_sent, inject_layer, random_dir,
                read_layers, W_U, target_ids, random_ids
            )
            pair_result["random_dir_causal"] = random_causal

            # Inject antonym direction (norm-matched to neg_dir)
            if antonym_dir is not None and antonym_dir_norm > 1e-10:
                antonym_causal = inject_and_read(
                    model, tokenizer, device, pos_sent, inject_layer, antonym_dir,
                    read_layers, W_U, target_ids, random_ids
                )
                pair_result["antonym_dir_causal"] = antonym_causal

            # Compute neg_dir vs random selectivity
            for rl_key in neg_causal:
                neg_top = max(neg_causal[rl_key].get("target_vs_random", {}).items(),
                             key=lambda x: x[1], default=("N/A", 0))
                rand_top = max(random_causal.get(rl_key, {}).get("target_vs_random", {}).items(),
                              key=lambda x: x[1], default=("N/A", 0))
                antonym_top_val = 0
                antonym_top_tok = "N/A"
                if rl_key in pair_result["antonym_dir_causal"]:
                    antonym_top = max(pair_result["antonym_dir_causal"][rl_key].get("target_vs_random", {}).items(),
                                     key=lambda x: x[1], default=("N/A", 0))
                    antonym_top_tok, antonym_top_val = antonym_top

                log(f"    {rl_key}: neg_top={neg_top[0]}({neg_top[1]:.2f}x), "
                    f"rand_top={rand_top[0]}({rand_top[1]:.2f}x), "
                    f"antonym_top={antonym_top_tok}({antonym_top_val:.2f}x)")

            # Check double negation effect: for double negation, neg_tokens should include positive words
            if neg_type == "double_negation":
                # The negation of "bad" → "not bad" should activate positive words
                # Check if neg_dir from "not bad" sentence activates "okay", "acceptable"
                for rl_key in neg_causal:
                    pos_token_effects = {tok: neg_causal[rl_key].get("target_vs_random", {}).get(tok, 0)
                                        for tok in pos_tokens if tok in target_ids}
                    neg_token_effects = {tok: neg_causal[rl_key].get("target_vs_random", {}).get(tok, 0)
                                        for tok in neg_tokens if tok in target_ids}
                    log(f"    {rl_key} double_neg: pos_effects={pos_token_effects}, neg_effects={neg_token_effects}")

            type_results[f"pair_{pidx}"] = pair_result
            pair_idx += 1

            torch.cuda.empty_cache()

        results[neg_type] = type_results

    return results


# =====================================================================
# MAIN
# =====================================================================

def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase317_{model_name}.log")

    log(f"=== Phase 317: Comprehensive Test for {model_name} ===")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"Model: n_layers={info.n_layers}, d_model={info.d_model}")

    # Test 1: Attribute with parallel templates
    t0 = time.time()
    test1_results = test1_attribute_parallel(model, tokenizer, device, info)
    t1 = time.time()
    log(f"Test 1 completed in {t1-t0:.1f}s")

    # Test 2: Context × Causal interaction
    t0 = time.time()
    test2_results = test2_context_causal(model, tokenizer, device, info)
    t1 = time.time()
    log(f"Test 2 completed in {t1-t0:.1f}s")

    # Test 3: Expanded negation
    t0 = time.time()
    test3_results = test3_negation_expanded(model, tokenizer, device, info)
    t1 = time.time()
    log(f"Test 3 completed in {t1-t0:.1f}s")

    # Save results
    all_results = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "test1_attribute_parallel": test1_results,
        "test2_context_causal": test2_results,
        "test3_negation_expanded": test3_results,
    }

    out_path = RESULT_DIR / f"{model_name}_phase317.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # ========== SUMMARY ==========
    log("\n" + "=" * 70)
    log(f"PHASE 317 SUMMARY - {model_name}")
    log("=" * 70)

    # Test 1 Summary
    log("\nTest 1: Attribute Activation (PARALLEL templates)")
    for tmpl_name, tmpl_data in test1_results.items():
        log(f"  Template: {tmpl_name}")
        for li_str, li_data in sorted(tmpl_data.items(), key=lambda x: int(x[0])):
            ratio = li_data["ratio_vs_random"]
            log(f"    L{li_str}: ratio={ratio:.2f} (real={li_data['real_mean_dist']:.4f}, "
                f"random={li_data['random_mean_dist']:.4f}, n={li_data['n_real']})")

    # Test 2 Summary
    log("\nTest 2: Context × Causal Interaction")
    interaction_ratios = []
    for pair_key, pair_data in test2_results.items():
        for rl_key, rl_data in pair_data.get("interaction_ratios", {}).items():
            ratio = rl_data.get("attr_dir_ratio", 0)
            interaction_ratios.append(ratio)
            log(f"  {pair_key}/{rl_key}: attr_dir_ratio={ratio:.2f}, "
                f"random_ratio={rl_data.get('random_dir_ratio', 0):.2f}")

    if interaction_ratios:
        mean_ratio = np.mean(interaction_ratios)
        log(f"  MEAN interaction ratio (attr_dir): {mean_ratio:.2f}")
        log(f"  → {'Context gating CONFIRMED' if mean_ratio > 1.2 else 'Context gating NOT confirmed'} "
            f"(threshold: 1.2x)")

    # Test 3 Summary
    log("\nTest 3: Expanded Negation")
    for neg_type, type_data in test3_results.items():
        log(f"  Type: {neg_type}")
        neg_vs_random_all = []
        for pair_key, pair_data in type_data.items():
            for rl_key, rl_data in pair_data.get("neg_dir_causal", {}).items():
                neg_top = max(rl_data.get("target_vs_random", {}).items(),
                             key=lambda x: x[1], default=("N/A", 0))
                neg_vs_random_all.append(neg_top[1])
                # Compare with random direction
                rand_data = pair_data.get("random_dir_causal", {}).get(rl_key, {})
                rand_top = max(rand_data.get("target_vs_random", {}).items(),
                              key=lambda x: x[1], default=("N/A", 0))
                selectivity = neg_top[1] / rand_top[1] if rand_top[1] > 0.01 else 0
                # Log mid-layer and last layer
                rl_num = int(rl_key.replace("L", ""))
                if rl_num >= info.n_layers - 4 or rl_num == info.n_layers // 3:
                    log(f"    {pair_key}/{rl_key}: neg_top={neg_top[0]}({neg_top[1]:.2f}x), "
                        f"rand_top={rand_top[0]}({rand_top[1]:.2f}x), selectivity={selectivity:.2f}")

        if neg_vs_random_all:
            log(f"    Mean neg_dir vs random: {np.mean(neg_vs_random_all):.2f}x")

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
