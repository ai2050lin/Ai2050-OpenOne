"""
Phase 315-R2: Confirmation Test for Key Findings
=================================================
Round 2 confirmation for Phase 315's most important findings:
1. Attribute context activation (L2 ratio=5x) — confirm with more pairs
2. Function context activation — fix template, try better prompts
3. Negation direction causal test — inject negation direction, measure opposition effect
4. Context × Causal interaction — test if causal efficacy increases in context

Usage:
  python tests/glm5/phase315r2_confirm.py qwen3
  python tests/glm5/phase315r2_confirm.py glm4
  python tests/glm5/phase315r2_confirm.py deepseek7b
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
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase315r2_confirm")
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
# EXPANDED ATTRIBUTE PAIRS (50+ pairs)
# =====================================================================
EXPANDED_ATTR_PAIRS = [
    # Color
    ("apple", "red"), ("banana", "yellow"), ("sky", "blue"), ("grass", "green"),
    ("snow", "white"), ("night", "dark"), ("fire", "orange"), ("ocean", "deep_blue"),
    ("sunset", "golden"), ("lemon", "yellow"),
    # Temperature
    ("ice", "cold"), ("fire", "hot"), ("summer", "warm"), ("winter", "cold"),
    ("desert", "hot"), ("arctic", "freezing"),
    # Texture
    ("silk", "smooth"), ("sandpaper", "rough"), ("glass", "smooth"), ("stone", "hard"),
    ("cotton", "soft"), ("steel", "hard"), ("feather", "light_weight"), ("lead", "heavy"),
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
    # Age
    ("baby", "young"), ("elder", "old"), ("wine", "aged"), ("flower", "fresh"),
    # Emotional
    ("joy", "bright"), ("grief", "dark"), ("anger", "hot"), ("calm", "cool"),
    # Additional
    ("diamond", "hard"), ("cloud", "soft"), ("iron", "strong"), ("thread", "thin"),
    ("forest", "dense"), ("desert", "empty"), ("storm", "violent"), ("lake", "calm"),
]

# Random control pairs (cross-category)
EXPANDED_RANDOM_PAIRS = [
    ("apple", "hammer"), ("dog", "key"), ("knife", "river"), ("car", "happy"),
    ("table", "cold"), ("fire", "cat"), ("silk", "bus"), ("book", "lemon"),
    ("lamp", "eagle"), ("door", "sweet"), ("stone", "shovel"), ("snow", "sword"),
    ("feather", "bridge"), ("honey", "chair"), ("phone", "sky"), ("rose", "pen"),
    ("lock", "hawk"), ("boat", "sour"), ("cup", "lily"), ("shovel", "smooth"),
    ("diamond", "turtle"), ("cloud", "needle"), ("iron", "sunset"), ("thread", "arctic"),
    ("forest", "plate"), ("desert", "ball"), ("storm", "baby"), ("lake", "salt"),
    ("cheetah", "glass"), ("rocket", "sandpaper"), ("drum", "winter"), ("breeze", "steel"),
]


# =====================================================================
# FIXED FUNCTION TEMPLATES
# =====================================================================
FUNCTION_TEMPLATES = {
    "static": "the {obj} was there",
    "function_designed": "A {obj} is designed for",
    "function_purpose": "The purpose of a {obj} is to",
    "function_using": "Using a {obj}, you can",
    "function_tool": "The {obj} is a tool for",
}

FUNCTION_PAIRS = [
    ("knife", "cut"), ("key", "open"), ("car", "drive"), ("hammer", "hit"),
    ("pen", "write"), ("cup", "drink"), ("phone", "call"), ("door", "enter"),
    ("book", "read"), ("shovel", "dig"), ("boat", "sail"), ("lamp", "light"),
    ("lock", "secure"), ("bridge", "cross"), ("rope", "tie"), ("needle", "sew"),
    ("brush", "paint"), ("ladder", "climb"), ("mirror", "reflect"), ("filter", "clean"),
    ("scale", "weigh"), ("clock", "time"), ("compass", "navigate"), ("anchor", "moor"),
]


# =====================================================================
# NEGATION DIRECTION CAUSAL TEST
# =====================================================================
NEGATION_CAUSAL_PAIRS = [
    {"positive": "they were very happy about it",
     "negative": "they were not happy about it",
     "positive_tokens": ["happy", "glad", "pleased", "joyful"],
     "negative_tokens": ["not", "unhappy", "sad", "disappointed"]},
    {"positive": "the result was very good",
     "negative": "the result was not good",
     "positive_tokens": ["good", "great", "excellent", "positive"],
     "negative_tokens": ["not", "bad", "poor", "negative"]},
    {"positive": "the place was very safe",
     "negative": "the place was not safe",
     "positive_tokens": ["safe", "secure", "protected"],
     "negative_tokens": ["not", "dangerous", "unsafe", "risky"]},
    {"positive": "the task was very possible",
     "negative": "the task was not possible",
     "positive_tokens": ["possible", "feasible", "achievable"],
     "negative_tokens": ["not", "impossible", "unfeasible"]},
    {"positive": "the room was very clean",
     "negative": "the room was not clean",
     "positive_tokens": ["clean", "tidy", "neat"],
     "negative_tokens": ["not", "dirty", "messy"]},
]


def load_model_bf16(model_name):
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
    return model, tokenizer, device


def get_rep_at_layer(model, tokenizer, device, sentence, layer_idx):
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
# TEST 1: CONFIRM ATTRIBUTE CONTEXT ACTIVATION (50+ pairs)
# =====================================================================

def test_attribute_confirmation(model, tokenizer, device, model_info):
    log("=" * 60)
    log("TEST 1: Attribute Context Activation Confirmation (50+ pairs)")
    log("=" * 60)

    n_layers = model_info.n_layers
    if n_layers >= 36:
        test_layers = [2, 6, 12, 18, 24]
    elif n_layers >= 24:
        test_layers = [2, 6, 12, 16, n_layers - 2]
    else:
        test_layers = [2, 6, 12, 16, n_layers - 2]

    results = {}

    for li in test_layers:
        # Attribute fill context
        attr_dists = []
        for obj, attr in EXPANDED_ATTR_PAIRS:
            sent1 = f"The {obj} has the quality of being {attr}"
            sent2 = f"The {attr} is a quality"
            h1 = get_rep_at_layer(model, tokenizer, device, sent1, li)
            h2 = get_rep_at_layer(model, tokenizer, device, sent2, li)
            if h1 is not None and h2 is not None:
                n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
                if n1 > 1e-10 and n2 > 1e-10:
                    attr_dists.append(1.0 - np.dot(h1, h2) / (n1 * n2))

        # Static context
        static_dists = []
        for obj, attr in EXPANDED_ATTR_PAIRS:
            sent1 = f"the {obj} was there"
            sent2 = f"the {attr} was there"
            h1 = get_rep_at_layer(model, tokenizer, device, sent1, li)
            h2 = get_rep_at_layer(model, tokenizer, device, sent2, li)
            if h1 is not None and h2 is not None:
                n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
                if n1 > 1e-10 and n2 > 1e-10:
                    static_dists.append(1.0 - np.dot(h1, h2) / (n1 * n2))

        # Random baseline
        random_dists = []
        for obj, attr in EXPANDED_RANDOM_PAIRS:
            sent1 = f"the {obj} was there"
            sent2 = f"the {attr} was there"
            h1 = get_rep_at_layer(model, tokenizer, device, sent1, li)
            h2 = get_rep_at_layer(model, tokenizer, device, sent2, li)
            if h1 is not None and h2 is not None:
                n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
                if n1 > 1e-10 and n2 > 1e-10:
                    random_dists.append(1.0 - np.dot(h1, h2) / (n1 * n2))

        static_mean = np.mean(static_dists) if static_dists else 0
        attr_mean = np.mean(attr_dists) if attr_dists else 0
        random_mean = np.mean(random_dists) if random_dists else 0

        static_ratio = random_mean / static_mean if static_mean > 1e-10 else 0
        attr_ratio = random_mean / attr_mean if attr_mean > 1e-10 else 0

        results[str(li)] = {
            "n_attr_pairs": len(attr_dists),
            "n_random_pairs": len(random_dists),
            "static_mean_dist": float(static_mean),
            "attribute_fill_mean_dist": float(attr_mean),
            "random_mean_dist": float(random_mean),
            "static_ratio_vs_random": float(static_ratio),
            "attribute_fill_ratio_vs_random": float(attr_ratio),
        }

        log(f"  L{li}: static={static_mean:.3f}(r={static_ratio:.2f}), "
            f"attr_fill={attr_mean:.3f}(r={attr_ratio:.2f}), "
            f"random={random_mean:.3f}, n_attr={len(attr_dists)}")

    return results


# =====================================================================
# TEST 2: FIXED FUNCTION TEMPLATES
# =====================================================================

def test_function_templates(model, tokenizer, device, model_info):
    log("=" * 60)
    log("TEST 2: Function Template Comparison")
    log("=" * 60)

    n_layers = model_info.n_layers
    if n_layers >= 36:
        test_layers = [2, 6, 12]
    else:
        test_layers = [2, 6, 12]

    results = {}

    for li in test_layers:
        layer_results = {}

        for tmpl_name, tmpl in FUNCTION_TEMPLATES.items():
            pair_dists = []
            for obj, func in FUNCTION_PAIRS:
                try:
                    sent1 = tmpl.format(obj=obj)
                    sent2 = tmpl.format(obj=func)
                except (KeyError, IndexError):
                    continue

                h1 = get_rep_at_layer(model, tokenizer, device, sent1, li)
                h2 = get_rep_at_layer(model, tokenizer, device, sent2, li)
                if h1 is not None and h2 is not None:
                    n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        pair_dists.append(1.0 - np.dot(h1, h2) / (n1 * n2))

            # Random baseline
            random_dists = []
            import random as rng
            rng.seed(42)
            random_pairs = rng.sample(EXPANDED_RANDOM_PAIRS, min(20, len(EXPANDED_RANDOM_PAIRS)))
            for obj, attr in random_pairs:
                try:
                    sent1 = tmpl.format(obj=obj)
                    sent2 = tmpl.format(obj=attr)
                except:
                    continue
                h1 = get_rep_at_layer(model, tokenizer, device, sent1, li)
                h2 = get_rep_at_layer(model, tokenizer, device, sent2, li)
                if h1 is not None and h2 is not None:
                    n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        random_dists.append(1.0 - np.dot(h1, h2) / (n1 * n2))

            mean_dist = np.mean(pair_dists) if pair_dists else 0
            random_mean = np.mean(random_dists) if random_dists else 0
            ratio = random_mean / mean_dist if mean_dist > 1e-10 else 0

            layer_results[tmpl_name] = {
                "mean_dist": float(mean_dist),
                "random_mean": float(random_mean),
                "ratio_vs_random": float(ratio),
                "n_pairs": len(pair_dists),
            }

            log(f"  L{li} {tmpl_name}: dist={mean_dist:.3f}, random={random_mean:.3f}, ratio={ratio:.2f}")

        results[str(li)] = layer_results

    return results


# =====================================================================
# TEST 3: NEGATION DIRECTION CAUSAL TEST
# =====================================================================

def test_negation_causal(model, tokenizer, device, model_info):
    log("=" * 60)
    log("TEST 3: Negation Direction Causal Test")
    log("=" * 60)

    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Use mid-layer for injection
    if n_layers >= 36:
        inject_layer = 12
        read_layers = [6, 12, 18, 24, n_layers - 2]
    else:
        inject_layer = 8
        read_layers = [4, 8, 12, 16, n_layers - 2]

    log(f"Inject layer: {inject_layer}, Read layers: {read_layers}")

    from model_utils import get_W_U
    W_U = get_W_U(model, model_info.name)

    results = {}

    for pair_data in NEGATION_CAUSAL_PAIRS:
        pos_sent = pair_data["positive"]
        neg_sent = pair_data["negative"]
        pos_tokens = pair_data["positive_tokens"]
        neg_tokens = pair_data["negative_tokens"]

        log(f"\n  Positive: '{pos_sent}'")
        log(f"  Negative: '{neg_sent}'")

        # Extract negation direction at inject layer
        h_pos = get_rep_at_layer(model, tokenizer, device, pos_sent, inject_layer)
        h_neg = get_rep_at_layer(model, tokenizer, device, neg_sent, inject_layer)

        if h_pos is None or h_neg is None:
            log(f"  WARNING: Could not extract representations, skipping")
            continue

        neg_dir = h_neg - h_pos  # negation direction
        neg_dir_norm = np.linalg.norm(neg_dir)
        log(f"  Negation direction norm: {neg_dir_norm:.6f}")

        if neg_dir_norm < 1e-10:
            continue

        # Token IDs
        pos_ids = {}
        for tok in pos_tokens:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                pos_ids[tok] = ids[0]

        neg_ids = {}
        for tok in neg_tokens:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                neg_ids[tok] = ids[0]

        random_ids = np.random.choice(W_U.shape[0], size=20, replace=False)

        # W_U projection
        wu_proj = W_U @ neg_dir  # [vocab_size]

        pos_effects = {tok: float(wu_proj[tid]) for tok, tid in pos_ids.items()}
        neg_effects = {tok: float(wu_proj[tid]) for tok, tid in neg_ids.items()}
        random_effects = [float(wu_proj[rid]) for rid in random_ids]
        random_abs_mean = float(np.mean(np.abs(random_effects)))

        # Hook-based causal injection
        layers = get_layers(model)
        hook_data = {}

        for read_li in read_layers:
            # Clean run (positive sentence)
            captured_clean = {}
            def make_hook(cdict, name):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        cdict[name] = output[0].detach()
                    else:
                        cdict[name] = output.detach()
                return hook_fn

            read_layer = layers[read_li]
            h_read = read_layer.register_forward_hook(make_hook(captured_clean, "h"))

            inp = tokenizer(pos_sent, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inp)
            h_read.remove()

            # Injected run (inject negation direction into positive sentence)
            inject_vec = torch.tensor(neg_dir, dtype=torch.bfloat16, device=device)
            captured_inject = {}

            inject_layer_obj = layers[inject_layer]

            def inject_hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    h[0, -1, :] += inject_vec.to(h.dtype)
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    h[0, -1, :] += inject_vec.to(h.dtype)
                    return h

            h_inject = inject_layer_obj.register_forward_hook(inject_hook)
            h_read2 = read_layer.register_forward_hook(make_hook(captured_inject, "h"))

            with torch.no_grad():
                model(**inp)

            h_inject.remove()
            h_read2.remove()

            if "h" in captured_clean and "h" in captured_inject:
                h_clean = captured_clean["h"][0, -1].cpu().float().numpy()
                h_injected = captured_inject["h"][0, -1].cpu().float().numpy()
                delta_h = h_injected - h_clean
                delta_logits = W_U @ delta_h

                delta_pos = {tok: float(delta_logits[tid]) for tok, tid in pos_ids.items()}
                delta_neg = {tok: float(delta_logits[tid]) for tok, tid in neg_ids.items()}
                delta_random = [float(delta_logits[rid]) for rid in random_ids]
                delta_random_abs_mean = float(np.mean(np.abs(delta_random)))

                hook_data[f"L{read_li}"] = {
                    "delta_h_norm": float(np.linalg.norm(delta_h)),
                    "pos_logit_deltas": delta_pos,
                    "neg_logit_deltas": delta_neg,
                    "random_abs_mean": delta_random_abs_mean,
                    "neg_vs_random": {tok: abs(v) / delta_random_abs_mean if delta_random_abs_mean > 1e-10 else 0
                                      for tok, v in delta_neg.items()},
                    "pos_vs_random": {tok: abs(v) / delta_random_abs_mean if delta_random_abs_mean > 1e-10 else 0
                                      for tok, v in delta_pos.items()},
                }

                log(f"    L{read_li}: delta_norm={np.linalg.norm(delta_h):.3f}, "
                    f"neg_top={max(delta_neg.items(), key=lambda x: abs(x[1]))}, "
                    f"pos_top={max(delta_pos.items(), key=lambda x: abs(x[1]))}")

        pair_key = pos_sent[:30]
        results[pair_key] = {
            "positive": pos_sent,
            "negative": neg_sent,
            "neg_dir_norm": float(neg_dir_norm),
            "W_U_projection": {
                "pos_effects": pos_effects,
                "neg_effects": neg_effects,
                "random_abs_mean": random_abs_mean,
            },
            "hook_causal": hook_data,
        }

    return results


# =====================================================================
# MAIN
# =====================================================================

def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase315r2_{model_name}.log")

    log(f"=== Phase 315-R2: Confirmation for {model_name} ===")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"Model: n_layers={info.n_layers}, d_model={info.d_model}")

    # Test 1: Attribute confirmation
    t0 = time.time()
    attr_results = test_attribute_confirmation(model, tokenizer, device, info)
    t1 = time.time()
    log(f"Test 1 completed in {t1-t0:.1f}s")

    # Test 2: Function templates
    t0 = time.time()
    func_results = test_function_templates(model, tokenizer, device, info)
    t1 = time.time()
    log(f"Test 2 completed in {t1-t0:.1f}s")

    # Test 3: Negation causal
    t0 = time.time()
    neg_results = test_negation_causal(model, tokenizer, device, info)
    t1 = time.time()
    log(f"Test 3 completed in {t1-t0:.1f}s")

    # Save
    all_results = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "test1_attribute_confirmation": attr_results,
        "test2_function_templates": func_results,
        "test3_negation_causal": neg_results,
    }

    out_path = RESULT_DIR / f"{model_name}_confirm.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # Summary
    log("\n" + "=" * 70)
    log(f"PHASE 315-R2 SUMMARY - {model_name}")
    log("=" * 70)

    log("\nTest 1: Attribute Confirmation")
    for li_str, data in sorted(attr_results.items(), key=lambda x: int(x[0])):
        log(f"  L{li_str}: static_ratio={data['static_ratio_vs_random']:.2f}, "
            f"attr_fill_ratio={data['attribute_fill_ratio_vs_random']:.2f} "
            f"(n_attr={data['n_attr_pairs']}, n_random={data['n_random_pairs']})")

    log("\nTest 2: Function Templates")
    for li_str, li_data in sorted(func_results.items(), key=lambda x: int(x[0])):
        for tmpl_name, tmpl_data in li_data.items():
            log(f"  L{li_str} {tmpl_name}: ratio={tmpl_data['ratio_vs_random']:.2f}")

    log("\nTest 3: Negation Causal")
    for pair_key, pair_data in neg_results.items():
        log(f"  '{pair_key}': neg_dir_norm={pair_data['neg_dir_norm']:.4f}")
        for li_key, li_data in pair_data.get("hook_causal", {}).items():
            neg_top = max(li_data.get("neg_vs_random", {}).items(), key=lambda x: x[1], default=("N/A", 0))
            log(f"    {li_key}: neg_top={neg_top[0]}({neg_top[1]:.2f}x)")

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
