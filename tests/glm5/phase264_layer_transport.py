"""
Phase 264: Layer Transport Theory — Jacobian, Activation Paths, MLP Intervention
==================================================================================

Combining insights from Analysis 1 (gauge geometry, Jacobian transport) and
Analysis 2 (activation path mapping, encoding as computation path):

  Part 1 (264a): Jacobian Direction Transport
    - Compute JVP (Jacobian-vector product) at sampled layers
    - For number, animacy, tense, and random directions
    - Measure: preservation, gauge covariance, transport gain
    - Key question: do probe directions transform consistently through layers?

  Part 2 (264b): Activation Path Mapping
    - Map which MLP neurons activate for 5 concept categories (50 words)
    - Compute intra vs inter-category path overlap (Jaccard)
    - Identify category-specific neurons
    - Decode key-value semantics of shared neurons
    - Key question: do words in same category share computation paths?

  Part 3 (264c): MLP-Level Causal Intervention
    - Extract number direction at each MLP layer (not just embedding)
    - Inject direction at layers 0, 5, 10, 15, 20, 25
    - Compare trajectory monotonicity and bidirectional control vs embedding
    - Key question: does MLP-level intervention give better causal control?

Usage:
  python tests/glm5/phase264_layer_transport.py --model qwen3 --part 1
  python tests/glm5/phase264_layer_transport.py --model glm4 --part 2
  python tests/glm5/phase264_layer_transport.py --model deepseek7b --part 3
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase264_layer_transport")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Word lists =====
SING_VERBS = ["runs", "walks", "sits", "is", "has", "does", "goes", "was", "eats", "makes"]
PLUR_VERBS = ["run", "walk", "sit", "are", "have", "do", "go", "were", "eat", "make"]

TRAIN_SING = [
    "cat", "dog", "bird", "fish", "child", "woman", "man", "person",
    "teacher", "doctor", "student", "writer", "artist", "driver", "worker",
    "tree", "flower", "river", "mountain", "book", "car", "house", "door",
    "girl", "boy", "king", "queen", "hero", "friend", "mother", "father",
    "sister", "brother", "scientist", "engineer", "lawyer", "nurse",
    "apple", "orange", "banana", "grape", "peach", "cherry", "lemon",
    "horse", "sheep", "goose", "mouse", "tooth", "foot"
]
TRAIN_PLUR = [
    "cats", "dogs", "birds", "fish", "children", "women", "men", "people",
    "teachers", "doctors", "students", "writers", "artists", "drivers", "workers",
    "trees", "flowers", "rivers", "mountains", "books", "cars", "houses", "doors",
    "girls", "boys", "kings", "queens", "heroes", "friends", "mothers", "fathers",
    "sisters", "brothers", "scientists", "engineers", "lawyers", "nurses",
    "apples", "oranges", "bananas", "grapes", "peaches", "cherries", "lemons",
    "horses", "sheep", "geese", "mice", "teeth", "feet"
]
TEST_SING = [
    "bear", "eagle", "rabbit", "tiger", "whale", "fox", "deer", "wolf",
    "snake", "crow", "ant", "owl", "penguin", "dolphin", "spider",
    "lamp", "clock", "plate", "cup", "glass", "pillow", "blanket",
    "hammer", "rope", "ring", "coin", "letter", "map", "photo", "key"
]

ANIMATE = [
    "dog", "cat", "bird", "fish", "child", "woman", "man", "person",
    "teacher", "doctor", "student", "girl", "boy", "king", "queen",
    "hero", "friend", "mother", "father", "sister", "brother",
    "horse", "sheep", "goose", "mouse", "lion", "bear", "eagle",
    "rabbit", "monkey", "elephant", "tiger", "whale", "dolphin",
    "ant", "bee", "cow", "pig", "chicken", "duck"
]
INANIMATE = [
    "tree", "flower", "river", "mountain", "book", "car", "house", "door",
    "apple", "orange", "city", "country", "knife", "leaf", "loaf",
    "stone", "table", "water", "chair", "cloud", "fire", "rock",
    "road", "bridge", "wall", "building", "window", "paper", "glass",
    "metal", "wood", "plastic", "fabric", "soil", "sand", "ice",
    "snow", "rain", "wind", "light", "lamp"
]

PRESENT_WORDS = [
    "runs", "walks", "sits", "eats", "makes", "takes", "gives",
    "comes", "goes", "sees", "knows", "thinks", "says", "gets",
    "finds", "tells", "asks", "seems", "feels", "leaves", "calls"
]
PAST_WORDS = [
    "ran", "walked", "sat", "ate", "made", "took", "gave",
    "came", "went", "saw", "knew", "thought", "said", "got",
    "found", "told", "asked", "seemed", "felt", "left", "called"
]

# Concept categories for activation path mapping
CONCEPT_CATEGORIES = {
    "animals": ["dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "duck", "hen"],
    "fruits": ["apple", "banana", "orange", "grape", "peach", "cherry", "lemon", "pear", "plum", "mango"],
    "tools": ["hammer", "saw", "drill", "wrench", "ruler", "knife", "axe", "chisel", "screw", "nail"],
    "body_parts": ["hand", "foot", "head", "arm", "leg", "eye", "ear", "nose", "mouth", "finger"],
    "clothing": ["shirt", "pants", "dress", "coat", "hat", "shoe", "sock", "belt", "scarf", "glove"],
}


def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_input_device(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_special_token_offset(tokenizer, prompt_text):
    no_special = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    with_special = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
    return with_special - no_special


def safe_get_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def load_model_for_phase264(model_name, need_attn_weights=False):
    """Load model with BF16 + device_map='auto' + flash attention"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS, get_model_info, get_layers

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if need_attn_weights:
        attn_impls = ["eager"]
    else:
        attn_impls = ["flash_attention_2", "eager"]

    model = None
    for attn_impl in attn_impls:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            log_time(f"  Loaded with attn_implementation={attn_impl}")
            break
        except Exception as e:
            log_time(f"  {attn_impl} failed: {str(e)[:80]}, trying next...")
            continue

    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")

    model.eval()
    info = get_model_info(model, model_name)

    config = model.config
    n_heads = getattr(config, 'num_attention_heads', 32)
    head_dim = getattr(config, 'head_dim', info.d_model // n_heads)
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)

    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  class={info.model_class}, layers={info.n_layers}, d_model={info.d_model}, "
             f"n_heads={n_heads}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, info, n_heads, head_dim


def train_probe(data_dict):
    """Train linear probe and return (accuracy, direction)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    keys = list(data_dict.keys())
    if len(keys) != 2:
        return None, None
    d0, d1 = data_dict[keys[0]], data_dict[keys[1]]
    if len(d0) < 5 or len(d1) < 5:
        return None, None
    X = np.array(d0 + d1)
    y = np.array([1] * len(d0) + [0] * len(d1))
    probe = LogisticRegression(max_iter=2000, C=1.0)
    cv = min(5, min(len(d0), len(d1)))
    scores = cross_val_score(probe, X, y, cv=cv)
    probe.fit(X, y)
    direction = probe.coef_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    return round(float(np.mean(scores)), 4), direction


def get_sampled_layers(n_layers, n_sample=7):
    """Get uniformly sampled layer indices + first + last"""
    if n_layers <= n_sample:
        return list(range(n_layers))
    step = max(1, (n_layers - 1) // (n_sample - 1))
    layers = list(range(0, n_layers - 1, step))
    if (n_layers - 1) not in layers:
        layers.append(n_layers - 1)
    return sorted(set(layers))


# ============================================================
# Part 1: Jacobian Direction Transport
# ============================================================

def run_part1(model_name):
    """
    CRITICAL: How do probe directions transform through layer computations?

    For each layer l and direction v_l (probe direction at layer l):
    1. Perturb h_l by +eps*v_l and -eps*v_l
    2. Measure h_{l+1} after each perturbation
    3. Compute JVP = (h_{l+1}(+) - h_{l+1}(-)) / (2*eps)
    4. Measure:
       - cos(JVP, v_l): preservation (auto-alignment)
       - cos(JVP, v_{l+1}): gauge covariance (transport aligns with next-layer probe)
       - ||JVP||/||v_l||: transport gain (amplification or compression)

    Key hypothesis:
    - If gauge covariance is high: direction is "transported" consistently → gauge-covariant quantity
    - If gauge covariance is low but preservation is high: direction is invariant → fixed point of layer
    - If both low: direction is just a local projection, not a transported quantity
    """
    import torch
    from model_utils import get_layers

    log_time(f"=== Part 1: Jacobian Direction Transport for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim = load_model_for_phase264(model_name)
    layers = get_layers(model)
    embed_layer = model.get_input_embeddings()
    input_device = get_input_device(model)

    n_layers = info.n_layers
    d_model = info.d_model
    sampled_layers = get_sampled_layers(n_layers, n_sample=7)
    log_time(f"  n_layers={n_layers}, sampled_layers={sampled_layers}")

    # Step 1: Extract probe directions at all layers
    log_time("Step 1: Extracting probe directions at all layers...")

    # Use 30 sing/plur pairs for direction extraction
    direction_words_sing = TRAIN_SING[:30]
    direction_words_plur = TRAIN_PLUR[:30]

    # Collect hidden states at subject position for all layers
    all_layer_states = {f"L{l}": {"sing": [], "plur": []} for l in range(n_layers + 1)}

    for s_word, p_word in zip(direction_words_sing, direction_words_plur):
        for word, label in [(s_word, "sing"), (p_word, "plur")]:
            prompt = f"The {word} sits" if label == "sing" else f"The {word} sit"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            offset = get_special_token_offset(tokenizer, prompt)
            subj_pos = 1 + offset

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
            except Exception:
                continue

            if out.hidden_states:
                for l in range(len(out.hidden_states)):
                    hs = out.hidden_states[l][0, subj_pos, :].float().cpu().numpy()
                    all_layer_states[f"L{l}"][label].append(hs)

        torch.cuda.empty_cache()

    # Train probes at each layer
    probe_directions = {}
    probe_accuracies = {}
    for l in range(n_layers + 1):
        key = f"L{l}"
        acc, direction = train_probe(all_layer_states[key])
        if direction is not None:
            probe_directions[l] = direction
            probe_accuracies[l] = acc

    log_time(f"  Extracted probe directions at {len(probe_directions)} layers")
    for l in sorted(probe_directions.keys()):
        if l in sampled_layers or l == 0:
            log_time(f"    L{l}: probe_acc={probe_accuracies.get(l, 'N/A')}")

    # Also extract animacy and tense directions at sampled layers
    anim_directions = {}
    tense_directions = {}

    # Animacy at sampled layers
    for l_target in sampled_layers:
        anim_states = {"animate": [], "inanimate": []}
        for a_word, ia_word in zip(ANIMATE[:25], INANIMATE[:25]):
            for word, label in [(a_word, "animate"), (ia_word, "inanimate")]:
                prompt = f"The {word}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)
                offset = get_special_token_offset(tokenizer, prompt)
                subj_pos = 1 + offset

                try:
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attn_mask,
                                   output_hidden_states=True)
                except Exception:
                    continue

                if out.hidden_states and l_target < len(out.hidden_states):
                    hs = out.hidden_states[l_target][0, subj_pos, :].float().cpu().numpy()
                    anim_states[label].append(hs)

            torch.cuda.empty_cache()

        acc, direction = train_probe(anim_states)
        if direction is not None:
            anim_directions[l_target] = direction

    # Tense at sampled layers
    for l_target in sampled_layers:
        tense_states = {"present": [], "past": []}
        for p_word, pa_word in zip(PRESENT_WORDS, PAST_WORDS):
            for word, label in [(p_word, "present"), (pa_word, "past")]:
                prompt = f"Yesterday it {word}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)

                try:
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attn_mask,
                                   output_hidden_states=True)
                except Exception:
                    continue

                if out.hidden_states and l_target < len(out.hidden_states):
                    last_pos = -1
                    hs = out.hidden_states[l_target][0, last_pos, :].float().cpu().numpy()
                    tense_states[label].append(hs)

            torch.cuda.empty_cache()

        acc, direction = train_probe(tense_states)
        if direction is not None:
            tense_directions[l_target] = direction

    log_time(f"  Animacy directions at {len(anim_directions)} layers, tense at {len(tense_directions)} layers")

    # Step 2: Compute Jacobian-Vector Products
    log_time("Step 2: Computing JVPs at sampled layers...")

    # Test prompts for JVP
    test_prompts = [
        "The cat sits",
        "The dogs run",
        "The bird flies",
        "The teacher walks",
        "The children play",
    ]

    eps = 1e-3  # Perturbation size for numerical JVP
    jvp_results = {}

    # For each sampled layer (except the last), compute JVP
    jvp_layers = [l for l in sampled_layers if l < n_layers - 1]

    for layer_idx in jvp_layers:
        log_time(f"  Computing JVPs at layer {layer_idx}...")

        # Get direction at this layer
        directions_at_layer = {}
        if layer_idx in probe_directions:
            directions_at_layer["number"] = probe_directions[layer_idx]
        if layer_idx in anim_directions:
            directions_at_layer["animacy"] = anim_directions[layer_idx]
        if layer_idx in tense_directions:
            directions_at_layer["tense"] = tense_directions[layer_idx]
        # Random direction
        rng = np.random.RandomState(42 + layer_idx)
        random_dir = rng.randn(d_model).astype(np.float32)
        random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-10)
        directions_at_layer["random"] = random_dir

        layer_device = next(layers[layer_idx].parameters()).device

        for dir_name, direction in directions_at_layer.items():
            # Compute JVP averaged over test prompts
            jvp_accumulator = np.zeros(d_model, dtype=np.float64)
            h_next_baseline_accum = np.zeros(d_model, dtype=np.float64)
            n_valid = 0

            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)
                offset = get_special_token_offset(tokenizer, prompt)
                subj_pos = 1 + offset

                direction_tensor = torch.tensor(direction, dtype=model.dtype, device=layer_device)

                # === +eps perturbation ===
                captured_plus = {}

                def make_pre_hook_plus(eps_v, dir_t, pos):
                    def hook(module, args):
                        hidden_states = args[0].clone()
                        hidden_states[:, pos, :] += eps_v * dir_t.to(hidden_states.device)
                        return (hidden_states,) + args[1:]
                    return hook

                def make_post_hook_plus(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured_plus[key] = output[0].detach().float().cpu()
                        else:
                            captured_plus[key] = output.detach().float().cpu()
                    return hook

                hooks_plus = []
                hooks_plus.append(layers[layer_idx].register_forward_pre_hook(
                    make_pre_hook_plus(eps, direction_tensor, subj_pos)))
                hooks_plus.append(layers[layer_idx].register_forward_hook(
                    make_post_hook_plus("h_next")))

                try:
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attn_mask)
                except Exception as e:
                    log_time(f"    +eps forward failed at L{layer_idx}/{dir_name}: {str(e)[:60]}")
                    for h in hooks_plus: h.remove()
                    continue

                for h in hooks_plus: h.remove()

                if "h_next" not in captured_plus:
                    continue

                h_next_plus = captured_plus["h_next"][0, subj_pos, :].numpy()

                # === -eps perturbation ===
                captured_minus = {}

                def make_pre_hook_minus(eps_v, dir_t, pos):
                    def hook(module, args):
                        hidden_states = args[0].clone()
                        hidden_states[:, pos, :] -= eps_v * dir_t.to(hidden_states.device)
                        return (hidden_states,) + args[1:]
                    return hook

                def make_post_hook_minus(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured_minus[key] = output[0].detach().float().cpu()
                        else:
                            captured_minus[key] = output.detach().float().cpu()
                    return hook

                hooks_minus = []
                hooks_minus.append(layers[layer_idx].register_forward_pre_hook(
                    make_pre_hook_minus(eps, direction_tensor, subj_pos)))
                hooks_minus.append(layers[layer_idx].register_forward_hook(
                    make_post_hook_minus("h_next")))

                try:
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attn_mask)
                except Exception as e:
                    log_time(f"    -eps forward failed at L{layer_idx}/{dir_name}: {str(e)[:60]}")
                    for h in hooks_minus: h.remove()
                    continue

                for h in hooks_minus: h.remove()

                if "h_next" not in captured_minus:
                    continue

                h_next_minus = captured_minus["h_next"][0, subj_pos, :].numpy()

                # JVP = (h_{l+1}(+) - h_{l+1}(-)) / (2*eps)
                jvp = (h_next_plus - h_next_minus) / (2 * eps)
                jvp_accumulator += jvp
                n_valid += 1

                del captured_plus, captured_minus, h_next_plus, h_next_minus
                torch.cuda.empty_cache()

            if n_valid > 0:
                jvp_mean = jvp_accumulator / n_valid
                jvp_norm = np.linalg.norm(jvp_mean)
                dir_norm = np.linalg.norm(direction)

                # Preservation: cos(JVP, v_l)
                preservation = float(np.dot(jvp_mean, direction) / (jvp_norm * dir_norm + 1e-10))

                # Gauge covariance: cos(JVP, v_{l+1})
                gauge_covariance = None
                if (layer_idx + 1) in probe_directions and dir_name == "number":
                    v_next = probe_directions[layer_idx + 1]
                    gauge_covariance = float(np.dot(jvp_mean, v_next) / (jvp_norm * np.linalg.norm(v_next) + 1e-10))

                # Transport gain: ||JVP|| / ||v||
                transport_gain = jvp_norm / (dir_norm + 1e-10)

                jvp_results[f"L{layer_idx}_{dir_name}"] = {
                    "preservation": round(preservation, 4),
                    "gauge_covariance": round(gauge_covariance, 4) if gauge_covariance is not None else None,
                    "transport_gain": round(transport_gain, 4),
                    "jvp_norm": round(jvp_norm, 4),
                    "n_valid_prompts": n_valid,
                }

                gc_str = f"{gauge_covariance:.4f}" if gauge_covariance is not None else "N/A"
                log_time(f"    L{layer_idx}/{dir_name}: preservation={preservation:.4f}, "
                         f"gauge_cov={gc_str}, gain={transport_gain:.4f}")

    # Step 3: Summary analysis
    log_time("\nStep 3: Jacobian Transport Summary...")

    # Group by direction type
    direction_types = ["number", "animacy", "tense", "random"]
    summary = {}
    for dir_type in direction_types:
        entries = {k: v for k, v in jvp_results.items() if k.endswith(f"_{dir_type}")}
        if not entries:
            continue

        preservation_vals = [v["preservation"] for v in entries.values()]
        gauge_vals = [v["gauge_covariance"] for v in entries.values() if v["gauge_covariance"] is not None]
        gain_vals = [v["transport_gain"] for v in entries.values()]

        summary[dir_type] = {
            "mean_preservation": round(float(np.mean(preservation_vals)), 4),
            "mean_transport_gain": round(float(np.mean(gain_vals)), 4),
            "mean_gauge_covariance": round(float(np.mean(gauge_vals)), 4) if gauge_vals else None,
            "n_layers_tested": len(entries),
        }

        log_time(f"  {dir_type}: preservation={summary[dir_type]['mean_preservation']:.4f}, "
                 f"gain={summary[dir_type]['mean_transport_gain']:.4f}, "
                 f"gauge_cov={summary[dir_type]['mean_gauge_covariance']}")

    # Key comparison: number vs random
    if "number" in summary and "random" in summary:
        num_pres = abs(summary["number"]["mean_preservation"])
        rand_pres = abs(summary["random"]["mean_preservation"])
        specificity_ratio = num_pres / (rand_pres + 1e-10)

        log_time(f"\n  ★ Number vs Random preservation ratio: {specificity_ratio:.2f}")
        log_time(f"    Number: {num_pres:.4f}, Random: {rand_pres:.4f}")
        if specificity_ratio > 2.0:
            log_time(f"    → Number direction is SIGNIFICANTLY better preserved than random")
        else:
            log_time(f"    → Number direction is NOT significantly better preserved than random")

    results = {
        "model": model_name,
        "n_layers": n_layers,
        "sampled_layers": sampled_layers,
        "probe_accuracies": {str(k): v for k, v in probe_accuracies.items()},
        "jvp_results": jvp_results,
        "summary": summary,
        "eps": eps,
    }

    out_path = RESULT_DIR / f"{model_name}_part1_jacobian_transport.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 1 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Part 2: Activation Path Mapping
# ============================================================

def run_part2(model_name):
    """
    Map which MLP neurons are activated by different concept categories.

    For each word:
    1. Run model with hooks on MLP down_proj to capture intermediate activations
    2. Find top-k activated neurons per layer
    3. Compute intra vs inter-category path overlap (Jaccard similarity)
    4. Identify category-specific neurons
    5. Decode key-value semantics of shared neurons
    """
    import torch
    from model_utils import get_layers, get_layer_weights, get_W_U

    log_time(f"=== Part 2: Activation Path Mapping for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim = load_model_for_phase264(model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)

    n_layers = info.n_layers
    d_model = info.d_model
    intermediate_size = info.intermediate_size

    log_time(f"  n_layers={n_layers}, d_model={d_model}, intermediate_size={intermediate_size}")

    top_k = 30  # Top-k activated neurons per layer

    # Step 1: Collect MLP intermediate activations for each word
    log_time("Step 1: Collecting MLP activations for 50 concept words...")

    word_activations = {}  # word -> {layer_idx: {neuron_idx: activation_value}}
    word_categories = {}   # word -> category

    for category, words in CONCEPT_CATEGORIES.items():
        log_time(f"  Processing category: {category} ({len(words)} words)")
        for word in words:
            word_categories[word] = category
            prompt = f"The {word}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            offset = get_special_token_offset(tokenizer, prompt)
            subj_pos = 1 + offset

            # Capture MLP intermediate activations via down_proj hook
            captured = {}

            def make_mlp_hook(layer_idx):
                def hook(module, input, output):
                    # input[0] is the intermediate activation before down_proj
                    # shape: [batch, seq_len, intermediate_size]
                    if isinstance(input, tuple) and len(input) > 0:
                        intermediate = input[0].detach().float().cpu()
                        captured[layer_idx] = intermediate[0, subj_pos, :].numpy()
                return hook

            hooks = []
            for li in range(n_layers):
                if hasattr(layers[li], 'mlp') and hasattr(layers[li].mlp, 'down_proj'):
                    hooks.append(layers[li].mlp.down_proj.register_forward_hook(make_mlp_hook(li)))

            try:
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attn_mask)
            except Exception as e:
                log_time(f"    Forward failed for '{word}': {str(e)[:60]}")
                for h in hooks: h.remove()
                continue

            for h in hooks: h.remove()

            # Find top-k activated neurons per layer
            word_activations[word] = {}
            for layer_idx, activations in captured.items():
                abs_activations = np.abs(activations)
                top_indices = np.argsort(abs_activations)[-top_k:][::-1]
                word_activations[word][layer_idx] = {
                    int(idx): float(activations[idx]) for idx in top_indices
                }

            torch.cuda.empty_cache()

    log_time(f"  Collected activations for {len(word_activations)} words")

    # Step 2: Compute path overlap (Jaccard similarity)
    log_time("Step 2: Computing path overlap...")

    # Convert to sets of (layer, neuron) pairs for Jaccard
    def get_active_set(word_acts, top_n=None):
        """Get set of (layer, neuron) pairs for a word"""
        active = set()
        for layer_idx, neurons in word_acts.items():
            if top_n:
                sorted_neurons = sorted(neurons.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
                for neuron_idx, _ in sorted_neurons:
                    active.add((layer_idx, neuron_idx))
            else:
                for neuron_idx in neurons:
                    active.add((layer_idx, neuron_idx))
        return active

    # Compute Jaccard similarity
    def jaccard(set_a, set_b):
        if len(set_a) == 0 and len(set_b) == 0:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    # Get active sets for all words
    word_active_sets = {w: get_active_set(acts, top_n=20) for w, acts in word_activations.items()}

    # Intra-category overlap
    intra_overlaps = {cat: [] for cat in CONCEPT_CATEGORIES}
    inter_overlaps = []

    all_words = list(word_active_sets.keys())
    for i, w1 in enumerate(all_words):
        for j, w2 in enumerate(all_words):
            if i >= j:
                continue
            cat1 = word_categories.get(w1, "unknown")
            cat2 = word_categories.get(w2, "unknown")
            sim = jaccard(word_active_sets[w1], word_active_sets[w2])

            if cat1 == cat2:
                intra_overlaps[cat1].append(sim)
            else:
                inter_overlaps.append(sim)

    # Summary
    overlap_summary = {}
    for cat, overlaps in intra_overlaps.items():
        if overlaps:
            overlap_summary[f"intra_{cat}"] = {
                "mean": round(float(np.mean(overlaps)), 4),
                "std": round(float(np.std(overlaps)), 4),
                "n_pairs": len(overlaps),
            }
            log_time(f"  Intra-{cat}: mean_jaccard={np.mean(overlaps):.4f} ± {np.std(overlaps):.4f}")

    if inter_overlaps:
        overlap_summary["inter_category"] = {
            "mean": round(float(np.mean(inter_overlaps)), 4),
            "std": round(float(np.std(inter_overlaps)), 4),
            "n_pairs": len(inter_overlaps),
        }
        log_time(f"  Inter-category: mean_jaccard={np.mean(inter_overlaps):.4f} ± {np.std(inter_overlaps):.4f}")

    # Key comparison
    intra_means = [v["mean"] for k, v in overlap_summary.items() if k.startswith("intra_")]
    inter_mean = overlap_summary.get("inter_category", {}).get("mean", 0)

    if intra_means:
        mean_intra = float(np.mean(intra_means))
        log_time(f"\n  ★ Intra-category mean: {mean_intra:.4f}, Inter-category mean: {inter_mean:.4f}")
        ratio = mean_intra / (inter_mean + 1e-10)
        log_time(f"    Intra/Inter ratio: {ratio:.2f}")
        if ratio > 1.5:
            log_time(f"    → Words in same category SHARE significantly more neurons → Path encoding confirmed")
        else:
            log_time(f"    → No significant path sharing within categories")

    # Step 3: Identify category-specific neurons
    log_time("Step 3: Identifying category-specific neurons...")

    category_specific_neurons = {}
    for cat in CONCEPT_CATEGORIES:
        cat_words = [w for w in all_words if word_categories.get(w) == cat]
        other_words = [w for w in all_words if word_categories.get(w) != cat]

        if not cat_words or not other_words:
            continue

        # For each (layer, neuron), check if it's activated by all words in this category
        # and not activated by most words in other categories
        cat_neuron_sets = [word_active_sets[w] for w in cat_words]
        other_neuron_sets = [word_active_sets[w] for w in other_words]

        # Find neurons that appear in >= 80% of category words and <= 20% of other words
        cat_neuron_count = defaultdict(int)
        other_neuron_count = defaultdict(int)

        for ns in cat_neuron_sets:
            for n in ns:
                cat_neuron_count[n] += 1

        for ns in other_neuron_sets:
            for n in ns:
                other_neuron_count[n] += 1

        specific_neurons = []
        for neuron, count in cat_neuron_count.items():
            cat_freq = count / len(cat_words)
            other_freq = other_neuron_count.get(neuron, 0) / len(other_words)
            if cat_freq >= 0.7 and other_freq <= 0.3:
                specific_neurons.append({
                    "layer": int(neuron[0]),
                    "neuron": int(neuron[1]),
                    "cat_freq": round(cat_freq, 4),
                    "other_freq": round(other_freq, 4),
                    "specificity": round(cat_freq - other_freq, 4),
                })

        specific_neurons.sort(key=lambda x: x["specificity"], reverse=True)
        category_specific_neurons[cat] = specific_neurons[:10]  # Top 10

        log_time(f"  {cat}: {len(specific_neurons)} specific neurons (top: "
                 f"L{specific_neurons[0]['layer']}_N{specific_neurons[0]['neuron']} "
                 f"if specific_neurons else 'none'), "
                 f"specificity={specific_neurons[0]['specificity'] if specific_neurons else 'N/A'}")

    # Step 4: Decode key-value semantics of top category-specific neurons
    log_time("Step 4: Decoding key-value semantics...")

    W_U = None
    try:
        W_U = get_W_U(model, model_name)
    except Exception as e:
        log_time(f"  Could not load W_U: {e}")

    key_value_decodings = {}
    if W_U is not None:
        for cat, neurons in category_specific_neurons.items():
            if not neurons:
                continue

            cat_decodings = []
            for neuron_info in neurons[:5]:  # Top 5 per category
                layer_idx = neuron_info["layer"]
                neuron_idx = neuron_info["neuron"]

                try:
                    lw = get_layer_weights(layers[layer_idx], d_model, info.mlp_type)

                    # Key direction: what activates this neuron
                    # For SiLU-gated MLP: neuron activates when gate_proj input is high
                    if lw.W_gate is not None:
                        key_direction = lw.W_gate[neuron_idx, :]
                    else:
                        key_direction = None

                    # Value direction: what this neuron writes to residual
                    value_direction = lw.W_down[:, neuron_idx]

                    # Decode by projecting onto W_U
                    def decode_direction(direction, top_n=5):
                        if direction is None or W_U is None:
                            return []
                        logits = direction @ W_U.T
                        top_ids = np.argsort(logits)[-top_n:][::-1]
                        return [tokenizer.decode([int(i)]).strip() for i in top_ids]

                    key_tokens = decode_direction(key_direction) if key_direction is not None else []
                    value_tokens = decode_direction(value_direction)

                    cat_decodings.append({
                        "layer": layer_idx,
                        "neuron": neuron_idx,
                        "specificity": neuron_info["specificity"],
                        "key_tokens": key_tokens,
                        "value_tokens": value_tokens,
                    })

                except Exception as e:
                    log_time(f"    Failed to decode L{layer_idx}_N{neuron_idx}: {str(e)[:60]}")
                    continue

            key_value_decodings[cat] = cat_decodings

            for dec in cat_decodings:
                log_time(f"    L{dec['layer']}_N{dec['neuron']} ({cat}, spec={dec['specificity']:.3f}): "
                         f"key={dec['key_tokens'][:3]}, value={dec['value_tokens'][:3]}")

    # Step 5: Layer distribution analysis
    log_time("Step 5: Layer distribution of category-specific neurons...")

    layer_distribution = {cat: defaultdict(int) for cat in CONCEPT_CATEGORIES}
    for cat, neurons in category_specific_neurons.items():
        for n in neurons:
            layer_distribution[cat][n["layer"]] += 1

    for cat in CONCEPT_CATEGORIES:
        dist = layer_distribution[cat]
        total = sum(dist.values())
        if total > 0:
            # Which layers have the most specific neurons?
            top_layers = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]
            log_time(f"  {cat}: {total} specific neurons, concentrated at layers "
                     f"{[(l, c) for l, c in top_layers]}")

    results = {
        "model": model_name,
        "n_layers": n_layers,
        "intermediate_size": intermediate_size,
        "top_k_per_layer": top_k,
        "categories": list(CONCEPT_CATEGORIES.keys()),
        "overlap_summary": overlap_summary,
        "category_specific_counts": {cat: len(v) for cat, v in category_specific_neurons.items()},
        "category_specific_neurons": {cat: v[:5] for cat, v in category_specific_neurons.items()},
        "key_value_decodings": key_value_decodings,
        "layer_distribution": {cat: {str(k): v for k, v in dist.items()}
                               for cat, dist in layer_distribution.items()},
    }

    out_path = RESULT_DIR / f"{model_name}_part2_activation_paths.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 2 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Part 3: MLP-Level Causal Intervention
# ============================================================

def run_part3(model_name):
    """
    CRITICAL TEST: Does injecting the number direction at MLP layers
    give better causal control than injecting at the embedding layer?

    Phase 263 showed that embedding-level injection is non-monotonic and
    lacks bidirectional control. This experiment tests whether MLP-level
    injection (where the "internal representation" lives) gives better results.

    Method:
    1. Extract number direction at multiple MLP layers
    2. For each layer, inject the direction and measure grammar score change
    3. Scan alpha in [-10, 10] to test trajectory monotonicity
    4. Compare with embedding-level injection (from Phase 263)
    """
    import torch
    from model_utils import get_layers

    log_time(f"=== Part 3: MLP-Level Causal Intervention for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim = load_model_for_phase264(model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)

    n_layers = info.n_layers
    d_model = info.d_model
    sampled_layers = get_sampled_layers(n_layers, n_sample=7)
    # Only test intervention at layers that have room for remaining computation
    intervention_layers = [l for l in sampled_layers if l < n_layers - 2]

    log_time(f"  Intervention layers: {intervention_layers}")

    # Step 1: Extract number direction at each intervention layer
    log_time("Step 1: Extracting number directions at each layer...")

    # Collect hidden states at subject position for all layers
    all_layer_states = {l: {"sing": [], "plur": []} for l in intervention_layers}

    for s_word, p_word in zip(TRAIN_SING[:25], TRAIN_PLUR[:25]):
        for word, label in [(s_word, "sing"), (p_word, "plur")]:
            prompt = f"The {word} sits" if label == "sing" else f"The {word} sit"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            offset = get_special_token_offset(tokenizer, prompt)
            subj_pos = 1 + offset

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
            except Exception:
                continue

            if out.hidden_states:
                for l in intervention_layers:
                    if l + 1 < len(out.hidden_states):
                        hs = out.hidden_states[l + 1][0, subj_pos, :].float().cpu().numpy()
                        all_layer_states[l][label].append(hs)

        torch.cuda.empty_cache()

    # Train probes at each layer
    layer_directions = {}
    layer_probe_accs = {}
    for l in intervention_layers:
        acc, direction = train_probe(all_layer_states[l])
        if direction is not None:
            layer_directions[l] = direction
            layer_probe_accs[l] = acc

    log_time(f"  Extracted directions at {len(layer_directions)} layers")
    for l in sorted(layer_directions.keys()):
        log_time(f"    L{l}: probe_acc={layer_probe_accs[l]}")

    # Also get embedding direction for comparison
    embed_layer = model.get_input_embeddings()
    sing_embeds = []
    plur_embeds = []
    for s, p in zip(TRAIN_SING[:30], TRAIN_PLUR[:30]):
        s_ids = tokenizer.encode(s, add_special_tokens=False)
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        if len(s_ids) == 1 and len(p_ids) == 1:
            with torch.no_grad():
                s_emb = embed_layer.weight[s_ids[0]].detach().float().cpu().numpy()
                p_emb = embed_layer.weight[p_ids[0]].detach().float().cpu().numpy()
            sing_embeds.append(s_emb)
            plur_embeds.append(p_emb)

    embed_data = {"sing": sing_embeds, "plur": plur_embeds}
    embed_probe_acc, embed_direction = train_probe(embed_data)
    log_time(f"  Embedding probe: {embed_probe_acc}")

    # Get verb token IDs
    sing_verb_ids = [safe_get_token_id(tokenizer, v) for v in SING_VERBS if safe_get_token_id(tokenizer, v) is not None]
    plur_verb_ids = [safe_get_token_id(tokenizer, v) for v in PLUR_VERBS if safe_get_token_id(tokenizer, v) is not None]

    # Step 2: Trajectory scan at each injection layer
    log_time("Step 2: Scanning alpha trajectories at each injection layer...")

    alpha_values = list(np.arange(-10, 10.5, 2.0))  # 11 points from -10 to 10
    test_words = TEST_SING[:10]

    # For each injection layer, scan alpha and measure grammar score
    trajectory_results = {}

    # Include embedding-level injection as baseline
    injection_points = [("embed", -1)] + [(f"L{l}", l) for l in sorted(layer_directions.keys())]

    for point_name, layer_idx in injection_points:
        log_time(f"  Injection at {point_name}...")

        # Get the direction for this injection point
        if layer_idx == -1:
            direction = embed_direction
        else:
            direction = layer_directions.get(layer_idx)
            if direction is None:
                log_time(f"    No direction at {point_name}, skipping")
                continue

        # Collect trajectory data
        mean_scores = []

        for alpha in alpha_values:
            score_changes = []

            for word in test_words:
                prompt = f"The {word}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)

                offset = get_special_token_offset(tokenizer, prompt)
                subj_pos = 1 + offset

                # Baseline score
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask)
                baseline_logits = out.logits[0, -1, :].float().cpu().numpy()
                bl_sing = float(np.mean([baseline_logits[tid] for tid in sing_verb_ids]))
                bl_plur = float(np.mean([baseline_logits[tid] for tid in plur_verb_ids]))
                bl_score = bl_sing - bl_plur

                # Intervention
                if layer_idx == -1:
                    # Embedding-level injection
                    with torch.no_grad():
                        base_embed = embed_layer(input_ids).detach().clone()
                    seq_len = input_ids.shape[1]
                    position_ids = torch.arange(seq_len, device=input_device).unsqueeze(0)
                    modified_embed = base_embed.clone()
                    delta_tensor = torch.tensor(alpha * direction, dtype=base_embed.dtype, device=input_device)
                    modified_embed[0, subj_pos, :] += delta_tensor

                    with torch.no_grad():
                        out = model(inputs_embeds=modified_embed.to(model.dtype),
                                   attention_mask=attn_mask, position_ids=position_ids)
                else:
                    # MLP-level injection: modify output of layer l
                    captured_logits = {}

                    def make_injection_hook(dir_vec, alpha_val, pos):
                        dir_tensor = torch.tensor(dir_vec, dtype=model.dtype,
                                                   device=next(layers[0].parameters()).device)
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                modified = output[0].clone()
                                modified[:, pos, :] += alpha_val * dir_tensor.to(modified.device)
                                return (modified,) + output[1:]
                            else:
                                modified = output.clone()
                                modified[:, pos, :] += alpha_val * dir_tensor.to(modified.device)
                                return modified
                        return hook

                    hook = layers[layer_idx].register_forward_hook(
                        make_injection_hook(direction, alpha, subj_pos))

                    try:
                        with torch.no_grad():
                            out = model(input_ids=input_ids, attention_mask=attn_mask)
                    except Exception:
                        hook.remove()
                        continue

                    hook.remove()

                logits = out.logits[0, -1, :].float().cpu().numpy()
                sing_v = float(np.mean([logits[tid] for tid in sing_verb_ids]))
                plur_v = float(np.mean([logits[tid] for tid in plur_verb_ids]))
                score = sing_v - plur_v
                score_changes.append(score - bl_score)

                torch.cuda.empty_cache()

            mean_score = float(np.mean(score_changes)) if score_changes else 0.0
            mean_scores.append(mean_score)

        # Analyze trajectory
        alpha_arr = np.array(alpha_values, dtype=float)
        score_arr = np.array(mean_scores)

        # Monotonicity
        diffs = np.diff(score_arr)
        n_increasing = sum(1 for d in diffs if d > 0)
        monotonicity = n_increasing / len(diffs) if len(diffs) > 0 else 0

        # Correlation with alpha
        if np.std(score_arr) > 1e-10:
            correlation = float(np.corrcoef(alpha_arr, score_arr)[0, 1])
            r_squared = correlation ** 2
        else:
            correlation = 0.0
            r_squared = 0.0

        # Bidirectional test: does add (positive alpha) and subtract (negative alpha)
        # have opposite effects?
        pos_score = np.mean([s for a, s in zip(alpha_values, mean_scores) if a > 0])
        neg_score = np.mean([s for a, s in zip(alpha_values, mean_scores) if a < 0])
        bidirectional = (pos_score * neg_score < 0)  # Opposite signs

        trajectory_results[point_name] = {
            "alpha_values": [float(a) for a in alpha_values],
            "mean_scores": [round(s, 4) for s in mean_scores],
            "monotonicity": round(monotonicity, 4),
            "correlation": round(correlation, 4),
            "r_squared": round(r_squared, 4),
            "pos_alpha_mean": round(pos_score, 4),
            "neg_alpha_mean": round(neg_score, 4),
            "bidirectional": bidirectional,
            "probe_acc": layer_probe_accs.get(layer_idx, embed_probe_acc) if layer_idx >= 0 else embed_probe_acc,
        }

        interp = ("REAL_SEMANTIC_AXIS" if monotonicity > 0.85 and r_squared > 0.9
                  else "PARTIALLY_SEMANTIC" if monotonicity > 0.7 and r_squared > 0.7
                  else "LOCAL_STATISTICAL_BOUNDARY")

        log_time(f"    {point_name}: mono={monotonicity:.3f}, R²={r_squared:.3f}, "
                 f"bidirectional={bidirectional}, class={interp}")

    # Step 3: Compare embedding vs MLP-level intervention
    log_time("\nStep 3: Comparing embedding vs MLP-level intervention...")

    embed_result = trajectory_results.get("embed", {})
    mlp_results = {k: v for k, v in trajectory_results.items() if k != "embed"}

    comparison = {
        "embed_r_squared": embed_result.get("r_squared", 0),
        "embed_monotonicity": embed_result.get("monotonicity", 0),
        "embed_bidirectional": embed_result.get("bidirectional", False),
    }

    # Find the MLP layer with best R²
    if mlp_results:
        best_mlp = max(mlp_results.items(), key=lambda x: x[1]["r_squared"])
        comparison["best_mlp_layer"] = best_mlp[0]
        comparison["best_mlp_r_squared"] = best_mlp[1]["r_squared"]
        comparison["best_mlp_monotonicity"] = best_mlp[1]["monotonicity"]
        comparison["best_mlp_bidirectional"] = best_mlp[1]["bidirectional"]

        # Improvement ratio
        if embed_result.get("r_squared", 0) > 0:
            improvement = best_mlp[1]["r_squared"] / embed_result["r_squared"]
            comparison["r_squared_improvement_ratio"] = round(improvement, 2)
        else:
            comparison["r_squared_improvement_ratio"] = float('inf')

        log_time(f"  Embed: R²={embed_result.get('r_squared', 0):.4f}, mono={embed_result.get('monotonicity', 0):.3f}")
        log_time(f"  Best MLP ({best_mlp[0]}): R²={best_mlp[1]['r_squared']:.4f}, "
                 f"mono={best_mlp[1]['monotonicity']:.3f}")
        log_time(f"  Improvement ratio: {comparison.get('r_squared_improvement_ratio', 'N/A')}")

        if comparison.get("r_squared_improvement_ratio", 0) > 2.0:
            log_time(f"  ★ MLP-level intervention SIGNIFICANTLY better than embedding-level")
        else:
            log_time(f"  MLP-level intervention NOT significantly better than embedding-level")
    else:
        log_time("  No MLP results available for comparison")

    results = {
        "model": model_name,
        "n_layers": n_layers,
        "intervention_layers": intervention_layers,
        "alpha_range": [-10, 10],
        "trajectory_results": trajectory_results,
        "comparison": comparison,
    }

    out_path = RESULT_DIR / f"{model_name}_part3_mlp_intervention.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 3 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 264: Layer Transport Theory")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--part", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    if args.part == 1:
        run_part1(args.model)
    elif args.part == 2:
        run_part2(args.model)
    elif args.part == 3:
        run_part3(args.model)


if __name__ == "__main__":
    main()
