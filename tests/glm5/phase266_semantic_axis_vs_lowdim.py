"""
Phase 266: Semantic Axis vs Low-Dimensional Artifact — The Decisive Experiment
==============================================================================

This is the most critical experiment in the entire research series.

  Part 1 (266a): Random Direction Control Test (HIGHEST PRIORITY)
    - In low-entropy layers where semantic axes have R²>0.99,
      test whether RANDOM directions also have high R²
    - If random R² ≈ semantic R² → "semantic axis" is low-dimensional trivial effect
    - If random R² ≪ semantic R² → true semantic selection mechanism exists
    - This is the fork in the road for the entire theory

  Part 2 (266b): Multi-Layer Entropy-Matched Control
    - Test at layers with DIFFERENT entropy levels (high, medium, low)
    - If R²(semantic) - R²(random) is constant across entropy → semantic axis is genuine
    - If R²(semantic) - R²(random) → 0 at low entropy → low-dim artifact

  Part 3 (266c): Multi-Direction Joint Control (Intervention Capacity)
    - Can we simultaneously control number + animacy?
    - If yes → features are in orthogonal subspaces (genuine decomposition)
    - If no → features share the same low-dimensional manifold (artifact)

  Part 4 (266d): Local Effective Dimension (Participation Ratio)
    - Measure the true dimensionality of the hidden state manifold at each layer
    - Participation ratio, local rank, Fisher dimension
    - This quantifies whether deep layers are truly "low-dimensional"

Usage:
  python tests/glm5/phase266_semantic_axis_vs_lowdim.py --model qwen3 --part 1
  python tests/glm5/phase266_semantic_axis_vs_lowdim.py --model glm4 --part 2
  python tests/glm5/phase266_semantic_axis_vs_lowdim.py --model deepseek7b --part 3
  python tests/glm5/phase266_semantic_axis_vs_lowdim.py --model qwen3 --part 4
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase266_semantic_axis_vs_lowdim")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Word lists (same as Phase 265) =====
SING_WORDS = [
    "cat", "dog", "bird", "fish", "child", "woman", "man", "person",
    "teacher", "doctor", "student", "writer", "artist", "driver", "worker",
    "tree", "flower", "river", "mountain", "book", "car", "house", "door",
    "girl", "boy", "king", "queen", "hero", "friend", "mother", "father",
]
PLUR_WORDS = [
    "cats", "dogs", "birds", "fish", "children", "women", "men", "people",
    "teachers", "doctors", "students", "writers", "artists", "drivers", "workers",
    "trees", "flowers", "rivers", "mountains", "books", "cars", "houses", "doors",
    "girls", "boys", "kings", "queens", "heroes", "friends", "mothers", "fathers",
]
SING_VERBS = ["runs", "walks", "sits", "is", "has", "does", "goes", "was", "eats", "makes"]
PLUR_VERBS = ["run", "walk", "sit", "are", "have", "do", "go", "were", "eat", "make"]
TEST_SING = [
    "bear", "eagle", "rabbit", "tiger", "whale", "fox", "deer", "wolf",
    "snake", "crow", "ant", "owl", "penguin", "dolphin", "spider",
]

ANIMATE = [
    "dog", "cat", "bird", "fish", "child", "woman", "man", "person",
    "teacher", "doctor", "student", "girl", "boy", "king", "queen",
    "hero", "friend", "mother", "father", "sister", "brother",
    "horse", "sheep", "goose", "mouse", "lion", "bear", "eagle",
    "rabbit", "monkey", "elephant", "tiger", "whale", "dolphin",
    "ant", "bee", "cow", "pig", "chicken", "duck",
]
INANIMATE = [
    "rock", "stone", "table", "chair", "book", "car", "house", "door",
    "lamp", "clock", "plate", "cup", "glass", "pillow", "blanket",
    "hammer", "rope", "ring", "coin", "letter", "map", "photo", "key",
    "wall", "bridge", "window", "paper", "metal", "wood", "plastic",
    "fabric", "soil", "sand", "ice", "snow", "rain", "wind", "light",
    "cloud", "river",
]
ANIMATE_VERBS = ["thinks", "feels", "runs", "walks", "speaks", "believes", "decides", "wants", "loves", "hopes"]
INANIMATE_VERBS = ["sits", "lies", "stands", "rests", "hangs", "falls", "rolls", "breaks", "cracks", "shines"]

CONCRETE = [
    "apple", "table", "dog", "house", "car", "tree", "water", "stone",
    "knife", "book", "shirt", "chair", "door", "window", "lamp", "cup",
    "plate", "flower", "bird", "fish", "mountain", "river", "cloud", "rain",
    "fire", "snow", "ice", "sand", "metal", "wood",
]
ABSTRACT = [
    "freedom", "justice", "love", "truth", "beauty", "wisdom", "courage",
    "honesty", "loyalty", "patience", "anger", "fear", "hope", "peace",
    "power", "knowledge", "time", "space", "mind", "soul",
    "idea", "dream", "memory", "reason", "logic", "faith", "trust",
    "pride", "shame", "guilt",
]
CONCRETE_VERBS = ["holds", "touches", "sees", "carries", "drops", "picks", "throws", "catches", "lifts", "moves"]
ABSTRACT_VERBS = ["understands", "believes", "feels", "thinks", "knows", "means", "represents", "expresses", "defines", "explains"]

# Test prompts for interventions
NUMBER_TEST = [f"The {w}" for w in TEST_SING[:10]]
ANIMATE_TEST = [f"The {w}" for w in ["puppy", "kitten", "parrot", "salmon", "infant"][:5]]
INANIMATE_TEST = [f"The {w}" for w in ["boulder", "couch", "novel", "truck", "candle"][:5]]
CONCRETE_TEST = [f"The {w}" for w in CONCRETE[:5]]
ABSTRACT_TEST = [f"The {w}" for w in ABSTRACT[:5]]

# General prompts for entropy measurement
GENERAL_PROMPTS = [
    "The scientist discovered a new",
    "In the morning the old man",
    "A beautiful garden with many",
    "The children played in the",
    "She walked slowly through the",
    "The river flows past the",
    "He opened the door and",
    "The teacher explained the",
    "They watched the sun set",
    "The city was full of",
    "A small bird sat on",
    "The book tells a story",
    "Every year the family",
    "The doctor said the patient",
    "After dinner the couple",
]


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


def load_model_bf16(model_name):
    """Load model with BF16 + device_map='auto' + flash attention"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS, get_model_info, get_layers

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (BF16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash_attention_2 first, fall back to eager
    model = None
    for attn_impl in ["flash_attention_2", "eager"]:
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

    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  class={info.model_class}, layers={info.n_layers}, d_model={info.d_model}, "
             f"GPU={gpu_mem:.2f}GB")

    return model, tokenizer, info


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


def extract_probe_direction(model, tokenizer, info, input_device,
                            class_a_words, class_b_words, prompt_a, prompt_b,
                            target_layer, pos_func, n_train=25):
    """Extract a single probe direction at one specific layer"""
    import torch
    from model_utils import get_layers

    layers = get_layers(model)
    states_a = []
    states_b = []

    for a_word, b_word in zip(class_a_words[:n_train], class_b_words[:n_train]):
        for word, label in [(a_word, "A"), (b_word, "B")]:
            prompt = prompt_a if label == "A" else prompt_b
            prompt = prompt.format(word)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            offset = get_special_token_offset(tokenizer, prompt)
            subj_pos = pos_func(prompt, offset)
            n_tokens = input_ids.shape[1]
            if subj_pos < 0:
                subj_pos = n_tokens + subj_pos
            if subj_pos < 0 or subj_pos >= n_tokens:
                continue

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
            except Exception:
                continue

            if out.hidden_states and target_layer + 1 < len(out.hidden_states):
                hs = out.hidden_states[target_layer + 1][0, subj_pos, :].float().cpu().numpy()
                if label == "A":
                    states_a.append(hs)
                else:
                    states_b.append(hs)

        torch.cuda.empty_cache()

    acc, direction = train_probe({"A": states_a, "B": states_b})
    return acc, direction


def measure_intervention_r2(model, tokenizer, info, input_device,
                           direction, target_layer, test_prompts,
                           verb_ids_a, verb_ids_b, alpha_values,
                           pos_func, n_heads=None, head_dim=None,
                           inject_at="layer"):
    """
    Measure R² of direction intervention at a specific layer.
    
    Args:
        direction: numpy array [d_model], normalized
        target_layer: layer index (0-based)
        inject_at: "layer" for transformer layer output, "mlp" for MLP output
    
    Returns:
        dict with r_squared, monotonicity, bidirectional, trajectory
    """
    import torch
    from model_utils import get_layers

    layers = get_layers(model)
    
    if inject_at == "mlp":
        target_module = layers[target_layer].mlp
    else:
        target_module = layers[target_layer]

    mean_scores = []

    for alpha in alpha_values:
        score_changes = []

        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            offset = get_special_token_offset(tokenizer, prompt)
            subj_pos = pos_func(prompt, offset)
            n_tokens = input_ids.shape[1]
            if subj_pos < 0:
                subj_pos = n_tokens + subj_pos
            if subj_pos < 0 or subj_pos >= n_tokens:
                continue

            # Baseline
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask)
            baseline_logits = out.logits[0, -1, :].float().cpu().numpy()
            bl_a = float(np.mean([baseline_logits[tid] for tid in verb_ids_a if tid < len(baseline_logits)]))
            bl_b = float(np.mean([baseline_logits[tid] for tid in verb_ids_b if tid < len(baseline_logits)]))
            bl_score = bl_a - bl_b

            # Intervention
            dir_tensor = torch.tensor(direction, dtype=model.dtype,
                                      device=next(target_module.parameters()).device)

            def make_hook(d, a, p):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        modified = output[0].clone()
                        modified[:, p, :] += a * d.to(modified.device)
                        return (modified,) + output[1:]
                    else:
                        modified = output.clone()
                        modified[:, p, :] += a * d.to(modified.device)
                        return modified
                return hook

            hook = target_module.register_forward_hook(make_hook(dir_tensor, alpha, subj_pos))

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask)
            except Exception:
                hook.remove()
                continue

            hook.remove()

            logits = out.logits[0, -1, :].float().cpu().numpy()
            a_v = float(np.mean([logits[tid] for tid in verb_ids_a if tid < len(logits)]))
            b_v = float(np.mean([logits[tid] for tid in verb_ids_b if tid < len(logits)]))
            score = a_v - b_v
            score_changes.append(score - bl_score)

            torch.cuda.empty_cache()

        mean_score = float(np.mean(score_changes)) if score_changes else 0.0
        mean_scores.append(mean_score)

    # Analyze trajectory
    alpha_arr = np.array(alpha_values, dtype=float)
    score_arr = np.array(mean_scores)

    diffs = np.diff(score_arr)
    n_increasing = sum(1 for d in diffs if d > 0)
    monotonicity = n_increasing / len(diffs) if len(diffs) > 0 else 0

    if np.std(score_arr) > 1e-10:
        correlation = float(np.corrcoef(alpha_arr, score_arr)[0, 1])
        r_squared = correlation ** 2
    else:
        correlation = 0.0
        r_squared = 0.0

    pos_score = float(np.mean([s for a, s in zip(alpha_values, mean_scores) if a > 0])) if any(a > 0 for a in alpha_values) else 0.0
    neg_score = float(np.mean([s for a, s in zip(alpha_values, mean_scores) if a < 0])) if any(a < 0 for a in alpha_values) else 0.0
    bidirectional = bool(pos_score * neg_score < 0)

    return {
        "r_squared": round(float(r_squared), 4),
        "monotonicity": round(float(monotonicity), 4),
        "correlation": round(float(correlation), 4),
        "bidirectional": bidirectional,
        "mean_scores": [round(s, 4) for s in mean_scores],
        "alpha_values": [float(a) for a in alpha_values],
    }


def compute_entropy_at_layer(model, tokenizer, info, input_device, prompt, target_layer):
    """Compute next-token distribution entropy at a specific layer using logit lens"""
    import torch
    from model_utils import get_W_U

    W_U = get_W_U(model, info.name if hasattr(info, 'name') else None)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attn_mask = inputs["attention_mask"].to(input_device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                   output_hidden_states=True)

    if target_layer + 1 >= len(out.hidden_states):
        return None

    hs = out.hidden_states[target_layer + 1][0, -1, :].float().cpu().numpy()

    # Logit lens: logits = hs @ W_U^T
    logits = hs @ W_U.T
    max_logit = np.max(logits)
    probs = np.exp(logits - max_logit)
    probs = probs / np.sum(probs)

    entropy = -np.sum(probs * np.log(probs + 1e-30))
    top1_prob = float(np.max(probs))
    effective_support = int(np.sum(probs > 1e-4))

    torch.cuda.empty_cache()

    return {
        "entropy": float(entropy),
        "top1_prob": top1_prob,
        "effective_support": effective_support,
    }


# ============================================================
# Part 1: Random Direction Control Test (THE DECISIVE EXPERIMENT)
# ============================================================

def run_part1(model_name):
    """
    THE DECISIVE EXPERIMENT: Random Direction Control Test
    
    If random directions at low-entropy layers also have R²>0.9,
    then "semantic axis" is just a low-dimensional trivial effect.
    
    If random directions have R²≈0, then true semantic selection exists.
    
    Controls:
    - matched norm (all unit vectors)
    - matched layer (same layer as semantic axis)
    - same readout (same verb pairs)
    - same alpha range
    - 30 random directions for statistical power
    - Also test at MULTIPLE layers with different entropy levels
    """
    import torch
    from model_utils import get_layers, release_model

    log_time(f"=== Part 1: Random Direction Control Test for {model_name} ===")
    log_time(f"  THIS IS THE DECISIVE EXPERIMENT")

    model, tokenizer, info = load_model_bf16(model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)
    n_layers = info.n_layers
    d_model = info.d_model

    # Define layers to test: low entropy (deep), medium entropy (mid), high entropy (early)
    # Based on Phase 265 results:
    layer_configs = {
        "qwen3": {
            "low_entropy": 33,    # effective_support≈2
            "medium_entropy": 25, # effective_support≈1222
            "high_entropy": 9,    # effective_support≈98686
        },
        "glm4": {
            "low_entropy": 39,
            "medium_entropy": 30,
            "high_entropy": 10,
        },
        "deepseek7b": {
            "low_entropy": 27,
            "medium_entropy": 5,
            "high_entropy": 0,
        },
    }

    cfg = layer_configs.get(model_name, {
        "low_entropy": n_layers - 3,
        "medium_entropy": n_layers // 2,
        "high_entropy": max(0, n_layers // 6),
    })

    # Features to test (using number as primary — we already know it's REAL_SEMANTIC_AXIS)
    verb_ids_sing = [safe_get_token_id(tokenizer, v) for v in SING_VERBS]
    verb_ids_plur = [safe_get_token_id(tokenizer, v) for v in PLUR_VERBS]
    verb_ids_sing = [v for v in verb_ids_sing if v is not None]
    verb_ids_plur = [v for v in verb_ids_plur if v is not None]

    animacy_verb_a = [safe_get_token_id(tokenizer, v) for v in ANIMATE_VERBS]
    animacy_verb_b = [safe_get_token_id(tokenizer, v) for v in INANIMATE_VERBS]
    animacy_verb_a = [v for v in animacy_verb_a if v is not None]
    animacy_verb_b = [v for v in animacy_verb_b if v is not None]

    alpha_values = list(np.arange(-10, 10.5, 2.0))  # 11 points
    n_random = 30  # 30 random directions for statistical power

    results = {}

    for entropy_level, target_layer in cfg.items():
        if target_layer >= n_layers:
            log_time(f"  Skipping {entropy_level}: L{target_layer} >= n_layers={n_layers}")
            continue

        log_time(f"\n{'='*60}")
        log_time(f"Entropy level: {entropy_level}, Layer: L{target_layer}")
        log_time(f"{'='*60}")

        # Step 1: Measure actual entropy at this layer
        entropies = []
        for prompt in GENERAL_PROMPTS[:5]:
            ent = compute_entropy_at_layer(model, tokenizer, info, input_device, prompt, target_layer)
            if ent:
                entropies.append(ent)

        if entropies:
            avg_entropy = np.mean([e["entropy"] for e in entropies])
            avg_support = np.mean([e["effective_support"] for e in entropies])
            log_time(f"  Layer entropy: {avg_entropy:.2f}, effective_support: {avg_support:.0f}")
        else:
            log_time(f"  WARNING: Could not measure entropy, continuing...")

        # Step 2: Extract SEMANTIC direction for number
        log_time(f"  Extracting number semantic direction at L{target_layer}...")
        probe_acc, sem_direction = extract_probe_direction(
            model, tokenizer, info, input_device,
            SING_WORDS, PLUR_WORDS, "The {} sits", "The {} sit",
            target_layer, lambda p, off: 1 + off, n_train=25
        )
        if sem_direction is None:
            log_time(f"  WARNING: Could not extract number direction at L{target_layer}")
            continue
        log_time(f"  Number probe_acc: {probe_acc}")

        # Step 3: Measure R² for SEMANTIC direction
        log_time(f"  Measuring R² for SEMANTIC (number) direction...")
        sem_result = measure_intervention_r2(
            model, tokenizer, info, input_device,
            sem_direction, target_layer, NUMBER_TEST,
            verb_ids_sing, verb_ids_plur, alpha_values,
            lambda p, off: 1 + off
        )
        log_time(f"  SEMANTIC R²: {sem_result['r_squared']:.4f}, "
                 f"mono: {sem_result['monotonicity']:.4f}, "
                 f"bidir: {sem_result['bidirectional']}")

        # Step 4: Extract SEMANTIC direction for animacy
        log_time(f"  Extracting animacy semantic direction at L{target_layer}...")
        probe_acc_anim, sem_direction_anim = extract_probe_direction(
            model, tokenizer, info, input_device,
            ANIMATE, INANIMATE, "The {} thinks", "The {} sits",
            target_layer, lambda p, off: 1 + off, n_train=25
        )
        sem_result_anim = None
        if sem_direction_anim is not None:
            log_time(f"  Animacy probe_acc: {probe_acc_anim}")
            log_time(f"  Measuring R² for SEMANTIC (animacy) direction...")
            sem_result_anim = measure_intervention_r2(
                model, tokenizer, info, input_device,
                sem_direction_anim, target_layer,
                ANIMATE_TEST + INANIMATE_TEST,
                animacy_verb_a, animacy_verb_b, alpha_values,
                lambda p, off: 1 + off
            )
            log_time(f"  ANIMACY SEMANTIC R²: {sem_result_anim['r_squared']:.4f}")

        # Step 5: Generate RANDOM directions and measure R²
        log_time(f"  Generating {n_random} random directions and measuring R²...")
        random_r2_values = []
        random_results = []

        for i in range(n_random):
            # Generate random unit vector in d_model space
            random_dir = np.random.randn(d_model).astype(np.float32)
            random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-10)

            rand_result = measure_intervention_r2(
                model, tokenizer, info, input_device,
                random_dir, target_layer, NUMBER_TEST,
                verb_ids_sing, verb_ids_plur, alpha_values,
                lambda p, off: 1 + off
            )
            random_r2_values.append(rand_result['r_squared'])
            random_results.append(rand_result)

            if (i + 1) % 5 == 0:
                log_time(f"    {i+1}/{n_random} random directions done, "
                         f"mean R²={np.mean(random_r2_values):.4f}")

            torch.cuda.empty_cache()

        # Step 6: Also test SUBSPACE-MATCHED random directions
        # These are random directions in the SAME subspace as the semantic direction
        # Generate random directions orthogonal to the semantic direction
        log_time(f"  Generating {n_random//2} orthogonal random directions...")
        ortho_r2_values = []

        for i in range(n_random // 2):
            random_dir = np.random.randn(d_model).astype(np.float32)
            # Remove projection onto semantic direction
            proj = np.dot(random_dir, sem_direction) * sem_direction
            random_dir = random_dir - proj
            norm = np.linalg.norm(random_dir)
            if norm < 1e-10:
                continue
            random_dir = random_dir / norm

            ortho_result = measure_intervention_r2(
                model, tokenizer, info, input_device,
                random_dir, target_layer, NUMBER_TEST,
                verb_ids_sing, verb_ids_plur, alpha_values,
                lambda p, off: 1 + off
            )
            ortho_r2_values.append(ortho_result['r_squared'])

            if (i + 1) % 5 == 0:
                log_time(f"    {i+1}/{n_random//2} orthogonal directions done, "
                         f"mean R²={np.mean(ortho_r2_values):.4f}")

            torch.cuda.empty_cache()

        # Step 7: Statistical test
        random_r2_mean = np.mean(random_r2_values)
        random_r2_std = np.std(random_r2_values)
        random_r2_max = np.max(random_r2_values)
        ortho_r2_mean = np.mean(ortho_r2_values) if ortho_r2_values else 0
        sem_r2 = sem_result['r_squared']

        # Effect size: how many standard deviations is the semantic R² above random?
        if random_r2_std > 1e-10:
            effect_size = (sem_r2 - random_r2_mean) / random_r2_std
        else:
            effect_size = float('inf') if sem_r2 > random_r2_mean else 0

        # Classification
        if random_r2_mean > 0.8:
            verdict = "LOW_DIM_TRIVIAL_EFFECT"
            verdict_detail = "Random directions also have high R² → semantic axis is low-dim artifact"
        elif random_r2_mean < 0.1 and sem_r2 > 0.9:
            verdict = "TRUE_SEMANTIC_SELECTION"
            verdict_detail = "Random R²≈0 but semantic R²>0.9 → genuine semantic mechanism"
        elif random_r2_mean < 0.3 and sem_r2 - random_r2_mean > 0.5:
            verdict = "PARTIALLY_SEMANTIC"
            verdict_detail = "Random R² low-ish, semantic R² much higher → partially genuine"
        else:
            verdict = "MIXED_EFFECT"
            verdict_detail = f"Random R²={random_r2_mean:.3f}, Semantic R²={sem_r2:.3f}"

        layer_result = {
            "layer": int(target_layer),
            "entropy_level": entropy_level,
            "layer_entropy": round(float(avg_entropy), 4) if entropies else None,
            "effective_support": round(float(avg_support), 1) if entropies else None,
            "semantic_r2": round(float(sem_r2), 4),
            "semantic_monotonicity": round(float(sem_result['monotonicity']), 4),
            "semantic_bidirectional": bool(sem_result['bidirectional']),
            "animacy_semantic_r2": round(float(sem_result_anim['r_squared']), 4) if sem_result_anim else None,
            "random_r2_mean": round(float(random_r2_mean), 4),
            "random_r2_std": round(float(random_r2_std), 4),
            "random_r2_max": round(float(random_r2_max), 4),
            "ortho_r2_mean": round(float(ortho_r2_mean), 4) if ortho_r2_values else None,
            "effect_size_sigma": round(float(effect_size), 2),
            "verdict": verdict,
            "n_random": int(n_random),
        }

        results[entropy_level] = layer_result

        log_time(f"\n  *** RESULT for {entropy_level} (L{target_layer}) ***")
        log_time(f"  Semantic (number) R²: {sem_r2:.4f}")
        if sem_result_anim:
            log_time(f"  Semantic (animacy) R²: {sem_result_anim['r_squared']:.4f}")
        log_time(f"  Random R²: mean={random_r2_mean:.4f}, std={random_r2_std:.4f}, max={random_r2_max:.4f}")
        log_time(f"  Orthogonal R²: mean={ortho_r2_mean:.4f}")
        log_time(f"  Effect size: {effect_size:.1f}σ")
        log_time(f"  VERDICT: {verdict}")
        log_time(f"  Detail: {verdict_detail}")

    # Overall verdict
    log_time(f"\n{'='*60}")
    log_time(f"OVERALL VERDICT for {model_name}")
    log_time(f"{'='*60}")

    for level, res in results.items():
        log_time(f"  {level} (L{res['layer']}): "
                 f"Semantic R²={res['semantic_r2']:.4f}, "
                 f"Random R²={res['random_r2_mean']:.4f}, "
                 f"Effect={res['effect_size_sigma']}σ → {res['verdict']}")

    # Cross-layer comparison
    low_r = results.get("low_entropy", {})
    med_r = results.get("medium_entropy", {})
    high_r = results.get("high_entropy", {})

    if low_r and med_r:
        gap_low = low_r["semantic_r2"] - low_r["random_r2_mean"]
        gap_med = med_r["semantic_r2"] - med_r["random_r2_mean"]

        log_time(f"\n  CRITICAL COMPARISON:")
        log_time(f"  Low-entropy layer:  Semantic-gap = {gap_low:.4f}")
        log_time(f"  Medium-entropy layer: Semantic-gap = {gap_med:.4f}")

        if gap_low < gap_med * 0.3:
            log_time(f"  → Gap SHRINKS at low entropy → partially low-dim effect")
        elif gap_low > gap_med:
            log_time(f"  → Gap GROWS at low entropy → semantic selection strengthens")
        else:
            log_time(f"  → Gap STABLE → semantic selection is consistent")

    # Save results
    result_file = RESULT_DIR / f"part1_{model_name}_random_direction.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_time(f"  Results saved to {result_file}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================
# Part 2: Multi-Layer Entropy-Matched Control
# ============================================================

def run_part2(model_name):
    """
    Test at ALL layers from 0 to n-1, measuring both:
    1. Entropy at each layer
    2. Semantic direction R² vs Random direction R²
    
    This gives us the FULL PICTURE of how the gap evolves.
    """
    import torch
    from model_utils import get_layers, release_model

    log_time(f"=== Part 2: Full-Layer Entropy-Matched Control for {model_name} ===")

    model, tokenizer, info = load_model_bf16(model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)
    n_layers = info.n_layers
    d_model = info.d_model

    verb_ids_sing = [safe_get_token_id(tokenizer, v) for v in SING_VERBS]
    verb_ids_plur = [safe_get_token_id(tokenizer, v) for v in PLUR_VERBS]
    verb_ids_sing = [v for v in verb_ids_sing if v is not None]
    verb_ids_plur = [v for v in verb_ids_plur if v is not None]

    alpha_values = list(np.arange(-10, 10.5, 2.0))
    n_random_per_layer = 15  # 15 random directions per layer

    # Sample layers (every 3rd layer + key layers)
    all_layers = list(range(0, n_layers, 3))
    if (n_layers - 1) not in all_layers:
        all_layers.append(n_layers - 1)
    all_layers = sorted(set(all_layers))

    log_time(f"  Testing {len(all_layers)} layers: {all_layers}")

    layer_results = []

    for layer_idx in all_layers:
        log_time(f"\n  --- Layer {layer_idx} ---")

        # Step 1: Measure entropy
        entropies = []
        for prompt in GENERAL_PROMPTS[:3]:
            ent = compute_entropy_at_layer(model, tokenizer, info, input_device, prompt, layer_idx)
            if ent:
                entropies.append(ent)

        avg_entropy = np.mean([e["entropy"] for e in entropies]) if entropies else None
        avg_support = np.mean([e["effective_support"] for e in entropies]) if entropies else None
        log_time(f"    Entropy: {avg_entropy:.2f}, Support: {avg_support:.0f}" if avg_entropy else "    Entropy: N/A")

        # Step 2: Extract semantic direction
        probe_acc, sem_direction = extract_probe_direction(
            model, tokenizer, info, input_device,
            SING_WORDS, PLUR_WORDS, "The {} sits", "The {} sit",
            layer_idx, lambda p, off: 1 + off, n_train=20
        )

        if sem_direction is None:
            log_time(f"    No valid probe direction, skipping")
            layer_results.append({
                "layer": int(layer_idx),
                "entropy": round(float(avg_entropy), 4) if avg_entropy else None,
                "support": round(float(avg_support), 1) if avg_support else None,
                "semantic_r2": None,
                "random_r2_mean": None,
                "probe_acc": None,
            })
            continue

        # Step 3: Measure semantic R²
        sem_result = measure_intervention_r2(
            model, tokenizer, info, input_device,
            sem_direction, layer_idx, NUMBER_TEST,
            verb_ids_sing, verb_ids_plur, alpha_values,
            lambda p, off: 1 + off
        )
        sem_r2 = sem_result['r_squared']
        log_time(f"    Semantic R²: {sem_r2:.4f}")

        # Step 4: Measure random R²
        random_r2s = []
        for i in range(n_random_per_layer):
            random_dir = np.random.randn(d_model).astype(np.float32)
            random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-10)

            rand_result = measure_intervention_r2(
                model, tokenizer, info, input_device,
                random_dir, layer_idx, NUMBER_TEST,
                verb_ids_sing, verb_ids_plur, alpha_values,
                lambda p, off: 1 + off
            )
            random_r2s.append(rand_result['r_squared'])
            torch.cuda.empty_cache()

        random_mean = np.mean(random_r2s)
        random_std = np.std(random_r2s)
        gap = sem_r2 - random_mean

        log_time(f"    Random R²: mean={random_mean:.4f}, std={random_std:.4f}")
        log_time(f"    GAP (semantic - random): {gap:.4f}")

        layer_results.append({
            "layer": int(layer_idx),
            "entropy": round(float(avg_entropy), 4) if avg_entropy else None,
            "support": round(float(avg_support), 1) if avg_support else None,
            "probe_acc": round(float(probe_acc), 4) if probe_acc else None,
            "semantic_r2": round(float(sem_r2), 4),
            "semantic_mono": round(float(sem_result['monotonicity']), 4),
            "semantic_bidir": bool(sem_result['bidirectional']),
            "random_r2_mean": round(float(random_mean), 4),
            "random_r2_std": round(float(random_std), 4),
            "gap": round(float(gap), 4),
            "n_random": int(n_random_per_layer),
        })

        # Periodic memory cleanup
        if layer_idx % 6 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Analyze the evolution
    log_time(f"\n{'='*60}")
    log_time(f"FULL-LAYER EVOLUTION for {model_name}")
    log_time(f"{'='*60}")

    valid_results = [r for r in layer_results if r["semantic_r2"] is not None]
    for r in valid_results:
        bar_sem = "█" * int(r["semantic_r2"] * 20)
        bar_rand = "░" * int(r["random_r2_mean"] * 20) if r["random_r2_mean"] else ""
        log_time(f"  L{r['layer']:2d}: Sem={r['semantic_r2']:.3f} {bar_sem} | "
                 f"Rnd={r['random_r2_mean']:.3f} {bar_rand} | "
                 f"Gap={r['gap']:.3f} | "
                 f"H={r['entropy']:.1f}" if r['entropy'] else
                 f"  L{r['layer']:2d}: Sem={r['semantic_r2']:.3f} {bar_sem} | "
                 f"Rnd={r['random_r2_mean']:.3f} {bar_rand} | "
                 f"Gap={r['gap']:.3f}")

    # Find the transition point where gap becomes significant
    transition_layer = None
    for r in valid_results:
        if r["gap"] > 0.5 and r["semantic_r2"] > 0.9:
            transition_layer = r["layer"]
            break

    if transition_layer:
        log_time(f"\n  Transition layer (gap>0.5, sem>0.9): L{transition_layer}")
    else:
        log_time(f"\n  No clear transition found")

    # Save results
    result_file = RESULT_DIR / f"part2_{model_name}_full_layer_evolution.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(layer_results, f, indent=2, ensure_ascii=False)
    log_time(f"  Results saved to {result_file}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    return layer_results


# ============================================================
# Part 3: Multi-Direction Joint Control (Intervention Capacity)
# ============================================================

def run_part3(model_name):
    """
    Test whether multiple features can be simultaneously and independently controlled.
    
    If number and animacy directions are orthogonal and independently controllable,
    it supports the "genuine feature decomposition" theory.
    
    If they interfere, it suggests features share the same low-dimensional manifold.
    """
    import torch
    from model_utils import get_layers, release_model

    log_time(f"=== Part 3: Multi-Direction Joint Control for {model_name} ===")

    model, tokenizer, info = load_model_bf16(model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)
    n_layers = info.n_layers
    d_model = info.d_model

    # Target layer: deep layer where we know semantic axes exist
    target_layers = {
        "qwen3": [30, 35],
        "glm4": [36, 39],
        "deepseek7b": [27],
    }
    test_layers = target_layers.get(model_name, [n_layers - 3])

    verb_ids_sing = [safe_get_token_id(tokenizer, v) for v in SING_VERBS]
    verb_ids_plur = [safe_get_token_id(tokenizer, v) for v in PLUR_VERBS]
    animacy_verb_a = [safe_get_token_id(tokenizer, v) for v in ANIMATE_VERBS]
    animacy_verb_b = [safe_get_token_id(tokenizer, v) for v in INANIMATE_VERBS]
    verb_ids_sing = [v for v in verb_ids_sing if v is not None]
    verb_ids_plur = [v for v in verb_ids_plur if v is not None]
    animacy_verb_a = [v for v in animacy_verb_a if v is not None]
    animacy_verb_b = [v for v in animacy_verb_b if v is not None]

    results = {}

    for target_layer in test_layers:
        if target_layer >= n_layers:
            continue

        log_time(f"\n  --- Joint Control at L{target_layer} ---")

        # Extract directions
        _, number_dir = extract_probe_direction(
            model, tokenizer, info, input_device,
            SING_WORDS, PLUR_WORDS, "The {} sits", "The {} sit",
            target_layer, lambda p, off: 1 + off, n_train=25
        )
        _, animacy_dir = extract_probe_direction(
            model, tokenizer, info, input_device,
            ANIMATE, INANIMATE, "The {} thinks", "The {} sits",
            target_layer, lambda p, off: 1 + off, n_train=25
        )

        if number_dir is None or animacy_dir is None:
            log_time(f"  Could not extract both directions, skipping L{target_layer}")
            continue

        # Measure cosine between the two semantic directions
        cos_between = float(np.dot(number_dir, animacy_dir) /
                           (np.linalg.norm(number_dir) * np.linalg.norm(animacy_dir)))
        log_time(f"  Cosine between number & animacy: {cos_between:.4f}")

        # Test 1: Single direction control (baseline)
        log_time(f"  Test 1: Single direction baseline...")

        number_only = measure_intervention_r2(
            model, tokenizer, info, input_device,
            number_dir, target_layer, NUMBER_TEST,
            verb_ids_sing, verb_ids_plur,
            list(np.arange(-10, 10.5, 2.0)),
            lambda p, off: 1 + off
        )

        animacy_test_prompts = ANIMATE_TEST + INANIMATE_TEST
        animacy_only = measure_intervention_r2(
            model, tokenizer, info, input_device,
            animacy_dir, target_layer, animacy_test_prompts,
            animacy_verb_a, animacy_verb_b,
            list(np.arange(-10, 10.5, 2.0)),
            lambda p, off: 1 + off
        )

        log_time(f"  Number-only R²: {number_only['r_squared']:.4f}")
        log_time(f"  Animacy-only R²: {animacy_only['r_squared']:.4f}")

        # Test 2: Cross-interference — inject number direction, check animacy readout
        log_time(f"  Test 2: Cross-interference (number→animacy)...")
        number_on_animacy = measure_intervention_r2(
            model, tokenizer, info, input_device,
            number_dir, target_layer, animacy_test_prompts,
            animacy_verb_a, animacy_verb_b,
            list(np.arange(-10, 10.5, 2.0)),
            lambda p, off: 1 + off
        )
        log_time(f"  Number direction → animacy readout R²: {number_on_animacy['r_squared']:.4f}")

        # Test 3: Cross-interference — inject animacy direction, check number readout
        log_time(f"  Test 3: Cross-interference (animacy→number)...")
        animacy_on_number = measure_intervention_r2(
            model, tokenizer, info, input_device,
            animacy_dir, target_layer, NUMBER_TEST,
            verb_ids_sing, verb_ids_plur,
            list(np.arange(-10, 10.5, 2.0)),
            lambda p, off: 1 + off
        )
        log_time(f"  Animacy direction → number readout R²: {animacy_on_number['r_squared']:.4f}")

        # Test 4: Joint injection
        log_time(f"  Test 4: Joint injection (number + animacy)...")
        # Use prompts that are ambiguous (neutral nouns)
        joint_test_prompts = ["The thing", "The object", "The item", "The entity", "The one"]
        alpha_values_joint = list(np.arange(-10, 10.5, 2.0))

        # Joint: inject both directions simultaneously
        # Measure both readouts
        joint_number_scores = []
        joint_animacy_scores = []

        for alpha in alpha_values_joint:
            n_changes = []
            a_changes = []

            for prompt in joint_test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)
                offset = get_special_token_offset(tokenizer, prompt)
                subj_pos = 1 + offset
                n_tokens = input_ids.shape[1]
                if subj_pos < 0:
                    subj_pos = n_tokens + subj_pos

                # Baseline
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask)
                bl = out.logits[0, -1, :].float().cpu().numpy()
                bl_n = float(np.mean([bl[tid] for tid in verb_ids_sing if tid < len(bl)])) - \
                       float(np.mean([bl[tid] for tid in verb_ids_plur if tid < len(bl)]))
                bl_a = float(np.mean([bl[tid] for tid in animacy_verb_a if tid < len(bl)])) - \
                       float(np.mean([bl[tid] for tid in animacy_verb_b if tid < len(bl)]))

                # Joint injection
                n_dir_t = torch.tensor(number_dir, dtype=model.dtype,
                                       device=next(layers[0].parameters()).device)
                a_dir_t = torch.tensor(animacy_dir, dtype=model.dtype,
                                       device=next(layers[0].parameters()).device)

                def make_joint_hook(nd, ad, a_val, p):
                    def hook(module, inp, output):
                        if isinstance(output, tuple):
                            modified = output[0].clone()
                            modified[:, p, :] += a_val * nd.to(modified.device)
                            modified[:, p, :] += a_val * ad.to(modified.device)
                            return (modified,) + output[1:]
                        else:
                            modified = output.clone()
                            modified[:, p, :] += a_val * nd.to(modified.device)
                            modified[:, p, :] += a_val * ad.to(modified.device)
                            return modified
                    return hook

                hook = layers[target_layer].register_forward_hook(
                    make_joint_hook(n_dir_t, a_dir_t, alpha, subj_pos))

                try:
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attn_mask)
                except Exception:
                    hook.remove()
                    continue
                hook.remove()

                lg = out.logits[0, -1, :].float().cpu().numpy()
                n_score = float(np.mean([lg[tid] for tid in verb_ids_sing if tid < len(lg)])) - \
                          float(np.mean([lg[tid] for tid in verb_ids_plur if tid < len(lg)]))
                a_score = float(np.mean([lg[tid] for tid in animacy_verb_a if tid < len(lg)])) - \
                          float(np.mean([lg[tid] for tid in animacy_verb_b if tid < len(lg)]))

                n_changes.append(n_score - bl_n)
                a_changes.append(a_score - bl_a)
                torch.cuda.empty_cache()

            joint_number_scores.append(float(np.mean(n_changes)) if n_changes else 0)
            joint_animacy_scores.append(float(np.mean(a_changes)) if a_changes else 0)

        # Analyze joint results
        alpha_arr = np.array(alpha_values_joint, dtype=float)
        n_arr = np.array(joint_number_scores)
        a_arr = np.array(joint_animacy_scores)

        def compute_r2(x, y):
            if np.std(y) < 1e-10:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1] ** 2)

        joint_number_r2 = compute_r2(alpha_arr, n_arr)
        joint_animacy_r2 = compute_r2(alpha_arr, a_arr)

        log_time(f"  Joint injection: number R²={joint_number_r2:.4f}, animacy R²={joint_animacy_r2:.4f}")

        # Summary
        cross_interference = max(number_on_animacy['r_squared'], animacy_on_number['r_squared'])

        if cross_interference < 0.1:
            ortho_verdict = "ORTHOGONAL_CONTROL"
            ortho_detail = "Features are independently controllable → genuine decomposition"
        elif cross_interference < 0.3:
            ortho_verdict = "PARTIALLY_ORTHOGONAL"
            ortho_detail = "Features mostly independent with some leakage"
        else:
            ortho_verdict = "SHARED_MANIFOLD"
            ortho_detail = "Features interfere → share same low-dim manifold"

        layer_result = {
            "layer": int(target_layer),
            "cos_between_features": round(float(cos_between), 4),
            "number_only_r2": round(float(number_only['r_squared']), 4),
            "animacy_only_r2": round(float(animacy_only['r_squared']), 4),
            "number_on_animacy_r2": round(float(number_on_animacy['r_squared']), 4),
            "animacy_on_number_r2": round(float(animacy_on_number['r_squared']), 4),
            "cross_interference": round(float(cross_interference), 4),
            "joint_number_r2": round(float(joint_number_r2), 4),
            "joint_animacy_r2": round(float(joint_animacy_r2), 4),
            "ortho_verdict": ortho_verdict,
        }
        results[f"L{target_layer}"] = layer_result

        log_time(f"\n  *** JOINT CONTROL RESULT at L{target_layer} ***")
        log_time(f"  Cos between features: {cos_between:.4f}")
        log_time(f"  Number-only R²: {number_only['r_squared']:.4f}")
        log_time(f"  Animacy-only R²: {animacy_only['r_squared']:.4f}")
        log_time(f"  Cross-interference: {cross_interference:.4f}")
        log_time(f"  Verdict: {ortho_verdict} — {ortho_detail}")

    # Save results
    result_file = RESULT_DIR / f"part3_{model_name}_joint_control.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_time(f"  Results saved to {result_file}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================
# Part 4: Local Effective Dimension (Participation Ratio)
# ============================================================

def run_part4(model_name):
    """
    Measure the true dimensionality of hidden state space at each layer.
    
    Methods:
    1. Participation Ratio (PR) = (Σλ_i)² / Σλ_i²
       where λ_i are eigenvalues of the covariance matrix
    2. Effective Rank = number of eigenvalues needed to explain 95% variance
    3. Local intrinsic dimension via PCA
    
    This directly tests: "Are deep layers truly low-dimensional?"
    """
    import torch
    from model_utils import get_layers, release_model

    log_time(f"=== Part 4: Local Effective Dimension for {model_name} ===")

    model, tokenizer, info = load_model_bf16(model_name)
    input_device = get_input_device(model)
    n_layers = info.n_layers
    d_model = info.d_model

    # Collect hidden states from many diverse prompts
    diverse_prompts = GENERAL_PROMPTS + [
        "The cat sat on the", "A large tree grows", "She never expected the",
        "Water flows downhill through", "The king commanded his", "Music filled the",
        "Stars shine bright in", "Food was prepared by", "The engine started to",
        "Knowledge comes from", "Freedom requires", "Time passes like",
        "Mountains rise above the", "The story begins with", "He found the answer",
    ]

    # Sample layers
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if (n_layers - 1) not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    log_time(f"  Sampling {len(sample_layers)} layers with {len(diverse_prompts)} prompts")

    layer_dims = {}

    for target_layer in sample_layers:
        log_time(f"  Processing L{target_layer}...")

        # Collect hidden states
        all_vectors = []

        for prompt in diverse_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
            except Exception:
                continue

            if target_layer + 1 < len(out.hidden_states):
                # Get last token position
                hs = out.hidden_states[target_layer + 1][0, -1, :].float().cpu().numpy()
                all_vectors.append(hs)

            torch.cuda.empty_cache()

        if len(all_vectors) < 10:
            log_time(f"    Not enough vectors ({len(all_vectors)}), skipping")
            continue

        # Stack into matrix
        X = np.array(all_vectors)  # [n_prompts, d_model]
        n_samples = X.shape[0]

        # Center the data
        X_centered = X - X.mean(axis=0, keepdims=True)

        # Method 1: Participation Ratio
        # Covariance matrix: C = X^T X / n
        # But d_model might be large, so use SVD of X instead
        # X = U S V^T, eigenvalues of C = S^2 / n
        try:
            from scipy.sparse.linalg import svds
            k = min(100, min(n_samples - 1, d_model) - 1)
            k = max(k, 10)
            U, S, Vt = svds(X_centered, k=k)
            S = np.sort(S)[::-1]  # Sort descending
            eigenvalues = S ** 2 / n_samples

            # Participation Ratio = (Σλ)² / Σλ²
            sum_lambda = np.sum(eigenvalues)
            sum_lambda_sq = np.sum(eigenvalues ** 2)
            pr = sum_lambda ** 2 / max(sum_lambda_sq, 1e-30)

            # Effective Rank (95% variance)
            cumvar = np.cumsum(eigenvalues) / sum_lambda
            eff_rank_95 = int(np.searchsorted(cumvar, 0.95)) + 1
            eff_rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1

            # Top-k energy distribution
            top1_energy = eigenvalues[0] / sum_lambda
            top5_energy = np.sum(eigenvalues[:5]) / sum_lambda
            top10_energy = np.sum(eigenvalues[:10]) / sum_lambda

        except Exception as e:
            log_time(f"    SVD failed: {e}")
            pr = 0
            eff_rank_95 = 0
            eff_rank_99 = 0
            top1_energy = 0
            top5_energy = 0
            top10_energy = 0

        layer_dims[f"L{target_layer}"] = {
            "layer": int(target_layer),
            "n_samples": int(n_samples),
            "participation_ratio": round(float(pr), 2),
            "effective_rank_95": int(eff_rank_95),
            "effective_rank_99": int(eff_rank_99),
            "top1_energy": round(float(top1_energy), 4),
            "top5_energy": round(float(top5_energy), 4),
            "top10_energy": round(float(top10_energy), 4),
        }

        log_time(f"    PR={pr:.1f}, EffRank95={eff_rank_95}, "
                 f"Top1={top1_energy:.3f}, Top5={top5_energy:.3f}, Top10={top10_energy:.3f}")

        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    log_time(f"\n{'='*60}")
    log_time(f"DIMENSION EVOLUTION for {model_name}")
    log_time(f"{'='*60}")

    for key, dims in sorted(layer_dims.items(), key=lambda x: x[1]["layer"]):
        log_time(f"  {key}: PR={dims['participation_ratio']:.1f}, "
                 f"EffRank95={dims['effective_rank_95']}, "
                 f"Top10%={dims['top10_energy']:.1%}")

    # Compare early vs deep layers
    early_layers = {k: v for k, v in layer_dims.items() if v["layer"] < n_layers // 3}
    deep_layers = {k: v for k, v in layer_dims.items() if v["layer"] > 2 * n_layers // 3}

    if early_layers and deep_layers:
        early_pr = np.mean([v["participation_ratio"] for v in early_layers.values()])
        deep_pr = np.mean([v["participation_ratio"] for v in deep_layers.values()])
        early_rank = np.mean([v["effective_rank_95"] for v in early_layers.values()])
        deep_rank = np.mean([v["effective_rank_95"] for v in deep_layers.values()])

        log_time(f"\n  Early layers: PR={early_pr:.1f}, EffRank95={early_rank:.1f}")
        log_time(f"  Deep layers: PR={deep_pr:.1f}, EffRank95={deep_rank:.1f}")

        if deep_pr < early_pr * 0.5:
            log_time(f"  → Deep layers are GENUINELY low-dimensional (PR drops >50%)")
        elif deep_pr < early_pr * 0.8:
            log_time(f"  → Deep layers are SOMEWHAT lower-dimensional")
        else:
            log_time(f"  → Deep layers maintain similar dimensionality")

    # Save results
    result_file = RESULT_DIR / f"part4_{model_name}_effective_dimension.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(layer_dims, f, indent=2, ensure_ascii=False)
    log_time(f"  Results saved to {result_file}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    return layer_dims


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 266: Semantic Axis vs Low-Dim Artifact")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--part", type=int, required=True,
                       choices=[1, 2, 3, 4],
                       help="Part number (1=random control, 2=full-layer, 3=joint, 4=dimension)")
    args = parser.parse_args()

    log_time(f"Phase 266: {args.model} Part {args.part}")
    log_time(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.part == 1:
        run_part1(args.model)
    elif args.part == 2:
        run_part2(args.model)
    elif args.part == 3:
        run_part3(args.model)
    elif args.part == 4:
        run_part4(args.model)

    log_time(f"Phase 266 Part {args.part} for {args.model} complete!")
    log_time(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
