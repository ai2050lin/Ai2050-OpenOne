"""
Phase 265: Constraint Geometry — Multi-Feature MLP Intervention + Future Space Entropy + Constraint Conflict
============================================================================================================

Combining the most critical gaps identified by both analyses:

  Part 1 (265a): Multi-Feature Deep MLP Intervention
    - Test animacy, tense, concreteness at deep layers (not just number!)
    - CRITICAL: Is "deep semantic axis" a universal principle or a number-specific artifact?
    - If animacy/tense also reach R²>0.9 at deep layers → universal principle
    - If not → number is special, different features have different encoding mechanisms

  Part 2 (265b): Future Space Entropy Collapse
    - Measure how the token probability distribution narrows across layers
    - At each layer, compute entropy of the "next-token distribution"
    - Key test for Analysis 1's "constraint propagation" theory:
      * If entropy drops dramatically in deep layers → constraint convergence
      * If entropy stays constant → deep linearization is NOT from convergence

  Part 3 (265c): Constraint Conflict Experiment
    - Grammatically inconsistent inputs: "The rocks eats..."
    - Track which layer detects the conflict, which layer tries to fix it
    - Directly observe constraint propagation in action

Usage:
  python tests/glm5/phase265_constraint_geometry.py --model qwen3 --part 1
  python tests/glm5/phase265_constraint_geometry.py --model glm4 --part 2
  python tests/glm5/phase265_constraint_geometry.py --model deepseek7b --part 3
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase265_constraint_geometry")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Word lists for multi-feature probing =====

# Number (same as Phase 264)
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

# Animacy
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

# Animacy test: animate subjects can "do voluntary actions", inanimate cannot
# Use verb preference as readout: animate → "thinks/feels/runs", inanimate → "sits/lies/stands"
ANIMATE_VERBS = ["thinks", "feels", "runs", "walks", "speaks", "believes", "decides", "wants", "loves", "hopes"]
INANIMATE_VERBS = ["sits", "lies", "stands", "rests", "hangs", "falls", "rolls", "breaks", "cracks", "shines"]

TEST_ANIMATE = ["puppy", "kitten", "parrot", "salmon", "infant", "lady", "gentleman", "scholar"]
TEST_INANIMATE = ["boulder", "couch", "novel", "truck", "candle", "boulder", "marble", "crystal"]

# Tense
PRESENT_WORDS = [
    "runs", "walks", "sits", "eats", "makes", "takes", "gives",
    "comes", "goes", "sees", "knows", "thinks", "says", "gets",
    "finds", "tells", "asks", "seems", "feels", "leaves", "calls",
]
PAST_WORDS = [
    "ran", "walked", "sat", "ate", "made", "took", "gave",
    "came", "went", "saw", "knew", "thought", "said", "got",
    "found", "told", "asked", "seemed", "felt", "left", "called",
]

# Tense readout: present-tense contexts → present verbs, past-tense contexts → past verbs
PRESENT_CONTEXT_VERBS = ["is", "does", "has", "goes", "comes", "makes", "takes"]
PAST_CONTEXT_VERBS = ["was", "did", "had", "went", "came", "made", "took"]

# Concreteness
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

# Concreteness test: concrete → "holds/touches/sees", abstract → "understands/believes/feels"
CONCRETE_VERBS = ["holds", "touches", "sees", "carries", "drops", "picks", "throws", "catches", "lifts", "moves"]
ABSTRACT_VERBS = ["understands", "believes", "feels", "thinks", "knows", "means", "represents", "expresses", "defines", "explains"]

# Sentiment (for additional validation)
POSITIVE = [
    "joy", "happiness", "love", "hope", "peace", "success", "victory",
    "health", "beauty", "kindness", "smile", "laugh", "gift", "friend",
    "sunshine", "bloom", "warmth", "comfort", "dream", "wonder",
]
NEGATIVE = [
    "sadness", "pain", "hate", "despair", "war", "failure", "defeat",
    "disease", "ugliness", "cruelty", "frown", "cry", "loss", "enemy",
    "storm", "decay", "cold", "suffering", "nightmare", "horror",
]

POSITIVE_VERBS = ["loves", "enjoys", "celebrates", "appreciates", "embraces"]
NEGATIVE_VERBS = ["hates", "suffers", "fears", "rejects", "avoids"]

# Constraint conflict examples
CONFLICT_PAIRS = [
    # (grammatical_singular, grammatical_plural, conflict_singular_verb, conflict_plural_verb)
    ("The cat sits", "The cats sit", "The cat sit", "The cats sits"),
    ("The dog runs", "The dogs run", "The dog run", "The dogs runs"),
    ("The bird flies", "The birds fly", "The bird fly", "The birds flies"),
    ("The child walks", "The children walk", "The child walk", "The children walks"),
    ("The woman is", "The women are", "The woman are", "The women is"),
    ("The man has", "The men have", "The man have", "The men has"),
    ("The fish swims", "The fish swim", "The fishes swims", "The fish swims"),
    ("The hero goes", "The heroes go", "The hero go", "The heroes goes"),
]

# Animacy constraint conflicts
ANIMACY_CONFLICT = [
    # (animate_verb, inanimate_verb)
    ("The rock thinks", "The dog sits"),   # Inanimate + animate verb
    ("The table believes", "The child rests"),  # Inanimate + animate verb
    ("The stone decides", "The cat lies"),  # Inanimate + animate verb
    ("The chair wants", "The bird stands"),  # Inanimate + animate verb
    ("The cloud hopes", "The fish swims"),  # Inanimate + animate verb
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

    n_heads = getattr(model.config, 'num_attention_heads', 32)
    head_dim = getattr(model.config, 'head_dim', info.d_model // n_heads)

    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  class={info.model_class}, layers={info.n_layers}, d_model={info.d_model}, "
             f"GPU={gpu_mem:.2f}GB")

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


def collect_hidden_states_at_layers(model, tokenizer, prompts, positions, target_layers,
                                     input_device, batch_size=5):
    """Collect hidden states at specific layers for a list of prompts"""
    import torch

    all_states = {l: [] for l in target_layers}

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_positions = positions[i:i+batch_size]

        for prompt, pos in zip(batch_prompts, batch_positions):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
            except Exception:
                continue

            if out.hidden_states:
                for l in target_layers:
                    if l < len(out.hidden_states):
                        hs = out.hidden_states[l][0, pos, :].float().cpu().numpy()
                        all_states[l].append(hs)

            torch.cuda.empty_cache()

    return all_states


def extract_probe_directions_for_feature(model, tokenizer, info, input_device,
                                          feature_name, class_a_words, class_b_words,
                                          prompt_template_a, prompt_template_b,
                                          pos_func, n_train=25):
    """Extract probe directions at all layers for a given feature"""
    import torch

    n_layers = info.n_layers
    d_model = info.d_model
    sampled_layers = get_sampled_layers(n_layers, n_sample=8)

    log_time(f"  Extracting {feature_name} directions at {len(sampled_layers)} layers...")

    # Collect hidden states
    class_a_states = {l: [] for l in range(n_layers + 1)}
    class_b_states = {l: [] for l in range(n_layers + 1)}

    for a_word, b_word in zip(class_a_words[:n_train], class_b_words[:n_train]):
        for word, label in [(a_word, "A"), (b_word, "B")]:
            if label == "A":
                prompt = prompt_template_a.format(word)
            else:
                prompt = prompt_template_b.format(word)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            offset = get_special_token_offset(tokenizer, prompt)
            subj_pos = pos_func(prompt, offset)

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
            except Exception:
                continue

            if out.hidden_states:
                for l in range(len(out.hidden_states)):
                    hs = out.hidden_states[l][0, subj_pos, :].float().cpu().numpy()
                    if label == "A":
                        class_a_states[l].append(hs)
                    else:
                        class_b_states[l].append(hs)

        torch.cuda.empty_cache()

    # Train probes
    directions = {}
    accuracies = {}
    for l in range(n_layers + 1):
        data_dict = {"A": class_a_states[l], "B": class_b_states[l]}
        acc, direction = train_probe(data_dict)
        if direction is not None:
            directions[l] = direction
            accuracies[l] = acc

    log_time(f"  {feature_name}: {len(directions)} layers with valid directions")

    return directions, accuracies


# ============================================================
# Part 1: Multi-Feature Deep MLP Intervention
# ============================================================

def run_part1(model_name):
    """
    CRITICAL: Test animacy, tense, concreteness, sentiment at deep layers.
    
    Phase 264 found number direction has R²>0.9 at deep MLP layers.
    But number is the simplest binary feature. Is this universal?
    
    For each feature:
    1. Extract probe direction at all layers
    2. Inject at deep layers (L{n-5}, L{n-3}, L{n-1}) and shallow layers
    3. Measure R², monotonicity, bidirectional control
    4. Classify as REAL_SEMANTIC_AXIS or LOCAL_STATISTICAL_BOUNDARY
    """
    import torch
    from model_utils import get_layers

    log_time(f"=== Part 1: Multi-Feature Deep MLP Intervention for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim = load_model_bf16(model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)
    embed_layer = model.get_input_embeddings()

    n_layers = info.n_layers
    d_model = info.d_model
    sampled_layers = get_sampled_layers(n_layers, n_sample=8)

    # Deep layers to test (last 30% of layers)
    deep_start = int(n_layers * 0.7)
    deep_layers = [l for l in sampled_layers if l >= deep_start]
    # Also include some mid/shallow layers for comparison
    shallow_layers = [l for l in sampled_layers if l < deep_start]
    test_layers = shallow_layers[-2:] + deep_layers  # Last 2 shallow + all deep
    test_layers = sorted(set(test_layers))

    log_time(f"  n_layers={n_layers}, test_layers={test_layers}")

    # Define features with their prompt templates, position functions, and verb readouts
    features = {}

    # Feature 1: Number (baseline - should replicate Phase 264)
    features["number"] = {
        "class_a_words": SING_WORDS,
        "class_b_words": PLUR_WORDS,
        "prompt_a": "The {} sits",   # singular + singular verb
        "prompt_b": "The {} sit",     # plural + plural verb
        "pos_func": lambda p, off: 1 + off,  # subject position
        "verb_ids_a": [safe_get_token_id(tokenizer, v) for v in SING_VERBS],
        "verb_ids_b": [safe_get_token_id(tokenizer, v) for v in PLUR_VERBS],
        "test_prompts": [f"The {w}" for w in TEST_SING[:10]],
        "test_pos_func": lambda p, off: 1 + off,
        "readout_name": "singular - plural verb",
    }

    # Feature 2: Animacy
    features["animacy"] = {
        "class_a_words": ANIMATE,
        "class_b_words": INANIMATE,
        "prompt_a": "The {} thinks",  # animate + animate verb
        "prompt_b": "The {} sits",    # inanimate + inanimate verb
        "pos_func": lambda p, off: 1 + off,
        "verb_ids_a": [safe_get_token_id(tokenizer, v) for v in ANIMATE_VERBS],
        "verb_ids_b": [safe_get_token_id(tokenizer, v) for v in INANIMATE_VERBS],
        "test_prompts": [f"The {w}" for w in TEST_ANIMATE[:5]] + [f"The {w}" for w in TEST_INANIMATE[:5]],
        "test_pos_func": lambda p, off: 1 + off,
        "readout_name": "animate - inanimate verb",
    }

    # Feature 3: Tense
    features["tense"] = {
        "class_a_words": PRESENT_WORDS,
        "class_b_words": PAST_WORDS,
        "prompt_a": "Today it {}",    # present context
        "prompt_b": "Yesterday it {}", # past context
        "pos_func": lambda p, off: -1 + off,  # verb position (last token)
        "verb_ids_a": [safe_get_token_id(tokenizer, v) for v in PRESENT_CONTEXT_VERBS],
        "verb_ids_b": [safe_get_token_id(tokenizer, v) for v in PAST_CONTEXT_VERBS],
        "test_prompts": [
            "Every day the cat", "Each morning the dog",
            "Always the bird", "Regularly the student",
            "Today the teacher",
        ],
        "test_pos_func": lambda p, off: -1 + off,
        "readout_name": "present - past verb",
    }

    # Feature 4: Concreteness
    features["concreteness"] = {
        "class_a_words": CONCRETE,
        "class_b_words": ABSTRACT,
        "prompt_a": "The {} weighs",   # concrete + physical verb
        "prompt_b": "The {} matters",  # abstract + mental verb
        "pos_func": lambda p, off: 1 + off,
        "verb_ids_a": [safe_get_token_id(tokenizer, v) for v in CONCRETE_VERBS],
        "verb_ids_b": [safe_get_token_id(tokenizer, v) for v in ABSTRACT_VERBS],
        "test_prompts": [
            f"The {w}" for w in CONCRETE[:5]
        ] + [
            f"The {w}" for w in ABSTRACT[:5]
        ],
        "test_pos_func": lambda p, off: 1 + off,
        "readout_name": "concrete - abstract verb",
    }

    # Feature 5: Sentiment
    features["sentiment"] = {
        "class_a_words": POSITIVE,
        "class_b_words": NEGATIVE,
        "prompt_a": "The {} brings",   # positive + positive verb
        "prompt_b": "The {} causes",   # negative + negative verb
        "pos_func": lambda p, off: 1 + off,
        "verb_ids_a": [safe_get_token_id(tokenizer, v) for v in POSITIVE_VERBS],
        "verb_ids_b": [safe_get_token_id(tokenizer, v) for v in NEGATIVE_VERBS],
        "test_prompts": [
            f"The {w}" for w in POSITIVE[:5]
        ] + [
            f"The {w}" for w in NEGATIVE[:5]
        ],
        "test_pos_func": lambda p, off: 1 + off,
        "readout_name": "positive - negative verb",
    }

    # Filter verb IDs to valid ones
    for feat_name, feat in features.items():
        feat["verb_ids_a"] = [v for v in feat["verb_ids_a"] if v is not None]
        feat["verb_ids_b"] = [v for v in feat["verb_ids_b"] if v is not None]
        if not feat["verb_ids_a"] or not feat["verb_ids_b"]:
            log_time(f"  WARNING: {feat_name} has no valid verb IDs, skipping")
            continue

    all_results = {}

    for feat_name, feat in features.items():
        if not feat["verb_ids_a"] or not feat["verb_ids_b"]:
            continue

        log_time(f"\n{'='*50}")
        log_time(f"Feature: {feat_name}")
        log_time(f"{'='*50}")

        # Step 1: Extract probe directions at test layers
        log_time(f"  Extracting {feat_name} directions...")

        feat_directions = {}
        feat_probe_accs = {}

        # Collect hidden states at each test layer
        for l in test_layers:
            states_a = {"A": [], "B": []}

            for a_word, b_word in zip(feat["class_a_words"][:25], feat["class_b_words"][:25]):
                for word, label in [(a_word, "A"), (b_word, "B")]:
                    prompt = feat["prompt_a" if label == "A" else "prompt_b"].format(word)
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                    input_ids = inputs["input_ids"].to(input_device)
                    attn_mask = inputs["attention_mask"].to(input_device)

                    offset = get_special_token_offset(tokenizer, prompt)
                    subj_pos = feat["pos_func"](prompt, offset)
                    # Ensure valid position
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

                    if out.hidden_states and l + 1 < len(out.hidden_states):
                        hs = out.hidden_states[l + 1][0, subj_pos, :].float().cpu().numpy()
                        states_a[label].append(hs)

                torch.cuda.empty_cache()

            acc, direction = train_probe(states_a)
            if direction is not None:
                feat_directions[l] = direction
                feat_probe_accs[l] = acc
                log_time(f"    L{l}: probe_acc={acc}")

        # Also get embedding direction
        embed_a = []
        embed_b = []
        for a_word, b_word in zip(feat["class_a_words"][:30], feat["class_b_words"][:30]):
            a_ids = tokenizer.encode(a_word, add_special_tokens=False)
            b_ids = tokenizer.encode(b_word, add_special_tokens=False)
            if len(a_ids) == 1 and len(b_ids) == 1:
                with torch.no_grad():
                    embed_a.append(embed_layer.weight[a_ids[0]].detach().float().cpu().numpy())
                    embed_b.append(embed_layer.weight[b_ids[0]].detach().float().cpu().numpy())

        embed_acc, embed_direction = train_probe({"A": embed_a, "B": embed_b})
        log_time(f"    Embedding: probe_acc={embed_acc}")

        # Step 2: Intervention at each layer
        log_time(f"  Scanning intervention trajectories for {feat_name}...")

        alpha_values = list(np.arange(-10, 10.5, 2.0))  # 11 points
        trajectory_results = {}

        injection_points = [("embed", -1)] + [(f"L{l}", l) for l in sorted(feat_directions.keys())]

        for point_name, layer_idx in injection_points:
            if layer_idx == -1:
                direction = embed_direction
                if direction is None:
                    continue
            else:
                direction = feat_directions.get(layer_idx)
                if direction is None:
                    continue

            mean_scores = []

            for alpha in alpha_values:
                score_changes = []

                for prompt in feat["test_prompts"]:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                    input_ids = inputs["input_ids"].to(input_device)
                    attn_mask = inputs["attention_mask"].to(input_device)

                    offset = get_special_token_offset(tokenizer, prompt)
                    subj_pos = feat["test_pos_func"](prompt, offset)
                    n_tokens = input_ids.shape[1]
                    if subj_pos < 0:
                        subj_pos = n_tokens + subj_pos
                    if subj_pos < 0 or subj_pos >= n_tokens:
                        continue

                    # Baseline score
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attn_mask)
                    baseline_logits = out.logits[0, -1, :].float().cpu().numpy()
                    bl_a = float(np.mean([baseline_logits[tid] for tid in feat["verb_ids_a"] if tid < len(baseline_logits)]))
                    bl_b = float(np.mean([baseline_logits[tid] for tid in feat["verb_ids_b"] if tid < len(baseline_logits)]))
                    bl_score = bl_a - bl_b

                    # Intervention
                    if layer_idx == -1:
                        with torch.no_grad():
                            base_embed = embed_layer(input_ids).detach().clone()
                        seq_len = input_ids.shape[1]
                        position_ids = torch.arange(seq_len, device=input_device).unsqueeze(0)
                        modified_embed = base_embed.clone()
                        delta_tensor = torch.tensor(alpha * direction, dtype=base_embed.dtype, device=input_device)
                        modified_embed[0, subj_pos, :] += delta_tensor

                        try:
                            with torch.no_grad():
                                out = model(inputs_embeds=modified_embed.to(model.dtype),
                                           attention_mask=attn_mask, position_ids=position_ids)
                        except Exception:
                            continue
                    else:
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
                    a_v = float(np.mean([logits[tid] for tid in feat["verb_ids_a"] if tid < len(logits)]))
                    b_v = float(np.mean([logits[tid] for tid in feat["verb_ids_b"] if tid < len(logits)]))
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

            pos_score = np.mean([s for a, s in zip(alpha_values, mean_scores) if a > 0]) if any(a > 0 for a in alpha_values) else 0
            neg_score = np.mean([s for a, s in zip(alpha_values, mean_scores) if a < 0]) if any(a < 0 for a in alpha_values) else 0
            bidirectional = (pos_score * neg_score < 0)

            interp = ("REAL_SEMANTIC_AXIS" if monotonicity > 0.85 and r_squared > 0.9
                      else "PARTIALLY_SEMANTIC" if monotonicity > 0.7 and r_squared > 0.7
                      else "LOCAL_STATISTICAL_BOUNDARY")

            trajectory_results[point_name] = {
                "alpha_values": [float(a) for a in alpha_values],
                "mean_scores": [round(s, 4) for s in mean_scores],
                "monotonicity": round(monotonicity, 4),
                "correlation": round(correlation, 4),
                "r_squared": round(r_squared, 4),
                "pos_alpha_mean": round(pos_score, 4),
                "neg_alpha_mean": round(neg_score, 4),
                "bidirectional": bidirectional,
                "interpretation": interp,
                "probe_acc": feat_probe_accs.get(layer_idx, embed_acc) if layer_idx >= 0 else embed_acc,
            }

            log_time(f"    {point_name}: mono={monotonicity:.3f}, R²={r_squared:.3f}, "
                     f"bidir={bidirectional}, class={interp}")

        all_results[feat_name] = trajectory_results

        # Feature summary
        embed_r2 = trajectory_results.get("embed", {}).get("r_squared", 0)
        best_mlp = max(
            [(k, v) for k, v in trajectory_results.items() if k != "embed"],
            key=lambda x: x[1]["r_squared"],
            default=(None, {"r_squared": 0})
        )

        log_time(f"\n  ★ {feat_name} Summary:")
        log_time(f"    Embed R²={embed_r2:.4f}, Best MLP ({best_mlp[0]}) R²={best_mlp[1]['r_squared']:.4f}")
        if best_mlp[1]['r_squared'] > 0.9 and best_mlp[1]['monotonicity'] > 0.85:
            log_time(f"    → DEEP SEMANTIC AXIS CONFIRMED for {feat_name}")
        elif best_mlp[1]['r_squared'] > 0.7:
            log_time(f"    → PARTIAL semantic axis for {feat_name}")
        else:
            log_time(f"    → NO deep semantic axis for {feat_name}")

    # Save all results
    out_path = RESULT_DIR / f"{model_name}_part1_multi_feature_intervention.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 1 saved to {out_path}")

    # Cross-feature comparison
    log_time(f"\n{'='*60}")
    log_time(f"CROSS-FEATURE COMPARISON for {model_name}")
    log_time(f"{'='*60}")
    log_time(f"{'Feature':<15} {'Embed R²':>10} {'Best MLP R²':>12} {'Best Layer':>12} {'Classification':>25}")
    log_time(f"{'-'*74}")

    for feat_name, traj in all_results.items():
        embed_r2 = traj.get("embed", {}).get("r_squared", 0)
        best_mlp = max(
            [(k, v) for k, v in traj.items() if k != "embed"],
            key=lambda x: x[1]["r_squared"],
            default=(None, {"r_squared": 0, "interpretation": "N/A"})
        )
        best_layer = best_mlp[0] if best_mlp[0] else "N/A"
        best_r2 = best_mlp[1]["r_squared"]
        interp = best_mlp[1].get("interpretation", "N/A")
        log_time(f"{feat_name:<15} {embed_r2:>10.4f} {best_r2:>12.4f} {best_layer:>12} {interp:>25}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return all_results


# ============================================================
# Part 2: Future Space Entropy Collapse
# ============================================================

def run_part2(model_name):
    """
    Measure how the token probability distribution narrows across layers.
    
    Key question from Analysis 1:
    - Does entropy drop dramatically in deep layers → constraint convergence?
    - Or does entropy stay constant → deep linearization is NOT from convergence?
    
    Method:
    For each layer l, extract the "logit lens" projection:
      logits_l = h_l @ W_U
    Then compute:
      entropy_l = -Σ p(token_i) * log(p(token_i))
      where p(token_i) = softmax(logits_l / temperature)
    
    Also measure:
    - top-1 probability: how much probability mass on the top token
    - effective support size: exp(entropy)
    - rank of the eventually-chosen token
    """
    import torch
    from model_utils import get_layers, get_W_U

    log_time(f"=== Part 2: Future Space Entropy Collapse for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim = load_model_bf16(model_name)
    input_device = get_input_device(model)

    n_layers = info.n_layers
    d_model = info.d_model

    # Get unembedding matrix
    W_U = get_W_U(model, model_name)
    log_time(f"  W_U shape: {W_U.shape}")

    # Test prompts (diverse set)
    test_prompts = [
        "The cat",
        "The dogs",
        "A beautiful",
        "The scientist",
        "Yesterday it",
        "She walked to the",
        "The old building",
        "Children love to",
        "The weather is",
        "In the morning,",
        "He decided to",
        "The river flows",
        "After the storm,",
        "They discovered a",
        "The city was",
    ]

    # Collect entropy at each layer
    all_entropies = {i: [] for i in range(n_layers + 1)}
    all_top1_probs = {i: [] for i in range(n_layers + 1)}
    all_effective_sizes = {i: [] for i in range(n_layers + 1)}
    all_final_token_ranks = {i: [] for i in range(n_layers + 1)}

    temperature = 1.0

    for pi, prompt in enumerate(test_prompts):
        log_time(f"  Processing prompt {pi+1}/{len(test_prompts)}: '{prompt}'")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=True)

        if not out.hidden_states:
            continue

        # Get final token prediction for rank tracking
        final_logits = out.logits[0, -1, :].float().cpu().numpy()
        final_top_id = int(np.argmax(final_logits))

        # For each layer, compute logit lens
        for l in range(len(out.hidden_states)):
            hs = out.hidden_states[l][0, -1, :].float().cpu().numpy()  # last position

            # Logit lens: project through unembedding
            logits = hs @ W_U.T

            # Compute softmax probabilities
            logits_shifted = logits - np.max(logits)  # numerical stability
            exp_logits = np.exp(logits_shifted / temperature)
            probs = exp_logits / np.sum(exp_logits)

            # Entropy
            log_probs = np.log(probs + 1e-30)
            entropy = -np.sum(probs * log_probs)

            # Top-1 probability
            top1_prob = float(np.max(probs))

            # Effective support size
            effective_size = float(np.exp(entropy))

            # Rank of final predicted token
            sorted_ids = np.argsort(logits)[::-1]
            rank = int(np.where(sorted_ids == final_top_id)[0][0]) + 1 if final_top_id in sorted_ids else len(logits)

            all_entropies[l].append(entropy)
            all_top1_probs[l].append(top1_prob)
            all_effective_sizes[l].append(effective_size)
            all_final_token_ranks[l].append(rank)

        torch.cuda.empty_cache()

    # Compute statistics
    layer_stats = {}
    for l in range(n_layers + 1):
        if all_entropies[l]:
            layer_stats[l] = {
                "mean_entropy": round(float(np.mean(all_entropies[l])), 4),
                "std_entropy": round(float(np.std(all_entropies[l])), 4),
                "mean_top1_prob": round(float(np.mean(all_top1_probs[l])), 4),
                "mean_effective_size": round(float(np.mean(all_effective_sizes[l])), 2),
                "mean_final_rank": round(float(np.mean(all_final_token_ranks[l])), 2),
                "n_prompts": len(all_entropies[l]),
            }

    # Print key results
    log_time(f"\n  Layer-wise entropy collapse:")
    log_time(f"  {'Layer':<8} {'Entropy':>10} {'Top1%':>10} {'Eff.Size':>12} {'FinalRank':>12}")
    log_time(f"  {'-'*52}")

    # Print sampled layers
    print_layers = [0] + list(range(5, n_layers, max(1, n_layers // 8))) + [n_layers]
    print_layers = sorted(set([l for l in print_layers if l in layer_stats]))

    for l in print_layers:
        s = layer_stats[l]
        log_time(f"  L{l:<7} {s['mean_entropy']:>10.4f} {s['mean_top1_prob']*100:>9.2f}% "
                 f"{s['mean_effective_size']:>12.2f} {s['mean_final_rank']:>12.2f}")

    # Key analysis: entropy drop
    first_layer_entropy = layer_stats.get(0, {}).get("mean_entropy", 0)
    last_layer_entropy = layer_stats.get(n_layers, {}).get("mean_entropy", 0)
    entropy_drop = first_layer_entropy - last_layer_entropy
    entropy_ratio = last_layer_entropy / (first_layer_entropy + 1e-10)

    # Find the layer of maximum entropy drop rate
    max_drop_rate = 0
    max_drop_layer = 0
    for l in range(1, n_layers + 1):
        if l in layer_stats and l-1 in layer_stats:
            drop_rate = layer_stats[l-1]["mean_entropy"] - layer_stats[l]["mean_entropy"]
            if drop_rate > max_drop_rate:
                max_drop_rate = drop_rate
                max_drop_layer = l

    log_time(f"\n  ★ Entropy Analysis:")
    log_time(f"    First layer entropy: {first_layer_entropy:.4f}")
    log_time(f"    Last layer entropy:  {last_layer_entropy:.4f}")
    log_time(f"    Total entropy drop:  {entropy_drop:.4f} ({entropy_ratio:.2%} of original)")
    log_time(f"    Max drop rate at layer {max_drop_layer}: {max_drop_rate:.4f}/layer")

    # Critical test: Is deep-layer linearization from constraint convergence?
    # If entropy drops dramatically → constraint convergence (Analysis 1 correct)
    # If entropy stays roughly constant → deep linearization is NOT from convergence

    deep_start = int(n_layers * 0.7)
    mid_entropy = layer_stats.get(n_layers // 2, {}).get("mean_entropy", 0)
    deep_entropy = layer_stats.get(deep_start, {}).get("mean_entropy", 0)
    last_entropy = layer_stats.get(n_layers, {}).get("mean_entropy", 0)

    deep_to_last_drop = deep_entropy - last_entropy
    first_to_mid_drop = first_layer_entropy - mid_entropy

    log_time(f"\n  ★ Constraint Convergence Test:")
    log_time(f"    First→Mid drop: {first_to_mid_drop:.4f}")
    log_time(f"    Deep→Last drop: {deep_to_last_drop:.4f}")

    if deep_to_last_drop > first_to_mid_drop * 0.5:
        log_time(f"    → Entropy continues to drop significantly in deep layers")
        log_time(f"    → Constraint convergence IS happening in deep layers")
        log_time(f"    → Analysis 1's 'constraint convergence' theory SUPPORTED")
    else:
        log_time(f"    → Entropy mostly stabilizes before deep layers")
        log_time(f"    → Deep linearization is NOT from constraint convergence")
        log_time(f"    → Analysis 1's 'constraint convergence' theory NOT supported")
        log_time(f"    → Deep semantic axes emerge WITHOUT entropy collapse")

    results = {
        "model": model_name,
        "n_layers": n_layers,
        "n_prompts": len(test_prompts),
        "temperature": temperature,
        "layer_stats": {str(k): v for k, v in layer_stats.items()},
        "entropy_analysis": {
            "first_layer_entropy": round(first_layer_entropy, 4),
            "last_layer_entropy": round(last_layer_entropy, 4),
            "total_drop": round(entropy_drop, 4),
            "ratio": round(entropy_ratio, 4),
            "max_drop_layer": max_drop_layer,
            "max_drop_rate": round(max_drop_rate, 4),
            "deep_start_layer": deep_start,
            "first_to_mid_drop": round(first_to_mid_drop, 4),
            "deep_to_last_drop": round(deep_to_last_drop, 4),
        },
    }

    out_path = RESULT_DIR / f"{model_name}_part2_entropy_collapse.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 2 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Part 3: Constraint Conflict Experiment
# ============================================================

def run_part3(model_name):
    """
    Observe how the model processes grammatically inconsistent inputs.
    
    Analysis 1's core claim: language is a constraint propagation system.
    If true, grammatically inconsistent inputs should show:
    1. A specific layer where the conflict is detected
    2. Subsequent layers trying to "fix" the conflict
    3. A "repair" pattern in the hidden state trajectory
    
    Method:
    - Compare hidden state trajectories for:
      * Grammatical: "The cat sits" (singular subject + singular verb)
      * Ungrammatical: "The cat sit" (singular subject + plural verb)
      * Conflict: "The rocks eats" (plural subject + singular verb)
    - At each layer, measure the "grammar consistency signal":
      How much does the model favor grammatical continuation?
    - Track which layer first detects the conflict
    """
    import torch
    from model_utils import get_layers, get_W_U

    log_time(f"=== Part 3: Constraint Conflict Experiment for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim = load_model_bf16(model_name)
    input_device = get_input_device(model)

    n_layers = info.n_layers
    d_model = info.d_model

    # Get unembedding matrix for logit lens
    W_U = get_W_U(model, model_name)

    # Get verb token IDs
    sing_verb_ids = [safe_get_token_id(tokenizer, v) for v in SING_VERBS if safe_get_token_id(tokenizer, v) is not None]
    plur_verb_ids = [safe_get_token_id(tokenizer, v) for v in PLUR_VERBS if safe_get_token_id(tokenizer, v) is not None]

    # For each conflict pair, compare grammatical vs ungrammatical
    conflict_results = []

    for pair in CONFLICT_PAIRS:
        gram_sing, gram_plur, ungram_sing, ungram_plur = pair

        log_time(f"\n  Testing: {gram_sing} vs {ungram_sing}")

        # Process all 4 variants
        variants = {
            "gram_sing": gram_sing,
            "gram_plur": gram_plur,
            "ungram_sing": ungram_sing,
            "ungram_plur": ungram_plur,
        }

        variant_layer_scores = {}

        for var_name, prompt in variants.items():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)

            if not out.hidden_states:
                continue

            # For each layer, compute grammar consistency score via logit lens
            layer_scores = {}
            for l in range(len(out.hidden_states)):
                hs = out.hidden_states[l][0, -1, :].float().cpu().numpy()
                logits = hs @ W_U.T

                # Grammar consistency: sing_verbs - plur_verbs
                sing_score = float(np.mean([logits[tid] for tid in sing_verb_ids if tid < len(logits)]))
                plur_score = float(np.mean([logits[tid] for tid in plur_verb_ids if tid < len(logits)]))
                grammar_signal = sing_score - plur_score

                layer_scores[l] = grammar_signal

            variant_layer_scores[var_name] = layer_scores
            torch.cuda.empty_cache()

        # Analyze: when does the model detect the conflict?
        # For singular subject: grammatical has high sing>plur, ungrammatical should show conflict
        gram_sing_scores = variant_layer_scores.get("gram_sing", {})
        ungram_sing_scores = variant_layer_scores.get("ungram_sing", {})

        # The conflict signal: difference between grammatical and ungrammatical
        conflict_signal = {}
        for l in range(n_layers + 1):
            if l in gram_sing_scores and l in ungram_sing_scores:
                # For "The cat sits" (grammatical): should show high sing>plur
                # For "The cat sit" (ungrammatical): should show low sing>plur
                # Conflict = grammatical - ungrammatical
                conflict_signal[l] = gram_sing_scores[l] - ungram_sing_scores[l]

        # Find the layer where conflict signal first becomes significant
        first_conflict_layer = None
        conflict_threshold = 0.5  # Minimum difference to count as conflict detection
        for l in range(n_layers + 1):
            if l in conflict_signal and abs(conflict_signal[l]) > conflict_threshold:
                first_conflict_layer = l
                break

        # Track the trajectory of grammar signal
        # For grammatical: should stay high
        # For ungrammatical: should show a dip or reversal
        result = {
            "pair": list(pair),
            "grammar_signal_trajectory": {},
            "conflict_signal": {str(k): round(v, 4) for k, v in conflict_signal.items()},
            "first_conflict_layer": first_conflict_layer,
        }

        # Sample every few layers for readability
        sample_rate = max(1, n_layers // 10)
        for var_name, scores in variant_layer_scores.items():
            result["grammar_signal_trajectory"][var_name] = {
                str(l): round(scores[l], 4)
                for l in range(0, n_layers + 1, sample_rate)
                if l in scores
            }

        conflict_results.append(result)

        if first_conflict_layer is not None:
            log_time(f"    First conflict detected at L{first_conflict_layer}")
        else:
            log_time(f"    No significant conflict detection found")

        # Log sample of grammar signal trajectory
        for var_name in ["gram_sing", "ungram_sing"]:
            scores = variant_layer_scores.get(var_name, {})
            sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]
            sample_vals = [round(scores.get(l, 0), 3) for l in sample_layers]
            log_time(f"    {var_name}: L0={sample_vals[0]}, L{n_layers//4}={sample_vals[1]}, "
                     f"L{n_layers//2}={sample_vals[2]}, L{3*n_layers//4}={sample_vals[3]}, "
                     f"L{n_layers}={sample_vals[4]}")

    # Also test animacy conflicts
    animacy_results = []

    for conflict_prompt, control_prompt in ANIMACY_CONFLICT:
        log_time(f"\n  Animacy conflict: '{conflict_prompt}' vs '{control_prompt}'")

        pair_scores = {}
        for var_name, prompt in [("conflict", conflict_prompt), ("control", control_prompt)]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)

            if not out.hidden_states:
                continue

            layer_scores = {}
            for l in range(len(out.hidden_states)):
                hs = out.hidden_states[l][0, -1, :].float().cpu().numpy()
                logits = hs @ W_U.T

                anim_score = float(np.mean([logits[tid] for tid in [safe_get_token_id(tokenizer, v) for v in ANIMATE_VERBS if safe_get_token_id(tokenizer, v)] if tid < len(logits)]))
                inanim_score = float(np.mean([logits[tid] for tid in [safe_get_token_id(tokenizer, v) for v in INANIMATE_VERBS if safe_get_token_id(tokenizer, v)] if tid < len(logits)]))

                layer_scores[l] = {
                    "animate_signal": anim_score - inanim_score,
                }

            pair_scores[var_name] = layer_scores
            torch.cuda.empty_cache()

        animacy_results.append({
            "conflict_prompt": conflict_prompt,
            "control_prompt": control_prompt,
            "sample_trajectory": {
                var: {
                    str(l): {k: round(v, 4) for k, v in scores[l].items()}
                    for l in range(0, n_layers + 1, max(1, n_layers // 8))
                    if l in scores
                }
                for var, scores in pair_scores.items()
            },
        })

    # Summary
    log_time(f"\n{'='*60}")
    log_time(f"CONSTRAINT CONFLICT SUMMARY for {model_name}")
    log_time(f"{'='*60}")

    first_conflict_layers = [r["first_conflict_layer"] for r in conflict_results if r["first_conflict_layer"] is not None]
    if first_conflict_layers:
        log_time(f"  Number conflict detected at layers: {first_conflict_layers}")
        log_time(f"  Mean first detection layer: {np.mean(first_conflict_layers):.1f}")
        log_time(f"  Relative position: {np.mean(first_conflict_layers)/n_layers:.1%}")
    else:
        log_time(f"  No significant conflict detection found")

    results = {
        "model": model_name,
        "n_layers": n_layers,
        "number_conflicts": conflict_results,
        "animacy_conflicts": animacy_results,
        "summary": {
            "first_conflict_layers": first_conflict_layers,
            "mean_first_detection_layer": round(float(np.mean(first_conflict_layers)), 1) if first_conflict_layers else None,
            "relative_position": round(float(np.mean(first_conflict_layers) / n_layers), 4) if first_conflict_layers else None,
        },
    }

    out_path = RESULT_DIR / f"{model_name}_part3_constraint_conflict.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 3 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    parser.add_argument("--part", choices=["1", "2", "3"], required=True)
    args = parser.parse_args()

    if args.part == "1":
        run_part1(args.model)
    elif args.part == "2":
        run_part2(args.model)
    elif args.part == "3":
        run_part3(args.model)


if __name__ == "__main__":
    main()
