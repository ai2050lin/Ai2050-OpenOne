"""
Phase 263: Trajectory Continuity, Bootstrap Stability & Multi-Feature Orthogonality
====================================================================================

Addressing the DEEPEST remaining questions from Phase 262:

  Part 1: Continuous Trajectory Experiment (CRITICAL)
    - Scan alpha in [-20, 20] along Δ_number direction
    - Measure logit, entropy, and verb prediction at each alpha
    - Test: is the response monotonic and continuous?
    → If yes: direction is a real semantic axis
    → If discontinuous: direction is just a local statistical boundary

  Part 2: Bootstrap Direction Stability (CRITICAL)
    - Extract number direction from 10 different random subsets of training words
    - Compute pairwise cosine between all 10 directions
    - Test: is the direction stable across different word sets?
    → If stable (cos > 0.8): direction is robust, not dataset accident
    → If unstable (cos < 0.3): direction is dataset-specific, unreliable

  Part 3: Multi-Feature Orthogonality Map (HIGH PRIORITY)
    - Test 5+ features: number, animacy, tense, concreteness, frequency
    - Compute pairwise direction cosine in MLP space
    - Build complete "feature subspace map"
    → Tests if orthogonal coding is universal across feature types

  Part 4: Bidirectional Causal Transport + Random Direction Deep Dive
    - Test ADDING and SUBTRACTING Δ_number
    - Test Δ_animacy as additional control
    - Test multiple random directions for statistical significance
    → Resolves direction reversal and random direction puzzles

Usage:
  python tests/glm5/phase263_trajectory_stability.py --model qwen3 --part 1
  python tests/glm5/phase263_trajectory_stability.py --model glm4 --part 2
  python tests/glm5/phase263_trajectory_stability.py --model deepseek7b --part 3
  python tests/glm5/phase263_trajectory_stability.py --model qwen3 --part 4
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase263_trajectory_stability")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Verb tokens =====
SING_VERBS = ["runs", "walks", "sits", "is", "has", "does", "goes", "was", "eats", "makes"]
PLUR_VERBS = ["run", "walk", "sit", "are", "have", "do", "go", "were", "eat", "make"]

# ===== Training subjects =====
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

# ===== Test subjects =====
TEST_SING = [
    "bear", "eagle", "rabbit", "tiger", "whale", "fox", "deer", "wolf",
    "snake", "crow", "ant", "owl", "penguin", "dolphin", "spider",
    "lamp", "clock", "plate", "cup", "glass", "pillow", "blanket",
    "hammer", "rope", "ring", "coin", "letter", "map", "photo", "key"
]

# ===== Animate/Inanimate =====
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

# ===== Tense words (present vs past) =====
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

# ===== Concrete vs Abstract =====
CONCRETE_WORDS = [
    "table", "chair", "house", "car", "tree", "dog", "book", "phone",
    "water", "food", "door", "wall", "road", "river", "mountain",
    "cloud", "fire", "stone", "glass", "metal", "paper", "wood",
    "knife", "cup", "plate", "lamp", "clock", "pillow", "blanket", "hammer"
]
ABSTRACT_WORDS = [
    "freedom", "justice", "truth", "beauty", "love", "time", "mind",
    "knowledge", "power", "idea", "reason", "hope", "fear", "anger",
    "peace", "war", "life", "death", "faith", "doubt", "honor",
    "duty", "right", "wrong", "good", "evil", "virtue", "wisdom", "courage", "joy"
]

# ===== High vs Low frequency (approximate by word length as proxy) =====
HIGH_FREQ_WORDS = [
    "the", "a", "is", "was", "he", "she", "it", "they", "we", "you",
    "do", "did", "can", "will", "would", "could", "should", "may",
    "not", "no", "all", "some", "any", "each", "every", "one", "two",
    "first", "new", "old", "big", "small", "good", "bad", "long", "short"
]
LOW_FREQ_WORDS = [
    "lexicon", "paradigm", "synthesis", "epistemology", "hermeneutics",
    "metamorphosis", "juxtaposition", "quintessential", "surreptitious",
    "perspicacious", "magnanimous", "obfuscate", "recalcitrant", "sycophant",
    "pusillanimous", "sesquipedalian", "verisimilitude", "persiflage",
    "tergiversation", "pulchritude", "defenestration", "logorrhea",
    "borborygmus", "floccinaucinihilipilification", "antidisestablishmentarianism"
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


def load_model_for_phase263(model_name, need_attn_weights=False):
    """Load model with BF16 + device_map='auto' + flash attention when possible"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS, get_model_info, get_layers

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (need_attn_weights={need_attn_weights})...")

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
    kv_group_size = n_heads // n_kv_heads

    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  class={info.model_class}, layers={info.n_layers}, d_model={info.d_model}, "
             f"n_heads={n_heads}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size


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


# ============================================================
# Part 1: Continuous Trajectory Experiment
# ============================================================

def run_part1(model_name):
    """
    CRITICAL EXPERIMENT: Is the Δ_number direction a real semantic axis
    or just a local statistical boundary?

    Method:
    1. Extract Δ_number from training embeddings
    2. For 10 test words, scan alpha in [-20, 20] with fine resolution
    3. At each alpha, measure:
       - sing_verb_logit - plur_verb_logit (grammar score)
       - entropy of next-token distribution
       - top-1 predicted token
    4. Test monotonicity and continuity

    If direction is a real semantic axis:
      → score should change monotonically with alpha
      → entropy should be minimal at extremes (confident prediction)
      → smooth transition between sing/plur verb dominance

    If direction is a local statistical boundary:
      → score changes erratically
      → high entropy throughout
      → no smooth transition
    """
    import torch

    log_time(f"=== Part 1: Continuous Trajectory for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_for_phase263(model_name)
    embed_layer = model.get_input_embeddings()
    embed_device = embed_layer.weight.device

    # Step 1: Extract Δ_number
    log_time("Step 1: Extracting Δ_number from training embeddings...")
    sing_embeds = []
    plur_embeds = []
    for s, p in zip(TRAIN_SING, TRAIN_PLUR):
        s_ids = tokenizer.encode(s, add_special_tokens=False)
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        if len(s_ids) == 1 and len(p_ids) == 1:
            with torch.no_grad():
                s_emb = embed_layer.weight[s_ids[0]].detach().float().cpu().numpy()
                p_emb = embed_layer.weight[p_ids[0]].detach().float().cpu().numpy()
            sing_embeds.append(s_emb)
            plur_embeds.append(p_emb)

    sing_embeds = np.array(sing_embeds)
    plur_embeds = np.array(plur_embeds)

    # Probe direction
    embed_data = {"sing": sing_embeds.tolist(), "plur": plur_embeds.tolist()}
    probe_acc, delta_number = train_probe(embed_data)
    log_time(f"  Embedding probe accuracy: {probe_acc}")

    # Random direction control
    rng = np.random.RandomState(42)
    delta_random = rng.randn(info.d_model).astype(np.float32)
    delta_random = delta_random / (np.linalg.norm(delta_random) + 1e-10)

    # Get verb token IDs
    sing_verb_ids = [safe_get_token_id(tokenizer, v) for v in SING_VERBS if safe_get_token_id(tokenizer, v) is not None]
    plur_verb_ids = [safe_get_token_id(tokenizer, v) for v in PLUR_VERBS if safe_get_token_id(tokenizer, v) is not None]

    # Step 2: Fine-grained alpha scan
    alpha_values = list(np.arange(-20, 20.5, 1.0))  # 41 points from -20 to 20
    test_words = TEST_SING[:15]  # Use 15 test words for manageable compute

    log_time(f"Step 2: Scanning {len(alpha_values)} alpha values for {len(test_words)} test words")

    results = {
        "model": model_name,
        "alpha_values": alpha_values,
        "test_words": test_words,
        "probe_acc": probe_acc,
        "trajectory_data": {},
    }

    for direction_name, direction_vec in [("delta_number", delta_number), ("random", delta_random)]:
        trajectory = []

        for wi, word in enumerate(test_words):
            prompt = f"The {word}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(embed_device)
            attn_mask = inputs["attention_mask"].to(embed_device)
            seq_len = input_ids.shape[1]
            offset = get_special_token_offset(tokenizer, prompt)
            subj_pos = 1 + offset

            with torch.no_grad():
                base_embed = embed_layer(input_ids).detach().clone()

            position_ids = torch.arange(seq_len, device=embed_device).unsqueeze(0)

            # Baseline
            with torch.no_grad():
                out = model(inputs_embeds=base_embed.to(model.dtype),
                           attention_mask=attn_mask, position_ids=position_ids)
            baseline_logits = out.logits[0, -1, :].float().cpu().numpy()
            bl_sing = float(np.mean([baseline_logits[tid] for tid in sing_verb_ids]))
            bl_plur = float(np.mean([baseline_logits[tid] for tid in plur_verb_ids]))
            bl_score = bl_sing - bl_plur

            word_trajectory = []
            for alpha in alpha_values:
                modified_embed = base_embed.clone()
                delta_tensor = torch.tensor(alpha * direction_vec, dtype=base_embed.dtype, device=embed_device)
                modified_embed[0, subj_pos, :] += delta_tensor

                with torch.no_grad():
                    out = model(inputs_embeds=modified_embed.to(model.dtype),
                               attention_mask=attn_mask, position_ids=position_ids)
                logits = out.logits[0, -1, :].float().cpu().numpy()

                sing_v = float(np.mean([logits[tid] for tid in sing_verb_ids]))
                plur_v = float(np.mean([logits[tid] for tid in plur_verb_ids]))
                score = sing_v - plur_v

                # Entropy
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)
                entropy = -np.sum(probs * np.log(probs + 1e-10))

                # Top-1 prediction
                top1_id = int(np.argmax(logits))
                top1_token = tokenizer.decode([top1_id]).strip()

                word_trajectory.append({
                    "alpha": float(alpha),
                    "score": round(score, 4),
                    "score_change": round(score - bl_score, 4),
                    "sing_verb_logit": round(sing_v, 4),
                    "plur_verb_logit": round(plur_v, 4),
                    "entropy": round(float(entropy), 4),
                    "top1_token": top1_token,
                })

            trajectory.append({"word": word, "baseline_score": round(bl_score, 4), "data": word_trajectory})

            if wi % 5 == 0:
                # Quick monotonicity check for this word
                scores = [d["score"] for d in word_trajectory]
                # Check if scores are roughly monotonic
                diffs = np.diff(scores)
                n_positive = sum(1 for d in diffs if d > 0)
                n_negative = sum(1 for d in diffs if d < 0)
                direction_label = "MONOTONIC↑" if n_positive > 0.8 * len(diffs) else \
                                  "MONOTONIC↓" if n_negative > 0.8 * len(diffs) else "NON-MONOTONIC"
                log_time(f"  Word {wi+1}/{len(test_words)}: '{word}' | "
                         f"baseline={bl_score:.2f}, trend={direction_label}, "
                         f"score_range=[{min(scores):.2f}, {max(scores):.2f}]")

        results["trajectory_data"][direction_name] = trajectory

    # Step 3: Monotonicity and continuity analysis
    log_time("\nStep 3: Analyzing trajectory properties...")

    for direction_name in ["delta_number", "random"]:
        trajectory = results["trajectory_data"][direction_name]

        # Aggregate: for each alpha, compute mean score across words
        mean_scores = []
        for ai, alpha in enumerate(alpha_values):
            scores_at_alpha = [t["data"][ai]["score"] for t in trajectory]
            mean_scores.append(float(np.mean(scores_at_alpha)))

        # Monotonicity: fraction of consecutive pairs where score increases with alpha
        diffs = np.diff(mean_scores)
        n_increasing = sum(1 for d in diffs if d > 0)
        monotonicity = n_increasing / len(diffs) if len(diffs) > 0 else 0

        # Check if monotonic in the expected direction:
        # Higher alpha = more "plural-like" direction in probe → should decrease score
        # (because probe points toward "sing" class)
        # But Phase 262 showed REVERSAL, so we test if it's consistently monotonic either way

        # Continuity: max absolute jump between consecutive alphas
        max_jump = float(np.max(np.abs(diffs)))

        # Linearity: R² of score vs alpha
        alpha_arr = np.array(alpha_values, dtype=float)
        score_arr = np.array(mean_scores)
        correlation = np.corrcoef(alpha_arr, score_arr)[0, 1]
        r_squared = correlation ** 2

        # Entropy analysis: mean entropy at each alpha
        mean_entropies = []
        for ai, alpha in enumerate(alpha_values):
            entropies_at_alpha = [t["data"][ai]["entropy"] for t in trajectory]
            mean_entropies.append(float(np.mean(entropies_at_alpha)))

        # Entropy should be minimal at extremes (confident) and maximal near crossover
        min_entropy_alpha = alpha_values[np.argmin(mean_entropies)]

        results[f"analysis_{direction_name}"] = {
            "monotonicity": round(monotonicity, 4),
            "correlation": round(float(correlation), 4),
            "r_squared": round(float(r_squared), 4),
            "max_jump": round(max_jump, 4),
            "mean_entropy_range": [round(min(mean_entropies), 4), round(max(mean_entropies), 4)],
            "min_entropy_at_alpha": float(min_entropy_alpha),
            "score_range": [round(min(mean_scores), 4), round(max(mean_scores), 4)],
            "interpretation": (
                "REAL_SEMANTIC_AXIS" if monotonicity > 0.85 and r_squared > 0.9
                else "PARTIALLY_SEMANTIC" if monotonicity > 0.7 and r_squared > 0.7
                else "LOCAL_STATISTICAL_BOUNDARY"
            ),
        }

        log_time(f"  {direction_name}: monotonicity={monotonicity:.3f}, R²={r_squared:.3f}, "
                 f"max_jump={max_jump:.3f}, corr={correlation:.3f}")
        log_time(f"  Interpretation: {results[f'analysis_{direction_name}']['interpretation']}")

    # Save
    out_path = RESULT_DIR / f"{model_name}_part1_trajectory.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 1 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Part 2: Bootstrap Direction Stability
# ============================================================

def run_part2(model_name):
    """
    CRITICAL EXPERIMENT: Is the number direction robust across different word sets?

    Method:
    1. Create 10 different random subsets (30 out of 50 word pairs each)
    2. Extract probe direction from each subset at embedding and MLP layers
    3. Compute pairwise cosine between all 10 directions
    4. Test stability

    If stable (mean pairwise cos > 0.8): direction is robust
    If unstable (mean pairwise cos < 0.3): direction is dataset accident
    """
    import torch
    from model_utils import get_layers

    log_time(f"=== Part 2: Bootstrap Direction Stability for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_for_phase263(model_name)
    input_device = get_input_device(model)
    layers = get_layers(model)
    embed_layer = model.get_input_embeddings()

    n_subsets = 10
    subset_size = 30  # Use 30 out of 50 pairs
    rng = np.random.RandomState(2026)

    # Generate subsets
    all_indices = list(range(len(TRAIN_SING)))
    subsets = []
    for _ in range(n_subsets):
        idx = rng.choice(all_indices, size=subset_size, replace=False)
        subsets.append(idx)

    log_time(f"Created {n_subsets} subsets of {subset_size} word pairs each")

    # Extract directions at multiple layers
    state_names = ["embed", "L0_attn_out", "L0_mlp_out", "L1_mlp_out"]
    all_directions = {name: [] for name in state_names}

    for si, subset_idx in enumerate(subsets):
        log_time(f"  Subset {si+1}/{n_subsets}...")

        sing_subset = [TRAIN_SING[i] for i in subset_idx]
        plur_subset = [TRAIN_PLUR[i] for i in subset_idx]

        # Collect hidden states for this subset
        states = {name: {"sing": [], "plur": []} for name in state_names}

        for s_word, p_word in zip(sing_subset, plur_subset):
            for word, label in [(s_word, "sing"), (p_word, "plur")]:
                prompt = f"The {word} sits" if label == "sing" else f"The {word} sit"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)

                offset = get_special_token_offset(tokenizer, prompt)
                subj_pos = 1 + offset

                captured = {}
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key + "_out"] = output[0].detach().float().cpu()
                        else:
                            captured[key + "_out"] = output.detach().float().cpu()
                    return hook

                hooks = []
                hooks.append(layers[0].self_attn.register_forward_hook(make_hook("L0_attn")))
                hooks.append(layers[0].register_forward_hook(make_hook("L0_full")))
                hooks.append(layers[1].self_attn.register_forward_hook(make_hook("L1_attn")))
                hooks.append(layers[1].register_forward_hook(make_hook("L1_full")))

                try:
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                except Exception:
                    for h in hooks: h.remove()
                    continue
                for h in hooks: h.remove()

                if out.hidden_states and len(out.hidden_states) > 0:
                    states["embed"][label].append(out.hidden_states[0][0, subj_pos, :].float().cpu().numpy())

                for cap_name, state_name in [("L0_attn_out", "L0_attn_out"), ("L0_full_out", "L0_mlp_out"),
                                              ("L1_full_out", "L1_mlp_out")]:
                    if cap_name in captured:
                        states[state_name][label].append(captured[cap_name][0, subj_pos, :].float().cpu().numpy())

        # Train probes for this subset
        for name in state_names:
            acc, direction = train_probe(states[name])
            if direction is not None:
                all_directions[name].append(direction)
            else:
                all_directions[name].append(None)

    # Compute pairwise cosine for each layer
    stability_results = {}
    for name in state_names:
        directions = [d for d in all_directions[name] if d is not None]
        n_valid = len(directions)

        if n_valid < 2:
            stability_results[name] = {"error": "Not enough valid directions"}
            continue

        pairwise_cos = []
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                cos = float(np.dot(directions[i], directions[j]))
                pairwise_cos.append(cos)

        mean_cos = float(np.mean(pairwise_cos))
        std_cos = float(np.std(pairwise_cos))
        min_cos = float(np.min(pairwise_cos))
        max_cos = float(np.max(pairwise_cos))

        # Also compute average direction (consensus direction)
        avg_direction = np.mean(directions, axis=0)
        avg_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-10)

        # Alignment of each subset direction with average
        alignment_with_avg = [float(np.dot(d, avg_direction)) for d in directions]

        stability_results[name] = {
            "n_valid_subsets": n_valid,
            "pairwise_cosine": {
                "mean": round(mean_cos, 4),
                "std": round(std_cos, 4),
                "min": round(min_cos, 4),
                "max": round(max_cos, 4),
            },
            "alignment_with_consensus": {
                "mean": round(float(np.mean(alignment_with_avg)), 4),
                "min": round(float(np.min(alignment_with_avg)), 4),
            },
            "interpretation": (
                "ROBUST_DIRECTION" if mean_cos > 0.8
                else "MODERATELY_STABLE" if mean_cos > 0.5
                else "UNSTABLE_DATASET_DEPENDENT"
            ),
        }

        log_time(f"  {name}: pairwise_cos mean={mean_cos:.4f}±{std_cos:.4f}, "
                 f"range=[{min_cos:.4f}, {max_cos:.4f}], "
                 f"consensus_align={np.mean(alignment_with_avg):.4f}")
        log_time(f"    Interpretation: {stability_results[name]['interpretation']}")

    # Also test ANIMACY direction stability for comparison
    log_time("\nTesting animacy direction stability...")
    anim_state_names = ["embed", "L0_mlp_out", "L1_mlp_out"]
    anim_all_directions = {name: [] for name in anim_state_names}

    n_anim = min(len(ANIMATE), len(INANIMATE))
    anim_indices = list(range(n_anim))

    for si in range(n_subsets):
        idx = rng.choice(anim_indices, size=min(25, n_anim), replace=False)
        anim_subset = [ANIMATE[i] for i in idx]
        inanim_subset = [INANIMATE[i] for i in idx]

        states = {name: {"animate": [], "inanimate": []} for name in anim_state_names}

        for a_word, ia_word in zip(anim_subset, inanim_subset):
            for word, label in [(a_word, "animate"), (ia_word, "inanimate")]:
                prompt = f"The {word}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)

                offset = get_special_token_offset(tokenizer, prompt)
                subj_pos = 1 + offset

                captured = {}
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key + "_out"] = output[0].detach().float().cpu()
                        else:
                            captured[key + "_out"] = output.detach().float().cpu()
                    return hook

                hooks = []
                hooks.append(layers[0].self_attn.register_forward_hook(make_hook("L0_attn")))
                hooks.append(layers[0].register_forward_hook(make_hook("L0_full")))
                hooks.append(layers[1].register_forward_hook(make_hook("L1_full")))

                try:
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                except Exception:
                    for h in hooks: h.remove()
                    continue
                for h in hooks: h.remove()

                if out.hidden_states and len(out.hidden_states) > 0:
                    states["embed"][label].append(out.hidden_states[0][0, subj_pos, :].float().cpu().numpy())

                for cap_name, state_name in [("L0_full_out", "L0_mlp_out"), ("L1_full_out", "L1_mlp_out")]:
                    if cap_name in captured:
                        states[state_name][label].append(captured[cap_name][0, subj_pos, :].float().cpu().numpy())

        for name in anim_state_names:
            acc, direction = train_probe(states[name])
            if direction is not None:
                anim_all_directions[name].append(direction)
            else:
                anim_all_directions[name].append(None)

    for name in anim_state_names:
        directions = [d for d in anim_all_directions[name] if d is not None]
        n_valid = len(directions)
        if n_valid < 2:
            continue

        pairwise_cos = []
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                cos = float(np.dot(directions[i], directions[j]))
                pairwise_cos.append(cos)

        mean_cos = float(np.mean(pairwise_cos))
        log_time(f"  Animacy {name}: pairwise_cos mean={mean_cos:.4f}")

        stability_results[f"animacy_{name}"] = {
            "n_valid_subsets": n_valid,
            "pairwise_cosine_mean": round(mean_cos, 4),
            "interpretation": (
                "ROBUST_DIRECTION" if mean_cos > 0.8
                else "MODERATELY_STABLE" if mean_cos > 0.5
                else "UNSTABLE_DATASET_DEPENDENT"
            ),
        }

    results = {
        "model": model_name,
        "n_subsets": n_subsets,
        "subset_size": subset_size,
        "stability": stability_results,
    }

    out_path = RESULT_DIR / f"{model_name}_part2_bootstrap_stability.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 2 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Part 3: Multi-Feature Orthogonality Map
# ============================================================

def run_part3(model_name):
    """
    Build a complete "feature subspace map" in MLP space.

    Test 5 features:
    1. Number (singular vs plural) — grammar
    2. Animacy (animate vs inanimate) — semantics
    3. Tense (present vs past) — grammar
    4. Concreteness (concrete vs abstract) — semantics
    5. Frequency (high vs low frequency) — functional

    For each pair of features, compute direction cosine in MLP space.
    → If all pairs are near-orthogonal: orthogonal coding is universal
    → If some pairs are correlated: features have dependencies
    """
    import torch
    from model_utils import get_layers

    log_time(f"=== Part 3: Multi-Feature Orthogonality Map for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_for_phase263(model_name)
    input_device = get_input_device(model)
    layers = get_layers(model)

    # Define features
    features = {
        "number": (TRAIN_SING[:40], TRAIN_PLUR[:40], "grammar"),
        "animacy": (ANIMATE[:40], INANIMATE[:40], "semantics"),
        "tense": (PRESENT_WORDS, PAST_WORDS, "grammar"),
        "concreteness": (CONCRETE_WORDS, ABSTRACT_WORDS, "semantics"),
        "frequency": (HIGH_FREQ_WORDS[:25], LOW_FREQ_WORDS[:25], "functional"),
    }

    # Collect hidden states for each feature at key layers
    target_layers = ["embed", "L0_attn_out", "L0_mlp_out", "L1_mlp_out"]

    feature_directions = {layer: {} for layer in target_layers}
    feature_probe_accs = {layer: {} for layer in target_layers}

    for feat_name, (class_A, class_B, feat_type) in features.items():
        log_time(f"  Processing feature: {feat_name} ({feat_type}), {len(class_A)}+{len(class_B)} words")

        states = {name: {"A": [], "B": []} for name in target_layers}

        for word_A, word_B in zip(class_A, class_B):
            for word, label in [(word_A, "A"), (word_B, "B")]:
                # For words that might be multi-token, use simple context
                prompt = f"The {word}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)

                offset = get_special_token_offset(tokenizer, prompt)
                subj_pos = 1 + offset

                captured = {}
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key + "_out"] = output[0].detach().float().cpu()
                        else:
                            captured[key + "_out"] = output.detach().float().cpu()
                    return hook

                hooks = []
                hooks.append(layers[0].self_attn.register_forward_hook(make_hook("L0_attn")))
                hooks.append(layers[0].register_forward_hook(make_hook("L0_full")))
                hooks.append(layers[1].register_forward_hook(make_hook("L1_full")))

                try:
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                except Exception:
                    for h in hooks: h.remove()
                    continue
                for h in hooks: h.remove()

                if out.hidden_states and len(out.hidden_states) > 0:
                    states["embed"][label].append(out.hidden_states[0][0, subj_pos, :].float().cpu().numpy())

                for cap_name, state_name in [("L0_attn_out", "L0_attn_out"), ("L0_full_out", "L0_mlp_out"),
                                              ("L1_full_out", "L1_mlp_out")]:
                    if cap_name in captured:
                        states[state_name][label].append(captured[cap_name][0, subj_pos, :].float().cpu().numpy())

        # Train probes
        for name in target_layers:
            acc, direction = train_probe(states[name])
            if direction is not None:
                feature_directions[name][feat_name] = direction
                feature_probe_accs[name][feat_name] = acc
                log_time(f"    {name} probe: {acc}")
            else:
                log_time(f"    {name} probe: FAILED (not enough data)")

    # Compute pairwise feature direction cosine at each layer
    orthogonality_map = {}
    for layer_name in target_layers:
        feat_names = list(feature_directions[layer_name].keys())
        n_feats = len(feat_names)

        pairwise = {}
        for i in range(n_feats):
            for j in range(i + 1, n_feats):
                f1, f2 = feat_names[i], feat_names[j]
                cos = float(np.dot(feature_directions[layer_name][f1],
                                   feature_directions[layer_name][f2]))
                pairwise[f"{f1}_vs_{f2}"] = round(cos, 4)

        # Mean absolute cosine (measure of overall orthogonality)
        abs_cosines = [abs(v) for v in pairwise.values()]
        mean_abs_cos = round(float(np.mean(abs_cosines)), 4) if abs_cosines else None

        orthogonality_map[layer_name] = {
            "pairwise_cosine": pairwise,
            "mean_abs_cosine": mean_abs_cos,
            "n_features": n_feats,
        }

        log_time(f"\n  {layer_name} feature orthogonality (mean |cos|={mean_abs_cos}):")
        for pair, cos in sorted(pairwise.items(), key=lambda x: abs(x[1]), reverse=True):
            log_time(f"    {pair}: {cos}")

    # Key analysis: Are grammar-grammar pairs less orthogonal than semantics-semantics?
    # Or are all pairs similarly orthogonal?
    mlp_pairwise = orthogonality_map.get("L0_mlp_out", {}).get("pairwise_cosine", {})

    grammar_feats = [f for f, (_, _, t) in features.items() if t == "grammar"]
    semantic_feats = [f for f, (_, _, t) in features.items() if t == "semantics"]

    grammar_grammar_cos = []
    semantic_semantic_cos = []
    grammar_semantic_cos = []

    for pair, cos in mlp_pairwise.items():
        f1, f2 = pair.split("_vs_")
        t1 = features.get(f1, (None, None, None))[2]
        t2 = features.get(f2, (None, None, None))[2]
        if t1 == "grammar" and t2 == "grammar":
            grammar_grammar_cos.append(abs(cos))
        elif t1 == "semantics" and t2 == "semantics":
            semantic_semantic_cos.append(abs(cos))
        elif t1 and t2:
            grammar_semantic_cos.append(abs(cos))

    type_analysis = {
        "grammar_grammar_mean_abs_cos": round(float(np.mean(grammar_grammar_cos)), 4) if grammar_grammar_cos else None,
        "semantic_semantic_mean_abs_cos": round(float(np.mean(semantic_semantic_cos)), 4) if semantic_semantic_cos else None,
        "grammar_semantic_mean_abs_cos": round(float(np.mean(grammar_semantic_cos)), 4) if grammar_semantic_cos else None,
    }

    log_time(f"\n  Cross-type analysis (L0 MLP):")
    for k, v in type_analysis.items():
        log_time(f"    {k}: {v}")

    # "Two representation systems" consistency check across features
    consistency_check = {}
    for feat_name in feature_directions.get("embed", {}):
        if feat_name in feature_directions.get("L0_mlp_out", {}) and feat_name in feature_directions.get("L1_mlp_out", {}):
            embed_vs_mlp = float(np.dot(feature_directions["embed"][feat_name],
                                        feature_directions["L0_mlp_out"][feat_name]))
            mlp_consecutive = float(np.dot(feature_directions["L0_mlp_out"][feat_name],
                                           feature_directions["L1_mlp_out"][feat_name]))
            consistency_check[feat_name] = {
                "embed_vs_L0_mlp": round(embed_vs_mlp, 4),
                "L0_mlp_vs_L1_mlp": round(mlp_consecutive, 4),
                "follows_two_system_pattern": embed_vs_mlp < 0.4 and mlp_consecutive > 0.7,
            }

    log_time(f"\n  'Two representation systems' consistency:")
    for feat, check in consistency_check.items():
        pattern = "YES" if check["follows_two_system_pattern"] else "NO"
        log_time(f"    {feat}: embed_vs_mlp={check['embed_vs_L0_mlp']}, "
                 f"mlp_stability={check['L0_mlp_vs_L1_mlp']}, pattern={pattern}")

    results = {
        "model": model_name,
        "features_tested": list(features.keys()),
        "probe_accuracies": feature_probe_accs,
        "orthogonality_map": orthogonality_map,
        "type_analysis": type_analysis,
        "two_systems_consistency": consistency_check,
    }

    out_path = RESULT_DIR / f"{model_name}_part3_orthogonality_map.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 3 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Part 4: Bidirectional Causal Transport + Random Direction Analysis
# ============================================================

def run_part4(model_name):
    """
    Deep dive into causal transport directionality.

    Tests:
    1. ADD Δ_number → should decrease sing-plur score (more plural-like)
    2. SUBTRACT Δ_number → should increase sing-plur score (more singular-like)
    3. ADD Δ_animacy → control: should NOT change number prediction
    4. ADD 5 different random directions → statistical baseline
    5. NEGATE random direction → test if negation flips effect

    This resolves:
    - Phase 262's direction reversal puzzle
    - Phase 262's Qwen3 random direction 52% effect
    - Whether effects are truly direction-specific
    """
    import torch

    log_time(f"=== Part 4: Bidirectional Causal Transport for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_for_phase263(model_name)
    embed_layer = model.get_input_embeddings()
    embed_device = embed_layer.weight.device

    # Extract directions
    log_time("Step 1: Extracting directions...")

    # Number direction
    sing_embeds = []
    plur_embeds = []
    for s, p in zip(TRAIN_SING, TRAIN_PLUR):
        s_ids = tokenizer.encode(s, add_special_tokens=False)
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        if len(s_ids) == 1 and len(p_ids) == 1:
            with torch.no_grad():
                s_emb = embed_layer.weight[s_ids[0]].detach().float().cpu().numpy()
                p_emb = embed_layer.weight[p_ids[0]].detach().float().cpu().numpy()
            sing_embeds.append(s_emb)
            plur_embeds.append(p_emb)

    embed_data = {"sing": sing_embeds, "plur": plur_embeds}
    probe_acc_num, delta_number = train_probe(embed_data)
    log_time(f"  Number probe: {probe_acc_num}")

    # Animacy direction
    anim_embeds = []
    inanim_embeds = []
    for a, ia in zip(ANIMATE[:40], INANIMATE[:40]):
        a_ids = tokenizer.encode(a, add_special_tokens=False)
        ia_ids = tokenizer.encode(ia, add_special_tokens=False)
        if len(a_ids) == 1 and len(ia_ids) == 1:
            with torch.no_grad():
                a_emb = embed_layer.weight[a_ids[0]].detach().float().cpu().numpy()
                ia_emb = embed_layer.weight[ia_ids[0]].detach().float().cpu().numpy()
            anim_embeds.append(a_emb)
            inanim_embeds.append(ia_emb)

    anim_data = {"animate": anim_embeds, "inanimate": inanim_embeds}
    probe_acc_anim, delta_animacy = train_probe(anim_data)
    log_time(f"  Animacy probe: {probe_acc_anim}")

    # Cross-direction cosine
    cross_cos = float(np.dot(delta_number, delta_animacy))
    log_time(f"  Cosine(number_dir, animacy_dir): {cross_cos:.4f}")

    # Random directions (5 different seeds)
    rng = np.random.RandomState(2026)
    random_directions = []
    for i in range(5):
        r = rng.randn(info.d_model).astype(np.float32)
        r = r / (np.linalg.norm(r) + 1e-10)
        random_directions.append(r)

    # Get verb token IDs
    sing_verb_ids = [safe_get_token_id(tokenizer, v) for v in SING_VERBS if safe_get_token_id(tokenizer, v) is not None]
    plur_verb_ids = [safe_get_token_id(tokenizer, v) for v in PLUR_VERBS if safe_get_token_id(tokenizer, v) is not None]

    # Test conditions
    alpha = 8.0
    test_words = TEST_SING[:20]

    conditions = {
        "add_number": delta_number,
        "subtract_number": -delta_number,
        "add_animacy": delta_animacy,
        "subtract_animacy": -delta_animacy,
        "add_random_1": random_directions[0],
        "add_random_2": random_directions[1],
        "add_random_3": random_directions[2],
        "add_random_4": random_directions[3],
        "add_random_5": random_directions[4],
        "negate_random_1": -random_directions[0],
    }

    results = {
        "model": model_name,
        "alpha": alpha,
        "n_test_words": len(test_words),
        "number_probe_acc": probe_acc_num,
        "animacy_probe_acc": probe_acc_anim,
        "cross_direction_cosine": round(cross_cos, 4),
        "per_condition": {},
    }

    for cond_name, direction in conditions.items():
        score_changes = []

        for wi, word in enumerate(test_words):
            prompt = f"The {word}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(embed_device)
            attn_mask = inputs["attention_mask"].to(embed_device)
            seq_len = input_ids.shape[1]
            offset = get_special_token_offset(tokenizer, prompt)
            subj_pos = 1 + offset

            with torch.no_grad():
                base_embed = embed_layer(input_ids).detach().clone()

            position_ids = torch.arange(seq_len, device=embed_device).unsqueeze(0)

            # Baseline
            with torch.no_grad():
                out = model(inputs_embeds=base_embed.to(model.dtype),
                           attention_mask=attn_mask, position_ids=position_ids)
            baseline_logits = out.logits[0, -1, :].float().cpu().numpy()
            bl_sing = float(np.mean([baseline_logits[tid] for tid in sing_verb_ids]))
            bl_plur = float(np.mean([baseline_logits[tid] for tid in plur_verb_ids]))
            bl_score = bl_sing - bl_plur

            # Intervention
            modified_embed = base_embed.clone()
            delta_tensor = torch.tensor(alpha * direction, dtype=base_embed.dtype, device=embed_device)
            modified_embed[0, subj_pos, :] += delta_tensor

            with torch.no_grad():
                out = model(inputs_embeds=modified_embed.to(model.dtype),
                           attention_mask=attn_mask, position_ids=position_ids)
            logits = out.logits[0, -1, :].float().cpu().numpy()

            sing_v = float(np.mean([logits[tid] for tid in sing_verb_ids]))
            plur_v = float(np.mean([logits[tid] for tid in plur_verb_ids]))
            score = sing_v - plur_v

            score_changes.append(score - bl_score)

        mean_change = float(np.mean(score_changes))
        std_change = float(np.std(score_changes))
        n_decreased = sum(1 for c in score_changes if c < 0)

        results["per_condition"][cond_name] = {
            "mean_score_change": round(mean_change, 4),
            "std_score_change": round(std_change, 4),
            "n_decreased": n_decreased,
            "n_total": len(score_changes),
            "frac_decreased": round(n_decreased / len(score_changes), 4),
        }

        log_time(f"  {cond_name}: mean_change={mean_change:+.4f}, "
                 f"frac_decreased={n_decreased}/{len(score_changes)}")

    # Key analysis
    add_num = results["per_condition"]["add_number"]["mean_score_change"]
    sub_num = results["per_condition"]["subtract_number"]["mean_score_change"]
    add_anim = results["per_condition"]["add_animacy"]["mean_score_change"]
    sub_anim = results["per_condition"]["subtract_animacy"]["mean_score_change"]

    random_means = [results["per_condition"][f"add_random_{i+1}"]["mean_score_change"] for i in range(5)]
    random_mean = float(np.mean(random_means))
    random_std = float(np.std(random_means))

    # Is add_number vs subtract_number symmetric?
    asymmetry = abs(add_num + sub_num) / (abs(add_num) + abs(sub_num) + 1e-10)

    # Is number effect significantly different from random?
    number_z_score = (add_num - random_mean) / (random_std + 1e-10)

    # Is animacy effect on number prediction near zero?
    animacy_interference = abs(add_anim)

    results["key_analysis"] = {
        "add_vs_subtract_asymmetry": round(asymmetry, 4),
        "number_z_score_vs_random": round(number_z_score, 4),
        "animacy_interference_on_number": round(animacy_interference, 4),
        "random_direction_mean": round(random_mean, 4),
        "random_direction_std": round(random_std, 4),
        "interpretation": {
            "bidirectional_causal": abs(add_num) > 0.5 and abs(sub_num) > 0.5 and add_num * sub_num < 0,
            "direction_specific": abs(number_z_score) > 2.0,
            "feature_selective": animacy_interference < abs(add_num) * 0.5,
            "symmetric": asymmetry < 0.3,
        },
    }

    log_time(f"\n  Key analysis:")
    log_time(f"    Add Δ_number: {add_num:+.4f}, Subtract Δ_number: {sub_num:+.4f}")
    log_time(f"    Asymmetry: {asymmetry:.4f}")
    log_time(f"    Number Z-score vs random: {number_z_score:.2f}")
    log_time(f"    Animacy interference: {animacy_interference:.4f}")
    log_time(f"    Random direction mean: {random_mean:+.4f} ± {random_std:.4f}")

    for k, v in results["key_analysis"]["interpretation"].items():
        log_time(f"    {k}: {v}")

    out_path = RESULT_DIR / f"{model_name}_part4_bidirectional.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 4 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 263: Trajectory & Stability")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--part", type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    if args.part == 1:
        run_part1(args.model)
    elif args.part == 2:
        run_part2(args.model)
    elif args.part == 3:
        run_part3(args.model)
    elif args.part == 4:
        run_part4(args.model)


if __name__ == "__main__":
    main()
