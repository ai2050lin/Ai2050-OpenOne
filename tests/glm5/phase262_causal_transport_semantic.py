"""
Phase 262: Cross-Prompt Causal Transport & Semantic Extension
=============================================================

4 experiment parts addressing the deepest questions from Phase 260-261:

  Part 1: Cross-Prompt Causal Transport (CRITICAL — validates probe = real direction)
    - Extract Δ_number from training words' embeddings
    - Apply to test words at embedding level (causal intervention)
    - Measure effect on verb prediction (dose-response curve)
    - Random direction control
    → If Δ_number causally changes verb prediction on NEW words,
      it's a real abstract number direction, not lexical memorization

  Part 2: Semantic (Animacy) Extension (MOST IMPORTANT LONG-TERM)
    - Probe animacy (animate vs inanimate) at each layer
    - Direction cosine analysis: embed vs attn vs mlp
    - MLP direction stability across layers
    → Tests if "two representation systems" is universal or grammar-specific

  Part 3: GLM4 L0 Rotation Responsible Heads
    - Compute per-head contribution to L0 attention output
    - Measure alignment with number direction (cosine)
    - Find heads with negative cosine (rotation mechanism)
    → Fills mechanism gap for GLM4's direction flip

  Part 4: Suppression-Promotion Pair Test
    - Per-head logit contribution at verb position
    - Simulate ablation: no_suppressor, no_promoter, no_both
    - Distinguish balance pair vs functional division
    → Completes grammar mechanism picture

Usage:
  python tests/glm5/phase262_causal_transport_semantic.py --model qwen3 --part 1
  python tests/glm5/phase262_causal_transport_semantic.py --model glm4 --part 2
  python tests/glm5/phase262_causal_transport_semantic.py --model glm4 --part 3
  python tests/glm5/phase262_causal_transport_semantic.py --model qwen3 --part 4
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase262_causal_semantic")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Verb tokens for logit analysis =====
SING_VERBS = ["runs", "walks", "sits", "is", "has", "does", "goes", "was", "eats", "makes"]
PLUR_VERBS = ["run", "walk", "sit", "are", "have", "do", "go", "were", "eat", "make"]

# ===== Training subjects for extracting Δ_number (50 pairs) =====
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

# ===== Test subjects (NOT in training set — 30 words) =====
TEST_SING = [
    "bear", "eagle", "rabbit", "tiger", "whale", "fox", "deer", "wolf",
    "snake", "crow", "ant", "owl", "penguin", "dolphin", "spider",
    "lamp", "clock", "plate", "cup", "glass", "pillow", "blanket",
    "hammer", "rope", "ring", "coin", "letter", "map", "photo", "key"
]

# ===== Grammar prompts for Part 3/4 =====
SINGULAR_SUBJECTS = TRAIN_SING + TEST_SING
PLURAL_SUBJECTS = TRAIN_PLUR + [
    "bears", "eagles", "rabbits", "tigers", "whales", "foxes", "deer", "wolves",
    "snakes", "crows", "ants", "owls", "penguins", "dolphins", "spiders",
    "lamps", "clocks", "plates", "cups", "glasses", "pillows", "blankets",
    "hammers", "ropes", "rings", "coins", "letters", "maps", "photos", "keys"
]

# ===== Animate/Inanimate for Part 2 =====
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


def load_model_for_phase262(model_name, need_attn_weights=False):
    """
    Load model with appropriate attention implementation.
    - need_attn_weights=True: must use eager (for attention pattern access)
    - need_attn_weights=False: try flash_attention_2 first, fallback to eager
    """
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


def compute_separation(data_sing, data_plur, direction):
    """Compute separation (Cohen's d) along direction"""
    proj_s = np.dot(data_sing, direction) / (np.linalg.norm(direction) + 1e-10)
    proj_p = np.dot(data_plur, direction) / (np.linalg.norm(direction) + 1e-10)
    mean_diff = np.mean(proj_s) - np.mean(proj_p)
    pooled_std = np.sqrt((np.var(proj_s) + np.var(proj_p)) / 2 + 1e-10)
    return float(mean_diff / pooled_std)


# ============================================================
# Part 1: Cross-Prompt Causal Transport
# ============================================================

def run_part1(model_name):
    """
    CRITICAL EXPERIMENT: Validate that probe direction = real causal direction.

    Method:
    1. Extract Δ_number from training words' embeddings
    2. Apply Δ_number to TEST words (never seen during extraction)
    3. Measure if verb prediction changes (sing→plur shift)
    4. Compare with random direction control

    If Δ_number causally changes verb prediction on new words,
    it's a real abstract number direction, not lexical memorization.
    """
    import torch

    log_time(f"=== Part 1: Cross-Prompt Causal Transport for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_for_phase262(model_name)
    embed_layer = model.get_input_embeddings()
    embed_device = embed_layer.weight.device

    # Step 1: Extract Δ_number from training word embeddings
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
    log_time(f"  Collected {len(sing_embeds)} sing + {len(plur_embeds)} plur embeddings")

    # Method A: Mean difference direction
    delta_mean = np.mean(plur_embeds, axis=0) - np.mean(sing_embeds, axis=0)
    delta_mean_norm = delta_mean / (np.linalg.norm(delta_mean) + 1e-10)

    # Method B: Probe direction
    embed_data = {"sing": sing_embeds.tolist(), "plur": plur_embeds.tolist()}
    probe_acc, delta_probe = train_probe(embed_data)

    log_time(f"  Embedding probe accuracy: {probe_acc}")
    log_time(f"  Δ_mean direction norm: {np.linalg.norm(delta_mean):.2f}")
    log_time(f"  Cosine(mean_dir, probe_dir): {float(np.dot(delta_mean_norm, delta_probe)):.4f}")

    # Use probe direction (more robust than mean difference)
    delta_number = delta_probe

    # Generate random direction control (same norm as delta_mean)
    rng = np.random.RandomState(42)
    delta_random = rng.randn(info.d_model).astype(np.float32)
    delta_random = delta_random / (np.linalg.norm(delta_random) + 1e-10)

    # Get verb token IDs
    sing_verb_ids = []
    plur_verb_ids = []
    for v in SING_VERBS:
        tid = safe_get_token_id(tokenizer, v)
        if tid is not None:
            sing_verb_ids.append(tid)
    for v in PLUR_VERBS:
        tid = safe_get_token_id(tokenizer, v)
        if tid is not None:
            plur_verb_ids.append(tid)
    log_time(f"  Sing verb tokens: {len(sing_verb_ids)}, Plur verb tokens: {len(plur_verb_ids)}")

    # Step 2: Test causal transport on test words
    alpha_values = [1.0, 2.0, 4.0, 8.0, 16.0]
    log_time(f"Step 2: Testing causal transport on {len(TEST_SING)} test words, alphas={alpha_values}")

    results = {
        "model": model_name,
        "n_train_pairs": len(sing_embeds),
        "n_test_words": len(TEST_SING),
        "alpha_values": alpha_values,
        "embedding_probe_acc": probe_acc,
        "cosine_mean_vs_probe": round(float(np.dot(delta_mean_norm, delta_probe)), 4),
        "per_word_results": [],
        "aggregate": {},
    }

    for wi, word in enumerate(TEST_SING):
        prompt = f"The {word}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(embed_device)
        attn_mask = inputs["attention_mask"].to(embed_device)
        seq_len = input_ids.shape[1]

        offset = get_special_token_offset(tokenizer, prompt)
        subj_pos = 1 + offset

        # Get embedding
        with torch.no_grad():
            base_embed = embed_layer(input_ids).detach().clone()

        # Baseline: run model, get verb logits
        position_ids = torch.arange(seq_len, device=embed_device).unsqueeze(0)
        with torch.no_grad():
            out = model(inputs_embeds=base_embed.to(model.dtype),
                       attention_mask=attn_mask, position_ids=position_ids)
        baseline_logits = out.logits[0, -1, :].float().cpu().numpy()

        # Compute baseline verb scores
        bl_sing = float(np.mean([baseline_logits[tid] for tid in sing_verb_ids]))
        bl_plur = float(np.mean([baseline_logits[tid] for tid in plur_verb_ids]))
        bl_score = bl_sing - bl_plur  # positive = more singular-like

        word_result = {
            "word": word,
            "baseline_sing_verb_logit": round(bl_sing, 4),
            "baseline_plur_verb_logit": round(bl_plur, 4),
            "baseline_score": round(bl_score, 4),
            "delta_transport": [],
            "random_transport": [],
        }

        # Delta_number intervention: add alpha * delta_number to subject position
        for alpha in alpha_values:
            modified_embed = base_embed.clone()
            delta_tensor = torch.tensor(alpha * delta_number, dtype=base_embed.dtype, device=embed_device)
            modified_embed[0, subj_pos, :] += delta_tensor

            with torch.no_grad():
                out = model(inputs_embeds=modified_embed.to(model.dtype),
                           attention_mask=attn_mask, position_ids=position_ids)
            logits = out.logits[0, -1, :].float().cpu().numpy()

            sing_v = float(np.mean([logits[tid] for tid in sing_verb_ids]))
            plur_v = float(np.mean([logits[tid] for tid in plur_verb_ids]))
            score = sing_v - plur_v

            word_result["delta_transport"].append({
                "alpha": alpha,
                "score": round(score, 4),
                "score_change": round(score - bl_score, 4),
                "sing_verb_logit": round(sing_v, 4),
                "plur_verb_logit": round(plur_v, 4),
            })

        # Random direction control (2 alpha values)
        for alpha in [4.0, 8.0]:
            modified_embed = base_embed.clone()
            rand_tensor = torch.tensor(alpha * delta_random, dtype=base_embed.dtype, device=embed_device)
            modified_embed[0, subj_pos, :] += rand_tensor

            with torch.no_grad():
                out = model(inputs_embeds=modified_embed.to(model.dtype),
                           attention_mask=attn_mask, position_ids=position_ids)
            logits = out.logits[0, -1, :].float().cpu().numpy()

            sing_v = float(np.mean([logits[tid] for tid in sing_verb_ids]))
            plur_v = float(np.mean([logits[tid] for tid in plur_verb_ids]))
            score = sing_v - plur_v

            word_result["random_transport"].append({
                "alpha": alpha,
                "score": round(score, 4),
                "score_change": round(score - bl_score, 4),
            })

        results["per_word_results"].append(word_result)

        if wi % 5 == 0:
            # Log progress with current word's results
            dt = word_result["delta_transport"]
            rt = word_result["random_transport"]
            log_time(f"  Word {wi+1}/{len(TEST_SING)}: '{word}' | "
                     f"baseline={bl_score:.2f}, "
                     f"delta(α=8)={dt[3]['score_change']:+.2f}, "
                     f"rand(α=8)={rt[1]['score_change']:+.2f}")

    # Step 3: Aggregate results
    log_time("Step 3: Aggregating results...")

    # Delta transport: for each alpha, compute mean score change across test words
    for ai, alpha in enumerate(alpha_values):
        changes = [r["delta_transport"][ai]["score_change"] for r in results["per_word_results"]]
        n_decreased = sum(1 for c in changes if c < 0)  # score decreased = more plural-like
        results["aggregate"][f"delta_alpha{alpha}"] = {
            "mean_change": round(float(np.mean(changes)), 4),
            "std_change": round(float(np.std(changes)), 4),
            "n_decreased": n_decreased,
            "n_total": len(changes),
            "frac_decreased": round(n_decreased / len(changes), 4),
        }

    # Random control
    for ri, alpha in enumerate([4.0, 8.0]):
        changes = [r["random_transport"][ri]["score_change"] for r in results["per_word_results"]]
        n_decreased = sum(1 for c in changes if c < 0)
        results["aggregate"][f"random_alpha{alpha}"] = {
            "mean_change": round(float(np.mean(changes)), 4),
            "std_change": round(float(np.std(changes)), 4),
            "n_decreased": n_decreased,
            "n_total": len(changes),
            "frac_decreased": round(n_decreased / len(changes), 4),
        }

    # Key comparison: delta vs random at alpha=8
    delta_changes_8 = [r["delta_transport"][3]["score_change"] for r in results["per_word_results"]]
    random_changes_8 = [r["random_transport"][1]["score_change"] for r in results["per_word_results"]]
    results["aggregate"]["key_comparison_alpha8"] = {
        "delta_mean": round(float(np.mean(delta_changes_8)), 4),
        "random_mean": round(float(np.mean(random_changes_8)), 4),
        "delta_frac_decreased": round(sum(1 for c in delta_changes_8 if c < 0) / len(delta_changes_8), 4),
        "random_frac_decreased": round(sum(1 for c in random_changes_8 if c < 0) / len(random_changes_8), 4),
        "interpretation": (
            "REAL_ABSTRACT_DIRECTION" if np.mean(delta_changes_8) < -0.5 and
            sum(1 for c in delta_changes_8 if c < 0) / len(delta_changes_8) > 0.7
            else "WEAK_OR_LEXICAL"
        ),
    }

    log_time("Aggregate results:")
    for key, val in results["aggregate"].items():
        log_time(f"  {key}: {val}")

    # Save
    out_path = RESULT_DIR / f"{model_name}_part1_causal_transport.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 1 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Part 2: Semantic (Animacy) Extension
# ============================================================

def run_part2(model_name):
    """
    Test if MLP builds independent semantic direction for animacy.
    Parallel analysis to Phase 261 Part 2 but for animacy instead of number.

    Key question: Does "two representation systems" (embed space vs MLP internal space)
    apply to semantic features, or is it grammar-specific?
    """
    import torch
    from model_utils import get_layers

    log_time(f"=== Part 2: Semantic (Animacy) Extension for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_for_phase262(model_name)
    input_device = get_input_device(model)
    layers = get_layers(model)

    n_words = min(len(ANIMATE), len(INANIMATE))
    anim_words = ANIMATE[:n_words]
    inanim_words = INANIMATE[:n_words]
    log_time(f"Testing {n_words} animate + {n_words} inanimate words")

    # Collect states at key layers
    mid_layer = info.n_layers // 2
    late_layer = info.n_layers - 5
    state_names = ["embed", "L0_attn_out", "L0_mlp_out", "L1_attn_out", "L1_mlp_out",
                   f"L{mid_layer}", f"L{late_layer}", f"L{info.n_layers}"]
    states = {name: {"animate": [], "inanimate": []} for name in state_names}

    for wi, (aw, iw) in enumerate(zip(anim_words, inanim_words)):
        for word, label in [(aw, "animate"), (iw, "inanimate")]:
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
                        if len(output) >= 2 and output[1] is not None:
                            captured[key + "_attn"] = output[1].detach().float().cpu()
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

            if out.hidden_states:
                if len(out.hidden_states) > 0:
                    states["embed"][label].append(out.hidden_states[0][0, subj_pos, :].float().cpu().numpy())
                if mid_layer < len(out.hidden_states):
                    states[f"L{mid_layer}"][label].append(out.hidden_states[mid_layer][0, subj_pos, :].float().cpu().numpy())
                if late_layer < len(out.hidden_states):
                    states[f"L{late_layer}"][label].append(out.hidden_states[late_layer][0, subj_pos, :].float().cpu().numpy())
                if info.n_layers < len(out.hidden_states):
                    states[f"L{info.n_layers}"][label].append(out.hidden_states[info.n_layers][0, subj_pos, :].float().cpu().numpy())

            for cap_name, state_name in [("L0_attn_out", "L0_attn_out"), ("L0_full_out", "L0_mlp_out"),
                                          ("L1_attn_out", "L1_attn_out"), ("L1_full_out", "L1_mlp_out")]:
                if cap_name in captured:
                    states[state_name][label].append(captured[cap_name][0, subj_pos, :].float().cpu().numpy())

        if wi % 10 == 0:
            log_time(f"  Word pair {wi+1}/{n_words}")

    # Train probes and extract directions
    directions = {}
    probe_accs = {}
    for name in state_names:
        acc, direction = train_probe(states[name])
        if acc is not None:
            directions[name] = direction
            probe_accs[name] = acc

    log_time("Animacy probe accuracies:")
    for name in state_names:
        if name in probe_accs:
            log_time(f"  {name}: {probe_accs[name]}")

    # Pairwise direction cosine
    cos_matrix = {}
    state_list = list(directions.keys())
    for i, s1 in enumerate(state_list):
        for j, s2 in enumerate(state_list):
            if j <= i:
                continue
            cos = float(np.dot(directions[s1], directions[s2]))
            cos_matrix[f"{s1}_vs_{s2}"] = round(cos, 4)

    # Key comparisons (parallel to Phase 261 number analysis)
    key_pairs = [
        "embed_vs_L0_attn_out", "embed_vs_L0_mlp_out", "L0_attn_out_vs_L0_mlp_out",
        "L0_mlp_out_vs_L1_mlp_out", "embed_vs_L" + str(info.n_layers)
    ]

    log_time("Key animacy direction cosines:")
    for pair in key_pairs:
        if pair in cos_matrix:
            log_time(f"  {pair}: {cos_matrix[pair]}")

    # Also do NUMBER analysis on same data for direct comparison
    # Use training subjects to extract number direction at same layers
    log_time("Running NUMBER direction analysis for comparison...")
    n_num_prompts = 60
    num_states = {name: {"sing": [], "plur": []} for name in state_names}

    for i in range(min(n_num_prompts, len(TRAIN_SING))):
        for subj, verb, label in [(TRAIN_SING[i], "sits", "sing"), (TRAIN_PLUR[i], "sit", "plur")]:
            prompt = f"The {subj} {verb}"
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

            if out.hidden_states:
                if len(out.hidden_states) > 0:
                    num_states["embed"][label].append(out.hidden_states[0][0, subj_pos, :].float().cpu().numpy())
                if mid_layer < len(out.hidden_states):
                    num_states[f"L{mid_layer}"][label].append(out.hidden_states[mid_layer][0, subj_pos, :].float().cpu().numpy())
                if late_layer < len(out.hidden_states):
                    num_states[f"L{late_layer}"][label].append(out.hidden_states[late_layer][0, subj_pos, :].float().cpu().numpy())
                if info.n_layers < len(out.hidden_states):
                    num_states[f"L{info.n_layers}"][label].append(out.hidden_states[info.n_layers][0, subj_pos, :].float().cpu().numpy())

            for cap_name, state_name in [("L0_attn_out", "L0_attn_out"), ("L0_full_out", "L0_mlp_out"),
                                          ("L1_attn_out", "L1_attn_out"), ("L1_full_out", "L1_mlp_out")]:
                if cap_name in captured:
                    num_states[state_name][label].append(captured[cap_name][0, subj_pos, :].float().cpu().numpy())

        if i % 20 == 0:
            log_time(f"  Number prompt {i+1}/{n_num_prompts}")

    # Number direction cosine
    num_directions = {}
    num_probe_accs = {}
    for name in state_names:
        acc, direction = train_probe(num_states[name])
        if acc is not None:
            num_directions[name] = direction
            num_probe_accs[name] = acc

    num_cos_matrix = {}
    num_state_list = list(num_directions.keys())
    for i, s1 in enumerate(num_state_list):
        for j, s2 in enumerate(num_state_list):
            if j <= i:
                continue
            cos = float(np.dot(num_directions[s1], num_directions[s2]))
            num_cos_matrix[f"{s1}_vs_{s2}"] = round(cos, 4)

    # Cross-feature direction cosine: are number and animacy directions the same?
    cross_cos = {}
    for name in state_names:
        if name in directions and name in num_directions:
            cross_cos[name] = round(float(np.dot(directions[name], num_directions[name])), 4)

    log_time("Number direction cosines (for comparison):")
    for pair in key_pairs:
        if pair in num_cos_matrix:
            log_time(f"  {pair}: {num_cos_matrix[pair]}")

    log_time("Cross-feature cosine (animacy_dir vs number_dir at same layer):")
    for name, cos in sorted(cross_cos.items()):
        log_time(f"  {name}: {cos}")

    # Summary comparison
    results = {
        "model": model_name,
        "n_animacy_words": n_words,
        "n_number_prompts": n_num_prompts,
        "animacy_probe_accuracies": probe_accs,
        "animacy_direction_cosine": cos_matrix,
        "number_probe_accuracies": num_probe_accs,
        "number_direction_cosine": num_cos_matrix,
        "cross_feature_cosine": cross_cos,
        "key_comparison": {
            "animacy_embed_vs_L0_mlp": cos_matrix.get("embed_vs_L0_mlp_out", None),
            "animacy_L0_mlp_vs_L1_mlp": cos_matrix.get("L0_mlp_out_vs_L1_mlp_out", None),
            "number_embed_vs_L0_mlp": num_cos_matrix.get("embed_vs_L0_mlp_out", None),
            "number_L0_mlp_vs_L1_mlp": num_cos_matrix.get("L0_mlp_out_vs_L1_mlp_out", None),
        },
    }

    out_path = RESULT_DIR / f"{model_name}_part2_semantic_extension.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 2 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Part 3: GLM4 L0 Rotation Responsible Heads
# ============================================================

def run_part3(model_name):
    """
    Find which L0 heads are responsible for rotating the number direction
    to near-orthogonal (cos=-0.007 in GLM4).

    Method:
    1. Compute per-head contribution to L0 attention output at subject position
    2. Measure each head's alignment with the embedding number direction
    3. Heads with large magnitude + negative cosine are rotation responsible
    """
    import torch
    from model_utils import get_layers

    log_time(f"=== Part 3: GLM4 L0 Rotation Responsible Heads for {model_name} ===")

    # Must use eager for attention weights
    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_for_phase262(
        model_name, need_attn_weights=True
    )
    input_device = get_input_device(model)
    layers = get_layers(model)

    n_prompts = 60
    sing_prompts = [f"The {TRAIN_SING[i]} sits" for i in range(min(n_prompts, len(TRAIN_SING)))]
    plur_prompts = [f"The {TRAIN_PLUR[i]} sit" for i in range(min(n_prompts, len(TRAIN_PLUR)))]
    all_prompts = sing_prompts + plur_prompts
    labels_list = ["sing"] * len(sing_prompts) + ["plur"] * len(plur_prompts)

    log_time(f"Total prompts: {len(all_prompts)}")

    # Collect embedding and attention data
    embed_data = {"sing": [], "plur": []}
    attn_data = {"sing": [], "plur": []}

    # Per-head contributions
    head_contribs = defaultdict(lambda: {"sing": [], "plur": []})

    # Get L0 weights
    try:
        W_O = layers[0].self_attn.o_proj.weight.detach().cpu().float().numpy()
        W_V = layers[0].self_attn.v_proj.weight.detach().cpu().float().numpy()
    except Exception as e:
        log_time(f"ERROR: Cannot access L0 weights: {e}")
        del model; gc.collect(); torch.cuda.empty_cache()
        return {"error": str(e)}

    for pi, (prompt, label) in enumerate(zip(all_prompts, labels_list)):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        offset = get_special_token_offset(tokenizer, prompt)
        subj_pos = 1 + offset

        captured = {}
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    if len(output) >= 2 and output[1] is not None:
                        captured[key + "_attn_w"] = output[1].detach().float().cpu()
                    captured[key + "_out"] = output[0].detach().float().cpu()
                else:
                    captured[key + "_out"] = output.detach().float().cpu()
            return hook

        hooks = []
        hooks.append(layers[0].self_attn.register_forward_hook(make_hook("L0_attn")))

        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True, output_attentions=True)
        except Exception:
            for h in hooks: h.remove()
            continue
        for h in hooks: h.remove()

        if out.hidden_states and len(out.hidden_states) > 0:
            embed_vec = out.hidden_states[0][0, subj_pos, :].float().cpu().numpy()
            embed_data[label].append(embed_vec)

        if "L0_attn_out" in captured:
            attn_vec = captured["L0_attn_out"][0, subj_pos, :].float().cpu().numpy()
            attn_data[label].append(attn_vec)

        # Compute per-head contributions
        if "L0_attn_attn_w" in captured and out.hidden_states is not None:
            attn_weights = captured["L0_attn_attn_w"]  # [1, n_heads, seq, seq]

            # Get layernormed input for computing V
            embed_tensor = out.hidden_states[0]  # [1, seq, d_model]
            ln = layers[0].input_layernorm
            ln_device = ln.weight.device

            with torch.no_grad():
                normalized = ln(embed_tensor.to(ln_device).to(ln.weight.dtype))
            normalized_np = normalized.float().cpu().numpy()  # [1, seq, d_model]

            # Compute V = W_V @ normalized.T  → [n_kv*head_dim, seq]
            V_all = W_V @ normalized_np[0].T  # [n_kv*head_dim, seq]

            seq_len = attn_weights.shape[-1]
            for h in range(min(n_heads, attn_weights.shape[1])):
                # Attention from subject position
                attn_h = attn_weights[0, h, subj_pos, :].numpy()  # [seq]

                # Value vectors for this head's KV group
                kv_h = h // kv_group_size
                V_h = V_all[kv_h * head_dim:(kv_h + 1) * head_dim, :]  # [head_dim, seq]

                # z_h = weighted sum of value vectors
                z_h = V_h @ attn_h  # [head_dim]

                # W_O for this head
                W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]  # [d_model, head_dim]

                # Contribution at subject position
                contrib_h = W_O_h @ z_h  # [d_model]

                head_contribs[h][label].append(contrib_h)

        if pi % 20 == 0:
            log_time(f"  Prompt {pi+1}/{len(all_prompts)}")

    # Baseline number direction
    embed_acc, embed_dir = train_probe(embed_data)
    attn_acc, attn_dir = train_probe(attn_data)

    if embed_dir is None or attn_dir is None:
        log_time("ERROR: Cannot compute baseline directions")
        del model; gc.collect(); torch.cuda.empty_cache()
        return {"error": "Cannot compute baseline directions"}

    cos_baseline = float(np.dot(embed_dir, attn_dir))
    log_time(f"Baseline: embed_probe={embed_acc}, attn_probe={attn_acc}, cos(embed,attn)={cos_baseline:.4f}")

    # Analyze each head's contribution
    head_analysis = {}
    for h in range(n_heads):
        sing_contribs = head_contribs[h]["sing"]
        plur_contribs = head_contribs[h]["plur"]

        if not sing_contribs or not plur_contribs:
            continue

        # Mean contribution
        mean_contrib = np.mean(sing_contribs + plur_contribs, axis=0)
        magnitude = float(np.linalg.norm(mean_contrib))

        # Alignment with embedding number direction
        cos_embed = float(np.dot(mean_contrib / (magnitude + 1e-10), embed_dir))

        # Alignment with attention number direction
        cos_attn = float(np.dot(mean_contrib / (magnitude + 1e-10), attn_dir)) if attn_dir is not None else 0

        # Sing/plur difference in contribution
        mean_sing = np.mean(sing_contribs, axis=0)
        mean_plur = np.mean(plur_contribs, axis=0)
        diff_vec = mean_sing - mean_plur
        diff_magnitude = float(np.linalg.norm(diff_vec))
        cos_diff_embed = float(np.dot(diff_vec / (diff_magnitude + 1e-10), embed_dir))

        # Number sensitivity: how much does this head distinguish sing/plur?
        # Compute per-prompt contribution, project on embed_dir
        sing_proj = [float(np.dot(c, embed_dir)) for c in sing_contribs]
        plur_proj = [float(np.dot(c, embed_dir)) for c in plur_contribs]
        number_sensitivity = abs(np.mean(sing_proj) - np.mean(plur_proj)) / (np.std(sing_proj + plur_proj) + 1e-10)

        head_analysis[h] = {
            "magnitude": round(magnitude, 4),
            "cos_with_embed_dir": round(cos_embed, 4),
            "cos_with_attn_dir": round(cos_attn, 4),
            "sing_plur_diff_magnitude": round(diff_magnitude, 4),
            "cos_diff_with_embed_dir": round(cos_diff_embed, 4),
            "number_sensitivity": round(float(number_sensitivity), 4),
            "mean_sing_proj_on_embed": round(float(np.mean(sing_proj)), 4),
            "mean_plur_proj_on_embed": round(float(np.mean(plur_proj)), 4),
        }

    # Sort heads by different criteria
    sorted_by_magnitude = sorted(head_analysis.items(), key=lambda x: -x[1]["magnitude"])
    sorted_by_neg_cos = sorted(head_analysis.items(), key=lambda x: x[1]["cos_with_embed_dir"])
    sorted_by_sensitivity = sorted(head_analysis.items(), key=lambda x: -x[1]["number_sensitivity"])

    log_time("\nTop 10 heads by LARGEST magnitude (biggest contributors):")
    for h, s in sorted_by_magnitude[:10]:
        log_time(f"  H{h}: mag={s['magnitude']}, cos_embed={s['cos_with_embed_dir']}, "
                 f"sensitivity={s['number_sensitivity']}")

    log_time("\nTop 10 heads by MOST NEGATIVE cos_with_embed_dir (rotation heads):")
    for h, s in sorted_by_neg_cos[:10]:
        log_time(f"  H{h}: cos_embed={s['cos_with_embed_dir']}, mag={s['magnitude']}, "
                 f"sensitivity={s['number_sensitivity']}")

    log_time("\nTop 10 heads by number sensitivity:")
    for h, s in sorted_by_sensitivity[:10]:
        log_time(f"  H{h}: sensitivity={s['number_sensitivity']}, cos_embed={s['cos_with_embed_dir']}, "
                 f"mag={s['magnitude']}")

    # Identify rotation responsible heads: large magnitude + negative cos_embed
    rotation_heads = []
    for h, s in head_analysis.items():
        if s["magnitude"] > np.median([v["magnitude"] for v in head_analysis.values()]) and s["cos_with_embed_dir"] < -0.1:
            rotation_heads.append((h, s))
    rotation_heads.sort(key=lambda x: x[1]["cos_with_embed_dir"])

    log_time(f"\nIdentified {len(rotation_heads)} rotation responsible heads (mag > median, cos < -0.1):")
    for h, s in rotation_heads:
        log_time(f"  H{h}: cos_embed={s['cos_with_embed_dir']}, mag={s['magnitude']}, "
                 f"sensitivity={s['number_sensitivity']}")

    results = {
        "model": model_name,
        "n_prompts": n_prompts,
        "baseline_cosine_embed_vs_attn": round(cos_baseline, 4),
        "head_analysis": {str(k): v for k, v in head_analysis.items()},
        "rotation_responsible_heads": [(h, s) for h, s in rotation_heads],
    }

    out_path = RESULT_DIR / f"{model_name}_part3_rotation_heads.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 3 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Part 4: Suppression-Promotion Pair Test
# ============================================================

def run_part4(model_name):
    """
    Test whether L35_H0 (suppressor) and L34_H23 (promoter) form a balance pair
    or a functional division pair.

    Method:
    - Compute per-head logit contribution at verb position
    - Simulate ablation at logit level
    - Compare 4 conditions: baseline, no_suppressor, no_promoter, no_both

    If no_both ≈ baseline → balance pair (explanation A)
    If no_both << baseline → functional division (explanation B)
    """
    import torch
    from model_utils import get_layers, get_W_U

    log_time(f"=== Part 4: Suppression-Promotion Pair Test for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_for_phase262(
        model_name, need_attn_weights=True
    )
    input_device = get_input_device(model)
    layers = get_layers(model)

    # Get W_U
    try:
        W_U = get_W_U(model, model_name)
    except Exception as e:
        log_time(f"ERROR: Cannot get W_U: {e}")
        del model; gc.collect(); torch.cuda.empty_cache()
        return {"error": str(e)}

    n_prompts = 60
    sing_prompts = [f"The {TRAIN_SING[i]} runs" for i in range(min(n_prompts, len(TRAIN_SING)))]
    plur_prompts = [f"The {TRAIN_PLUR[i]} run" for i in range(min(n_prompts, len(TRAIN_PLUR)))]
    all_prompts = sing_prompts + plur_prompts
    labels_list = ["sing"] * len(sing_prompts) + ["plur"] * len(plur_prompts)

    # Get verb token IDs
    sing_verb_ids = [safe_get_token_id(tokenizer, v) for v in SING_VERBS if safe_get_token_id(tokenizer, v) is not None]
    plur_verb_ids = [safe_get_token_id(tokenizer, v) for v in PLUR_VERBS if safe_get_token_id(tokenizer, v) is not None]

    # Target layers for head analysis
    # For Qwen3: L34, L35. For other models: last 2 layers
    if model_name == "qwen3":
        target_layers = [34, 35]
    else:
        target_layers = [info.n_layers - 2, info.n_layers - 1]

    # Check which layers have accessible weights
    accessible_layers = []
    for li in target_layers:
        try:
            w_o = layers[li].self_attn.o_proj.weight
            if hasattr(w_o, 'is_meta') and w_o.is_meta:
                log_time(f"  L{li} weights on meta device, skipping")
                continue
            w_v = layers[li].self_attn.v_proj.weight
            if hasattr(w_v, 'is_meta') and w_v.is_meta:
                log_time(f"  L{li} V weights on meta device, skipping")
                continue
            accessible_layers.append(li)
        except Exception as e:
            log_time(f"  L{li} weight access failed: {str(e)[:60]}")

    # Also try lower layers as fallback
    if not accessible_layers:
        fallback_layers = [info.n_layers // 2, 0]
        for li in fallback_layers:
            try:
                w_o = layers[li].self_attn.o_proj.weight
                if hasattr(w_o, 'is_meta') and w_o.is_meta:
                    continue
                accessible_layers.append(li)
            except Exception:
                continue

    log_time(f"Accessible layers for head analysis: {accessible_layers}")

    if not accessible_layers:
        log_time("ERROR: No accessible layers for head analysis")
        del model; gc.collect(); torch.cuda.empty_cache()
        return {"error": "No accessible layers"}

    # Collect per-head logit contributions
    all_head_contribs = defaultdict(lambda: {"sing": [], "plur": []})

    for pi, (prompt, label) in enumerate(zip(all_prompts, labels_list)):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        offset = get_special_token_offset(tokenizer, prompt)
        no_special = tokenizer.encode(prompt, add_special_tokens=False)
        verb_pos = len(no_special) - 1 + offset

        for li in accessible_layers:
            try:
                W_O = layers[li].self_attn.o_proj.weight.detach().cpu().float().numpy()
                W_V = layers[li].self_attn.v_proj.weight.detach().cpu().float().numpy()
            except Exception:
                continue

            captured = {}
            def make_attn_hook():
                def hook(module, input, output):
                    if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                        captured["attn_w"] = output[1].detach().float().cpu()
                return hook

            hook_handle = layers[li].self_attn.register_forward_hook(make_attn_hook())

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True, output_attentions=True)
            except Exception:
                hook_handle.remove()
                continue
            hook_handle.remove()

            if out.hidden_states is None or li >= len(out.hidden_states) or "attn_w" not in captured:
                continue

            resid_pre = out.hidden_states[li][0].float().cpu().numpy()
            attn_w = captured["attn_w"]

            for hi in range(min(n_heads, attn_w.shape[1])):
                W_O_h = W_O[:, hi * head_dim:(hi + 1) * head_dim]
                kv_h = hi // kv_group_size
                W_V_h = W_V[kv_h * head_dim:(kv_h + 1) * head_dim, :]

                attn_from_verb = attn_w[0, hi, verb_pos, :].numpy()
                V_h_input = W_V_h @ resid_pre.T
                z_h = V_h_input @ attn_from_verb
                head_contrib = W_O_h @ z_h
                logit_effect = W_U @ head_contrib

                sv_eff = [logit_effect[tid] for tid in sing_verb_ids]
                pv_eff = [logit_effect[tid] for tid in plur_verb_ids]

                all_head_contribs[(li, hi)][label].append({
                    "sing_verb_mean": round(float(np.mean(sv_eff)), 4) if sv_eff else 0,
                    "plur_verb_mean": round(float(np.mean(pv_eff)), 4) if pv_eff else 0,
                    "verb_diff": round(float(np.mean(sv_eff) - np.mean(pv_eff)), 4) if sv_eff and pv_eff else 0,
                })

        if pi % 10 == 0:
            log_time(f"  Prompt {pi+1}/{len(all_prompts)}")

    # Aggregate per-head contributions
    head_summary = {}
    for (li, hi), data in all_head_contribs.items():
        sing_diffs = [d["verb_diff"] for d in data.get("sing", [])]
        plur_diffs = [d["verb_diff"] for d in data.get("plur", [])]
        sing_sverb = [d["sing_verb_mean"] for d in data.get("sing", [])]
        plur_sverb = [d["sing_verb_mean"] for d in data.get("plur", [])]

        head_summary[f"L{li}_H{hi}"] = {
            "verb_diff_sing": round(float(np.mean(sing_diffs)), 4) if sing_diffs else None,
            "verb_diff_plur": round(float(np.mean(plur_diffs)), 4) if plur_diffs else None,
            "sing_verb_effect_sing": round(float(np.mean(sing_sverb)), 4) if sing_sverb else None,
            "sing_verb_effect_plur": round(float(np.mean(plur_sverb)), 4) if plur_sverb else None,
            "mean_verb_diff": round(float(np.mean(sing_diffs + plur_diffs)), 4) if sing_diffs or plur_diffs else None,
        }

    # Identify key heads
    # Suppressor: large negative mean_verb_diff
    # Promoter: large positive mean_verb_diff
    # Grammar head: large sing/plur verb_diff difference

    sorted_by_diff = sorted(head_summary.items(), key=lambda x: x[1].get("mean_verb_diff", 0) or 0)

    log_time("\nTop 5 suppressor heads (most negative verb_diff):")
    for name, s in sorted_by_diff[:5]:
        log_time(f"  {name}: mean_diff={s.get('mean_verb_diff')}, "
                 f"sing_diff={s.get('verb_diff_sing')}, plur_diff={s.get('verb_diff_plur')}")

    log_time("\nTop 5 promoter heads (most positive verb_diff):")
    for name, s in sorted_by_diff[-5:]:
        log_time(f"  {name}: mean_diff={s.get('mean_verb_diff')}, "
                 f"sing_diff={s.get('verb_diff_sing')}, plur_diff={s.get('verb_diff_plur')}")

    # Simulate ablation at logit level
    # For each prompt, compute total verb logit with and without specific heads
    log_time("\nSimulating ablation conditions...")

    # We need per-prompt per-head logit effects
    # Re-collect with more detail
    per_prompt_results = []

    for pi, (prompt, label) in enumerate(zip(all_prompts[:30], labels_list[:30])):  # Use 30 prompts for speed
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        # Get baseline verb logits
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        baseline_logits = out.logits[0, -1, :].float().cpu().numpy()
        bl_sing = float(np.mean([baseline_logits[tid] for tid in sing_verb_ids]))
        bl_plur = float(np.mean([baseline_logits[tid] for tid in plur_verb_ids]))

        prompt_result = {
            "prompt": prompt, "label": label,
            "baseline_sing_verb": round(bl_sing, 4),
            "baseline_plur_verb": round(bl_plur, 4),
            "baseline_score": round(bl_sing - bl_plur, 4),
        }

        # Collect head logit contributions for ablation simulation
        head_logit_effects = {}  # (li, hi) -> logit_effect array

        for li in accessible_layers:
            try:
                W_O = layers[li].self_attn.o_proj.weight.detach().cpu().float().numpy()
                W_V = layers[li].self_attn.v_proj.weight.detach().cpu().float().numpy()
            except Exception:
                continue

            captured = {}
            def make_hook():
                def hook(module, input, output):
                    if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                        captured["attn_w"] = output[1].detach().float().cpu()
                return hook

            hook_handle = layers[li].self_attn.register_forward_hook(make_hook())

            try:
                with torch.no_grad():
                    out2 = model(input_ids=input_ids, attention_mask=attn_mask,
                                output_hidden_states=True, output_attentions=True)
            except Exception:
                hook_handle.remove()
                continue
            hook_handle.remove()

            if out2.hidden_states is None or li >= len(out2.hidden_states) or "attn_w" not in captured:
                continue

            resid_pre = out2.hidden_states[li][0].float().cpu().numpy()
            attn_w = captured["attn_w"]

            for hi in range(min(n_heads, attn_w.shape[1])):
                W_O_h = W_O[:, hi * head_dim:(hi + 1) * head_dim]
                kv_h_idx = hi // kv_group_size
                W_V_h = W_V[kv_h_idx * head_dim:(kv_h_idx + 1) * head_dim, :]

                offset = get_special_token_offset(tokenizer, prompt)
                no_special = tokenizer.encode(prompt, add_special_tokens=False)
                verb_pos = len(no_special) - 1 + offset

                attn_from_verb = attn_w[0, hi, verb_pos, :].numpy()
                V_h_input = W_V_h @ resid_pre.T
                z_h = V_h_input @ attn_from_verb
                head_contrib = W_O_h @ z_h
                logit_effect = W_U @ head_contrib

                head_logit_effects[(li, hi)] = logit_effect

        # Simulate 4 conditions
        # Find suppressor and promoter heads
        suppressor_key = None
        promoter_key = None

        for (li, hi), effect in head_logit_effects.items():
            sv_eff = np.mean([effect[tid] for tid in sing_verb_ids])
            pv_eff = np.mean([effect[tid] for tid in plur_verb_ids])
            mean_diff = sv_eff - pv_eff

            if mean_diff < -2.0 and suppressor_key is None:
                suppressor_key = (li, hi)
            if mean_diff > 2.0 and promoter_key is None:
                promoter_key = (li, hi)

        # Compute verb scores under each condition
        def compute_verb_score(head_effects, exclude_keys=None):
            total_logit = np.zeros(W_U.shape[0])
            for key, effect in head_effects.items():
                if exclude_keys and key in exclude_keys:
                    continue
                total_logit += effect
            sv = float(np.mean([total_logit[tid] for tid in sing_verb_ids]))
            pv = float(np.mean([total_logit[tid] for tid in plur_verb_ids]))
            return round(sv - pv, 4), round(sv, 4), round(pv, 4)

        baseline_score, _, _ = compute_verb_score(head_logit_effects)
        no_supp_score, _, _ = compute_verb_score(head_logit_effects, {suppressor_key} if suppressor_key else None)
        no_prom_score, _, _ = compute_verb_score(head_logit_effects, {promoter_key} if promoter_key else None)
        no_both_score, _, _ = compute_verb_score(head_logit_effects,
                                                  {suppressor_key, promoter_key} if suppressor_key and promoter_key else None)

        prompt_result["suppressor_head"] = f"L{suppressor_key[0]}_H{suppressor_key[1]}" if suppressor_key else None
        prompt_result["promoter_head"] = f"L{promoter_key[0]}_H{promoter_key[1]}" if promoter_key else None
        prompt_result["baseline_score"] = baseline_score
        prompt_result["no_suppressor_score"] = no_supp_score
        prompt_result["no_promoter_score"] = no_prom_score
        prompt_result["no_both_score"] = no_both_score

        per_prompt_results.append(prompt_result)

        if pi % 10 == 0:
            log_time(f"  Ablation prompt {pi+1}/30")

    # Aggregate ablation results
    if per_prompt_results:
        baseline_scores = [r["baseline_score"] for r in per_prompt_results if r["baseline_score"] is not None]
        no_supp_scores = [r["no_suppressor_score"] for r in per_prompt_results if r["no_suppressor_score"] is not None]
        no_prom_scores = [r["no_promoter_score"] for r in per_prompt_results if r["no_promoter_score"] is not None]
        no_both_scores = [r["no_both_score"] for r in per_prompt_results if r["no_both_score"] is not None]

        ablation_summary = {
            "baseline_mean": round(float(np.mean(baseline_scores)), 4) if baseline_scores else None,
            "no_suppressor_mean": round(float(np.mean(no_supp_scores)), 4) if no_supp_scores else None,
            "no_promoter_mean": round(float(np.mean(no_prom_scores)), 4) if no_prom_scores else None,
            "no_both_mean": round(float(np.mean(no_both_scores)), 4) if no_both_scores else None,
        }

        # Determine interpretation
        if ablation_summary["no_both_mean"] is not None and ablation_summary["baseline_mean"] is not None:
            recovery = abs(ablation_summary["no_both_mean"] - ablation_summary["baseline_mean"])
            disruption = max(
                abs(ablation_summary.get("no_suppressor_mean", 0) - ablation_summary["baseline_mean"]),
                abs(ablation_summary.get("no_promoter_mean", 0) - ablation_summary["baseline_mean"])
            ) if ablation_summary.get("no_suppressor_mean") and ablation_summary.get("no_promoter_mean") else 0

            if recovery < 0.3 * disruption:
                ablation_summary["interpretation"] = "BALANCE_PAIR (no_both ≈ baseline → effects cancel)"
            else:
                ablation_summary["interpretation"] = "FUNCTIONAL_DIVISION (no_both still disrupted → independent functions)"

        log_time("\nAblation summary:")
        for key, val in ablation_summary.items():
            log_time(f"  {key}: {val}")
    else:
        ablation_summary = {"error": "No per-prompt results"}

    results = {
        "model": model_name,
        "accessible_layers": accessible_layers,
        "head_summary": head_summary,
        "per_prompt_ablation": per_prompt_results,
        "ablation_summary": ablation_summary,
    }

    out_path = RESULT_DIR / f"{model_name}_part4_suppression_promotion.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 4 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 262: Causal Transport & Semantic Extension")
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
