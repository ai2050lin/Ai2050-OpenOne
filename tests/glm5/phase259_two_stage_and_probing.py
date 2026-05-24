"""
Phase 259: Two-Stage Computation Hypothesis & Probing Classifier
================================================================

Based on Phase 258 deep review, 4 experiment plans:

  Part 1: Two-stage computation verification
    - Ablate driver heads → measure follower attribution change
    - Reverse test: ablate followers → measure driver attribution change
    - For Qwen3: drivers=[L35_H8,L31_H17,L34_H15], followers=[L35_H0,L35_H28]
    - For GLM4/DS7B: auto-identify via ablation first

  Part 2: Infrastructure head deep analysis
    - L34_H15 (Qwen3) attention pattern across diverse prompts
    - OV circuit with real residual stream (not W_E)
    - Decode head output at attended positions
    - Find equivalents for GLM4/DS7B

  Part 3: Probing classifier for subject residual stream (HIGHEST PRIORITY)
    - 60+ singular + 60+ plural subject prompts
    - Extract subject hidden state at all layers
    - Train linear probe per layer (LogisticRegression)
    - Per-layer accuracy → grammar information accumulation curve
    - Probe weight vector = "number direction" in residual stream

  Part 4: Q/K alignment with probe number direction
    - cos(number_direction, grammar_head_query_at_verb)
    - cos(number_direction, grammar_head_key_at_subject)
    - Test: is the grammar head actively searching for number information?

Usage:
  python tests/glm5/phase259_two_stage_and_probing.py --model qwen3 --part 1
  python tests/glm5/phase259_two_stage_and_probing.py --model qwen3 --part all
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase259_two_stage_probing")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Utility Functions
# ============================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_model_safe(model_name):
    """Load model with bfloat16 + device_map=auto + flash attention"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} from {cfg['path']}...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
            log_time(f"Loaded with attn_implementation={attn_impl}")
            break
        except Exception as e:
            log_time(f"  {attn_impl} failed: {str(e)[:80]}, trying next...")
            continue

    model.eval()
    from model_utils import get_model_info
    info = get_model_info(model, model_name)
    log_time(f"{model_name}: class={info.model_class}, layers={info.n_layers}, "
             f"d_model={info.d_model}, vocab={info.vocab_size}")

    config = model.config
    n_heads = getattr(config, 'num_attention_heads', 32)
    head_dim = getattr(config, 'head_dim', info.d_model // n_heads)
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    kv_group_size = n_heads // n_kv_heads  # heads per KV group

    log_time(f"  n_heads={n_heads}, head_dim={head_dim}, n_kv_heads={n_kv_heads}, kv_group_size={kv_group_size}")
    return model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size

def get_W_U_safe(model, model_name):
    from model_utils import get_W_U
    return get_W_U(model, model_name)

def release_model_safe(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released, GPU cleared")

def save_result(model_name, part, data):
    fname = RESULT_DIR / f"{model_name}_part{part}.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
    log_time(f"Results saved to {fname}")

def safe_decode(tokenizer, token_id):
    try:
        r = tokenizer.decode([token_id])
        return r.strip() if r else f"<tok_{token_id}>"
    except:
        return f"<tok_{token_id}>"

def get_input_device(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_weight_to_numpy(weight_tensor, model_name=None, layer_name=None):
    import torch
    if weight_tensor.is_meta:
        from model_utils import MODEL_CONFIGS
        import glob, os
        from safetensors import safe_open
        model_path = MODEL_CONFIGS.get(model_name, {}).get("path", None)
        if model_path and layer_name:
            sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
            for sf_file in sf_files:
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    if layer_name in sf.keys():
                        w = sf.get_tensor(layer_name)
                        return w.float().numpy()
        raise ValueError(f"Cannot load meta tensor {layer_name} for {model_name}")
    return weight_tensor.detach().cpu().float().numpy()

def find_word_positions_by_decode(prompt_text, words, tok):
    """Find token positions by decoding each token and matching word substrings."""
    prompt_toks = tok.encode(prompt_text, add_special_tokens=False)
    positions = []
    for word in words:
        word_lower = word.lower()
        char_start = prompt_text.find(word)
        if char_start < 0:
            char_start = prompt_text.lower().find(word_lower)
        if char_start < 0:
            continue
        char_end = char_start + len(word)
        cum_chars = 0
        for ti in range(len(prompt_toks)):
            decoded = tok.decode(prompt_toks[:ti+1])
            tok_char_end = len(decoded)
            tok_char_start = len(tok.decode(prompt_toks[:ti])) if ti > 0 else 0
            if tok_char_start < char_end and tok_char_end > char_start:
                positions.append(ti)
    return sorted(set(positions))

def get_kv_head_idx(query_head, kv_group_size):
    """Get the KV head index for a given query head (handles GQA)."""
    return query_head // kv_group_size

def get_W_V_h(W_V, query_head, head_dim, kv_group_size):
    """Extract W_V for a specific query head, handling GQA."""
    kv_h = get_kv_head_idx(query_head, kv_group_size)
    return W_V[kv_h * head_dim:(kv_h + 1) * head_dim, :]  # [head_dim, d_model]

def get_W_K_h(W_K, query_head, head_dim, kv_group_size):
    """Extract W_K for a specific query head, handling GQA."""
    kv_h = get_kv_head_idx(query_head, kv_group_size)
    return W_K[kv_h * head_dim:(kv_h + 1) * head_dim, :]  # [head_dim, d_model]

def get_grammar_heads_for_model(model_name, n_layers, n_heads):
    """Load grammar heads from Phase 257 results, with model-specific defaults."""
    p257_dir = Path("results/phase257_grammar_geometry")
    part1_path = p257_dir / f"{model_name}_part1.json"
    grammar_heads = []

    if part1_path.exists():
        with open(part1_path, 'r', encoding='utf-8') as f:
            p257_data = json.load(f)
        for label, count in p257_data.get("consistent_grammar_heads", []):
            parts = label.replace("L", "").split("_H")
            if len(parts) == 2:
                grammar_heads.append((int(parts[0]), int(parts[1])))
    else:
        # Fallback: last 1/3 layers
        for li in range(n_layers * 2 // 3, n_layers):
            for h in range(min(8, n_heads)):
                grammar_heads.append((li, h))

    return grammar_heads

# Model-specific driver/follower heads (from Phase 258 ablation)
MODEL_HEAD_ROLES = {
    "qwen3": {
        "drivers": [(35, 8), (31, 17), (34, 15)],
        "followers": [(35, 0), (35, 28)],
        "infrastructure": (34, 15),
    },
    "glm4": {
        "drivers": None,   # Will auto-identify
        "followers": None,
        "infrastructure": None,
    },
    "deepseek7b": {
        "drivers": None,
        "followers": None,
        "infrastructure": None,
    },
}

# ============================================================
# Grammar Prompts for Probing Classifier
# ============================================================

# Singular subject prompts: (prompt, subject_word)
SINGULAR_PROMPTS = [
    # Simple nouns with "The"
    ("The cat", "cat"), ("The dog", "dog"), ("The bird", "bird"),
    ("The fish", "fish"), ("The tree", "tree"), ("The man", "man"),
    ("The woman", "woman"), ("The girl", "girl"), ("The boy", "boy"),
    ("The child", "child"), ("The house", "house"), ("The book", "book"),
    ("The car", "car"), ("The train", "train"), ("The room", "room"),
    ("The sun", "sun"), ("The moon", "moon"), ("The star", "star"),
    ("The river", "river"), ("The mountain", "mountain"),
    # Pronouns
    ("She", "She"), ("He", "He"), ("It", "It"),
    # With adjectives
    ("The big cat", "cat"), ("The small dog", "dog"), ("The red bird", "bird"),
    ("The old man", "man"), ("The young girl", "girl"), ("The tall tree", "tree"),
    ("The new book", "book"), ("The fast car", "car"), ("The slow train", "train"),
    ("The dark room", "room"),
    # Demonstratives
    ("This cat", "cat"), ("That dog", "dog"), ("This bird", "bird"),
    ("That tree", "tree"), ("This man", "man"), ("That woman", "woman"),
    # Possessives
    ("My cat", "cat"), ("His dog", "dog"), ("Her bird", "bird"),
    ("My friend", "friend"), ("His mother", "mother"), ("Her father", "father"),
    # Every/Each
    ("Every cat", "cat"), ("Each dog", "dog"), ("Every child", "child"),
    ("Each student", "student"), ("Every teacher", "teacher"),
    # Uncountable (treated as singular)
    ("Water", "Water"), ("Fire", "Fire"), ("Love", "Love"),
    ("Music", "Music"), ("The water", "water"), ("The fire", "fire"),
    ("The food", "food"), ("The bread", "bread"),
    # More simple nouns
    ("The teacher", "teacher"), ("The doctor", "doctor"),
    ("The student", "student"), ("The worker", "worker"),
    ("The player", "player"), ("The singer", "singer"),
    ("The farmer", "farmer"), ("The dancer", "dancer"),
    ("The writer", "writer"), ("The leader", "leader"),
    # Longer context
    ("The cat on the mat", "cat"), ("The dog in the park", "dog"),
    ("She always", "She"), ("He never", "He"), ("It usually", "It"),
    ("The man with the hat", "man"), ("The girl by the door", "girl"),
]

PLURAL_PROMPTS = [
    # Simple nouns with "The"
    ("The cats", "cats"), ("The dogs", "dogs"), ("The birds", "birds"),
    ("The fish", "fish"), ("The trees", "trees"), ("The men", "men"),
    ("The women", "women"), ("The girls", "girls"), ("The boys", "boys"),
    ("The children", "children"), ("The houses", "houses"), ("The books", "books"),
    ("The cars", "cars"), ("The trains", "trains"), ("The rooms", "rooms"),
    ("The suns", "suns"), ("The moons", "moons"), ("The stars", "stars"),
    ("The rivers", "rivers"), ("The mountains", "mountains"),
    # Pronouns
    ("They", "They"), ("We", "We"),
    # With adjectives
    ("The big cats", "cats"), ("The small dogs", "dogs"), ("The red birds", "birds"),
    ("The old men", "men"), ("The young girls", "girls"), ("The tall trees", "trees"),
    ("The new books", "books"), ("The fast cars", "cars"), ("The slow trains", "trains"),
    ("The dark rooms", "rooms"),
    # Demonstratives
    ("These cats", "cats"), ("Those dogs", "dogs"), ("These birds", "birds"),
    ("Those trees", "trees"), ("These men", "men"), ("Those women", "women"),
    # Possessives
    ("My cats", "cats"), ("His dogs", "dogs"), ("Her birds", "birds"),
    ("My friends", "friends"), ("His parents", "parents"), ("Her sisters", "sisters"),
    # Quantifiers
    ("Some people", "people"), ("Many children", "children"),
    ("Few students", "students"), ("Both men", "men"),
    ("Several birds", "birds"), ("Various trees", "trees"),
    # Number words
    ("Two cats", "cats"), ("Three dogs", "dogs"), ("Five birds", "birds"),
    ("Ten trees", "trees"), ("Six men", "men"),
    # More nouns
    ("The teachers", "teachers"), ("The doctors", "doctors"),
    ("The students", "students"), ("The workers", "workers"),
    ("The players", "players"), ("The singers", "singers"),
    ("The farmers", "farmers"), ("The dancers", "dancers"),
    ("The writers", "writers"), ("The leaders", "leaders"),
    # Longer context
    ("The cats on the mat", "cats"), ("The dogs in the park", "dogs"),
    ("They always", "They"), ("We never", "We"),
    ("The men with the hats", "men"), ("The girls by the doors", "girls"),
]


# ============================================================
# Part 1: Two-Stage Computation Verification
# ============================================================

def part1_two_stage(model_name):
    """
    Test the two-stage hypothesis: driver heads compute grammar first,
    then follower heads read the results.

    Method:
    1. Compute baseline attribution for all grammar heads
    2. Ablate driver heads → measure follower attribution change
    3. Ablate follower heads → measure driver attribution change (reverse)
    4. If follower attribution drops after driver ablation → two-stage confirmed
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)

    results = {"model": model_name, "n_layers": info.n_layers, "n_heads": n_heads}

    # Load grammar heads
    grammar_heads = get_grammar_heads_for_model(model_name, info.n_layers, n_heads)
    top_heads = grammar_heads[:15]  # Top 15 grammar heads
    log_time(f"Grammar heads for two-stage test: {[(f'L{l}_H{h}') for l,h in top_heads]}")

    # Get driver/follower roles
    roles = MODEL_HEAD_ROLES.get(model_name, {})
    drivers = roles.get("drivers")
    followers = roles.get("followers")

    # If roles not defined, auto-identify via ablation
    if drivers is None or followers is None:
        log_time("Auto-identifying driver/follower heads via ablation...")
        drivers, followers = auto_identify_roles(model_name, model, tokenizer, info, n_heads, head_dim, kv_group_size, W_U, layers, input_device, top_heads)
        results["auto_identified_roles"] = {
            "drivers": [(f"L{l}_H{h}") for l,h in drivers],
            "followers": [(f"L{l}_H{h}") for l,h in followers],
        }

    log_time(f"Driver heads: {[(f'L{l}_H{h}') for l,h in drivers]}")
    log_time(f"Follower heads: {[(f'L{l}_H{h}') for l,h in followers]}")

    results["driver_heads"] = [(f"L{l}_H{h}") for l,h in drivers]
    results["follower_heads"] = [(f"L{l}_H{h}") for l,h in followers]

    # Test prompts for attribution measurement
    test_prompts = [
        ("The cat", " sits", " sit"),
        ("The dogs", " run", " runs"),
        ("She", " walks", " walk"),
        ("They", " eat", " eats"),
        ("The girl that the boys like", " walks", " walk"),
        ("The cats that the dog chases", " sit", " sits"),
        ("The man", " works", " work"),
        ("The women", " speak", " speaks"),
    ]

    # ---- Step 1: Compute baseline attribution for all heads ----
    log_time("\n=== Step 1: Baseline attribution ===")
    baseline_attribution = defaultdict(list)  # head_label -> [attribution across prompts]

    for pi, (prefix, correct_verb, wrong_verb) in enumerate(test_prompts):
        correct_ids = tokenizer.encode(correct_verb, add_special_tokens=False)
        wrong_ids = tokenizer.encode(wrong_verb, add_special_tokens=False)
        if not correct_ids or not wrong_ids:
            continue
        correct_id = correct_ids[0]
        wrong_id = wrong_ids[0]
        if correct_id >= W_U.shape[0] or wrong_id >= W_U.shape[0]:
            continue

        target_dir = W_U[correct_id]
        competitor_dir = W_U[wrong_id]
        diff_dir = target_dir - competitor_dir

        # Capture head outputs
        head_outputs = {}

        def make_capture_hook(capture_dict, li, h):
            def hook(module, input, output):
                inp = input[0]
                batch, seq, _ = inp.shape
                head_outs = inp.view(batch, seq, n_heads, head_dim)
                capture_dict[f"L{li}_H{h}"] = head_outs[0, -1, h, :].detach().float().cpu().numpy()
            return hook

        inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        hooks = []
        hooked = set()
        for li, h in top_heads:
            key = (li, h)
            if key not in hooked:
                hooks.append(layers[li].self_attn.o_proj.register_forward_hook(
                    make_capture_hook(head_outputs, li, h)))
                hooked.add(key)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask)

        for hk in hooks:
            hk.remove()

        # Compute attribution for each head
        for li, h in top_heads:
            h_label = f"L{li}_H{h}"
            if h_label not in head_outputs:
                continue
            h_out = head_outputs[h_label]

            w_o = layers[li].self_attn.o_proj.weight
            W_O = safe_weight_to_numpy(w_o, model_name, f"model.layers.{li}.self_attn.o_proj.weight")
            W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]

            attribution = float(diff_dir @ W_O_h @ h_out)
            baseline_attribution[h_label].append(attribution)

        del head_outputs, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (pi + 1) % 4 == 0:
            log_time(f"  Baseline: {pi+1}/{len(test_prompts)} prompts done")

    # Compute mean baseline attribution
    mean_baseline = {}
    for h_label, attrs in baseline_attribution.items():
        mean_baseline[h_label] = float(np.mean(attrs))

    log_time("\nBaseline attribution (mean across prompts):")
    for h_label in sorted(mean_baseline.keys(), key=lambda x: abs(mean_baseline[x]), reverse=True)[:10]:
        log_time(f"  {h_label}: {mean_baseline[h_label]:.4f}")

    # ---- Step 2: Ablate drivers → measure follower attribution ----
    log_time("\n=== Step 2: Ablate drivers → measure follower attribution ===")

    # Get driver head contributions to subtract
    driver_deltas = {}  # head_label -> delta_vector (d_model,)
    for li, h in drivers:
        h_label = f"L{li}_H{h}"
        # We need the W_O_h and head output for each prompt
        # Store per-prompt deltas
        pass

    follower_attribution_after_driver_ablation = defaultdict(list)

    for pi, (prefix, correct_verb, wrong_verb) in enumerate(test_prompts):
        correct_ids = tokenizer.encode(correct_verb, add_special_tokens=False)
        wrong_ids = tokenizer.encode(wrong_verb, add_special_tokens=False)
        if not correct_ids or not wrong_ids:
            continue
        correct_id = correct_ids[0]
        wrong_id = wrong_ids[0]
        if correct_id >= W_U.shape[0] or wrong_id >= W_U.shape[0]:
            continue

        target_dir = W_U[correct_id]
        competitor_dir = W_U[wrong_id]
        diff_dir = target_dir - competitor_dir

        inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        # First, capture driver head outputs for this prompt
        driver_head_outs = {}

        def make_driver_capture_hook(capture_dict, li, h):
            def hook(module, input, output):
                inp = input[0]
                batch, seq, _ = inp.shape
                head_outs = inp.view(batch, seq, n_heads, head_dim)
                capture_dict[f"L{li}_H{h}"] = head_outs[0, -1, h, :].detach().float().cpu()
            return hook

        hooks_cap = []
        for li, h in drivers:
            hooks_cap.append(layers[li].self_attn.o_proj.register_forward_hook(
                make_driver_capture_hook(driver_head_outs, li, h)))

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attn_mask)

        for hk in hooks_cap:
            hk.remove()

        # Compute driver deltas
        driver_deltas_this = []
        for li, h in drivers:
            h_label = f"L{li}_H{h}"
            if h_label not in driver_head_outs:
                continue
            h_out = driver_head_outs[h_label]
            w_o = layers[li].self_attn.o_proj.weight
            W_O = safe_weight_to_numpy(w_o, model_name, f"model.layers.{li}.self_attn.o_proj.weight")
            W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]
            delta = W_O_h @ h_out.numpy()  # [d_model]
            driver_deltas_this.append((li, delta))

        # Now ablate all drivers simultaneously and measure follower attribution
        ablation_applied = [False]
        follower_head_outputs = {}

        def make_follower_capture_hook(capture_dict, li, h):
            def hook(module, input, output):
                inp = input[0]
                batch, seq, _ = inp.shape
                head_outs = inp.view(batch, seq, n_heads, head_dim)
                capture_dict[f"L{li}_H{h}"] = head_outs[0, -1, h, :].detach().float().cpu().numpy()
            return hook

        # Create ablation hooks for each driver head
        def make_driver_ablation_hook(delta, layer_idx):
            applied = [False]
            def hook(module, input, output):
                if applied[0]:
                    return output
                if isinstance(output, tuple):
                    hidden = output[0].clone()
                    hidden[0, -1, :] -= torch.tensor(delta, dtype=hidden.dtype, device=hidden.device)
                    applied[0] = True
                    return (hidden,) + output[1:]
                return output
            return hook

        hooks_abl = []
        # Add ablation hooks for drivers
        for li, delta in driver_deltas_this:
            hooks_abl.append(layers[li].self_attn.register_forward_hook(
                make_driver_ablation_hook(delta, li)))
        # Add capture hooks for followers
        for li, h in followers:
            hooks_abl.append(layers[li].self_attn.o_proj.register_forward_hook(
                make_follower_capture_hook(follower_head_outputs, li, h)))

        with torch.no_grad():
            ablated_out = model(input_ids=input_ids, attention_mask=attn_mask)

        for hk in hooks_abl:
            hk.remove()

        # Compute follower attribution after driver ablation
        for li, h in followers:
            h_label = f"L{li}_H{h}"
            if h_label not in follower_head_outputs:
                continue
            h_out = follower_head_outputs[h_label]
            w_o = layers[li].self_attn.o_proj.weight
            W_O = safe_weight_to_numpy(w_o, model_name, f"model.layers.{li}.self_attn.o_proj.weight")
            W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]
            attribution = float(diff_dir @ W_O_h @ h_out)
            follower_attribution_after_driver_ablation[h_label].append(attribution)

        del driver_head_outs, follower_head_outputs, ablated_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (pi + 1) % 4 == 0:
            log_time(f"  Driver ablation: {pi+1}/{len(test_prompts)} prompts done")

    # ---- Step 3: Ablate followers → measure driver attribution ----
    log_time("\n=== Step 3: Ablate followers → measure driver attribution ===")

    driver_attribution_after_follower_ablation = defaultdict(list)

    for pi, (prefix, correct_verb, wrong_verb) in enumerate(test_prompts):
        correct_ids = tokenizer.encode(correct_verb, add_special_tokens=False)
        wrong_ids = tokenizer.encode(wrong_verb, add_special_tokens=False)
        if not correct_ids or not wrong_ids:
            continue
        correct_id = correct_ids[0]
        wrong_id = wrong_ids[0]
        if correct_id >= W_U.shape[0] or wrong_id >= W_U.shape[0]:
            continue

        target_dir = W_U[correct_id]
        competitor_dir = W_U[wrong_id]
        diff_dir = target_dir - competitor_dir

        inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        # Capture follower head outputs
        follower_head_outs = {}

        def make_fcap_hook(capture_dict, li, h):
            def hook(module, input, output):
                inp = input[0]
                batch, seq, _ = inp.shape
                head_outs = inp.view(batch, seq, n_heads, head_dim)
                capture_dict[f"L{li}_H{h}"] = head_outs[0, -1, h, :].detach().float().cpu()
            return hook

        hooks_fcap = []
        for li, h in followers:
            hooks_fcap.append(layers[li].self_attn.o_proj.register_forward_hook(
                make_fcap_hook(follower_head_outs, li, h)))

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attn_mask)

        for hk in hooks_fcap:
            hk.remove()

        # Compute follower deltas
        follower_deltas_this = []
        for li, h in followers:
            h_label = f"L{li}_H{h}"
            if h_label not in follower_head_outs:
                continue
            h_out = follower_head_outs[h_label]
            w_o = layers[li].self_attn.o_proj.weight
            W_O = safe_weight_to_numpy(w_o, model_name, f"model.layers.{li}.self_attn.o_proj.weight")
            W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]
            delta = W_O_h @ h_out.numpy()
            follower_deltas_this.append((li, delta))

        # Ablate followers and measure driver attribution
        driver_head_outputs_after = {}

        def make_dcap_hook(capture_dict, li, h):
            def hook(module, input, output):
                inp = input[0]
                batch, seq, _ = inp.shape
                head_outs = inp.view(batch, seq, n_heads, head_dim)
                capture_dict[f"L{li}_H{h}"] = head_outs[0, -1, h, :].detach().float().cpu().numpy()
            return hook

        def make_follower_ablation_hook(delta, layer_idx):
            applied = [False]
            def hook(module, input, output):
                if applied[0]:
                    return output
                if isinstance(output, tuple):
                    hidden = output[0].clone()
                    hidden[0, -1, :] -= torch.tensor(delta, dtype=hidden.dtype, device=hidden.device)
                    applied[0] = True
                    return (hidden,) + output[1:]
                return output
            return hook

        hooks_fabl = []
        for li, delta in follower_deltas_this:
            hooks_fabl.append(layers[li].self_attn.register_forward_hook(
                make_follower_ablation_hook(delta, li)))
        for li, h in drivers:
            hooks_fabl.append(layers[li].self_attn.o_proj.register_forward_hook(
                make_dcap_hook(driver_head_outputs_after, li, h)))

        with torch.no_grad():
            ablated_out = model(input_ids=input_ids, attention_mask=attn_mask)

        for hk in hooks_fabl:
            hk.remove()

        for li, h in drivers:
            h_label = f"L{li}_H{h}"
            if h_label not in driver_head_outputs_after:
                continue
            h_out = driver_head_outputs_after[h_label]
            w_o = layers[li].self_attn.o_proj.weight
            W_O = safe_weight_to_numpy(w_o, model_name, f"model.layers.{li}.self_attn.o_proj.weight")
            W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]
            attribution = float(diff_dir @ W_O_h @ h_out)
            driver_attribution_after_follower_ablation[h_label].append(attribution)

        del follower_head_outs, driver_head_outputs_after, ablated_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Summary ----
    log_time("\n=== Two-Stage Hypothesis Results ===")

    summary = {}

    # Follower attribution: baseline vs after driver ablation
    for li, h in followers:
        h_label = f"L{li}_H{h}"
        base = mean_baseline.get(h_label, 0.0)
        after_driver = float(np.mean(follower_attribution_after_driver_ablation.get(h_label, [0.0])))
        change = after_driver - base
        summary[h_label] = {
            "role": "follower",
            "baseline_attribution": round(base, 4),
            "after_driver_ablation": round(after_driver, 4),
            "change": round(change, 4),
            "change_pct": round(change / max(abs(base), 1e-6) * 100, 1),
        }
        log_time(f"  Follower {h_label}: baseline={base:.4f}, after_driver_ablation={after_driver:.4f}, "
                 f"change={change:.4f} ({change/max(abs(base),1e-6)*100:.1f}%)")

    # Driver attribution: baseline vs after follower ablation
    for li, h in drivers:
        h_label = f"L{li}_H{h}"
        base = mean_baseline.get(h_label, 0.0)
        after_follower = float(np.mean(driver_attribution_after_follower_ablation.get(h_label, [0.0])))
        change = after_follower - base
        summary[h_label] = {
            "role": "driver",
            "baseline_attribution": round(base, 4),
            "after_follower_ablation": round(after_follower, 4),
            "change": round(change, 4),
            "change_pct": round(change / max(abs(base), 1e-6) * 100, 1),
        }
        log_time(f"  Driver {h_label}: baseline={base:.4f}, after_follower_ablation={after_follower:.4f}, "
                 f"change={change:.4f} ({change/max(abs(base),1e-6)*100:.1f}%)")

    results["two_stage_summary"] = summary

    save_result(model_name, 1, results)
    release_model_safe(model)
    return results


def auto_identify_roles(model_name, model, tokenizer, info, n_heads, head_dim, kv_group_size, W_U, layers, input_device, grammar_heads):
    """Auto-identify driver/follower heads via ablation test."""
    log_time("Running ablation to identify driver/follower heads...")

    test_prompts = [
        ("The cat", " sits", " sit"),
        ("The dogs", " run", " runs"),
        ("She", " walks", " walk"),
        ("They", " eat", " eats"),
        ("The girl that the boys like", " walks", " walk"),
        ("The cats that the dog chases", " sit", " sits"),
    ]

    import torch
    head_ablation_effects = defaultdict(list)

    for prefix, correct_verb, wrong_verb in test_prompts:
        correct_ids = tokenizer.encode(correct_verb, add_special_tokens=False)
        wrong_ids = tokenizer.encode(wrong_verb, add_special_tokens=False)
        if not correct_ids or not wrong_ids:
            continue
        correct_id, wrong_id = correct_ids[0], wrong_ids[0]

        inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        with torch.no_grad():
            base_out = model(input_ids=input_ids, attention_mask=attn_mask)
        base_logits = base_out.logits[0, -1].float().cpu().numpy()
        base_logit_diff = float(base_logits[correct_id] - base_logits[wrong_id])

        for li, h in grammar_heads[:15]:
            # Capture head output
            head_output = {}
            def make_cap_hook(cd, l, hh):
                def hook(module, input, output):
                    inp = input[0]
                    batch, seq, _ = inp.shape
                    ho = inp.view(batch, seq, n_heads, head_dim)
                    cd["out"] = ho[0, -1, hh, :].detach().float().cpu()
                return hook

            h_cap = layers[li].self_attn.o_proj.register_forward_hook(make_cap_hook(head_output, li, h))
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attn_mask)
            h_cap.remove()

            if "out" not in head_output:
                continue

            h_out = head_output["out"]
            w_o = layers[li].self_attn.o_proj.weight
            W_O = safe_weight_to_numpy(w_o, model_name, f"model.layers.{li}.self_attn.o_proj.weight")
            W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]
            delta = W_O_h @ h_out.numpy()

            # Ablate
            applied = [False]
            def make_abl_hook(d):
                a = [False]
                def hook(module, input, output):
                    if a[0]: return output
                    if isinstance(output, tuple):
                        hidden = output[0].clone()
                        hidden[0, -1, :] -= torch.tensor(d, dtype=hidden.dtype, device=hidden.device)
                        a[0] = True
                        return (hidden,) + output[1:]
                    return output
                return hook

            h_abl = layers[li].self_attn.register_forward_hook(make_abl_hook(delta))
            with torch.no_grad():
                abl_out = model(input_ids=input_ids, attention_mask=attn_mask)
            h_abl.remove()

            abl_logits = abl_out.logits[0, -1].float().cpu().numpy()
            abl_logit_diff = float(abl_logits[correct_id] - abl_logits[wrong_id])
            change = abl_logit_diff - base_logit_diff
            head_ablation_effects[f"L{li}_H{h}"].append(change)

            del head_output, abl_out
            gc.collect()

        del base_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Classify heads
    drivers = []
    followers = []
    for h_label, changes in head_ablation_effects.items():
        mean_change = np.mean(changes)
        n_hurt = sum(1 for c in changes if c < -0.01)
        if n_hurt >= len(test_prompts) * 0.5 and mean_change < -0.02:
            drivers.append(h_label)
        else:
            followers.append(h_label)

    # Convert to (layer, head) tuples
    driver_tuples = []
    follower_tuples = []
    for h_label in drivers[:5]:
        parts = h_label.replace("L", "").split("_H")
        driver_tuples.append((int(parts[0]), int(parts[1])))
    for h_label in followers[:5]:
        parts = h_label.replace("L", "").split("_H")
        follower_tuples.append((int(parts[0]), int(parts[1])))

    log_time(f"  Auto-identified drivers: {drivers[:5]}")
    log_time(f"  Auto-identified followers: {followers[:5]}")

    return driver_tuples, follower_tuples


# ============================================================
# Part 2: Infrastructure Head Deep Analysis
# ============================================================

def part2_infrastructure_head(model_name):
    """
    Deep analysis of the infrastructure head (L34_H15 for Qwen3).
    - Attention pattern across diverse prompts
    - OV circuit with real residual stream context
    - Decode head output at attended positions
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)

    results = {"model": model_name, "n_layers": info.n_layers}

    # Get infrastructure head
    roles = MODEL_HEAD_ROLES.get(model_name, {})
    infra_head = roles.get("infrastructure")

    if infra_head is None:
        # Find the head with highest % hurt in ablation (from Phase 258 or auto)
        # For now, use a heuristic: find head in late layers with consistent attention
        if model_name == "glm4":
            infra_head = (38, 28)  # Top grammar head with subject attention
        elif model_name == "deepseek7b":
            infra_head = (27, 7)   # Top grammar head
        else:
            infra_head = (34, 15)  # Default Qwen3

    li, h = infra_head
    h_label = f"L{li}_H{h}"
    log_time(f"Infrastructure head: {h_label}")

    results["infrastructure_head"] = h_label

    # Diverse prompts for analysis
    analysis_prompts = [
        ("The cat", "cat"),
        ("The cats", "cats"),
        ("The dog", "dog"),
        ("The dogs", "dogs"),
        ("She", "She"),
        ("They", "They"),
        ("The girl that the boys like", "girl"),
        ("The cats that the dog chases", "cats"),
        ("The man with the hat", "man"),
        ("The women in the room", "women"),
        ("Yesterday, she", "she"),
        ("Last night, he", "he"),
        ("She is taller", "taller"),
        ("The teacher", "teacher"),
        ("The students", "students"),
    ]

    # ---- A: Attention pattern analysis ----
    log_time("\n=== A: Attention pattern of infrastructure head ===")
    attn_analysis = []

    for pi, (prompt, subj_word) in enumerate(analysis_prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        # Calculate special token offset
        actual_seq_len = input_ids.shape[1]
        special_token_offset = actual_seq_len - len(prompt_tokens)
        verb_pos = len(prompt_tokens) - 1 + special_token_offset

        # Find subject position (add offset for special tokens)
        subj_positions_raw = find_word_positions_by_decode(prompt, [subj_word], tokenizer)
        subj_positions = [p + special_token_offset for p in subj_positions_raw]

        # Capture attention weights
        captured_attn = {}

        def make_attn_hook(cd):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    aw = output[1]
                    if aw is not None:
                        cd["attn"] = aw.detach().float().cpu()
            return hook

        hook_a = layers[li].self_attn.register_forward_hook(make_attn_hook(captured_attn))

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)

        hook_a.remove()

        if "attn" not in captured_attn:
            log_time(f"  Prompt {pi}: no attention captured, skipping")
            continue

        attn_w = captured_attn["attn"]  # [batch, n_heads, seq_q, seq_k]
        if attn_w.dim() != 4:
            continue

        # Attention from verb position for this head
        head_attn = attn_w[0, h, verb_pos, :].numpy()

        # Top-5 attended positions
        top5_pos = np.argsort(head_attn)[-5:][::-1]
        top5_info = []
        all_tokens_with_special = input_ids[0].cpu().tolist()
        for pos in top5_pos:
            tok_str = safe_decode(tokenizer, int(all_tokens_with_special[pos])) if pos < len(all_tokens_with_special) else "<pad>"
            is_subj = pos in subj_positions
            top5_info.append({
                "position": int(pos),
                "token": tok_str,
                "attn_weight": round(float(head_attn[pos]), 4),
                "is_subject": is_subj,
            })

        subj_attn = float(np.mean([head_attn[p] for p in subj_positions])) if subj_positions else 0.0
        first_attn = float(head_attn[special_token_offset]) if len(head_attn) > special_token_offset else 0.0

        attn_analysis.append({
            "prompt": prompt,
            "subject_positions": subj_positions,
            "verb_position": verb_pos,
            "subject_attn": round(subj_attn, 4),
            "first_token_attn": round(first_attn, 4),
            "top5_attended": top5_info,
        })

        log_time(f"  Prompt '{prompt}': subj_attn={subj_attn:.4f}, first_attn={first_attn:.4f}, "
                 f"top1={top5_info[0]['token']}({top5_info[0]['attn_weight']:.3f})")

        del captured_attn, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["attention_analysis"] = attn_analysis

    # Summarize attention patterns
    subj_attns = [a["subject_attn"] for a in attn_analysis if a["subject_attn"] > 0]
    first_attns = [a["first_token_attn"] for a in attn_analysis]
    top1_is_subj = sum(1 for a in attn_analysis
                       if any(t["is_subject"] for t in a["top5_attended"][:1]))
    top3_has_subj = sum(1 for a in attn_analysis
                        if any(t["is_subject"] for t in a["top5_attended"][:3]))

    results["attention_summary"] = {
        "mean_subject_attn": round(float(np.mean(subj_attns)), 4) if subj_attns else 0.0,
        "mean_first_token_attn": round(float(np.mean(first_attns)), 4),
        "top1_is_subject_count": f"{top1_is_subj}/{len(attn_analysis)}",
        "top3_has_subject_count": f"{top3_has_subj}/{len(attn_analysis)}",
    }

    log_time(f"\nAttention summary: mean_subj_attn={results['attention_summary']['mean_subject_attn']:.4f}, "
             f"top1_is_subj={results['attention_summary']['top1_is_subject_count']}, "
             f"top3_has_subj={results['attention_summary']['top3_has_subject_count']}")

    # ---- B: OV circuit with real residual stream ----
    log_time("\n=== B: OV circuit with real residual stream ===")

    ov_analysis = []

    for pi, (prompt, subj_word) in enumerate(analysis_prompts[:8]):  # Limit to 8 prompts
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        # Calculate special token offset
        actual_seq_len = input_ids.shape[1]
        special_token_offset = actual_seq_len - len(prompt_tokens)
        verb_pos = len(prompt_tokens) - 1 + special_token_offset
        subj_positions_raw = find_word_positions_by_decode(prompt, [subj_word], tokenizer)
        subj_positions = [p + special_token_offset for p in subj_positions_raw]
        all_tokens_with_special = input_ids[0].cpu().tolist()

        # Capture attention weights and residual stream
        captured_attn = {}
        captured_resid = {}

        def make_attn_hook2(cd):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    aw = output[1]
                    if aw is not None:
                        cd["attn"] = aw.detach().float().cpu()
            return hook

        def make_resid_hook(cd, layer_idx):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    cd["resid"] = input[0].detach().float().cpu().numpy()
            return hook

        hook_a = layers[li].self_attn.register_forward_hook(make_attn_hook2(captured_attn))
        hook_r = layers[li].register_forward_hook(make_resid_hook(captured_resid, li))

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)

        hook_a.remove()
        hook_r.remove()

        if "attn" not in captured_attn or "resid" not in captured_resid:
            del captured_attn, captured_resid, out
            continue

        attn_w = captured_attn["attn"]
        resid = captured_resid["resid"]  # [1, seq, d_model]

        if attn_w.dim() != 4:
            del captured_attn, captured_resid, out
            continue

        head_attn = attn_w[0, h, verb_pos, :].numpy()

        # Top attended position
        top_pos = int(np.argmax(head_attn))

        # Get hidden state at top attended position
        attended_hidden = resid[0, top_pos, :]  # [d_model]

        # OV circuit: W_O_h @ W_V_h @ attended_hidden
        w_o = layers[li].self_attn.o_proj.weight
        w_v = layers[li].self_attn.v_proj.weight
        W_O = safe_weight_to_numpy(w_o, model_name, f"model.layers.{li}.self_attn.o_proj.weight")
        W_V = safe_weight_to_numpy(w_v, model_name, f"model.layers.{li}.self_attn.v_proj.weight")

        W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]  # [d_model, head_dim]
        W_V_h = get_W_V_h(W_V, h, head_dim, kv_group_size)  # [head_dim, d_model]

        # Head output = W_O_h @ (W_V_h @ attended_hidden)
        head_out = W_O_h @ (W_V_h @ attended_hidden)  # [d_model]

        # Decode with W_U
        logit_effect = W_U @ head_out  # [vocab_size]
        top10_ids = np.argsort(logit_effect)[-10:][::-1]
        top10_tokens = [(safe_decode(tokenizer, int(tid)), round(float(logit_effect[tid]), 2))
                       for tid in top10_ids]

        # Check singular/plural verb logit effect
        sing_verb_ids = [tokenizer.encode(w, add_special_tokens=False)[0]
                        for w in ["sits", "runs", "walks", "eats"] if tokenizer.encode(w, add_special_tokens=False)]
        plur_verb_ids = [tokenizer.encode(w, add_special_tokens=False)[0]
                        for w in ["sit", "run", "walk", "eat"] if tokenizer.encode(w, add_special_tokens=False)]

        sing_effect = [float(logit_effect[vid]) for vid in sing_verb_ids if vid < len(logit_effect)]
        plur_effect = [float(logit_effect[vid]) for vid in plur_verb_ids if vid < len(logit_effect)]

        number_effect = (np.mean(sing_effect) - np.mean(plur_effect)) if (sing_effect and plur_effect) else 0.0

        ov_analysis.append({
            "prompt": prompt,
            "top_attended_pos": top_pos,
            "top_attended_token": safe_decode(tokenizer, int(all_tokens_with_special[top_pos])) if top_pos < len(all_tokens_with_special) else "<pad>",
            "top_attended_attn": round(float(head_attn[top_pos]), 4),
            "is_subject": top_pos in subj_positions,
            "OV_top10_tokens": top10_tokens,
            "OV_number_effect": round(float(number_effect), 4),
            "OV_sing_verb_effect": round(float(np.mean(sing_effect)), 4) if sing_effect else None,
            "OV_plur_verb_effect": round(float(np.mean(plur_effect)), 4) if plur_effect else None,
        })

        log_time(f"  '{prompt}': attend to pos {top_pos} ({safe_decode(tokenizer, int(prompt_tokens[top_pos])) if top_pos < len(prompt_tokens) else '<pad>'}), "
                 f"is_subj={top_pos in subj_positions}, number_effect={number_effect:.4f}, "
                 f"top3={top10_tokens[:3]}")

        del captured_attn, captured_resid, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["ov_analysis"] = ov_analysis

    # OV summary
    subj_ov = [o for o in ov_analysis if o["is_subject"]]
    nonsubj_ov = [o for o in ov_analysis if not o["is_subject"]]

    results["ov_summary"] = {
        "n_prompts_attend_to_subject": f"{len(subj_ov)}/{len(ov_analysis)}",
        "mean_number_effect_when_subj": round(float(np.mean([o["OV_number_effect"] for o in subj_ov])), 4) if subj_ov else None,
        "mean_number_effect_when_nonsubj": round(float(np.mean([o["OV_number_effect"] for o in nonsubj_ov])), 4) if nonsubj_ov else None,
    }

    log_time(f"\nOV summary: attends_to_subj={results['ov_summary']['n_prompts_attend_to_subject']}, "
             f"number_effect(subj)={results['ov_summary']['mean_number_effect_when_subj']}, "
             f"number_effect(nonsubj)={results['ov_summary']['mean_number_effect_when_nonsubj']}")

    save_result(model_name, 2, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 3: Probing Classifier for Subject Residual Stream
# ============================================================

def part3_probing_classifier(model_name):
    """
    Train linear probing classifiers to decode number (singular/plural)
    from subject position's hidden state at each layer.

    This is the highest priority experiment — it directly answers:
    "What grammatical information does the subject's residual stream carry?"
    """
    import torch
    from model_utils import get_layers
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_safe(model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)

    results = {"model": model_name, "n_layers": info.n_layers, "d_model": info.d_model}

    # ---- Collect hidden states at subject position for all layers ----
    log_time(f"Collecting subject hidden states for {len(SINGULAR_PROMPTS)} singular + "
             f"{len(PLURAL_PROMPTS)} plural prompts...")

    # Data structure: {layer_idx: {"singular": [hidden_states], "plural": [hidden_states]}}
    all_hidden = defaultdict(lambda: {"singular": [], "plural": []})
    valid_prompts = {"singular": 0, "plural": 0}
    failed_prompts = []

    for number_label, prompts in [("singular", SINGULAR_PROMPTS), ("plural", PLURAL_PROMPTS)]:
        for pi, (prompt, subj_word) in enumerate(prompts):
            # Find subject position
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            subj_positions = find_word_positions_by_decode(prompt, [subj_word], tokenizer)

            if not subj_positions:
                failed_prompts.append((number_label, prompt, "no subject position"))
                continue

            # Use first subject position
            subj_pos = subj_positions[0]

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            # Calculate special token offset (GLM4=2, DS7B=1, Qwen3=0)
            actual_seq_len = input_ids.shape[1]
            no_special_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            special_token_offset = actual_seq_len - no_special_len
            adjusted_subj_pos = subj_pos + special_token_offset

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
            except Exception as e:
                failed_prompts.append((number_label, prompt, str(e)[:50]))
                continue

            # Extract hidden state at subject position for each layer
            if out.hidden_states:
                for layer_idx, hs in enumerate(out.hidden_states):
                    # hs: [1, seq, d_model]
                    hidden_vec = hs[0, adjusted_subj_pos, :].float().cpu().numpy()
                    all_hidden[layer_idx][number_label].append(hidden_vec)

            valid_prompts[number_label] += 1

            del out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (valid_prompts[number_label]) % 20 == 0:
                log_time(f"  {number_label}: {valid_prompts[number_label]} prompts done")

    log_time(f"Valid prompts: singular={valid_prompts['singular']}, plural={valid_prompts['plural']}")
    log_time(f"Failed prompts: {len(failed_prompts)}")
    if failed_prompts:
        for label, prompt, reason in failed_prompts[:5]:
            log_time(f"  Failed: [{label}] '{prompt}' — {reason}")

    results["n_valid_singular"] = valid_prompts["singular"]
    results["n_valid_plural"] = valid_prompts["plural"]
    results["n_failed"] = len(failed_prompts)

    # ---- Train probing classifier per layer ----
    log_time("\n=== Training linear probe per layer ===")

    layer_probing_results = {}
    n_layers = info.n_layers

    for layer_idx in range(n_layers + 1):  # +1 for embedding layer (layer 0)
        sing_data = all_hidden[layer_idx]["singular"]
        plur_data = all_hidden[layer_idx]["plural"]

        if not sing_data or not plur_data:
            continue

        # Prepare data
        X = np.array(sing_data + plur_data)  # [n_samples, d_model]
        y = np.array([1] * len(sing_data) + [0] * len(plur_data))  # 1=singular, 0=plural

        # Train/test split (80/20)
        n_total = len(y)
        n_train = int(n_total * 0.8)
        indices = np.random.RandomState(42).permutation(n_total)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train logistic regression with L2 regularization
        try:
            probe = LogisticRegression(
                C=1.0, max_iter=1000, solver='lbfgs',
                penalty='l2', random_state=42
            )
            probe.fit(X_train, y_train)

            # Test accuracy
            test_acc = probe.score(X_test, y_test)

            # Cross-validation accuracy (5-fold)
            cv_scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs',
                                  penalty='l2', random_state=42),
                X, y, cv=5
            )
            cv_acc = float(np.mean(cv_scores))

            # Probe weight vector (the "number direction")
            number_direction = probe.coef_[0]  # [d_model]
            number_direction_norm = float(np.linalg.norm(number_direction))

            # Normalize for alignment analysis
            if number_direction_norm > 1e-10:
                number_direction_normalized = number_direction / number_direction_norm
            else:
                number_direction_normalized = number_direction

            layer_probing_results[layer_idx] = {
                "test_accuracy": round(float(test_acc), 4),
                "cv_accuracy": round(cv_acc, 4),
                "cv_std": round(float(np.std(cv_scores)), 4),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "number_direction_norm": round(number_direction_norm, 4),
            }

            # Save normalized number direction for alignment analysis
            layer_probing_results[layer_idx]["number_direction"] = number_direction_normalized.tolist()

            if layer_idx % 5 == 0 or layer_idx >= n_layers - 2:
                log_time(f"  Layer {layer_idx}/{n_layers}: test_acc={test_acc:.3f}, "
                         f"cv_acc={cv_acc:.3f}±{np.std(cv_scores):.3f}, "
                         f"dir_norm={number_direction_norm:.2f}")

        except Exception as e:
            log_time(f"  Layer {layer_idx}: probe training failed — {str(e)[:60]}")
            layer_probing_results[layer_idx] = {"error": str(e)[:100]}

    results["layer_probing"] = layer_probing_results

    # ---- Find key layers ----
    log_time("\n=== Key layer analysis ===")

    valid_layers = {k: v for k, v in layer_probing_results.items() if "cv_accuracy" in v}
    if valid_layers:
        # Peak accuracy layer
        peak_layer = max(valid_layers, key=lambda k: valid_layers[k]["cv_accuracy"])
        peak_acc = valid_layers[peak_layer]["cv_accuracy"]

        # First layer above 90%
        first_90_layer = None
        for li in sorted(valid_layers.keys()):
            if valid_layers[li]["cv_accuracy"] >= 0.90:
                first_90_layer = li
                break

        # First layer above 80%
        first_80_layer = None
        for li in sorted(valid_layers.keys()):
            if valid_layers[li]["cv_accuracy"] >= 0.80:
                first_80_layer = li
                break

        # Layers with accuracy > 95%
        high_acc_layers = sorted([li for li, v in valid_layers.items() if v["cv_accuracy"] > 0.95])

        results["probing_summary"] = {
            "peak_layer": peak_layer,
            "peak_accuracy": round(peak_acc, 4),
            "first_90_layer": first_90_layer,
            "first_80_layer": first_80_layer,
            "high_accuracy_layers_95": high_acc_layers,
        }

        log_time(f"  Peak: Layer {peak_layer} (cv_acc={peak_acc:.3f})")
        log_time(f"  First >90%: Layer {first_90_layer}")
        log_time(f"  First >80%: Layer {first_80_layer}")
        log_time(f"  Layers >95%: {high_acc_layers}")

        # Print accuracy curve
        log_time("\n  Accuracy curve:")
        for li in sorted(valid_layers.keys()):
            acc = valid_layers[li]["cv_accuracy"]
            bar = "█" * int(acc * 40)
            log_time(f"    L{li:2d}: {acc:.3f} {bar}")

    # Save number directions for Part 4 alignment analysis
    directions_file = RESULT_DIR / f"{model_name}_number_directions.npy"
    directions_data = {}
    for li, v in valid_layers.items():
        if "number_direction" in v:
            directions_data[li] = np.array(v["number_direction"])
    if directions_data:
        np.savez(directions_file.with_suffix('.npz'), **{f"L{k}": v for k, v in directions_data.items()})
        log_time(f"Number directions saved to {directions_file.with_suffix('.npz')}")

    save_result(model_name, 3, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 4: Q/K Alignment with Probe Number Direction
# ============================================================

def part4_qk_alignment(model_name):
    """
    Analyze alignment between the probing number direction and
    grammar head Q/K vectors.

    Key tests:
    1. cos(number_direction, grammar_head_query_at_verb) — is the query searching for number?
    2. cos(number_direction, grammar_head_key_at_subject) — does the subject broadcast number?
    3. Compare driver vs follower head alignment
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)

    results = {"model": model_name, "n_layers": info.n_layers}

    # Load number directions from Part 3
    directions_file = RESULT_DIR / f"{model_name}_number_directions.npz"
    if not directions_file.exists():
        log_time("ERROR: Number directions file not found. Run Part 3 first.")
        save_result(model_name, 4, {"error": "number_directions_not_found"})
        release_model_safe(model)
        return results

    directions_data = np.load(directions_file)
    number_directions = {}
    for key in directions_data.files:
        layer_idx = int(key.replace("L", ""))
        number_directions[layer_idx] = directions_data[key]

    log_time(f"Loaded number directions for {len(number_directions)} layers")

    # Get grammar heads and their roles
    grammar_heads = get_grammar_heads_for_model(model_name, info.n_layers, n_heads)
    roles = MODEL_HEAD_ROLES.get(model_name, {})
    drivers = roles.get("drivers", grammar_heads[:3])
    followers = roles.get("followers", grammar_heads[3:5])

    if drivers is None:
        drivers = grammar_heads[:3]
    if followers is None:
        followers = grammar_heads[3:5]

    log_time(f"Driver heads: {[(f'L{l}_H{h}') for l,h in drivers]}")
    log_time(f"Follower heads: {[(f'L{l}_H{h}') for l,h in followers]}")

    # Test prompts
    test_prompts = [
        ("The cat", "cat", " sits", " sit"),
        ("The cats", "cats", " sit", " sits"),
        ("The dog", "dog", " runs", " run"),
        ("The dogs", "dogs", " run", " runs"),
        ("She", "She", " walks", " walk"),
        ("They", "They", " walk", " walks"),
        ("The girl that the boys like", "girl", " walks", " walk"),
        ("The cats that the dog chases", "cats", " sit", " sits"),
    ]

    # For each grammar head, collect Q vectors at verb position and K vectors at subject position
    all_heads = list(set(drivers + followers))
    head_q_at_verb = defaultdict(list)   # head_label -> [q_vectors across prompts]
    head_k_at_subj = defaultdict(list)   # head_label -> [k_vectors across prompts]

    for pi, (prompt, subj_word, correct_verb, wrong_verb) in enumerate(test_prompts):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        subj_positions_raw = find_word_positions_by_decode(prompt, [subj_word], tokenizer)

        if not subj_positions_raw:
            continue

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        # Calculate special token offset
        actual_seq_len = input_ids.shape[1]
        special_token_offset = actual_seq_len - len(prompt_tokens)
        verb_pos = len(prompt_tokens) - 1 + special_token_offset
        subj_pos = subj_positions_raw[0] + special_token_offset

        # Capture residual stream at grammar head layers
        captured_resid = {}

        def make_resid_hook(cd, layer_idx):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    cd[layer_idx] = input[0].detach().float().cpu().numpy()
            return hook

        hooks = []
        hooked_layers = set()
        for li, h in all_heads:
            if li not in hooked_layers:
                hooks.append(layers[li].register_forward_hook(make_resid_hook(captured_resid, li)))
                hooked_layers.add(li)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)

        for hk in hooks:
            hk.remove()

        # Compute Q and K vectors for each grammar head
        for li, h in all_heads:
            h_label = f"L{li}_H{h}"

            # Get W_Q and W_K
            w_q = layers[li].self_attn.q_proj.weight
            w_k = layers[li].self_attn.k_proj.weight
            W_Q = safe_weight_to_numpy(w_q, model_name, f"model.layers.{li}.self_attn.q_proj.weight")
            W_K = safe_weight_to_numpy(w_k, model_name, f"model.layers.{li}.self_attn.k_proj.weight")

            W_Q_h = W_Q[h * head_dim:(h + 1) * head_dim, :]  # [head_dim, d_model]
            W_K_h = get_W_K_h(W_K, h, head_dim, kv_group_size)  # [head_dim, d_model]

            # Get hidden state at this layer
            if li in captured_resid:
                hidden = captured_resid[li][0]  # [seq, d_model]
            elif out.hidden_states and li < len(out.hidden_states):
                hidden = out.hidden_states[li][0].float().cpu().numpy()
            else:
                continue

            # Q at verb position: q = W_Q_h @ hidden[verb_pos]
            q_verb = W_Q_h @ hidden[verb_pos]  # [head_dim]
            head_q_at_verb[h_label].append(q_verb)

            # K at subject position: k = W_K_h @ hidden[subj_pos]
            k_subj = W_K_h @ hidden[subj_pos]  # [head_dim]
            head_k_at_subj[h_label].append(k_subj)

        del captured_resid, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Compute alignment ----
    log_time("\n=== Q/K Alignment with Number Direction ===")

    alignment_results = {}

    for li, h in all_heads:
        h_label = f"L{li}_H{h}"
        role = "driver" if (li, h) in drivers else ("follower" if (li, h) in followers else "other")

        # Get number direction at this layer
        if li not in number_directions:
            log_time(f"  {h_label}: no number direction at layer {li}, skipping")
            continue

        num_dir = number_directions[li]  # [d_model]

        # Q alignment: cos(number_direction, mean_q_at_verb) projected through W_Q_h^T
        # Actually, number_direction is in d_model space, q is in head_dim space
        # We need to project number_direction into head_dim space: W_Q_h @ number_direction
        w_q = layers[li].self_attn.q_proj.weight
        W_Q = safe_weight_to_numpy(w_q, model_name, f"model.layers.{li}.self_attn.q_proj.weight")
        W_Q_h = W_Q[h * head_dim:(h + 1) * head_dim, :]  # [head_dim, d_model]

        w_k = layers[li].self_attn.k_proj.weight
        W_K = safe_weight_to_numpy(w_k, model_name, f"model.layers.{li}.self_attn.k_proj.weight")
        W_K_h = get_W_K_h(W_K, h, head_dim, kv_group_size)  # [head_dim, d_model]

        # Project number direction into Q and K subspaces
        num_dir_in_q_space = W_Q_h @ num_dir  # [head_dim]
        num_dir_in_k_space = W_K_h @ num_dir  # [head_dim]

        # Average Q vector at verb position
        if h_label in head_q_at_verb and head_q_at_verb[h_label]:
            mean_q = np.mean(head_q_at_verb[h_label], axis=0)  # [head_dim]
            q_alignment = cosine_sim(num_dir_in_q_space, mean_q)
        else:
            q_alignment = None

        # Average K vector at subject position
        if h_label in head_k_at_subj and head_k_at_subj[h_label]:
            mean_k = np.mean(head_k_at_subj[h_label], axis=0)  # [head_dim]
            k_alignment = cosine_sim(num_dir_in_k_space, mean_k)
        else:
            k_alignment = None

        # Also: alignment of individual Q/K vectors (not just mean)
        q_aligns = []
        if h_label in head_q_at_verb:
            for q_vec in head_q_at_verb[h_label]:
                q_aligns.append(cosine_sim(num_dir_in_q_space, q_vec))

        k_aligns = []
        if h_label in head_k_at_subj:
            for k_vec in head_k_at_subj[h_label]:
                k_aligns.append(cosine_sim(num_dir_in_k_space, k_vec))

        alignment_results[h_label] = {
            "role": role,
            "layer": li,
            "head": h,
            "q_alignment_mean": round(q_alignment, 4) if q_alignment is not None else None,
            "k_alignment_mean": round(k_alignment, 4) if k_alignment is not None else None,
            "q_alignment_per_prompt": [round(a, 4) for a in q_aligns],
            "k_alignment_per_prompt": [round(a, 4) for a in k_aligns],
            "q_alignment_std": round(float(np.std(q_aligns)), 4) if q_aligns else None,
            "k_alignment_std": round(float(np.std(k_aligns)), 4) if k_aligns else None,
        }

        log_time(f"  {h_label} ({role}): Q_align={q_alignment:.3f}±{np.std(q_aligns):.3f}, "
                 f"K_align={k_alignment:.3f}±{np.std(k_aligns):.3f}" if q_alignment is not None and k_alignment is not None else
                 f"  {h_label} ({role}): incomplete data")

    results["alignment_results"] = alignment_results

    # ---- Summary: driver vs follower alignment ----
    log_time("\n=== Driver vs Follower Alignment Summary ===")

    driver_q = [alignment_results[h]["q_alignment_mean"] for h in alignment_results
                if alignment_results[h]["role"] == "driver" and alignment_results[h]["q_alignment_mean"] is not None]
    driver_k = [alignment_results[h]["k_alignment_mean"] for h in alignment_results
                if alignment_results[h]["role"] == "driver" and alignment_results[h]["k_alignment_mean"] is not None]
    follower_q = [alignment_results[h]["q_alignment_mean"] for h in alignment_results
                  if alignment_results[h]["role"] == "follower" and alignment_results[h]["q_alignment_mean"] is not None]
    follower_k = [alignment_results[h]["k_alignment_mean"] for h in alignment_results
                  if alignment_results[h]["role"] == "follower" and alignment_results[h]["k_alignment_mean"] is not None]

    results["alignment_summary"] = {
        "driver_mean_Q_alignment": round(float(np.mean(driver_q)), 4) if driver_q else None,
        "driver_mean_K_alignment": round(float(np.mean(driver_k)), 4) if driver_k else None,
        "follower_mean_Q_alignment": round(float(np.mean(follower_q)), 4) if follower_q else None,
        "follower_mean_K_alignment": round(float(np.mean(follower_k)), 4) if follower_k else None,
    }

    log_time(f"  Drivers: Q_align={results['alignment_summary']['driver_mean_Q_alignment']}, "
             f"K_align={results['alignment_summary']['driver_mean_K_alignment']}")
    log_time(f"  Followers: Q_align={results['alignment_summary']['follower_mean_Q_alignment']}, "
             f"K_align={results['alignment_summary']['follower_mean_K_alignment']}")

    # ---- Additional: Per-layer probe accuracy comparison (for layering analysis) ----
    log_time("\n=== Per-layer probe accuracy (for layering analysis) ===")

    # Load Part 3 results
    part3_path = RESULT_DIR / f"{model_name}_part3.json"
    if part3_path.exists():
        with open(part3_path, 'r', encoding='utf-8') as f:
            part3_data = json.load(f)

        layer_probing = part3_data.get("layer_probing", {})
        accuracy_curve = {}
        for li_str, v in layer_probing.items():
            li = int(li_str)
            if "cv_accuracy" in v:
                accuracy_curve[li] = v["cv_accuracy"]

        results["accuracy_curve"] = accuracy_curve

        # Analyze curve shape
        if accuracy_curve:
            sorted_layers = sorted(accuracy_curve.keys())
            log_time(f"  Accuracy curve for {model_name}:")
            for li in sorted_layers:
                acc = accuracy_curve[li]
                bar = "█" * int(acc * 50)
                log_time(f"    L{li:2d}: {acc:.3f} {bar}")

            # Find inflection points (where accuracy rises fastest)
            if len(sorted_layers) > 2:
                diffs = []
                for i in range(1, len(sorted_layers)):
                    d = accuracy_curve[sorted_layers[i]] - accuracy_curve[sorted_layers[i-1]]
                    diffs.append((sorted_layers[i-1], sorted_layers[i], round(d, 4)))

                max_rise = max(diffs, key=lambda x: x[2])
                results["max_accuracy_rise"] = {
                    "from_layer": max_rise[0],
                    "to_layer": max_rise[1],
                    "accuracy_increase": max_rise[2],
                }
                log_time(f"  Max accuracy rise: L{max_rise[0]}→L{max_rise[1]} (+{max_rise[2]:.3f})")

    save_result(model_name, 4, results)
    release_model_safe(model)
    return results


# ============================================================
# Main
# ============================================================

PART_FUNCTIONS = {
    1: part1_two_stage,
    2: part2_infrastructure_head,
    3: part3_probing_classifier,
    4: part4_qk_alignment,
}

def main():
    parser = argparse.ArgumentParser(description="Phase 259: Two-Stage Computation & Probing Classifier")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="Model to test")
    parser.add_argument("--part", type=str, required=True,
                       help="Part number (1-4) or 'all'")
    args = parser.parse_args()

    model_name = args.model

    if args.part == "all":
        parts = [1, 2, 3, 4]
    else:
        parts = [int(args.part)]

    log_time(f"Phase 259: Two-Stage Computation & Probing Classifier")
    log_time(f"Model: {model_name}, Parts: {parts}")
    log_time(f"=" * 60)

    for part_num in parts:
        if part_num not in PART_FUNCTIONS:
            log_time(f"Unknown part: {part_num}, skipping")
            continue

        log_time(f"\n{'#' * 60}")
        log_time(f"# Starting Part {part_num}")
        log_time(f"{'#' * 60}")

        try:
            result = PART_FUNCTIONS[part_num](model_name)
            log_time(f"Part {part_num} completed successfully!")
        except Exception as e:
            log_time(f"Part {part_num} FAILED: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()
        import torch
        torch.cuda.empty_cache()
        time.sleep(2)

    log_time(f"\nPhase 259 completed for {model_name}!")

if __name__ == "__main__":
    main()
