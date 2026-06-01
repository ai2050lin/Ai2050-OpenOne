"""
Phase 321: Fine-Grained Layer Scan + Readout Alignment
========================================================

Combines Phase 321 (fine layer scan) and Phase 322 (readout alignment) into one script.

Based on Phase 320b results, the optimal injection layers are:
- GLM4: attribute L4, function L12-16, negation L0
- Qwen3: attribute L0, function L4/L20
- DS7B: attribute L9-12, negation L0

This script:
1. Fine-scans around optimal layers with 1-layer resolution
2. Computes readout alignment: cos(d_relation, W_U[target_token])
3. Computes cluster readout: cos(d_relation, W_U[cluster_center])
4. Tests whether readout alignment explains cross-model causal efficacy differences

Usage:
  python tests/glm5/phase321_fine_layer_readout.py qwen3
  python tests/glm5/phase321_fine_layer_readout.py glm4
  python tests/glm5/phase321_fine_layer_readout.py deepseek7b
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model, get_W_U

RESULT_DIR = Path("results/phase321_fine_readout")
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


# ===== Stimuli =====
ATTRIBUTE_PAIRS = [
    # (noun, attribute_value, attribute_type)
    ("apple", "red", "color"), ("sky", "blue", "color"), ("fire", "hot", "temperature"),
    ("ice", "cold", "temperature"), ("lemon", "sour", "taste"), ("honey", "sweet", "taste"),
    ("silk", "smooth", "texture"), ("sandpaper", "rough", "texture"),
    ("grass", "green", "color"), ("snow", "white", "color"),
    ("oven", "hot", "temperature"), ("milk", "cold", "temperature"),
    ("vinegar", "sour", "taste"), ("sugar", "sweet", "taste"),
    ("velvet", "smooth", "texture"), ("gravel", "rough", "texture"),
]

ATTRIBUTE_TARGET_PAIRS = [
    ("strawberry", "red", "color"), ("ocean", "blue", "color"), ("stove", "hot", "temperature"),
    ("frost", "cold", "temperature"), ("grapefruit", "sour", "taste"), ("candy", "sweet", "taste"),
    ("satin", "smooth", "texture"), ("concrete", "rough", "texture"),
    ("emerald", "green", "color"), ("cloud", "white", "color"),
]

# Attribute cluster words for readout alignment
COLOR_WORDS = ["red", "blue", "green", "white", "black", "yellow", "orange", "purple", "brown", "pink"]
TEMP_WORDS = ["hot", "cold", "warm", "cool", "freezing", "boiling", "tepid", "chilly"]
TASTE_WORDS = ["sweet", "sour", "bitter", "salty", "savory", "spicy", "tangy", "umami"]
TEXTURE_WORDS = ["smooth", "rough", "soft", "hard", "silky", "coarse", "fine", "gritty"]

FUNCTION_PAIRS = [
    ("knife", "cut"), ("pen", "write"), ("car", "drive"),
    ("phone", "call"), ("key", "unlock"), ("lamp", "illuminate"),
    ("camera", "capture"), ("brush", "paint"),
    ("scissors", "cut"), ("pencil", "write"),
    ("bus", "drive"), ("flashlight", "illuminate"),
]

# Action cluster words for readout alignment
CUT_WORDS = ["cut", "slice", "chop", "dice", "carve", "trim", "shear", "sever"]
WRITE_WORDS = ["write", "draw", "sketch", "draft", "inscribe", "scribble", "compose", "note"]
DRIVE_WORDS = ["drive", "ride", "steer", "pilot", "navigate", "cruise", "travel", "commute"]

NEGATION_ADJECTIVES = [
    "happy", "good", "clean", "safe", "warm",
    "fast", "strong", "bright", "easy", "soft",
    "smart", "kind", "rich", "brave", "calm",
    "fair", "loud", "sharp", "thick", "deep",
]

NEG_WORDS_CLUSTER = ["not", "never", "no", "neither", "nor", "none", "nobody", "nothing"]
NEG_ADJ_WORDS = ["unhappy", "sad", "bad", "dirty", "unsafe", "cold"]


# ===== Fine scan layers =====
def get_fine_scan_layers(model_name, n_layers):
    """Return layers for fine scanning based on Phase 320b results."""
    if model_name == "glm4":
        # GLM4: 40 layers
        # Attribute peak at L4, scan L1-L9
        # Function peak at L12-16, scan L9-L19
        # Negation peak at L0, scan L0-L5
        attr_layers = list(range(1, 10))
        func_layers = list(range(9, 20))
        neg_layers = list(range(0, 6))
    elif model_name == "qwen3":
        # Qwen3: 36 layers
        # Attribute peak at L0, scan L0-L10
        # Function peak at L4/L20, scan L0-L10 and L16-L24
        attr_layers = list(range(0, 11))
        func_layers = list(range(0, 11)) + list(range(16, 25))
        neg_layers = list(range(0, 11))
    elif model_name == "deepseek7b":
        # DS7B: 28 layers
        # Attribute best at L9, scan L3-L15
        # Function best at L0, scan L0-L8
        # Negation best at L0, scan L0-L8
        attr_layers = list(range(3, 16))
        func_layers = list(range(0, 9))
        neg_layers = list(range(0, 9))
    else:
        attr_layers = list(range(0, min(12, n_layers)))
        func_layers = list(range(0, min(12, n_layers)))
        neg_layers = list(range(0, min(6, n_layers)))

    # Ensure no layer exceeds max
    attr_layers = [l for l in attr_layers if l < n_layers]
    func_layers = [l for l in func_layers if l < n_layers]
    neg_layers = [l for l in neg_layers if l < n_layers]

    return attr_layers, func_layers, neg_layers


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    attn_impl = "flash_attention_2"
    log(f"Loading {model_name} (bf16 + device_map=auto + {attn_impl})...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for impl in [attn_impl, "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=impl,
            )
            log(f"  Loaded with {impl}")
            break
        except Exception as e:
            log(f"  {impl} failed, trying next...")
            continue

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Model: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def extract_rep_at_layers(model, tokenizer, device, sentences, target_layers, label=""):
    """Extract representations at specific layers."""
    layers_list = get_layers(model)
    cache = {}
    captured = {}

    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[li] = output[0].detach().float().cpu()
            else:
                captured[li] = output.detach().float().cpu()
        return hook_fn

    hooks = [layers_list[li].register_forward_hook(make_hook(li)) for li in target_layers]

    try:
        for idx, sent in enumerate(sentences):
            inp = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128).to(device)
            captured.clear()
            with torch.no_grad():
                model(**inp)

            for li in target_layers:
                if li in captured:
                    cache[(sent, li)] = captured[li][0, -1].numpy()

            if (idx + 1) % 30 == 0 or idx == len(sentences) - 1:
                log(f"    {label} Extracted {idx+1}/{len(sentences)}, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

            if (idx + 1) % 60 == 0:
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    return cache


def inject_and_get_logits(model, tokenizer, device, prompt, direction, layer_idx, alpha, top_k=20):
    """Inject direction at a specific layer and get logits."""
    layers_list = get_layers(model)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)

    injected_logits = None

    def hook_fn(module, input, output):
        nonlocal injected_logits
        hidden = output[0] if isinstance(output, tuple) else output
        d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
        hidden_modified = hidden.clone()
        hidden_modified[0, -1, :] += (alpha * d_tensor).to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        return hidden_modified

    hook = layers_list[layer_idx].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            out = model(**inp)
        injected_logits = out.logits[0, -1].float().cpu().numpy()
    finally:
        hook.remove()

    return injected_logits


def get_baseline_logits(model, tokenizer, device, prompt):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1].float().cpu().numpy()
    return logits


def compute_readout_alignment(direction, W_U, tokenizer, target_words):
    """
    Compute readout alignment between a direction and W_U rows for target words.
    
    Returns dict with:
    - mean_cos: mean cosine similarity with target word W_U rows
    - max_cos: max cosine similarity
    - individual: per-word cosine
    """
    d_norm = np.linalg.norm(direction)
    if d_norm < 1e-10:
        return {"mean_cos": 0, "max_cos": 0, "individual": {}}

    d_unit = direction / d_norm
    results = {}
    for word in target_words:
        tok_ids = tokenizer.encode(word, add_special_tokens=False)
        if not tok_ids:
            continue
        w_vec = W_U[tok_ids[0]]
        w_norm = np.linalg.norm(w_vec)
        if w_norm < 1e-10:
            continue
        cos_val = float(np.dot(d_unit, w_vec) / w_norm)
        results[word] = cos_val

    if not results:
        return {"mean_cos": 0, "max_cos": 0, "individual": {}}

    return {
        "mean_cos": float(np.mean(list(results.values()))),
        "max_cos": float(max(results.values())),
        "min_cos": float(min(results.values())),
        "individual": results,
    }


def compute_cluster_readout(direction, W_U, tokenizer, cluster_words):
    """Compute readout alignment with a cluster of words."""
    return compute_readout_alignment(direction, W_U, tokenizer, cluster_words)


def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase321_{model_name}.log")

    log(f"=== Phase 321: Fine Layer Scan + Readout Alignment for {model_name} ===")

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}, class={info.model_class}")
    t_load = time.time() - t0

    # Get W_U for readout alignment
    log("Loading W_U for readout alignment...")
    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")

    # Get fine scan layers
    attr_layers, func_layers, neg_layers = get_fine_scan_layers(model_name, info.n_layers)
    log(f"  Attribute scan layers: {attr_layers}")
    log(f"  Function scan layers: {func_layers}")
    log(f"  Negation scan layers: {neg_layers}")

    # Collect all sentences
    all_sentences = set()
    for noun, val, _ in ATTRIBUTE_PAIRS:
        all_sentences.add(f"the {noun} is {val}")
        all_sentences.add(f"the {noun} is just an object")
    for noun, val, _ in ATTRIBUTE_TARGET_PAIRS:
        all_sentences.add(f"the {noun} is {val}")
        all_sentences.add(f"the {noun} is just an object")
    for tool, action in FUNCTION_PAIRS:
        all_sentences.add(f"people use the {tool} to {action}")
        all_sentences.add(f"people use the {tool}")
    for adj in NEGATION_ADJECTIVES:
        all_sentences.add(f"very {adj}")
        all_sentences.add(f"not {adj}")
        all_sentences.add(f"never {adj}")

    all_sentences = sorted(all_sentences)
    log(f"  Total sentences: {len(all_sentences)}")

    # Need to extract at ALL target layers (union)
    all_target_layers = sorted(set(attr_layers + func_layers + neg_layers))
    log(f"  Total target layers: {len(all_target_layers)}")

    # Extract representations
    log("Extracting representations at fine-grained layers...")
    t0 = time.time()
    cache = extract_rep_at_layers(model, tokenizer, device, all_sentences, all_target_layers, label="All")
    t_extract = time.time() - t0
    log(f"  Extraction time: {t_extract:.1f}s")

    results = {}
    alpha = 2.0

    # ===================================================================
    # Part A: Attribute Fine Scan + Readout Alignment
    # ===================================================================
    log("\n" + "="*60)
    log("Part A: Attribute Fine Scan + Readout Alignment")
    log("="*60)

    attr_results = []
    attr_readout_results = []

    # Use first 8 source pairs for fine scan
    for src_idx, (src_noun, src_val, src_type) in enumerate(ATTRIBUTE_PAIRS[:8]):
        if src_idx >= len(ATTRIBUTE_TARGET_PAIRS):
            break
        tgt_noun, tgt_val, tgt_type = ATTRIBUTE_TARGET_PAIRS[src_idx]

        target_prompt = f"The {tgt_noun} is"
        baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)

        tgt_val_ids = tokenizer.encode(tgt_val, add_special_tokens=False)
        if not tgt_val_ids:
            continue
        baseline_val_logit = float(baseline_logits[tgt_val_ids[0]])

        for li in attr_layers:
            sent_pos = f"the {src_noun} is {src_val}"
            sent_neg = f"the {src_noun} is just an object"
            k_pos = (sent_pos, li)
            k_neg = (sent_neg, li)

            if k_pos not in cache or k_neg not in cache:
                continue

            d_attr = cache[k_pos] - cache[k_neg]
            d_norm = np.linalg.norm(d_attr)
            if d_norm < 1e-10:
                continue
            d_attr_unit = d_attr / d_norm

            # Causal injection
            inj_logits = inject_and_get_logits(
                model, tokenizer, device, target_prompt,
                d_attr_unit, li, alpha, top_k=30
            )
            delta = float(inj_logits[tgt_val_ids[0]] - baseline_val_logit)

            # Readout alignment
            # 1. Direct target word
            direct_readout = compute_readout_alignment(d_attr, W_U, tokenizer, [src_val, tgt_val])
            # 2. Attribute cluster
            if src_type == "color":
                cluster = COLOR_WORDS
            elif src_type == "temperature":
                cluster = TEMP_WORDS
            elif src_type == "taste":
                cluster = TASTE_WORDS
            else:
                cluster = TEXTURE_WORDS
            cluster_readout = compute_cluster_readout(d_attr, W_U, tokenizer, cluster)

            attr_results.append({
                "source": f"{src_noun}→{src_val}",
                "target": f"{tgt_noun}→{tgt_val}",
                "layer": li,
                "delta": round(delta, 4),
                "positive": delta > 0,
                "direct_readout_mean": round(direct_readout["mean_cos"], 4),
                "cluster_readout_mean": round(cluster_readout["mean_cos"], 4),
            })

        if src_idx == 0:
            log(f"  First source detailed results:")
            for r in attr_results:
                if r["source"] == f"{ATTRIBUTE_PAIRS[0][0]}→{ATTRIBUTE_PAIRS[0][1]}":
                    log(f"    L{r['layer']}: delta={r['delta']:.4f}, direct_ro={r['direct_readout_mean']:.4f}, cluster_ro={r['cluster_readout_mean']:.4f}")

    results["attribute_fine"] = attr_results

    # Summary by layer
    log("\n  Attribute fine scan by layer (alpha=2.0):")
    layer_data = {}
    for r in attr_results:
        li = r["layer"]
        if li not in layer_data:
            layer_data[li] = {"deltas": [], "direct_ro": [], "cluster_ro": []}
        layer_data[li]["deltas"].append(r["delta"])
        layer_data[li]["direct_ro"].append(r["direct_readout_mean"])
        layer_data[li]["cluster_ro"].append(r["cluster_readout_mean"])

    for li in sorted(layer_data.keys()):
        d = layer_data[li]
        mean_d = np.mean(d["deltas"])
        pos_frac = np.mean([1 if x > 0 else 0 for x in d["deltas"]])
        mean_dro = np.mean(d["direct_ro"])
        mean_cro = np.mean(d["cluster_ro"])
        log(f"  L{li}: delta={mean_d:.4f}({pos_frac:.0%}), direct_ro={mean_dro:.4f}, cluster_ro={mean_cro:.4f}")

    # ===================================================================
    # Part B: Function Fine Scan + Readout Alignment
    # ===================================================================
    log("\n" + "="*60)
    log("Part B: Function Fine Scan + Readout Alignment")
    log("="*60)

    func_results = []

    for src_idx, (tool, action) in enumerate(FUNCTION_PAIRS[:8]):
        target_prompt = f"People use the {tool} to"
        baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)

        action_ids = tokenizer.encode(action, add_special_tokens=False)
        if not action_ids:
            continue
        baseline_action_logit = float(baseline_logits[action_ids[0]])

        for li in func_layers:
            sent_pos = f"people use the {tool} to {action}"
            sent_neg = f"people use the {tool}"
            k_pos = (sent_pos, li)
            k_neg = (sent_neg, li)

            if k_pos not in cache or k_neg not in cache:
                continue

            d_func = cache[k_pos] - cache[k_neg]
            d_norm = np.linalg.norm(d_func)
            if d_norm < 1e-10:
                continue
            d_func_unit = d_func / d_norm

            # Causal injection
            inj_logits = inject_and_get_logits(
                model, tokenizer, device, target_prompt,
                d_func_unit, li, alpha, top_k=30
            )
            delta = float(inj_logits[action_ids[0]] - baseline_action_logit)

            # Readout alignment
            direct_readout = compute_readout_alignment(d_func, W_U, tokenizer, [action])
            # Action cluster
            if action == "cut":
                cluster = CUT_WORDS
            elif action == "write":
                cluster = WRITE_WORDS
            elif action == "drive":
                cluster = DRIVE_WORDS
            else:
                cluster = [action]
            cluster_readout = compute_cluster_readout(d_func, W_U, tokenizer, cluster)

            func_results.append({
                "source": f"{tool}→{action}",
                "layer": li,
                "delta": round(delta, 4),
                "positive": delta > 0,
                "direct_readout_mean": round(direct_readout["mean_cos"], 4),
                "cluster_readout_mean": round(cluster_readout["mean_cos"], 4),
            })

    results["function_fine"] = func_results

    # Summary by layer
    log("\n  Function fine scan by layer (alpha=2.0):")
    layer_data = {}
    for r in func_results:
        li = r["layer"]
        if li not in layer_data:
            layer_data[li] = {"deltas": [], "direct_ro": [], "cluster_ro": []}
        layer_data[li]["deltas"].append(r["delta"])
        layer_data[li]["direct_ro"].append(r["direct_readout_mean"])
        layer_data[li]["cluster_ro"].append(r["cluster_readout_mean"])

    for li in sorted(layer_data.keys()):
        d = layer_data[li]
        mean_d = np.mean(d["deltas"])
        pos_frac = np.mean([1 if x > 0 else 0 for x in d["deltas"]])
        mean_dro = np.mean(d["direct_ro"])
        mean_cro = np.mean(d["cluster_ro"])
        log(f"  L{li}: delta={mean_d:.4f}({pos_frac:.0%}), direct_ro={mean_dro:.4f}, cluster_ro={mean_cro:.4f}")

    # ===================================================================
    # Part C: Negation Fine Scan + Readout Alignment
    # ===================================================================
    log("\n" + "="*60)
    log("Part C: Negation Fine Scan + Readout Alignment")
    log("="*60)

    neg_results = []

    for adj in NEGATION_ADJECTIVES[:8]:
        sent_pos = f"very {adj}"
        sent_neg = f"not {adj}"
        prompt = f"very {adj}"

        baseline_logits = get_baseline_logits(model, tokenizer, device, prompt)

        neg_word_ids = {}
        for nw in NEG_WORDS_CLUSTER:
            ids = tokenizer.encode(nw, add_special_tokens=False)
            if ids:
                neg_word_ids[nw] = ids[0]

        adj_ids = tokenizer.encode(adj, add_special_tokens=False)
        adj_id = adj_ids[0] if adj_ids else None

        for li in neg_layers:
            k_pos = (sent_pos, li)
            k_neg = (sent_neg, li)

            if k_pos not in cache or k_neg not in cache:
                continue

            d_neg = cache[k_neg] - cache[k_pos]
            d_norm = np.linalg.norm(d_neg)
            if d_norm < 1e-10:
                continue
            d_neg_unit = d_neg / d_norm

            # Causal injection
            inj_logits = inject_and_get_logits(
                model, tokenizer, device, prompt,
                d_neg_unit, li, alpha, top_k=30
            )

            max_neg_delta = 0
            for nw, nid in neg_word_ids.items():
                delta = float(inj_logits[nid] - baseline_logits[nid])
                max_neg_delta = max(max_neg_delta, delta)

            adj_delta = float(inj_logits[adj_id] - baseline_logits[adj_id]) if adj_id else 0

            # Readout alignment
            neg_readout = compute_readout_alignment(d_neg, W_U, tokenizer, NEG_WORDS_CLUSTER)
            neg_adj_readout = compute_readout_alignment(d_neg, W_U, tokenizer, NEG_ADJ_WORDS)

            neg_results.append({
                "adjective": adj,
                "layer": li,
                "max_neg_delta": round(max_neg_delta, 4),
                "adj_delta": round(adj_delta, 4),
                "neg_word_readout_mean": round(neg_readout["mean_cos"], 4),
                "neg_adj_readout_mean": round(neg_adj_readout["mean_cos"], 4),
            })

    results["negation_fine"] = neg_results

    # Summary by layer
    log("\n  Negation fine scan by layer (alpha=2.0):")
    layer_data = {}
    for r in neg_results:
        li = r["layer"]
        if li not in layer_data:
            layer_data[li] = {"neg_deltas": [], "adj_deltas": [], "neg_ro": [], "neg_adj_ro": []}
        layer_data[li]["neg_deltas"].append(r["max_neg_delta"])
        layer_data[li]["adj_deltas"].append(r["adj_delta"])
        layer_data[li]["neg_ro"].append(r["neg_word_readout_mean"])
        layer_data[li]["neg_adj_ro"].append(r["neg_adj_readout_mean"])

    for li in sorted(layer_data.keys()):
        d = layer_data[li]
        mean_neg = np.mean(d["neg_deltas"])
        mean_adj = np.mean(d["adj_deltas"])
        mean_neg_ro = np.mean(d["neg_ro"])
        mean_neg_adj_ro = np.mean(d["neg_adj_ro"])
        log(f"  L{li}: neg_delta={mean_neg:.4f}, adj_delta={mean_adj:.4f}, neg_ro={mean_neg_ro:.4f}, neg_adj_ro={mean_neg_adj_ro:.4f}")

    # ===================================================================
    # Part D: Cross-model Readout Comparison (without injection)
    # ===================================================================
    log("\n" + "="*60)
    log("Part D: Direction-Readout Alignment (no injection, all layers)")
    log("="*60)

    # For a subset of pairs, compute readout alignment at each layer
    readout_layers = all_target_layers
    readout_alignment_data = []

    for noun, val, atype in ATTRIBUTE_PAIRS[:4]:
        for li in readout_layers:
            k_pos = (f"the {noun} is {val}", li)
            k_neg = (f"the {noun} is just an object", li)
            if k_pos not in cache or k_neg not in cache:
                continue
            d = cache[k_pos] - cache[k_neg]
            d_n = np.linalg.norm(d)
            if d_n < 1e-10:
                continue

            # cos with W_U[val_token]
            val_ids = tokenizer.encode(val, add_special_tokens=False)
            if val_ids:
                w_val = W_U[val_ids[0]]
                w_norm = np.linalg.norm(w_val)
                cos_val = float(np.dot(d, w_val) / (d_n * w_norm)) if w_norm > 0 else 0
            else:
                cos_val = 0

            # cos with random direction baseline
            rand_dir = np.random.randn(d.shape[0])
            rand_dir /= np.linalg.norm(rand_dir)
            rand_cos = float(np.dot(d / d_n, rand_dir))

            readout_alignment_data.append({
                "pair": f"{noun}→{val}",
                "type": atype,
                "layer": li,
                "cos_with_target": round(cos_val, 4),
                "cos_with_random": round(rand_cos, 4),
            })

    results["readout_alignment"] = readout_alignment_data

    log("\n  Readout alignment by layer (attribute):")
    layer_ra = {}
    for r in readout_alignment_data:
        li = r["layer"]
        if li not in layer_ra:
            layer_ra[li] = {"target_cos": [], "random_cos": []}
        layer_ra[li]["target_cos"].append(r["cos_with_target"])
        layer_ra[li]["random_cos"].append(r["cos_with_random"])

    for li in sorted(layer_ra.keys()):
        d = layer_ra[li]
        log(f"  L{li}: target_cos={np.mean(d['target_cos']):.4f}, random_cos={np.mean(d['random_cos']):.4f}, diff={np.mean(d['target_cos'])-np.mean(d['random_cos']):.4f}")

    # Save results
    output = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "alpha": alpha,
        "attr_layers": attr_layers,
        "func_layers": func_layers,
        "neg_layers": neg_layers,
        "results": results,
    }

    out_path = RESULT_DIR / f"{model_name}_phase321.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # Print overall summary
    log("\n" + "="*60)
    log(f"PHASE 321 SUMMARY - {model_name}")
    log("="*60)

    # Find best layer for each type
    log("\nBest injection layer (max mean_delta) with readout alignment:")

    for rtype, key in [("attribute", "attribute_fine"), ("function", "function_fine")]:
        data = results.get(key, [])
        if not data:
            continue
        layer_deltas = {}
        layer_ro = {}
        for r in data:
            li = r["layer"]
            if li not in layer_deltas:
                layer_deltas[li] = []
                layer_ro[li] = []
            layer_deltas[li].append(r["delta"])
            layer_ro[li].append(r.get("direct_readout_mean", 0))

        best_li = max(layer_deltas.keys(), key=lambda li: np.mean(layer_deltas[li]))
        best_delta = np.mean(layer_deltas[best_li])
        best_ro = np.mean(layer_ro.get(best_li, [0]))
        log(f"  {rtype}: best=L{best_li}, delta={best_delta:.4f}, direct_readout={best_ro:.4f}")

    neg_data = results.get("negation_fine", [])
    if neg_data:
        layer_neg = {}
        layer_neg_ro = {}
        for r in neg_data:
            li = r["layer"]
            if li not in layer_neg:
                layer_neg[li] = []
                layer_neg_ro[li] = []
            layer_neg[li].append(r["max_neg_delta"])
            layer_neg_ro[li].append(r.get("neg_word_readout_mean", 0))

        best_li = max(layer_neg.keys(), key=lambda li: np.mean(layer_neg[li]))
        best_delta = np.mean(layer_neg[best_li])
        best_ro = np.mean(layer_neg_ro.get(best_li, [0]))
        log(f"  negation: best=L{best_li}, neg_delta={best_delta:.4f}, neg_readout={best_ro:.4f}")

    # Readout alignment summary
    log("\nReadout alignment summary (target vs random):")
    if readout_alignment_data:
        target_mean = np.mean([r["cos_with_target"] for r in readout_alignment_data])
        random_mean = np.mean([r["cos_with_random"] for r in readout_alignment_data])
        log(f"  Mean target alignment: {target_mean:.4f}")
        log(f"  Mean random alignment: {random_mean:.4f}")
        log(f"  Alignment above random: {target_mean - random_mean:.4f}")

    # Cleanup
    del cache
    del W_U
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released.")

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
