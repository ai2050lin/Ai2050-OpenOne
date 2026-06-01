"""
Phase 319b: Relation-Specific Template Validation
===================================================

Critical follow-up to Phase 319: Phase 319 showed that "the {w} was there"
template produces template-specific directions (cross-template cosine only 0.08-0.27).
This test uses RELATION-SPECIFIC templates to extract direction vectors.

Key question: Does using natural, relation-activating templates produce
directions that are above random baseline?

Design:
- 4 relation types x 20 pairs x 2 templates (neutral vs relation-specific)
- Compare direction quality between neutral and relation-specific templates
- Measure: pairwise cosine, LOO cosine, cross-template consistency

Templates:
  same_class:
    neutral: "the {A} was there" (same as Phase 318)
    specific: "{A} and {B} are both things" (activates category membership)

  attribute:
    neutral: "the {A} was there" / "the {B} was there" (word substitution)
    specific: "the {N} is {A}" vs "the {N} is just a thing" (attribute activation)

  function:
    neutral: "the {A} was there" / "the {B} was there"
    specific: "people use the {T} to {V}" vs "people use the {T}" (function activation)

  antonym:
    neutral: "the {A} was there" / "the {B} was there"
    specific: "{A} is the opposite of {B}" (activates antonym relation)

Plus: control condition with "they mentioned the {w}" (different neutral template)

Usage:
  python tests/glm5/phase319b_relation_templates.py qwen3
  python tests/glm5/phase319b_relation_templates.py glm4
  python tests/glm5/phase319b_relation_templates.py deepseek7b
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase319b_templates")
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

# 20 pairs per type
SAME_CLASS_PAIRS = [
    ("apple", "orange"), ("banana", "grape"), ("lemon", "cherry"),
    ("peach", "plum"), ("pear", "mango"),
    ("dog", "cat"), ("lion", "tiger"), ("eagle", "hawk"),
    ("whale", "dolphin"), ("rabbit", "squirrel"),
    ("car", "bus"), ("train", "subway"), ("bicycle", "motorcycle"),
    ("airplane", "helicopter"), ("boat", "ship"),
    ("rain", "snow"), ("storm", "thunder"), ("fog", "mist"),
    ("wind", "breeze"), ("frost", "hail"),
]

ATTRIBUTE_PAIRS = [
    ("apple", "red"), ("sky", "blue"), ("fire", "hot"),
    ("ice", "cold"), ("silk", "smooth"), ("sandpaper", "rough"),
    ("glass", "clear"), ("stone", "hard"), ("cotton", "soft"),
    ("feather", "light"), ("lead", "heavy"), ("mountain", "tall"),
    ("lemon", "sour"), ("honey", "sweet"), ("chili", "spicy"),
    ("coffee", "bitter"), ("salt", "salty"), ("diamond", "brilliant"),
    ("night", "dark"), ("steel", "strong"),
]

FUNCTION_PAIRS = [
    ("knife", "cut"), ("scissors", "snip"), ("pen", "write"),
    ("pencil", "draw"), ("car", "drive"), ("bicycle", "ride"),
    ("phone", "call"), ("radio", "broadcast"), ("key", "unlock"),
    ("lock", "secure"), ("cup", "drink"), ("bowl", "contain"),
    ("lamp", "illuminate"), ("flashlight", "shine"),
    ("clock", "measure"), ("watch", "display"),
    ("umbrella", "protect"), ("shield", "defend"),
    ("camera", "capture"), ("microphone", "record"),
]

ANTONYM_PAIRS = [
    ("happy", "sad"), ("hot", "cold"), ("big", "small"),
    ("tall", "short"), ("fast", "slow"), ("light", "dark"),
    ("strong", "weak"), ("young", "old"), ("rich", "poor"),
    ("clean", "dirty"), ("hard", "soft"), ("loud", "quiet"),
    ("open", "closed"), ("full", "empty"), ("love", "hate"),
    ("win", "lose"), ("rise", "fall"), ("create", "destroy"),
    ("accept", "reject"), ("include", "exclude"),
]

# Template definitions
# Each template pair: (sentence_A, sentence_B) for computing direction A→B
TEMPLATES = {
    "same_class": {
        "neutral": lambda A, B: (f"the {A} was there", f"the {B} was there"),
        "specific": lambda A, B: (f"{A} and {B} are both things", f"{B} and {A} are both things"),
        "category": lambda A, B: (f"{A} is a kind of thing", f"{B} is a kind of thing"),
    },
    "attribute": {
        "neutral": lambda N, A: (f"the {N} was there", f"the {A} was there"),
        "specific": lambda N, A: (f"the {N} is {A}", f"the {N} is just an object"),
        "descriptive": lambda N, A: (f"the {N} has the quality of being {A}", f"the {N} is just an object"),
    },
    "function": {
        "neutral": lambda T, V: (f"the {T} was there", f"the {V} was there"),
        "specific": lambda T, V: (f"people use the {T} to {V}", f"people use the {T} for something"),
        "purpose": lambda T, V: (f"the {T} is for {V}ing", f"the {T} is for something"),
    },
    "antonym": {
        "neutral": lambda A, B: (f"the {A} was there", f"the {B} was there"),
        "specific": lambda A, B: (f"{A} is the opposite of {B}", f"{B} is the opposite of {A}"),
        "contrast": lambda A, B: (f"not {A} but {B}", f"not {B} but {A}"),
    },
}

# Random control (20 pairs, matched to attribute pattern: noun→adjective)
RANDOM_CONTROL = [
    ("table", "blue"), ("mountain", "quick"), ("river", "soft"),
    ("cloud", "sharp"), ("forest", "sweet"), ("chair", "dark"),
    ("book", "cold"), ("door", "bright"), ("garden", "rough"),
    ("window", "warm"), ("stone", "light"), ("ocean", "quiet"),
    ("street", "heavy"), ("bridge", "thin"), ("tower", "gentle"),
    ("valley", "strong"), ("island", "slow"), ("desert", "smooth"),
    ("planet", "loud"), ("castle", "fast"),
]


# =====================================================================
# MODEL LOADING
# =====================================================================

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

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation=attn_impl,
        )
        log(f"  Loaded with {attn_impl}")
    except Exception as e:
        log(f"  flash_attention_2 failed, falling back to sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="sdpa",
        )
        log(f"  Loaded with sdpa")

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
# REPRESENTATION EXTRACTION
# =====================================================================

def get_target_layers(n_layers):
    if n_layers >= 36:
        return [6, 12, 18, 24, n_layers - 2]
    elif n_layers >= 28:
        return [4, 8, 12, 16, n_layers - 2]
    else:
        return [2, 4, 8, 12, n_layers - 2]


def extract_representations(model, tokenizer, device, items, target_layers, label=""):
    layers = get_layers(model)
    cache = {}
    n_items = len(items)
    captured = {}

    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[li] = output[0].detach().float().cpu()
            else:
                captured[li] = output.detach().float().cpu()
        return hook_fn

    hooks = [layers[li].register_forward_hook(make_hook(li)) for li in target_layers]

    try:
        for idx, item in enumerate(items):
            inp = tokenizer(item, return_tensors="pt", truncation=True, max_length=128).to(device)
            captured.clear()
            with torch.no_grad():
                model(**inp)

            for li in target_layers:
                if li in captured:
                    cache[(item, li)] = captured[li][0, -1].numpy()

            if (idx + 1) % 30 == 0 or idx == n_items - 1:
                log(f"    {label} Extracted {idx+1}/{n_items}, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

            if (idx + 1) % 80 == 0:
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    return cache


# =====================================================================
# ANALYSIS FUNCTIONS
# =====================================================================

def pca_analysis(D):
    n, d = D.shape
    if n < 2:
        return {"error": "need >=2"}
    k = min(n - 1, d)
    D_c = D - D.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(D_c, full_matrices=False)
    eig = (S ** 2) / (n - 1)
    total = eig.sum()
    if total < 1e-20:
        return {"error": "zero variance"}
    ratio = eig / total
    cum = np.cumsum(ratio)
    return {
        "dim_at_50": int(np.searchsorted(cum, 0.50) + 1),
        "dim_at_80": int(np.searchsorted(cum, 0.80) + 1),
        "dim_at_90": int(np.searchsorted(cum, 0.90) + 1),
        "top1_explained": float(ratio[0]),
    }


def loo_cosine_analysis(D):
    n = D.shape[0]
    if n < 3:
        return {"error": "need >=3"}
    G = D @ D.T
    cosines = []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        A = G[np.ix_(mask, mask)]
        b = G[i, mask]
        try:
            x = np.linalg.solve(A + 1e-8 * np.eye(n-1), b)
            proj_sq = float(b @ x)
            d_sq = float(G[i, i])
            cosines.append(min(np.sqrt(max(proj_sq, 0)) / np.sqrt(max(d_sq, 1e-20)), 1.0))
        except np.linalg.LinAlgError:
            cosines.append(0.0)
    return {"mean": float(np.mean(cosines)), "median": float(np.median(cosines))}


def pairwise_cosine_analysis(D):
    n = D.shape[0]
    if n < 2:
        return {"error": "need >=2"}
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    D_norm = D / norms
    cos_matrix = D_norm @ D_norm.T
    mask = ~np.eye(n, dtype=bool)
    cos_vals = cos_matrix[mask]
    return {
        "mean": float(np.mean(cos_vals)),
        "median": float(np.median(cos_vals)),
        "std": float(np.std(cos_vals)),
    }


def norm_stats(norms):
    if len(norms) == 0:
        return {"error": "empty"}
    return {"mean": float(np.mean(norms)), "std": float(np.std(norms))}


# =====================================================================
# COLLECT ALL UNIQUE ITEMS
# =====================================================================

def collect_all_unique_items():
    items = set()

    # Relation-specific templates for each type
    for rel_type, pairs, templates in [
        ("same_class", SAME_CLASS_PAIRS, TEMPLATES["same_class"]),
        ("attribute", ATTRIBUTE_PAIRS, TEMPLATES["attribute"]),
        ("function", FUNCTION_PAIRS, TEMPLATES["function"]),
        ("antonym", ANTONYM_PAIRS, TEMPLATES["antonym"]),
    ]:
        for tmpl_name, tmpl_fn in templates.items():
            for A, B in pairs:
                sentA, sentB = tmpl_fn(A, B)
                items.add(sentA)
                items.add(sentB)

    # Random control (neutral template only)
    for A, B in RANDOM_CONTROL:
        items.add(f"the {A} was there")
        items.add(f"the {B} was there")

    return sorted(items)


# =====================================================================
# MAIN ANALYSIS
# =====================================================================

def run_analysis(cache, target_layers):
    """Run the full template comparison analysis."""
    log("\nRunning relation-specific template analysis...")

    results = {}

    # Process each relation type
    for rel_type, pairs, templates in [
        ("same_class", SAME_CLASS_PAIRS, TEMPLATES["same_class"]),
        ("attribute", ATTRIBUTE_PAIRS, TEMPLATES["attribute"]),
        ("function", FUNCTION_PAIRS, TEMPLATES["function"]),
        ("antonym", ANTONYM_PAIRS, TEMPLATES["antonym"]),
    ]:
        type_result = {}

        # Compute directions for each template
        for tmpl_name, tmpl_fn in templates.items():
            tmpl_result = {}
            for li in target_layers:
                d_list, n_list = [], []
                for A, B in pairs:
                    sentA, sentB = tmpl_fn(A, B)
                    kA, kB = (sentA, li), (sentB, li)
                    if kA in cache and kB in cache:
                        d = cache[kB] - cache[kA]
                        n_list.append(float(np.linalg.norm(d)))
                        d_list.append(d)

                if len(d_list) < 3:
                    continue

                D = np.array(d_list)
                D_norms = np.linalg.norm(D, axis=1, keepdims=True)
                D_norms = np.maximum(D_norms, 1e-10)
                D_unit = D / D_norms

                tmpl_result[f"L{li}"] = {
                    "pca_normalized": pca_analysis(D_unit),
                    "loo_cosine": loo_cosine_analysis(D),
                    "pairwise_cosine": pairwise_cosine_analysis(D),
                    "norm_stats": norm_stats(n_list),
                }

                log(f"  {rel_type}/{tmpl_name}/L{li}: dim@80={tmpl_result[f'L{li}']['pca_normalized']['dim_at_80']}, "
                    f"LOO_cos={tmpl_result[f'L{li}']['loo_cosine']['mean']:.3f}, "
                    f"pair_cos={tmpl_result[f'L{li}']['pairwise_cosine']['mean']:.4f}, "
                    f"norm_mean={tmpl_result[f'L{li}']['norm_stats']['mean']:.2f}")

            type_result[tmpl_name] = tmpl_result

        # Cross-template direction consistency (deepest layer)
        deepest_li = target_layers[-1]
        tmpl_names = list(templates.keys())
        cross_result = {}

        # Get direction matrices for each template at deepest layer
        dirs_by_tmpl = {}
        for tmpl_name, tmpl_fn in templates.items():
            d_list = []
            for A, B in pairs:
                sentA, sentB = tmpl_fn(A, B)
                kA, kB = (sentA, deepest_li), (sentB, deepest_li)
                if kA in cache and kB in cache:
                    d_list.append(cache[kB] - cache[kA])
            if d_list:
                dirs_by_tmpl[tmpl_name] = np.array(d_list)

        for i in range(len(tmpl_names)):
            for j in range(i+1, len(tmpl_names)):
                tA, tB = tmpl_names[i], tmpl_names[j]
                if tA not in dirs_by_tmpl or tB not in dirs_by_tmpl:
                    continue
                DA, DB = dirs_by_tmpl[tA], dirs_by_tmpl[tB]
                n_pairs = min(DA.shape[0], DB.shape[0])
                cosines = []
                for k in range(n_pairs):
                    dA, dB = DA[k], DB[k]
                    nA, nB = np.linalg.norm(dA), np.linalg.norm(dB)
                    if nA > 1e-10 and nB > 1e-10:
                        cosines.append(float(np.dot(dA, dB) / (nA * nB)))
                if cosines:
                    cross_result[f"{tA}_vs_{tB}"] = {
                        "mean_cosine": float(np.mean(cosines)),
                        "median_cosine": float(np.median(cosines)),
                        "frac_positive": float(np.mean([c > 0 for c in cosines])),
                        "frac_above_03": float(np.mean([c > 0.3 for c in cosines])),
                        "frac_above_05": float(np.mean([c > 0.5 for c in cosines])),
                    }
                    cr = cross_result[f"{tA}_vs_{tB}"]
                    log(f"  {rel_type} {tA} vs {tB}: mean_cos={cr['mean_cosine']:.3f}, "
                        f"frac>0.3={cr['frac_above_03']:.2f}")

        type_result["cross_template"] = cross_result
        results[rel_type] = type_result

    # Random control baseline
    log("\nRandom control baseline...")
    random_result = {}
    for li in target_layers:
        d_list = []
        for A, B in RANDOM_CONTROL:
            sentA = f"the {A} was there"
            sentB = f"the {B} was there"
            kA, kB = (sentA, li), (sentB, li)
            if kA in cache and kB in cache:
                d_list.append(cache[kB] - cache[kA])
        if len(d_list) >= 3:
            D = np.array(d_list)
            D_norms = np.linalg.norm(D, axis=1, keepdims=True)
            D_norms = np.maximum(D_norms, 1e-10)
            D_unit = D / D_norms
            random_result[f"L{li}"] = {
                "pca_normalized": pca_analysis(D_unit),
                "loo_cosine": loo_cosine_analysis(D),
                "pairwise_cosine": pairwise_cosine_analysis(D),
            }
            log(f"  random_control/L{li}: dim@80={random_result[f'L{li}']['pca_normalized']['dim_at_80']}, "
                f"LOO_cos={random_result[f'L{li}']['loo_cosine']['mean']:.3f}, "
                f"pair_cos={random_result[f'L{li}']['pairwise_cosine']['mean']:.4f}")

    results["random_control"] = random_result

    return results


# =====================================================================
# MAIN
# =====================================================================

def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase319b_{model_name}.log")

    log(f"=== Phase 319b: Relation-Specific Templates for {model_name} ===")

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}, class={info.model_class}")
    t_load = time.time() - t0
    log(f"  Load time: {t_load:.1f}s")

    target_layers = get_target_layers(info.n_layers)
    log(f"  Target layers: {target_layers}")

    unique_items = collect_all_unique_items()
    log(f"  Total unique items: {len(unique_items)}")

    t0 = time.time()
    log("Extracting representations...")
    cache = extract_representations(model, tokenizer, device, unique_items, target_layers, label="All")
    t_extract = time.time() - t0
    log(f"  Extraction time: {t_extract:.1f}s")

    results = run_analysis(cache, target_layers)

    all_results = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "target_layers": target_layers,
        "extraction_time_s": round(t_extract, 1),
        "analysis": results,
    }

    out_path = RESULT_DIR / f"{model_name}_phase319b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    print_summary(all_results)

    release_model(model)
    del model, cache
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released.")

    return all_results


def print_summary(results):
    log("\n" + "=" * 70)
    log(f"PHASE 319b SUMMARY - {results['model']}")
    log("=" * 70)

    analysis = results["analysis"]

    # Per-type comparison: neutral vs specific (deepest layer)
    log("\n--- Relation-Specific vs Neutral Templates (deepest layer) ---")
    log(f"{'Type':<15} {'Template':<15} {'dim@80':<8} {'LOO_cos':<9} {'pair_cos':<10}")
    for rel_type in ["same_class", "attribute", "function", "antonym"]:
        if rel_type not in analysis:
            continue
        type_data = analysis[rel_type]
        for tmpl_name in ["neutral", "specific", "category", "descriptive", "purpose", "contrast"]:
            if tmpl_name not in type_data:
                continue
            tmpl_data = type_data[tmpl_name]
            # Get deepest layer
            deepest = None
            for lk in sorted(tmpl_data.keys(), key=lambda x: int(x.replace("L",""))):
                deepest = tmpl_data[lk]
            if deepest and "pca_normalized" in deepest:
                pca = deepest["pca_normalized"]
                loo = deepest.get("loo_cosine", {})
                pair = deepest.get("pairwise_cosine", {})
                log(f"{rel_type:<15} {tmpl_name:<15} {pca.get('dim_at_80','?'):<8} "
                    f"{loo.get('mean',0):.3f}   {pair.get('mean',0):.4f}")

    # Random control
    if "random_control" in analysis:
        rc = analysis["random_control"]
        deepest = None
        for lk in sorted(rc.keys(), key=lambda x: int(x.replace("L",""))):
            deepest = rc[lk]
        if deepest:
            pca = deepest.get("pca_normalized", {})
            loo = deepest.get("loo_cosine", {})
            pair = deepest.get("pairwise_cosine", {})
            log(f"{'random_control':<15} {'neutral':<15} {pca.get('dim_at_80','?'):<8} "
                f"{loo.get('mean',0):.3f}   {pair.get('mean',0):.4f}")

    # Cross-template consistency
    log("\n--- Cross-Template Consistency ---")
    for rel_type in ["same_class", "attribute", "function", "antonym"]:
        if rel_type not in analysis:
            continue
        cross = analysis[rel_type].get("cross_template", {})
        for key, cr in cross.items():
            log(f"  {rel_type} {key}: mean_cos={cr.get('mean_cosine',0):.3f}, "
                f"frac>0.3={cr.get('frac_above_03',0):.2f}")


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
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(10)
    else:
        run_model(model_name)
