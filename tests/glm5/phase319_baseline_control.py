"""
Phase 319: Random Baseline + Template Control + Negation Token Control
======================================================================

Addresses 3 critical gaps from Phase 318:
1. RANDOM BASELINE: Is same_class/antonym structure above random?
2. TEMPLATE CONTROL: Are relation directions template-invariant?
3. NEGATION TOKEN CONTROL: Is negation consistency from "not" token or negation operation?

Part A: Random Baselines (3 groups x 30 pairs)
  - random_mixed: 30 random word pairs (mixed POS)
  - random_noun_noun: 30 random noun→noun pairs
  - random_adj_adj: 30 random adjective→adjective pairs
  All using "the {w} was there" template (same as Phase 318)
  Compute: PCA dim, LOO cosine, pairwise cosine → compare with Phase 318

Part B: Multi-Template Consistency (4 types x 15 pairs x 3 templates)
  For same_class, attribute, function, antonym:
  - Template 1: "the {w} was there" (original)
  - Template 2: "they mentioned the {w}" (different neutral context)
  - Template 3: "they discussed the {w}" (another neutral context)
  Compute cross-template direction cosine for each pair

Part C: Negation Token Control (20 adjectives x 4 negation forms)
  - not: "very {adj}" → "not {adj}"
  - never: "very {adj}" → "never {adj}"
  - barely: "very {adj}" → "barely {adj}"
  - un-: "very {adj}" → "un{adj}" (morphological)
  Compute cross-form subspace overlap and pairwise consistency

Usage:
  python tests/glm5/phase319_baseline_control.py qwen3
  python tests/glm5/phase319_baseline_control.py glm4
  python tests/glm5/phase319_baseline_control.py deepseek7b
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

RESULT_DIR = Path("results/phase319_control")
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

TEMPLATES = {
    "t1": "the {w} was there",
    "t2": "they mentioned the {w}",
    "t3": "they discussed the {w}",
}

# Part A: Random baselines
RANDOM_MIXED = [
    ("table", "blue"), ("mountain", "quick"), ("river", "soft"),
    ("cloud", "sharp"), ("forest", "sweet"), ("chair", "dark"),
    ("book", "cold"), ("door", "bright"), ("garden", "rough"),
    ("window", "warm"), ("stone", "light"), ("ocean", "quiet"),
    ("street", "heavy"), ("bridge", "thin"), ("tower", "gentle"),
    ("valley", "strong"), ("island", "slow"), ("desert", "smooth"),
    ("planet", "loud"), ("forest", "small"), ("castle", "fast"),
    ("market", "clean"), ("village", "deep"), ("harbor", "wide"),
    ("meadow", "old"), ("canyon", "new"), ("temple", "young"),
    ("lake", "long"), ("field", "short"), ("cliff", "high"),
]

RANDOM_NOUN_NOUN = [
    ("mountain", "pencil"), ("river", "cabinet"), ("cloud", "mirror"),
    ("forest", "blanket"), ("chair", "volcano"), ("book", "feather"),
    ("door", "diamond"), ("garden", "umbrella"), ("window", "compass"),
    ("stone", "lantern"), ("ocean", "pillow"), ("street", "candle"),
    ("bridge", "sandwich"), ("tower", "bottle"), ("valley", "hammer"),
    ("island", "needle"), ("desert", "carpet"), ("planet", "wallet"),
    ("castle", "bicycle"), ("market", "telescope"), ("village", "guitar"),
    ("harbor", "blanket"), ("meadow", "rocket"), ("canyon", "sofa"),
    ("temple", "ladder"), ("lake", "helmet"), ("field", "curtain"),
    ("cliff", "glove"), ("cave", "brush"), ("swamp", "anchor"),
]

RANDOM_ADJ_ADJ = [
    ("happy", "wooden"), ("cold", "ancient"), ("bright", "hungry"),
    ("soft", "digital"), ("heavy", "mysterious"), ("fast", "empty"),
    ("dark", "plastic"), ("warm", "silent"), ("tall", "broken"),
    ("sharp", "lazy"), ("smooth", "wet"), ("loud", "invisible"),
    ("sweet", "frozen"), ("rough", "artificial"), ("thin", "proud"),
    ("deep", "golden"), ("wide", "angry"), ("old", "electronic"),
    ("clean", "romantic"), ("strong", "purple"), ("gentle", "hollow"),
    ("quiet", "sticky"), ("slow", "brilliant"), ("light", "fierce"),
    ("new", "fragile"), ("long", "cruel"), ("young", "sacred"),
    ("hot", "lonely"), ("rich", "dense"), ("safe", "bitter"),
]

# Part B: Multi-template pairs (15 per type, subset of Phase 318)
TEMPLATE_TEST_PAIRS = {
    "same_class": [
        ("apple", "orange"), ("banana", "grape"), ("lemon", "cherry"),
        ("dog", "cat"), ("lion", "tiger"), ("eagle", "hawk"),
        ("car", "bus"), ("train", "subway"), ("bicycle", "motorcycle"),
        ("chair", "table"), ("desk", "cabinet"), ("sofa", "couch"),
        ("rain", "snow"), ("storm", "thunder"), ("wind", "breeze"),
    ],
    "attribute": [
        ("apple", "red"), ("sky", "blue"), ("fire", "hot"),
        ("ice", "cold"), ("silk", "smooth"), ("sandpaper", "rough"),
        ("glass", "clear"), ("stone", "hard"), ("cotton", "soft"),
        ("feather", "light"), ("lead", "heavy"), ("mountain", "tall"),
        ("lemon", "sour"), ("honey", "sweet"), ("chili", "spicy"),
    ],
    "function": [
        ("knife", "cut"), ("pen", "write"), ("car", "drive"),
        ("phone", "call"), ("key", "unlock"), ("cup", "drink"),
        ("lamp", "illuminate"), ("clock", "measure"), ("umbrella", "protect"),
        ("camera", "capture"), ("oven", "bake"), ("shovel", "dig"),
        ("brush", "paint"), ("hammer", "strike"), ("saw", "divide"),
    ],
    "antonym": [
        ("happy", "sad"), ("hot", "cold"), ("big", "small"),
        ("fast", "slow"), ("light", "dark"), ("strong", "weak"),
        ("young", "old"), ("rich", "poor"), ("clean", "dirty"),
        ("hard", "soft"), ("loud", "quiet"), ("open", "closed"),
        ("full", "empty"), ("love", "hate"), ("rise", "fall"),
    ],
}

# Part C: Negation token control (20 adjectives)
NEGATION_CONTROL_ADJS = [
    "happy", "sad", "angry", "excited", "worried",
    "scared", "proud", "tired", "confused", "surprised",
    "safe", "good", "clean", "possible", "bright",
    "strong", "warm", "fair", "clear", "easy",
]

# Morphological negation mapping (un-/in-/im-/dis-)
MORPHOLOGICAL_NEG = {
    "happy": "unhappy", "sad": "unsad", "angry": "unangry",
    "excited": "unexcited", "worried": "unworried",
    "scared": "unscared", "proud": "unproud", "tired": "untired",
    "confused": "unconfused", "surprised": "unsurprised",
    "safe": "unsafe", "good": "ungood", "clean": "unclean",
    "possible": "impossible", "bright": "unbright",
    "strong": "unstrong", "warm": "unwarm", "fair": "unfair",
    "clear": "unclear", "easy": "uneasy",
}


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
        log(f"  flash_attention_2 failed ({e}), falling back to sdpa")
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
    """Extract last-token representations at target layers using hooks."""
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
# ANALYSIS FUNCTIONS (reused from Phase 318)
# =====================================================================

def pca_analysis(D):
    n, d = D.shape
    if n < 2:
        return {"error": "need >=2 directions"}
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
        "n_directions": n,
        "dim_at_50": int(np.searchsorted(cum, 0.50) + 1),
        "dim_at_80": int(np.searchsorted(cum, 0.80) + 1),
        "dim_at_90": int(np.searchsorted(cum, 0.90) + 1),
        "top1_explained": float(ratio[0]),
        "top3_explained": float(cum[min(2, len(cum)-1)]),
    }


def loo_cosine_analysis(D):
    n = D.shape[0]
    if n < 3:
        return {"error": "need >=3 directions"}
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
    return {
        "mean": float(np.mean(cosines)),
        "std": float(np.std(cosines)),
        "min": float(np.min(cosines)),
        "max": float(np.max(cosines)),
        "median": float(np.median(cosines)),
    }


def pairwise_cosine_analysis(D):
    n = D.shape[0]
    if n < 2:
        return {"error": "need >=2 directions"}
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    D_norm = D / norms
    cos_matrix = D_norm @ D_norm.T
    mask = ~np.eye(n, dtype=bool)
    cos_vals = cos_matrix[mask]
    return {
        "mean": float(np.mean(cos_vals)),
        "std": float(np.std(cos_vals)),
        "min": float(np.min(cos_vals)),
        "max": float(np.max(cos_vals)),
        "median": float(np.median(cos_vals)),
    }


def norm_stats(norms):
    if len(norms) == 0:
        return {"error": "empty"}
    return {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
    }


# =====================================================================
# PART A: RANDOM BASELINE ANALYSIS
# =====================================================================

def run_part_a_random_baseline(cache, target_layers):
    """Analyze random pair baselines vs Phase 318 relation types."""
    log("\n--- Part A: Random Baseline Analysis ---")

    groups = {
        "random_mixed": RANDOM_MIXED,
        "random_noun_noun": RANDOM_NOUN_NOUN,
        "random_adj_adj": RANDOM_ADJ_ADJ,
    }

    results = {}

    for group_name, pairs in groups.items():
        group_result = {}
        for li in target_layers:
            d_list, n_list = [], []
            for wordA, wordB in pairs:
                sentA = TEMPLATES["t1"].replace("{w}", wordA)
                sentB = TEMPLATES["t1"].replace("{w}", wordB)
                kA, kB = (sentA, li), (sentB, li)
                if kA in cache and kB in cache:
                    d = cache[kB] - cache[kA]
                    n_list.append(float(np.linalg.norm(d)))
                    d_list.append(d)

            if len(d_list) < 3:
                continue

            D = np.array(d_list)
            n_raw = np.array(n_list)

            # Normalize for angular analysis
            D_norms = np.linalg.norm(D, axis=1, keepdims=True)
            D_norms = np.maximum(D_norms, 1e-10)
            D_unit = D / D_norms

            layer_result = {
                "n_directions": D.shape[0],
                "norm_stats": norm_stats(n_raw),
                "pca_normalized": pca_analysis(D_unit),
                "pca_raw": pca_analysis(D),
                "loo_cosine": loo_cosine_analysis(D),
                "pairwise_cosine": pairwise_cosine_analysis(D),
            }
            group_result[f"L{li}"] = layer_result

            log(f"  {group_name}/L{li}: dim@80={layer_result['pca_normalized']['dim_at_80']}, "
                f"top1={layer_result['pca_normalized']['top1_explained']:.3f}, "
                f"LOO_cos={layer_result['loo_cosine']['mean']:.3f}, "
                f"pair_cos={layer_result['pairwise_cosine']['mean']:.4f}, "
                f"norm_mean={layer_result['norm_stats']['mean']:.2f}")

        results[group_name] = group_result

    return results


# =====================================================================
# PART B: MULTI-TEMPLATE CONSISTENCY
# =====================================================================

def run_part_b_template_control(cache, target_layers):
    """Test if relation directions are consistent across templates."""
    log("\n--- Part B: Multi-Template Consistency ---")

    results = {}

    for rel_type, pairs in TEMPLATE_TEST_PAIRS.items():
        type_result = {}

        # Compute directions for each template
        dirs_by_template = {}
        for tkey, template in TEMPLATES.items():
            d_list = []
            for wordA, wordB in pairs:
                sentA = template.replace("{w}", wordA)
                sentB = template.replace("{w}", wordB)
                # Use deepest layer for cross-template comparison
                deepest_li = target_layers[-1]
                kA, kB = (sentA, deepest_li), (sentB, deepest_li)
                if kA in cache and kB in cache:
                    d = cache[kB] - cache[kA]
                    d_list.append(d)
            if d_list:
                dirs_by_template[tkey] = np.array(d_list)

        # Per-layer subspace analysis for each template
        for tkey, template in TEMPLATES.items():
            tmpl_result = {}
            for li in target_layers:
                d_list, n_list = [], []
                for wordA, wordB in pairs:
                    sentA = template.replace("{w}", wordA)
                    sentB = template.replace("{w}", wordB)
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

            type_result[f"subspace_{tkey}"] = tmpl_result

        # Cross-template direction consistency (deepest layer)
        if len(dirs_by_template) >= 2:
            cross_result = {}
            tkeys = list(dirs_by_template.keys())
            for i in range(len(tkeys)):
                for j in range(i+1, len(tkeys)):
                    tA, tB = tkeys[i], tkeys[j]
                    DA = dirs_by_template[tA]
                    DB = dirs_by_template[tB]
                    # Compute pairwise cosine between same-pair directions across templates
                    n_pairs = min(DA.shape[0], DB.shape[0])
                    cosines = []
                    for k in range(n_pairs):
                        dA = DA[k]
                        dB = DB[k]
                        normA = np.linalg.norm(dA)
                        normB = np.linalg.norm(dB)
                        if normA > 1e-10 and normB > 1e-10:
                            cos = float(np.dot(dA, dB) / (normA * normB))
                            cosines.append(cos)

                    if cosines:
                        cross_result[f"{tA}_vs_{tB}"] = {
                            "mean_cosine": float(np.mean(cosines)),
                            "std_cosine": float(np.std(cosines)),
                            "median_cosine": float(np.median(cosines)),
                            "min_cosine": float(np.min(cosines)),
                            "max_cosine": float(np.max(cosines)),
                            "n_pairs": len(cosines),
                            "frac_positive": float(np.mean([c > 0 for c in cosines])),
                            "frac_above_03": float(np.mean([c > 0.3 for c in cosines])),
                            "frac_above_05": float(np.mean([c > 0.5 for c in cosines])),
                        }
                        cr = cross_result[f"{tA}_vs_{tB}"]
                        log(f"  {rel_type} {tA} vs {tB}: mean_cos={cr['mean_cosine']:.3f}, "
                            f"median={cr['median_cosine']:.3f}, "
                            f"frac>0.3={cr['frac_above_03']:.2f}, frac>0.5={cr['frac_above_05']:.2f}")

            type_result["cross_template"] = cross_result

        results[rel_type] = type_result

    return results


# =====================================================================
# PART C: NEGATION TOKEN CONTROL
# =====================================================================

def run_part_c_negation_token_control(cache, target_layers):
    """Test if negation consistency comes from 'not' token or negation operation."""
    log("\n--- Part C: Negation Token Control ---")

    results = {}

    # Build directions for each negation form
    neg_forms = {
        "not": lambda adj: ("they were very " + adj + " about it", "they were not " + adj + " about it"),
        "never": lambda adj: ("they were very " + adj + " about it", "they were never " + adj + " about it"),
        "barely": lambda adj: ("they were very " + adj + " about it", "they were barely " + adj + " about it"),
        "morphological": lambda adj: ("they were very " + adj + " about it",
                                       "they were " + MORPHOLOGICAL_NEG.get(adj, "un" + adj) + " about it"),
    }

    dirs_by_form = {}
    norms_by_form = {}

    for form_name, make_pair in neg_forms.items():
        d_by_layer = {}
        n_by_layer = {}
        for li in target_layers:
            d_list, n_list = [], []
            for adj in NEGATION_CONTROL_ADJS:
                pos_sent, neg_sent = make_pair(adj)
                kA, kB = (pos_sent, li), (neg_sent, li)
                if kA in cache and kB in cache:
                    d = cache[kB] - cache[kA]
                    n_list.append(float(np.linalg.norm(d)))
                    d_list.append(d)
            if d_list:
                d_by_layer[f"L{li}"] = np.array(d_list)
                n_by_layer[f"L{li}"] = np.array(n_list)

        dirs_by_form[form_name] = d_by_layer
        norms_by_form[form_name] = n_by_layer

    # Per-form subspace analysis
    for form_name in neg_forms:
        form_result = {}
        for li in target_layers:
            lk = f"L{li}"
            if lk not in dirs_by_form[form_name]:
                continue
            D = dirs_by_form[form_name][lk]
            n_raw = norms_by_form[form_name][lk]

            D_norms = np.linalg.norm(D, axis=1, keepdims=True)
            D_norms = np.maximum(D_norms, 1e-10)
            D_unit = D / D_norms

            form_result[lk] = {
                "pca_normalized": pca_analysis(D_unit),
                "loo_cosine": loo_cosine_analysis(D),
                "pairwise_cosine": pairwise_cosine_analysis(D),
                "norm_stats": norm_stats(n_raw),
            }

            log(f"  {form_name}/L{li}: dim@80={form_result[lk]['pca_normalized']['dim_at_80']}, "
                f"LOO_cos={form_result[lk]['loo_cosine']['mean']:.3f}, "
                f"pair_cos={form_result[lk]['pairwise_cosine']['mean']:.4f}, "
                f"norm_mean={form_result[lk]['norm_stats']['mean']:.2f}")

        results[f"subspace_{form_name}"] = form_result

    # Cross-form pairwise consistency (deepest layer)
    deepest_li = target_layers[-1]
    lk = f"L{deepest_li}"
    cross_form = {}

    form_names = list(neg_forms.keys())
    for i in range(len(form_names)):
        for j in range(i+1, len(form_names)):
            fA, fB = form_names[i], form_names[j]
            if lk not in dirs_by_form[fA] or lk not in dirs_by_form[fB]:
                continue
            DA = dirs_by_form[fA][lk]
            DB = dirs_by_form[fB][lk]

            # Same-adjective cosine between forms
            n_pairs = min(DA.shape[0], DB.shape[0])
            cosines = []
            for k in range(n_pairs):
                dA = DA[k]
                dB = DB[k]
                normA = np.linalg.norm(dA)
                normB = np.linalg.norm(dB)
                if normA > 1e-10 and normB > 1e-10:
                    cos = float(np.dot(dA, dB) / (normA * normB))
                    cosines.append(cos)

            if cosines:
                cross_form[f"{fA}_vs_{fB}"] = {
                    "mean_cosine": float(np.mean(cosines)),
                    "std_cosine": float(np.std(cosines)),
                    "median_cosine": float(np.median(cosines)),
                    "frac_positive": float(np.mean([c > 0 for c in cosines])),
                    "frac_above_03": float(np.mean([c > 0.3 for c in cosines])),
                    "frac_above_05": float(np.mean([c > 0.5 for c in cosines])),
                }
                cr = cross_form[f"{fA}_vs_{fB}"]
                log(f"  Negation {fA} vs {fB}: mean_cos={cr['mean_cosine']:.3f}, "
                    f"frac>0.3={cr['frac_above_03']:.2f}, frac>0.5={cr['frac_above_05']:.2f}")

    results["cross_form"] = cross_form

    # Subspace overlap between forms (principal angles)
    subspace_overlap = {}
    for li in target_layers:
        lk = f"L{li}"
        layer_overlap = {}
        for i in range(len(form_names)):
            for j in range(i+1, len(form_names)):
                fA, fB = form_names[i], form_names[j]
                if lk not in dirs_by_form[fA] or lk not in dirs_by_form[fB]:
                    continue
                DA = dirs_by_form[fA][lk]
                DB = dirs_by_form[fB][lk]
                if DA.shape[0] < 3 or DB.shape[0] < 3:
                    continue

                # Compute principal angles
                kA = min(10, DA.shape[0] - 1, DA.shape[1])
                kB = min(10, DB.shape[0] - 1, DB.shape[1])
                Ua, _, _ = np.linalg.svd(DA - DA.mean(axis=0), full_matrices=False)
                Ub, _, _ = np.linalg.svd(DB - DB.mean(axis=0), full_matrices=False)
                Ua = Ua[:, :kA]
                Ub = Ub[:, :kB]
                M = Ua.T @ Ub
                s = np.linalg.svd(M, compute_uv=False)
                s = np.clip(s, 0, 1)
                angles_deg = np.degrees(np.arccos(s))

                key = f"{fA}_vs_{fB}"
                layer_overlap[key] = {
                    "mean_angle": float(np.mean(angles_deg)),
                    "min_angle": float(np.min(angles_deg)),
                    "mean_cosine": float(np.mean(s)),
                }
                log(f"  L{li} Negation {key}: mean_angle={np.mean(angles_deg):.1f}deg, "
                    f"min_angle={np.min(angles_deg):.1f}deg, mean_cos={np.mean(s):.3f}")

        subspace_overlap[lk] = layer_overlap

    results["subspace_overlap"] = subspace_overlap

    return results


# =====================================================================
# COLLECT ALL UNIQUE ITEMS
# =====================================================================

def collect_all_unique_items():
    """Collect all unique text items for all three parts."""
    items = set()

    # Part A: random baselines (template t1 only)
    for group in [RANDOM_MIXED, RANDOM_NOUN_NOUN, RANDOM_ADJ_ADJ]:
        for wordA, wordB in group:
            items.add(TEMPLATES["t1"].replace("{w}", wordA))
            items.add(TEMPLATES["t1"].replace("{w}", wordB))

    # Part B: multi-template
    for rel_type, pairs in TEMPLATE_TEST_PAIRS.items():
        for tkey, template in TEMPLATES.items():
            for wordA, wordB in pairs:
                items.add(template.replace("{w}", wordA))
                items.add(template.replace("{w}", wordB))

    # Part C: negation token control
    for adj in NEGATION_CONTROL_ADJS:
        # positive sentence (shared)
        pos_sent = "they were very " + adj + " about it"
        items.add(pos_sent)
        # 4 negation forms
        items.add("they were not " + adj + " about it")
        items.add("they were never " + adj + " about it")
        items.add("they were barely " + adj + " about it")
        items.add("they were " + MORPHOLOGICAL_NEG.get(adj, "un" + adj) + " about it")

    return sorted(items)


# =====================================================================
# MAIN
# =====================================================================

def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase319_{model_name}.log")

    log(f"=== Phase 319: Baseline Control for {model_name} ===")

    # 1. Load model
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}, class={info.model_class}")
    t_load = time.time() - t0
    log(f"  Load time: {t_load:.1f}s")

    target_layers = get_target_layers(info.n_layers)
    log(f"  Target layers: {target_layers}")

    # 2. Collect unique items
    unique_items = collect_all_unique_items()
    log(f"  Total unique items: {len(unique_items)}")

    # 3. Extract representations (single pass)
    t0 = time.time()
    log("Extracting representations...")
    cache = extract_representations(model, tokenizer, device, unique_items, target_layers, label="All")
    t_extract = time.time() - t0
    log(f"  Extraction time: {t_extract:.1f}s, cached {len(cache)} representations")

    # 4. Run all three parts
    log("\nRunning Part A: Random Baseline...")
    part_a = run_part_a_random_baseline(cache, target_layers)

    log("\nRunning Part B: Template Control...")
    part_b = run_part_b_template_control(cache, target_layers)

    log("\nRunning Part C: Negation Token Control...")
    part_c = run_part_c_negation_token_control(cache, target_layers)

    # 5. Save results
    all_results = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "target_layers": target_layers,
        "n_unique_items": len(unique_items),
        "extraction_time_s": round(t_extract, 1),
        "part_a_random_baseline": part_a,
        "part_b_template_control": part_b,
        "part_c_negation_token_control": part_c,
    }

    out_path = RESULT_DIR / f"{model_name}_phase319.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # 6. Print summary
    print_summary(all_results)

    # 7. Release model
    release_model(model)
    del model, cache
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released.")

    return all_results


def print_summary(results):
    """Print concise summary."""
    log("\n" + "=" * 70)
    log(f"PHASE 319 SUMMARY - {results['model']}")
    log("=" * 70)

    # Part A summary
    log("\n--- Part A: Random Baseline vs Phase 318 (deepest layer) ---")
    log(f"{'Group':<20} {'dim@80':<8} {'top1%':<8} {'LOO_cos':<9} {'pair_cos':<10}")
    for group_name, group_data in results["part_a_random_baseline"].items():
        # Get deepest layer
        deepest = None
        for lk in sorted(group_data.keys(), key=lambda x: int(x.replace("L",""))):
            deepest = group_data[lk]
        if deepest:
            pca = deepest.get("pca_normalized", {})
            loo = deepest.get("loo_cosine", {})
            pair = deepest.get("pairwise_cosine", {})
            log(f"{group_name:<20} {pca.get('dim_at_80','?'):<8} "
                f"{pca.get('top1_explained',0):.3f}   "
                f"{loo.get('mean',0):.3f}   "
                f"{pair.get('mean',0):.4f}")

    log("\n  Phase 318 reference (deepest layer):")
    log(f"  {'same_class':<20} {'18-19':<8} {'0.07-0.11':<8} {'0.43-0.50':<9} {'0.002-0.023':<10}")
    log(f"  {'antonym':<20} {'16-18':<8} {'0.14-0.16':<8} {'0.51-0.62':<9} {'0.008-0.032':<10}")
    log(f"  {'weak_negation':<20} {'14-20':<8} {'0.12-0.20':<8} {'0.67-0.83':<9} {'0.27-0.66':<10}")

    # Part B summary
    log("\n--- Part B: Cross-Template Direction Consistency ---")
    for rel_type, type_data in results["part_b_template_control"].items():
        cross = type_data.get("cross_template", {})
        for key, cr in cross.items():
            log(f"  {rel_type} {key}: mean_cos={cr.get('mean_cosine',0):.3f}, "
                f"median={cr.get('median_cosine',0):.3f}, "
                f"frac>0.3={cr.get('frac_above_03',0):.2f}, frac>0.5={cr.get('frac_above_05',0):.2f}")

    # Part C summary
    log("\n--- Part C: Negation Token Control ---")
    cross = results["part_c_negation_token_control"].get("cross_form", {})
    for key, cr in cross.items():
        log(f"  {key}: mean_cos={cr.get('mean_cosine',0):.3f}, "
            f"frac>0.3={cr.get('frac_above_03',0):.2f}, frac>0.5={cr.get('frac_above_05',0):.2f}")

    overlap = results["part_c_negation_token_control"].get("subspace_overlap", {})
    for lk, layer_data in sorted(overlap.items()):
        for key, od in layer_data.items():
            log(f"  {lk} {key}: mean_angle={od.get('mean_angle',0):.1f}deg, "
                f"mean_cos={od.get('mean_cosine',0):.3f}")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        all_results = {}
        for mn in ["qwen3", "glm4", "deepseek7b"]:
            log(f"\n{'#'*70}")
            log(f"# Starting {mn}")
            log(f"{'#'*70}")
            try:
                r = run_model(mn)
                all_results[mn] = {"status": "ok", "n_layers": r["n_layers"]}
            except Exception as e:
                log(f"ERROR running {mn}: {e}")
                import traceback; traceback.print_exc()
                all_results[mn] = {"status": "error", "error": str(e)}
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(10)

        log(f"\n{'='*70}")
        log("ALL MODELS SUMMARY")
        log(f"{'='*70}")
        for mn, r in all_results.items():
            log(f"  {mn}: {r}")
    else:
        run_model(model_name)
