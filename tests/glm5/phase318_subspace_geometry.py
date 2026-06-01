"""
Phase 318: Relation Subspace Geometry Mapping
==============================================

Goal: Map the geometric structure of relation subspaces.
Answer: "Do same-type relation directions form a coherent low-dimensional subspace?"

6 relation types x 30 pairs each:
1. same_class: apple->orange, dog->cat
2. attribute: apple->red, ice->cold
3. function: knife->cut, pen->write
4. antonym: happy->sad, hot->cold
5. regular_negation: "very happy"->"not happy"
6. weak_negation: "very great"->"not great"

Analysis per type x layer:
- PCA: eigenvalues, explained variance, subspace dimension (50/80/90%)
- LOO cosine: alignment of each direction with group subspace
- Pairwise cosine: coherence within type
- Norm distribution

Cross-type analysis:
- Principal angles between type subspaces
- Projection ratios

Usage:
  python tests/glm5/phase318_subspace_geometry.py qwen3
  python tests/glm5/phase318_subspace_geometry.py glm4
  python tests/glm5/phase318_subspace_geometry.py deepseek7b
  python tests/glm5/phase318_subspace_geometry.py all
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

RESULT_DIR = Path("results/phase318_subspace")
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

WORD_TEMPLATE = "the {w} was there"

WORD_RELATIONS = {
    "same_class": [
        ("apple", "orange"), ("banana", "grape"), ("lemon", "cherry"),
        ("peach", "plum"), ("pear", "mango"),
        ("dog", "cat"), ("lion", "tiger"), ("eagle", "hawk"),
        ("whale", "dolphin"), ("rabbit", "squirrel"),
        ("car", "bus"), ("train", "subway"), ("bicycle", "motorcycle"),
        ("airplane", "helicopter"), ("boat", "ship"),
        ("chair", "table"), ("desk", "cabinet"), ("sofa", "couch"),
        ("bed", "mattress"), ("shelf", "bookcase"),
        ("rain", "snow"), ("storm", "thunder"), ("fog", "mist"),
        ("wind", "breeze"), ("frost", "hail"),
        ("red", "blue"), ("green", "yellow"), ("purple", "pink"),
        ("orange", "brown"), ("black", "white"),
    ],
    "attribute": [
        ("apple", "red"), ("banana", "yellow"), ("sky", "blue"),
        ("grass", "green"), ("snow", "white"),
        ("fire", "hot"), ("ice", "cold"), ("summer", "warm"),
        ("desert", "dry"), ("ocean", "vast"),
        ("silk", "smooth"), ("sandpaper", "rough"), ("glass", "clear"),
        ("stone", "hard"), ("cotton", "soft"),
        ("feather", "light"), ("lead", "heavy"), ("mountain", "tall"),
        ("ant", "tiny"), ("elephant", "huge"),
        ("lemon", "sour"), ("honey", "sweet"), ("salt", "salty"),
        ("chili", "spicy"), ("coffee", "bitter"),
        ("ball", "round"), ("sword", "sharp"), ("diamond", "brilliant"),
        ("night", "dark"), ("steel", "strong"),
    ],
    "function": [
        ("knife", "cut"), ("scissors", "snip"), ("pen", "write"),
        ("pencil", "draw"), ("car", "drive"), ("bicycle", "ride"),
        ("phone", "call"), ("radio", "broadcast"), ("key", "unlock"),
        ("lock", "secure"), ("cup", "drink"), ("bowl", "contain"),
        ("lamp", "illuminate"), ("flashlight", "shine"),
        ("clock", "measure"), ("watch", "display"),
        ("umbrella", "protect"), ("shield", "defend"),
        ("camera", "capture"), ("microphone", "record"),
        ("oven", "bake"), ("refrigerator", "cool"),
        ("shovel", "dig"), ("rake", "gather"),
        ("brush", "paint"), ("comb", "arrange"),
        ("hammer", "strike"), ("saw", "divide"),
        ("screwdriver", "tighten"), ("wrench", "loosen"),
    ],
    "antonym": [
        ("happy", "sad"), ("hot", "cold"), ("big", "small"),
        ("tall", "short"), ("fast", "slow"), ("light", "dark"),
        ("strong", "weak"), ("young", "old"), ("rich", "poor"),
        ("clean", "dirty"), ("hard", "soft"), ("loud", "quiet"),
        ("open", "closed"), ("full", "empty"), ("love", "hate"),
        ("win", "lose"), ("rise", "fall"), ("create", "destroy"),
        ("accept", "reject"), ("include", "exclude"),
        ("simple", "complex"), ("calm", "chaotic"),
        ("sharp", "dull"), ("thick", "thin"),
        ("rough", "smooth"), ("dry", "wet"),
        ("safe", "dangerous"), ("beautiful", "ugly"),
        ("brave", "cowardly"), ("narrow", "wide"),
    ],
}

SENTENCE_RELATIONS = {
    "regular_negation": [
        ("they were very happy about it", "they were not happy about it"),
        ("they were very sad about it", "they were not sad about it"),
        ("they were very angry about it", "they were not angry about it"),
        ("they were very excited about it", "they were not excited about it"),
        ("they were very worried about it", "they were not worried about it"),
        ("they were very confused about it", "they were not confused about it"),
        ("they were very surprised about it", "they were not surprised about it"),
        ("they were very tired about it", "they were not tired about it"),
        ("they were very scared about it", "they were not scared about it"),
        ("they were very proud about it", "they were not proud about it"),
        ("the place was very safe", "the place was not safe"),
        ("the result was very good", "the result was not good"),
        ("the room was very clean", "the room was not clean"),
        ("the task was very possible", "the task was not possible"),
        ("the room was very bright", "the room was not bright"),
        ("the man was very strong", "the man was not strong"),
        ("the weather was very warm", "the weather was not warm"),
        ("the door was very open", "the door was not open"),
        ("the situation was very fair", "the situation was not fair"),
        ("the answer was very clear", "the answer was not clear"),
        ("the path was very easy", "the path was not easy"),
        ("the food was very fresh", "the food was not fresh"),
        ("the water was very deep", "the water was not deep"),
        ("the sound was very loud", "the sound was not loud"),
        ("the surface was very smooth", "the surface was not smooth"),
        ("the material was very soft", "the material was not soft"),
        ("the light was very bright", "the light was not bright"),
        ("the wind was very gentle", "the wind was not gentle"),
        ("the price was very cheap", "the price was not cheap"),
        ("the speed was very fast", "the speed was not fast"),
    ],
    "weak_negation": [
        ("the movie was very great", "the movie was not great"),
        ("the food was very terrible", "the food was not terrible"),
        ("the plan was very perfect", "the plan was not perfect"),
        ("the design was very horrible", "the design was not horrible"),
        ("the idea was very amazing", "the idea was not amazing"),
        ("the performance was very awful", "the performance was not awful"),
        ("the book was very excellent", "the book was not excellent"),
        ("the song was very dreadful", "the song was not dreadful"),
        ("the view was very wonderful", "the view was not wonderful"),
        ("the experience was very disastrous", "the experience was not disastrous"),
        ("the result was very fantastic", "the result was not fantastic"),
        ("the outcome was very miserable", "the outcome was not miserable"),
        ("the show was very brilliant", "the show was not brilliant"),
        ("the event was very catastrophic", "the event was not catastrophic"),
        ("the product was very superb", "the product was not superb"),
        ("the service was very atrocious", "the service was not atrocious"),
        ("the building was very magnificent", "the building was not magnificent"),
        ("the decision was very appalling", "the decision was not appalling"),
        ("the achievement was very outstanding", "the achievement was not outstanding"),
        ("the mistake was very horrendous", "the mistake was not horrendous"),
        ("the discovery was very extraordinary", "the discovery was not extraordinary"),
        ("the failure was very abysmal", "the failure was not abysmal"),
        ("the success was very phenomenal", "the success was not phenomenal"),
        ("the novel was very marvelous", "the novel was not marvelous"),
        ("the speech was very remarkable", "the speech was not remarkable"),
        ("the painting was very exceptional", "the painting was not exceptional"),
        ("the concert was very impressive", "the concert was not impressive"),
        ("the theory was very flawless", "the theory was not flawless"),
        ("the garden was very immaculate", "the garden was not immaculate"),
        ("the storm was very devastating", "the storm was not devastating"),
    ],
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


def extract_representations(model, tokenizer, device, items, target_layers):
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

            if (idx + 1) % 20 == 0 or idx == n_items - 1:
                log(f"    Extracted {idx+1}/{n_items} items, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

            if (idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    return cache


def collect_all_unique_items():
    """Collect all unique text items for representation extraction."""
    items = set()
    for rel_type, pairs in WORD_RELATIONS.items():
        for wordA, wordB in pairs:
            items.add(WORD_TEMPLATE.replace("{w}", wordA))
            items.add(WORD_TEMPLATE.replace("{w}", wordB))
    for rel_type, pairs in SENTENCE_RELATIONS.items():
        for sentA, sentB in pairs:
            items.add(sentA)
            items.add(sentB)
    return sorted(items)


# =====================================================================
# DIRECTION COMPUTATION
# =====================================================================

def compute_directions(cache, target_layers):
    """Compute direction vectors d = h(B) - h(A) from cached reps."""
    directions = {}
    norms_raw = {}

    # Word-based
    for rel_type, pairs in WORD_RELATIONS.items():
        dl, nl = {}, {}
        for li in target_layers:
            d_list, n_list = [], []
            for wordA, wordB in pairs:
                sentA = WORD_TEMPLATE.replace("{w}", wordA)
                sentB = WORD_TEMPLATE.replace("{w}", wordB)
                kA, kB = (sentA, li), (sentB, li)
                if kA in cache and kB in cache:
                    d = cache[kB] - cache[kA]
                    n_list.append(float(np.linalg.norm(d)))
                    d_list.append(d)
            if d_list:
                dl[li] = np.array(d_list)
                nl[li] = np.array(n_list)
        directions[rel_type] = dl
        norms_raw[rel_type] = nl

    # Sentence-based
    for rel_type, pairs in SENTENCE_RELATIONS.items():
        dl, nl = {}, {}
        for li in target_layers:
            d_list, n_list = [], []
            for sentA, sentB in pairs:
                kA, kB = (sentA, li), (sentB, li)
                if kA in cache and kB in cache:
                    d = cache[kB] - cache[kA]
                    n_list.append(float(np.linalg.norm(d)))
                    d_list.append(d)
            if d_list:
                dl[li] = np.array(d_list)
                nl[li] = np.array(n_list)
        directions[rel_type] = dl
        norms_raw[rel_type] = nl

    return directions, norms_raw


# =====================================================================
# SUBSPACE ANALYSIS FUNCTIONS
# =====================================================================

def pca_analysis(D, n_components=None):
    """PCA on direction matrix D (n, d_model). Returns variance structure."""
    n, d = D.shape
    if n < 2:
        return {"error": "need >=2 directions"}
    k = min(n_components or n, n - 1, d)
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
        "eigenvalues_top10": [float(x) for x in eig[:10]],
        "explained_ratio_top10": [float(x) for x in ratio[:10]],
        "cumulative_top10": [float(x) for x in cum[:10]],
        "dim_at_50": int(np.searchsorted(cum, 0.50) + 1),
        "dim_at_80": int(np.searchsorted(cum, 0.80) + 1),
        "dim_at_90": int(np.searchsorted(cum, 0.90) + 1),
        "total_variance": float(total),
        "top1_explained": float(ratio[0]),
        "top3_explained": float(cum[min(2, len(cum)-1)]),
    }


def loo_cosine_analysis(D):
    """Leave-One-Out cosine using Gram matrix trick."""
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
    """Mean pairwise cosine between direction pairs."""
    n = D.shape[0]
    if n < 2:
        return {"error": "need >=2 directions"}
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    D_norm = D / norms
    cos_matrix = D_norm @ D_norm.T
    # Exclude diagonal
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
    """Statistics for norm distribution."""
    if len(norms) == 0:
        return {"error": "empty"}
    return {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "median": float(np.median(norms)),
    }


def principal_angles(subA, subB, n_dims=10):
    """
    Principal angles between two subspaces spanned by rows of subA and subB.
    Uses SVD of orthonormal basis products.

    Args:
        subA: (nA, d_model) direction matrix for type A
        subB: (nB, d_model) direction matrix for type B
        n_dims: max number of principal angles to compute

    Returns:
        dict with angles (degrees), mean_angle, min_angle, projection ratios
    """
    # Orthonormal bases via SVD
    kA = min(n_dims, subA.shape[0] - 1, subA.shape[1])
    kB = min(n_dims, subB.shape[0] - 1, subB.shape[1])
    if kA < 1 or kB < 1:
        return {"error": "insufficient dimensions"}

    Ua, _, _ = np.linalg.svd(subA - subA.mean(axis=0), full_matrices=False)
    Ub, _, _ = np.linalg.svd(subB - subB.mean(axis=0), full_matrices=False)
    Ua = Ua[:, :kA]  # (d_model, kA) via transpose
    Ub = Ub[:, :kB]

    # Principal angles: singular values of Ua.T @ Ub
    M = Ua.T @ Ub  # (kA, kB)
    s = np.linalg.svd(M, compute_uv=False)
    # s = cos(principal_angles)
    s = np.clip(s, 0, 1)
    angles_rad = np.arccos(s)
    angles_deg = np.degrees(angles_rad)

    # Projection ratios
    # How much of A's subspace projects onto B's subspace
    proj_AB = np.mean(np.sum((Ub @ Ub.T) @ Ua, axis=0) ** 2) / np.mean(np.sum(Ua ** 2, axis=0))
    proj_BA = np.mean(np.sum((Ua @ Ua.T) @ Ub, axis=0) ** 2) / np.mean(np.sum(Ub ** 2, axis=0))

    return {
        "angles_deg": [float(x) for x in angles_deg[:10]],
        "mean_angle": float(np.mean(angles_deg)),
        "min_angle": float(np.min(angles_deg)),
        "max_angle": float(np.max(angles_deg)),
        "cosines": [float(x) for x in s[:10]],
        "mean_cosine": float(np.mean(s)),
        "proj_AB": float(min(proj_AB, 1.0)) if not np.isnan(proj_AB) else 0.0,
        "proj_BA": float(min(proj_BA, 1.0)) if not np.isnan(proj_BA) else 0.0,
    }


# =====================================================================
# FULL ANALYSIS PIPELINE
# =====================================================================

def run_full_analysis(directions, norms_raw, target_layers):
    """Run all subspace analyses for one model."""

    all_types = list(WORD_RELATIONS.keys()) + list(SENTENCE_RELATIONS.keys())
    results = {"per_type": {}, "cross_type": {}}

    # --- Per-type analysis ---
    log("Running per-type subspace analysis...")
    for rel_type in all_types:
        type_result = {}
        for li in target_layers:
            if rel_type not in directions or li not in directions[rel_type]:
                continue
            D = directions[rel_type][li]
            n_raw = norms_raw[rel_type][li]

            # Normalize directions to unit vectors for angular analysis
            D_norms = np.linalg.norm(D, axis=1, keepdims=True)
            D_norms = np.maximum(D_norms, 1e-10)
            D_unit = D / D_norms

            layer_result = {
                "n_directions": D.shape[0],
                "norm_stats": norm_stats(n_raw),
                "pca_raw": pca_analysis(D),
                "pca_normalized": pca_analysis(D_unit),
                "loo_cosine": loo_cosine_analysis(D),
                "pairwise_cosine": pairwise_cosine_analysis(D),
            }
            type_result[f"L{li}"] = layer_result

            log(f"  {rel_type}/L{li}: dim@80={layer_result['pca_normalized']['dim_at_80']}, "
                f"top1={layer_result['pca_normalized']['top1_explained']:.3f}, "
                f"LOO_cos={layer_result['loo_cosine']['mean']:.3f}, "
                f"pair_cos={layer_result['pairwise_cosine']['mean']:.3f}, "
                f"norm_mean={layer_result['norm_stats']['mean']:.2f}")

        results["per_type"][rel_type] = type_result

    # --- Cross-type analysis ---
    log("Running cross-type subspace analysis...")
    type_names = list(directions.keys())
    for li in target_layers:
        layer_cross = {}
        for i, tA in enumerate(type_names):
            for j, tB in enumerate(type_names):
                if j <= i:
                    continue
                if li not in directions[tA] or li not in directions[tB]:
                    continue
                DA = directions[tA][li]
                DB = directions[tB][li]
                if DA.shape[0] < 3 or DB.shape[0] < 3:
                    continue

                pa = principal_angles(DA, DB, n_dims=10)
                key = f"{tA}_vs_{tB}"
                layer_cross[key] = pa

                log(f"  L{li} {key}: mean_angle={pa.get('mean_angle', 0):.1f}deg, "
                    f"min_angle={pa.get('min_angle', 0):.1f}deg, "
                    f"mean_cos={pa.get('mean_cosine', 0):.3f}, "
                    f"proj_AB={pa.get('proj_AB', 0):.3f}")

        results["cross_type"][f"L{li}"] = layer_cross

    return results


# =====================================================================
# MAIN
# =====================================================================

def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase318_{model_name}.log")

    log(f"=== Phase 318: Subspace Geometry for {model_name} ===")

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

    # 3. Extract representations
    t0 = time.time()
    log("Extracting representations...")
    cache = extract_representations(model, tokenizer, device, unique_items, target_layers)
    t_extract = time.time() - t0
    log(f"  Extraction time: {t_extract:.1f}s, cached {len(cache)} representations")

    # 4. Compute directions
    log("Computing direction vectors...")
    directions, norms_raw = compute_directions(cache, target_layers)
    for rel_type in directions:
        for li in target_layers:
            if li in directions[rel_type]:
                log(f"  {rel_type}/L{li}: {directions[rel_type][li].shape[0]} directions, "
                    f"d_model={directions[rel_type][li].shape[1]}")

    # 5. Run full analysis
    log("Running subspace analysis...")
    t0 = time.time()
    analysis = run_full_analysis(directions, norms_raw, target_layers)
    t_analysis = time.time() - t0
    log(f"  Analysis time: {t_analysis:.1f}s")

    # 6. Save results
    all_results = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "target_layers": target_layers,
        "n_unique_items": len(unique_items),
        "extraction_time_s": round(t_extract, 1),
        "analysis_time_s": round(t_analysis, 1),
        "analysis": analysis,
    }

    out_path = RESULT_DIR / f"{model_name}_phase318.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # 7. Print summary
    print_summary(all_results)

    # 8. Release model
    release_model(model)
    del model, cache, directions, norms_raw
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released.")

    return all_results


def print_summary(results):
    """Print a concise summary of the subspace analysis."""
    log("\n" + "=" * 70)
    log(f"PHASE 318 SUMMARY - {results['model']}")
    log("=" * 70)

    analysis = results["analysis"]

    # Per-type summary table
    log("\n--- Per-type Subspace Properties (normalized PCA) ---")
    header = f"{'Type':<20} {'Layer':<6} {'dim@50':<8} {'dim@80':<8} {'dim@90':<8} {'top1%':<8} {'top3%':<8} {'LOO_cos':<9} {'pair_cos':<9} {'norm_mean':<10}"
    log(header)

    for rel_type, type_data in analysis["per_type"].items():
        for layer_key, lr in sorted(type_data.items()):
            pca = lr.get("pca_normalized", {})
            loo = lr.get("loo_cosine", {})
            pair = lr.get("pairwise_cosine", {})
            ns = lr.get("norm_stats", {})
            if "error" in pca:
                continue
            log(f"{rel_type:<20} {layer_key:<6} {pca.get('dim_at_50','?'):<8} "
                f"{pca.get('dim_at_80','?'):<8} {pca.get('dim_at_90','?'):<8} "
                f"{pca.get('top1_explained',0):.3f}   {pca.get('top3_explained',0):.3f}   "
                f"{loo.get('mean',0):.3f}   {pair.get('mean',0):.3f}   "
                f"{ns.get('mean',0):.2f}")

    # Cross-type summary
    log("\n--- Cross-type Principal Angles (most aligned pairs) ---")
    for layer_key, cross_data in sorted(analysis["cross_type"].items()):
        # Find top-3 most aligned pairs (smallest mean angle)
        pairs = [(k, v) for k, v in cross_data.items() if "mean_angle" in v]
        pairs.sort(key=lambda x: x[1]["mean_angle"])
        log(f"\n  {layer_key}:")
        for k, v in pairs[:5]:
            log(f"    {k}: mean_angle={v['mean_angle']:.1f}deg, min_angle={v['min_angle']:.1f}deg, "
                f"mean_cos={v['mean_cosine']:.3f}, proj_AB={v['proj_AB']:.3f}")
        # Also show most orthogonal pairs
        log(f"  Most orthogonal:")
        for k, v in pairs[-3:]:
            log(f"    {k}: mean_angle={v['mean_angle']:.1f}deg, min_angle={v['min_angle']:.1f}deg, "
                f"mean_cos={v['mean_cosine']:.3f}")


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

        # Run cross-model comparison
        log("\nRunning cross-model comparison...")
        try:
            run_cross_model_comparison()
        except Exception as e:
            log(f"Cross-model comparison failed: {e}")
            import traceback; traceback.print_exc()
    else:
        run_model(model_name)


def run_cross_model_comparison():
    """Compare subspace structures across models."""
    models = ["qwen3", "glm4", "deepseek7b"]
    data = {}
    for mn in models:
        path = RESULT_DIR / f"{mn}_phase318.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data[mn] = json.load(f)

    if len(data) < 2:
        log("Need at least 2 models for comparison, skipping")
        return

    log(f"\nCross-model comparison: {list(data.keys())}")

    # Compare subspace dimensions across models
    log("\n--- Subspace Dimension Comparison (dim@80, normalized) ---")
    all_types = set()
    for mn in data:
        all_types.update(data[mn]["analysis"]["per_type"].keys())

    for rel_type in sorted(all_types):
        parts = []
        for mn in data:
            type_data = data[mn]["analysis"]["per_type"].get(rel_type, {})
            # Get deepest layer
            deepest = None
            for lk in sorted(type_data.keys(), key=lambda x: int(x.replace("L",""))):
                deepest = type_data[lk]
            if deepest:
                pca = deepest.get("pca_normalized", {})
                dim80 = pca.get("dim_at_80", "?")
                parts.append(f"{mn}:dim@80={dim80}")
        log(f"  {rel_type}: {', '.join(parts)}")

    # Compare norm ranges across models
    log("\n--- Direction Norm Comparison (deepest layer) ---")
    for rel_type in sorted(all_types):
        parts = []
        for mn in data:
            type_data = data[mn]["analysis"]["per_type"].get(rel_type, {})
            deepest = None
            for lk in sorted(type_data.keys(), key=lambda x: int(x.replace("L",""))):
                deepest = type_data[lk]
            if deepest:
                ns = deepest.get("norm_stats", {})
                parts.append(f"{mn}:norm={ns.get('mean',0):.1f}")
        log(f"  {rel_type}: {', '.join(parts)}")

    log("\nCross-model comparison complete.")
