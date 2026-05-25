"""
Phase 278: Global Language Dynamics Mapping — Objective Phenomena Puzzle
========================================================================

Phase 277 discovered:
- Middle layers NOT rank-1 (var_top1 ≈ 0.19-0.23), U-curve
- Negation is NOT dynamical reversal (pearson_r ≈ 0.999)
- Scalar profiles almost identical across dimensions
- Trajectories strongly diverge (12-356x)

Phase 278 adds 4 blocks of OBJECTIVE measurements:

Block A: Trajectory Bifurcation Tree
  - Pairwise cosine distance matrix at each layer
  - Hierarchical clustering → ARI per layer
  - FIND: at which exact layer do categories separate?

Block B: Context-Dependent Dynamics
  - Same token in 4 grammatical contexts
  - Trajectory deviation at each layer
  - FIND: how much does context change dynamics?

Block C: Multi-Direction Spectrum
  - SVD of increment matrix → top-5 directions at each layer
  - Project each token onto these directions
  - FIND: what fraction of variance in top-1 vs top-3 vs top-5?
  - FIND: do higher-order directions carry dimension information?

Block D: Attractor Basin Radius
  - Perturb h_l at sample layers, measure final deviation
  - 5 magnitudes, 3 random directions
  - FIND: how robust are trajectories to perturbation?
  - FIND: is basin radius layer-dependent?

Usage:
  python tests/glm5/phase278_global_dynamics.py qwen3
  python tests/glm5/phase278_global_dynamics.py glm4
  python tests/glm5/phase278_global_dynamics.py deepseek7b
"""
import sys, os, json, gc, time, warnings
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_utils import MODEL_CONFIGS, get_model_info, get_layers

RESULT_DIR = Path("results/phase278_global_dynamics")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None

def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ===== Token Definitions (expanded from Phase 277) =====

LANGUAGE_DIMENSIONS = {
    "entities_animal": ["dog", "cat", "lion", "eagle", "fish", "snake", "horse", "whale"],
    "entities_tool": ["hammer", "knife", "wheel", "rope", "nail", "saw", "axe", "drill"],
    "entities_place": ["city", "river", "mountain", "forest", "ocean", "desert", "lake", "island"],
    "actions_physical": ["run", "jump", "eat", "build", "cut", "throw", "swim", "climb"],
    "actions_mental": ["think", "believe", "know", "dream", "fear", "hope", "doubt", "remember"],
    "abstractions": ["truth", "justice", "freedom", "beauty", "time", "love", "death", "power"],
    "logic": ["and", "or", "not", "if", "so", "but", "yet", "because"],
    "reference": ["this", "that", "he", "she", "it", "they", "we", "you"],
    "tense": ["go", "went", "going", "gone", "eat", "ate", "eating", "eaten"],
    "negation_pos": ["good", "happy", "alive", "open", "light"],
    "negation_neg": ["bad", "sad", "dead", "closed", "dark"],
}

ALL_TOKENS = []
TOKEN_TO_DIM = {}
DIM_NAMES = list(LANGUAGE_DIMENSIONS.keys())

# Broad categories for bifurcation analysis
BROAD_CATEGORIES = {
    "entities_animal": "entity", "entities_tool": "entity", "entities_place": "entity",
    "actions_physical": "action", "actions_mental": "action",
    "abstractions": "abstract", "logic": "function",
    "reference": "function", "tense": "function",
    "negation_pos": "negation", "negation_neg": "negation",
}

for dim, tokens in LANGUAGE_DIMENSIONS.items():
    for t in tokens:
        ALL_TOKENS.append(t)
        TOKEN_TO_DIM[t] = dim

# Context templates for Block B
CONTEXT_TOKENS = [
    "dog", "cat", "hammer", "city", "run", "think", "truth", "and",
    "he", "go", "went", "good", "bad", "not", "if", "eat", "ate",
    "this", "beauty", "fear",
]

CONTEXT_TEMPLATES = [
    ("The {X} is", "det_pres"),
    ("A {X} was", "indef_past"),
    ("My {X} can", "poss_modal"),
    ("{X}", "bare"),
]

# Negation pairs
NEGATION_PAIRS = [
    ("good", "bad"), ("happy", "sad"), ("alive", "dead"),
    ("open", "closed"), ("light", "dark"),
]

# Tense groups
TENSE_GROUPS = [
    ("go", "went", "going", "gone"),
    ("eat", "ate", "eating", "eaten"),
]


# ===== Model Loading =====

def load_model_bf16(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bfloat16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "eager"
    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
        log_time(f"  flash_attn available, using {attn_impl}")
    except ImportError:
        log_time(f"  flash_attn not available, using {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=attn_impl,
    )
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"{model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device


# ===== Trajectory Extraction =====

def extract_trajectory(model, tokenizer, device, prompt):
    """Extract h_l at ALL layers for a prompt. Returns h_dict[l] = np.array [d_model]."""
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        except Exception as e:
            log_time(f"  WARNING: Forward failed for '{prompt}': {e}")
            return None

    hs = outputs.hidden_states
    h_dict = {}
    for l in range(len(hs)):
        h_dict[l] = hs[l][0, -1, :].float().cpu().numpy()
    return h_dict


def collect_all_trajectories(model, tokenizer, device, model_name):
    """Extract full trajectories for ALL tokens in baseline context."""
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    log_time(f"Collecting trajectories: {len(ALL_TOKENS)} tokens, {n_layers} layers")

    all_h = {}
    all_delta = {}

    t0 = time.time()
    for ti, word in enumerate(ALL_TOKENS):
        prompt = f"The {word} is"
        h_dict = extract_trajectory(model, tokenizer, device, prompt)
        if h_dict is not None:
            all_h[word] = h_dict
            all_delta[word] = {}
            for l in range(n_layers):
                if l + 1 in h_dict and l in h_dict:
                    all_delta[word][l] = h_dict[l + 1] - h_dict[l]

        if (ti + 1) % 15 == 0 or ti == len(ALL_TOKENS) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (ti + 1) * (len(ALL_TOKENS) - ti - 1)
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log_time(f"  Traj progress: {ti+1}/{len(ALL_TOKENS)}, "
                     f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s, GPU={gpu_mem:.2f}GB")

    log_time(f"Trajectories collected: {len(all_h)}/{len(ALL_TOKENS)} tokens")
    return all_h, all_delta, n_layers, d_model


# ===== Block A: Trajectory Bifurcation Tree =====

def block_a_bifurcation(all_h, n_layers, model_name):
    """
    At each layer, cluster tokens and measure ARI vs true categories.
    FIND: at which layer do categories become distinguishable?
    """
    log_time("=== Block A: Trajectory Bifurcation Tree ===")

    valid_tokens = [w for w in ALL_TOKENS if w in all_h]
    n_tokens = len(valid_tokens)

    # True labels
    broad_labels = [BROAD_CATEGORIES.get(TOKEN_TO_DIM[w], "other") for w in valid_tokens]
    fine_labels = [TOKEN_TO_DIM[w] for w in valid_tokens]

    # Unique label sets
    unique_broad = sorted(set(broad_labels))
    unique_fine = sorted(set(fine_labels))
    broad_map = {l: i for i, l in enumerate(unique_broad)}
    fine_map = {l: i for i, l in enumerate(unique_fine)}
    broad_true = [broad_map[l] for l in broad_labels]
    fine_true = [fine_map[l] for l in fine_labels]

    per_layer_ari = {}

    for l in range(n_layers + 1):
        # Build distance matrix
        vecs = np.array([all_h[w][l] for w in valid_tokens if l in all_h[w]])
        tokens_at_layer = [w for w in valid_tokens if l in all_h[w]]

        if len(vecs) < 5:
            continue

        # Normalize for cosine distance
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        vecs_norm = vecs / norms

        # Pairwise cosine distance
        cos_sim = vecs_norm @ vecs_norm.T
        cos_dist = 1.0 - cos_sim
        np.fill_diagonal(cos_dist, 0)
        cos_dist = np.clip(cos_dist, 0, 2)

        # Hierarchical clustering
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            from sklearn.metrics import adjusted_rand_score

            # Convert to condensed form
            condensed = squareform(cos_dist, checks=False)

            # Ward linkage
            Z = linkage(condensed, method='average')

            # Broad ARI: n_broad clusters
            broad_pred = fcluster(Z, t=len(unique_broad), criterion='maxclust')
            ari_broad = adjusted_rand_score(broad_true, broad_pred)

            # Fine ARI: n_fine clusters
            fine_pred = fcluster(Z, t=len(unique_fine), criterion='maxclust')
            ari_fine = adjusted_rand_score(fine_true, fine_pred)

            # Also compute: pairwise within vs between cosine similarity
            within_sims = []
            between_sims = []
            for i in range(len(tokens_at_layer)):
                for j in range(i + 1, len(tokens_at_layer)):
                    di = TOKEN_TO_DIM[tokens_at_layer[i]]
                    dj = TOKEN_TO_DIM[tokens_at_layer[j]]
                    if di == dj:
                        within_sims.append(cos_sim[i, j])
                    else:
                        between_sims.append(cos_sim[i, j])

            per_layer_ari[str(l)] = {
                "ari_broad": float(ari_broad),
                "ari_fine": float(ari_fine),
                "within_sim": float(np.mean(within_sims)) if within_sims else None,
                "between_sim": float(np.mean(between_sims)) if between_sims else None,
                "sim_delta": float(np.mean(within_sims) - np.mean(between_sims))
                           if within_sims and between_sims else None,
            }
        except Exception as e:
            log_time(f"  L{l} clustering failed: {e}")
            continue

    # Find bifurcation layers
    ari_broad_vals = {int(k): v["ari_broad"] for k, v in per_layer_ari.items()}
    ari_fine_vals = {int(k): v["ari_fine"] for k, v in per_layer_ari.items()}
    sim_delta_vals = {int(k): v["sim_delta"] for k, v in per_layer_ari.items() if v["sim_delta"] is not None}

    # First layer where ARI_broad > 0.3
    bifurcation_broad = None
    for l in sorted(ari_broad_vals.keys()):
        if ari_broad_vals[l] > 0.3:
            bifurcation_broad = l
            break

    bifurcation_fine = None
    for l in sorted(ari_fine_vals.keys()):
        if ari_fine_vals[l] > 0.3:
            bifurcation_fine = l
            break

    # Peak ARI layer
    peak_broad_layer = max(ari_broad_vals, key=ari_broad_vals.get) if ari_broad_vals else None
    peak_fine_layer = max(ari_fine_vals, key=ari_fine_vals.get) if ari_fine_vals else None

    results = {
        "model": model_name,
        "n_tokens": n_tokens,
        "n_broad_categories": len(unique_broad),
        "n_fine_categories": len(unique_fine),
        "bifurcation_broad_layer": bifurcation_broad,
        "bifurcation_fine_layer": bifurcation_fine,
        "peak_broad_ari_layer": peak_broad_layer,
        "peak_broad_ari_value": float(ari_broad_vals[peak_broad_layer]) if peak_broad_layer is not None else None,
        "peak_fine_ari_layer": peak_fine_layer,
        "peak_fine_ari_value": float(ari_fine_vals[peak_fine_layer]) if peak_fine_layer is not None else None,
        "per_layer": per_layer_ari,
    }

    out_path = RESULT_DIR / f"{model_name}_block_a_bifurcation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log_time(f"  Bifurcation broad (ARI>0.3): L{bifurcation_broad}")
    log_time(f"  Bifurcation fine (ARI>0.3): L{bifurcation_fine}")
    log_time(f"  Peak broad ARI: L{peak_broad_layer} = {results['peak_broad_ari_value']:.4f}")
    log_time(f"  Peak fine ARI: L{peak_fine_layer} = {results['peak_fine_ari_value']:.4f}")

    # Show key layers
    for l in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]:
        if str(l) in per_layer_ari:
            pl = per_layer_ari[str(l)]
            log_time(f"    L{l}: ari_broad={pl['ari_broad']:.3f}, "
                     f"ari_fine={pl['ari_fine']:.3f}, "
                     f"sim_delta={pl.get('sim_delta', 'N/A')}")

    return results


# ===== Block B: Context-Dependent Dynamics =====

def block_b_context(all_h, n_layers, d_model, model, tokenizer, device, model_name):
    """
    Same token in different contexts. Measure trajectory deviation.
    FIND: how much does context change dynamics at each layer?
    """
    log_time("=== Block B: Context-Dependent Dynamics ===")

    context_results = {}
    t0 = time.time()

    for ci, (template, ctx_name) in enumerate(CONTEXT_TEMPLATES):
        log_time(f"  Context '{ctx_name}': {template}")

        context_h = {}  # word -> layer -> h

        for ti, word in enumerate(CONTEXT_TOKENS):
            prompt = template.replace("{X}", word)
            h_dict = extract_trajectory(model, tokenizer, device, prompt)
            if h_dict is not None:
                context_h[word] = h_dict

            if (ti + 1) % 10 == 0:
                elapsed = time.time() - t0
                log_time(f"    {ctx_name}: {ti+1}/{len(CONTEXT_TOKENS)}, "
                         f"elapsed={elapsed:.1f}s")

        context_results[ctx_name] = context_h

    # Compare trajectories across contexts
    # For each token, compute pairwise cosine distance between contexts at each layer
    cross_context = {}

    for word in CONTEXT_TOKENS:
        word_contexts = {}
        for ctx_name, ctx_h in context_results.items():
            if word in ctx_h:
                word_contexts[ctx_name] = ctx_h[word]

        if len(word_contexts) < 2:
            continue

        ctx_names = list(word_contexts.keys())
        per_layer_dist = {}

        for l in range(n_layers + 1):
            vecs = []
            valid_ctxs = []
            for cn in ctx_names:
                if l in word_contexts[cn]:
                    vecs.append(word_contexts[cn][l])
                    valid_ctxs.append(cn)

            if len(vecs) < 2:
                continue

            vecs = np.array(vecs)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms > 1e-10, norms, 1.0)
            vecs_norm = vecs / norms

            # Pairwise cosine distance
            cos_sim = vecs_norm @ vecs_norm.T
            cos_dist = 1.0 - cos_sim
            np.fill_diagonal(cos_dist, 0)

            # Mean off-diagonal distance = context sensitivity
            mask = ~np.eye(len(vecs), dtype=bool)
            mean_dist = float(np.mean(cos_dist[mask])) if mask.any() else 0.0

            # Also: distance from baseline (det_pres) context
            baseline_idx = valid_ctxs.index("det_pres") if "det_pres" in valid_ctxs else None
            if baseline_idx is not None:
                dists_from_baseline = [cos_dist[baseline_idx, j]
                                      for j in range(len(valid_ctxs)) if j != baseline_idx]
                mean_dist_from_baseline = float(np.mean(dists_from_baseline))
            else:
                mean_dist_from_baseline = None

            per_layer_dist[str(l)] = {
                "mean_cross_context_dist": mean_dist,
                "dist_from_baseline": mean_dist_from_baseline,
                "n_contexts": len(valid_ctxs),
            }

        cross_context[word] = per_layer_dist

    # Aggregate: mean context sensitivity per layer
    layer_sensitivity = {}
    for l in range(n_layers + 1):
        dists = []
        baseline_dists = []
        for word, pld in cross_context.items():
            if str(l) in pld:
                dists.append(pld[str(l)]["mean_cross_context_dist"])
                if pld[str(l)]["dist_from_baseline"] is not None:
                    baseline_dists.append(pld[str(l)]["dist_from_baseline"])

        if dists:
            layer_sensitivity[str(l)] = {
                "mean_sensitivity": float(np.mean(dists)),
                "mean_dist_from_baseline": float(np.mean(baseline_dists)) if baseline_dists else None,
                "n_tokens": len(dists),
            }

    # Per-dimension context sensitivity
    dim_sensitivity = defaultdict(list)
    for word in cross_context:
        dim = TOKEN_TO_DIM.get(word, "unknown")
        for l_key, ld in cross_context[word].items():
            dim_sensitivity[dim].append(ld["mean_cross_context_dist"])

    dim_sensitivity_summary = {}
    for dim, dists in dim_sensitivity.items():
        dim_sensitivity_summary[dim] = {
            "mean_sensitivity": float(np.mean(dists)),
            "n_observations": len(dists),
        }

    results = {
        "model": model_name,
        "n_tokens_tested": len(cross_context),
        "n_contexts": len(CONTEXT_TEMPLATES),
        "layer_sensitivity": layer_sensitivity,
        "dim_sensitivity": dim_sensitivity_summary,
        "per_token": {w: {
            l: d for l, d in pld.items()
        } for w, pld in cross_context.items()},
    }

    out_path = RESULT_DIR / f"{model_name}_block_b_context.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    if layer_sensitivity:
        early_layers = [l for l in range(min(5, n_layers + 1)) if str(l) in layer_sensitivity]
        mid_layers = [l for l in range(n_layers // 2 - 2, n_layers // 2 + 3) if str(l) in layer_sensitivity]
        late_layers = [l for l in range(max(0, n_layers - 4), n_layers + 1) if str(l) in layer_sensitivity]

        early_sens = np.mean([layer_sensitivity[str(l)]["mean_sensitivity"] for l in early_layers]) if early_layers else None
        mid_sens = np.mean([layer_sensitivity[str(l)]["mean_sensitivity"] for l in mid_layers]) if mid_layers else None
        late_sens = np.mean([layer_sensitivity[str(l)]["mean_sensitivity"] for l in late_layers]) if late_layers else None

        log_time(f"  Context sensitivity: early={early_sens:.4f}, mid={mid_sens:.4f}, late={late_sens:.4f}")

        # Peak sensitivity layer
        peak_layer = max(layer_sensitivity.keys(), key=lambda l: layer_sensitivity[l]["mean_sensitivity"])
        log_time(f"  Peak sensitivity at L{peak_layer}: {layer_sensitivity[peak_layer]['mean_sensitivity']:.4f}")

    log_time(f"  Dimension sensitivities:")
    for dim in sorted(dim_sensitivity_summary.keys()):
        log_time(f"    {dim}: {dim_sensitivity_summary[dim]['mean_sensitivity']:.4f}")

    return results


# ===== Block C: Multi-Direction Spectrum =====

def block_c_multidirection(all_delta, n_layers, model_name):
    """
    SVD of increment matrix at each layer.
    Project tokens onto top-5 directions.
    FIND: variance spectrum, direction-dimension correlations.
    """
    log_time("=== Block C: Multi-Direction Spectrum ===")

    valid_tokens = [w for w in ALL_TOKENS if w in all_delta]
    per_layer_spectrum = {}
    direction_dim_correlation = {}

    for l in range(n_layers):
        # Build increment matrix
        deltas = []
        valid_for_layer = []
        for w in valid_tokens:
            if l in all_delta[w]:
                deltas.append(all_delta[w][l])
                valid_for_layer.append(w)

        if len(deltas) < 5:
            continue

        D = np.column_stack(deltas)  # [d_model, N]

        try:
            U, sigma, Vt = np.linalg.svd(D, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        total_var = np.sum(sigma ** 2)
        if total_var < 1e-20:
            continue

        # Cumulative variance explained
        cumvar = np.cumsum(sigma ** 2) / total_var

        # Variance in top-K
        var_topK = {}
        for K in [1, 2, 3, 5, 10]:
            if K <= len(sigma):
                var_topK[str(K)] = float(cumvar[K - 1])

        # Project each token onto top-5 directions
        n_dirs = min(5, U.shape[1])
        projections = U[:, :n_dirs].T @ D  # [n_dirs, N]

        # Correlate each direction's projection with language dimensions
        dim_corrs = {}
        for d_idx in range(n_dirs):
            proj_vals = projections[d_idx, :]  # [N]

            # For each broad category, compute mean projection
            category_means = {}
            for cat_name in ["entity", "action", "abstract", "function", "negation"]:
                cat_tokens_idx = [i for i, w in enumerate(valid_for_layer)
                                  if BROAD_CATEGORIES.get(TOKEN_TO_DIM[w], "") == cat_name]
                if cat_tokens_idx:
                    category_means[cat_name] = float(np.mean(proj_vals[cat_tokens_idx]))

            dim_corrs[f"dir_{d_idx}"] = {
                "category_means": category_means,
                "projection_std": float(np.std(proj_vals)),
                "projection_range": float(np.max(proj_vals) - np.min(proj_vals)),
            }

        # Direction similarity: cosine between top-K direction and dimension-mean direction
        # For each dimension, compute mean delta, then project onto U directions
        dim_mean_deltas = {}
        for dim in DIM_NAMES:
            dim_idx = [i for i, w in enumerate(valid_for_layer) if TOKEN_TO_DIM[w] == dim]
            if len(dim_idx) >= 2:
                dim_mean = np.mean(D[:, dim_idx], axis=1)
                dim_mean_deltas[dim] = dim_mean

        per_layer_spectrum[str(l)] = {
            "var_topK": var_topK,
            "sigma_top10": [float(s) for s in sigma[:10]],
            "direction_correlations": dim_corrs,
            "n_dirs_significant": int(np.sum(sigma ** 2 / total_var > 0.01)),  # directions > 1% variance
        }

    # Aggregate: mean var_topK across layers
    all_var_top1 = [per_layer_spectrum[k]["var_topK"]["1"] for k in per_layer_spectrum if "1" in per_layer_spectrum[k]["var_topK"]]
    all_var_top3 = [per_layer_spectrum[k]["var_topK"]["3"] for k in per_layer_spectrum if "3" in per_layer_spectrum[k]["var_topK"]]
    all_var_top5 = [per_layer_spectrum[k]["var_topK"]["5"] for k in per_layer_spectrum if "5" in per_layer_spectrum[k]["var_topK"]]
    all_n_sig = [per_layer_spectrum[k]["n_dirs_significant"] for k in per_layer_spectrum]

    # Find layer with maximum n_dirs_significant (most multi-directional)
    max_sig_layer = max(per_layer_spectrum.keys(), key=lambda k: per_layer_spectrum[k]["n_dirs_significant"]) if per_layer_spectrum else None

    results = {
        "model": model_name,
        "mean_var_top1": float(np.mean(all_var_top1)) if all_var_top1 else None,
        "mean_var_top3": float(np.mean(all_var_top3)) if all_var_top3 else None,
        "mean_var_top5": float(np.mean(all_var_top5)) if all_var_top5 else None,
        "mean_n_significant_dirs": float(np.mean(all_n_sig)) if all_n_sig else None,
        "max_significant_dirs_layer": int(max_sig_layer) if max_sig_layer else None,
        "max_significant_dirs_value": per_layer_spectrum[max_sig_layer]["n_dirs_significant"] if max_sig_layer else None,
        "per_layer": per_layer_spectrum,
    }

    out_path = RESULT_DIR / f"{model_name}_block_c_multidirection.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log_time(f"  Mean var_top1={results['mean_var_top1']:.4f}, "
             f"var_top3={results['mean_var_top3']:.4f}, "
             f"var_top5={results['mean_var_top5']:.4f}")
    log_time(f"  Mean n_significant_dirs={results['mean_n_significant_dirs']:.1f}")
    log_time(f"  Max multi-dir layer: L{max_sig_layer} "
             f"({results['max_significant_dirs_value']} dirs)" if max_sig_layer else "  No layers")

    # Show key layers
    for l in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
        if str(l) in per_layer_spectrum:
            pl = per_layer_spectrum[str(l)]
            log_time(f"    L{l}: var_top1={pl['var_topK'].get('1', 'N/A'):.3f}, "
                     f"var_top3={pl['var_topK'].get('3', 'N/A'):.3f}, "
                     f"var_top5={pl['var_topK'].get('5', 'N/A'):.3f}, "
                     f"n_sig={pl['n_dirs_significant']}")

    return results


# ===== Block D: Attractor Basin Radius =====

def block_d_basin(all_h, n_layers, d_model, model, tokenizer, device, model_name):
    """
    Perturb hidden states at sample layers, measure final deviation.
    FIND: how robust are trajectories to perturbation?
    FIND: is basin radius layer-dependent?
    """
    log_time("=== Block D: Attractor Basin Radius ===")

    # Representative tokens: 2 per broad category
    basin_tokens = [
        "dog", "city",       # entity
        "run", "think",      # action
        "truth", "time",     # abstract
        "and", "he",         # function
        "good", "bad",       # negation
    ]
    basin_tokens = [t for t in basin_tokens if t in all_h]

    # Sample layers: early, early-mid, mid, late-mid, late
    sample_layers = sorted(set([
        0, n_layers // 5, n_layers // 2,
        4 * n_layers // 5, n_layers - 2
    ]))

    # Perturbation magnitudes (relative to h norm)
    perturb_magnitudes = [0.01, 0.05, 0.1, 0.5, 1.0]
    n_random_dirs = 3

    log_time(f"  Tokens: {len(basin_tokens)}, Layers: {sample_layers}, "
             f"Magnitudes: {perturb_magnitudes}, Dirs: {n_random_dirs}")

    layers_list = get_layers(model)
    basin_data = {}

    t0_total = time.time()

    for word in basin_tokens:
        log_time(f"  Processing '{word}'...")
        word_data = {}

        # Get clean trajectory
        clean_h = all_h[word]

        # Clean final state from output_hidden_states (same method used for all_h)
        if n_layers not in clean_h:
            log_time(f"  WARNING: No clean final state for '{word}', skipping")
            continue
        h_final_clean = clean_h[n_layers]

        for p_layer in sample_layers:
            if p_layer not in clean_h:
                continue

            h_l = clean_h[p_layer]
            h_l_norm = np.linalg.norm(h_l)

            layer_data = {}

            for mag in perturb_magnitudes:
                abs_mag = mag * h_l_norm
                deviations = []

                for dir_idx in range(n_random_dirs):
                    # Random direction
                    np.random.seed(dir_idx * 1000 + int(mag * 100))
                    v = np.random.randn(d_model).astype(np.float32)
                    v = v / np.linalg.norm(v) * abs_mag

                    # Perturb and run forward from layer p_layer
                    final_state = _run_perturbed_forward(
                        model, tokenizer, device, word,
                        p_layer, v, layers_list
                    )

                    if final_state is not None:
                        deviation = float(np.linalg.norm(final_state - h_final_clean))
                        deviations.append(deviation)

                if deviations:
                    layer_data[str(mag)] = {
                        "mean_deviation": float(np.mean(deviations)),
                        "std_deviation": float(np.std(deviations)),
                        "max_deviation": float(np.max(deviations)),
                        "n_trials": len(deviations),
                        "relative_deviation": float(np.mean(deviations) / max(np.linalg.norm(h_final_clean), 1e-10)),
                    }

            # Also measure: clean baseline deviation at final layer
            # (how much does perturbation at layer l affect the final state)
            layer_data["clean_h_norm"] = float(h_l_norm)
            layer_data["final_h_norm"] = float(np.linalg.norm(h_final_clean))

            word_data[str(p_layer)] = layer_data

            # Progress
            elapsed = time.time() - t0_total
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log_time(f"    '{word}' L{p_layer}: "
                     f"dev(0.01)={layer_data.get('0.01', {}).get('relative_deviation', 'N/A')}, "
                     f"dev(1.0)={layer_data.get('1.0', {}).get('relative_deviation', 'N/A')}, "
                     f"elapsed={elapsed:.1f}s, GPU={gpu_mem:.2f}GB")

        basin_data[word] = word_data

    # Compute basin radius estimate:
    # For each token × layer, find the magnitude where relative deviation exceeds 0.1
    basin_radius = {}
    for word in basin_data:
        for l_key in basin_data[word]:
            ld = basin_data[word][l_key]
            # Interpolate to find threshold
            mags = []
            rel_devs = []
            for mag_str in perturb_magnitudes:
                if str(mag_str) in ld and "relative_deviation" in ld[str(mag_str)]:
                    mags.append(mag_str)
                    rel_devs.append(ld[str(mag_str)]["relative_deviation"])

            if len(mags) >= 2:
                # Linear interpolation for threshold 0.1
                threshold = 0.1
                radius = None
                for i in range(len(mags) - 1):
                    if rel_devs[i] < threshold <= rel_devs[i + 1]:
                        # Interpolate
                        frac = (threshold - rel_devs[i]) / max(rel_devs[i + 1] - rel_devs[i], 1e-10)
                        radius = mags[i] + frac * (mags[i + 1] - mags[i])
                        break
                if radius is None and rel_devs[0] >= threshold:
                    radius = 0.0  # Even smallest perturbation causes large deviation
                elif radius is None:
                    radius = max(mags)  # Even largest perturbation is within basin

                if l_key not in basin_radius:
                    basin_radius[l_key] = {}
                basin_radius[l_key][word] = radius

    # Average basin radius per layer
    avg_basin_radius = {}
    for l_key in basin_radius:
        radii = [v for v in basin_radius[l_key].values() if v is not None]
        if radii:
            avg_basin_radius[l_key] = {
                "mean": float(np.mean(radii)),
                "std": float(np.std(radii)),
                "min": float(np.min(radii)),
                "max": float(np.max(radii)),
                "n_tokens": len(radii),
            }

    results = {
        "model": model_name,
        "n_tokens": len(basin_tokens),
        "sample_layers": sample_layers,
        "perturb_magnitudes": perturb_magnitudes,
        "basin_radius_per_layer": avg_basin_radius,
        "basin_data": basin_data,
    }

    out_path = RESULT_DIR / f"{model_name}_block_d_basin.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log_time(f"  Basin radius per layer:")
    for l_key in sorted(avg_basin_radius.keys(), key=lambda x: int(x)):
        br = avg_basin_radius[l_key]
        log_time(f"    L{l_key}: mean={br['mean']:.4f}, "
                 f"range=[{br['min']:.4f}, {br['max']:.4f}]")

    return results


def _run_perturbed_forward(model, tokenizer, device, word, perturb_layer,
                           perturb_vector, layers_list):
    """
    Run forward pass with perturbation at a specific layer.
    Returns final hidden state at last token position (AFTER final norm).
    Uses output_hidden_states=True to get the same final state as extract_trajectory.
    """
    prompt = f"The {word} is"
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    perturb_applied = [False]

    def perturb_hook(module, input, output):
        if not perturb_applied[0]:
            if isinstance(output, tuple):
                h = output[0].clone()
                pv = torch.tensor(perturb_vector, dtype=h.dtype, device=h.device)
                h[0, -1, :] += pv
                perturb_applied[0] = True
                return (h,) + output[1:]
            else:
                h = output.clone()
                pv = torch.tensor(perturb_vector, dtype=h.dtype, device=h.device)
                h[0, -1, :] += pv
                perturb_applied[0] = True
                return h
        return output

    perturb_handle = layers_list[perturb_layer].register_forward_hook(perturb_hook)

    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
            final_state = outputs.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
        except Exception:
            final_state = None

    perturb_handle.remove()

    return final_state


# ===== Main =====

def main():
    global _log_file

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"

    log_path = RESULT_DIR / f"{model_name}_phase278.log"
    _log_file = str(log_path)

    log_time(f"Phase 278: Global Language Dynamics Mapping")
    log_time(f"Model: {model_name}, Tokens: {len(ALL_TOKENS)}")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log_time(f"Model info: n_layers={info.n_layers}, d_model={info.d_model}, "
             f"class={info.model_class}")

    # Collect trajectories (baseline context)
    log_time("=" * 60)
    all_h, all_delta, n_layers, d_model = collect_all_trajectories(
        model, tokenizer, device, model_name
    )

    # Block A: Bifurcation Tree
    log_time("=" * 60)
    results_a = block_a_bifurcation(all_h, n_layers, model_name)

    # Block B: Context Dependence
    log_time("=" * 60)
    results_b = block_b_context(all_h, n_layers, d_model, model, tokenizer, device, model_name)

    # Block C: Multi-Direction Spectrum
    log_time("=" * 60)
    results_c = block_c_multidirection(all_delta, n_layers, model_name)

    # Block D: Basin Radius
    log_time("=" * 60)
    results_d = block_d_basin(all_h, n_layers, d_model, model, tokenizer, device, model_name)

    # Final Summary
    log_time("=" * 60)
    log_time("PHASE 278 OBJECTIVE RESULTS")
    log_time("=" * 60)

    log_time(f"Block A — Bifurcation:")
    log_time(f"  broad bifurcation layer: {results_a['bifurcation_broad_layer']}")
    log_time(f"  fine bifurcation layer: {results_a['bifurcation_fine_layer']}")
    log_time(f"  peak broad ARI: L{results_a['peak_broad_ari_layer']} = {results_a['peak_broad_ari_value']}")

    log_time(f"Block B — Context:")
    ls = results_b.get("layer_sensitivity", {})
    if ls:
        peak_l = max(ls.keys(), key=lambda k: ls[k]["mean_sensitivity"])
        log_time(f"  peak sensitivity layer: L{peak_l} = {ls[peak_l]['mean_sensitivity']:.4f}")

    log_time(f"Block C — Multi-Direction:")
    log_time(f"  mean var_top1={results_c['mean_var_top1']:.4f}")
    log_time(f"  mean var_top3={results_c['mean_var_top3']:.4f}")
    log_time(f"  mean var_top5={results_c['mean_var_top5']:.4f}")
    log_time(f"  mean n_significant_dirs={results_c['mean_n_significant_dirs']:.1f}")

    log_time(f"Block D — Basin Radius:")
    for l_key in sorted(results_d.get("basin_radius_per_layer", {}).keys(), key=lambda x: int(x)):
        br = results_d["basin_radius_per_layer"][l_key]
        log_time(f"  L{l_key}: mean radius={br['mean']:.4f}")

    # Release
    del all_h, all_delta, model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released. Phase 278 complete.")


if __name__ == "__main__":
    main()
