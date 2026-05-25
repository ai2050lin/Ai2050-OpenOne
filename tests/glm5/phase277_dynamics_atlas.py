"""
Phase 277: Conditional Dynamics Atlas — Full Trajectory + Expanded Semantics
=============================================================================

Phase 276 discovery: Jacobian is rank-1 with universal spectral structure but
concept-specific direction. This leads to a CRITICAL PREDICTION:

    If J_l ≈ σ_l * u_l * v_l^T  (rank-1),
    then δ_l(x) = h_{l+1}(x) - h_l(x) ≈ c_l(x) * u_l

i.e., ALL tokens share the SAME update direction at each layer,
differing only in the scalar c_l(x).

This is testable: compute D_l = [δ_l(x_1), ..., δ_l(x_N)] and check
if it's rank-1 (top-1 singular value explains >90% variance).

Expanded semantic coverage: 59 tokens across 11 language dimensions:
  entities (animal/tool/place), actions (physical/mental),
  abstractions, logic, tense, reference, negation (pos/neg)

Experiments:
A. Universal Direction Test — Is D_l rank-1 across tokens?
B. Scalar Profile Atlas — Token-conditioned scalar c_l(x)
C. Trajectory Topology — Curvature, divergence, phase transitions
D. DMD Mode Analysis — Global dynamical modes per token
E. Language Dimension Signatures — Negation, logic, tense effects

Usage:
  python tests/glm5/phase277_dynamics_atlas.py qwen3
  python tests/glm5/phase277_dynamics_atlas.py glm4
  python tests/glm5/phase277_dynamics_atlas.py deepseek7b
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

from model_utils import MODEL_CONFIGS, get_model_info

RESULT_DIR = Path("results/phase277_dynamics_atlas")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None

def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ===== Language Dimensions =====

LANGUAGE_DIMENSIONS = {
    "entities_animal": ["dog", "cat", "lion", "eagle", "fish"],
    "entities_tool": ["hammer", "knife", "wheel", "rope", "nail"],
    "entities_place": ["city", "river", "mountain", "forest", "ocean"],
    "actions_physical": ["run", "jump", "eat", "build", "cut"],
    "actions_mental": ["think", "believe", "know", "dream", "fear"],
    "abstractions": ["truth", "justice", "freedom", "beauty", "time"],
    "logic": ["and", "or", "not", "if", "so"],
    "tense": ["go", "went", "going", "gone"],
    "reference": ["this", "that", "he", "she"],
    "negation_pos": ["good", "happy", "alive", "open", "light"],
    "negation_neg": ["bad", "sad", "dead", "closed", "dark"],
}

ALL_TOKENS = []
TOKEN_TO_DIM = {}
DIM_NAMES = list(LANGUAGE_DIMENSIONS.keys())

for dim, tokens in LANGUAGE_DIMENSIONS.items():
    for t in tokens:
        ALL_TOKENS.append(t)
        TOKEN_TO_DIM[t] = dim

NEGATION_PAIRS = [
    ("good", "bad"), ("happy", "sad"), ("alive", "dead"),
    ("open", "closed"), ("light", "dark"),
]

TENSE_GROUPS = [
    ("go", "went", "going", "gone"),
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

    # Try flash_attention_2 first, fallback to eager
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

def extract_trajectory(model, tokenizer, device, word, n_layers):
    """
    Extract h_l at ALL layers for a single token.
    Returns: h_dict[l] = np.array [d_model] for l = 0, 1, ..., n_layers
    """
    prompt = f"The {word} is"
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        except Exception as e:
            log_time(f"  WARNING: Forward failed for '{word}': {e}")
            return None

    hs = outputs.hidden_states
    # hs[0] = embedding output, hs[1] = after layer 0, ..., hs[n_layers] = after layer n_layers-1
    h_dict = {}
    for l in range(len(hs)):
        h_dict[l] = hs[l][0, -1, :].float().cpu().numpy()

    return h_dict


def collect_all_trajectories(model, tokenizer, device, model_name):
    """Extract full trajectories for ALL tokens."""
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    log_time(f"Collecting trajectories: {len(ALL_TOKENS)} tokens, {n_layers} layers, d={d_model}")

    all_h = {}  # word -> layer -> h

    t0 = time.time()
    for ti, word in enumerate(ALL_TOKENS):
        h_dict = extract_trajectory(model, tokenizer, device, word, n_layers)
        if h_dict is not None:
            all_h[word] = h_dict
        else:
            log_time(f"  SKIP '{word}' — forward failed")

        if (ti + 1) % 10 == 0 or ti == len(ALL_TOKENS) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (ti + 1) * (len(ALL_TOKENS) - ti - 1)
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log_time(f"  Progress: {ti+1}/{len(ALL_TOKENS)}, "
                     f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s, GPU={gpu_mem:.2f}GB")

    # Compute increments δ_l = h_{l+1} - h_l
    all_delta = {}  # word -> layer -> δ
    for word in all_h:
        all_delta[word] = {}
        for l in range(n_layers):
            if l + 1 in all_h[word] and l in all_h[word]:
                all_delta[word][l] = all_h[word][l + 1] - all_h[word][l]

    # Save trajectory stats
    traj_stats = {}
    for word in all_h:
        traj_stats[word] = {
            "n_layers": len(all_h[word]),
            "h_norms": {str(l): float(np.linalg.norm(all_h[word][l]))
                       for l in sorted(all_h[word].keys())},
        }
    out_path = RESULT_DIR / f"{model_name}_trajectory_stats.json"
    with open(out_path, "w") as f:
        json.dump(traj_stats, f, indent=2)

    log_time(f"Trajectories collected: {len(all_h)}/{len(ALL_TOKENS)} tokens")
    return all_h, all_delta, n_layers, d_model


# ===== Exp A: Universal Direction Test =====

def test_universal_direction(all_delta, n_layers, model_name):
    """
    For each layer l, compute D_l = [δ_l(x_1), ..., δ_l(x_N)]
    and test if it's rank-1 (universal update direction).
    """
    log_time("=== Exp A: Universal Direction Test ===")

    valid_tokens = [w for w in ALL_TOKENS if w in all_delta]
    per_layer = {}
    universal_dirs = {}  # layer -> u_l (top-1 left singular vector)

    for l in range(n_layers):
        # Build D_l matrix: [d_model, N_tokens]
        deltas = []
        valid_for_layer = []
        for w in valid_tokens:
            if l in all_delta[w]:
                deltas.append(all_delta[w][l])
                valid_for_layer.append(w)

        if len(deltas) < 3:
            continue

        D = np.column_stack(deltas)  # [d_model, N]

        try:
            U, sigma, Vt = np.linalg.svd(D, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        total_var = np.sum(sigma ** 2)
        if total_var < 1e-20:
            continue

        # Variance explained by top-K directions
        var_top1 = sigma[0] ** 2 / total_var
        var_top3 = np.sum(sigma[:min(3, len(sigma))] ** 2) / total_var
        var_top5 = np.sum(sigma[:min(5, len(sigma))] ** 2) / total_var

        # Pairwise cosine similarity of δ vectors
        cos_matrix = np.zeros((len(deltas), len(deltas)))
        for i in range(len(deltas)):
            for j in range(len(deltas)):
                ni = np.linalg.norm(deltas[i])
                nj = np.linalg.norm(deltas[j])
                if ni > 1e-10 and nj > 1e-10:
                    cos_matrix[i, j] = float(np.dot(deltas[i], deltas[j]) / (ni * nj))

        # Mean off-diagonal cosine
        mask = ~np.eye(len(deltas), dtype=bool)
        mean_cos = float(np.mean(cos_matrix[mask])) if mask.any() else 0.0

        # Project each δ onto u_l (top-1 direction)
        u_l = U[:, 0]
        projections = u_l @ D  # [N]
        signs = np.sign(projections)
        neg_fraction = float(np.sum(signs < 0) / len(signs)) if len(signs) > 0 else 0.0

        per_layer[str(l)] = {
            "n_tokens": len(deltas),
            "var_top1": float(var_top1),
            "var_top3": float(var_top3),
            "var_top5": float(var_top5),
            "mean_pairwise_cos": mean_cos,
            "top1_sv": float(sigma[0]),
            "top2_sv": float(sigma[1]) if len(sigma) > 1 else 0.0,
            "neg_fraction": neg_fraction,
            "sigma_spectrum": [float(s) for s in sigma[:10]],
        }
        universal_dirs[l] = u_l

    # Summary
    var_top1_vals = [per_layer[k]["var_top1"] for k in per_layer]
    mean_cos_vals = [per_layer[k]["mean_pairwise_cos"] for k in per_layer]

    summary = {
        "model": model_name,
        "n_layers": n_layers,
        "global_var_top1_mean": float(np.mean(var_top1_vals)) if var_top1_vals else None,
        "global_var_top1_std": float(np.std(var_top1_vals)) if var_top1_vals else None,
        "global_mean_cos_mean": float(np.mean(mean_cos_vals)) if mean_cos_vals else None,
        "per_layer": per_layer,
    }

    out_path = RESULT_DIR / f"{model_name}_exp_a_universal_dir.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    log_time(f"  Global var_top1: mean={summary['global_var_top1_mean']:.4f} "
             f"± {summary['global_var_top1_std']:.4f}")
    log_time(f"  Global mean pairwise cos: {summary['global_mean_cos_mean']:.4f}")
    log_time(f"  Per-layer var_top1 range: "
             f"[{min(var_top1_vals):.4f}, {max(var_top1_vals):.4f}]")

    # Show early/mid/late layers
    for l_key in sorted(per_layer.keys(), key=lambda x: int(x)):
        l = int(l_key)
        if l in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            pl = per_layer[l_key]
            log_time(f"    L{l}: var_top1={pl['var_top1']:.4f}, "
                     f"var_top3={pl['var_top3']:.4f}, "
                     f"mean_cos={pl['mean_pairwise_cos']:.4f}, "
                     f"neg_frac={pl['neg_fraction']:.3f}")

    return summary, universal_dirs


# ===== Exp B: Scalar Profile Atlas =====

def build_scalar_atlas(all_delta, universal_dirs, n_layers, model_name):
    """
    Project each δ_l(x) onto u_l to get c_l(x).
    Build scalar profile matrix C[l, x].
    """
    log_time("=== Exp B: Scalar Profile Atlas ===")

    valid_tokens = [w for w in ALL_TOKENS if w in all_delta]
    layers_with_dir = sorted(universal_dirs.keys())

    # Build scalar profile matrix
    scalar_matrix = np.zeros((len(layers_with_dir), len(valid_tokens)))
    for li, l in enumerate(layers_with_dir):
        u_l = universal_dirs[l]
        for ti, w in enumerate(valid_tokens):
            if l in all_delta[w]:
                c = float(np.dot(u_l, all_delta[w][l]))
                scalar_matrix[li, ti] = c

    # Correlation matrix of scalar profiles (token × token)
    # Each token's "fingerprint" is its column in scalar_matrix
    if scalar_matrix.shape[1] > 1 and scalar_matrix.shape[0] > 1:
        from scipy.stats import spearmanr
        corr_matrix, _ = spearmanr(scalar_matrix, axis=0)
        if np.isscalar(corr_matrix):
            corr_matrix = np.array([[corr_matrix]])
    else:
        corr_matrix = np.eye(len(valid_tokens))

    # Within-dimension vs between-dimension scalar correlation
    within_corrs = []
    between_corrs = []

    for i, wi in enumerate(valid_tokens):
        for j, wj in enumerate(valid_tokens):
            if i >= j:
                continue
            di, dj = TOKEN_TO_DIM[wi], TOKEN_TO_DIM[wj]
            if i < corr_matrix.shape[0] and j < corr_matrix.shape[1]:
                c = corr_matrix[i, j]
                if di == dj:
                    within_corrs.append(c)
                else:
                    between_corrs.append(c)

    # Cluster by scalar profile
    clustering_result = {}
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        from sklearn.preprocessing import StandardScaler

        # True labels: group by broad category
        broad_cats = {
            "entities_animal": 0, "entities_tool": 0, "entities_place": 0,
            "actions_physical": 1, "actions_mental": 1,
            "abstractions": 2, "logic": 3, "tense": 4,
            "reference": 5, "negation_pos": 6, "negation_neg": 6,
        }
        true_labels = [broad_cats.get(TOKEN_TO_DIM[w], 0) for w in valid_tokens]

        # Fine labels: exact dimension
        dim_to_idx = {d: i for i, d in enumerate(DIM_NAMES)}
        fine_labels = [dim_to_idx[TOKEN_TO_DIM[w]] for w in valid_tokens]

        X = scalar_matrix.T  # [N_tokens, N_layers]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Broad clustering
        n_broad = len(set(broad_cats.values()))
        km = KMeans(n_clusters=n_broad, random_state=42, n_init=10)
        broad_pred = km.fit_predict(X_scaled)
        ari_broad = adjusted_rand_score(true_labels, broad_pred)

        # Fine clustering
        n_fine = len(set(fine_labels))
        km2 = KMeans(n_clusters=n_fine, random_state=42, n_init=10)
        fine_pred = km2.fit_predict(X_scaled)
        ari_fine = adjusted_rand_score(fine_labels, fine_pred)

        clustering_result = {
            "broad_ari": float(ari_broad),
            "fine_ari": float(ari_fine),
            "broad_labels": [int(l) for l in broad_pred],
            "fine_labels": [int(l) for l in fine_pred],
            "true_broad": true_labels,
            "true_fine": fine_labels,
        }
        log_time(f"  Scalar profile clustering: broad ARI={ari_broad:.4f}, "
                 f"fine ARI={ari_fine:.4f}")
    except Exception as e:
        log_time(f"  Clustering failed: {e}")

    # Save scalar profile matrix
    profile_data = {
        "tokens": valid_tokens,
        "token_dims": [TOKEN_TO_DIM[w] for w in valid_tokens],
        "layers": layers_with_dir,
        "scalar_matrix_shape": list(scalar_matrix.shape),
        "within_corr_mean": float(np.mean(within_corrs)) if within_corrs else None,
        "between_corr_mean": float(np.mean(between_corrs)) if between_corrs else None,
        "delta_corr": float(np.mean(within_corrs) - np.mean(between_corrs))
                      if within_corrs and between_corrs else None,
        "clustering": clustering_result,
    }

    # Save full scalar matrix as numpy
    np.save(RESULT_DIR / f"{model_name}_scalar_matrix.npy", scalar_matrix)

    out_path = RESULT_DIR / f"{model_name}_exp_b_scalar_atlas.json"
    with open(out_path, "w") as f:
        json.dump(profile_data, f, indent=2)

    log_time(f"  Within-dim scalar corr: {profile_data['within_corr_mean']}")
    log_time(f"  Between-dim scalar corr: {profile_data['between_corr_mean']}")
    log_time(f"  Delta: {profile_data['delta_corr']}")

    return profile_data, scalar_matrix


# ===== Exp C: Trajectory Topology =====

def analyze_trajectory_topology(all_h, all_delta, n_layers, model_name):
    """
    Analyze trajectory curvature, divergence, and phase transitions.
    """
    log_time("=== Exp C: Trajectory Topology ===")

    valid_tokens = [w for w in ALL_TOKENS if w in all_h and w in all_delta]

    # 1. Curvature: angle between consecutive increments
    curvature = {}  # word -> layer -> angle (degrees)
    for w in valid_tokens:
        curvature[w] = {}
        for l in range(n_layers - 1):
            if l in all_delta[w] and l + 1 in all_delta[w]:
                d1 = all_delta[w][l]
                d2 = all_delta[w][l + 1]
                n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_angle = np.clip(np.dot(d1, d2) / (n1 * n2), -1, 1)
                    curvature[w][l] = float(np.degrees(np.arccos(cos_angle)))

    # Average curvature per layer
    avg_curvature = {}
    max_curvature_layer = 0
    max_curvature_val = 0
    for l in range(n_layers - 1):
        vals = [curvature[w][l] for w in valid_tokens if l in curvature[w]]
        if vals:
            avg_curvature[str(l)] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "max": float(np.max(vals)),
            }
            if float(np.mean(vals)) > max_curvature_val:
                max_curvature_val = float(np.mean(vals))
                max_curvature_layer = l

    log_time(f"  Max curvature at L{max_curvature_layer}: {max_curvature_val:.1f}°")

    # 2. Pairwise hidden state distance across layers
    # Sample 15 token pairs for efficiency
    sample_pairs = []
    # Within-dimension pairs
    for dim in DIM_NAMES:
        dim_tokens = [w for w in LANGUAGE_DIMENSIONS[dim] if w in valid_tokens]
        if len(dim_tokens) >= 2:
            sample_pairs.append((dim_tokens[0], dim_tokens[1], "within"))
    # Between-dimension pairs
    dims_with_tokens = [d for d in DIM_NAMES
                        if any(w in valid_tokens for w in LANGUAGE_DIMENSIONS[d])]
    for i in range(min(5, len(dims_with_tokens) - 1)):
        d1, d2 = dims_with_tokens[i], dims_with_tokens[i + 1]
        t1 = [w for w in LANGUAGE_DIMENSIONS[d1] if w in valid_tokens][0]
        t2 = [w for w in LANGUAGE_DIMENSIONS[d2] if w in valid_tokens][0]
        sample_pairs.append((t1, t2, "between"))

    distance_profiles = {"within": defaultdict(list), "between": defaultdict(list)}
    for wa, wb, pair_type in sample_pairs:
        for l in range(n_layers + 1):
            if l in all_h[wa] and l in all_h[wb]:
                dist = float(np.linalg.norm(all_h[wa][l] - all_h[wb][l]))
                distance_profiles[pair_type][str(l)].append(dist)

    # Average distance profiles
    avg_distance = {}
    for ptype in ["within", "between"]:
        avg_distance[ptype] = {}
        for l_key in distance_profiles[ptype]:
            vals = distance_profiles[ptype][l_key]
            if vals:
                avg_distance[ptype][l_key] = float(np.mean(vals))

    # 3. Layer divergence: does distance increase or decrease with depth?
    early_layers = [str(l) for l in range(min(5, n_layers)) if str(l) in avg_distance.get("within", {})]
    late_layers = [str(l) for l in range(max(0, n_layers - 5), n_layers + 1) if str(l) in avg_distance.get("within", {})]

    early_dist_within = np.mean([avg_distance["within"][l] for l in early_layers]) if early_layers else 0
    late_dist_within = np.mean([avg_distance["within"][l] for l in late_layers]) if late_layers else 0
    early_dist_between = np.mean([avg_distance["between"][l] for l in early_layers]) if early_layers else 0
    late_dist_between = np.mean([avg_distance["between"][l] for l in late_layers]) if late_layers else 0

    results = {
        "model": model_name,
        "curvature": {
            "max_curvature_layer": max_curvature_layer,
            "max_curvature_value": max_curvature_val,
            "avg_curvature": avg_curvature,
        },
        "distance": {
            "early_within": float(early_dist_within),
            "late_within": float(late_dist_within),
            "early_between": float(early_dist_between),
            "late_between": float(late_dist_between),
            "within_ratio": float(late_dist_within / max(early_dist_within, 1e-10)),
            "between_ratio": float(late_dist_between / max(early_dist_between, 1e-10)),
        },
    }

    out_path = RESULT_DIR / f"{model_name}_exp_c_topology.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log_time(f"  Distance within: early={early_dist_within:.2f}, late={late_dist_within:.2f}, "
             f"ratio={results['distance']['within_ratio']:.3f}")
    log_time(f"  Distance between: early={early_dist_between:.2f}, late={late_dist_between:.2f}, "
             f"ratio={results['distance']['between_ratio']:.3f}")

    return results


# ===== Exp D: DMD Mode Analysis =====

def dmd_analysis(all_h, n_layers, d_model, model_name):
    """
    Dynamic Mode Decomposition on full trajectories.
    For each token, fit h_{l+1} ≈ A h_l and extract modes.
    """
    log_time("=== Exp D: DMD Mode Analysis ===")

    valid_tokens = [w for w in ALL_TOKENS if w in all_h]
    r = min(15, n_layers - 1)  # DMD rank

    token_dmd = {}
    all_eigenvalues = []
    all_mode_corrs = []

    for w in valid_tokens:
        # Build trajectory matrix H: [d_model, n_layers+1]
        H_list = []
        for l in range(n_layers + 1):
            if l in all_h[w]:
                H_list.append(all_h[w][l])

        if len(H_list) < r + 2:
            continue

        H = np.column_stack(H_list)  # [d_model, N]

        H1 = H[:, :-1]  # [d_model, N-1]
        H2 = H[:, 1:]   # [d_model, N-1]

        # Reduced DMD
        try:
            U, S, Vt = np.linalg.svd(H1, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        U_r = U[:, :r]
        S_r = S[:r]
        V_r = Vt[:r, :].T

        # Avoid division by zero
        S_r_safe = np.where(S_r > 1e-10, S_r, 1e-10)

        A_tilde = U_r.T @ H2 @ V_r @ np.diag(1.0 / S_r_safe)

        try:
            eigenvalues, W = np.linalg.eig(A_tilde)
        except np.linalg.LinAlgError:
            continue

        # DMD modes
        modes = U_r @ W  # [d_model, r]

        # Mode amplitudes
        try:
            amplitudes = np.linalg.lstsq(modes, H[:, 0], rcond=None)[0]
        except Exception:
            amplitudes = np.ones(r)

        # Sort by amplitude magnitude
        sort_idx = np.argsort(np.abs(amplitudes))[::-1]
        eigenvalues = eigenvalues[sort_idx]
        amplitudes = amplitudes[sort_idx]
        modes = modes[:, sort_idx]

        token_dmd[w] = {
            "eigenvalues": [{"real": float(e.real), "imag": float(e.imag),
                            "magnitude": float(abs(e)), "phase": float(np.angle(e))}
                           for e in eigenvalues[:10]],
            "amplitudes": [float(abs(a)) for a in amplitudes[:10]],
            "top_mode_energy": float(abs(amplitudes[0]) ** 2 / max(np.sum(np.abs(amplitudes) ** 2), 1e-10)),
        }

        all_eigenvalues.append(eigenvalues[:5])

    # Cross-token mode correlation
    if len(token_dmd) >= 2:
        # Compare top-mode directions
        mode_vectors = {}
        for w in valid_tokens:
            if w in all_h:
                H_list = [all_h[w][l] for l in range(n_layers + 1) if l in all_h[w]]
                if len(H_list) >= 2:
                    H = np.column_stack(H_list)
                    H1 = H[:, :-1]
                    try:
                        U, S, Vt = np.linalg.svd(H1, full_matrices=False)
                        mode_vectors[w] = U[:, 0]  # top-1 mode direction
                    except Exception:
                        pass

        mode_corr_within = []
        mode_corr_between = []
        words_with_modes = list(mode_vectors.keys())

        for i, wi in enumerate(words_with_modes):
            for j, wj in enumerate(words_with_modes):
                if i >= j:
                    continue
                vi = mode_vectors[wi]
                vj = mode_vectors[wj]
                ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
                if ni > 1e-10 and nj > 1e-10:
                    cos_val = float(np.dot(vi, vj) / (ni * nj))
                    di, dj = TOKEN_TO_DIM.get(wi, ""), TOKEN_TO_DIM.get(wj, "")
                    if di == dj:
                        mode_corr_within.append(cos_val)
                    else:
                        mode_corr_between.append(cos_val)

        mode_corr_summary = {
            "within_mean": float(np.mean(mode_corr_within)) if mode_corr_within else None,
            "between_mean": float(np.mean(mode_corr_between)) if mode_corr_between else None,
            "delta": float(np.mean(mode_corr_within) - np.mean(mode_corr_between))
                    if mode_corr_within and mode_corr_between else None,
        }
    else:
        mode_corr_summary = {"within_mean": None, "between_mean": None, "delta": None}

    # Eigenvalue statistics
    eig_stats = {}
    for w, dmd in token_dmd.items():
        eigs = dmd["eigenvalues"]
        if eigs:
            mags = [e["magnitude"] for e in eigs]
            eig_stats[w] = {
                "top1_magnitude": mags[0],
                "top1_mode_energy": dmd["top_mode_energy"],
                "n_growing": sum(1 for m in mags if m > 1.0),
                "n_decaying": sum(1 for m in mags if m < 1.0),
                "n_neutral": sum(1 for m in mags if abs(m - 1.0) < 0.05),
            }

    results = {
        "model": model_name,
        "dmd_rank": r,
        "n_tokens_analyzed": len(token_dmd),
        "mode_correlation": mode_corr_summary,
        "eigenvalue_stats": eig_stats,
        "per_token_dmd": {w: {
            "eigenvalues": d["eigenvalues"][:5],
            "top_mode_energy": d["top_mode_energy"],
        } for w, d in token_dmd.items()},
    }

    out_path = RESULT_DIR / f"{model_name}_exp_d_dmd.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    mc = mode_corr_summary
    log_time(f"  DMD mode correlation: within={mc['within_mean']}, "
             f"between={mc['between_mean']}, delta={mc['delta']}")

    # Show eigenvalue patterns for key tokens
    key_tokens = ["dog", "think", "and", "not", "good", "bad", "go", "went"]
    for kt in key_tokens:
        if kt in eig_stats:
            es = eig_stats[kt]
            log_time(f"    '{kt}': top1_mag={es['top1_magnitude']:.3f}, "
                     f"top1_energy={es['top1_mode_energy']:.3f}, "
                     f"growing={es['n_growing']}, decaying={es['n_decaying']}")

    return results


# ===== Exp E: Language Dimension Signatures =====

def analyze_dimension_signatures(all_delta, universal_dirs, scalar_matrix, n_layers, model_name):
    """
    Compare dynamics across language dimensions.
    Focus on negation, logic, tense, reference.
    """
    log_time("=== Exp E: Language Dimension Signatures ===")

    valid_tokens = [w for w in ALL_TOKENS if w in all_delta]
    layers_with_dir = sorted(universal_dirs.keys())

    results = {"model": model_name}

    # 1. Negation pairs: are good/bad opposites in scalar space?
    negation_results = []
    for pos, neg in NEGATION_PAIRS:
        if pos not in all_delta or neg not in all_delta:
            continue

        # Scalar profile correlation
        pos_scalars = []
        neg_scalars = []
        for l in layers_with_dir:
            if l in all_delta[pos] and l in all_delta[neg]:
                u_l = universal_dirs[l]
                c_pos = float(np.dot(u_l, all_delta[pos][l]))
                c_neg = float(np.dot(u_l, all_delta[neg][l]))
                pos_scalars.append(c_pos)
                neg_scalars.append(c_neg)

        if len(pos_scalars) > 2:
            from scipy.stats import spearmanr, pearsonr
            p_corr, p_pval = pearsonr(pos_scalars, neg_scalars)
            s_corr, s_pval = spearmanr(pos_scalars, neg_scalars)

            # Fraction of layers where signs are opposite
            opposite_frac = float(np.sum(np.array(pos_scalars) * np.array(neg_scalars) < 0)
                                  / len(pos_scalars))

            negation_results.append({
                "pos": pos, "neg": neg,
                "pearson_r": float(p_corr),
                "spearman_r": float(s_corr),
                "opposite_fraction": opposite_frac,
                "pos_scalar_mean": float(np.mean(pos_scalars)),
                "neg_scalar_mean": float(np.mean(neg_scalars)),
            })

    results["negation_pairs"] = negation_results

    avg_opposite = np.mean([nr["opposite_fraction"] for nr in negation_results]) if negation_results else None
    avg_pearson = np.mean([nr["pearson_r"] for nr in negation_results]) if negation_results else None
    log_time(f"  Negation pairs: avg opposite fraction={avg_opposite}, "
             f"avg pearson_r={avg_pearson}")
    for nr in negation_results:
        log_time(f"    {nr['pos']}/{nr['neg']}: pearson={nr['pearson_r']:.3f}, "
                 f"opposite_frac={nr['opposite_fraction']:.3f}")

    # 2. Tense: scalar profile comparison within tense group
    tense_results = []
    for group in TENSE_GROUPS:
        group_tokens = [t for t in group if t in all_delta]
        if len(group_tokens) < 2:
            continue

        group_data = {}
        for t in group_tokens:
            scalars = []
            for l in layers_with_dir:
                if l in all_delta[t]:
                    u_l = universal_dirs[l]
                    c = float(np.dot(u_l, all_delta[t][l]))
                    scalars.append(c)
            group_data[t] = scalars

        # Pairwise correlations within tense group
        tense_corrs = []
        t_tokens = list(group_data.keys())
        for i in range(len(t_tokens)):
            for j in range(i + 1, len(t_tokens)):
                if len(group_data[t_tokens[i]]) > 2 and len(group_data[t_tokens[j]]) > 2:
                    from scipy.stats import pearsonr
                    r, _ = pearsonr(group_data[t_tokens[i]], group_data[t_tokens[j]])
                    tense_corrs.append(float(r))

        tense_results.append({
            "group": group,
            "valid_tokens": t_tokens,
            "pairwise_correlations": tense_corrs,
            "mean_corr": float(np.mean(tense_corrs)) if tense_corrs else None,
        })

    results["tense_groups"] = tense_results
    for tr in tense_results:
        log_time(f"  Tense group {tr['group']}: mean corr={tr['mean_corr']}")

    # 3. Logic tokens: different from content tokens?
    logic_tokens = [w for w in LANGUAGE_DIMENSIONS["logic"] if w in all_delta]
    content_tokens = [w for w in valid_tokens
                     if TOKEN_TO_DIM[w].startswith("entities") or TOKEN_TO_DIM[w].startswith("actions")]
    content_tokens = content_tokens[:10]  # sample

    if logic_tokens and content_tokens:
        logic_scalar_corr = []
        content_scalar_corr = []

        for i, wa in enumerate(logic_tokens):
            for j, wb in enumerate(logic_tokens):
                if i >= j:
                    continue
                sa = [float(np.dot(universal_dirs[l], all_delta[wa][l]))
                      for l in layers_with_dir if l in all_delta[wa]]
                sb = [float(np.dot(universal_dirs[l], all_delta[wb][l]))
                      for l in layers_with_dir if l in all_delta[wb]]
                min_len = min(len(sa), len(sb))
                if min_len > 2:
                    from scipy.stats import pearsonr
                    r, _ = pearsonr(sa[:min_len], sb[:min_len])
                    logic_scalar_corr.append(float(r))

        for i, wa in enumerate(content_tokens):
            for j, wb in enumerate(content_tokens):
                if i >= j:
                    continue
                sa = [float(np.dot(universal_dirs[l], all_delta[wa][l]))
                      for l in layers_with_dir if l in all_delta[wa]]
                sb = [float(np.dot(universal_dirs[l], all_delta[wb][l]))
                      for l in layers_with_dir if l in all_delta[wb]]
                min_len = min(len(sa), len(sb))
                if min_len > 2:
                    from scipy.stats import pearsonr
                    r, _ = pearsonr(sa[:min_len], sb[:min_len])
                    content_scalar_corr.append(float(r))

        results["logic_vs_content"] = {
            "logic_within_corr": float(np.mean(logic_scalar_corr)) if logic_scalar_corr else None,
            "content_within_corr": float(np.mean(content_scalar_corr)) if content_scalar_corr else None,
        }
        lc = results["logic_vs_content"]
        log_time(f"  Logic within-scalar corr: {lc['logic_within_corr']}")
        log_time(f"  Content within-scalar corr: {lc['content_within_corr']}")

    # 4. Dimension-level scalar profile statistics
    dim_scalar_stats = {}
    for dim in DIM_NAMES:
        dim_tokens = [w for w in LANGUAGE_DIMENSIONS[dim] if w in all_delta]
        if not dim_tokens:
            continue

        dim_scalars = []
        for w in dim_tokens:
            s = [float(np.dot(universal_dirs[l], all_delta[w][l]))
                 for l in layers_with_dir if l in all_delta[w]]
            if s:
                dim_scalars.append(np.mean(s))

        dim_scalar_stats[dim] = {
            "mean_scalar": float(np.mean(dim_scalars)) if dim_scalars else None,
            "std_scalar": float(np.std(dim_scalars)) if dim_scalars else None,
            "n_tokens": len(dim_tokens),
        }

    results["dimension_scalar_stats"] = dim_scalar_stats
    log_time("  Dimension mean scalars:")
    for dim, stats in dim_scalar_stats.items():
        log_time(f"    {dim}: mean={stats['mean_scalar']}, std={stats['std_scalar']}, "
                 f"n={stats['n_tokens']}")

    out_path = RESULT_DIR / f"{model_name}_exp_e_dimensions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ===== Main =====

def main():
    global _log_file

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"

    log_path = RESULT_DIR / f"{model_name}_phase277.log"
    _log_file = str(log_path)

    log_time(f"Phase 277: Conditional Dynamics Atlas")
    log_time(f"Model: {model_name}, Tokens: {len(ALL_TOKENS)}")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log_time(f"Model info: n_layers={info.n_layers}, d_model={info.d_model}, "
             f"class={info.model_class}")

    # Collect trajectories
    log_time("=" * 60)
    all_h, all_delta, n_layers, d_model = collect_all_trajectories(
        model, tokenizer, device, model_name
    )

    # Exp A: Universal Direction Test
    log_time("=" * 60)
    summary_a, universal_dirs = test_universal_direction(all_delta, n_layers, model_name)

    # Exp B: Scalar Profile Atlas
    log_time("=" * 60)
    profile_b, scalar_matrix = build_scalar_atlas(
        all_delta, universal_dirs, n_layers, model_name
    )

    # Exp C: Trajectory Topology
    log_time("=" * 60)
    results_c = analyze_trajectory_topology(all_h, all_delta, n_layers, model_name)

    # Exp D: DMD Mode Analysis
    log_time("=" * 60)
    results_d = dmd_analysis(all_h, n_layers, d_model, model_name)

    # Exp E: Language Dimension Signatures
    log_time("=" * 60)
    results_e = analyze_dimension_signatures(
        all_delta, universal_dirs, scalar_matrix, n_layers, model_name
    )

    # Final Summary
    log_time("=" * 60)
    log_time("FINAL SUMMARY")
    log_time("=" * 60)

    log_time(f"Exp A — Universal Direction:")
    log_time(f"  var_top1 (mean): {summary_a['global_var_top1_mean']:.4f}")
    log_time(f"  mean pairwise cos: {summary_a['global_mean_cos_mean']:.4f}")

    log_time(f"Exp B — Scalar Profile:")
    log_time(f"  within-dim corr: {profile_b['within_corr_mean']}")
    log_time(f"  between-dim corr: {profile_b['between_corr_mean']}")
    log_time(f"  delta: {profile_b['delta_corr']}")
    if profile_b.get("clustering"):
        log_time(f"  broad ARI: {profile_b['clustering']['broad_ari']}")
        log_time(f"  fine ARI: {profile_b['clustering']['fine_ari']}")

    log_time(f"Exp C — Topology:")
    dc = results_c["distance"]
    log_time(f"  distance within ratio (late/early): {dc['within_ratio']}")
    log_time(f"  distance between ratio (late/early): {dc['between_ratio']}")
    log_time(f"  max curvature layer: {results_c['curvature']['max_curvature_layer']}")

    mc = results_d["mode_correlation"]
    log_time(f"Exp D — DMD:")
    log_time(f"  mode corr within: {mc['within_mean']}")
    log_time(f"  mode corr between: {mc['between_mean']}")

    neg_avg = results_e.get("negation_pairs", [])
    if neg_avg:
        log_time(f"Exp E — Negation:")
        log_time(f"  avg opposite fraction: {np.mean([n['opposite_fraction'] for n in neg_avg])}")
        log_time(f"  avg pearson_r: {np.mean([n['pearson_r'] for n in neg_avg])}")

    # Release model
    del all_h, all_delta, universal_dirs, scalar_matrix
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released. Phase 277 complete.")


if __name__ == "__main__":
    main()
