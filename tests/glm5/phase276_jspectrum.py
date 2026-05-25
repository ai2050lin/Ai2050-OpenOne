"""
Phase 276: Jacobian Spectrum & Conditional Operator Analysis
=============================================================

Upgrade from Phase 275: analyze the FULL SPECTRAL STRUCTURE of the conditional
Jacobian, not just cosine similarity.

Key insight: Phase 275 showed Within > Between in Jacobian cosine (delta +0.11~0.20),
but cosine is a scalar that misses dynamics STRUCTURE. The Jacobian spectrum tells us:
- Which directions are AMPLIFIED (unstable modes, σ > 1)
- Which directions are COMPRESSED (stable/attractor modes, σ < 1)
- The EFFECTIVE RANK of the dynamics
- The CONDITION NUMBER (anisotropy)

Experiments:
A. Jacobian Spectrum — SVD of JV matrix. Extract singular values, effective rank,
   condition number, spectral entropy, stable/unstable dimension ratio.
B. Conditional Operator Distance — Frobenius & subspace distance between operators.
C. Dynamical Clustering — Cluster tokens by Jacobian spectral features vs embeddings.
D. Critical Layer Search — Find layers where spectral structure changes most.

OPTIMIZATION: Single data collection pass (JV matrices shared by all experiments).

Usage:
  python tests/glm5/phase276_jspectrum.py qwen3
  python tests/glm5/phase276_jspectrum.py glm4
  python tests/glm5/phase276_jspectrum.py deepseek7b
"""
import sys, os, json, gc, time, warnings, random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_utils import MODEL_CONFIGS, get_model_info, get_W_U, get_layers

RESULT_DIR = Path("results/phase276_jspectrum")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None

def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ===== Stimuli =====

CATEGORIES = {
    "fruits": ["apple", "banana", "orange", "grape", "mango"],
    "animals": ["dog", "cat", "lion", "tiger", "wolf"],
    "vehicles": ["car", "bus", "train", "plane", "bike"],
}

WITHIN_PAIRS = [
    ("apple", "banana"), ("apple", "orange"), ("banana", "grape"),
    ("dog", "cat"), ("lion", "tiger"), ("wolf", "fox"),
    ("car", "bus"), ("train", "plane"), ("bike", "truck"),
]

BETWEEN_PAIRS = [
    ("apple", "dog"), ("banana", "car"), ("orange", "lion"),
    ("grape", "train"), ("mango", "bike"), ("dog", "bus"),
]

ALL_WORDS = []
for cat_words in CATEGORIES.values():
    ALL_WORDS.extend(cat_words)

WORD_TO_CATEGORY = {}
for cat, words in CATEGORIES.items():
    for w in words:
        WORD_TO_CATEGORY[w] = cat

# Perturbation parameters — 32 vectors for better spectral estimation
N_PERTURBATIONS = 32
EPSILON = 1.0


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
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"{model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# ===== Core: Baseline Forward Pass =====

def run_baseline(model, tokenizer, device, word, n_layers):
    prompt = f"The {word} is"
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    captured = {}
    layers = get_layers(model)

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[name] = output[0].detach().float().cpu()
            else:
                captured[name] = output.detach().float().cpu()
        return hook

    hooks = []
    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    for h in hooks:
        h.remove()

    h_dict = {}
    for li in range(n_layers):
        key = f"L{li}"
        if key in captured:
            h_dict[li] = captured[key][0, -1, :].numpy()

    logits = outputs.logits[0, -1].float().cpu().numpy()
    return h_dict, logits


# ===== Core: Estimate JV matrix at a layer =====

def estimate_jacobian_at_layer(model, tokenizer, device, word, target_layer,
                                perturb_vecs, n_layers, baseline_h):
    """
    Estimate Jacobian J_l^(token) at target_layer using K perturbation vectors.
    Returns the JV matrix [d_model, K] where each column is Jv_k.
    """
    prompt = f"The {word} is"
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)
    layers = get_layers(model)

    jv_vectors = []

    for ki, pv in enumerate(perturb_vecs):
        captured = {}

        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                captured["h_next"] = output[0].detach().float().cpu()
            else:
                captured["h_next"] = output.detach().float().cpu()

        perturb_tensor = torch.tensor(pv, dtype=torch.float32)

        def perturbation_prehook(module, args):
            hidden_states = args[0]
            perturbed = hidden_states.clone()
            p = perturb_tensor.to(device=perturbed.device, dtype=perturbed.dtype)
            perturbed[0, -1, :] += EPSILON * p
            return (perturbed,) + args[1:]

        hooks = []
        hooks.append(layers[target_layer].register_forward_hook(capture_hook))
        hooks.append(layers[target_layer].register_forward_pre_hook(perturbation_prehook))

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks:
            h.remove()

        if "h_next" in captured:
            h_next_perturbed = captured["h_next"][0, -1, :].numpy()
            next_layer = target_layer + 1 if target_layer + 1 < n_layers else target_layer
            h_next_baseline = baseline_h.get(next_layer)
            if h_next_baseline is not None:
                jv = (h_next_perturbed - h_next_baseline) / EPSILON
                jv_vectors.append(jv)
            else:
                jv_vectors.append(np.zeros_like(pv))
        else:
            jv_vectors.append(np.zeros_like(pv))

    JV = np.column_stack(jv_vectors) if jv_vectors else np.zeros((len(perturb_vecs[0]), len(perturb_vecs)))
    return JV


# ===== Unified Data Collection =====

def collect_all_jv_data(model, tokenizer, device, model_name):
    """
    Single pass: collect JV matrices and baseline hidden states for all (word, layer).
    Returns: all_JV, all_spectral, all_baseline_h
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    # Sample layers
    sampled_layers = []
    for i in range(1, n_layers - 1):
        if i % max(1, (n_layers - 2) // 8) == 0 or i == n_layers // 2:
            sampled_layers.append(i)
    sampled_layers = sorted(set(sampled_layers))
    if len(sampled_layers) > 9:
        step = len(sampled_layers) // 8
        sampled_layers = sampled_layers[::step][:9]
    log_time(f"Sampled layers: {sampled_layers}")

    # Pre-generate perturbation vectors
    rng = np.random.RandomState(42)
    perturb_vecs = []
    for k in range(N_PERTURBATIONS):
        v = rng.randn(d_model).astype(np.float32)
        v = v / np.linalg.norm(v)
        perturb_vecs.append(v)
    log_time(f"Generated {len(perturb_vecs)} perturbation vectors, dim={d_model}")

    all_JV = {}        # word -> layer_idx -> JV [d_model, K]
    all_spectral = {}   # word -> layer_idx -> spectral features
    all_baseline_h = {} # word -> layer_idx -> h (numpy)

    for wi, word in enumerate(ALL_WORDS):
        log_time(f"  Word {wi+1}/{len(ALL_WORDS)}: '{word}' — collecting data...")
        baseline_h, baseline_logits = run_baseline(model, tokenizer, device, word, n_layers)
        all_baseline_h[word] = {str(l): baseline_h[l].tolist() for l in baseline_h}

        all_JV[word] = {}
        all_spectral[word] = {}

        for li, layer_idx in enumerate(sampled_layers):
            if li % 2 == 0:
                log_time(f"    L{layer_idx} ({li+1}/{len(sampled_layers)})...")

            JV = estimate_jacobian_at_layer(
                model, tokenizer, device, word, layer_idx,
                perturb_vecs, n_layers, baseline_h
            )
            all_JV[word][str(layer_idx)] = JV

            # SVD of JV
            try:
                U, sigma, Vt = np.linalg.svd(JV, full_matrices=False)
            except np.linalg.LinAlgError:
                log_time(f"    WARNING: SVD failed for {word} L{layer_idx}")
                all_spectral[word][str(layer_idx)] = {
                    "singular_values": None, "effective_rank": None,
                    "condition_number": None, "spectral_entropy": None,
                    "top1_sv": None, "top5_sv_ratio": None, "total_energy": None,
                    "n_unstable": None, "n_stable": None, "n_dead": None,
                    "unstable_ratio": None, "stable_ratio": None,
                }
                continue

            sv = sigma.tolist()
            threshold = 0.1 * sigma[0] if len(sigma) > 0 and sigma[0] > 0 else 1e-10
            effective_rank = int(np.sum(sigma > threshold))

            if len(sigma) > 1 and sigma[-1] > 1e-10:
                condition_number = float(sigma[0] / sigma[-1])
            else:
                condition_number = -1  # infinity marker

            if len(sigma) > 0 and np.sum(sigma) > 0:
                p = sigma / np.sum(sigma)
                p = p[p > 0]
                entropy = -np.sum(p * np.log(p)) / np.log(len(sigma))
            else:
                entropy = 0.0

            top1_sv = float(sigma[0]) if len(sigma) > 0 else 0.0
            top5_ratio = float(np.sum(sigma[:5]**2) / np.sum(sigma**2)) if len(sigma) >= 5 else 1.0
            total_energy = float(np.sum(sigma**2))

            n_unstable = int(np.sum(sigma > 1.0))
            n_stable = int(np.sum((sigma > 0.01) & (sigma <= 1.0)))
            n_dead = int(np.sum(sigma <= 0.01))

            all_spectral[word][str(layer_idx)] = {
                "singular_values": sv[:10],
                "effective_rank": effective_rank,
                "condition_number": condition_number,
                "spectral_entropy": float(entropy),
                "top1_sv": top1_sv,
                "top5_sv_ratio": top5_ratio,
                "total_energy": total_energy,
                "n_unstable": n_unstable,
                "n_stable": n_stable,
                "n_dead": n_dead,
                "unstable_ratio": float(n_unstable / max(len(sigma), 1)),
                "stable_ratio": float(n_stable / max(len(sigma), 1)),
            }

        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        log_time(f"  '{word}' done. GPU={gpu_mem:.2f}GB")

    # Save spectral data (for Exp C/D reuse)
    out_path = RESULT_DIR / f"{model_name}_spectral.json"
    with open(out_path, "w") as f:
        json.dump(all_spectral, f)
    log_time(f"Spectral data saved to {out_path}")

    # Save JV stats (not full matrices — too large)
    jv_stats = {}
    for word in ALL_WORDS:
        jv_stats[word] = {}
        for layer_idx in sampled_layers:
            JV = all_JV.get(word, {}).get(str(layer_idx))
            if JV is not None:
                jv_stats[word][str(layer_idx)] = {
                    "shape": list(JV.shape),
                    "frobenius_norm": float(np.linalg.norm(JV)),
                }
    out_path = RESULT_DIR / f"{model_name}_jv_stats.json"
    with open(out_path, "w") as f:
        json.dump(jv_stats, f, indent=2)

    return all_JV, all_spectral, sampled_layers, n_layers, d_model


# ===== Exp A: Jacobian Spectrum Analysis =====

def analyze_spectrum(all_spectral, sampled_layers, model_name):
    """Analyze spectral feature similarity: within vs between categories."""

    def get_spectral_vector(word, layer_idx):
        if word not in all_spectral:
            return None
        ld = all_spectral[word].get(str(layer_idx))
        if ld is None or ld.get("singular_values") is None:
            return None
        return np.array([
            ld["effective_rank"], ld["spectral_entropy"], ld["top1_sv"],
            ld["top5_sv_ratio"], ld["total_energy"],
            ld["unstable_ratio"], ld["stable_ratio"],
        ], dtype=np.float32)

    def get_sv_vector(word, layer_idx):
        if word not in all_spectral:
            return None
        ld = all_spectral[word].get(str(layer_idx))
        if ld is None or ld.get("singular_values") is None:
            return None
        return np.array(ld["singular_values"], dtype=np.float32)

    per_layer = {}
    per_layer_sv_cos = {}

    for layer_idx in sampled_layers:
        # Spectral feature cosine
        within_feats, between_feats = [], []
        for wa, wb in WITHIN_PAIRS:
            fa = get_spectral_vector(wa, layer_idx)
            fb = get_spectral_vector(wb, layer_idx)
            if fa is not None and fb is not None:
                na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
                if na > 1e-10 and nb > 1e-10:
                    within_feats.append(float(np.dot(fa, fb) / (na * nb)))
        for wa, wb in BETWEEN_PAIRS:
            fa = get_spectral_vector(wa, layer_idx)
            fb = get_spectral_vector(wb, layer_idx)
            if fa is not None and fb is not None:
                na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
                if na > 1e-10 and nb > 1e-10:
                    between_feats.append(float(np.dot(fa, fb) / (na * nb)))

        per_layer[str(layer_idx)] = {
            "within_mean": float(np.mean(within_feats)) if within_feats else None,
            "between_mean": float(np.mean(between_feats)) if between_feats else None,
            "within_n": len(within_feats),
            "between_n": len(between_feats),
            "delta": float(np.mean(within_feats) - np.mean(between_feats))
                if within_feats and between_feats else None,
        }

        # Singular value vector cosine
        within_sv, between_sv = [], []
        for wa, wb in WITHIN_PAIRS:
            sa = get_sv_vector(wa, layer_idx)
            sb = get_sv_vector(wb, layer_idx)
            if sa is not None and sb is not None:
                na, nb = np.linalg.norm(sa), np.linalg.norm(sb)
                if na > 1e-10 and nb > 1e-10:
                    within_sv.append(float(np.dot(sa, sb) / (na * nb)))
        for wa, wb in BETWEEN_PAIRS:
            sa = get_sv_vector(wa, layer_idx)
            sb = get_sv_vector(wb, layer_idx)
            if sa is not None and sb is not None:
                na, nb = np.linalg.norm(sa), np.linalg.norm(sb)
                if na > 1e-10 and nb > 1e-10:
                    between_sv.append(float(np.dot(sa, sb) / (na * nb)))

        per_layer_sv_cos[str(layer_idx)] = {
            "within_mean": float(np.mean(within_sv)) if within_sv else None,
            "between_mean": float(np.mean(between_sv)) if between_sv else None,
            "delta": float(np.mean(within_sv) - np.mean(between_sv))
                if within_sv and between_sv else None,
        }

    # Per-layer average spectral features
    per_layer_avg = {}
    for layer_idx in sampled_layers:
        feats = {}
        for feat_name in ["effective_rank", "spectral_entropy", "top1_sv", "top5_sv_ratio",
                          "total_energy", "unstable_ratio", "stable_ratio", "n_unstable", "n_stable", "n_dead"]:
            vals = []
            for word in ALL_WORDS:
                ld = all_spectral.get(word, {}).get(str(layer_idx), {})
                v = ld.get(feat_name)
                if v is not None:
                    vals.append(v)
            feats[feat_name] = {
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals)) if vals else None,
            }
        per_layer_avg[str(layer_idx)] = feats

    within_all = [per_layer[k]["within_mean"] for k in per_layer if per_layer[k]["within_mean"] is not None]
    between_all = [per_layer[k]["between_mean"] for k in per_layer if per_layer[k]["between_mean"] is not None]
    within_sv_all = [per_layer_sv_cos[k]["within_mean"] for k in per_layer_sv_cos if per_layer_sv_cos[k]["within_mean"] is not None]
    between_sv_all = [per_layer_sv_cos[k]["between_mean"] for k in per_layer_sv_cos if per_layer_sv_cos[k]["between_mean"] is not None]

    summary = {
        "model": model_name,
        "sampled_layers": sampled_layers,
        "spectral_feature_similarity": {
            "within_mean": float(np.mean(within_all)) if within_all else None,
            "between_mean": float(np.mean(between_all)) if between_all else None,
            "delta": float(np.mean(within_all) - np.mean(between_all))
                if within_all and between_all else None,
        },
        "sv_cosine_similarity": {
            "within_mean": float(np.mean(within_sv_all)) if within_sv_all else None,
            "between_mean": float(np.mean(between_sv_all)) if between_sv_all else None,
            "delta": float(np.mean(within_sv_all) - np.mean(between_sv_all))
                if within_sv_all and between_sv_all else None,
        },
        "per_layer_spectral_similarity": per_layer,
        "per_layer_sv_cosine": per_layer_sv_cos,
        "per_layer_avg_features": per_layer_avg,
    }

    out_path = RESULT_DIR / f"{model_name}_exp_a_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_time(f"Exp A summary saved to {out_path}")

    log_time(f"=== Exp A: Jacobian Spectrum ({model_name}) ===")
    sfs = summary["spectral_feature_similarity"]
    log_time(f"  Spectral feature sim: within={sfs['within_mean']}, "
             f"between={sfs['between_mean']}, delta={sfs['delta']}")
    svcs = summary["sv_cosine_similarity"]
    log_time(f"  SV cosine sim: within={svcs['within_mean']}, "
             f"between={svcs['between_mean']}, delta={svcs['delta']}")

    log_time("  Per-layer averages:")
    for layer_idx in sampled_layers:
        pl = per_layer_avg[str(layer_idx)]
        log_time(f"    L{layer_idx}: eff_rank={pl['effective_rank']['mean']:.1f}, "
                 f"entropy={pl['spectral_entropy']['mean']:.3f}, "
                 f"top1_sv={pl['top1_sv']['mean']:.2f}, "
                 f"unstable%={pl['unstable_ratio']['mean']:.3f}, "
                 f"delta={per_layer[str(layer_idx)].get('delta', 'N/A')}")

    return summary


# ===== Exp B: Conditional Operator Distance =====

def analyze_operator_distance(all_JV, sampled_layers, model_name):
    """Compute operator distance between token pairs using JV matrices."""

    per_layer = {}

    for layer_idx in sampled_layers:
        within_frob, between_frob = [], []
        within_cos, between_cos = [], []
        within_proj, between_proj = [], []

        for wa, wb in WITHIN_PAIRS:
            JVa = all_JV.get(wa, {}).get(str(layer_idx))
            JVb = all_JV.get(wb, {}).get(str(layer_idx))
            if JVa is None or JVb is None:
                continue

            diff = JVa - JVb
            frob_dist = np.linalg.norm(diff) / max(np.linalg.norm(JVa), np.linalg.norm(JVb), 1e-10)
            within_frob.append(float(frob_dist))

            Ua, sa, _ = np.linalg.svd(JVa, full_matrices=False)
            proj_b = Ua @ (Ua.T @ JVb)
            cos_sub = np.linalg.norm(proj_b) / max(np.linalg.norm(JVb), 1e-10)
            within_cos.append(float(cos_sub))

            M = Ua.T @ JVb
            if M.shape[0] > 0 and M.shape[1] > 0:
                try:
                    s = np.linalg.svd(M, compute_uv=False)
                    within_proj.append(float(s[0]))
                except Exception:
                    pass

        for wa, wb in BETWEEN_PAIRS:
            JVa = all_JV.get(wa, {}).get(str(layer_idx))
            JVb = all_JV.get(wb, {}).get(str(layer_idx))
            if JVa is None or JVb is None:
                continue

            diff = JVa - JVb
            frob_dist = np.linalg.norm(diff) / max(np.linalg.norm(JVa), np.linalg.norm(JVb), 1e-10)
            between_frob.append(float(frob_dist))

            Ua, sa, _ = np.linalg.svd(JVa, full_matrices=False)
            proj_b = Ua @ (Ua.T @ JVb)
            cos_sub = np.linalg.norm(proj_b) / max(np.linalg.norm(JVb), 1e-10)
            between_cos.append(float(cos_sub))

            M = Ua.T @ JVb
            if M.shape[0] > 0 and M.shape[1] > 0:
                try:
                    s = np.linalg.svd(M, compute_uv=False)
                    between_proj.append(float(s[0]))
                except Exception:
                    pass

        per_layer[str(layer_idx)] = {
            "frobenius_distance": {
                "within_mean": float(np.mean(within_frob)) if within_frob else None,
                "between_mean": float(np.mean(between_frob)) if between_frob else None,
                "delta": float(np.mean(within_frob) - np.mean(between_frob))
                    if within_frob and between_frob else None,
            },
            "subspace_cosine": {
                "within_mean": float(np.mean(within_cos)) if within_cos else None,
                "between_mean": float(np.mean(between_cos)) if between_cos else None,
                "delta": float(np.mean(within_cos) - np.mean(between_cos))
                    if within_cos and between_cos else None,
            },
            "principal_angle_cos": {
                "within_mean": float(np.mean(within_proj)) if within_proj else None,
                "between_mean": float(np.mean(between_proj)) if between_proj else None,
                "delta": float(np.mean(within_proj) - np.mean(between_proj))
                    if within_proj and between_proj else None,
            },
        }

    within_frob_all = [per_layer[k]["frobenius_distance"]["within_mean"] for k in per_layer
                       if per_layer[k]["frobenius_distance"]["within_mean"] is not None]
    between_frob_all = [per_layer[k]["frobenius_distance"]["between_mean"] for k in per_layer
                        if per_layer[k]["frobenius_distance"]["between_mean"] is not None]
    within_cos_all = [per_layer[k]["subspace_cosine"]["within_mean"] for k in per_layer
                      if per_layer[k]["subspace_cosine"]["within_mean"] is not None]
    between_cos_all = [per_layer[k]["subspace_cosine"]["between_mean"] for k in per_layer
                       if per_layer[k]["subspace_cosine"]["between_mean"] is not None]

    summary = {
        "model": model_name,
        "sampled_layers": sampled_layers,
        "frobenius_distance": {
            "within_mean": float(np.mean(within_frob_all)) if within_frob_all else None,
            "between_mean": float(np.mean(between_frob_all)) if between_frob_all else None,
            "delta": float(np.mean(within_frob_all) - np.mean(between_frob_all))
                if within_frob_all and between_frob_all else None,
        },
        "subspace_cosine": {
            "within_mean": float(np.mean(within_cos_all)) if within_cos_all else None,
            "between_mean": float(np.mean(between_cos_all)) if between_cos_all else None,
            "delta": float(np.mean(within_cos_all) - np.mean(between_cos_all))
                if within_cos_all and between_cos_all else None,
        },
        "per_layer": per_layer,
    }

    out_path = RESULT_DIR / f"{model_name}_exp_b_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_time(f"Exp B summary saved to {out_path}")

    log_time(f"=== Exp B: Operator Distance ({model_name}) ===")
    fd = summary["frobenius_distance"]
    log_time(f"  Frobenius: within={fd['within_mean']}, between={fd['between_mean']}, "
             f"delta={fd['delta']}")
    sc = summary["subspace_cosine"]
    log_time(f"  Subspace cosine: within={sc['within_mean']}, between={sc['between_mean']}, "
             f"delta={sc['delta']}")
    for layer_idx in sampled_layers:
        pl = per_layer[str(layer_idx)]
        fd_l = pl["frobenius_distance"]
        sc_l = pl["subspace_cosine"]
        pa_l = pl["principal_angle_cos"]
        log_time(f"    L{layer_idx}: frob_delta={fd_l.get('delta')}, "
                 f"subspace_delta={sc_l.get('delta')}, "
                 f"principal_delta={pa_l.get('delta')}")

    return summary


# ===== Exp C: Dynamical Clustering =====

def analyze_clustering(model, tokenizer, device, model_name, all_spectral, sampled_layers, n_layers):
    """Cluster tokens by Jacobian spectral features vs embeddings."""

    # Build feature vectors for each word
    spectral_features = {}
    embedding_features = {}

    for wi, word in enumerate(ALL_WORDS):
        log_time(f"  Clustering: word {wi+1}/{len(ALL_WORDS)}: '{word}'...")
        baseline_h, baseline_logits = run_baseline(model, tokenizer, device, word, n_layers)

        # Spectral feature vector: concat across all sampled layers
        feat_list = []
        for layer_idx in sampled_layers:
            ld = all_spectral.get(word, {}).get(str(layer_idx), {})
            if ld.get("singular_values") is not None:
                feat_list.extend([
                    ld["effective_rank"], ld["spectral_entropy"], ld["top1_sv"],
                    ld["top5_sv_ratio"], ld["total_energy"],
                    ld["unstable_ratio"], ld["stable_ratio"],
                ])
        if feat_list:
            spectral_features[word] = np.array(feat_list, dtype=np.float32)

        mid_layer = n_layers // 2
        h_mid = baseline_h.get(mid_layer)
        if h_mid is not None:
            embedding_features[word] = h_mid

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    true_labels = [0 if WORD_TO_CATEGORY[w] == "fruits"
                   else 1 if WORD_TO_CATEGORY[w] == "animals"
                   else 2 for w in ALL_WORDS]

    results = {}

    # 1. Spectral feature clustering
    if len(spectral_features) >= len(ALL_WORDS):
        X_spectral = np.array([spectral_features[w] for w in ALL_WORDS])
        scaler = StandardScaler()
        X_spectral = scaler.fit_transform(X_spectral)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        spectral_labels = km.fit_predict(X_spectral)
        ari_spectral = adjusted_rand_score(true_labels, spectral_labels)
        results["spectral_clustering"] = {
            "ari": float(ari_spectral),
            "labels": [int(l) for l in spectral_labels],
            "true_labels": true_labels,
        }
        log_time(f"  Spectral clustering ARI: {ari_spectral:.4f}")

    # 2. Embedding clustering
    if len(embedding_features) >= len(ALL_WORDS):
        X_embed = np.array([embedding_features[w] for w in ALL_WORDS])
        from sklearn.decomposition import PCA
        n_pca = min(10, X_embed.shape[1], X_embed.shape[0])
        pca = PCA(n_components=n_pca, random_state=42)
        X_embed_reduced = pca.fit_transform(X_embed)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        embed_labels = km.fit_predict(X_embed_reduced)
        ari_embed = adjusted_rand_score(true_labels, embed_labels)
        results["embedding_clustering"] = {
            "ari": float(ari_embed),
            "labels": [int(l) for l in embed_labels],
            "n_pca": n_pca,
        }
        log_time(f"  Embedding clustering ARI: {ari_embed:.4f}")

    # 3. Per-layer Jacobian spectral clustering
    per_layer_ari = {}
    for layer_idx in sampled_layers:
        feat_list = []
        valid_words = []
        for word in ALL_WORDS:
            ld = all_spectral.get(word, {}).get(str(layer_idx), {})
            if ld.get("singular_values") is not None:
                feat_list.append([
                    ld["effective_rank"], ld["spectral_entropy"], ld["top1_sv"],
                    ld["top5_sv_ratio"], ld["total_energy"],
                    ld["unstable_ratio"], ld["stable_ratio"],
                ])
                valid_words.append(word)
        if len(valid_words) >= 3:
            X = np.array(feat_list, dtype=np.float32)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            valid_true = [0 if WORD_TO_CATEGORY[w] == "fruits"
                         else 1 if WORD_TO_CATEGORY[w] == "animals"
                         else 2 for w in valid_words]
            ari = adjusted_rand_score(valid_true, labels)
            per_layer_ari[str(layer_idx)] = float(ari)

    results["per_layer_spectral_ari"] = per_layer_ari

    out_path = RESULT_DIR / f"{model_name}_exp_c_clustering.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log_time(f"=== Exp C: Dynamical Clustering ({model_name}) ===")
    sc = results.get("spectral_clustering", {})
    ec = results.get("embedding_clustering", {})
    log_time(f"  Spectral ARI: {sc.get('ari', 'N/A')}")
    log_time(f"  Embedding ARI: {ec.get('ari', 'N/A')}")
    for layer_idx in sampled_layers:
        ari = per_layer_ari.get(str(layer_idx), "N/A")
        log_time(f"    L{layer_idx}: ARI={ari}")

    return results


# ===== Exp D: Critical Layer Search =====

def analyze_critical_layers(all_spectral, sampled_layers, model_name):
    """Find layers where Jacobian spectral structure changes most."""

    transition_data = {}

    for word in ALL_WORDS:
        word_data = {}
        for i in range(1, len(sampled_layers)):
            l_prev = sampled_layers[i - 1]
            l_curr = sampled_layers[i]

            prev = all_spectral.get(word, {}).get(str(l_prev), {})
            curr = all_spectral.get(word, {}).get(str(l_curr), {})

            if prev.get("singular_values") is None or curr.get("singular_values") is None:
                continue

            feat_prev = [prev["effective_rank"], prev["spectral_entropy"], prev["top1_sv"],
                        prev["top5_sv_ratio"], prev["total_energy"], prev["unstable_ratio"],
                        prev["stable_ratio"]]
            feat_curr = [curr["effective_rank"], curr["spectral_entropy"], curr["top1_sv"],
                        curr["top5_sv_ratio"], curr["total_energy"], curr["unstable_ratio"],
                        curr["stable_ratio"]]

            delta = np.array(feat_curr) - np.array(feat_prev)
            delta_norm = float(np.linalg.norm(delta))

            key = f"L{l_prev}_to_L{l_curr}"
            word_data[key] = {
                "delta_norm": delta_norm,
                "prev_layer": l_prev,
                "curr_layer": l_curr,
            }

        transition_data[word] = word_data

    # Average transition magnitude
    transition_avg = {}
    for i in range(1, len(sampled_layers)):
        l_prev = sampled_layers[i - 1]
        l_curr = sampled_layers[i]
        key = f"L{l_prev}_to_L{l_curr}"

        norms = [transition_data[w][key]["delta_norm"]
                 for w in ALL_WORDS
                 if key in transition_data.get(w, {})]

        if norms:
            transition_avg[key] = {
                "mean_delta_norm": float(np.mean(norms)),
                "std_delta_norm": float(np.std(norms)),
                "prev_layer": l_prev,
                "curr_layer": l_curr,
            }

    # Within/between similarity of transition profiles
    transition_similarity = {"within": [], "between": []}

    for wa, wb in WITHIN_PAIRS:
        ta = transition_data.get(wa, {})
        tb = transition_data.get(wb, {})
        if not ta or not tb:
            continue
        common_keys = set(ta.keys()) & set(tb.keys())
        if not common_keys:
            continue
        vec_a = np.array([ta[k]["delta_norm"] for k in sorted(common_keys)])
        vec_b = np.array([tb[k]["delta_norm"] for k in sorted(common_keys)])
        na, nb = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
        if na > 1e-10 and nb > 1e-10:
            transition_similarity["within"].append(float(np.dot(vec_a, vec_b) / (na * nb)))

    for wa, wb in BETWEEN_PAIRS:
        ta = transition_data.get(wa, {})
        tb = transition_data.get(wb, {})
        if not ta or not tb:
            continue
        common_keys = set(ta.keys()) & set(tb.keys())
        if not common_keys:
            continue
        vec_a = np.array([ta[k]["delta_norm"] for k in sorted(common_keys)])
        vec_b = np.array([tb[k]["delta_norm"] for k in sorted(common_keys)])
        na, nb = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
        if na > 1e-10 and nb > 1e-10:
            transition_similarity["between"].append(float(np.dot(vec_a, vec_b) / (na * nb)))

    results = {
        "model": model_name,
        "sampled_layers": sampled_layers,
        "transition_avg": transition_avg,
        "transition_similarity": {
            "within_mean": float(np.mean(transition_similarity["within"])) if transition_similarity["within"] else None,
            "between_mean": float(np.mean(transition_similarity["between"])) if transition_similarity["between"] else None,
            "delta": float(np.mean(transition_similarity["within"]) - np.mean(transition_similarity["between"]))
                if transition_similarity["within"] and transition_similarity["between"] else None,
        },
    }

    out_path = RESULT_DIR / f"{model_name}_exp_d_critical.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log_time(f"=== Exp D: Critical Layers ({model_name}) ===")
    log_time("  Transition magnitudes:")
    for key in sorted(transition_avg.keys()):
        ta = transition_avg[key]
        log_time(f"    {key}: {ta['mean_delta_norm']:.4f} +/- {ta['std_delta_norm']:.4f}")
    ts = results["transition_similarity"]
    log_time(f"  Transition profile sim: within={ts.get('within_mean')}, "
             f"between={ts.get('between_mean')}, delta={ts.get('delta')}")

    return results


# ===== Main =====

def main():
    global _log_file

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"

    log_path = RESULT_DIR / f"{model_name}_phase276.log"
    _log_file = str(log_path)

    log_time(f"Phase 276: Jacobian Spectrum & Conditional Operator Analysis")
    log_time(f"Model: {model_name}")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log_time(f"Model info: n_layers={info.n_layers}, d_model={info.d_model}, "
             f"class={info.model_class}")

    # ===== Single data collection pass =====
    log_time("=" * 60)
    log_time("Collecting JV matrices and spectral data (single pass)...")
    log_time("=" * 60)
    all_JV, all_spectral, sampled_layers, n_layers, d_model = collect_all_jv_data(
        model, tokenizer, device, model_name
    )

    # ===== Analysis (no more model inference needed) =====

    # Exp A: Jacobian Spectrum
    log_time("=" * 60)
    log_time("Analyzing Exp A: Jacobian Spectrum")
    log_time("=" * 60)
    summary_a = analyze_spectrum(all_spectral, sampled_layers, model_name)

    # Exp B: Conditional Operator Distance
    log_time("=" * 60)
    log_time("Analyzing Exp B: Conditional Operator Distance")
    log_time("=" * 60)
    summary_b = analyze_operator_distance(all_JV, sampled_layers, model_name)

    # Exp C: Dynamical Clustering (needs baseline hidden states for embedding)
    log_time("=" * 60)
    log_time("Running Exp C: Dynamical Clustering")
    log_time("=" * 60)
    results_c = analyze_clustering(model, tokenizer, device, model_name, all_spectral, sampled_layers, n_layers)

    # Exp D: Critical Layer Search
    log_time("=" * 60)
    log_time("Analyzing Exp D: Critical Layer Search")
    log_time("=" * 60)
    results_d = analyze_critical_layers(all_spectral, sampled_layers, model_name)

    # ===== Final Summary =====
    log_time("=" * 60)
    log_time("FINAL SUMMARY")
    log_time("=" * 60)

    sfs = summary_a["spectral_feature_similarity"]
    log_time(f"Spectral feature sim: within={sfs['within_mean']}, "
             f"between={sfs['between_mean']}, delta={sfs['delta']}")

    svcs = summary_a["sv_cosine_similarity"]
    log_time(f"SV cosine sim: within={svcs['within_mean']}, "
             f"between={svcs['between_mean']}, delta={svcs['delta']}")

    fd = summary_b["frobenius_distance"]
    log_time(f"Operator Frobenius dist: within={fd['within_mean']}, "
             f"between={fd['between_mean']}, delta={fd['delta']}")

    sc = summary_b["subspace_cosine"]
    log_time(f"Subspace cosine: within={sc['within_mean']}, "
             f"between={sc['between_mean']}, delta={sc['delta']}")

    spectral_ari = results_c.get("spectral_clustering", {}).get("ari", "N/A")
    embed_ari = results_c.get("embedding_clustering", {}).get("ari", "N/A")
    log_time(f"Clustering ARI: spectral={spectral_ari}, embedding={embed_ari}")

    ts = results_d.get("transition_similarity", {})
    log_time(f"Transition profile: within={ts.get('within_mean')}, "
             f"between={ts.get('between_mean')}, delta={ts.get('delta')}")

    # Release model
    del all_JV  # Free large numpy arrays
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released. Phase 276 complete.")


if __name__ == "__main__":
    main()
