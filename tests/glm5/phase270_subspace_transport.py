"""
Phase 270: Subspace Transport Analysis — The Decisive Experiment
================================================================

CORE QUESTION (from Phase 268 critique):
  "W_U-invisible ≠ causally irrelevant"
  The residual stream ROTATES across layers. V_inv at layer l may be
  rotated into V_vis at layer l+1 by attention/MLP.

  If V_inv components get transported into V_vis → "dark matter" is
  just "delayed visible computation" — Phase 268 overstates the finding.

  If V_inv components STAY invisible → Phase 268 finding stands.

EXPERIMENT DESIGN:
  For each layer l → l+1:
  1. Decompose h_l = h_l_vis + h_l_inv  (in V_vis / V_inv of W_U)
  2. Decompose h_{l+1} = h_{l+1}_vis + h_{l+1}_inv
  3. Measure: how much of h_{l+1}_vis is predicted by h_l_inv?
     → "Transport from invisible to visible"
  4. Compare: how much of h_{l+1}_vis is predicted by h_l_vis?
     → "Persistence in visible"

  Also: "Cumulative transport" — how much of h_L_vis (final layer)
  is predicted by h_l_inv at each intermediate layer?

KEY METRICS:
  - Transport rate: fraction of V_vis(l+1) variance explained by V_inv(l)
  - Transport gain: transport rate at layer l+1 vs at layer l
  - Accumulated transport: total fraction of final V_vis explained
    by intermediate V_inv

PR concern (also from critique):
  PR ≈ 20-30 doesn't mean hidden states are truly 20-30 dimensional.
  We add: full covariance spectrum visualization.

=== Usage ===
  python tests/glm5/phase270_subspace_transport.py qwen3
  python tests/glm5/phase270_subspace_transport.py glm4
  python tests/glm5/phase270_subspace_transport.py deepseek7b
"""
import sys, os, json, gc, time, warnings, random
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase270_subspace_transport")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None

def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ===== Prompt Generation (reuse Phase 268) =====

def generate_diverse_prompts(n=1000):
    prompts = []

    SING = [
        "cat", "dog", "bird", "fish", "tree", "house", "car", "book",
        "phone", "chair", "table", "door", "river", "mountain", "cloud",
        "fire", "earth", "stone", "glass", "wood", "paper", "food",
        "king", "queen", "child", "woman", "man", "doctor", "teacher",
        "soldier", "artist", "writer", "singer", "farmer", "builder",
        "driver", "pilot", "baker", "hunter", "sailor", "nurse", "cook",
        "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit", "mouse",
        "snake", "frog", "duck", "goose", "swan", "crow", "hawk", "eagle",
    ]
    PLUR = [s + "s" for s in SING[:40]]

    VERBS_S = ["sits", "runs", "walks", "eats", "drinks", "sleeps", "thinks",
               "knows", "wants", "needs", "loves", "hates", "makes", "finds",
               "reads", "writes", "speaks", "hears", "sees", "feels"]
    VERBS_P = ["sit", "run", "walk", "eat", "drink", "sleep", "think",
               "know", "want", "need", "love", "hate", "make", "find",
               "read", "write", "speak", "hear", "see", "feel"]

    ANIMATE = ["cat", "dog", "bird", "horse", "cow", "child", "woman", "man",
               "boy", "girl", "baby", "friend", "teacher", "doctor", "king",
               "queen", "prince", "soldier", "artist", "writer"]
    INANIMATE = ["rock", "chair", "table", "door", "wall", "road", "bridge",
                 "tower", "boat", "ship", "train", "car", "phone", "book",
                 "pen", "clock", "shirt", "pants", "cup", "plate"]

    ADJS = ["big", "small", "red", "blue", "old", "new", "good", "bad",
            "fast", "slow", "hot", "cold", "dark", "bright", "hard", "soft",
            "long", "short", "tall", "wide", "heavy", "light", "rich", "poor"]

    ANIMALS = ["cat", "dog", "bird", "horse", "cow", "pig", "sheep", "goat",
               "duck", "hen", "fox", "wolf", "bear", "deer", "rabbit", "mouse",
               "lion", "tiger", "elephant", "monkey", "whale", "dolphin", "shark",
               "snake", "frog", "bee", "ant", "spider", "fly", "butterfly"]
    TOOLS = ["hammer", "wrench", "saw", "drill", "screwdriver", "pliers",
             "chisel", "axe", "shovel", "rake", "hoe", "spade", "knife",
             "scissors", "needle", "thread", "rope", "wire", "nail", "screw",
             "bolt", "nut", "glue", "tape", "ruler", "compass", "level",
             "wrench", "clamp", "vise"]

    for noun in SING[:30]:
        for verb in VERBS_S[:4]:
            prompts.append(f"The {noun} {verb}")
    for noun in PLUR[:30]:
        for verb in VERBS_P[:4]:
            prompts.append(f"The {noun} {verb}")
    for word in ANIMATE:
        prompts.append(f"The {word} thinks about tomorrow")
    for word in INANIMATE:
        prompts.append(f"The {word} sits on the shelf")
    for noun in SING[:15]:
        prompts.append(f"The {noun} will go tomorrow")
        prompts.append(f"The {noun} went yesterday")
        prompts.append(f"The {noun} is going now")
        prompts.append(f"The {noun} has gone already")
    for adj in ADJS[:20]:
        for noun in SING[:6]:
            prompts.append(f"The {adj} {noun} is here")
    for noun in SING[:20]:
        prompts.append(f"Is the {noun} here?")
        prompts.append(f"Where is the {noun}?")
    for noun in SING[:10]:
        prompts.append(f"When the {noun} arrived, everyone was surprised")
        prompts.append(f"Although the {noun} was small, it was powerful")
        prompts.append(f"The {noun} that I saw was interesting")
    for verb in ["eat", "run", "think", "know", "want", "see", "hear", "feel"]:
        prompts.append(f"I {verb} the answer")
        prompts.append(f"You {verb} the answer")
        prompts.append(f"He {verb}s the answer")
        prompts.append(f"She {verb}s the answer")
        prompts.append(f"We {verb} the answer")
        prompts.append(f"They {verb} the answer")
    for noun in SING[:10]:
        prompts.append(f"The {noun} was seen by everyone")
        prompts.append(f"If the {noun} comes, we will be happy")
    OBJECTS = ["apple", "ball", "key", "lamp", "mirror", "rope", "clock",
               "blanket", "pillow", "hammer", "nail", "brush", "comb",
               "soap", "towel", "ring", "coin", "stamp", "letter", "map"]
    LOCATIONS = ["table", "shelf", "floor", "wall", "box", "bag", "drawer",
                 "closet", "garden", "kitchen", "bedroom", "office", "yard",
                 "street", "park", "school", "church", "market", "station", "bridge"]
    for obj in OBJECTS[:10]:
        for loc in LOCATIONS[:4]:
            prompts.append(f"The {obj} is on the {loc}")
    for subj in SING[:10]:
        for verb in ["eats", "finds", "sees", "loves", "hates"]:
            for obj in ["food", "water", "shelter", "friend", "answer"]:
                prompts.append(f"The {subj} {verb} the {obj}")
    for animal in ANIMALS[:10]:
        prompts.append(f"The {animal} was running in the field")
        prompts.append(f"A {animal} can be found in nature")
        prompts.append(f"I saw a {animal} near the lake")
        prompts.append(f"The {animal} is a living creature")
    for tool in TOOLS[:10]:
        prompts.append(f"The {tool} was lying on the workbench")
        prompts.append(f"A {tool} can be used for building")
        prompts.append(f"I need a {tool} for this project")
        prompts.append(f"The {tool} is made of metal or wood")
    for noun_s in SING[:20]:
        prompts.append(f"The {noun_s} is very interesting")
    for noun_p in PLUR[:20]:
        prompts.append(f"The {noun_p} are very interesting")
    for noun in SING[:10]:
        prompts.append(f"The {noun} that I saw was beautiful")
        prompts.append(f"The {noun} which was here is gone")
        prompts.append(f"The {noun} who came was friendly")
    for noun in SING[:10]:
        prompts.append(f"The {noun} does not exist")
        prompts.append(f"No {noun} was found there")

    random.seed(42)
    random.shuffle(prompts)
    return prompts[:n]


# ===== Model Loading =====

def load_model_bf16(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS, get_model_info

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (BF16 + device_map=auto + flash)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
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
            log_time(f"  Loaded with attn_implementation={attn_impl}")
            break
        except Exception as e:
            log_time(f"  {attn_impl} failed: {str(e)[:120]}, trying next...")
            continue

    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")

    model.eval()
    info = get_model_info(model, model_name)

    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  class={info.model_class}, layers={info.n_layers}, d_model={info.d_model}, "
             f"vocab={info.vocab_size}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, info


def get_input_device(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Core: Hidden State Collection =====

def collect_hidden_states(model, tokenizer, input_device, prompts, n_layers, model_name):
    """Collect last-token hidden states at each layer for all prompts."""
    import torch

    n_prompts = len(prompts)
    n_total = n_layers + 1
    hidden_states_per_layer = {l: [] for l in range(n_total)}

    log_time(f"Collecting hidden states for {n_prompts} prompts, {n_total} layers...")
    t_start = time.time()

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=True)

        last_pos = int(attn_mask.sum().item()) - 1

        for l in range(n_total):
            h = out.hidden_states[l][0, last_pos, :].detach().float().cpu().numpy()
            hidden_states_per_layer[l].append(h)

        del out
        torch.cuda.empty_cache()

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_prompts - i - 1) / rate if rate > 0 else 0
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log_time(f"  {i+1}/{n_prompts} done ({rate:.1f} prompt/s, ETA={eta:.0f}s, GPU={gpu_mem:.1f}GB)")

        gc.collect()

    return hidden_states_per_layer


# ===== Core: W_U Subspace Decomposition =====

def compute_wu_subspaces(W_U, n_components=200):
    """Compute V_vis and V_inv projection matrices from W_U SVD.

    Returns:
        Vt_vis: [n_components, d_model] — orthonormal basis of W_U row space (top-k)
        P_vis: [d_model, d_model] — projection matrix onto V_vis
        P_inv: [d_model, d_model] — projection matrix onto V_inv (orthogonal complement)
    """
    from sklearn.utils.extmath import randomized_svd

    log_time(f"Computing W_U subspaces (n_components={n_components})...")
    U, S, Vt = randomized_svd(W_U.astype(np.float32), n_components=n_components,
                               random_state=42)

    # Vt shape: [n_components, d_model]
    # V_vis is spanned by rows of Vt (top-k right singular vectors of W_U)
    Vt_vis = Vt  # [k, d_model]

    # Projection onto V_vis: P_vis = Vt_vis.T @ Vt_vis
    # Projection onto V_inv: P_inv = I - P_vis
    d_model = W_U.shape[1]
    log_time(f"  W_U SVD: top-{n_components} singular values capture "
             f"{np.sum(S[:n_components])**2 / np.sum(S**2) * 100:.1f}% of W_U energy")

    # For efficiency, we work with Vt_vis directly rather than forming full d_model x d_model matrices
    # Projection: h_vis = Vt_vis.T @ (Vt_vis @ h)
    # h_inv = h - h_vis

    return Vt_vis, S


def project_to_vis(h, Vt_vis):
    """Project hidden state(s) onto V_vis subspace.
    h: [N, d_model] or [d_model]
    Returns: h_vis [same shape]
    """
    if h.ndim == 1:
        coeffs = Vt_vis @ h  # [k]
        return Vt_vis.T @ coeffs
    else:
        coeffs = h @ Vt_vis.T  # [N, k]
        return coeffs @ Vt_vis  # [N, d_model]


def project_to_inv(h, Vt_vis):
    """Project hidden state(s) onto V_inv (orthogonal complement of V_vis).
    h: [N, d_model] or [d_model]
    Returns: h_inv [same shape]
    """
    return h - project_to_vis(h, Vt_vis)


# ===== Core: Subspace Transport Analysis =====

def compute_subspace_transport(hidden_states_per_layer, Vt_vis, n_total_layers):
    """
    THE DECISIVE EXPERIMENT:
    For each layer pair (l, l+1), measure how much V_inv(l) contributes
    to V_vis(l+1).

    Also: cumulative transport from each layer l to the final layer L.
    """
    log_time(f"\n{'='*70}")
    log_time("SUBSPACE TRANSPORT ANALYSIS")
    log_time(f"{'='*70}")

    n_total = n_total_layers
    results = {}

    # ===== 1. Adjacent-layer transport =====
    log_time("\n--- Adjacent-layer transport (l → l+1) ---")

    transport_adjacent = {}

    for l in range(n_total - 1):
        H_l = np.array(hidden_states_per_layer[l])       # [N, d_model]
        H_l1 = np.array(hidden_states_per_layer[l + 1])   # [N, d_model]

        N = H_l.shape[0]

        # Decompose h_l
        H_l_vis = project_to_vis(H_l, Vt_vis)   # [N, d_model]
        H_l_inv = project_to_inv(H_l, Vt_vis)    # [N, d_model]

        # Decompose h_{l+1}
        H_l1_vis = project_to_vis(H_l1, Vt_vis)  # [N, d_model]
        H_l1_inv = project_to_inv(H_l1, Vt_vis)  # [N, d_model]

        # Center all for regression
        H_l_vis_c = H_l_vis - H_l_vis.mean(axis=0)
        H_l_inv_c = H_l_inv - H_l_inv.mean(axis=0)
        H_l1_vis_c = H_l1_vis - H_l1_vis.mean(axis=0)

        # --- Metric 1: Variance decomposition ---
        # Total variance of h_{l+1}_vis
        total_var_l1_vis = np.sum(H_l1_vis_c ** 2)

        # Variance explained by h_l_vis (persistence)
        # R² of: h_{l+1}_vis = α h_l_vis + noise
        # Use multivariate regression: each dim of h_{l+1}_vis regressed on h_l_vis
        # For simplicity, compute R² as: 1 - ||residual||²/||target||²
        # using ridge regression

        from sklearn.linear_model import Ridge

        # Persistence: h_l_vis → h_{l+1}_vis
        ridge = Ridge(alpha=1.0)
        ridge.fit(H_l_vis_c, H_l1_vis_c)
        pred_vis = ridge.predict(H_l_vis_c)
        resid_vis = H_l1_vis_c - pred_vis
        r2_persistence = 1.0 - np.sum(resid_vis ** 2) / max(total_var_l1_vis, 1e-20)

        # Transport: h_l_inv → h_{l+1}_vis
        ridge2 = Ridge(alpha=1.0)
        ridge2.fit(H_l_inv_c, H_l1_vis_c)
        pred_inv = ridge2.predict(H_l_inv_c)
        resid_inv = H_l1_vis_c - pred_inv
        r2_transport = 1.0 - np.sum(resid_inv ** 2) / max(total_var_l1_vis, 1e-20)

        # Combined: both h_l_vis and h_l_inv → h_{l+1}_vis
        H_l_both = np.concatenate([H_l_vis_c, H_l_inv_c], axis=1)
        ridge3 = Ridge(alpha=1.0)
        ridge3.fit(H_l_both, H_l1_vis_c)
        pred_both = ridge3.predict(H_l_both)
        resid_both = H_l1_vis_c - pred_both
        r2_combined = 1.0 - np.sum(resid_both ** 2) / max(total_var_l1_vis, 1e-20)

        # --- Metric 2: Fraction of Δh in each subspace ---
        delta_h = H_l1 - H_l  # [N, d_model]
        delta_h_centered = delta_h - delta_h.mean(axis=0)
        total_delta_var = np.sum(delta_h_centered ** 2)

        if total_delta_var > 1e-20:
            delta_vis_frac = np.sum(project_to_vis(delta_h_centered, Vt_vis) ** 2) / total_delta_var
            delta_inv_frac = np.sum(project_to_inv(delta_h_centered, Vt_vis) ** 2) / total_delta_var
        else:
            delta_vis_frac = 0.0
            delta_inv_frac = 0.0

        # --- Metric 3: Angle between V_inv(l) update and V_vis ---
        # The update from V_inv(l) is: Δh_inv_contribution = h_{l+1} - h_l - (contribution of h_l_vis)
        # Approximate: just measure cosine of Δh with V_vis basis
        delta_h_vis_component = project_to_vis(delta_h_centered, Vt_vis)
        delta_h_inv_component = project_to_inv(delta_h_centered, Vt_vis)

        vis_norm = np.sum(delta_h_vis_component ** 2)
        inv_norm = np.sum(delta_h_inv_component ** 2)
        total_norm = vis_norm + inv_norm

        transport_ratio = r2_transport / max(r2_persistence + r2_transport, 1e-20)

        transport_adjacent[l] = {
            "r2_persistence": float(r2_persistence),
            "r2_transport": float(r2_transport),
            "r2_combined": float(r2_combined),
            "transport_ratio": float(transport_ratio),
            "delta_vis_frac": float(delta_vis_frac),
            "delta_inv_frac": float(delta_inv_frac),
        }

        if l < 3 or (l + 1) % 5 == 0 or l == n_total - 2:
            log_time(f"  L{l}→L{l+1}: persist_R²={r2_persistence:.4f}, "
                     f"transport_R²={r2_transport:.4f}, combined_R²={r2_combined:.4f}, "
                     f"transport_ratio={transport_ratio:.4f}, "
                     f"Δh_vis={delta_vis_frac:.3f}, Δh_inv={delta_inv_frac:.3f}")

    # ===== 2. Cumulative transport: layer l → final layer =====
    log_time("\n--- Cumulative transport (l → final layer L) ---")

    transport_cumulative = {}

    H_L = np.array(hidden_states_per_layer[n_total - 1])  # [N, d_model]
    H_L_vis = project_to_vis(H_L, Vt_vis)
    H_L_vis_c = H_L_vis - H_L_vis.mean(axis=0)
    total_var_L_vis = np.sum(H_L_vis_c ** 2)

    for l in range(n_total):
        H_l = np.array(hidden_states_per_layer[l])
        H_l_vis = project_to_vis(H_l, Vt_vis)
        H_l_inv = project_to_inv(H_l, Vt_vis)

        H_l_vis_c = H_l_vis - H_l_vis.mean(axis=0)
        H_l_inv_c = H_l_inv - H_l_inv.mean(axis=0)

        # Persistence: h_l_vis → h_L_vis
        ridge_p = Ridge(alpha=1.0)
        ridge_p.fit(H_l_vis_c, H_L_vis_c)
        pred_p = ridge_p.predict(H_l_vis_c)
        r2_persist_cum = 1.0 - np.sum((H_L_vis_c - pred_p) ** 2) / max(total_var_L_vis, 1e-20)

        # Transport: h_l_inv → h_L_vis
        ridge_t = Ridge(alpha=1.0)
        ridge_t.fit(H_l_inv_c, H_L_vis_c)
        pred_t = ridge_t.predict(H_l_inv_c)
        r2_transport_cum = 1.0 - np.sum((H_L_vis_c - pred_t) ** 2) / max(total_var_L_vis, 1e-20)

        # Combined
        H_l_both = np.concatenate([H_l_vis_c, H_l_inv_c], axis=1)
        ridge_b = Ridge(alpha=1.0)
        ridge_b.fit(H_l_both, H_L_vis_c)
        pred_b = ridge_b.predict(H_l_both)
        r2_combined_cum = 1.0 - np.sum((H_L_vis_c - pred_b) ** 2) / max(total_var_L_vis, 1e-20)

        transport_cumulative[l] = {
            "r2_persistence": float(r2_persist_cum),
            "r2_transport": float(r2_transport_cum),
            "r2_combined": float(r2_combined_cum),
            "transport_ratio": float(r2_transport_cum / max(r2_persist_cum + r2_transport_cum, 1e-20)),
        }

        if l < 3 or (l + 1) % 5 == 0 or l == n_total - 1:
            log_time(f"  L{l}→L{n_total-1}: persist_R²={r2_persist_cum:.4f}, "
                     f"transport_R²={r2_transport_cum:.4f}, combined_R²={r2_combined_cum:.4f}, "
                     f"transport_ratio={transport_cumulative[l]['transport_ratio']:.4f}")

    # ===== 3. Covariance spectrum (addressing PR critique) =====
    log_time("\n--- Covariance spectrum (addressing PR critique) ---")

    spectrum_data = {}
    for l in range(n_total):
        H = np.array(hidden_states_per_layer[l])
        H_c = H - H.mean(axis=0)
        N, d = H_c.shape

        # Use Gram matrix for N < d
        if N < d:
            G = H_c @ H_c.T
            eigs = np.linalg.eigvalsh(G)
            eigs = np.maximum(eigs, 0)
            eigs = np.sort(eigs)[::-1]
        else:
            C = H_c.T @ H_c
            eigs = np.linalg.eigvalsh(C)
            eigs = np.maximum(eigs, 0)
            eigs = np.sort(eigs)[::-1]

        total_var = np.sum(eigs)
        if total_var > 1e-20:
            cumvar = np.cumsum(eigs) / total_var
            # Full spectrum shape info
            pr = (np.sum(eigs) ** 2) / np.sum(eigs ** 2)

            # Key percentiles of the spectrum
            n_50 = int(np.searchsorted(cumvar, 0.50) + 1)
            n_90 = int(np.searchsorted(cumvar, 0.90) + 1)
            n_99 = int(np.searchsorted(cumvar, 0.99) + 1)

            # Tail energy: fraction in components beyond n_99
            tail_energy = 1.0 - cumvar[n_99 - 1] if n_99 < len(cumvar) else 0.0

            # Log-percentile spectrum
            percentiles = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]
            spec_at_pct = {}
            for p in percentiles:
                idx = int(np.searchsorted(cumvar, p))
                spec_at_pct[f"n_{int(p*1000)}"] = min(idx + 1, len(eigs))
        else:
            pr = 0
            n_50 = n_90 = n_99 = 0
            tail_energy = 0.0
            spec_at_pct = {}

        # Top 20 eigenvalues for visualization
        top_eigs = [float(x) for x in eigs[:20]]

        spectrum_data[l] = {
            "pr": float(pr),
            "n_50var": n_50,
            "n_90var": n_90,
            "n_99var": n_99,
            "tail_energy_pct": float(tail_energy * 100),
            "top20_eigenvalues": top_eigs,
            "n_samples": N,
            "d_model": d,
        }

        if l < 3 or (l + 1) % 5 == 0 or l == n_total - 1:
            log_time(f"  L{l}: PR={pr:.1f}, n50={n_50}, n90={n_90}, n99={n_99}, "
                     f"tail_energy={tail_energy*100:.2f}%")

    # ===== 4. Variance in V_vis vs V_inv per layer =====
    log_time("\n--- Variance decomposition per layer ---")

    var_decomp = {}
    for l in range(n_total):
        H = np.array(hidden_states_per_layer[l])
        H_c = H - H.mean(axis=0)
        total_var = np.sum(H_c ** 2)

        H_vis = project_to_vis(H_c, Vt_vis)
        H_inv = project_to_inv(H_c, Vt_vis)

        vis_var = np.sum(H_vis ** 2)
        inv_var = np.sum(H_inv ** 2)

        var_decomp[l] = {
            "total_var": float(total_var),
            "vis_var": float(vis_var),
            "inv_var": float(inv_var),
            "vis_frac": float(vis_var / max(total_var, 1e-20)),
            "inv_frac": float(inv_var / max(total_var, 1e-20)),
        }

        if l < 3 or (l + 1) % 5 == 0 or l == n_total - 1:
            log_time(f"  L{l}: vis_frac={var_decomp[l]['vis_frac']:.4f}, "
                     f"inv_frac={var_decomp[l]['inv_frac']:.4f}")

    results = {
        "adjacent_transport": transport_adjacent,
        "cumulative_transport": transport_cumulative,
        "spectrum": spectrum_data,
        "variance_decomposition": var_decomp,
    }

    return results


# ===== Main =====

def run_model(model_name):
    """Run the complete Phase 270 analysis for one model."""
    global _log_file
    _log_file = RESULT_DIR / f"{model_name}_log.txt"

    import torch
    from model_utils import get_W_U, release_model

    log_time(f"\n{'='*70}")
    log_time(f"Phase 270: SUBSPACE TRANSPORT — {model_name}")
    log_time(f"{'='*70}")

    # Generate prompts
    n_prompts = 1000
    prompts = generate_diverse_prompts(n=n_prompts)
    log_time(f"Generated {len(prompts)} diverse prompts")

    # Load model
    model, tokenizer, info = load_model_bf16(model_name)
    input_device = get_input_device(model)
    n_layers = info.n_layers
    d_model = info.d_model
    n_total = n_layers + 1

    # Step 1: Collect hidden states
    t0 = time.time()
    hidden_states = collect_hidden_states(model, tokenizer, input_device,
                                           prompts, n_layers, model_name)
    t_collect = time.time() - t0
    log_time(f"Hidden state collection: {t_collect:.1f}s")

    # Step 2: Get W_U
    log_time("Getting W_U...")
    W_U = get_W_U(model, model_name)
    log_time(f"  W_U shape={W_U.shape}")

    # Step 3: Compute W_U subspaces
    n_components = min(200, min(W_U.shape) - 1)
    Vt_vis, S_vis = compute_wu_subspaces(W_U, n_components=n_components)

    # Step 4: Run subspace transport analysis
    t0 = time.time()
    results = compute_subspace_transport(hidden_states, Vt_vis, n_total)
    t_transport = time.time() - t0
    log_time(f"Subspace transport analysis: {t_transport:.1f}s")

    # ===== Print Summary =====
    log_time(f"\n{'='*70}")
    log_time(f"SUMMARY — {model_name}")
    log_time(f"{'='*70}")

    # Key question: does V_inv at layer l predict V_vis at final layer?
    cum = results["cumulative_transport"]
    log_time("\nCUMULATIVE TRANSPORT (L{l} → Final):")
    log_time(f"{'Layer':>6} {'Persist_R2':>12} {'Transport_R2':>13} {'Combined_R2':>12} {'T_ratio':>10}")
    log_time("-" * 55)

    for l in range(0, n_total, max(1, n_total // 20)):
        c = cum.get(str(l), cum.get(l))
        if c:
            log_time(f"L{l:>4} {c['r2_persistence']:>12.4f} {c['r2_transport']:>13.4f} "
                     f"{c['r2_combined']:>12.4f} {c['transport_ratio']:>10.4f}")

    # Also print last layer explicitly
    c = cum.get(str(n_total - 1), cum.get(n_total - 1))
    if c:
        log_time(f"L{n_total-1:>4} {c['r2_persistence']:>12.4f} {c['r2_transport']:>13.4f} "
                 f"{c['r2_combined']:>12.4f} {c['transport_ratio']:>10.4f}")

    # Adjacent transport summary
    adj = results["adjacent_transport"]
    log_time("\nADJACENT TRANSPORT (L{l} → L{l+1}):")
    log_time(f"{'Layer':>6} {'Persist_R2':>12} {'Transport_R2':>13} {'T_ratio':>10} {'Δh_vis':>8} {'Δh_inv':>8}")
    log_time("-" * 60)

    for l in range(0, n_total - 1, max(1, (n_total - 1) // 20)):
        a = adj.get(str(l), adj.get(l))
        if a:
            log_time(f"L{l:>4}→L{l+1:<3} {a['r2_persistence']:>12.4f} {a['r2_transport']:>13.4f} "
                     f"{a['transport_ratio']:>10.4f} {a['delta_vis_frac']:>8.3f} {a['delta_inv_frac']:>8.3f}")

    # Diagnosis
    log_time("\nDIAGNOSIS:")

    # Check if cumulative transport grows across layers
    transport_values = []
    for l in range(n_total):
        c = cum.get(str(l), cum.get(l))
        if c:
            transport_values.append(c['r2_transport'])

    if len(transport_values) > 2:
        # Transport from early layers vs late layers
        early_transport = np.mean(transport_values[:len(transport_values)//4])
        late_transport = np.mean(transport_values[-len(transport_values)//4:])
        mid_transport = np.mean(transport_values[len(transport_values)//4:len(transport_values)//2])

        log_time(f"  Early-layer transport R² (L0-L{n_total//4}): {early_transport:.4f}")
        log_time(f"  Mid-layer transport R² (L{n_total//4}-L{n_total//2}): {mid_transport:.4f}")
        log_time(f"  Late-layer transport R² (L{n_total-n_total//4}-L{n_total-1}): {late_transport:.4f}")

        if early_transport > late_transport:
            log_time("  → Early V_inv is MORE predictive of final V_vis than late V_inv")
            log_time("  → Suggests V_inv at early layers gets ROTATED into V_vis at later layers")
            log_time("  → 'Dark matter' is partially 'delayed visible computation'")
        elif late_transport > early_transport * 2:
            log_time("  → Late V_inv is MUCH MORE predictive of final V_vis than early V_inv")
            log_time("  → V_inv stays mostly invisible; rotation is weak")
            log_time("  → Phase 268 finding stands: most computation stays in V_inv")
        else:
            log_time("  → Mixed signal: both early and late V_inv predict final V_vis similarly")

    # Save results
    # Convert integer keys to strings for JSON
    def stringify_keys(d):
        if isinstance(d, dict):
            return {str(k): stringify_keys(v) for k, v in d.items()}
        return d

    results_clean = stringify_keys(results)

    result_file = RESULT_DIR / f"{model_name}_subspace_transport.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False)
    log_time(f"Results saved: {result_file}")

    # Save hidden states for potential re-analysis
    hs_file = RESULT_DIR / f"{model_name}_hidden_states.npz"
    save_dict = {}
    for l in range(n_total):
        save_dict[f"L{l}"] = np.array(hidden_states[l])
    np.savez_compressed(hs_file, **save_dict)
    log_time(f"Hidden states saved: {hs_file.name}")

    # Release model
    release_model(model)
    del hidden_states
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released, GPU memory freed")

    _log_file = None
    return results_clean


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase270_subspace_transport.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in ("qwen3", "glm4", "deepseek7b"):
        print(f"Unknown model: {model_name}. Available: qwen3, glm4, deepseek7b")
        sys.exit(1)

    results = run_model(model_name)
    log_time(f"\nPhase 270 complete for {model_name}")
