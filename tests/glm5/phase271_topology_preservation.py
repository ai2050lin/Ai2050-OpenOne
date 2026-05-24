"""
Phase 271: Relational Topology Preservation & Transport R² Sanity Check
=====================================================================

Three core experiments addressing the critique:

A. Transport R² Permutation Test
   - Phase 270 found Transport R²≈1.0, but Ridge from high-dim (2360+) to low-dim (200)
     may be a regression artifact.
   - TEST: Permute input-output pairings. If R² drops to ~0, the original is real.
   - Also: Random subspace control — project onto random orthogonal subspace of same dim.

B. Cross-layer Topology Preservation (Mantel Test)
   - Core test of "relative encoding": are concept distances preserved across layers?
   - If topology is preserved despite coordinate rotation → relative encoding supported.
   - Within-category (apple↔banana) vs between-category (apple↔car) preservation.

C. V_vis vs V_inv Topology
   - Is relational structure preserved differently in V_vis vs V_inv?
   - Does V_inv maintain topology even though it's "not readable" by W_U?

Usage:
  python tests/glm5/phase271_topology_preservation.py qwen3
  python tests/glm5/phase271_topology_preservation.py glm4
  python tests/glm5/phase271_topology_preservation.py deepseek7b
"""
import sys, os, json, gc, time, warnings, random
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_utils import MODEL_CONFIGS, get_model_info, get_W_U

RESULT_DIR = Path("results/phase271_topology_preservation")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None

def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ===== Word Categories for Topology Analysis =====

CATEGORIES = {
    "fruits": [
        "apple", "banana", "orange", "grape", "mango",
        "pear", "peach", "cherry", "lemon", "lime",
        "plum", "kiwi", "fig", "guava", "papaya",
    ],
    "animals": [
        "dog", "cat", "lion", "tiger", "bear",
        "wolf", "fox", "deer", "horse", "cow",
        "rabbit", "mouse", "rat", "pig", "sheep",
    ],
    "vehicles": [
        "car", "bus", "train", "plane", "bike",
        "truck", "boat", "ship", "taxi", "van",
        "jeep", "tram", "sled", "raft", "canoe",
    ],
    "tools": [
        "hammer", "drill", "saw", "ruler", "knife",
        "wrench", "shovel", "pliers", "level", "clamp",
        "chisel", "rake", "mallet", "clamp", "spade",
    ],
    "body": [
        "head", "hand", "foot", "arm", "leg",
        "eye", "ear", "nose", "neck", "back",
        "chest", "knee", "elbow", "wrist", "ankle",
    ],
}

# Additional diverse prompts for Transport R² test (1000 prompts like Phase 270)
def generate_diverse_prompts(n=1000):
    """Generate diverse prompts for the Transport R² sanity check."""
    prompts = []
    SING = [
        "cat", "dog", "bird", "fish", "tree", "house", "car", "book",
        "phone", "chair", "table", "door", "river", "mountain", "cloud",
        "fire", "earth", "stone", "glass", "wood", "paper", "food",
        "king", "queen", "child", "woman", "man", "doctor", "teacher",
        "soldier", "artist", "writer", "singer", "farmer", "builder",
        "driver", "pilot", "baker", "hunter", "sailor", "nurse", "cook",
        "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit", "mouse",
    ]
    PLUR = [s + "s" for s in SING[:40]]
    VERBS_S = ["sits", "runs", "walks", "eats", "drinks", "sleeps", "thinks",
               "knows", "wants", "needs", "loves", "hates", "makes", "finds"]
    VERBS_P = ["sit", "run", "walk", "eat", "drink", "sleep", "think",
               "know", "want", "need", "love", "hate", "make", "find"]

    for noun in SING[:30]:
        for verb in VERBS_S[:4]:
            prompts.append(f"The {noun} {verb}")
    for noun in PLUR[:30]:
        for verb in VERBS_P[:4]:
            prompts.append(f"The {noun} {verb}")
    for noun in SING[:15]:
        prompts.append(f"The {noun} will go tomorrow")
        prompts.append(f"The {noun} went yesterday")
        prompts.append(f"The {noun} is going now")
    ADJS = ["big", "small", "red", "blue", "old", "new", "good", "bad",
            "fast", "slow", "hot", "cold", "dark", "bright"]
    for adj in ADJS[:10]:
        for noun in SING[:6]:
            prompts.append(f"The {adj} {noun} is here")
    for noun in SING[:20]:
        prompts.append(f"The {noun} is very interesting")
    for verb in ["eat", "run", "think", "know", "want", "see", "hear", "feel"]:
        for pronoun in ["I", "You", "He", "She", "We", "They"]:
            prompts.append(f"{pronoun} {verb} the answer")
    for noun in SING[:10]:
        prompts.append(f"The {noun} was seen by everyone")
        prompts.append(f"If the {noun} comes, we will be happy")
    for noun in SING[:20]:
        prompts.append(f"The {noun} does not exist")

    random.seed(42)
    random.shuffle(prompts)
    return prompts[:n]


# ===== Model Loading =====

def load_model_bf16(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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


# ===== Hidden State Collection =====

def collect_hidden_states(model, tokenizer, input_device, prompts, n_layers, batch_label=""):
    """Collect last-token hidden states at each layer for all prompts."""
    import torch

    n_prompts = len(prompts)
    n_total = n_layers + 1
    hidden_states_per_layer = {l: [] for l in range(n_total)}

    log_time(f"Collecting hidden states for {n_prompts} {batch_label} prompts, {n_total} layers...")
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
            log_time(f"  {batch_label} {i+1}/{n_prompts} ({rate:.1f} prompt/s, ETA={eta:.0f}s, GPU={gpu_mem:.1f}GB)")

        gc.collect()

    # Convert to arrays
    for l in range(n_total):
        hidden_states_per_layer[l] = np.array(hidden_states_per_layer[l])

    return hidden_states_per_layer


# ===== W_U Subspace Computation =====

def compute_wu_subspaces(model, model_name, n_components=200):
    """Compute W_U SVD and V_vis/V_inv decomposition."""
    W_U = get_W_U(model, model_name)
    log_time(f"W_U shape: {W_U.shape}")

    U, S, Vt = randomized_svd(W_U.astype(np.float32), n_components=n_components, random_state=42)
    Vt_vis = Vt  # [n_components, d_model]

    log_time(f"  W_U SVD: top-{n_components} singular values, "
             f"top-5: {S[:5].tolist()}, energy={np.sum(S[:n_components])**2 / np.sum(S**2) * 100:.1f}%")

    return Vt_vis, S


# ===== Experiment A: Transport R² Permutation Test =====

def experiment_a_transport_sanity(hidden_states, Vt_vis, n_layers, n_permutations=200):
    """
    Test whether Transport R²≈1.0 is a statistical artifact.

    Method: Permutation test
    - Original: Ridge(h_inv(l)) -> h_vis(L_final)
    - Shuffled: Break input-output pairing by permuting indices

    Also: Random subspace control
    - Instead of V_inv, use a random orthogonal subspace of same dimension
    - If random subspace also gives high R², V_inv is not special
    """
    import torch

    log_time("=" * 60)
    log_time("Experiment A: Transport R² Permutation Test")
    log_time("=" * 60)

    final_layer = n_layers  # Last hidden state layer index
    d_model = Vt_vis.shape[1]
    n_components = Vt_vis.shape[0]

    results = {}

    # Test layers
    test_layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]))

    for l in test_layers:
        log_time(f"  Testing Layer {l} -> Layer {final_layer}")

        h_l = hidden_states[l + 1]  # +1 because index 0 is embedding
        h_final = hidden_states[final_layer]

        # Project to subspaces
        h_l_vis_coeff = h_l @ Vt_vis.T  # [n_samples, n_components]
        h_l_vis_recon = h_l_vis_coeff @ Vt_vis
        h_l_inv = h_l - h_l_vis_recon  # [n_samples, d_model]

        h_final_vis_coeff = h_final @ Vt_vis.T  # [n_samples, n_components]

        n_samples = h_l.shape[0]

        # 1. Original Transport R²: h_l_inv -> h_final_vis
        ridge = Ridge(alpha=1.0)
        ridge.fit(h_l_inv, h_final_vis_coeff)
        r2_transport = ridge.score(h_l_inv, h_final_vis_coeff)

        # 2. Original Persistence R²: h_l_vis -> h_final_vis
        ridge_p = Ridge(alpha=1.0)
        ridge_p.fit(h_l_vis_coeff, h_final_vis_coeff)
        r2_persist = ridge_p.score(h_l_vis_coeff, h_final_vis_coeff)

        # 3. Combined: h_l -> h_final_vis
        ridge_c = Ridge(alpha=1.0)
        ridge_c.fit(h_l, h_final_vis_coeff)
        r2_combined = ridge_c.score(h_l, h_final_vis_coeff)

        # 4. Permutation test: shuffle output pairings
        log_time(f"    Running {n_permutations} permutations...")
        r2_shuffled = []
        for perm_i in range(n_permutations):
            perm_idx = np.random.permutation(n_samples)
            h_final_vis_shuffled = h_final_vis_coeff[perm_idx]
            ridge_s = Ridge(alpha=1.0)
            ridge_s.fit(h_l_inv, h_final_vis_shuffled)
            r2_shuffled.append(ridge_s.score(h_l_inv, h_final_vis_shuffled))

        r2_shuffled = np.array(r2_shuffled)
        p_value = float(np.mean(r2_shuffled >= r2_transport))

        # 5. Random subspace control: random orthogonal subspace of same dimension as V_inv
        #    (d_model - n_components dimensions)
        random_dim = d_model - n_components
        log_time(f"    Random subspace control (dim={random_dim})...")
        # Generate random orthonormal basis for random subspace
        np.random.seed(42)
        Q, _ = np.linalg.qr(np.random.randn(d_model, random_dim).astype(np.float32))

        # Project h_l onto random subspace
        h_l_random = h_l @ Q  # [n_samples, random_dim]

        ridge_rand = Ridge(alpha=1.0)
        ridge_rand.fit(h_l_random, h_final_vis_coeff)
        r2_random = ridge_rand.score(h_l_random, h_final_vis_coeff)

        # 6. Permutation test for random subspace
        r2_random_shuffled = []
        for perm_i in range(n_permutations):
            perm_idx = np.random.permutation(n_samples)
            h_final_vis_shuffled = h_final_vis_coeff[perm_idx]
            ridge_rs = Ridge(alpha=1.0)
            ridge_rs.fit(h_l_random, h_final_vis_shuffled)
            r2_random_shuffled.append(ridge_rs.score(h_l_random, h_final_vis_shuffled))

        r2_random_shuffled = np.array(r2_random_shuffled)
        p_value_random = float(np.mean(r2_random_shuffled >= r2_random))

        results[str(l)] = {
            "r2_transport_original": float(r2_transport),
            "r2_persist_original": float(r2_persist),
            "r2_combined_original": float(r2_combined),
            "r2_shuffled_mean": float(np.mean(r2_shuffled)),
            "r2_shuffled_std": float(np.std(r2_shuffled)),
            "r2_shuffled_max": float(np.max(r2_shuffled)),
            "p_value_transport": p_value,
            "r2_random_subspace": float(r2_random),
            "r2_random_shuffled_mean": float(np.mean(r2_random_shuffled)),
            "p_value_random": p_value_random,
            "n_samples": n_samples,
            "n_permutations": n_permutations,
        }

        log_time(f"    Transport R²: {r2_transport:.4f} (p={p_value:.4f})")
        log_time(f"    Persistence R²: {r2_persist:.4f}")
        log_time(f"    Combined R²: {r2_combined:.4f}")
        log_time(f"    Shuffled R²: {np.mean(r2_shuffled):.4f} ± {np.std(r2_shuffled):.4f} (max={np.max(r2_shuffled):.4f})")
        log_time(f"    Random subspace R²: {r2_random:.4f} (p={p_value_random:.4f})")
        log_time(f"    Random shuffled R²: {np.mean(r2_random_shuffled):.4f} ± {np.std(r2_random_shuffled):.4f}")

    return results


# ===== Experiment B: Cross-layer Topology Preservation =====

def experiment_b_topology_preservation(hidden_states, Vt_vis, n_layers, categories, word_to_idx):
    """
    Test whether relational structure is preserved across layers.

    Core test of "relative encoding":
    - Compute pairwise distance matrices at each layer
    - Measure cross-layer correlation (Mantel-like test using Spearman)
    - Compare within-category vs between-category preservation
    - Compare V_vis vs V_inv topology
    """
    log_time("=" * 60)
    log_time("Experiment B: Cross-layer Topology Preservation")
    log_time("=" * 60)

    n_samples = len(word_to_idx)
    n_total = n_layers + 1

    # ---- Step 1: Compute distance matrices at each layer ----
    log_time("  Computing distance matrices at each layer...")

    dist_full = {}    # Full space cosine distances
    dist_vis = {}     # V_vis subspace cosine distances
    dist_inv = {}     # V_inv subspace cosine distances

    for l in range(n_total):
        h_l = hidden_states[l]  # [n_samples, d_model]

        # Full space
        d_full = pdist(h_l, metric='cosine')
        dist_full[l] = squareform(d_full)

        # V_vis
        h_l_vis = h_l @ Vt_vis.T  # [n_samples, n_components]
        d_vis = pdist(h_l_vis, metric='cosine')
        dist_vis[l] = squareform(d_vis)

        # V_inv
        h_l_vis_recon = (h_l @ Vt_vis.T) @ Vt_vis
        h_l_inv = h_l - h_l_vis_recon
        d_inv = pdist(h_l_inv, metric='cosine')
        dist_inv[l] = squareform(d_inv)

    # ---- Step 2: Cross-layer topology correlation (Mantel test) ----
    log_time("  Computing cross-layer Mantel correlations...")

    reference_layer = n_layers  # Final layer
    test_layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]))

    upper_tri_idx = np.triu_indices(n_samples, k=1)

    mantel_results = {"full": {}, "vis": {}, "inv": {}}

    for l in test_layers:
        # Full space
        corr_full, p_full = spearmanr(
            dist_full[l][upper_tri_idx],
            dist_full[reference_layer][upper_tri_idx]
        )
        mantel_results["full"][str(l)] = {"spearman_r": float(corr_full), "p_value": float(p_full)}

        # V_vis space
        corr_vis, p_vis = spearmanr(
            dist_vis[l][upper_tri_idx],
            dist_vis[reference_layer][upper_tri_idx]
        )
        mantel_results["vis"][str(l)] = {"spearman_r": float(corr_vis), "p_value": float(p_vis)}

        # V_inv space
        corr_inv, p_inv = spearmanr(
            dist_inv[l][upper_tri_idx],
            dist_inv[reference_layer][upper_tri_idx]
        )
        mantel_results["inv"][str(l)] = {"spearman_r": float(corr_inv), "p_value": float(p_inv)}

        log_time(f"    L{l} vs L{reference_layer}: Full={corr_full:.4f}, V_vis={corr_vis:.4f}, V_inv={corr_inv:.4f}")

    # ---- Step 3: Within-category vs between-category preservation ----
    log_time("  Computing within vs between category preservation...")

    # Build index pairs
    within_pairs = []
    between_pairs = []
    same_category_pairs = {}  # {cat_name: [(i,j), ...]}

    for cat_name, words in categories.items():
        cat_indices = [word_to_idx[w] for w in words if w in word_to_idx]
        same_category_pairs[cat_name] = []
        for i_idx in range(len(cat_indices)):
            for j_idx in range(i_idx + 1, len(cat_indices)):
                within_pairs.append((cat_indices[i_idx], cat_indices[j_idx]))
                same_category_pairs[cat_name].append((cat_indices[i_idx], cat_indices[j_idx]))

    cat_names = list(categories.keys())
    for ci in range(len(cat_names)):
        for cj in range(ci + 1, len(cat_names)):
            indices_i = [word_to_idx[w] for w in categories[cat_names[ci]] if w in word_to_idx]
            indices_j = [word_to_idx[w] for w in categories[cat_names[cj]] if w in word_to_idx]
            for ii in indices_i:
                for jj in indices_j:
                    between_pairs.append((ii, jj))

    within_between_results = {}

    for l in test_layers:
        # Within-category distances
        within_dist_l = np.array([dist_full[l][i, j] for i, j in within_pairs])
        within_dist_ref = np.array([dist_full[reference_layer][i, j] for i, j in within_pairs])

        corr_within, p_within = spearmanr(within_dist_l, within_dist_ref) if len(within_pairs) > 2 else (0.0, 1.0)

        # Between-category distances
        between_dist_l = np.array([dist_full[l][i, j] for i, j in between_pairs])
        between_dist_ref = np.array([dist_full[reference_layer][i, j] for i, j in between_pairs])

        corr_between, p_between = spearmanr(between_dist_l, between_dist_ref) if len(between_pairs) > 2 else (0.0, 1.0)

        # Also in V_vis and V_inv
        within_vis_l = np.array([dist_vis[l][i, j] for i, j in within_pairs])
        within_vis_ref = np.array([dist_vis[reference_layer][i, j] for i, j in within_pairs])
        corr_within_vis, _ = spearmanr(within_vis_l, within_vis_ref) if len(within_pairs) > 2 else (0.0, 1.0)

        within_inv_l = np.array([dist_inv[l][i, j] for i, j in within_pairs])
        within_inv_ref = np.array([dist_inv[reference_layer][i, j] for i, j in within_pairs])
        corr_within_inv, _ = spearmanr(within_inv_l, within_inv_ref) if len(within_pairs) > 2 else (0.0, 1.0)

        between_vis_l = np.array([dist_vis[l][i, j] for i, j in between_pairs])
        between_vis_ref = np.array([dist_vis[reference_layer][i, j] for i, j in between_pairs])
        corr_between_vis, _ = spearmanr(between_vis_l, between_vis_ref) if len(between_pairs) > 2 else (0.0, 1.0)

        between_inv_l = np.array([dist_inv[l][i, j] for i, j in between_pairs])
        between_inv_ref = np.array([dist_inv[reference_layer][i, j] for i, j in between_pairs])
        corr_between_inv, _ = spearmanr(between_inv_l, between_inv_ref) if len(between_pairs) > 2 else (0.0, 1.0)

        within_between_results[str(l)] = {
            "within_full": float(corr_within),
            "between_full": float(corr_between),
            "within_vis": float(corr_within_vis),
            "within_inv": float(corr_within_inv),
            "between_vis": float(corr_between_vis),
            "between_inv": float(corr_between_inv),
        }

        log_time(f"    L{l}: Within={corr_within:.4f}, Between={corr_between:.4f}, "
                 f"Diff={corr_within - corr_between:.4f} | "
                 f"W_vis={corr_within_vis:.4f}, W_inv={corr_within_inv:.4f} | "
                 f"B_vis={corr_between_vis:.4f}, B_inv={corr_between_inv:.4f}")

    # ---- Step 4: Per-category topology preservation ----
    log_time("  Computing per-category topology preservation...")

    per_category_results = {}
    for cat_name, pairs in same_category_pairs.items():
        cat_result = {}
        for l in test_layers:
            if len(pairs) < 3:
                cat_result[str(l)] = {"full": 0.0, "vis": 0.0, "inv": 0.0, "n_pairs": len(pairs)}
                continue

            cat_dist_l = np.array([dist_full[l][i, j] for i, j in pairs])
            cat_dist_ref = np.array([dist_full[reference_layer][i, j] for i, j in pairs])
            corr_cat, _ = spearmanr(cat_dist_l, cat_dist_ref)

            cat_vis_l = np.array([dist_vis[l][i, j] for i, j in pairs])
            cat_vis_ref = np.array([dist_vis[reference_layer][i, j] for i, j in pairs])
            corr_cat_vis, _ = spearmanr(cat_vis_l, cat_vis_ref)

            cat_inv_l = np.array([dist_inv[l][i, j] for i, j in pairs])
            cat_inv_ref = np.array([dist_inv[reference_layer][i, j] for i, j in pairs])
            corr_cat_inv, _ = spearmanr(cat_inv_l, cat_inv_ref)

            cat_result[str(l)] = {
                "full": float(corr_cat),
                "vis": float(corr_cat_vis),
                "inv": float(corr_cat_inv),
                "n_pairs": len(pairs),
            }

        per_category_results[cat_name] = cat_result
        l0_str = str(test_layers[0])
        lf_str = str(test_layers[-1])
        if l0_str in cat_result and lf_str in cat_result:
            log_time(f"    {cat_name}: L0_full={cat_result[l0_str]['full']:.4f}, "
                     f"Lf_full={cat_result[lf_str]['full']:.4f}, "
                     f"L0_inv={cat_result[l0_str]['inv']:.4f}, "
                     f"Lf_inv={cat_result[lf_str]['inv']:.4f}")

    # ---- Step 5: Nearest neighbor preservation ----
    log_time("  Computing nearest neighbor preservation (k=5)...")

    k = 5
    nn_results = {}

    for l in test_layers:
        nn_preserve_count = 0
        nn_total = 0

        for i in range(n_samples):
            nn_l = set(np.argsort(dist_full[l][i])[1:k + 1])
            nn_ref = set(np.argsort(dist_full[reference_layer][i])[1:k + 1])
            overlap = len(nn_l & nn_ref)
            nn_preserve_count += overlap
            nn_total += k

        nn_frac = nn_preserve_count / nn_total if nn_total > 0 else 0
        nn_results[str(l)] = float(nn_frac)

        # Also in V_vis and V_inv
        nn_vis_count = 0
        nn_inv_count = 0
        for i in range(n_samples):
            nn_l_vis = set(np.argsort(dist_vis[l][i])[1:k + 1])
            nn_ref_vis = set(np.argsort(dist_vis[reference_layer][i])[1:k + 1])
            nn_vis_count += len(nn_l_vis & nn_ref_vis)

            nn_l_inv = set(np.argsort(dist_inv[l][i])[1:k + 1])
            nn_ref_inv = set(np.argsort(dist_inv[reference_layer][i])[1:k + 1])
            nn_inv_count += len(nn_l_inv & nn_ref_inv)

        nn_vis_frac = nn_vis_count / nn_total if nn_total > 0 else 0
        nn_inv_frac = nn_inv_count / nn_total if nn_total > 0 else 0

        log_time(f"    L{l}: NN_preserve Full={nn_frac:.4f}, V_vis={nn_vis_frac:.4f}, V_inv={nn_inv_frac:.4f}")

    # ---- Step 6: Layer-by-layer topology stability ----
    log_time("  Computing adjacent-layer topology stability...")

    adjacent_mantel = {}
    for l in range(n_total - 1):
        corr_adj, _ = spearmanr(
            dist_full[l][upper_tri_idx],
            dist_full[l + 1][upper_tri_idx]
        )
        corr_adj_vis, _ = spearmanr(
            dist_vis[l][upper_tri_idx],
            dist_vis[l + 1][upper_tri_idx]
        )
        corr_adj_inv, _ = spearmanr(
            dist_inv[l][upper_tri_idx],
            dist_inv[l + 1][upper_tri_idx]
        )
        adjacent_mantel[str(l)] = {
            "full": float(corr_adj),
            "vis": float(corr_adj_vis),
            "inv": float(corr_adj_inv),
        }

    # Print summary for adjacent layers (key layers only)
    for l in test_layers[:-1]:
        if str(l) in adjacent_mantel:
            a = adjacent_mantel[str(l)]
            log_time(f"    L{l}->L{l+1}: Full={a['full']:.4f}, V_vis={a['vis']:.4f}, V_inv={a['inv']:.4f}")

    return {
        "mantel_correlation": mantel_results,
        "within_between": within_between_results,
        "per_category": per_category_results,
        "nearest_neighbor": nn_results,
        "adjacent_stability": adjacent_mantel,
    }


# ===== Experiment C: V_vis vs V_inv Cross-space Topology Agreement =====

def experiment_c_cross_space_agreement(hidden_states, Vt_vis, n_layers, word_to_idx):
    """
    Does V_inv topology agree with V_vis topology at the same layer?
    If V_inv carries the same relational structure as V_vis → supports
    "computational workspace" interpretation.
    """
    log_time("=" * 60)
    log_time("Experiment C: V_vis vs V_inv Cross-space Topology Agreement")
    log_time("=" * 60)

    n_samples = len(word_to_idx)
    n_total = n_layers + 1
    upper_tri_idx = np.triu_indices(n_samples, k=1)

    results = {}
    test_layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]))

    for l in test_layers:
        h_l = hidden_states[l]
        h_l_vis = h_l @ Vt_vis.T
        h_l_vis_recon = h_l_vis @ Vt_vis
        h_l_inv = h_l - h_l_vis_recon

        # Distance matrices
        dist_full = squareform(pdist(h_l, metric='cosine'))
        dist_vis = squareform(pdist(h_l_vis, metric='cosine'))
        dist_inv = squareform(pdist(h_l_inv, metric='cosine'))

        # Cross-space correlations
        corr_full_vis, p_fv = spearmanr(dist_full[upper_tri_idx], dist_vis[upper_tri_idx])
        corr_full_inv, p_fi = spearmanr(dist_full[upper_tri_idx], dist_inv[upper_tri_idx])
        corr_vis_inv, p_vi = spearmanr(dist_vis[upper_tri_idx], dist_inv[upper_tri_idx])

        # Variance fractions
        vis_frac = np.sum(np.linalg.norm(h_l_vis, axis=1)**2) / np.sum(np.linalg.norm(h_l, axis=1)**2)
        inv_frac = 1.0 - vis_frac

        results[str(l)] = {
            "full_vis_spearman": float(corr_full_vis),
            "full_inv_spearman": float(corr_full_inv),
            "vis_inv_spearman": float(corr_vis_inv),
            "vis_variance_frac": float(vis_frac),
            "inv_variance_frac": float(inv_frac),
        }

        log_time(f"    L{l}: Full↔V_vis={corr_full_vis:.4f}, Full↔V_inv={corr_full_inv:.4f}, "
                 f"V_vis↔V_inv={corr_vis_inv:.4f} | var: vis={vis_frac:.3f}, inv={inv_frac:.3f}")

    return results


# ===== Main =====

def main():
    if len(sys.argv) < 2:
        print("Usage: python phase271_topology_preservation.py <model_key> [qwen3|glm4|deepseek7b]")
        sys.exit(1)

    model_key = sys.argv[1].lower()
    if model_key not in MODEL_CONFIGS:
        print(f"Unknown model: {model_key}. Choose from: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    global _log_file
    log_path = RESULT_DIR / f"{model_key}_phase271.log"
    _log_file = str(log_path)

    print(f"\n{'=' * 60}")
    print(f"Phase 271: Topology Preservation & Transport Sanity - {model_key}")
    print(f"{'=' * 60}")
    log_time(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- Load model ----
    model, tokenizer, info = load_model_bf16(model_key)
    n_layers = info.n_layers
    d_model = info.d_model
    input_device = get_input_device(model)

    # ---- Compute W_U subspaces ----
    N_COMPONENTS = 200
    Vt_vis, S_wu = compute_wu_subspaces(model, model_key, N_COMPONENTS)

    # ====================================================================
    # Part 1: Transport R² Sanity Check (diverse prompts, 1000)
    # ====================================================================
    log_time("\n--- Part 1: Transport R² Sanity Check (1000 diverse prompts) ---")

    diverse_prompts = generate_diverse_prompts(1000)
    hidden_states_diverse = collect_hidden_states(
        model, tokenizer, input_device, diverse_prompts, n_layers, batch_label="diverse"
    )

    sanity_results = experiment_a_transport_sanity(
        hidden_states_diverse, Vt_vis, n_layers, n_permutations=200
    )

    # Save Part 1 results immediately
    with open(RESULT_DIR / f"{model_key}_transport_sanity.json", 'w') as f:
        json.dump(sanity_results, f, indent=2, default=str)
    log_time(f"  Part 1 results saved.")

    # Free memory
    del hidden_states_diverse
    gc.collect()

    # ====================================================================
    # Part 2: Topology Preservation (category words, 75 sentences)
    # ====================================================================
    log_time("\n--- Part 2: Topology Preservation (category words) ---")

    # Build word sentences and index mapping
    all_words = []
    word_to_idx = {}
    word_sentences = []

    for cat_name, words in CATEGORIES.items():
        for word in words:
            if word in word_to_idx:
                continue
            idx = len(word_sentences)
            word_to_idx[word] = idx
            all_words.append(word)
            word_sentences.append(f"The {word} is")

    log_time(f"  Total unique words: {len(all_words)}")
    log_time(f"  Categories: {list(CATEGORIES.keys())}")
    log_time(f"  Sentences: {len(word_sentences)}")

    hidden_states_words = collect_hidden_states(
        model, tokenizer, input_device, word_sentences, n_layers, batch_label="words"
    )

    # Run topology experiments
    topo_results = experiment_b_topology_preservation(
        hidden_states_words, Vt_vis, n_layers, CATEGORIES, word_to_idx
    )

    cross_space_results = experiment_c_cross_space_agreement(
        hidden_states_words, Vt_vis, n_layers, word_to_idx
    )

    # Save Part 2 results
    topo_output = {
        "model": model_key,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_components": N_COMPONENTS,
        "n_words": len(all_words),
        "categories": {k: len(v) for k, v in CATEGORIES.items()},
        "experiment_b_topology": topo_results,
        "experiment_c_cross_space": cross_space_results,
    }

    with open(RESULT_DIR / f"{model_key}_topology_preservation.json", 'w') as f:
        json.dump(topo_output, f, indent=2, default=str)
    log_time(f"  Part 2 results saved.")

    # ---- Cleanup ----
    del model, tokenizer, hidden_states_words
    gc.collect()
    torch.cuda.empty_cache()

    log_time(f"Phase 271 complete for {model_key}")
    _log_file = None


if __name__ == "__main__":
    import torch
    main()
