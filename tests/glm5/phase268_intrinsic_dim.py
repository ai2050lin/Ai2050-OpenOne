"""
Phase 268: 决定性实验 — Hidden-State Intrinsic Dimension vs Logit-Lens Entropy
==============================================================================

这是整个研究系列的 DECISIVE EXPERIMENT。
Phase 266证明深层"语义轴"是低维空间平凡效应后，现在必须回答：

  "entropy collapse（熵塌缩）是真实维度塌缩，还是W_U投影假象？"

如果PR（参与率）高但eff_support低 → W_U投影假象 → "语义轴理论"崩塌
如果PR与eff_support同步下降 → 真实维度塌缩 → Goldilocks区真实

三个核心指标：
1. Participation Ratio (PR): hidden state空间的有效维度
2. Logit-lens entropy + effective support: Phase 265测量的
3. W_U alignment: hidden state方差在W_U可见方向中的比例

=== 用法 ===
  python tests/glm5/phase268_intrinsic_dim.py qwen3
  python tests/glm5/phase268_intrinsic_dim.py glm4
  python tests/glm5/phase268_intrinsic_dim.py deepseek7b
"""
import sys, os, json, gc, time, warnings, random
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase268_intrinsic_dim")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Logging =====
_log_file = None

def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ===== Prompt Generation (1000 diverse prompts) =====

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

    # Category 1: Subject-verb (number) — ~240 prompts
    for noun in SING[:30]:
        for verb in VERBS_S[:4]:
            prompts.append(f"The {noun} {verb}")
    for noun in PLUR[:30]:
        for verb in VERBS_P[:4]:
            prompts.append(f"The {noun} {verb}")

    # Category 2: Animacy — ~40 prompts
    for word in ANIMATE:
        prompts.append(f"The {word} thinks about tomorrow")
    for word in INANIMATE:
        prompts.append(f"The {word} sits on the shelf")

    # Category 3: Tense — ~60 prompts
    for noun in SING[:15]:
        prompts.append(f"The {noun} will go tomorrow")
        prompts.append(f"The {noun} went yesterday")
        prompts.append(f"The {noun} is going now")
        prompts.append(f"The {noun} has gone already")

    # Category 4: Adjective-noun — ~120 prompts
    for adj in ADJS[:20]:
        for noun in SING[:6]:
            prompts.append(f"The {adj} {noun} is here")

    # Category 5: Questions — ~40 prompts
    for noun in SING[:20]:
        prompts.append(f"Is the {noun} here?")
        prompts.append(f"Where is the {noun}?")

    # Category 6: Complex — ~30 prompts
    for noun in SING[:10]:
        prompts.append(f"When the {noun} arrived, everyone was surprised")
        prompts.append(f"Although the {noun} was small, it was powerful")
        prompts.append(f"The {noun} that I saw was interesting")

    # Category 7: Persons — ~48 prompts
    for verb in ["eat", "run", "think", "know", "want", "see", "hear", "feel"]:
        prompts.append(f"I {verb} the answer")
        prompts.append(f"You {verb} the answer")
        prompts.append(f"He {verb}s the answer")
        prompts.append(f"She {verb}s the answer")
        prompts.append(f"We {verb} the answer")
        prompts.append(f"They {verb} the answer")

    # Category 8: Passive/conditional — ~30 prompts
    for noun in SING[:10]:
        prompts.append(f"The {noun} was seen by everyone")
        prompts.append(f"If the {noun} comes, we will be happy")

    # Category 9: Objects/locations — ~40 prompts
    OBJECTS = ["apple", "ball", "key", "lamp", "mirror", "rope", "clock",
               "blanket", "pillow", "hammer", "nail", "brush", "comb",
               "soap", "towel", "ring", "coin", "stamp", "letter", "map"]
    LOCATIONS = ["table", "shelf", "floor", "wall", "box", "bag", "drawer",
                 "closet", "garden", "kitchen", "bedroom", "office", "yard",
                 "street", "park", "school", "church", "market", "station", "bridge"]
    for obj in OBJECTS[:10]:
        for loc in LOCATIONS[:4]:
            prompts.append(f"The {obj} is on the {loc}")

    # Category 10: Transitive — ~50 prompts
    for subj in SING[:10]:
        for verb in ["eats", "finds", "sees", "loves", "hates"]:
            for obj in ["food", "water", "shelter", "friend", "answer"]:
                prompts.append(f"The {subj} {verb} the {obj}")

    # Category 11: Semantic categories — ~120 prompts
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

    # Category 12: Number agreement — ~80 prompts
    for noun_s in SING[:20]:
        prompts.append(f"The {noun_s} is very interesting")
    for noun_p in PLUR[:20]:
        prompts.append(f"The {noun_p} are very interesting")

    # Category 13: Relative clauses — ~30 prompts
    for noun in SING[:10]:
        prompts.append(f"The {noun} that I saw was beautiful")
        prompts.append(f"The {noun} which was here is gone")
        prompts.append(f"The {noun} who came was friendly")

    # Category 14: Negation — ~20 prompts
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

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        devices = {}
        for k, v in dmap.items():
            dv = str(v)
            devices[dv] = devices.get(dv, 0) + 1
        log_time(f"  Device map: {devices}")

    return model, tokenizer, info


def get_input_device(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Core Measurements =====

def compute_participation_ratio(eigenvalues):
    """PR = (Σ λ)² / Σ λ² — effective number of significant dimensions"""
    eigenvalues = np.maximum(eigenvalues, 0)
    s1 = np.sum(eigenvalues)
    s2 = np.sum(eigenvalues ** 2)
    if s2 < 1e-20:
        return 0.0
    return (s1 ** 2) / s2


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

        # Intermediate save every 250 prompts
        if (i + 1) % 250 == 0:
            interim_file = RESULT_DIR / f"{model_name}_hidden_interim_{i+1}.npz"
            save_dict = {}
            for l in range(n_total):
                save_dict[f"L{l}"] = np.array(hidden_states_per_layer[l])
            np.savez_compressed(interim_file, **save_dict)
            log_time(f"  Intermediate saved: {interim_file.name}")

        gc.collect()

    # Final save
    final_file = RESULT_DIR / f"{model_name}_hidden_states.npz"
    save_dict = {}
    for l in range(n_total):
        save_dict[f"L{l}"] = np.array(hidden_states_per_layer[l])
    np.savez_compressed(final_file, **save_dict)
    log_time(f"  Full hidden states saved: {final_file.name} ({n_prompts} prompts x {n_total} layers)")

    return hidden_states_per_layer


def compute_layer_metrics(hidden_states_per_layer, n_total_layers):
    """Compute Participation Ratio and PCA spectrum for each layer."""
    log_time(f"Computing intrinsic dimension metrics for {n_total_layers} layers...")

    results = {}
    for l in range(n_total_layers):
        H = np.array(hidden_states_per_layer[l])  # (N, d_model)
        N, d = H.shape

        H_centered = H - H.mean(axis=0, keepdims=True)

        # Use Gram matrix approach when N < d (more efficient)
        if N < d:
            G = H_centered @ H_centered.T  # N x N
            eigenvalues = np.linalg.eigvalsh(G)
        else:
            C = H_centered.T @ H_centered  # d x d
            eigenvalues = np.linalg.eigvalsh(C)

        eigenvalues = np.maximum(eigenvalues, 0)
        sorted_eigs = np.sort(eigenvalues)[::-1]

        # Participation Ratio
        pr = compute_participation_ratio(eigenvalues)

        # PCA spectrum
        total_var = np.sum(sorted_eigs)
        if total_var > 1e-20:
            cumvar = np.cumsum(sorted_eigs) / total_var
            n90 = int(np.searchsorted(cumvar, 0.90) + 1)
            n95 = int(np.searchsorted(cumvar, 0.95) + 1)
            n99 = int(np.searchsorted(cumvar, 0.99) + 1)
        else:
            n90 = n95 = n99 = 0

        # Eigenvalue decay
        if sorted_eigs[0] > 1e-20:
            threshold_01 = sorted_eigs[0] * 0.01
            n_above_1pct = int(np.sum(sorted_eigs > threshold_01))
        else:
            n_above_1pct = 0

        # Top eigenvalue ratio
        if len(sorted_eigs) > 1 and sorted_eigs[1] > 1e-20:
            top_ratio = float(sorted_eigs[0] / sorted_eigs[1])
        else:
            top_ratio = 0.0

        results[l] = {
            "pr": float(pr),
            "n_samples": N,
            "d_model": d,
            "n_90var": n90,
            "n_95var": n95,
            "n_99var": n99,
            "n_above_1pct": n_above_1pct,
            "top_eigenvalue_ratio": top_ratio,
            "pr_capped": bool(pr > 0.8 * N),
            "top10_eigenvalues": [float(x) for x in sorted_eigs[:10]],
        }

        if l < 3 or (l + 1) % 5 == 0 or l == n_total_layers - 1:
            log_time(f"  L{l}: PR={pr:.1f}, n90={n90}, n95={n95}, n99={n99}, "
                     f"n>1%={n_above_1pct}, top_ratio={top_ratio:.1f}"
                     + (" [CAPPED-by-N]" if pr > 0.8 * N else ""))

    return results


def get_layer_norm_params(model, model_name):
    """Get final layer norm weight and bias for logit lens."""
    ln = None
    if model_name == "glm4":
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder'):
            enc = model.transformer.encoder
            if hasattr(enc, 'final_layernorm'):
                ln = enc.final_layernorm
    else:
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            ln = model.model.norm

    if ln is not None:
        w_tensor = ln.weight
        # Handle meta device (CPU offload with device_map=auto)
        if w_tensor.is_meta:
            log_time("  Layer norm weight on meta device, loading from safetensors...")
            from model_utils import MODEL_CONFIGS
            import glob
            from safetensors import safe_open
            model_path = MODEL_CONFIGS[model_name]["path"]
            sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
            for sf_file in sf_files:
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    for key in sf.keys():
                        if key == 'model.norm.weight':
                            w = sf.get_tensor(key).float().numpy()
                            b = None
                            if 'model.norm.bias' in sf.keys():
                                b = sf.get_tensor('model.norm.bias').float().numpy()
                            log_time(f"  Loaded layer norm from safetensors: weight shape={w.shape}")
                            return w, b
            log_time("  WARNING: Could not find layer norm in safetensors")
            return None, None

        w = w_tensor.detach().float().cpu().numpy()
        b = None
        if hasattr(ln, 'bias') and ln.bias is not None:
            if not ln.bias.is_meta:
                b = ln.bias.detach().float().cpu().numpy()
        return w, b

    log_time("  WARNING: Could not find final layer norm")
    return None, None


def layer_norm_numpy(h, weight, bias, eps=1e-5):
    mean = np.mean(h, axis=-1, keepdims=True)
    var = np.var(h, axis=-1, keepdims=True)
    h_norm = (h - mean) / np.sqrt(var + eps)
    result = weight * h_norm
    if bias is not None:
        result = result + bias
    return result


def compute_logit_entropy(hidden_states_per_layer, W_U, ln_weight, ln_bias, n_total_layers, chunk_size=50):
    """Compute logit-lens entropy and effective support at each layer."""
    vocab_size = W_U.shape[0]
    log_time(f"Computing logit-lens entropy for {n_total_layers} layers, vocab={vocab_size}...")

    results = {}

    for l in range(n_total_layers):
        H = np.array(hidden_states_per_layer[l])
        N = H.shape[0]

        entropies = []
        eff_supports = []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            H_chunk = H[start:end]

            if ln_weight is not None:
                H_proj = np.array([layer_norm_numpy(h, ln_weight, ln_bias) for h in H_chunk])
            else:
                H_proj = H_chunk

            logits = H_proj @ W_U.T

            logits_max = np.max(logits, axis=1, keepdims=True)
            shifted = logits - logits_max
            exp_vals = np.exp(shifted)
            probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

            log_probs = np.log(probs + 1e-20)
            ent = -np.sum(probs * log_probs, axis=1)
            entropies.extend(ent.tolist())

            sorted_probs = np.sort(probs, axis=1)[:, ::-1]
            cumprobs = np.cumsum(sorted_probs, axis=1)
            for i in range(end - start):
                eff_sup = int(np.searchsorted(cumprobs[i], 0.95) + 1)
                eff_supports.append(eff_sup)

        results[l] = {
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "median_entropy": float(np.median(entropies)),
            "mean_effective_support": float(np.mean(eff_supports)),
            "median_effective_support": float(np.median(eff_supports)),
        }

        if l < 3 or (l + 1) % 5 == 0 or l == n_total_layers - 1:
            log_time(f"  L{l}: entropy={results[l]['mean_entropy']:.2f}, "
                     f"eff_support={results[l]['mean_effective_support']:.0f}")

    return results


def compute_wu_alignment(hidden_states_per_layer, W_U, n_total_layers, n_components=200):
    """Compute fraction of hidden state variance visible to W_U at each layer."""
    from sklearn.utils.extmath import randomized_svd

    log_time(f"Computing W_U alignment (n_components={n_components})...")

    log_time(f"  Computing SVD of W_U (shape={W_U.shape})...")
    U, S, Vt = randomized_svd(W_U.astype(np.float32), n_components=n_components,
                               random_state=42)

    total_S = np.sum(S)
    cumvar_S = np.cumsum(S) / total_S
    wu_rank_90 = int(np.searchsorted(cumvar_S, 0.90) + 1)
    wu_rank_95 = int(np.searchsorted(cumvar_S, 0.95) + 1)
    wu_rank_99 = int(np.searchsorted(cumvar_S, 0.99) + 1)
    log_time(f"  W_U effective rank: n90={wu_rank_90}, n95={wu_rank_95}, n99={wu_rank_99}")

    results = {}
    for l in range(n_total_layers):
        H = np.array(hidden_states_per_layer[l])
        H_centered = H - H.mean(axis=0, keepdims=True)

        total_var = np.sum(H_centered ** 2)

        if total_var < 1e-20:
            results[l] = {"wu_alignment": 0.0, "wu_alignment_k": n_components}
            continue

        H_proj = H_centered @ Vt.T
        projected_var = np.sum(H_proj ** 2)

        alignment = float(projected_var / total_var)
        results[l] = {
            "wu_alignment": alignment,
            "wu_alignment_k": n_components,
        }

        if l < 3 or (l + 1) % 5 == 0 or l == n_total_layers - 1:
            log_time(f"  L{l}: W_U alignment={alignment:.4f} ({alignment*100:.1f}%)")

    results[0]["wu_rank_90"] = wu_rank_90
    results[0]["wu_rank_95"] = wu_rank_95
    results[0]["wu_rank_99"] = wu_rank_99

    return results


# ===== Main =====

def run_model(model_name):
    """Run the complete Phase 268 analysis for one model."""
    global _log_file
    _log_file = RESULT_DIR / f"{model_name}_log.txt"

    import torch
    from model_utils import get_W_U, release_model

    log_time(f"\n{'='*70}")
    log_time(f"Phase 268: DECISIVE — Intrinsic Dimension — {model_name}")
    log_time(f"{'='*70}")

    # Generate prompts (1000 for robust statistics)
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

    # Step 2: Compute intrinsic dimension (PR + PCA)
    t0 = time.time()
    dim_results = compute_layer_metrics(hidden_states, n_total)
    t_dim = time.time() - t0
    log_time(f"Intrinsic dimension computation: {t_dim:.1f}s")

    # Step 3: Get W_U and layer norm
    log_time("Getting W_U and layer norm...")
    W_U = get_W_U(model, model_name)
    log_time(f"  W_U shape={W_U.shape}, dtype={W_U.dtype}")

    ln_weight, ln_bias = get_layer_norm_params(model, model_name)
    if ln_weight is not None:
        log_time(f"  Layer norm: weight shape={ln_weight.shape}, "
                 f"bias={'yes' if ln_bias is not None else 'no'}")

    # Step 4: Compute logit-lens entropy
    t0 = time.time()
    entropy_results = compute_logit_entropy(hidden_states, W_U, ln_weight, ln_bias, n_total)
    t_entropy = time.time() - t0
    log_time(f"Logit entropy computation: {t_entropy:.1f}s")

    # Step 5: Compute W_U alignment
    t0 = time.time()
    alignment_results = compute_wu_alignment(hidden_states, W_U, n_total)
    t_align = time.time() - t0
    log_time(f"W_U alignment computation: {t_align:.1f}s")

    # Step 6: Print comparison table
    log_time(f"\n{'='*70}")
    log_time(f"=== DECISIVE: PR vs Effective Support vs W_U Alignment — {model_name} ===")
    log_time(f"{'='*70}")
    log_time(f"{'Layer':>6} {'PR':>8} {'n95':>5} {'n>1%':>6} {'Entropy':>8} {'EffSup':>8} "
             f"{'WU_Alin':>8} {'PR/ES':>8} {'Diagnosis':>22}")
    log_time("-" * 90)

    for l in range(n_total):
        pr = dim_results[l]["pr"]
        n95 = dim_results[l]["n_95var"]
        n_1pct = dim_results[l]["n_above_1pct"]
        entropy = entropy_results[l]["mean_entropy"]
        eff_sup = entropy_results[l]["mean_effective_support"]
        alignment = alignment_results[l]["wu_alignment"]

        pr_es_ratio = pr / max(eff_sup, 1)

        # Diagnosis
        if pr > 100 and eff_sup < 10:
            diag = "WU_ARTIFACT!"
        elif pr < 50 and eff_sup < 10:
            diag = "REAL_COLLAPSE"
        elif pr > 100 and eff_sup > 100:
            diag = "HIGH_DIM"
        elif alignment < 0.5 and pr > 100:
            diag = "WU_INVISIBLE"
        elif pr_es_ratio > 10:
            diag = "PR>>ES:ARTIFACT?"
        elif 50 <= pr <= 500 and 50 <= eff_sup <= 5000:
            diag = "GOLDILOCKS?"
        else:
            diag = f"PR={pr:.0f},ES={eff_sup:.0f}"

        log_time(f"L{l:>4} {pr:>8.1f} {n95:>5} {n_1pct:>6} {entropy:>8.2f} {eff_sup:>8.0f} "
                 f"{alignment:>8.3f} {pr_es_ratio:>8.1f} {diag:>22}")

    # Summary
    pr_first = dim_results[0]["pr"]
    pr_last = dim_results[n_total - 1]["pr"]
    es_first = entropy_results[0]["mean_effective_support"]
    es_last = entropy_results[n_total - 1]["mean_effective_support"]
    al_first = alignment_results[0]["wu_alignment"]
    al_last = alignment_results[n_total - 1]["wu_alignment"]

    pr_values = [dim_results[l]["pr"] for l in range(n_total)]
    min_pr = min(pr_values)
    min_pr_layer = pr_values.index(min_pr)

    # Find where PR drops below 50% of first layer
    pr_threshold = pr_first * 0.5
    collapse_layer = None
    for l in range(n_total):
        if dim_results[l]["pr"] < pr_threshold:
            collapse_layer = l
            break

    # Key diagnostic
    if pr_last > 100 and es_last < 10:
        conclusion = "W_U PROJECTION ARTIFACT: PR stays high but eff_support drops → entropy collapse is NOT real dimensional collapse"
    elif pr_last < 50 and es_last < 10:
        conclusion = "REAL DIMENSIONAL COLLAPSE: PR drops in sync with eff_support → Goldilocks zone is real"
    elif al_last < 0.5 and pr_last > 100:
        conclusion = "WU_INVISIBLE: Model computes in W_U-invisible directions → logit lens severely distorted"
    else:
        conclusion = f"MIXED: PR={pr_last:.0f}, eff_support={es_last:.0f}, alignment={al_last:.3f}"

    # Find Goldilocks candidates
    goldilocks_layers = []
    for l in range(n_total):
        pr = dim_results[l]["pr"]
        es = entropy_results[l]["mean_effective_support"]
        if 50 <= pr <= 500 and 50 <= es <= 5000:
            goldilocks_layers.append(l)

    # Combine all results
    combined = {}
    for l in range(n_total):
        combined[f"L{l}"] = {
            "layer": l,
            **dim_results[l],
            **entropy_results[l],
            **alignment_results[l],
        }

    # Save detailed results
    result_file = RESULT_DIR / f"{model_name}_full_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    log_time(f"Full results saved: {result_file}")

    # Save summary
    summary = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_prompts": len(prompts),
        "pr_first_layer": float(pr_first),
        "pr_last_layer": float(pr_last),
        "pr_min": float(min_pr),
        "pr_min_layer": int(min_pr_layer),
        "eff_support_first": float(es_first),
        "eff_support_last": float(es_last),
        "wu_alignment_first": float(al_first),
        "wu_alignment_last": float(al_last),
        "collapse_layer_50pct": collapse_layer,
        "goldilocks_layers": goldilocks_layers,
        "conclusion": conclusion,
        "timing": {
            "collect_s": round(t_collect, 1),
            "dim_s": round(t_dim, 1),
            "entropy_s": round(t_entropy, 1),
            "alignment_s": round(t_align, 1),
        }
    }

    summary_file = RESULT_DIR / f"{model_name}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_time(f"\n{'='*70}")
    log_time(f"CONCLUSION for {model_name}:")
    log_time(f"  {conclusion}")
    log_time(f"  PR: {pr_first:.0f} → {pr_last:.0f} (min={min_pr:.0f} at L{min_pr_layer})")
    log_time(f"  EffSupport: {es_first:.0f} → {es_last:.0f}")
    log_time(f"  WU_Alignment: {al_first:.3f} → {al_last:.3f}")
    if goldilocks_layers:
        log_time(f"  Goldilocks候选层: {goldilocks_layers}")
    log_time(f"{'='*70}")

    # Release model
    release_model(model)
    del hidden_states
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released, GPU memory freed")

    _log_file = None
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase268_intrinsic_dim.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in ("qwen3", "glm4", "deepseek7b"):
        print(f"Unknown model: {model_name}. Available: qwen3, glm4, deepseek7b")
        sys.exit(1)

    summary = run_model(model_name)
    log_time(f"\nPhase 268 complete for {model_name}")
