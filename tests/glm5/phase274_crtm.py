"""
Phase 274: CRTM — Conditional Routing Topology Mapping
========================================================

Core shift: From NOISING (destructive) to RESAMPLING (constructive).

Key insight: Word embeddings are "conditional keys", not "semantic vectors".
They don't store meaning — they CONTROL routing through the computational graph.

Algorithm layers:
A. Routing Activation Map — How does each token change the network state?
   Δ(W, L) = h_L("The W is") - h_L("The [neutral] is")
   This is the "routing fingerprint" of each word at each layer.

B. Activation Resampling (Patching) — THE core causal experiment.
   For pair (A, B): at layer L, replace A's residual with B's residual.
   Measure "concept shift": how much does the output move toward B?
   Layer with maximum shift = causal divergence layer.

C. Path Reuse Ratio — Compare routing fingerprints between within/between pairs.
   Reuse(A,B) = overlap of top-K active dimensions in Δ(A,L) and Δ(B,L).

D. Conditional Routing for Ambiguous Words — Same word, different contexts.
   "river bank" vs "bank account" — where do they diverge?

Usage:
  python tests/glm5/phase274_crtm.py qwen3
  python tests/glm5/phase274_crtm.py glm4
  python tests/glm5/phase274_crtm.py deepseek7b
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

RESULT_DIR = Path("results/phase274_crtm")
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
    "fruits": ["apple", "banana", "orange", "grape", "mango", "pear", "peach", "cherry", "lemon", "lime"],
    "animals": ["dog", "cat", "lion", "tiger", "bear", "wolf", "fox", "deer", "horse", "cow"],
    "vehicles": ["car", "bus", "train", "plane", "bike", "truck", "boat", "ship", "taxi", "van"],
}

WITHIN_PAIRS = [
    ("apple", "banana"), ("apple", "orange"), ("banana", "grape"),
    ("dog", "cat"), ("lion", "tiger"), ("wolf", "fox"),
    ("car", "bus"), ("train", "plane"), ("bike", "truck"),
    ("peach", "cherry"), ("bear", "wolf"), ("boat", "ship"),
]

BETWEEN_PAIRS = [
    ("apple", "dog"), ("banana", "car"), ("orange", "lion"),
    ("grape", "train"), ("mango", "bike"), ("peach", "wolf"),
    ("cherry", "fox"), ("lemon", "bus"), ("pear", "tiger"),
    ("lime", "ship"), ("apple", "car"), ("dog", "bus"),
]

# Ambiguous words with context pairs
AMBIGUOUS_PAIRS = [
    ("The fish sat on the river bank", "She deposited money in the bank"),
    ("The light from the sun was bright", "The box was very light to carry"),
    ("He likes to play football", "She wants to play the piano"),
    ("The bark of the dog was loud", "The bark of the tree was rough"),
    ("The spring season brings flowers", "The metal spring broke under pressure"),
    ("The bat flew in the cave", "He swung the baseball bat"),
    ("She broke the glass window", "He drank from a glass"),
    ("A fast runner won the race", "The human race spread globally"),
]

# Neutral baseline word — used to compute routing activation
NEUTRAL_WORD = "thing"


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


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Core Algorithm 1: Routing Activation Map =====

def compute_routing_activation(model, tokenizer, input_device, word, n_layers,
                                neutral_word=NEUTRAL_WORD, template="The {} is"):
    """
    Compute the routing activation map for a word.

    Δ(W, L) = h_L("The W is") - h_L("The [neutral] is")

    This measures what the word W "adds" to the computation at each layer,
    compared to a neutral baseline. This is the "routing fingerprint".

    Returns:
        routing_map: dict {layer_idx: np.array [d_model]}
        Also returns full hidden states for both conditions.
    """
    prompt_w = template.format(word)
    prompt_neutral = template.format(neutral_word)

    inputs_w = tokenizer(prompt_w, return_tensors="pt", truncation=True, max_length=64)
    inputs_n = tokenizer(prompt_neutral, return_tensors="pt", truncation=True, max_length=64)

    input_ids_w = inputs_w["input_ids"].to(input_device)
    attn_w = inputs_w["attention_mask"].to(input_device)
    input_ids_n = inputs_n["input_ids"].to(input_device)
    attn_n = inputs_n["attention_mask"].to(input_device)

    with torch.no_grad():
        out_w = model(input_ids=input_ids_w, attention_mask=attn_w, output_hidden_states=True)
        out_n = model(input_ids=input_ids_n, attention_mask=attn_n, output_hidden_states=True)

    hs_w = out_w.hidden_states  # tuple of [1, seq_len, d_model]
    hs_n = out_n.hidden_states

    routing_map = {}
    seq_pos = -1  # last token position

    for li in range(min(len(hs_w), len(hs_n))):
        # Get last token position for each
        pos_w = int(attn_w.sum().item()) - 1
        pos_n = int(attn_n.sum().item()) - 1
        delta = hs_w[li][0, pos_w, :].float().cpu().numpy() - hs_n[li][0, pos_n, :].float().cpu().numpy()
        routing_map[li] = delta

    del out_w, out_n, hs_w, hs_n
    torch.cuda.empty_cache()

    return routing_map


# ===== Core Algorithm 2: Activation Resampling (Patching) =====

def compute_resampling_effect(model, tokenizer, input_device,
                               word_A, word_B, n_layers,
                               template="The {} is"):
    """
    Activation Resampling (Patching):
    At each layer L, replace A's residual with B's residual.
    Measure "concept shift": how much does the output move toward B?

    Method:
    1. Run "The A is" cleanly → get all residuals and logit_A
    2. Run "The B is" cleanly → get all residuals and logit_B
    3. For each layer L:
       a. Run "The A is" but at layer L, replace residual with B's residual
       b. Measure logit(A_token) and logit(B_token) in the patched output
       c. Concept shift = logit(B_token) - logit(A_token) after patching
          (positive = moved toward B)

    Returns:
        resampling_map: dict {layer_idx: {concept_shift, logit_A_patched, logit_B_patched}}
    """
    prompt_A = template.format(word_A)
    prompt_B = template.format(word_B)

    inputs_A = tokenizer(prompt_A, return_tensors="pt", truncation=True, max_length=64)
    inputs_B = tokenizer(prompt_B, return_tensors="pt", truncation=True, max_length=64)

    input_ids_A = inputs_A["input_ids"].to(input_device)
    attn_A = inputs_A["attention_mask"].to(input_device)
    input_ids_B = inputs_B["input_ids"].to(input_device)
    attn_B = inputs_B["attention_mask"].to(input_device)

    # Target token IDs
    tok_A = tokenizer.encode(" " + word_A, add_special_tokens=False)
    if not tok_A:
        tok_A = tokenizer.encode(word_A, add_special_tokens=False)
    target_A = tok_A[0]

    tok_B = tokenizer.encode(" " + word_B, add_special_tokens=False)
    if not tok_B:
        tok_B = tokenizer.encode(word_B, add_special_tokens=False)
    target_B = tok_B[0]

    # Step 1: Clean forward passes
    with torch.no_grad():
        out_A = model(input_ids=input_ids_A, attention_mask=attn_A, output_hidden_states=True)
        out_B = model(input_ids=input_ids_B, attention_mask=attn_B, output_hidden_states=True)

    clean_logit_A = float(out_A.logits[0, -1, target_A].item())
    clean_logit_B = float(out_A.logits[0, -1, target_B].item())  # logit of B in A's context
    clean_logit_B_in_B = float(out_B.logits[0, -1, target_B].item())
    clean_logit_A_in_B = float(out_B.logits[0, -1, target_A].item())

    # Collect clean residuals at each layer input
    layers = get_layers(model)
    clean_residuals_A = {}
    clean_residuals_B = {}

    def make_capture_hook(store_dict, layer_idx):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                store_dict[layer_idx] = input[0].detach().clone()
        return hook

    # Capture A's residuals
    hooks_A = [layers[li].register_forward_hook(make_capture_hook(clean_residuals_A, li))
               for li in range(n_layers)]
    with torch.no_grad():
        model(input_ids=input_ids_A, attention_mask=attn_A)
    for h in hooks_A:
        h.remove()

    # Capture B's residuals
    hooks_B = [layers[li].register_forward_hook(make_capture_hook(clean_residuals_B, li))
               for li in range(n_layers)]
    with torch.no_grad():
        model(input_ids=input_ids_B, attention_mask=attn_B)
    for h in hooks_B:
        h.remove()

    del out_A, out_B
    torch.cuda.empty_cache()

    # Step 2: Resampling at each layer
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    resampling_map = {}

    for layer_idx in sample_layers:
        if layer_idx not in clean_residuals_B:
            continue

        # Create patching hook: at layer_idx, replace A's residual with B's residual
        patched_residual = clean_residuals_B[layer_idx].clone()

        def make_patch_hook(stored_residual, target_device):
            def hook(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    modified = list(input)
                    modified[0] = stored_residual.to(input[0].device, dtype=input[0].dtype)
                    return tuple(modified)
                return input
            return hook

        patch_hook = layers[layer_idx].register_forward_hook(
            make_patch_hook(patched_residual, input_device)
        )

        with torch.no_grad():
            patched_out = model(input_ids=input_ids_A, attention_mask=attn_A)

        patched_logit_A = float(patched_out.logits[0, -1, target_A].item())
        patched_logit_B = float(patched_out.logits[0, -1, target_B].item())

        # Concept shift: positive = moved toward B
        concept_shift = (patched_logit_B - patched_logit_A) - (clean_logit_B - clean_logit_A)

        # Also measure: how much did logit_A decrease? (A was "erased")
        logit_A_change = patched_logit_A - clean_logit_A
        logit_B_change = patched_logit_B - clean_logit_B

        resampling_map[str(layer_idx)] = {
            "concept_shift": concept_shift,
            "logit_A_change": logit_A_change,
            "logit_B_change": logit_B_change,
            "patched_logit_A": patched_logit_A,
            "patched_logit_B": patched_logit_B,
        }

        patch_hook.remove()
        del patched_out, patched_residual
        torch.cuda.empty_cache()

    del clean_residuals_A, clean_residuals_B
    torch.cuda.empty_cache()

    resampling_map["_clean_baseline"] = {
        "clean_logit_A": clean_logit_A,
        "clean_logit_B_in_A": clean_logit_B,
        "clean_logit_B_in_B": clean_logit_B_in_B,
        "clean_logit_A_in_B": clean_logit_A_in_B,
    }

    return resampling_map


# ===== Experiment A: Routing Activation Maps =====

def run_experiment_a(model, tokenizer, input_device, model_info):
    """
    Build routing activation maps for all words.
    Compare within-category vs between-category routing fingerprint similarity.
    """
    n_layers = model_info.n_layers
    all_words = []
    for cat_words in CATEGORIES.values():
        all_words.extend(cat_words)

    log_time(f"Exp A: Routing Activation Maps, {len(all_words)} words × {n_layers} layers")

    # Compute routing maps
    routing_maps = {}
    for wi, word in enumerate(all_words):
        rmap = compute_routing_activation(model, tokenizer, input_device, word, n_layers)
        routing_maps[word] = rmap

        if (wi + 1) % 10 == 0:
            log_time(f"  [{wi+1}/{len(all_words)}] routing maps done, "
                     f"GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Compute routing fingerprint similarity
    from scipy.stats import spearmanr

    # Sample layers for analysis
    sample_layers = list(range(0, n_layers, max(1, n_layers // 10)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)

    within_results = []
    between_results = []

    for a, b in WITHIN_PAIRS:
        if a in routing_maps and b in routing_maps:
            layer_corrs = []
            for li in sample_layers:
                if li in routing_maps[a] and li in routing_maps[b]:
                    delta_a = routing_maps[a][li]
                    delta_b = routing_maps[b][li]
                    r, _ = spearmanr(delta_a, delta_b)
                    layer_corrs.append(r)

            # Also compute cosine similarity of delta vectors per layer
            layer_cosines = []
            for li in sample_layers:
                if li in routing_maps[a] and li in routing_maps[b]:
                    delta_a = routing_maps[a][li]
                    delta_b = routing_maps[b][li]
                    na = np.linalg.norm(delta_a)
                    nb = np.linalg.norm(delta_b)
                    if na > 1e-10 and nb > 1e-10:
                        cos = float(np.dot(delta_a, delta_b) / (na * nb))
                        layer_cosines.append(cos)

            # Jaccard overlap of top-K dimensions
            K = 50
            layer_jaccards = []
            for li in sample_layers:
                if li in routing_maps[a] and li in routing_maps[b]:
                    top_a = set(np.argsort(np.abs(routing_maps[a][li]))[-K:])
                    top_b = set(np.argsort(np.abs(routing_maps[b][li]))[-K:])
                    jac = len(top_a & top_b) / len(top_a | top_b)
                    layer_jaccards.append(jac)

            within_results.append({
                "pair": f"{a}_{b}",
                "mean_spearman": float(np.mean(layer_corrs)) if layer_corrs else 0,
                "mean_cosine": float(np.mean(layer_cosines)) if layer_cosines else 0,
                "mean_jaccard": float(np.mean(layer_jaccards)) if layer_jaccards else 0,
                "layer_spearmans": {str(l): float(c) for l, c in zip(sample_layers, layer_corrs)},
                "layer_cosines": {str(l): float(c) for l, c in zip(sample_layers, layer_cosines)},
                "layer_jaccards": {str(l): float(c) for l, c in zip(sample_layers, layer_jaccards)},
            })

    for a, b in BETWEEN_PAIRS:
        if a in routing_maps and b in routing_maps:
            layer_corrs = []
            for li in sample_layers:
                if li in routing_maps[a] and li in routing_maps[b]:
                    delta_a = routing_maps[a][li]
                    delta_b = routing_maps[b][li]
                    r, _ = spearmanr(delta_a, delta_b)
                    layer_corrs.append(r)

            layer_cosines = []
            for li in sample_layers:
                if li in routing_maps[a] and li in routing_maps[b]:
                    delta_a = routing_maps[a][li]
                    delta_b = routing_maps[b][li]
                    na = np.linalg.norm(delta_a)
                    nb = np.linalg.norm(delta_b)
                    if na > 1e-10 and nb > 1e-10:
                        cos = float(np.dot(delta_a, delta_b) / (na * nb))
                        layer_cosines.append(cos)

            K = 50
            layer_jaccards = []
            for li in sample_layers:
                if li in routing_maps[a] and li in routing_maps[b]:
                    top_a = set(np.argsort(np.abs(routing_maps[a][li]))[-K:])
                    top_b = set(np.argsort(np.abs(routing_maps[b][li]))[-K:])
                    jac = len(top_a & top_b) / len(top_a | top_b)
                    layer_jaccards.append(jac)

            between_results.append({
                "pair": f"{a}_{b}",
                "mean_spearman": float(np.mean(layer_corrs)) if layer_corrs else 0,
                "mean_cosine": float(np.mean(layer_cosines)) if layer_cosines else 0,
                "mean_jaccard": float(np.mean(layer_jaccards)) if layer_jaccards else 0,
                "layer_spearmans": {str(l): float(c) for l, c in zip(sample_layers, layer_corrs)},
                "layer_cosines": {str(l): float(c) for l, c in zip(sample_layers, layer_cosines)},
                "layer_jaccards": {str(l): float(c) for l, c in zip(sample_layers, layer_jaccards)},
            })

    # Summary
    within_spearmans = [r["mean_spearman"] for r in within_results]
    between_spearmans = [r["mean_spearman"] for r in between_results]
    within_cosines = [r["mean_cosine"] for r in within_results]
    between_cosines = [r["mean_cosine"] for r in between_results]
    within_jaccards = [r["mean_jaccard"] for r in within_results]
    between_jaccards = [r["mean_jaccard"] for r in between_results]

    log_time(f"  Within: spearman={np.mean(within_spearmans):.4f}, "
             f"cosine={np.mean(within_cosines):.4f}, jaccard={np.mean(within_jaccards):.4f}")
    log_time(f"  Between: spearman={np.mean(between_spearmans):.4f}, "
             f"cosine={np.mean(between_cosines):.4f}, jaccard={np.mean(between_jaccards):.4f}")

    # Per-layer analysis: compute mean within/between at each layer
    per_layer_results = {}
    for li in sample_layers:
        w_sp = [r["layer_spearmans"].get(str(li), 0) for r in within_results]
        b_sp = [r["layer_spearmans"].get(str(li), 0) for r in between_results]
        w_cos = [r["layer_cosines"].get(str(li), 0) for r in within_results]
        b_cos = [r["layer_cosines"].get(str(li), 0) for r in between_results]
        w_jac = [r["layer_jaccards"].get(str(li), 0) for r in within_results]
        b_jac = [r["layer_jaccards"].get(str(li), 0) for r in between_results]

        per_layer_results[str(li)] = {
            "within_spearman": float(np.mean(w_sp)),
            "between_spearman": float(np.mean(b_sp)),
            "within_cosine": float(np.mean(w_cos)),
            "between_cosine": float(np.mean(b_cos)),
            "within_jaccard": float(np.mean(w_jac)),
            "between_jaccard": float(np.mean(b_jac)),
        }

    return {
        "within_pairs": within_results,
        "between_pairs": between_results,
        "per_layer": per_layer_results,
        "within_summary": {
            "spearman_mean": float(np.mean(within_spearmans)),
            "cosine_mean": float(np.mean(within_cosines)),
            "jaccard_mean": float(np.mean(within_jaccards)),
        },
        "between_summary": {
            "spearman_mean": float(np.mean(between_spearmans)),
            "cosine_mean": float(np.mean(between_cosines)),
            "jaccard_mean": float(np.mean(between_jaccards)),
        },
    }


# ===== Experiment B: Activation Resampling (Patching) =====

def run_experiment_b(model, tokenizer, input_device, model_info):
    """
    Core causal experiment: resampling/patching at each layer.
    For within and between pairs, measure concept shift.
    """
    n_layers = model_info.n_layers

    log_time(f"Exp B: Activation Resampling (Patching)")

    within_results = []
    between_results = []

    for pi, (a, b) in enumerate(WITHIN_PAIRS):
        log_time(f"  Within pair ({pi+1}/{len(WITHIN_PAIRS)}): {a}↔{b}")
        ret = compute_resampling_effect(model, tokenizer, input_device, a, b, n_layers)
        ret["pair"] = f"{a}_{b}"
        ret["type"] = "within"
        within_results.append(ret)
        gc.collect()
        torch.cuda.empty_cache()

    for pi, (a, b) in enumerate(BETWEEN_PAIRS):
        log_time(f"  Between pair ({pi+1}/{len(BETWEEN_PAIRS)}): {a}↔{b}")
        ret = compute_resampling_effect(model, tokenizer, input_device, a, b, n_layers)
        ret["pair"] = f"{a}_{b}"
        ret["type"] = "between"
        between_results.append(ret)
        gc.collect()
        torch.cuda.empty_cache()

    # Analyze: for each pair, find the layer with maximum concept shift
    within_shifts = []
    between_shifts = []
    within_peak_layers = []
    between_peak_layers = []

    for r in within_results:
        shifts = {l: r[l]["concept_shift"] for l in r if l.isdigit()}
        if shifts:
            peak_layer = max(shifts, key=lambda l: abs(shifts[l]))
            within_shifts.append(abs(shifts[peak_layer]))
            within_peak_layers.append(int(peak_layer))

    for r in between_results:
        shifts = {l: r[l]["concept_shift"] for l in r if l.isdigit()}
        if shifts:
            peak_layer = max(shifts, key=lambda l: abs(shifts[l]))
            between_shifts.append(abs(shifts[peak_layer]))
            between_peak_layers.append(int(peak_layer))

    log_time(f"  Within peak shift: {np.mean(within_shifts):.4f} at mean layer {np.mean(within_peak_layers):.1f}")
    log_time(f"  Between peak shift: {np.mean(between_shifts):.4f} at mean layer {np.mean(between_peak_layers):.1f}")

    return {
        "within_pairs": within_results,
        "between_pairs": between_results,
        "within_peak_shift": float(np.mean(within_shifts)) if within_shifts else 0,
        "between_peak_shift": float(np.mean(between_shifts)) if between_shifts else 0,
        "within_peak_layer": float(np.mean(within_peak_layers)) if within_peak_layers else 0,
        "between_peak_layer": float(np.mean(between_peak_layers)) if between_peak_layers else 0,
    }


# ===== Experiment C: Path Reuse Ratio =====

def run_experiment_c(model, tokenizer, input_device, model_info):
    """
    Compute Path Reuse Ratio between concept pairs.

    Reuse(A, B) = |active_dims(A) ∩ active_dims(B)| / |active_dims(A) ∪ active_dims(B)|

    where active_dims(W, L) = top-K dimensions of |Δ(W, L)|

    This directly measures: how much computational path do two concepts share?
    """
    n_layers = model_info.n_layers
    K = 50  # top-K dimensions

    all_pair_words = set()
    for a, b in WITHIN_PAIRS + BETWEEN_PAIRS:
        all_pair_words.add(a)
        all_pair_words.add(b)
    all_pair_words = sorted(all_pair_words)

    log_time(f"Exp C: Path Reuse Ratio, {len(all_pair_words)} words, K={K}")

    # Compute routing activation maps
    routing_maps = {}
    for wi, word in enumerate(all_pair_words):
        rmap = compute_routing_activation(model, tokenizer, input_device, word, n_layers)
        routing_maps[word] = rmap
        if (wi + 1) % 10 == 0:
            log_time(f"  [{wi+1}/{len(all_pair_words)}] done, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Compute path reuse for each pair
    sample_layers = list(range(0, n_layers, max(1, n_layers // 10)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)

    def compute_reuse(map_a, map_b, layer_idx, k=K):
        """Compute reuse ratio at a specific layer."""
        if layer_idx not in map_a or layer_idx not in map_b:
            return 0.0, 0.0, 0.0

        delta_a = np.abs(map_a[layer_idx])
        delta_b = np.abs(map_b[layer_idx])

        top_a = set(np.argsort(delta_a)[-k:])
        top_b = set(np.argsort(delta_b)[-k:])

        jaccard = len(top_a & top_b) / len(top_a | top_b)

        # Weighted reuse: consider activation magnitudes
        overlap_dims = top_a & top_b
        total_energy_a = np.sum(delta_a[list(top_a)])
        overlap_energy_a = np.sum(delta_a[list(overlap_dims)]) if overlap_dims else 0
        weighted_reuse_a = overlap_energy_a / max(total_energy_a, 1e-10)

        total_energy_b = np.sum(delta_b[list(top_b)])
        overlap_energy_b = np.sum(delta_b[list(overlap_dims)]) if overlap_dims else 0
        weighted_reuse_b = overlap_energy_b / max(total_energy_b, 1e-10)

        return jaccard, weighted_reuse_a, weighted_reuse_b

    within_reuse = []
    between_reuse = []

    for a, b in WITHIN_PAIRS:
        if a in routing_maps and b in routing_maps:
            layer_jaccards = []
            layer_weighted_a = []
            layer_weighted_b = []
            for li in sample_layers:
                jac, wa, wb = compute_reuse(routing_maps[a], routing_maps[b], li)
                layer_jaccards.append(jac)
                layer_weighted_a.append(wa)
                layer_weighted_b.append(wb)

            within_reuse.append({
                "pair": f"{a}_{b}",
                "mean_jaccard": float(np.mean(layer_jaccards)),
                "mean_weighted_reuse_A": float(np.mean(layer_weighted_a)),
                "mean_weighted_reuse_B": float(np.mean(layer_weighted_b)),
                "per_layer_jaccard": {str(l): float(j) for l, j in zip(sample_layers, layer_jaccards)},
            })

    for a, b in BETWEEN_PAIRS:
        if a in routing_maps and b in routing_maps:
            layer_jaccards = []
            layer_weighted_a = []
            layer_weighted_b = []
            for li in sample_layers:
                jac, wa, wb = compute_reuse(routing_maps[a], routing_maps[b], li)
                layer_jaccards.append(jac)
                layer_weighted_a.append(wa)
                layer_weighted_b.append(wb)

            between_reuse.append({
                "pair": f"{a}_{b}",
                "mean_jaccard": float(np.mean(layer_jaccards)),
                "mean_weighted_reuse_A": float(np.mean(layer_weighted_a)),
                "mean_weighted_reuse_B": float(np.mean(layer_weighted_b)),
                "per_layer_jaccard": {str(l): float(j) for l, j in zip(sample_layers, layer_jaccards)},
            })

    w_jac = [r["mean_jaccard"] for r in within_reuse]
    b_jac = [r["mean_jaccard"] for r in between_reuse]
    w_wa = [r["mean_weighted_reuse_A"] for r in within_reuse]
    b_wa = [r["mean_weighted_reuse_A"] for r in between_reuse]

    log_time(f"  Within jaccard: {np.mean(w_jac):.4f}, weighted_reuse: {np.mean(w_wa):.4f}")
    log_time(f"  Between jaccard: {np.mean(b_jac):.4f}, weighted_reuse: {np.mean(b_wa):.4f}")

    # Per-layer analysis
    per_layer = {}
    for li in sample_layers:
        w_j = [r["per_layer_jaccard"].get(str(li), 0) for r in within_reuse]
        b_j = [r["per_layer_jaccard"].get(str(li), 0) for r in between_reuse]
        per_layer[str(li)] = {
            "within_jaccard": float(np.mean(w_j)),
            "between_jaccard": float(np.mean(b_j)),
            "delta": float(np.mean(w_j) - np.mean(b_j)),
        }

    return {
        "within_reuse": within_reuse,
        "between_reuse": between_reuse,
        "within_summary": {
            "jaccard_mean": float(np.mean(w_jac)),
            "weighted_reuse_mean": float(np.mean(w_wa)),
        },
        "between_summary": {
            "jaccard_mean": float(np.mean(b_jac)),
            "weighted_reuse_mean": float(np.mean(b_wa)),
        },
        "per_layer": per_layer,
    }


# ===== Experiment D: Conditional Routing for Ambiguous Words =====

def run_experiment_d(model, tokenizer, input_device, model_info):
    """
    Same word, different context → where do they diverge?
    """
    n_layers = model_info.n_layers

    log_time(f"Exp D: Conditional Routing for Ambiguous Words")

    results = []

    for pi, (ctx1, ctx2) in enumerate(AMBIGUOUS_PAIRS):
        log_time(f"  Pair {pi+1}: '{ctx1[:40]}...' vs '{ctx2[:40]}...'")

        inputs1 = tokenizer(ctx1, return_tensors="pt", truncation=True, max_length=64)
        inputs2 = tokenizer(ctx2, return_tensors="pt", truncation=True, max_length=64)

        input_ids1 = inputs1["input_ids"].to(input_device)
        attn1 = inputs1["attention_mask"].to(input_device)
        input_ids2 = inputs2["input_ids"].to(input_device)
        attn2 = inputs2["attention_mask"].to(input_device)

        with torch.no_grad():
            out1 = model(input_ids=input_ids1, attention_mask=attn1, output_hidden_states=True)
            out2 = model(input_ids=input_ids2, attention_mask=attn2, output_hidden_states=True)

        hs1 = out1.hidden_states
        hs2 = out2.hidden_states

        # Compare per-layer
        from scipy.stats import spearmanr

        sample_layers = list(range(0, n_layers, max(1, n_layers // 10)))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)

        layer_metrics = {}
        for li in sample_layers:
            if li < len(hs1) and li < len(hs2):
                pos1 = int(attn1.sum().item()) - 1
                pos2 = int(attn2.sum().item()) - 1
                h1 = hs1[li][0, pos1, :].float().cpu().numpy()
                h2 = hs2[li][0, pos2, :].float().cpu().numpy()

                cos_dist = 1 - float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-10))
                r, _ = spearmanr(h1, h2)

                # Delta analysis
                delta = h1 - h2
                delta_norm = float(np.linalg.norm(delta))

                layer_metrics[str(li)] = {
                    "cosine_distance": cos_dist,
                    "spearman_r": float(r),
                    "delta_norm": delta_norm,
                }

        results.append({
            "context1": ctx1,
            "context2": ctx2,
            "layer_metrics": layer_metrics,
        })

        del out1, out2, hs1, hs2
        torch.cuda.empty_cache()

    return results


# ===== Main =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    global _log_file
    _log_file = str(RESULT_DIR / f"{model_name}_phase274.log")

    log_time(f"=== Phase 274: CRTM — {model_name} ===")

    model, tokenizer, device = load_model_bf16(model_name)
    input_device = get_input_device(model)
    model_info = get_model_info(model, model_name)

    log_time(f"Model: {model_info.model_class}, layers={model_info.n_layers}, "
             f"d_model={model_info.d_model}")

    results = {}

    # Experiment A: Routing Activation Maps
    log_time("--- Experiment A: Routing Activation Maps ---")
    results["exp_a_routing"] = run_experiment_a(model, tokenizer, input_device, model_info)
    gc.collect()
    torch.cuda.empty_cache()

    with open(RESULT_DIR / f"{model_name}_exp_a.json", "w", encoding="utf-8") as f:
        json.dump(results["exp_a_routing"], f, indent=2, ensure_ascii=False)

    # Experiment B: Activation Resampling
    log_time("--- Experiment B: Activation Resampling (Patching) ---")
    results["exp_b_resampling"] = run_experiment_b(model, tokenizer, input_device, model_info)
    gc.collect()
    torch.cuda.empty_cache()

    with open(RESULT_DIR / f"{model_name}_exp_b.json", "w", encoding="utf-8") as f:
        json.dump(results["exp_b_resampling"], f, indent=2, ensure_ascii=False)

    # Experiment C: Path Reuse Ratio
    log_time("--- Experiment C: Path Reuse Ratio ---")
    results["exp_c_reuse"] = run_experiment_c(model, tokenizer, input_device, model_info)
    gc.collect()
    torch.cuda.empty_cache()

    with open(RESULT_DIR / f"{model_name}_exp_c.json", "w", encoding="utf-8") as f:
        json.dump(results["exp_c_reuse"], f, indent=2, ensure_ascii=False)

    # Experiment D: Conditional Routing
    log_time("--- Experiment D: Conditional Routing for Ambiguous Words ---")
    results["exp_d_ambiguous"] = run_experiment_d(model, tokenizer, input_device, model_info)
    gc.collect()
    torch.cuda.empty_cache()

    with open(RESULT_DIR / f"{model_name}_exp_d.json", "w", encoding="utf-8") as f:
        json.dump(results["exp_d_ambiguous"], f, indent=2, ensure_ascii=False)

    # Save final
    final_path = RESULT_DIR / f"{model_name}_phase274.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log_time(f"=== Phase 274 Complete: {model_name} ===")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
