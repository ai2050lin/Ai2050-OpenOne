"""
Phase 273: Causal Path Decomposition
=====================================

Core shift: From statistical correlation to CAUSAL path analysis.

Method: Activation Patching (Noising)
- At each (layer, position), inject Gaussian noise into the residual stream
- Measure causal impact on target token logit
- Build "causal importance map" for each concept
- Compare maps: within-category vs between-category

Key questions:
1. Which (layer, position) pairs are causally necessary for producing a concept?
2. Do similar concepts (apple, banana) share causal paths?
3. Do different concepts (apple, car) have divergent causal paths?
4. At which layer does causal divergence happen?

Three experiments:
A. Causal Importance Map: For each (layer, last_pos), noise residual → logit impact
B. MLP Causal Tracing: For each (layer), noise MLP output → logit impact
C. Cross-concept Causal Overlap: Compare causal maps between within/between pairs

Usage:
  python tests/glm5/phase273_causal_path.py qwen3
  python tests/glm5/phase273_causal_path.py glm4
  python tests/glm5/phase273_causal_path.py deepseek7b
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

RESULT_DIR = Path("results/phase273_causal_path")
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

# Category words for path overlap analysis
CATEGORIES = {
    "fruits": ["apple", "banana", "orange", "grape", "mango", "pear", "peach", "cherry", "lemon", "lime"],
    "animals": ["dog", "cat", "lion", "tiger", "bear", "wolf", "fox", "deer", "horse", "cow"],
    "vehicles": ["car", "bus", "train", "plane", "bike", "truck", "boat", "ship", "taxi", "van"],
}

# For each word, define the expected next-token prediction and its token ID will be found dynamically
# Template: "The {word} is" → we measure logit of words like "red", "sweet" for fruits etc.
# Actually, we measure the logit of the word itself (how much the model "activates" the concept)
# Better: measure the logit of a typical attribute for each category

# Pairs for within/between comparison
WITHIN_PAIRS = [
    ("apple", "banana"), ("apple", "orange"), ("banana", "grape"),
    ("dog", "cat"), ("lion", "tiger"), ("wolf", "fox"),
    ("car", "bus"), ("train", "plane"), ("bike", "truck"),
]

BETWEEN_PAIRS = [
    ("apple", "dog"), ("banana", "car"), ("orange", "lion"),
    ("grape", "train"), ("mango", "bike"), ("peach", "wolf"),
    ("cherry", "fox"), ("lemon", "bus"), ("pear", "tiger"),
    ("lime", "ship"),
]


# ===== Model Loading =====

def load_model_bf16(model_name: str):
    """BF16 + device_map=auto loading for all models."""
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
    """Get device for input tensors."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Core Algorithm: Activation Patching =====

def compute_clean_logit(model, tokenizer, input_device, prompt, target_token_id):
    """Compute the logit of target_token for the clean (unpatched) input."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attn_mask = inputs["attention_mask"].to(input_device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits[0, -1, :]  # [vocab]
    target_logit = float(logits[target_token_id].item())

    del out
    torch.cuda.empty_cache()
    return target_logit


def compute_noised_logit_residual(model, tokenizer, input_device, prompt, target_token_id,
                                   noise_layer, noise_std=3.0, n_trials=5):
    """
    Noise the residual stream at a specific layer, measure impact on target logit.

    Method:
    1. Run clean forward pass, collect residual at noise_layer
    2. Add Gaussian noise with std=noise_std to residual at last position
    3. Continue forward from noise_layer with noised residual
    4. Measure logit change

    Returns:
        dict: {mean_logit_change, std_logit_change, mean_pct_change}
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attn_mask = inputs["attention_mask"].to(input_device)

    layers = get_layers(model)
    n_layers = len(layers)

    # Step 1: Run clean forward, capture residual at each layer
    clean_residuals = {}

    def make_capture_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                clean_residuals[layer_idx] = input[0].detach().clone()
        return hook

    capture_hooks = []
    for li in range(n_layers):
        capture_hooks.append(layers[li].register_forward_hook(make_capture_hook(li)))

    with torch.no_grad():
        clean_out = model(input_ids=input_ids, attention_mask=attn_mask)
    clean_logit = float(clean_out.logits[0, -1, target_token_id].item())

    for h in capture_hooks:
        h.remove()

    if noise_layer not in clean_residuals:
        return {"mean_logit_change": 0.0, "std_logit_change": 0.0, "mean_pct_change": 0.0}

    # Step 2: Run noised forward (multiple trials)
    logit_changes = []

    for trial in range(n_trials):
        # Hook to replace residual at noise_layer with noised version
        noised_residual = clean_residuals[noise_layer].clone()
        seq_len = noised_residual.shape[1]
        noise = torch.randn_like(noised_residual[:, -1:, :]) * noise_std
        noised_residual[:, -1:, :] += noise

        def make_noise_hook(stored_residual):
            def hook(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    modified = list(input)
                    modified[0] = stored_residual.to(input[0].device, dtype=input[0].dtype)
                    return tuple(modified)
                return input
            return hook

        noise_hook = layers[noise_layer].register_forward_hook(make_noise_hook(noised_residual))

        with torch.no_grad():
            noised_out = model(input_ids=input_ids, attention_mask=attn_mask)
        noised_logit = float(noised_out.logits[0, -1, target_token_id].item())

        noise_hook.remove()
        logit_changes.append(noised_logit - clean_logit)

        del noised_out, noised_residual
        torch.cuda.empty_cache()

    changes = np.array(logit_changes)
    mean_change = float(np.mean(changes))
    std_change = float(np.std(changes))
    pct_change = mean_change / max(abs(clean_logit), 1e-6) * 100

    del clean_residuals, clean_out
    torch.cuda.empty_cache()

    return {
        "mean_logit_change": mean_change,
        "std_logit_change": std_change,
        "mean_pct_change": pct_change,
        "clean_logit": clean_logit,
    }


def compute_noised_logit_mlp(model, tokenizer, input_device, prompt, target_token_id,
                              noise_layer, noise_std=3.0, n_trials=5):
    """
    Noise the MLP output at a specific layer, measure impact on target logit.

    Returns:
        dict: {mean_logit_change, std_logit_change}
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attn_mask = inputs["attention_mask"].to(input_device)

    layers = get_layers(model)

    # Step 1: Clean forward, capture MLP output
    clean_mlp_out = {}

    def make_mlp_capture_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                clean_mlp_out[layer_idx] = output[0].detach().clone()
            else:
                clean_mlp_out[layer_idx] = output.detach().clone()
        return hook

    mlp = layers[noise_layer].mlp if hasattr(layers[noise_layer], "mlp") else None
    if mlp is None:
        return {"mean_logit_change": 0.0, "std_logit_change": 0.0}

    capture_hook = mlp.register_forward_hook(make_mlp_capture_hook(noise_layer))

    with torch.no_grad():
        clean_out = model(input_ids=input_ids, attention_mask=attn_mask)
    clean_logit = float(clean_out.logits[0, -1, target_token_id].item())

    capture_hook.remove()

    if noise_layer not in clean_mlp_out:
        return {"mean_logit_change": 0.0, "std_logit_change": 0.0}

    # Step 2: Noised forward
    logit_changes = []

    for trial in range(n_trials):
        noised_mlp = clean_mlp_out[noise_layer].clone()
        noise = torch.randn_like(noised_mlp[:, -1:, :]) * noise_std
        noised_mlp[:, -1:, :] += noise

        def make_mlp_noise_hook(stored_mlp):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    modified = (stored_mlp.to(output[0].device, dtype=output[0].dtype),) + output[1:]
                    return modified
                return stored_mlp.to(output.device, dtype=output.dtype)
            return hook

        noise_hook = mlp.register_forward_hook(make_mlp_noise_hook(noised_mlp))

        with torch.no_grad():
            noised_out = model(input_ids=input_ids, attention_mask=attn_mask)
        noised_logit = float(noised_out.logits[0, -1, target_token_id].item())

        noise_hook.remove()
        logit_changes.append(noised_logit - clean_logit)

        del noised_out, noised_mlp
        torch.cuda.empty_cache()

    changes = np.array(logit_changes)
    mean_change = float(np.mean(changes))
    std_change = float(np.std(changes))

    del clean_mlp_out, clean_out
    torch.cuda.empty_cache()

    return {
        "mean_logit_change": mean_change,
        "std_logit_change": std_change,
        "clean_logit": clean_logit,
    }


# ===== Experiment A: Causal Importance Map =====

def run_experiment_a(model, tokenizer, input_device, model_info):
    """
    Build causal importance map for each word.
    For each (word, layer), noise residual at last position → measure logit impact.
    """
    n_layers = model_info.n_layers
    all_words = []
    for cat_words in CATEGORIES.values():
        all_words.extend(cat_words)

    # Sample layers to reduce computation
    sample_layers = list(range(0, n_layers, max(1, n_layers // 15)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    log_time(f"Exp A: Causal importance map, {len(all_words)} words × {len(sample_layers)} layers")

    results = {}

    for wi, word in enumerate(all_words):
        prompt = f"The {word} is"
        # Find the token ID of the word itself
        word_tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(word_tokens) == 0:
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
        target_token_id = word_tokens[0]

        word_results = {}
        for li, layer_idx in enumerate(sample_layers):
            ret = compute_noised_logit_residual(
                model, tokenizer, input_device, prompt, target_token_id,
                noise_layer=layer_idx, noise_std=3.0, n_trials=3
            )
            word_results[str(layer_idx)] = {
                "mean_logit_change": ret["mean_logit_change"],
                "std_logit_change": ret["std_logit_change"],
                "abs_impact": abs(ret["mean_logit_change"]),
            }
            if li % 5 == 0:
                log_time(f"  Word '{word}' layer {layer_idx}: impact={abs(ret['mean_logit_change']):.4f}")

        results[word] = word_results

        if (wi + 1) % 5 == 0:
            log_time(f"  [{wi+1}/{len(all_words)}] words done, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    results["_meta"] = {
        "sample_layers": sample_layers,
        "noise_std": 3.0,
        "n_trials": 3,
        "n_words": len(all_words),
    }

    return results


# ===== Experiment B: MLP Causal Tracing =====

def run_experiment_b(model, tokenizer, input_device, model_info):
    """
    MLP causal tracing: For each (word, layer), noise MLP output → measure logit impact.
    Compare with residual stream noise impact.
    """
    n_layers = model_info.n_layers
    # Use a subset of words
    test_words = ["apple", "banana", "dog", "cat", "car", "bus",
                  "orange", "lion", "train", "grape"]

    sample_layers = list(range(0, n_layers, max(1, n_layers // 15)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    log_time(f"Exp B: MLP causal tracing, {len(test_words)} words × {len(sample_layers)} layers")

    results = {}

    for wi, word in enumerate(test_words):
        prompt = f"The {word} is"
        word_tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(word_tokens) == 0:
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
        target_token_id = word_tokens[0]

        word_results = {}
        for li, layer_idx in enumerate(sample_layers):
            ret = compute_noised_logit_mlp(
                model, tokenizer, input_device, prompt, target_token_id,
                noise_layer=layer_idx, noise_std=3.0, n_trials=3
            )
            word_results[str(layer_idx)] = {
                "mean_logit_change": ret["mean_logit_change"],
                "std_logit_change": ret["std_logit_change"],
                "abs_impact": abs(ret["mean_logit_change"]),
            }

        results[word] = word_results

        if (wi + 1) % 3 == 0:
            log_time(f"  [{wi+1}/{len(test_words)}] words done (MLP), GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    results["_meta"] = {
        "sample_layers": sample_layers,
        "noise_std": 3.0,
        "n_trials": 3,
        "test_words": test_words,
    }

    return results


# ===== Experiment C: Cross-concept Causal Overlap =====

def run_experiment_c(model, tokenizer, input_device, model_info):
    """
    Compare causal importance maps between concept pairs.

    For each pair (A, B):
    1. Build causal map for A: impact_A[layer] = |logit change when noising layer|
    2. Build causal map for B: impact_B[layer] = |logit change when noising layer|
    3. Compute correlation between impact_A and impact_B across layers
    4. Compare within-category pairs vs between-category pairs

    This answers: Do similar concepts share causal paths?
    """
    n_layers = model_info.n_layers

    sample_layers = list(range(0, n_layers, max(1, n_layers // 15)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    # First compute causal maps for all unique words in pairs
    all_pair_words = set()
    for a, b in WITHIN_PAIRS + BETWEEN_PAIRS:
        all_pair_words.add(a)
        all_pair_words.add(b)
    all_pair_words = sorted(all_pair_words)

    log_time(f"Exp C: Cross-concept causal overlap, {len(all_pair_words)} unique words")

    # Compute causal maps
    causal_maps = {}
    for wi, word in enumerate(all_pair_words):
        prompt = f"The {word} is"
        word_tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(word_tokens) == 0:
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
        target_token_id = word_tokens[0]

        impacts = []
        for layer_idx in sample_layers:
            ret = compute_noised_logit_residual(
                model, tokenizer, input_device, prompt, target_token_id,
                noise_layer=layer_idx, noise_std=3.0, n_trials=3
            )
            impacts.append(abs(ret["mean_logit_change"]))

        causal_maps[word] = np.array(impacts)

        if (wi + 1) % 5 == 0:
            log_time(f"  [{wi+1}/{len(all_pair_words)}] causal maps done, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Compute pair-level overlap metrics
    from scipy.stats import spearmanr, pearsonr

    def compute_pair_overlap(map_a, map_b):
        """Compute overlap metrics between two causal importance maps."""
        # Rank correlation (are the same layers important?)
        sp_corr, sp_p = spearmanr(map_a, map_b)
        pe_corr, pe_p = pearsonr(map_a, map_b)

        # Jaccard overlap of top-K important layers
        k = max(1, len(map_a) // 3)
        top_a = set(np.argsort(map_a)[-k:])
        top_b = set(np.argsort(map_b)[-k:])
        jaccard = len(top_a & top_b) / len(top_a | top_b) if len(top_a | top_b) > 0 else 0

        # Cosine similarity of impact vectors
        norm_a = np.linalg.norm(map_a)
        norm_b = np.linalg.norm(map_b)
        cos_sim = float(np.dot(map_a, map_b) / max(norm_a * norm_b, 1e-10))

        return {
            "spearman_r": float(sp_corr),
            "spearman_p": float(sp_p),
            "pearson_r": float(pe_corr),
            "pearson_p": float(pe_p),
            "top_k_jaccard": float(jaccard),
            "cosine_sim": float(cos_sim),
        }

    within_results = []
    for a, b in WITHIN_PAIRS:
        if a in causal_maps and b in causal_maps:
            overlap = compute_pair_overlap(causal_maps[a], causal_maps[b])
            overlap["pair"] = f"{a}_{b}"
            overlap["type"] = "within"
            within_results.append(overlap)

    between_results = []
    for a, b in BETWEEN_PAIRS:
        if a in causal_maps and b in causal_maps:
            overlap = compute_pair_overlap(causal_maps[a], causal_maps[b])
            overlap["pair"] = f"{a}_{b}"
            overlap["type"] = "between"
            between_results.append(overlap)

    # Summary statistics
    def summarize(results_list, label):
        metrics = ["spearman_r", "pearson_r", "top_k_jaccard", "cosine_sim"]
        summary = {"label": label, "n_pairs": len(results_list)}
        for m in metrics:
            vals = [r[m] for r in results_list]
            summary[f"{m}_mean"] = float(np.mean(vals))
            summary[f"{m}_std"] = float(np.std(vals))
        return summary

    within_summary = summarize(within_results, "within")
    between_summary = summarize(between_results, "between")

    log_time(f"  Within-category: spearman_r={within_summary['spearman_r_mean']:.4f}, "
             f"cosine={within_summary['cosine_sim_mean']:.4f}, "
             f"jaccard={within_summary['top_k_jaccard_mean']:.4f}")
    log_time(f"  Between-category: spearman_r={between_summary['spearman_r_mean']:.4f}, "
             f"cosine={between_summary['cosine_sim_mean']:.4f}, "
             f"jaccard={between_summary['top_k_jaccard_mean']:.4f}")

    return {
        "within_pairs": within_results,
        "between_pairs": between_results,
        "within_summary": within_summary,
        "between_summary": between_summary,
        "_meta": {
            "sample_layers": sample_layers,
            "noise_std": 3.0,
            "n_trials": 3,
        }
    }


# ===== Experiment D: Causal Divergence Point =====

def run_experiment_d(model, tokenizer, input_device, model_info):
    """
    Find the causal divergence point: the layer where noising affects
    concept A but NOT concept B (and vice versa).

    Method: For a pair (A, B):
    - Noise at layer L → measure logit change for A's token AND B's token
    - If noising L affects A's token but not B's token → L is a divergence point
    - Differential impact = |impact_A(L)| - |impact_B(L)|
    """
    n_layers = model_info.n_layers

    sample_layers = list(range(0, n_layers, max(1, n_layers // 15)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    # Use within and between pairs
    test_pairs = WITHIN_PAIRS[:5] + BETWEEN_PAIRS[:5]

    log_time(f"Exp D: Causal divergence point, {len(test_pairs)} pairs × {len(sample_layers)} layers")

    results = []

    for pi, (word_a, word_b) in enumerate(test_pairs):
        prompt_a = f"The {word_a} is"
        prompt_b = f"The {word_b} is"

        tok_a = tokenizer.encode(" " + word_a, add_special_tokens=False)
        if len(tok_a) == 0:
            tok_a = tokenizer.encode(word_a, add_special_tokens=False)
        target_a = tok_a[0]

        tok_b = tokenizer.encode(" " + word_b, add_special_tokens=False)
        if len(tok_b) == 0:
            tok_b = tokenizer.encode(word_b, add_special_tokens=False)
        target_b = tok_b[0]

        pair_data = {"pair": f"{word_a}_{word_b}"}

        # Noise in context of prompt_a → measure impact on BOTH token_a and token_b logits
        impacts_a_context = {}
        impacts_b_context = {}

        for layer_idx in sample_layers:
            # Noise in prompt_a context
            ret_a = compute_noised_logit_residual(
                model, tokenizer, input_device, prompt_a, target_a,
                noise_layer=layer_idx, noise_std=3.0, n_trials=3
            )
            # Also measure impact on token_b when noising in prompt_a context
            ret_a_b = compute_noised_logit_residual(
                model, tokenizer, input_device, prompt_a, target_b,
                noise_layer=layer_idx, noise_std=3.0, n_trials=3
            )

            # Noise in prompt_b context
            ret_b = compute_noised_logit_residual(
                model, tokenizer, input_device, prompt_b, target_b,
                noise_layer=layer_idx, noise_std=3.0, n_trials=3
            )
            ret_b_a = compute_noised_logit_residual(
                model, tokenizer, input_device, prompt_b, target_a,
                noise_layer=layer_idx, noise_std=3.0, n_trials=3
            )

            impacts_a_context[str(layer_idx)] = {
                "impact_on_a": abs(ret_a["mean_logit_change"]),
                "impact_on_b": abs(ret_a_b["mean_logit_change"]),
                "differential": abs(ret_a["mean_logit_change"]) - abs(ret_a_b["mean_logit_change"]),
            }
            impacts_b_context[str(layer_idx)] = {
                "impact_on_b": abs(ret_b["mean_logit_change"]),
                "impact_on_a": abs(ret_b_a["mean_logit_change"]),
                "differential": abs(ret_b["mean_logit_change"]) - abs(ret_b_a["mean_logit_change"]),
            }

        # Find max differential layer
        diffs_a = [impacts_a_context[str(l)]["differential"] for l in sample_layers]
        diffs_b = [impacts_b_context[str(l)]["differential"] for l in sample_layers]

        max_diff_layer_a = sample_layers[int(np.argmax(diffs_a))]
        max_diff_layer_b = sample_layers[int(np.argmax(diffs_b))]

        pair_data["a_context_impacts"] = impacts_a_context
        pair_data["b_context_impacts"] = impacts_b_context
        pair_data["max_diff_layer_a_context"] = max_diff_layer_a
        pair_data["max_diff_layer_b_context"] = max_diff_layer_b
        pair_data["is_within"] = (word_a, word_b) in [(a, b) for a, b in WITHIN_PAIRS]

        results.append(pair_data)

        log_time(f"  Pair ({word_a}, {word_b}): max_diff_layer={max_diff_layer_a} (A ctx), "
                 f"{max_diff_layer_b} (B ctx)")

    return {
        "pairs": results,
        "_meta": {"sample_layers": sample_layers, "noise_std": 3.0, "n_trials": 3}
    }


# ===== Main =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    global _log_file
    _log_file = str(RESULT_DIR / f"{model_name}_phase273.log")

    log_time(f"=== Phase 273: Causal Path Decomposition — {model_name} ===")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    input_device = get_input_device(model)
    model_info = get_model_info(model, model_name)

    log_time(f"Model: {model_info.model_class}, layers={model_info.n_layers}, "
             f"d_model={model_info.d_model}")

    # ===== Run Experiments =====
    results = {}

    # Experiment A: Causal Importance Map
    log_time("--- Experiment A: Causal Importance Map ---")
    results["exp_a_causal_map"] = run_experiment_a(model, tokenizer, input_device, model_info)
    gc.collect()
    torch.cuda.empty_cache()
    log_time(f"Exp A done, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Save intermediate results
    with open(RESULT_DIR / f"{model_name}_exp_a.json", "w", encoding="utf-8") as f:
        json.dump(results["exp_a_causal_map"], f, indent=2, ensure_ascii=False)

    # Experiment B: MLP Causal Tracing
    log_time("--- Experiment B: MLP Causal Tracing ---")
    results["exp_b_mlp_tracing"] = run_experiment_b(model, tokenizer, input_device, model_info)
    gc.collect()
    torch.cuda.empty_cache()
    log_time(f"Exp B done, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    with open(RESULT_DIR / f"{model_name}_exp_b.json", "w", encoding="utf-8") as f:
        json.dump(results["exp_b_mlp_tracing"], f, indent=2, ensure_ascii=False)

    # Experiment C: Cross-concept Causal Overlap
    log_time("--- Experiment C: Cross-concept Causal Overlap ---")
    results["exp_c_causal_overlap"] = run_experiment_c(model, tokenizer, input_device, model_info)
    gc.collect()
    torch.cuda.empty_cache()
    log_time(f"Exp C done, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    with open(RESULT_DIR / f"{model_name}_exp_c.json", "w", encoding="utf-8") as f:
        json.dump(results["exp_c_causal_overlap"], f, indent=2, ensure_ascii=False)

    # Experiment D: Causal Divergence Point
    log_time("--- Experiment D: Causal Divergence Point ---")
    results["exp_d_divergence"] = run_experiment_d(model, tokenizer, input_device, model_info)
    gc.collect()
    torch.cuda.empty_cache()
    log_time(f"Exp D done, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    with open(RESULT_DIR / f"{model_name}_exp_d.json", "w", encoding="utf-8") as f:
        json.dump(results["exp_d_divergence"], f, indent=2, ensure_ascii=False)

    # ===== Save Final Results =====
    final_path = RESULT_DIR / f"{model_name}_phase273.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log_time(f"=== Phase 273 Complete: {model_name} ===")
    log_time(f"Results saved to {final_path}")

    # Release model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time(f"Model released, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")


if __name__ == "__main__":
    main()
