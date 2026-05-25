"""
Phase 275: CDRE — Conditional Dynamical Reverse Engineering
=============================================================

Core shift: From "what state is the network in" to "how does the token configure dynamics".

The Jacobian J_l^(t) = dh_{l+1}/dh_l conditioned on token t is the TRUE "computation rule".
Similar tokens should have similar conditional Jacobians → they configure similar dynamics.

Key insight: Token embeddings are NOT "semantic vectors" but "conditional keys" that
configure the network's local computation rules at each layer.

Experiments:
A. Conditional Jacobian Similarity — Do within-category tokens have more similar Jacobian responses?
   Inject SAME perturbation vector at layer l, measure h_{l+1} response.
   If within-category tokens respond more similarly → they configure similar local dynamics.

B. Attractor Convergence — Do perturbed states converge in deep layers?
   Inject perturbation at embedding level, run full forward pass.
   If states converge toward the same endpoint → attractor dynamics.
   Compare convergence: within vs between category.

C. Logit Response Fingerprint — How does each perturbation at each layer affect the output?
   Collect [delta_logit_1, ..., delta_logit_K] for each (token, layer).
   This K-dimensional "fingerprint" captures the global Jacobian projection.
   Compare fingerprints: within vs between category.

Method: register_forward_pre_hook to perturb h_l, register_forward_hook to capture h_{l+1}.
This directly estimates Jv ≈ (h_{l+1}(h_l + ev) - h_{l+1}(h_l)) / e  (finite difference).

Usage:
  python tests/glm5/phase275_cdre.py qwen3
  python tests/glm5/phase275_cdre.py glm4
  python tests/glm5/phase275_cdre.py deepseek7b
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

RESULT_DIR = Path("results/phase275_cdre")
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

# Perturbation parameters
N_PERTURBATIONS = 12   # Number of random perturbation vectors per layer
EPSILON = 1.0          # Perturbation magnitude
N_ATTRACTOR_PERTURBATIONS = 8  # For Exp B (embedding-level)


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


# ===== Core: Baseline Forward Pass =====

def run_baseline(model, tokenizer, device, word, n_layers):
    """Run baseline forward pass, cache h_l at each layer (last token position)."""
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

    # Extract last token hidden states
    h_dict = {}
    for li in range(n_layers):
        key = f"L{li}"
        if key in captured:
            h_dict[li] = captured[key][0, -1, :].numpy()  # [d_model]

    # Also get logits
    logits = outputs.logits[0, -1].float().cpu().numpy()

    return h_dict, logits


# ===== Exp A: Conditional Jacobian Similarity =====

def run_perturbed_jacobian(model, tokenizer, device, word, target_layer, perturbation_vec,
                           n_layers, baseline_h):
    """
    Perturb h_l at target_layer and measure the response at target_layer+1.

    Uses register_forward_pre_hook to inject perturbation into the layer's input.
    This directly estimates Jv ≈ (h_{l+1}(h_l + ev) - h_{l+1}(h_l)) / e

    Also captures logits for global response fingerprint.
    """
    prompt = f"The {word} is"
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    layers = get_layers(model)
    captured = {}

    # Hook: capture output of target_layer (h_{l+1})
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured["h_next"] = output[0].detach().float().cpu()
        else:
            captured["h_next"] = output.detach().float().cpu()

    # Pre-hook: inject perturbation into target_layer's input (h_l → h_l + ev)
    perturb_tensor = torch.tensor(perturbation_vec, dtype=torch.float32)

    def perturbation_prehook(module, args):
        # args is a tuple; args[0] is hidden_states
        hidden_states = args[0]
        perturbed = hidden_states.clone()
        # Add perturbation at last token position
        p = perturb_tensor.to(device=perturbed.device, dtype=perturbed.dtype)
        perturbed[0, -1, :] += EPSILON * p
        return (perturbed,) + args[1:]

    hooks = []
    hooks.append(layers[target_layer].register_forward_hook(capture_hook))
    hooks.append(layers[target_layer].register_forward_pre_hook(perturbation_prehook))

    # Also capture a few subsequent layers for propagation analysis
    prop_layers = []
    if target_layer + 1 < n_layers:
        prop_layers.append(target_layer + 1)
    if target_layer + 3 < n_layers:
        prop_layers.append(target_layer + 3)
    if n_layers - 1 not in prop_layers and n_layers - 1 > target_layer:
        prop_layers.append(n_layers - 1)

    for pl in prop_layers:
        def make_prop_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[name] = output[0].detach().float().cpu()
                else:
                    captured[name] = output.detach().float().cpu()
            return hook
        hooks.append(layers[pl].register_forward_hook(make_prop_hook(f"prop_L{pl}")))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    # Extract results
    result = {"target_layer": target_layer}

    # Local Jacobian response: delta_h_{l+1}
    if "h_next" in captured:
        h_next_perturbed = captured["h_next"][0, -1, :].numpy()
        h_next_baseline = baseline_h.get(target_layer + 1 if target_layer + 1 < n_layers else target_layer)
        if h_next_baseline is not None:
            delta_h_next = h_next_perturbed - h_next_baseline
            result["delta_h_next"] = delta_h_next.tolist()
            result["delta_h_next_norm"] = float(np.linalg.norm(delta_h_next))
            result["delta_h_next_cosine_with_perturb"] = float(
                np.dot(delta_h_next, perturbation_vec) /
                (np.linalg.norm(delta_h_next) * np.linalg.norm(perturbation_vec) + 1e-10)
            )
        else:
            result["delta_h_next"] = None
    else:
        result["delta_h_next"] = None

    # Propagation data
    for pl in prop_layers:
        key = f"prop_L{pl}"
        if key in captured:
            h_prop = captured[key][0, -1, :].numpy()
            result[f"h_L{pl}"] = h_prop.tolist()
            # Also compute delta relative to baseline
            h_baseline = baseline_h.get(pl)
            if h_baseline is not None:
                delta_prop = h_prop - h_baseline
                result[f"delta_L{pl}_norm"] = float(np.linalg.norm(delta_prop))

    # Logits
    logits = outputs.logits[0, -1].float().cpu().numpy()
    result["logits"] = logits.tolist()

    return result


def experiment_a(model, tokenizer, device, model_name):
    """
    Exp A: Conditional Jacobian Similarity

    For each token, at each sampled layer, apply K perturbation vectors.
    Measure the local response (delta h_{l+1}) and global response (delta logits).

    Compare: do within-category tokens have more similar Jacobian responses?
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    log_time(f"Exp A: n_layers={n_layers}, d_model={d_model}")

    # Sample layers (skip L0 and last layer for Jacobian — they're edge cases)
    sampled_layers = []
    for i in range(1, n_layers - 1):
        if i % max(1, (n_layers - 2) // 6) == 0 or i == n_layers // 2:
            sampled_layers.append(i)
    sampled_layers = sorted(set(sampled_layers))
    if len(sampled_layers) > 7:
        step = len(sampled_layers) // 6
        sampled_layers = sampled_layers[::step][:7]
    log_time(f"Sampled layers for Jacobian: {sampled_layers}")

    # Pre-generate perturbation vectors (shared across all tokens)
    rng = np.random.RandomState(42)
    perturb_vecs = []
    for k in range(N_PERTURBATIONS):
        v = rng.randn(d_model).astype(np.float32)
        v = v / np.linalg.norm(v)  # Unit vector
        perturb_vecs.append(v)
    log_time(f"Generated {len(perturb_vecs)} perturbation vectors, dim={d_model}")

    # Collect data
    all_data = {}
    for wi, word in enumerate(ALL_WORDS):
        log_time(f"  Word {wi+1}/{len(ALL_WORDS)}: '{word}' — running baseline...")
        baseline_h, baseline_logits = run_baseline(model, tokenizer, device, word, n_layers)

        word_data = {
            "baseline_logits": baseline_logits.tolist(),
            "jacobian_responses": {},
        }

        for li, layer_idx in enumerate(sampled_layers):
            layer_data = []
            for ki, pv in enumerate(perturb_vecs):
                if ki % 4 == 0:
                    log_time(f"    L{layer_idx}, perturb {ki+1}/{N_PERTURBATIONS}...")
                result = run_perturbed_jacobian(
                    model, tokenizer, device, word, layer_idx, pv,
                    n_layers, baseline_h
                )
                layer_data.append(result)

            word_data["jacobian_responses"][str(layer_idx)] = layer_data

        all_data[word] = word_data
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        log_time(f"  '{word}' done. GPU={gpu_mem:.2f}GB")

    # Save raw data
    out_path = RESULT_DIR / f"{model_name}_exp_a.json"
    with open(out_path, "w") as f:
        json.dump(all_data, f)
    log_time(f"Exp A raw data saved to {out_path}")

    # ===== Analysis =====
    log_time("Analyzing Jacobian similarity...")

    # For each pair of words, compute Jacobian response similarity at each layer
    def get_jacobian_response(word, layer_idx):
        """Get the K delta_h_next vectors for a word at a layer."""
        responses = []
        if word not in all_data:
            return None
        layer_data = all_data[word]["jacobian_responses"].get(str(layer_idx), [])
        for item in layer_data:
            if item.get("delta_h_next") is not None:
                responses.append(np.array(item["delta_h_next"]))
        return responses if len(responses) > 0 else None

    def compute_jacobian_similarity(responses_A, responses_B):
        """
        Compute similarity between two sets of Jacobian responses.
        Average cosine similarity of corresponding response vectors.
        """
        n = min(len(responses_A), len(responses_B))
        if n == 0:
            return None
        cosines = []
        for k in range(n):
            a, b = responses_A[k], responses_B[k]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-10 and nb > 1e-10:
                cosines.append(float(np.dot(a, b) / (na * nb)))
        return float(np.mean(cosines)) if cosines else None

    # Per-layer analysis
    per_layer = {}
    for layer_idx in sampled_layers:
        within_sims = []
        between_sims = []

        for wa, wb in WITHIN_PAIRS:
            ra = get_jacobian_response(wa, layer_idx)
            rb = get_jacobian_response(wb, layer_idx)
            if ra is not None and rb is not None:
                sim = compute_jacobian_similarity(ra, rb)
                if sim is not None:
                    within_sims.append(sim)

        for wa, wb in BETWEEN_PAIRS:
            ra = get_jacobian_response(wa, layer_idx)
            rb = get_jacobian_response(wb, layer_idx)
            if ra is not None and rb is not None:
                sim = compute_jacobian_similarity(ra, rb)
                if sim is not None:
                    between_sims.append(sim)

        per_layer[str(layer_idx)] = {
            "within_mean": float(np.mean(within_sims)) if within_sims else None,
            "between_mean": float(np.mean(between_sims)) if between_sims else None,
            "within_n": len(within_sims),
            "between_n": len(between_sims),
            "delta": float(np.mean(within_sims) - np.mean(between_sims))
                if within_sims and between_sims else None,
        }

    # Also compute logit fingerprint similarity (Exp C combined)
    def get_logit_fingerprint(word, layer_idx):
        """Get the K-dimensional logit change vector for a word at a layer."""
        if word not in all_data:
            return None
        baseline_logits = np.array(all_data[word]["baseline_logits"])
        layer_data = all_data[word]["jacobian_responses"].get(str(layer_idx), [])
        deltas = []
        for item in layer_data:
            if "logits" in item:
                delta_logit = np.array(item["logits"]) - baseline_logits
                deltas.append(delta_logit)
        return deltas if deltas else None

    def compute_logit_fingerprint_similarity(fps_A, fps_B):
        """
        Compare logit fingerprints: for each perturbation, compute cosine of logit change.
        Average over perturbations.
        """
        n = min(len(fps_A), len(fps_B))
        if n == 0:
            return None
        cosines = []
        for k in range(n):
            a, b = fps_A[k], fps_B[k]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-10 and nb > 1e-10:
                cosines.append(float(np.dot(a, b) / (na * nb)))
        return float(np.mean(cosines)) if cosines else None

    logit_per_layer = {}
    for layer_idx in sampled_layers:
        within_sims = []
        between_sims = []

        for wa, wb in WITHIN_PAIRS:
            fa = get_logit_fingerprint(wa, layer_idx)
            fb = get_logit_fingerprint(wb, layer_idx)
            if fa is not None and fb is not None:
                sim = compute_logit_fingerprint_similarity(fa, fb)
                if sim is not None:
                    within_sims.append(sim)

        for wa, wb in BETWEEN_PAIRS:
            fa = get_logit_fingerprint(wa, layer_idx)
            fb = get_logit_fingerprint(wb, layer_idx)
            if fa is not None and fb is not None:
                sim = compute_logit_fingerprint_similarity(fa, fb)
                if sim is not None:
                    between_sims.append(sim)

        logit_per_layer[str(layer_idx)] = {
            "within_mean": float(np.mean(within_sims)) if within_sims else None,
            "between_mean": float(np.mean(between_sims)) if between_sims else None,
            "within_n": len(within_sims),
            "between_n": len(between_sims),
            "delta": float(np.mean(within_sims) - np.mean(between_sims))
                if within_sims and between_sims else None,
        }

    # Summary
    within_all = [pl[k]["within_mean"] for pl in [per_layer] for k in pl
                  if pl[k]["within_mean"] is not None]
    between_all = [pl[k]["between_mean"] for pl in [per_layer] for k in pl
                   if pl[k]["between_mean"] is not None]

    summary = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "sampled_layers": sampled_layers,
        "n_perturbations": N_PERTURBATIONS,
        "epsilon": EPSILON,
        "jacobian_similarity": {
            "within_mean": float(np.mean(within_all)) if within_all else None,
            "between_mean": float(np.mean(between_all)) if between_all else None,
            "delta": float(np.mean(within_all) - np.mean(between_all))
                if within_all and between_all else None,
        },
        "per_layer_jacobian": per_layer,
        "per_layer_logit_fingerprint": logit_per_layer,
    }

    out_path = RESULT_DIR / f"{model_name}_exp_a_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_time(f"Exp A summary saved to {out_path}")

    # Print key results
    log_time(f"=== Exp A: Jacobian Similarity ({model_name}) ===")
    for layer_idx in sampled_layers:
        pl = per_layer[str(layer_idx)]
        w = pl.get("within_mean", "N/A")
        b = pl.get("between_mean", "N/A")
        d = pl.get("delta", "N/A")
        log_time(f"  L{layer_idx}: within={w}, between={b}, delta={d}")

    js = summary["jacobian_similarity"]
    log_time(f"  OVERALL: within={js['within_mean']}, between={js['between_mean']}, delta={js['delta']}")

    # Logit fingerprint results
    log_time(f"=== Exp C: Logit Fingerprint Similarity ({model_name}) ===")
    log_within = []
    log_between = []
    for layer_idx in sampled_layers:
        pl = logit_per_layer[str(layer_idx)]
        w = pl.get("within_mean")
        b = pl.get("between_mean")
        if w is not None:
            log_within.append(w)
        if b is not None:
            log_between.append(b)
        log_time(f"  L{layer_idx}: within={w}, between={b}, delta={pl.get('delta')}")
    if log_within and log_between:
        log_time(f"  OVERALL: within={np.mean(log_within):.4f}, "
                 f"between={np.mean(log_between):.4f}, "
                 f"delta={np.mean(log_within)-np.mean(log_between):+.4f}")

    return summary


# ===== Exp B: Attractor Convergence =====

def experiment_b(model, tokenizer, device, model_name):
    """
    Exp B: Attractor Convergence

    For each token, run multiple forward passes with different embedding perturbations.
    Measure convergence of hidden states at each layer:
    - If perturbed states converge to similar deep-layer states → attractor dynamics.
    - Compare: do within-category tokens converge to similar attractors?
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    log_time(f"Exp B: n_layers={n_layers}, d_model={d_model}")

    # Sample layers for convergence measurement
    conv_layers = sorted(set([0] + list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]))
    log_time(f"Convergence measurement layers: {conv_layers}")

    rng = np.random.RandomState(123)
    perturb_magnitude = 2.0  # Larger perturbation at embedding level

    all_data = {}
    for wi, word in enumerate(ALL_WORDS):
        log_time(f"  Word {wi+1}/{len(ALL_WORDS)}: '{word}' — running attractor test...")

        # Baseline
        baseline_h, baseline_logits = run_baseline(model, tokenizer, device, word, n_layers)

        # Run perturbed passes
        perturbed_states = {l: [] for l in conv_layers}
        perturbed_logits = []

        for pi in range(N_ATTRACTOR_PERTURBATIONS):
            # Create perturbed embedding
            prompt = f"The {word} is"
            toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
            input_ids = toks["input_ids"].to(device)
            attention_mask = toks["attention_mask"].to(device)

            # Get embedding and perturb
            embed_layer = model.get_input_embeddings()
            inputs_embeds = embed_layer(input_ids).detach().clone()

            # Add perturbation to last token embedding
            perturb_v = rng.randn(d_model).astype(np.float32)
            perturb_v = perturb_v / np.linalg.norm(perturb_v) * perturb_magnitude
            perturb_tensor = torch.tensor(perturb_v, dtype=inputs_embeds.dtype, device=device)
            inputs_embeds[0, -1, :] += perturb_tensor

            # Run forward pass with hooks
            layers_list = get_layers(model)
            captured = {}

            def make_hook(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[name] = output[0].detach().float().cpu()
                    else:
                        captured[name] = output.detach().float().cpu()
                return hook

            hooks = []
            for l in conv_layers:
                hooks.append(layers_list[l].register_forward_hook(make_hook(f"L{l}")))

            position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

            with torch.no_grad():
                outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                               position_ids=position_ids)

            for h in hooks:
                h.remove()

            # Collect states
            for l in conv_layers:
                key = f"L{l}"
                if key in captured:
                    perturbed_states[l].append(captured[key][0, -1, :].numpy())

            perturbed_logits.append(outputs.logits[0, -1].float().cpu().numpy())

        # Compute convergence metrics
        word_data = {"baseline_norm": {}, "convergence": {}, "inter_perturb_cosine": {}}

        for l in conv_layers:
            states = perturbed_states[l]
            if len(states) < 2:
                continue

            # 1. Convergence to baseline: average cosine with baseline at this layer
            baseline_state = baseline_h.get(l)
            if baseline_state is not None:
                cosines_to_baseline = []
                for s in states:
                    ns, nb = np.linalg.norm(s), np.linalg.norm(baseline_state)
                    if ns > 1e-10 and nb > 1e-10:
                        cosines_to_baseline.append(float(np.dot(s, baseline_state) / (ns * nb)))
                word_data["convergence"][str(l)] = {
                    "mean_cosine_to_baseline": float(np.mean(cosines_to_baseline)),
                    "std_cosine_to_baseline": float(np.std(cosines_to_baseline)),
                }

            # 2. Inter-perturbation convergence: average pairwise cosine between perturbed states
            pairwise_cosines = []
            for i in range(len(states)):
                for j in range(i + 1, len(states)):
                    ni, nj = np.linalg.norm(states[i]), np.linalg.norm(states[j])
                    if ni > 1e-10 and nj > 1e-10:
                        pairwise_cosines.append(float(np.dot(states[i], states[j]) / (ni * nj)))
            if pairwise_cosines:
                word_data["inter_perturb_cosine"][str(l)] = {
                    "mean": float(np.mean(pairwise_cosines)),
                    "std": float(np.std(pairwise_cosines)),
                }

            # Norm of baseline
            if baseline_state is not None:
                word_data["baseline_norm"][str(l)] = float(np.linalg.norm(baseline_state))

        all_data[word] = word_data
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        log_time(f"  '{word}' done. GPU={gpu_mem:.2f}GB")

    # Save raw data
    out_path = RESULT_DIR / f"{model_name}_exp_b.json"
    with open(out_path, "w") as f:
        json.dump(all_data, f)
    log_time(f"Exp B raw data saved to {out_path}")

    # ===== Analysis =====
    log_time("Analyzing attractor convergence...")

    # For within-category pairs: compute average cosine between their deep-layer states
    # (using the perturbed states to measure "attractor basin overlap")
    def get_word_conv_data(word):
        if word not in all_data:
            return None
        return all_data[word]

    # Convergence to baseline (attractor strength)
    # If deep layers have high cosine to baseline even with perturbation → strong attractor
    summary_data = {"per_layer_convergence": {}, "attractor_depth": {}}

    for l in conv_layers:
        conv_vals = []
        for word in ALL_WORDS:
            wd = get_word_conv_data(word)
            if wd and str(l) in wd.get("convergence", {}):
                conv_vals.append(wd["convergence"][str(l)]["mean_cosine_to_baseline"])
        if conv_vals:
            summary_data["per_layer_convergence"][str(l)] = {
                "mean": float(np.mean(conv_vals)),
                "std": float(np.std(conv_vals)),
            }

    # Inter-perturbation convergence (attractor basin tightness)
    for l in conv_layers:
        inter_vals = []
        for word in ALL_WORDS:
            wd = get_word_conv_data(word)
            if wd and str(l) in wd.get("inter_perturb_cosine", {}):
                inter_vals.append(wd["inter_perturb_cosine"][str(l)]["mean"])
        if inter_vals:
            summary_data["attractor_depth"][str(l)] = {
                "mean": float(np.mean(inter_vals)),
                "std": float(np.std(inter_vals)),
            }

    # Cross-token attractor comparison
    # For each pair, compute cosine between their deep-layer baseline states
    cross_attractor = {"within": [], "between": []}

    # Get baseline states at last layer
    last_layer = conv_layers[-1]
    baseline_states = {}
    for word in ALL_WORDS:
        # We need to re-extract baseline states — they're in the raw data
        # Actually, let's just use the convergence data
        pass

    # Simpler: compare convergence profiles between within/between pairs
    # If within-category tokens have more similar convergence curves → shared attractor basins
    for pair_type, pairs in [("within", WITHIN_PAIRS), ("between", BETWEEN_PAIRS)]:
        for wa, wb in pairs:
            da = get_word_conv_data(wa)
            db = get_word_conv_data(wb)
            if da is None or db is None:
                continue

            # Compare convergence profiles (cosine to baseline at each layer)
            conv_a = [da["convergence"].get(str(l), {}).get("mean_cosine_to_baseline")
                      for l in conv_layers if str(l) in da.get("convergence", {})]
            conv_b = [db["convergence"].get(str(l), {}).get("mean_cosine_to_baseline")
                      for l in conv_layers if str(l) in db.get("convergence", {})]

            if len(conv_a) == len(conv_b) and len(conv_a) > 0:
                # Cosine similarity of convergence profiles
                a, b = np.array(conv_a), np.array(conv_b)
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-10 and nb > 1e-10:
                    cross_attractor[pair_type].append(float(np.dot(a, b) / (na * nb)))

    summary_data["cross_attractor"] = {
        "within_mean": float(np.mean(cross_attractor["within"])) if cross_attractor["within"] else None,
        "between_mean": float(np.mean(cross_attractor["between"])) if cross_attractor["between"] else None,
        "within_n": len(cross_attractor["within"]),
        "between_n": len(cross_attractor["between"]),
        "delta": float(np.mean(cross_attractor["within"]) - np.mean(cross_attractor["between"]))
            if cross_attractor["within"] and cross_attractor["between"] else None,
    }

    # Save summary
    summary = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "conv_layers": conv_layers,
        "n_perturbations": N_ATTRACTOR_PERTURBATIONS,
        "perturb_magnitude": perturb_magnitude,
        **summary_data,
    }

    out_path = RESULT_DIR / f"{model_name}_exp_b_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_time(f"Exp B summary saved to {out_path}")

    # Print key results
    log_time(f"=== Exp B: Attractor Convergence ({model_name}) ===")

    log_time("  Convergence to baseline (cosine) by layer:")
    for l in conv_layers:
        cd = summary_data["per_layer_convergence"].get(str(l), {})
        if cd:
            log_time(f"    L{l}: mean_cosine_to_baseline={cd['mean']:.4f} ± {cd['std']:.4f}")

    log_time("  Inter-perturbation convergence (attractor basin tightness) by layer:")
    for l in conv_layers:
        ad = summary_data["attractor_depth"].get(str(l), {})
        if ad:
            log_time(f"    L{l}: mean_pairwise_cosine={ad['mean']:.4f} ± {ad['std']:.4f}")

    ca = summary_data["cross_attractor"]
    log_time(f"  Cross-attractor: within={ca['within_mean']}, between={ca['between_mean']}, "
             f"delta={ca['delta']}")

    return summary


# ===== Main =====

def main():
    global _log_file

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"

    # Setup log file
    log_path = RESULT_DIR / f"{model_name}_phase275.log"
    _log_file = str(log_path)

    log_time(f"Phase 275: CDRE — Conditional Dynamical Reverse Engineering")
    log_time(f"Model: {model_name}")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log_time(f"Model info: n_layers={info.n_layers}, d_model={info.d_model}, "
             f"class={info.model_class}")

    # ===== Run Experiments =====

    # Exp A + C: Conditional Jacobian Similarity + Logit Fingerprint
    log_time("=" * 60)
    log_time("Starting Exp A: Conditional Jacobian Similarity + Exp C: Logit Fingerprint")
    log_time("=" * 60)
    summary_a = experiment_a(model, tokenizer, device, model_name)

    # Exp B: Attractor Convergence
    log_time("=" * 60)
    log_time("Starting Exp B: Attractor Convergence")
    log_time("=" * 60)
    summary_b = experiment_b(model, tokenizer, device, model_name)

    # ===== Final Summary =====
    log_time("=" * 60)
    log_time("FINAL SUMMARY")
    log_time("=" * 60)

    js = summary_a["jacobian_similarity"]
    log_time(f"Jacobian Similarity: within={js['within_mean']}, between={js['between_mean']}, "
             f"delta={js['delta']}")

    ca = summary_b["cross_attractor"]
    log_time(f"Cross-Attractor: within={ca['within_mean']}, between={ca['between_mean']}, "
             f"delta={ca['delta']}")

    # Release model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released. Phase 275 complete.")


if __name__ == "__main__":
    main()
