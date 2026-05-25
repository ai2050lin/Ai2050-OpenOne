"""
Phase 272: Computational Path Reuse & Divergence
=================================================

Core shift: From representation analysis to computational structure analysis.

Three experiments:
A. Path Overlap Matrix — How much computation do different concepts share?
   - Attention head importance overlap (via output_attentions)
   - MLP residual contribution overlap (via hidden state deltas)
   - Within-category (apple↔banana) vs between-category (apple↔car)

B. Divergence Layer Detection — At which layer do related concepts diverge?
   - Track layer-by-layer cosine distance
   - Find "divergence point" for within vs between pairs

C. Context-Conditional Routing — Same word, different context → different paths?
   - "bank" in "river bank" vs "bank account"
   - Attention head routing differences
   - MLP contribution differences

Usage:
  python tests/glm5/phase272_path_reuse.py qwen3
  python tests/glm5/phase272_path_reuse.py glm4
  python tests/glm5/phase272_path_reuse.py deepseek7b
"""
import sys, os, json, gc, time, warnings, random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_utils import MODEL_CONFIGS, get_model_info, get_W_U, get_layers

RESULT_DIR = Path("results/phase272_path_reuse")
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
    "tools": ["hammer", "drill", "saw", "ruler", "knife", "wrench", "shovel", "pliers", "chisel", "spade"],
    "body": ["head", "hand", "foot", "arm", "leg", "eye", "ear", "nose", "neck", "back"],
}

CROSS_CATEGORY_PAIRS = [
    ("apple", "dog"), ("banana", "car"), ("orange", "hammer"),
    ("grape", "head"), ("mango", "bus"), ("peach", "drill"),
    ("lemon", "foot"), ("cherry", "train"), ("pear", "wolf"),
    ("lime", "saw"),
]

CONTEXT_CONDITIONAL = [
    ("The fish sat on the river bank", "She deposited money in the bank"),
    ("The light from the sun was bright", "The box was very light to carry"),
    ("He likes to play football", "She wants to play the piano"),
    ("The bark of the dog was loud", "The bark of the tree was rough"),
    ("A fast runner won the race", "The human race spread globally"),
    ("The spring season brings flowers", "The metal spring broke"),
    ("The bat flew in the cave", "He swung the baseball bat"),
    ("She broke the glass window", "He drank from a glass"),
    ("The match ended in a draw", "He struck a match to light the fire"),
    ("She painted the door red", "He read the book carefully"),
]


# ===== Model Loading (BF16 + device_map=auto + flash) =====

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


# ===== Core: Extract Per-Layer Computation Profile =====

def extract_computation_profile(model, tokenizer, input_device, prompt, n_layers):
    """
    Extract the computation profile for a single prompt.
    
    Returns:
        head_importance: {layer: [n_heads] L2 norm of attention from last pos}
        mlp_delta: {layer: [d_model] residual contribution of MLP}
        attn_delta: {layer: [d_model] residual contribution of attention}
        residual: {layer: [d_model] full residual stream}
        head_attn_weights: {layer: [n_heads, seq_len] attention weights from last pos}
    """
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attn_mask = inputs["attention_mask"].to(input_device)
    
    # Run with hidden states and attentions
    with torch.no_grad():
        out = model(
            input_ids=input_ids, attention_mask=attn_mask,
            output_hidden_states=True, output_attentions=True,
        )
    
    last_pos = int(attn_mask.sum().item()) - 1
    d_model = out.hidden_states[0].shape[-1]
    
    # Get n_heads from attention output
    n_heads = None
    if out.attentions and len(out.attentions) > 0:
        n_heads = out.attentions[0].shape[1]
    
    head_importance = {}
    head_attn_weights = {}
    mlp_delta = {}
    attn_delta = {}
    residual = {}
    
    # Residual stream at each layer
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, last_pos, :].detach().float().cpu().numpy()
        residual[li] = h
    
    # Layer contributions: delta = h_{l+1} - h_l
    for li in range(n_layers):
        # Total layer contribution
        delta = residual[li + 1] - residual[li]
        
        # We can't separate attn vs mlp from hidden_states alone,
        # but we can compute the total delta and analyze it
        mlp_delta[li] = delta  # This is the full layer delta (attn + mlp + layernorm)
        
        # Attention head importance from attention weights
        if out.attentions and li < len(out.attentions):
            attn_weights = out.attentions[li]  # [1, n_heads, seq_len, seq_len]
            # Attention FROM last position
            attn_from_last = attn_weights[0, :, last_pos, :last_pos+1]  # [n_heads, seq_len]
            head_imp = torch.norm(attn_from_last.float(), dim=-1).cpu().numpy()  # [n_heads]
            head_importance[li] = head_imp
            # Also store full attention pattern for overlap analysis
            head_attn_weights[li] = attn_from_last.float().cpu().numpy()  # [n_heads, seq_len]
    
    del out
    torch.cuda.empty_cache()
    
    return {
        "head_importance": head_importance,
        "head_attn_weights": head_attn_weights,
        "layer_delta": mlp_delta,  # renamed for clarity
        "residual": residual,
        "n_heads": n_heads,
    }


# ===== Experiment A: Path Overlap Matrix =====

def experiment_a_path_overlap(model, tokenizer, info, input_device):
    """
    Measure computational path overlap between concept pairs.
    
    Metrics:
    1. Attention head importance overlap: Spearman correlation of head importance vectors
    2. Attention pattern overlap: correlation of attention weight distributions
    3. Layer delta overlap: cosine similarity of per-layer residual changes
    4. Top-dimension overlap: Jaccard of top-50 most-changed residual dimensions
    
    Compare within-category (apple↔banana) vs between-category (apple↔car).
    """
    import torch
    from scipy.stats import spearmanr
    
    log_time("=" * 60)
    log_time("Experiment A: Path Overlap Matrix")
    log_time("=" * 60)
    
    n_layers = info.n_layers
    
    # Collect all words
    all_words = []
    word_category = {}
    for cat, words in CATEGORIES.items():
        for w in words:
            if w not in all_words:
                all_words.append(w)
                word_category[w] = cat
    
    template = "The {} is"
    
    # Extract profiles for all words
    profiles = {}
    log_time(f"  Extracting computation profiles for {len(all_words)} words...")
    
    for wi, word in enumerate(all_words):
        prompt = template.format(word)
        profile = extract_computation_profile(model, tokenizer, input_device, prompt, n_layers)
        profiles[word] = profile
        
        if (wi + 1) % 10 == 0:
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log_time(f"  {wi+1}/{len(all_words)} profiles extracted, GPU={gpu_mem:.1f}GB")
        
        gc.collect()
    
    # ---- Compute Overlap Metrics ----
    sample_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    
    log_time(f"  Computing overlaps at layers {sample_layers}...")
    
    def compute_head_importance_overlap(hi_a, hi_b):
        """Spearman correlation of head importance rankings."""
        if hi_a is None or hi_b is None:
            return 0.0
        if len(hi_a) != len(hi_b):
            return 0.0
        r, p = spearmanr(hi_a, hi_b)
        return float(r) if not np.isnan(r) else 0.0
    
    def compute_head_jaccard(hi_a, hi_b, pct=50):
        """Jaccard overlap of top-pct most important heads."""
        if hi_a is None or hi_b is None:
            return 0.0
        if len(hi_a) != len(hi_b):
            return 0.0
        thresh_a = np.percentile(hi_a, pct)
        thresh_b = np.percentile(hi_b, pct)
        active_a = set(np.where(hi_a >= thresh_a)[0])
        active_b = set(np.where(hi_b >= thresh_b)[0])
        if len(active_a | active_b) == 0:
            return 1.0
        return len(active_a & active_b) / len(active_a | active_b)
    
    def compute_attn_pattern_correlation(aw_a, aw_b):
        """Correlation of attention weight patterns (flattened)."""
        if aw_a is None or aw_b is None:
            return 0.0
        if aw_a.shape != aw_b.shape:
            return 0.0
        flat_a = aw_a.flatten()
        flat_b = aw_b.flatten()
        r, p = spearmanr(flat_a, flat_b)
        return float(r) if not np.isnan(r) else 0.0
    
    def compute_delta_cosine(delta_a, delta_b):
        """Cosine similarity of layer delta vectors."""
        na = np.linalg.norm(delta_a)
        nb = np.linalg.norm(delta_b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(delta_a, delta_b) / (na * nb))
    
    def compute_top_dim_jaccard(delta_a, delta_b, k=50):
        """Jaccard overlap of top-k most-changed residual dimensions."""
        abs_a = np.abs(delta_a)
        abs_b = np.abs(delta_b)
        top_a = set(np.argsort(abs_a)[-k:])
        top_b = set(np.argsort(abs_b)[-k:])
        if len(top_a | top_b) == 0:
            return 1.0
        return len(top_a & top_b) / len(top_a | top_b)
    
    # Compute overlaps for within-category and between-category pairs
    within_overlaps = defaultdict(list)
    between_overlaps = defaultdict(list)
    
    for li in sample_layers:
        # Within-category
        for cat, words in CATEGORIES.items():
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    wa, wb = words[i], words[j]
                    pa, pb = profiles.get(wa), profiles.get(wb)
                    if pa is None or pb is None:
                        continue
                    
                    hi_overlap = compute_head_importance_overlap(
                        pa["head_importance"].get(li), pb["head_importance"].get(li))
                    hj_overlap = compute_head_jaccard(
                        pa["head_importance"].get(li), pb["head_importance"].get(li))
                    ap_corr = compute_attn_pattern_correlation(
                        pa["head_attn_weights"].get(li), pb["head_attn_weights"].get(li))
                    delta_cos = compute_delta_cosine(
                        pa["layer_delta"].get(li, np.zeros(1)), pb["layer_delta"].get(li, np.zeros(1)))
                    top_jaccard = compute_top_dim_jaccard(
                        pa["layer_delta"].get(li, np.zeros(1)), pb["layer_delta"].get(li, np.zeros(1)))
                    
                    within_overlaps[li].append({
                        "head_importance_corr": hi_overlap,
                        "head_jaccard": hj_overlap,
                        "attn_pattern_corr": ap_corr,
                        "delta_cosine": delta_cos,
                        "top_dim_jaccard": top_jaccard,
                    })
        
        # Between-category
        for wa, wb in CROSS_CATEGORY_PAIRS:
            pa, pb = profiles.get(wa), profiles.get(wb)
            if pa is None or pb is None:
                continue
            
            hi_overlap = compute_head_importance_overlap(
                pa["head_importance"].get(li), pb["head_importance"].get(li))
            hj_overlap = compute_head_jaccard(
                pa["head_importance"].get(li), pb["head_importance"].get(li))
            ap_corr = compute_attn_pattern_correlation(
                pa["head_attn_weights"].get(li), pb["head_attn_weights"].get(li))
            delta_cos = compute_delta_cosine(
                pa["layer_delta"].get(li, np.zeros(1)), pb["layer_delta"].get(li, np.zeros(1)))
            top_jaccard = compute_top_dim_jaccard(
                pa["layer_delta"].get(li, np.zeros(1)), pb["layer_delta"].get(li, np.zeros(1)))
            
            between_overlaps[li].append({
                "head_importance_corr": hi_overlap,
                "head_jaccard": hj_overlap,
                "attn_pattern_corr": ap_corr,
                "delta_cosine": delta_cos,
                "top_dim_jaccard": top_jaccard,
            })
    
    # ---- Aggregate ----
    results = {"sample_layers": sample_layers, "within_category": {}, "between_category": {}}
    
    for li in sample_layers:
        w_items = within_overlaps.get(li, [])
        b_items = between_overlaps.get(li, [])
        
        w_agg = {}
        b_agg = {}
        for key in ["head_importance_corr", "head_jaccard", "attn_pattern_corr", "delta_cosine", "top_dim_jaccard"]:
            w_vals = [x[key] for x in w_items]
            b_vals = [x[key] for x in b_items]
            w_agg[key] = {"mean": float(np.mean(w_vals)), "std": float(np.std(w_vals)), "n": len(w_vals)}
            b_agg[key] = {"mean": float(np.mean(b_vals)), "std": float(np.std(b_vals)), "n": len(b_vals)}
        
        results["within_category"][str(li)] = w_agg
        results["between_category"][str(li)] = b_agg
        
        log_time(f"  L{li}: Within vs Between:")
        log_time(f"    Head Imp Corr:   {w_agg['head_importance_corr']['mean']:.4f} vs {b_agg['head_importance_corr']['mean']:.4f} (Δ={w_agg['head_importance_corr']['mean']-b_agg['head_importance_corr']['mean']:+.4f})")
        log_time(f"    Head Jaccard:    {w_agg['head_jaccard']['mean']:.4f} vs {b_agg['head_jaccard']['mean']:.4f} (Δ={w_agg['head_jaccard']['mean']-b_agg['head_jaccard']['mean']:+.4f})")
        log_time(f"    Attn Pat Corr:   {w_agg['attn_pattern_corr']['mean']:.4f} vs {b_agg['attn_pattern_corr']['mean']:.4f} (Δ={w_agg['attn_pattern_corr']['mean']-b_agg['attn_pattern_corr']['mean']:+.4f})")
        log_time(f"    Delta Cosine:    {w_agg['delta_cosine']['mean']:.4f} vs {b_agg['delta_cosine']['mean']:.4f} (Δ={w_agg['delta_cosine']['mean']-b_agg['delta_cosine']['mean']:+.4f})")
        log_time(f"    Top-Dim Jaccard: {w_agg['top_dim_jaccard']['mean']:.4f} vs {b_agg['top_dim_jaccard']['mean']:.4f} (Δ={w_agg['top_dim_jaccard']['mean']-b_agg['top_dim_jaccard']['mean']:+.4f})")
    
    # Per-category breakdown
    results["within_per_category"] = {}
    for cat in CATEGORIES:
        words = CATEGORIES[cat]
        cat_overlaps = defaultdict(list)
        for li in sample_layers:
            for i in range(len(words)):
                for j in range(i+1, len(words)):
                    wa, wb = words[i], words[j]
                    pa, pb = profiles.get(wa), profiles.get(wb)
                    if pa is None or pb is None:
                        continue
                    for key in ["head_importance_corr", "delta_cosine", "top_dim_jaccard"]:
                        if key == "head_importance_corr":
                            val = compute_head_importance_overlap(
                                pa["head_importance"].get(li), pb["head_importance"].get(li))
                        elif key == "delta_cosine":
                            val = compute_delta_cosine(
                                pa["layer_delta"].get(li, np.zeros(1)), pb["layer_delta"].get(li, np.zeros(1)))
                        elif key == "top_dim_jaccard":
                            val = compute_top_dim_jaccard(
                                pa["layer_delta"].get(li, np.zeros(1)), pb["layer_delta"].get(li, np.zeros(1)))
                        cat_overlaps[(cat, li, key)].append(val)
        
        results["within_per_category"][cat] = {}
        for li in sample_layers:
            results["within_per_category"][cat][str(li)] = {
                key: float(np.mean(cat_overlaps.get((cat, li, key), [0])))
                for key in ["head_importance_corr", "delta_cosine", "top_dim_jaccard"]
            }
    
    return results


# ===== Experiment B: Divergence Layer Detection =====

def experiment_b_divergence_detection(model, tokenizer, info, input_device):
    """
    Track at which layer related concepts start to diverge.
    
    For within-category pairs: apple vs banana → when do they start separating?
    For between-category pairs: apple vs car → when do they start separating?
    
    Hypothesis: Between-category pairs diverge earlier than within-category pairs.
    """
    import torch
    from scipy.stats import spearmanr
    
    log_time("=" * 60)
    log_time("Experiment B: Divergence Layer Detection")
    log_time("=" * 60)
    
    n_layers = info.n_layers
    d_model = info.d_model
    template = "The {} is"
    
    # Within-category pairs (5 per category)
    within_pairs = []
    for cat, words in CATEGORIES.items():
        for i in range(min(5, len(words) - 1)):
            within_pairs.append((words[i], words[i+1], cat, "within"))
    
    # Between-category pairs
    between_pairs = [(wa, wb, "cross", "between") for wa, wb in CROSS_CATEGORY_PAIRS]
    
    all_pairs = within_pairs + between_pairs
    
    divergence_results = []
    log_time(f"  Processing {len(all_pairs)} pairs...")
    
    for pi, (wa, wb, cat, pair_type) in enumerate(all_pairs):
        profile_a = extract_computation_profile(model, tokenizer, input_device,
                                                 template.format(wa), n_layers)
        profile_b = extract_computation_profile(model, tokenizer, input_device,
                                                 template.format(wb), n_layers)
        
        # Layer-by-layer divergence
        layer_divs = {}
        for li in range(n_layers + 1):
            h_a = profile_a["residual"].get(li)
            h_b = profile_b["residual"].get(li)
            if h_a is not None and h_b is not None:
                na = np.linalg.norm(h_a)
                nb = np.linalg.norm(h_b)
                cos_sim = np.dot(h_a, h_b) / (na * nb) if na > 1e-10 and nb > 1e-10 else 0
                cos_dist = 1.0 - cos_sim
            else:
                cos_dist = 1.0
            layer_divs[li] = float(cos_dist)
        
        # Layer-by-layer head importance divergence
        head_divs = {}
        for li in range(n_layers):
            hi_a = profile_a["head_importance"].get(li)
            hi_b = profile_b["head_importance"].get(li)
            if hi_a is not None and hi_b is not None:
                r, _ = spearmanr(hi_a, hi_b)
                head_divs[li] = float(r) if not np.isnan(r) else 0.0
            else:
                head_divs[li] = 0.0
        
        # Layer-by-layer delta divergence (cosine of layer contributions)
        delta_divs = {}
        for li in range(n_layers):
            d_a = profile_a["layer_delta"].get(li)
            d_b = profile_b["layer_delta"].get(li)
            if d_a is not None and d_b is not None:
                delta_cos = compute_delta_cosine(d_a, d_b)
                delta_divs[li] = float(delta_cos)
            else:
                delta_divs[li] = 0.0
        
        # Find divergence layer: first layer where cos_dist > 2× initial
        initial_div = layer_divs.get(0, 0.01)
        divergence_layer = None
        for li in range(1, n_layers + 1):
            if layer_divs.get(li, 0) > 2 * max(initial_div, 0.01):
                divergence_layer = li
                break
        
        result = {
            "word_a": wa, "word_b": wb, "category": cat, "pair_type": pair_type,
            "divergence_layer": divergence_layer,
            "initial_cos_dist": layer_divs.get(0, 0),
            "final_cos_dist": layer_divs.get(n_layers, 0),
            "layer_cos_distances": {str(k): v for k, v in layer_divs.items()},
            "layer_head_corr": {str(k): v for k, v in head_divs.items()},
            "layer_delta_cosine": {str(k): v for k, v in delta_divs.items()},
        }
        divergence_results.append(result)
        
        if (pi + 1) % 10 == 0:
            log_time(f"  {pi+1}/{len(all_pairs)} pairs processed")
        
        gc.collect()
    
    # Aggregate
    within_divs = [r for r in divergence_results if r["pair_type"] == "within"]
    between_divs = [r for r in divergence_results if r["pair_type"] == "between"]
    
    sample_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]))
    
    def avg_trajectory(results_list, key, n_layers):
        traj = defaultdict(list)
        for r in results_list:
            data = r.get(key, {})
            for li in range(n_layers + 1):
                if str(li) in data:
                    traj[li].append(data[str(li)])
        return {li: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                for li, v in traj.items() if v}
    
    w_cos_traj = avg_trajectory(within_divs, "layer_cos_distances", n_layers)
    b_cos_traj = avg_trajectory(between_divs, "layer_cos_distances", n_layers)
    w_head_traj = avg_trajectory(within_divs, "layer_head_corr", n_layers)
    b_head_traj = avg_trajectory(between_divs, "layer_head_corr", n_layers)
    w_delta_traj = avg_trajectory(within_divs, "layer_delta_cosine", n_layers)
    b_delta_traj = avg_trajectory(between_divs, "layer_delta_cosine", n_layers)
    
    within_div_layers = [r["divergence_layer"] for r in within_divs if r["divergence_layer"] is not None]
    between_div_layers = [r["divergence_layer"] for r in between_divs if r["divergence_layer"] is not None]
    
    summary = {
        "sample_layers": sample_layers,
        "within_cos_trajectory": {str(k): v for k, v in w_cos_traj.items() if k in sample_layers},
        "between_cos_trajectory": {str(k): v for k, v in b_cos_traj.items() if k in sample_layers},
        "within_head_corr_trajectory": {str(k): v for k, v in w_head_traj.items() if k in sample_layers},
        "between_head_corr_trajectory": {str(k): v for k, v in b_head_traj.items() if k in sample_layers},
        "within_delta_cos_trajectory": {str(k): v for k, v in w_delta_traj.items() if k in sample_layers},
        "between_delta_cos_trajectory": {str(k): v for k, v in b_delta_traj.items() if k in sample_layers},
        "within_divergence_layer_stats": {
            "mean": float(np.mean(within_div_layers)) if within_div_layers else None,
            "median": float(np.median(within_div_layers)) if within_div_layers else None,
            "n_found": len(within_div_layers), "n_total": len(within_divs),
        },
        "between_divergence_layer_stats": {
            "mean": float(np.mean(between_div_layers)) if between_div_layers else None,
            "median": float(np.median(between_div_layers)) if between_div_layers else None,
            "n_found": len(between_div_layers), "n_total": len(between_divs),
        },
        "per_pair": divergence_results,
    }
    
    for li in sample_layers:
        wc = w_cos_traj.get(li, {})
        bc = b_cos_traj.get(li, {})
        wh = w_head_traj.get(li, {})
        bh = b_head_traj.get(li, {})
        wd = w_delta_traj.get(li, {})
        bd = b_delta_traj.get(li, {})
        log_time(f"  L{li}: cos_dist W={wc.get('mean',0):.4f} B={bc.get('mean',0):.4f} Δ={bc.get('mean',0)-wc.get('mean',0):.4f} | "
                 f"head_corr W={wh.get('mean',0):.4f} B={bh.get('mean',0):.4f} | "
                 f"delta_cos W={wd.get('mean',0):.4f} B={bd.get('mean',0):.4f}")
    
    return summary


def compute_delta_cosine(delta_a, delta_b):
    na = np.linalg.norm(delta_a)
    nb = np.linalg.norm(delta_b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(delta_a, delta_b) / (na * nb))


# ===== Experiment C: Context-Conditional Routing =====

def experiment_c_context_routing(model, tokenizer, info, input_device):
    """
    Test whether the same word in different contexts activates different computation paths.
    """
    import torch
    from scipy.stats import spearmanr
    
    log_time("=" * 60)
    log_time("Experiment C: Context-Conditional Routing")
    log_time("=" * 60)
    
    n_layers = info.n_layers
    results = []
    
    log_time(f"  Processing {len(CONTEXT_CONDITIONAL)} context pairs...")
    
    for ci, (ctx_a, ctx_b) in enumerate(CONTEXT_CONDITIONAL):
        profile_a = extract_computation_profile(model, tokenizer, input_device, ctx_a, n_layers)
        profile_b = extract_computation_profile(model, tokenizer, input_device, ctx_b, n_layers)
        
        # Layer-by-layer divergence
        layer_divs = {}
        for li in range(n_layers + 1):
            h_a = profile_a["residual"].get(li)
            h_b = profile_b["residual"].get(li)
            if h_a is not None and h_b is not None:
                na = np.linalg.norm(h_a)
                nb = np.linalg.norm(h_b)
                cos_sim = np.dot(h_a, h_b) / (na * nb) if na > 1e-10 and nb > 1e-10 else 0
                cos_dist = 1.0 - cos_sim
            else:
                cos_dist = 1.0
            layer_divs[li] = float(cos_dist)
        
        # Head importance correlation per layer
        head_corrs = {}
        for li in range(n_layers):
            hi_a = profile_a["head_importance"].get(li)
            hi_b = profile_b["head_importance"].get(li)
            if hi_a is not None and hi_b is not None and len(hi_a) == len(hi_b):
                r, _ = spearmanr(hi_a, hi_b)
                head_corrs[li] = float(r) if not np.isnan(r) else 0.0
            else:
                head_corrs[li] = 0.0
        
        # Layer delta cosine per layer
        delta_cos = {}
        for li in range(n_layers):
            d_a = profile_a["layer_delta"].get(li)
            d_b = profile_b["layer_delta"].get(li)
            if d_a is not None and d_b is not None:
                delta_cos[li] = compute_delta_cosine(d_a, d_b)
            else:
                delta_cos[li] = 0.0
        
        # Top-dimension Jaccard per layer
        top_jaccard = {}
        for li in range(n_layers):
            d_a = profile_a["layer_delta"].get(li)
            d_b = profile_b["layer_delta"].get(li)
            if d_a is not None and d_b is not None and len(d_a) > 50:
                abs_a = np.abs(d_a)
                abs_b = np.abs(d_b)
                top_a = set(np.argsort(abs_a)[-50:])
                top_b = set(np.argsort(abs_b)[-50:])
                top_jaccard[li] = len(top_a & top_b) / len(top_a | top_b) if len(top_a | top_b) > 0 else 0
            else:
                top_jaccard[li] = 0.0
        
        result = {
            "context_a": ctx_a,
            "context_b": ctx_b,
            "layer_cos_distances": {str(k): v for k, v in layer_divs.items()},
            "layer_head_corr": {str(k): v for k, v in head_corrs.items()},
            "layer_delta_cosine": {str(k): v for k, v in delta_cos.items()},
            "layer_top_dim_jaccard": {str(k): v for k, v in top_jaccard.items()},
        }
        results.append(result)
        
        emb_div = layer_divs.get(0, 0)
        final_div = layer_divs.get(n_layers, 0)
        mid_head_corr = head_corrs.get(n_layers // 2, 0)
        mid_delta_cos = delta_cos.get(n_layers // 2, 0)
        log_time(f"  Pair {ci+1}: emb_div={emb_div:.4f}, final_div={final_div:.4f}, "
                 f"mid_head_corr={mid_head_corr:.4f}, mid_delta_cos={mid_delta_cos:.4f}")
        
        gc.collect()
    
    # Aggregate
    sample_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]))
    
    avg_divs = defaultdict(list)
    avg_head = defaultdict(list)
    avg_delta = defaultdict(list)
    avg_topj = defaultdict(list)
    
    for r in results:
        for li in sample_layers:
            if str(li) in r["layer_cos_distances"]:
                avg_divs[li].append(r["layer_cos_distances"][str(li)])
            if str(li) in r["layer_head_corr"]:
                avg_head[li].append(r["layer_head_corr"][str(li)])
            if str(li) in r["layer_delta_cosine"]:
                avg_delta[li].append(r["layer_delta_cosine"][str(li)])
            if str(li) in r["layer_top_dim_jaccard"]:
                avg_topj[li].append(r["layer_top_dim_jaccard"][str(li)])
    
    summary = {
        "sample_layers": sample_layers,
        "avg_cos_distance": {str(li): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                             for li, v in avg_divs.items()},
        "avg_head_corr": {str(li): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                          for li, v in avg_head.items()},
        "avg_delta_cosine": {str(li): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                             for li, v in avg_delta.items()},
        "avg_top_dim_jaccard": {str(li): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                for li, v in avg_topj.items()},
        "per_pair": results,
    }
    
    for li in sample_layers:
        log_time(f"  L{li}: cos_dist={np.mean(avg_divs.get(li,[0])):.4f}, "
                 f"head_corr={np.mean(avg_head.get(li,[0])):.4f}, "
                 f"delta_cos={np.mean(avg_delta.get(li,[0])):.4f}, "
                 f"top_jaccard={np.mean(avg_topj.get(li,[0])):.4f}")
    
    return summary


# ===== Main =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    global _log_file
    log_path = RESULT_DIR / f"{model_name}_phase272.log"
    _log_file = str(log_path)
    
    log_time(f"Phase 272: Path Reuse & Divergence - {model_name}")
    log_time(f"Log: {log_path}")
    
    import torch
    
    # ---- Experiment A: Path Overlap ----
    model, tokenizer, info = load_model_bf16(model_name)
    input_device = get_input_device(model)
    
    log_time("\nStarting Experiment A: Path Overlap Matrix")
    t0 = time.time()
    result_a = experiment_a_path_overlap(model, tokenizer, info, input_device)
    t_a = time.time() - t0
    log_time(f"Experiment A completed in {t_a:.0f}s")
    
    with open(RESULT_DIR / f"{model_name}_path_overlap.json", "w") as f:
        json.dump(result_a, f, indent=2, ensure_ascii=False)
    log_time(f"Saved: {model_name}_path_overlap.json")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released, reloading for Experiment B...")
    
    # ---- Experiment B: Divergence Detection ----
    model, tokenizer, info = load_model_bf16(model_name)
    input_device = get_input_device(model)
    
    log_time("\nStarting Experiment B: Divergence Layer Detection")
    t0 = time.time()
    result_b = experiment_b_divergence_detection(model, tokenizer, info, input_device)
    t_b = time.time() - t0
    log_time(f"Experiment B completed in {t_b:.0f}s")
    
    with open(RESULT_DIR / f"{model_name}_divergence.json", "w") as f:
        json.dump(result_b, f, indent=2, ensure_ascii=False)
    log_time(f"Saved: {model_name}_divergence.json")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released, reloading for Experiment C...")
    
    # ---- Experiment C: Context-Conditional Routing ----
    model, tokenizer, info = load_model_bf16(model_name)
    input_device = get_input_device(model)
    
    log_time("\nStarting Experiment C: Context-Conditional Routing")
    t0 = time.time()
    result_c = experiment_c_context_routing(model, tokenizer, info, input_device)
    t_c = time.time() - t0
    log_time(f"Experiment C completed in {t_c:.0f}s")
    
    with open(RESULT_DIR / f"{model_name}_context_routing.json", "w") as f:
        json.dump(result_c, f, indent=2, ensure_ascii=False)
    log_time(f"Saved: {model_name}_context_routing.json")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    log_time(f"\nPhase 272 complete for {model_name}!")
    log_time(f"  Exp A (Path Overlap): {t_a:.0f}s")
    log_time(f"  Exp B (Divergence): {t_b:.0f}s")
    log_time(f"  Exp C (Context Routing): {t_c:.0f}s")


if __name__ == "__main__":
    main()
