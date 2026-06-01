"""
Phase 322: Indirect Readout Path Tracing + Block Recomputation
==============================================================

Phase 321 discovered that readout alignment is INVERSELY correlated with causal efficacy:
- GLM4: strongest causal efficacy but lowest readout alignment
- Qwen3/DS7B: highest readout alignment but weakest causal efficacy

This means the causal mechanism is NOT direct alignment between direction and W_U,
but an INDIRECT path: direction → subsequent layer transformation → output.

This script traces the indirect readout path by:
1. Computing direction transformation across layers (how the direction rotates/scales)
2. Testing block recomputation: replacing a block of layers vs single-layer injection
3. Decomposing attention vs MLP contributions to the transformation

Usage:
  python tests/glm5/phase322_indirect_readout.py qwen3
  python tests/glm5/phase322_indirect_readout.py glm4
  python tests/glm5/phase322_indirect_readout.py deepseek7b
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model, get_W_U

RESULT_DIR = Path("results/phase322_indirect_readout")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except:
            pass


# ===== Stimuli =====
ATTRIBUTE_PAIRS = [
    ("apple", "red"), ("sky", "blue"), ("fire", "hot"),
    ("ice", "cold"), ("lemon", "sour"), ("honey", "sweet"),
    ("silk", "smooth"), ("sandpaper", "rough"),
]

ATTRIBUTE_TARGET_PAIRS = [
    ("strawberry", "red"), ("ocean", "blue"), ("stove", "hot"),
    ("frost", "cold"), ("grapefruit", "sour"), ("candy", "sweet"),
]

FUNCTION_PAIRS = [
    ("knife", "cut"), ("pen", "write"), ("car", "drive"),
    ("phone", "call"), ("key", "unlock"), ("lamp", "illuminate"),
]


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    attn_impl = "flash_attention_2"
    log(f"Loading {model_name} (bf16 + device_map=auto + {attn_impl})...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for impl in [attn_impl, "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=impl,
            )
            log(f"  Loaded with {impl}")
            break
        except Exception as e:
            log(f"  {impl} failed, trying next...")
            continue

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Model: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def extract_all_layer_reps(model, tokenizer, device, sentence, n_layers):
    """Extract representations at ALL layers for a single sentence."""
    layers_list = get_layers(model)
    cache = {}
    captured = {}

    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[li] = output[0].detach().float().cpu()
            else:
                captured[li] = output.detach().float().cpu()
        return hook_fn

    hooks = [layers_list[li].register_forward_hook(make_hook(li)) for li in range(n_layers)]

    try:
        inp = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
        captured.clear()
        with torch.no_grad():
            model(**inp)

        for li in range(n_layers):
            if li in captured:
                cache[li] = captured[li][0, -1].numpy()
    finally:
        for h in hooks:
            h.remove()

    return cache


def inject_at_layer_get_all_logits(model, tokenizer, device, prompt, direction, layer_idx, alpha):
    """Inject direction at a specific layer and get final logits."""
    layers_list = get_layers(model)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
        hidden_modified = hidden.clone()
        hidden_modified[0, -1, :] += (alpha * d_tensor).to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        return hidden_modified

    hook = layers_list[layer_idx].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1].float().cpu().numpy()
    finally:
        hook.remove()

    return logits


def get_baseline_logits(model, tokenizer, device, prompt):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1].float().cpu().numpy()
    return logits


def inject_and_get_layer_outputs(model, tokenizer, device, prompt, direction, inject_layer, alpha, n_layers):
    """Inject direction at inject_layer, collect outputs at all subsequent layers."""
    layers_list = get_layers(model)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    captured = {}

    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[li] = output[0].detach().float().cpu()
            else:
                captured[li] = output.detach().float().cpu()
        return hook_fn

    def inject_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
        hidden_modified = hidden.clone()
        hidden_modified[0, -1, :] += (alpha * d_tensor).to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        return hidden_modified

    hooks = []
    # Register injection hook
    hooks.append(layers_list[inject_layer].register_forward_hook(inject_hook))
    # Register capture hooks for all subsequent layers
    for li in range(inject_layer + 1, n_layers):
        hooks.append(layers_list[li].register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            model(**inp)
    finally:
        for h in hooks:
            h.remove()

    # Extract last token representations
    result = {}
    for li in captured:
        result[li] = captured[li][0, -1].numpy()

    return result


def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase322_{model_name}.log")

    log(f"=== Phase 322: Indirect Readout Path Tracing for {model_name} ===")

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}, class={info.model_class}")

    # Load W_U
    log("Loading W_U...")
    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")

    # Determine optimal injection layers from Phase 321
    if model_name == "glm4":
        opt_attr_layer = 3  # L2-3 is the peak
        opt_func_layer = 16  # L12-16
        opt_neg_layer = 0
    elif model_name == "qwen3":
        opt_attr_layer = 0  # L0
        opt_func_layer = 18  # L18
        opt_neg_layer = 0
    else:
        opt_attr_layer = 6  # L6
        opt_func_layer = 6  # L6
        opt_neg_layer = 0

    results = {}

    # ===================================================================
    # Part A: Direction Transformation Tracing
    # ===================================================================
    log("\n" + "="*60)
    log("Part A: Direction Transformation Tracing (Attribute)")
    log("="*60)

    # For each source pair, extract direction at optimal layer,
    # then inject and trace how it transforms through subsequent layers
    attr_transform_results = []

    for src_idx, (src_noun, src_val) in enumerate(ATTRIBUTE_PAIRS[:6]):
        if src_idx >= len(ATTRIBUTE_TARGET_PAIRS):
            break
        tgt_noun, tgt_val = ATTRIBUTE_TARGET_PAIRS[src_idx]

        # Get source direction at optimal layer
        sent_pos = f"the {src_noun} is {src_val}"
        sent_neg = f"the {src_noun} is just an object"

        cache_pos = extract_all_layer_reps(model, tokenizer, device, sent_pos, info.n_layers)
        cache_neg = extract_all_layer_reps(model, tokenizer, device, sent_neg, info.n_layers)

        if opt_attr_layer not in cache_pos or opt_attr_layer not in cache_neg:
            continue

        d_attr = cache_pos[opt_attr_layer] - cache_neg[opt_attr_layer]
        d_norm = np.linalg.norm(d_attr)
        if d_norm < 1e-10:
            continue
        d_attr_unit = d_attr / d_norm

        # Inject at optimal layer and trace subsequent layers
        target_prompt = f"The {tgt_noun} is"
        subsequent_reps = inject_and_get_layer_outputs(
            model, tokenizer, device, target_prompt,
            d_attr_unit, opt_attr_layer, 2.0, info.n_layers
        )

        # Also get baseline representations (no injection)
        baseline_reps = extract_all_layer_reps(model, tokenizer, device, target_prompt, info.n_layers)

        # Compute readout alignment at each subsequent layer
        tgt_val_ids = tokenizer.encode(tgt_val, add_special_tokens=False)
        src_val_ids = tokenizer.encode(src_val, add_special_tokens=False)

        for li in sorted(subsequent_reps.keys()):
            if li not in baseline_reps:
                continue

            # Delta at this layer = injected rep - baseline rep
            delta_at_li = subsequent_reps[li] - baseline_reps[li]
            delta_norm = np.linalg.norm(delta_at_li)

            # Readout alignment: cos(delta_at_li, W_U[target_val])
            cos_tgt = 0
            if tgt_val_ids:
                w_vec = W_U[tgt_val_ids[0]]
                w_norm = np.linalg.norm(w_vec)
                if w_norm > 0 and delta_norm > 0:
                    cos_tgt = float(np.dot(delta_at_li, w_vec) / (delta_norm * w_norm))

            cos_src = 0
            if src_val_ids:
                w_vec = W_U[src_val_ids[0]]
                w_norm = np.linalg.norm(w_vec)
                if w_norm > 0 and delta_norm > 0:
                    cos_src = float(np.dot(delta_at_li, w_vec) / (delta_norm * w_norm))

            # Angle between original direction and transformed direction
            if delta_norm > 0:
                cos_with_original = float(np.dot(delta_at_li / delta_norm, d_attr_unit))
            else:
                cos_with_original = 0

            attr_transform_results.append({
                "pair": f"{src_noun}→{src_val}→{tgt_noun}→{tgt_val}",
                "inject_layer": opt_attr_layer,
                "trace_layer": li,
                "layers_after": li - opt_attr_layer,
                "delta_norm": round(float(delta_norm), 4),
                "cos_with_target": round(cos_tgt, 4),
                "cos_with_source": round(cos_src, 4),
                "cos_with_original": round(cos_with_original, 4),
            })

        if src_idx == 0:
            log(f"  First pair transformation tracing:")
            for r in attr_transform_results:
                if r["pair"].startswith("apple"):
                    log(f"    L{r['trace_layer']}(+{r['layers_after']}): norm={r['delta_norm']:.4f}, cos_tgt={r['cos_with_target']:.4f}, cos_src={r['cos_with_source']:.4f}, cos_orig={r['cos_with_original']:.4f}")

        del cache_pos, cache_neg, subsequent_reps, baseline_reps
        torch.cuda.empty_cache()

    results["attr_transform"] = attr_transform_results

    # Summary: average readout alignment by layers_after_injection
    log("\n  Attribute direction transformation by distance from injection layer:")
    dist_data = {}
    for r in attr_transform_results:
        dist = r["layers_after"]
        if dist not in dist_data:
            dist_data[dist] = {"norms": [], "cos_tgt": [], "cos_src": [], "cos_orig": []}
        dist_data[dist]["norms"].append(r["delta_norm"])
        dist_data[dist]["cos_tgt"].append(r["cos_with_target"])
        dist_data[dist]["cos_src"].append(r["cos_with_source"])
        dist_data[dist]["cos_orig"].append(r["cos_with_original"])

    for dist in sorted(dist_data.keys()):
        d = dist_data[dist]
        log(f"  +{dist} layers: norm={np.mean(d['norms']):.4f}, cos_tgt={np.mean(d['cos_tgt']):.4f}, cos_src={np.mean(d['cos_src']):.4f}, cos_orig={np.mean(d['cos_orig']):.4f}")

    # ===================================================================
    # Part B: Block Recomputation Test (Attribute)
    # ===================================================================
    log("\n" + "="*60)
    log("Part B: Block Recomputation Test (Attribute)")
    log("="*60)

    # Compare: single-layer injection vs multi-layer block injection
    # Single: inject d at L_opt, let rest compute naturally
    # Block: replace h(clean, L_opt) with h(dirty, L_opt) at L_opt (equivalent to single injection)
    # Key test: inject at L_opt but ALSO replace the next K layers' outputs

    block_results = []

    for src_idx, (src_noun, src_val) in enumerate(ATTRIBUTE_PAIRS[:4]):
        if src_idx >= len(ATTRIBUTE_TARGET_PAIRS):
            break
        tgt_noun, tgt_val = ATTRIBUTE_TARGET_PAIRS[src_idx]

        # Get all representations for source and target
        sent_pos = f"the {src_noun} is {src_val}"
        sent_neg = f"the {src_noun} is just an object"

        cache_pos = extract_all_layer_reps(model, tokenizer, device, sent_pos, info.n_layers)
        cache_neg = extract_all_layer_reps(model, tokenizer, device, sent_neg, info.n_layers)

        target_prompt = f"The {tgt_noun} is"
        baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)

        tgt_val_ids = tokenizer.encode(tgt_val, add_special_tokens=False)
        if not tgt_val_ids:
            continue
        baseline_val_logit = float(baseline_logits[tgt_val_ids[0]])

        # Test injection at different layers around optimal
        for inject_li in range(max(0, opt_attr_layer - 2), min(info.n_layers, opt_attr_layer + 4)):
            if inject_li not in cache_pos or inject_li not in cache_neg:
                continue

            d_attr = cache_pos[inject_li] - cache_neg[inject_li]
            d_norm = np.linalg.norm(d_attr)
            if d_norm < 1e-10:
                continue
            d_attr_unit = d_attr / d_norm

            # Single-layer injection
            inj_logits = inject_at_layer_get_all_logits(
                model, tokenizer, device, target_prompt,
                d_attr_unit, inject_li, 2.0
            )
            single_delta = float(inj_logits[tgt_val_ids[0]] - baseline_val_logit)

            # Also test: injecting the FULL delta (not just the direction)
            # This tests whether the magnitude matters
            inj_logits_full = inject_at_layer_get_all_logits(
                model, tokenizer, device, target_prompt,
                d_attr / d_norm, inject_li, 4.0  # alpha=4 for stronger effect
            )
            full_delta = float(inj_logits_full[tgt_val_ids[0]] - baseline_val_logit)

            block_results.append({
                "pair": f"{src_noun}→{tgt_val}",
                "inject_layer": inject_li,
                "single_delta_alpha2": round(single_delta, 4),
                "single_delta_alpha4": round(full_delta, 4),
            })

        if src_idx == 0:
            log(f"  First pair block test:")
            for r in block_results:
                if r["pair"].startswith("apple"):
                    log(f"    L{r['inject_layer']}: alpha2={r['single_delta_alpha2']:.4f}, alpha4={r['single_delta_alpha4']:.4f}")

        del cache_pos, cache_neg
        torch.cuda.empty_cache()

    results["block_test"] = block_results

    # Summary
    log("\n  Block recomputation test by layer:")
    layer_data = {}
    for r in block_results:
        li = r["inject_layer"]
        if li not in layer_data:
            layer_data[li] = {"a2": [], "a4": []}
        layer_data[li]["a2"].append(r["single_delta_alpha2"])
        layer_data[li]["a4"].append(r["single_delta_alpha4"])

    for li in sorted(layer_data.keys()):
        d = layer_data[li]
        log(f"  L{li}: alpha2_mean={np.mean(d['a2']):.4f}, alpha4_mean={np.mean(d['a4']):.4f}")

    # ===================================================================
    # Part C: Negation Path Transformation
    # ===================================================================
    log("\n" + "="*60)
    log("Part C: Negation Path Transformation")
    log("="*60)

    neg_transform_results = []

    for adj in ["happy", "good", "clean", "safe"]:
        sent_pos = f"very {adj}"
        sent_neg = f"not {adj}"

        cache_pos = extract_all_layer_reps(model, tokenizer, device, sent_pos, info.n_layers)
        cache_neg = extract_all_layer_reps(model, tokenizer, device, sent_neg, info.n_layers)

        # Trace negation direction at L0
        if opt_neg_layer not in cache_pos or opt_neg_layer not in cache_neg:
            continue

        d_neg = cache_neg[opt_neg_layer] - cache_pos[opt_neg_layer]
        d_norm = np.linalg.norm(d_neg)
        if d_norm < 1e-10:
            continue
        d_neg_unit = d_neg / d_norm

        # Inject at L0 and trace
        prompt = f"very {adj}"
        subsequent_reps = inject_and_get_layer_outputs(
            model, tokenizer, device, prompt,
            d_neg_unit, opt_neg_layer, 2.0, info.n_layers
        )
        baseline_reps = extract_all_layer_reps(model, tokenizer, device, prompt, info.n_layers)

        neg_word_ids = {}
        for nw in ["not", "never", "no"]:
            ids = tokenizer.encode(nw, add_special_tokens=False)
            if ids:
                neg_word_ids[nw] = ids[0]

        adj_ids = tokenizer.encode(adj, add_special_tokens=False)
        adj_id = adj_ids[0] if adj_ids else None

        for li in sorted(subsequent_reps.keys()):
            if li not in baseline_reps:
                continue

            delta_at_li = subsequent_reps[li] - baseline_reps[li]
            delta_norm = np.linalg.norm(delta_at_li)

            # Readout with negation words
            max_neg_cos = 0
            for nw, nid in neg_word_ids.items():
                w_vec = W_U[nid]
                w_norm = np.linalg.norm(w_vec)
                if w_norm > 0 and delta_norm > 0:
                    cos_val = float(np.dot(delta_at_li, w_vec) / (delta_norm * w_norm))
                    max_neg_cos = max(max_neg_cos, cos_val)

            adj_cos = 0
            if adj_id:
                w_vec = W_U[adj_id]
                w_norm = np.linalg.norm(w_vec)
                if w_norm > 0 and delta_norm > 0:
                    adj_cos = float(np.dot(delta_at_li, w_vec) / (delta_norm * w_norm))

            neg_transform_results.append({
                "adjective": adj,
                "inject_layer": opt_neg_layer,
                "trace_layer": li,
                "layers_after": li - opt_neg_layer,
                "delta_norm": round(float(delta_norm), 4),
                "max_neg_cos": round(max_neg_cos, 4),
                "adj_cos": round(adj_cos, 4),
            })

        del cache_pos, cache_neg, subsequent_reps, baseline_reps
        torch.cuda.empty_cache()

    results["neg_transform"] = neg_transform_results

    # Summary
    log("\n  Negation direction transformation by distance from L0:")
    dist_data = {}
    for r in neg_transform_results:
        dist = r["layers_after"]
        if dist not in dist_data:
            dist_data[dist] = {"norms": [], "neg_cos": [], "adj_cos": []}
        dist_data[dist]["norms"].append(r["delta_norm"])
        dist_data[dist]["neg_cos"].append(r["max_neg_cos"])
        dist_data[dist]["adj_cos"].append(r["adj_cos"])

    for dist in sorted(dist_data.keys()):
        d = dist_data[dist]
        log(f"  +{dist} layers: norm={np.mean(d['norms']):.4f}, neg_cos={np.mean(d['neg_cos']):.4f}, adj_cos={np.mean(d['adj_cos']):.4f}")

    # ===================================================================
    # Part D: L1 vs L2 Transition Analysis (GLM4 specific)
    # ===================================================================
    log("\n" + "="*60)
    log("Part D: L1→L2 Transition Analysis")
    log("="*60)

    # For GLM4, attribute injection at L1 is negative but L2 is very positive.
    # What happens between L1 and L2?

    transition_results = []

    for src_idx, (src_noun, src_val) in enumerate(ATTRIBUTE_PAIRS[:4]):
        if src_idx >= len(ATTRIBUTE_TARGET_PAIRS):
            break
        tgt_noun, tgt_val = ATTRIBUTE_TARGET_PAIRS[src_idx]

        # Extract representations at ALL layers for source sentences
        sent_pos = f"the {src_noun} is {src_val}"
        sent_neg = f"the {src_noun} is just an object"

        cache_pos = extract_all_layer_reps(model, tokenizer, device, sent_pos, info.n_layers)
        cache_neg = extract_all_layer_reps(model, tokenizer, device, sent_neg, info.n_layers)

        # For each layer, compute direction and its readout with W_U
        for li in range(min(8, info.n_layers)):
            if li not in cache_pos or li not in cache_neg:
                continue

            d = cache_pos[li] - cache_neg[li]
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-10:
                continue
            d_unit = d / d_norm

            # Direct readout
            tgt_ids = tokenizer.encode(tgt_val, add_special_tokens=False)
            src_ids = tokenizer.encode(src_val, add_special_tokens=False)

            tgt_cos = 0
            if tgt_ids:
                w = W_U[tgt_ids[0]]
                wn = np.linalg.norm(w)
                if wn > 0:
                    tgt_cos = float(np.dot(d_unit, w) / wn)

            src_cos = 0
            if src_ids:
                w = W_U[src_ids[0]]
                wn = np.linalg.norm(w)
                if wn > 0:
                    src_cos = float(np.dot(d_unit, w) / wn)

            # Norm of the direction
            transition_results.append({
                "pair": f"{src_noun}→{src_val}",
                "layer": li,
                "direction_norm": round(float(d_norm), 4),
                "cos_with_target": round(tgt_cos, 4),
                "cos_with_source": round(src_cos, 4),
            })

        del cache_pos, cache_neg
        torch.cuda.empty_cache()

    results["transition"] = transition_results

    # Summary
    log("\n  Direction norm and readout by layer:")
    for r in transition_results:
        if r["pair"].startswith("apple"):
            log(f"    {r['pair']} L{r['layer']}: norm={r['direction_norm']:.4f}, cos_tgt={r['cos_with_target']:.4f}, cos_src={r['cos_with_source']:.4f}")

    # Save results
    output = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "opt_attr_layer": opt_attr_layer,
        "opt_func_layer": opt_func_layer,
        "opt_neg_layer": opt_neg_layer,
        "results": results,
    }

    out_path = RESULT_DIR / f"{model_name}_phase322.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # Print overall summary
    log("\n" + "="*60)
    log(f"PHASE 322 SUMMARY - {model_name}")
    log("="*60)

    # Key question: does readout alignment INCREASE as the direction propagates?
    if attr_transform_results:
        # Find layers where cos_with_target > cos at injection layer
        inj_layer_data = [r for r in attr_transform_results if r["layers_after"] == 1]
        far_layer_data = [r for r in attr_transform_results if r["layers_after"] >= 5]

        if inj_layer_data and far_layer_data:
            near_cos = np.mean([r["cos_with_target"] for r in inj_layer_data])
            far_cos = np.mean([r["cos_with_target"] for r in far_layer_data])
            log(f"  Readout alignment at +1 layer: {near_cos:.4f}")
            log(f"  Readout alignment at +5+ layers: {far_cos:.4f}")
            log(f"  Alignment change: {far_cos - near_cos:.4f}")
            if far_cos > near_cos:
                log(f"  → Readout alignment INCREASES through layers (indirect path confirmed)")
            else:
                log(f"  → Readout alignment DECREASES through layers")

    # Cleanup
    del W_U
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released.")

    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        for mn in ["qwen3", "glm4", "deepseek7b"]:
            try:
                run_model(mn)
            except Exception as e:
                log(f"ERROR running {mn}: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(10)
    else:
        run_model(model_name)
