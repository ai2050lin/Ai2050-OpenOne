"""
Phase 322b: Confirmation - Readout Gain Decomposition
======================================================

Phase 322 found that GLM4 has an "indirect readout path" where the direction
becomes more aligned with W_U[target] through subsequent layer transformation.

However, the alignment increase was very small (+0.007), which seems insufficient
to explain the huge causal efficacy (delta=1.6).

This confirmation test:
1. Compares attribute direction vs random direction vs W_U-aligned direction
2. Measures logit change per unit of injected norm
3. Tests whether the gain comes from norm amplification or direction alignment

Usage:
  python tests/glm5/phase322b_readout_gain.py qwen3
  python tests/glm5/phase322b_readout_gain.py glm4
  python tests/glm5/phase322b_readout_gain.py deepseek7b
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

RESULT_DIR = Path("results/phase322b_gain")
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


ATTRIBUTE_PAIRS = [
    ("apple", "red"), ("sky", "blue"), ("fire", "hot"),
    ("ice", "cold"), ("lemon", "sour"), ("honey", "sweet"),
    ("silk", "smooth"), ("sandpaper", "rough"),
]

ATTRIBUTE_TARGET_PAIRS = [
    ("strawberry", "red"), ("ocean", "blue"), ("stove", "hot"),
    ("frost", "cold"), ("grapefruit", "sour"), ("candy", "sweet"),
    ("satin", "smooth"), ("concrete", "rough"),
]


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=impl,
            )
            log(f"  Loaded {model_name} with {impl}")
            break
        except Exception as e:
            continue

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Model: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def extract_rep_at_layer(model, tokenizer, device, sentence, target_layer):
    """Extract representation at a single layer."""
    layers_list = get_layers(model)
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['rep'] = output[0].detach().float().cpu()
        else:
            captured['rep'] = output.detach().float().cpu()

    hook = layers_list[target_layer].register_forward_hook(hook_fn)

    try:
        inp = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            model(**inp)
        return captured['rep'][0, -1].numpy()
    finally:
        hook.remove()


def inject_and_get_logits(model, tokenizer, device, prompt, direction, layer_idx, alpha):
    """Inject direction and get logits."""
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


def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase322b_{model_name}.log")

    log(f"=== Phase 322b: Readout Gain Decomposition for {model_name} ===")

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}")

    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")

    # Determine optimal layers
    if model_name == "glm4":
        opt_attr_layer = 3
    elif model_name == "qwen3":
        opt_attr_layer = 0
    else:
        opt_attr_layer = 6

    results = {}
    alpha = 2.0

    # ===================================================================
    # Test 1: Attribute Direction vs Random Direction vs W_U-aligned Direction
    # ===================================================================
    log("\n" + "="*60)
    log("Test 1: Direction Type Comparison")
    log("="*60)

    comparison_results = []

    for src_idx, (src_noun, src_val) in enumerate(ATTRIBUTE_PAIRS[:6]):
        if src_idx >= len(ATTRIBUTE_TARGET_PAIRS):
            break
        tgt_noun, tgt_val = ATTRIBUTE_TARGET_PAIRS[src_idx]

        # Get attribute direction
        sent_pos = f"the {src_noun} is {src_val}"
        sent_neg = f"the {src_noun} is just an object"

        h_pos = extract_rep_at_layer(model, tokenizer, device, sent_pos, opt_attr_layer)
        h_neg = extract_rep_at_layer(model, tokenizer, device, sent_neg, opt_attr_layer)

        d_attr = h_pos - h_neg
        d_attr_norm = np.linalg.norm(d_attr)
        if d_attr_norm < 1e-10:
            continue
        d_attr_unit = d_attr / d_attr_norm

        # Get W_U-aligned direction (direct readout direction for target word)
        tgt_ids = tokenizer.encode(tgt_val, add_special_tokens=False)
        src_ids = tokenizer.encode(src_val, add_special_tokens=False)
        if not tgt_ids or not src_ids:
            continue

        w_tgt = W_U[tgt_ids[0]]
        w_tgt_norm = np.linalg.norm(w_tgt)
        w_src = W_U[src_ids[0]]
        w_src_norm = np.linalg.norm(w_src)

        d_wu_tgt = w_tgt / w_tgt_norm if w_tgt_norm > 0 else np.zeros_like(w_tgt)
        d_wu_src = w_src / w_src_norm if w_src_norm > 0 else np.zeros_like(w_src)

        # Generate matched random directions (5 random seeds)
        random_deltas = []
        np.random.seed(42 + src_idx)
        for _ in range(5):
            rand_dir = np.random.randn(d_attr.shape[0])
            rand_dir /= np.linalg.norm(rand_dir)
            random_deltas.append(rand_dir)

        # Baseline logits
        target_prompt = f"The {tgt_noun} is"
        baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
        baseline_tgt_logit = float(baseline_logits[tgt_ids[0]])
        baseline_src_logit = float(baseline_logits[src_ids[0]])

        # Test each direction type at optimal layer
        # 1. Attribute direction
        inj_logits = inject_and_get_logits(
            model, tokenizer, device, target_prompt, d_attr_unit, opt_attr_layer, alpha
        )
        attr_delta_tgt = float(inj_logits[tgt_ids[0]] - baseline_tgt_logit)
        attr_delta_src = float(inj_logits[src_ids[0]] - baseline_src_logit)

        # 2. W_U-target direction (direct readout alignment)
        inj_logits = inject_and_get_logits(
            model, tokenizer, device, target_prompt, d_wu_tgt, opt_attr_layer, alpha
        )
        wu_tgt_delta_tgt = float(inj_logits[tgt_ids[0]] - baseline_tgt_logit)
        wu_tgt_delta_src = float(inj_logits[src_ids[0]] - baseline_src_logit)

        # 3. W_U-source direction
        inj_logits = inject_and_get_logits(
            model, tokenizer, device, target_prompt, d_wu_src, opt_attr_layer, alpha
        )
        wu_src_delta_tgt = float(inj_logits[tgt_ids[0]] - baseline_tgt_logit)
        wu_src_delta_src = float(inj_logits[src_ids[0]] - baseline_src_logit)

        # 4. Random directions
        random_deltas_tgt = []
        random_deltas_src = []
        for rand_dir in random_deltas:
            inj_logits = inject_and_get_logits(
                model, tokenizer, device, target_prompt, rand_dir, opt_attr_layer, alpha
            )
            random_deltas_tgt.append(float(inj_logits[tgt_ids[0]] - baseline_tgt_logit))
            random_deltas_src.append(float(inj_logits[src_ids[0]] - baseline_src_logit))

        comparison_results.append({
            "pair": f"{src_noun}→{tgt_val}",
            "attr_delta_tgt": round(attr_delta_tgt, 4),
            "attr_delta_src": round(attr_delta_src, 4),
            "wu_tgt_delta_tgt": round(wu_tgt_delta_tgt, 4),
            "wu_tgt_delta_src": round(wu_tgt_delta_src, 4),
            "wu_src_delta_tgt": round(wu_src_delta_tgt, 4),
            "wu_src_delta_src": round(wu_src_delta_src, 4),
            "random_mean_delta_tgt": round(float(np.mean(random_deltas_tgt)), 4),
            "random_mean_delta_src": round(float(np.mean(random_deltas_src)), 4),
            "cos_attr_wu_tgt": round(float(np.dot(d_attr_unit, d_wu_tgt)), 4),
            "cos_attr_wu_src": round(float(np.dot(d_attr_unit, d_wu_src)), 4),
        })

        if src_idx == 0:
            log(f"  {src_noun}→{tgt_val}:")
            log(f"    attr_dir: delta_tgt={attr_delta_tgt:.4f}, delta_src={attr_delta_src:.4f}")
            log(f"    wu_tgt_dir: delta_tgt={wu_tgt_delta_tgt:.4f}, delta_src={wu_tgt_delta_src:.4f}")
            log(f"    wu_src_dir: delta_tgt={wu_src_delta_tgt:.4f}, delta_src={wu_src_delta_src:.4f}")
            log(f"    random_dir: delta_tgt={np.mean(random_deltas_tgt):.4f}, delta_src={np.mean(random_deltas_src):.4f}")
            log(f"    cos(attr, wu_tgt)={float(np.dot(d_attr_unit, d_wu_tgt)):.4f}")

    results["direction_comparison"] = comparison_results

    # Summary
    log("\n  Direction type comparison (mean across pairs):")
    for key in ["attr_delta_tgt", "wu_tgt_delta_tgt", "wu_src_delta_tgt", "random_mean_delta_tgt"]:
        vals = [r[key] for r in comparison_results]
        log(f"    {key}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

    # Compute gain: how much more effective is attribute direction vs random?
    attr_gains = [r["attr_delta_tgt"] - r["random_mean_delta_tgt"] for r in comparison_results]
    wu_gains = [r["wu_tgt_delta_tgt"] - r["random_mean_delta_tgt"] for r in comparison_results]
    log(f"\n  Gain above random:")
    log(f"    attr_direction: {np.mean(attr_gains):.4f}")
    log(f"    wu_tgt_direction: {np.mean(wu_gains):.4f}")

    # ===================================================================
    # Test 2: Readout Gain at Each Layer (where does the gain come from?)
    # ===================================================================
    log("\n" + "="*60)
    log("Test 2: Readout Gain by Layer (Inject at opt layer, readout at each layer)")
    log("="*60)

    # For GLM4, test attribute direction injection at L2 vs L3 vs L4
    # and W_U-target direction injection at the same layers
    # The key question: at which layer does the attribute direction become
    # more effective than the W_U-target direction?

    layer_comparison = []
    test_layers = list(range(max(0, opt_attr_layer - 2), min(info.n_layers, opt_attr_layer + 6)))

    for src_idx, (src_noun, src_val) in enumerate(ATTRIBUTE_PAIRS[:4]):
        if src_idx >= len(ATTRIBUTE_TARGET_PAIRS):
            break
        tgt_noun, tgt_val = ATTRIBUTE_TARGET_PAIRS[src_idx]

        tgt_ids = tokenizer.encode(tgt_val, add_special_tokens=False)
        if not tgt_ids:
            continue

        w_tgt = W_U[tgt_ids[0]]
        w_tgt_norm = np.linalg.norm(w_tgt)
        d_wu_tgt = w_tgt / w_tgt_norm if w_tgt_norm > 0 else np.zeros_like(w_tgt)

        target_prompt = f"The {tgt_noun} is"
        baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
        baseline_val_logit = float(baseline_logits[tgt_ids[0]])

        for li in test_layers:
            # Attribute direction at this layer
            h_pos = extract_rep_at_layer(model, tokenizer, device, f"the {src_noun} is {src_val}", li)
            h_neg = extract_rep_at_layer(model, tokenizer, device, f"the {src_noun} is just an object", li)
            d_attr = h_pos - h_neg
            d_norm = np.linalg.norm(d_attr)
            if d_norm < 1e-10:
                continue
            d_attr_unit = d_attr / d_norm

            # cos between attribute direction and W_U direction at this layer
            cos_attr_wu = float(np.dot(d_attr_unit, d_wu_tgt))

            # Inject attribute direction at this layer
            inj_logits = inject_and_get_logits(
                model, tokenizer, device, target_prompt, d_attr_unit, li, alpha
            )
            attr_delta = float(inj_logits[tgt_ids[0]] - baseline_val_logit)

            # Inject W_U-target direction at this layer
            inj_logits = inject_and_get_logits(
                model, tokenizer, device, target_prompt, d_wu_tgt, li, alpha
            )
            wu_delta = float(inj_logits[tgt_ids[0]] - baseline_val_logit)

            layer_comparison.append({
                "pair": f"{src_noun}→{tgt_val}",
                "inject_layer": li,
                "attr_delta": round(attr_delta, 4),
                "wu_delta": round(wu_delta, 4),
                "attr_minus_wu": round(attr_delta - wu_delta, 4),
                "cos_attr_wu": round(cos_attr_wu, 4),
            })

        torch.cuda.empty_cache()

    results["layer_comparison"] = layer_comparison

    # Summary by layer
    log("\n  Layer comparison (mean across pairs):")
    layer_data = {}
    for r in layer_comparison:
        li = r["inject_layer"]
        if li not in layer_data:
            layer_data[li] = {"attr": [], "wu": [], "diff": [], "cos": []}
        layer_data[li]["attr"].append(r["attr_delta"])
        layer_data[li]["wu"].append(r["wu_delta"])
        layer_data[li]["diff"].append(r["attr_minus_wu"])
        layer_data[li]["cos"].append(r["cos_attr_wu"])

    for li in sorted(layer_data.keys()):
        d = layer_data[li]
        log(f"  L{li}: attr_delta={np.mean(d['attr']):.4f}, wu_delta={np.mean(d['wu']):.4f}, diff={np.mean(d['diff']):.4f}, cos={np.mean(d['cos']):.4f}")

    # Save
    output = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "opt_attr_layer": opt_attr_layer,
        "results": results,
    }

    out_path = RESULT_DIR / f"{model_name}_phase322b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # Print overall summary
    log("\n" + "="*60)
    log(f"PHASE 322b SUMMARY - {model_name}")
    log("="*60)

    if comparison_results:
        attr_mean = np.mean([r["attr_delta_tgt"] for r in comparison_results])
        wu_mean = np.mean([r["wu_tgt_delta_tgt"] for r in comparison_results])
        rand_mean = np.mean([r["random_mean_delta_tgt"] for r in comparison_results])
        cos_mean = np.mean([r["cos_attr_wu_tgt"] for r in comparison_results])

        log(f"  Attribute direction: delta_tgt={attr_mean:.4f}")
        log(f"  W_U-target direction: delta_tgt={wu_mean:.4f}")
        log(f"  Random direction: delta_tgt={rand_mean:.4f}")
        log(f"  cos(attr_dir, wu_dir)={cos_mean:.4f}")

        if attr_mean > wu_mean:
            log(f"  → Attribute direction MORE effective than W_U direction (despite low direct alignment)")
            log(f"  → This confirms the INDIRECT READOUT PATH")
        else:
            log(f"  → W_U direction more effective (direct readout)")

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
