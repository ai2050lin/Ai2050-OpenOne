"""
Phase 361: full_resid Decomposition — Layer-State Contract Test
================================================================

Core question: What in full_resid makes it work when full_mlp doesn't?

6 patch conditions per layer x 2 directions (C2R, R2C):
  1. h_in_patch:         Replace pre-layer residual at last token
                          → entire layer recomputes from modified input
  2. attn_out_patch:     Replace attention output at last token only
                          → MLP computes naturally from (corrupt_h_in + clean_attn_out)
  3. h_after_attn_patch: Replace post-attn residual at last token
                          → LayerNorm + MLP recompute from clean h_after_attn
                          → residual after MLP still uses original h_after_attn
  4. mlp_input_recompute:Replace MLP input (post-LayerNorm) at last token
                          → MLP recomputes naturally from clean normed input
  5. mlp_out_patch:      Replace MLP output at last token (= full_mlp from Phase 359)
  6. full_resid_patch:   Replace entire layer output at last token (= full_resid)

Key comparisons:
  - attn_out vs full_resid: Does attention alone explain the gap?
  - h_in_patch vs full_resid: Was the info already in the residual input?
  - h_after_attn vs full_mlp: Does pre-LayerNorm patch differ from MLP output patch?
  - mlp_input_recompute vs full_mlp: Does MLP natural recomputation help?
  - attn_out vs mlp_out: Which is more important, attention or MLP?

Unified notation:
  C2R: effect = -Δgap / |base_gap| (positive = binding damaged)
  R2C: effect = +Δgap / |base_gap| (positive = binding rescued)
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Model Configs =====
MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560, "d_ff": 9728,
        "test_layers": [23, 27],
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096, "d_ff": 13696,
        "test_layers": [36, 38],
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584, "d_ff": 18944,
        "test_layers": [19, 21],
    },
}

# Full test pairs (42)
TEST_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("moon", "white", "black"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"),
    ("fire", "hot", "cold"), ("desert", "hot", "cold"), ("lava", "hot", "cold"),
    ("ice", "cold", "hot"), ("snow", "cold", "hot"), ("volcano", "hot", "cold"),
    ("furnace", "hot", "cold"), ("glacier", "cold", "hot"),
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
    ("swamp", "wet", "dry"), ("desert", "dry", "wet"),
    ("silk", "smooth", "rough"), ("sandpaper", "rough", "smooth"),
    ("glass", "smooth", "rough"), ("rock", "rough", "smooth"),
    ("velvet", "soft", "hard"), ("diamond", "hard", "soft"),
]

CORRUPTED_BASELINE = "The item"

CONDITIONS = [
    "h_in_patch",          # 1: replace pre-layer residual
    "attn_out_patch",      # 2: replace attention output
    "h_after_attn_patch",  # 3: replace post-attn residual (pre-LayerNorm)
    "mlp_input_recompute", # 4: replace MLP input (post-LayerNorm)
    "mlp_out_patch",       # 5: replace MLP output (= full_mlp)
    "full_resid_patch",    # 6: replace entire layer output (= full_resid)
]


def load_model_bf16(model_name):
    """BF16 + device_map=auto + flash attention"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl)
            log(f"  Loaded {model_name} with attn_impl={impl}")
            break
        except Exception as e:
            log(f"  Failed with {impl}: {str(e)[:80]}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def get_W_U(model, model_name):
    if hasattr(model, "lm_head"):
        w = model.lm_head.weight
        if not w.is_meta:
            return w.detach().cpu().float().numpy()
    import glob
    from safetensors import safe_open
    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            if 'lm_head.weight' in sf.keys():
                return sf.get_tensor('lm_head.weight').float().numpy()
    raise ValueError(f"Cannot load lm_head for {model_name}")


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers")


def find_post_attn_ln(layer):
    """Find post-attention LayerNorm module (different model architectures)."""
    for name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def compute_effect(delta_gap, base_gap, direction):
    abs_base = abs(base_gap)
    if abs_base < 1e-10:
        return 0.0
    if direction == "c2r":
        return -delta_gap / abs_base
    else:
        return delta_gap / abs_base


# ===== Capture All Intermediate Activations =====

def capture_layer_internals(model, tokenizer, device, prompt, target_layers):
    """
    Capture all intermediate activations at target layers:
      h_in, attn_out, h_after_attn, mlp_input, mlp_out, h_out
    All stored as numpy arrays [d_model] for last token position only.
    """
    layers = get_layers(model)
    captured = {}

    def make_fwd_hook_last(key):
        """Forward hook: capture output[0, -1, :] (last token)."""
        def hook(module, input, output):
            val = output[0] if isinstance(output, tuple) else output
            captured[key] = val[0, -1, :].detach().cpu().float().numpy()
        return hook

    def make_pre_hook_last(key):
        """Pre-hook: capture args[0][0, -1, :] (last token of first arg)."""
        def pre_hook(module, args):
            inp = args[0]
            captured[key] = inp[0, -1, :].detach().cpu().float().numpy()
        return pre_hook

    hooks = []
    for li in target_layers:
        layer = layers[li]
        post_attn_ln = find_post_attn_ln(layer)

        # h_in: input to the layer (= pre-layer residual)
        hooks.append(layer.register_forward_pre_hook(
            make_pre_hook_last(f"h_in_{li}")))

        # attn_out: output of self_attn module
        hooks.append(layer.self_attn.register_forward_hook(
            make_fwd_hook_last(f"attn_out_{li}")))

        # h_after_attn: input to post_attention_layernorm (= residual after attn)
        if post_attn_ln is not None:
            hooks.append(post_attn_ln.register_forward_pre_hook(
                make_pre_hook_last(f"h_after_attn_{li}")))
            # mlp_input: output of post_attention_layernorm (= MLP input)
            hooks.append(post_attn_ln.register_forward_hook(
                make_fwd_hook_last(f"mlp_input_{li}")))

        # mlp_out: output of MLP module
        hooks.append(layer.mlp.register_forward_hook(
            make_fwd_hook_last(f"mlp_out_{li}")))

        # h_out: output of the entire layer (= post-layer residual)
        hooks.append(layer.register_forward_hook(
            make_fwd_hook_last(f"h_out_{li}")))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()

    for h in hooks:
        h.remove()

    return captured, logits


# ===== Patch Functions =====

def _make_output_patch_hook(replacement):
    """Generic output patch hook: replace last token of output[0] with replacement."""
    def hook(module, input, output):
        val = output[0] if isinstance(output, tuple) else output
        modified = val.clone()
        rep_t = torch.tensor(replacement, dtype=modified.dtype, device=modified.device)
        modified[0, -1, :] = rep_t
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook


def _make_input_patch_hook(replacement):
    """Generic input patch hook: replace last token of args[0] with replacement."""
    def pre_hook(module, args):
        hidden_states = args[0]
        modified = hidden_states.clone()
        rep_t = torch.tensor(replacement, dtype=modified.dtype, device=modified.device)
        modified[0, -1, :] = rep_t
        return (modified,) + args[1:]
    return pre_hook


def run_patch_condition(model, tokenizer, device, prompt, target_layer, condition, replacements):
    """
    Run a single patch condition and return logits.

    Args:
        model: the model
        tokenizer: the tokenizer
        device: the device
        prompt: the prompt to run
        target_layer: layer index
        condition: one of CONDITIONS
        replacements: dict with keys like 'h_in', 'attn_out', 'h_after_attn',
                      'mlp_input', 'mlp_out', 'h_out' (numpy arrays [d_model])
    """
    layers = get_layers(model)
    layer = layers[target_layer]
    post_attn_ln = find_post_attn_ln(layer)
    hook_handles = []

    if condition == "h_in_patch":
        # Replace pre-layer residual at last token
        hook_handles.append(layer.register_forward_pre_hook(
            _make_input_patch_hook(replacements["h_in"])))

    elif condition == "attn_out_patch":
        # Replace attention output at last token
        hook_handles.append(layer.self_attn.register_forward_hook(
            _make_output_patch_hook(replacements["attn_out"])))

    elif condition == "h_after_attn_patch":
        # Replace post-attention residual (input to post_attn_ln) at last token
        if post_attn_ln is not None:
            hook_handles.append(post_attn_ln.register_forward_pre_hook(
                _make_input_patch_hook(replacements["h_after_attn"])))
        else:
            # Fallback: patch MLP input directly
            hook_handles.append(layer.mlp.register_forward_pre_hook(
                _make_input_patch_hook(replacements["mlp_input"])))

    elif condition == "mlp_input_recompute":
        # Replace MLP input (output of post_attn_ln) at last token
        if post_attn_ln is not None:
            hook_handles.append(post_attn_ln.register_forward_hook(
                _make_output_patch_hook(replacements["mlp_input"])))
        else:
            hook_handles.append(layer.mlp.register_forward_pre_hook(
                _make_input_patch_hook(replacements["mlp_input"])))

    elif condition == "mlp_out_patch":
        # Replace MLP output at last token (= full_mlp)
        hook_handles.append(layer.mlp.register_forward_hook(
            _make_output_patch_hook(replacements["mlp_out"])))

    elif condition == "full_resid_patch":
        # Replace entire layer output at last token (= full_resid)
        hook_handles.append(layer.register_forward_hook(
            _make_output_patch_hook(replacements["h_out"])))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()

    for h in hook_handles:
        h.remove()

    return logits


# ===== Main Experiment =====

def run_experiment(model_name):
    log(f"Phase 361: full_resid Decomposition ({model_name})")
    log("=" * 70)
    t0 = time.time()
    cfg = MODEL_CONFIGS[model_name]
    target_layers = cfg["test_layers"]

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)

    # Verify post_attn_ln exists for all target layers
    layers_obj = get_layers(model)
    for li in target_layers:
        ln = find_post_attn_ln(layers_obj[li])
        if ln is None:
            log(f"  WARNING: L{li} has no post_attention_layernorm! "
                f"h_after_attn_patch and mlp_input_recompute may fall back to MLP input patch.")
        else:
            log(f"  L{li}: post_attn_ln found ({type(ln).__name__})")

    n_test = len(TEST_PAIRS)
    log(f"\n  Main experiment: {n_test} pairs, {len(target_layers)} layers, "
        f"{len(CONDITIONS)} conditions x 2 directions")

    # Results storage
    results = {li: {cond: {"c2r": [], "r2c": []} for cond in CONDITIONS}
               for li in target_layers}

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue

        clean_prompt = f"The {obj}"

        # Capture all intermediate activations
        clean_acts, clean_logits = capture_layer_internals(
            model, tokenizer, device, clean_prompt, target_layers)
        corrupt_acts, corrupt_logits = capture_layer_internals(
            model, tokenizer, device, CORRUPTED_BASELINE, target_layers)

        clean_target = float(clean_logits[tid_t])
        clean_compet = float(clean_logits[tid_c])
        corrupt_target = float(corrupt_logits[tid_t])
        corrupt_compet = float(corrupt_logits[tid_c])
        clean_gap = clean_target - clean_compet
        corrupt_gap = corrupt_target - corrupt_compet
        base_gap = clean_gap - corrupt_gap

        if abs(base_gap) < 1e-10:
            del clean_acts, corrupt_acts
            gc.collect()
            torch.cuda.empty_cache()
            continue

        for li in target_layers:
            # Extract captured activations for this layer
            clean_repl = {
                "h_in": clean_acts.get(f"h_in_{li}"),
                "attn_out": clean_acts.get(f"attn_out_{li}"),
                "h_after_attn": clean_acts.get(f"h_after_attn_{li}"),
                "mlp_input": clean_acts.get(f"mlp_input_{li}"),
                "mlp_out": clean_acts.get(f"mlp_out_{li}"),
                "h_out": clean_acts.get(f"h_out_{li}"),
            }
            corrupt_repl = {
                "h_in": corrupt_acts.get(f"h_in_{li}"),
                "attn_out": corrupt_acts.get(f"attn_out_{li}"),
                "h_after_attn": corrupt_acts.get(f"h_after_attn_{li}"),
                "mlp_input": corrupt_acts.get(f"mlp_input_{li}"),
                "mlp_out": corrupt_acts.get(f"mlp_out_{li}"),
                "h_out": corrupt_acts.get(f"h_out_{li}"),
            }

            # Skip if any critical activation is missing
            if clean_repl["h_in"] is None or corrupt_repl["h_in"] is None:
                log(f"  WARNING: L{li} missing h_in activations, skipping")
                continue

            for cond in CONDITIONS:
                for direction, src_prompt, src_repl, ref_gap, ref_target, ref_compet in [
                    ("c2r", clean_prompt, corrupt_repl,
                     clean_gap, clean_target, clean_compet),
                    ("r2c", CORRUPTED_BASELINE, clean_repl,
                     corrupt_gap, corrupt_target, corrupt_compet)
                ]:
                    # Determine which replacement to use based on condition
                    if cond == "h_in_patch":
                        repl_key = "h_in"
                    elif cond == "attn_out_patch":
                        repl_key = "attn_out"
                    elif cond == "h_after_attn_patch":
                        repl_key = "h_after_attn"
                    elif cond == "mlp_input_recompute":
                        repl_key = "mlp_input"
                    elif cond == "mlp_out_patch":
                        repl_key = "mlp_out"
                    elif cond == "full_resid_patch":
                        repl_key = "h_out"
                    else:
                        continue

                    replacement = src_repl.get(repl_key)
                    if replacement is None:
                        continue

                    patch_logits = run_patch_condition(
                        model, tokenizer, device, src_prompt, li, cond,
                        {repl_key: replacement})
                    pt = float(patch_logits[tid_t])
                    pc = float(patch_logits[tid_c])
                    p_gap = pt - pc
                    delta_gap = p_gap - ref_gap
                    effect = compute_effect(delta_gap, base_gap, direction)
                    results[li][cond][direction].append({
                        "effect": effect,
                        "delta_gap": delta_gap,
                        "delta_t": pt - ref_target,
                        "delta_c": pc - ref_compet,
                    })

                gc.collect()
                torch.cuda.empty_cache()

        del clean_acts, corrupt_acts
        gc.collect()
        torch.cuda.empty_cache()

        if (pidx + 1) % 5 == 0:
            elapsed = time.time() - t0
            gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{pidx+1}/{n_test}] pairs done, elapsed={elapsed:.0f}s, GPU={gpu_gb:.1f}GB")

    # ================================================================
    # Summary & Analysis
    # ================================================================
    log(f"\n  {'='*100}")
    log(f"  Phase 361 Summary: {model_name}")
    log(f"  {'='*100}")

    # --- Per-condition results ---
    log(f"\n  --- Per-Condition Results (unified notation) ---")
    header = f"  {'Layer':>6} {'Dir':>5}"
    for cond in CONDITIONS:
        header += f" {cond:>20}"
    header += f" {'n':>4}"
    log(header)
    log(f"  {'-'*140}")

    summary = {}
    for li in target_layers:
        summary[li] = {}
        for direction in ["c2r", "r2c"]:
            row_str = f"  L{li:>4} {direction:>5}"
            n = 0
            for cond in CONDITIONS:
                vals = results[li][cond][direction]
                n = len(vals)
                if n == 0:
                    summary[li][f"{cond}_{direction}"] = {"n": 0, "mean": 0, "se": 0}
                    row_str += f" {'N/A':>20}"
                    continue
                effects = [v["effect"] for v in vals]
                mean_eff = float(np.mean(effects))
                se_eff = float(np.std(effects) / np.sqrt(n))
                summary[li][f"{cond}_{direction}"] = {
                    "n": n, "mean": mean_eff, "se": se_eff,
                    "mean_delta_gap": float(np.mean([v["delta_gap"] for v in vals])),
                }
                row_str += f" {mean_eff:>+20.4f}"
            row_str += f" {n:>4}"
            log(row_str)

    # --- Key Comparison 1: Decomposition of full_resid ---
    log(f"\n  --- Key Comparison: Decomposition of full_resid (R2C) ---")
    log(f"  {'Layer':>6} {'h_in':>10} {'attn_out':>10} {'h_aft_attn':>12} "
        f"{'mlp_inp_rc':>12} {'mlp_out':>10} {'full_resid':>12} {'attn vs resid':>15}")
    log(f"  {'-'*100}")

    decomp_results = {}
    for li in target_layers:
        hi = summary[li].get("h_in_patch_r2c", {}).get("mean", 0)
        ao = summary[li].get("attn_out_patch_r2c", {}).get("mean", 0)
        ha = summary[li].get("h_after_attn_patch_r2c", {}).get("mean", 0)
        mi = summary[li].get("mlp_input_recompute_r2c", {}).get("mean", 0)
        mo = summary[li].get("mlp_out_patch_r2c", {}).get("mean", 0)
        fr = summary[li].get("full_resid_patch_r2c", {}).get("mean", 0)

        # How much of full_resid is explained by each component
        if abs(fr) > 1e-10:
            attn_frac = ao / fr
            mlp_frac = mo / fr
        else:
            attn_frac = mlp_frac = 0

        decomp_results[li] = {
            "h_in": hi, "attn_out": ao, "h_after_attn": ha,
            "mlp_input_rc": mi, "mlp_out": mo, "full_resid": fr,
            "attn_fraction": attn_frac, "mlp_fraction": mlp_frac,
        }

        log(f"  L{li:>4} {hi:>+10.4f} {ao:>+10.4f} {ha:>+12.4f} "
            f"{mi:>+12.4f} {mo:>+10.4f} {fr:>+12.4f} "
            f"attn={attn_frac:>+.1%} mlp={mlp_frac:>+.1%}")

    # --- Key Comparison 2: attn_out vs mlp_out ---
    log(f"\n  --- Attention vs MLP Contribution (R2C) ---")
    log(f"  {'Layer':>6} {'attn_out':>10} {'mlp_out':>10} {'full_resid':>12} "
        f"{'attn>mlp?':>10} {'Dominant':>10}")
    log(f"  {'-'*70}")

    for li in target_layers:
        ao = summary[li].get("attn_out_patch_r2c", {}).get("mean", 0)
        mo = summary[li].get("mlp_out_patch_r2c", {}).get("mean", 0)
        fr = summary[li].get("full_resid_patch_r2c", {}).get("mean", 0)
        dominant = "attn" if abs(ao) > abs(mo) else "MLP" if abs(mo) > abs(ao) else "equal"
        log(f"  L{li:>4} {ao:>+10.4f} {mo:>+10.4f} {fr:>+12.4f} "
            f"{'YES' if ao > mo else 'NO':>10} {dominant:>10}")

    # --- Key Comparison 3: h_after_attn vs mlp_out (should be similar) ---
    log(f"\n  --- h_after_attn_patch vs mlp_out_patch (R2C) ---")
    log(f"  {'Layer':>6} {'h_aft_attn':>12} {'mlp_out':>10} {'Δ':>10} {'Same?':>8}")
    log(f"  {'-'*55}")

    for li in target_layers:
        ha = summary[li].get("h_after_attn_patch_r2c", {}).get("mean", 0)
        mo = summary[li].get("mlp_out_patch_r2c", {}).get("mean", 0)
        delta = ha - mo
        same = "YES" if abs(delta) < 0.02 else "NO"
        log(f"  L{li:>4} {ha:>+12.4f} {mo:>+10.4f} {delta:>+10.4f} {same:>8}")

    # --- Key Comparison 4: mlp_input_recompute vs mlp_out ---
    log(f"\n  --- mlp_input_recompute vs mlp_out_patch (R2C) ---")
    log(f"  {'Layer':>6} {'mlp_inp_rc':>12} {'mlp_out':>10} {'Δ':>10} {'Natural>patch?':>16}")
    log(f"  {'-'*65}")

    for li in target_layers:
        mi = summary[li].get("mlp_input_recompute_r2c", {}).get("mean", 0)
        mo = summary[li].get("mlp_out_patch_r2c", {}).get("mean", 0)
        delta = mi - mo
        better = "YES" if delta > 0.02 else ("marginal" if delta > 0 else "NO")
        log(f"  L{li:>4} {mi:>+12.4f} {mo:>+10.4f} {delta:>+10.4f} {better:>16}")

    # --- Bootstrap CI for key conditions ---
    log(f"\n  --- Bootstrap 95% CI (1000 resamples, R2C) ---")
    np.random.seed(42)
    n_bootstrap = 1000

    for li in target_layers:
        log(f"  Layer {li}:")
        for cond in ["h_in_patch", "attn_out_patch", "mlp_out_patch", "full_resid_patch"]:
            vals = results[li][cond]["r2c"]
            if len(vals) < 5:
                log(f"    {cond}: too few samples ({len(vals)})")
                continue
            effects = np.array([v["effect"] for v in vals])
            boot_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(effects, size=len(effects), replace=True)
                boot_means.append(float(np.mean(sample)))
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))
            mean_eff = float(np.mean(effects))
            log(f"    {cond:>20}: {mean_eff:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # --- Per-pair consistency: attn_out vs full_resid sign ---
    log(f"\n  --- Per-pair Consistency: attn_out R2C sign vs full_resid R2C sign ---")
    for li in target_layers:
        ao_vals = results[li]["attn_out_patch"]["r2c"]
        fr_vals = results[li]["full_resid_patch"]["r2c"]
        n = min(len(ao_vals), len(fr_vals))
        if n == 0:
            continue
        both_neg = sum(1 for i in range(n) if ao_vals[i]["effect"] < 0 and fr_vals[i]["effect"] < 0)
        ao_neg_fr_pos = sum(1 for i in range(n) if ao_vals[i]["effect"] < 0 and fr_vals[i]["effect"] > 0)
        ao_pos_fr_neg = sum(1 for i in range(n) if ao_vals[i]["effect"] > 0 and fr_vals[i]["effect"] < 0)
        both_pos = sum(1 for i in range(n) if ao_vals[i]["effect"] > 0 and fr_vals[i]["effect"] > 0)
        log(f"  L{li}: both_neg={both_neg}, ao_neg_fr_pos={ao_neg_fr_pos}, "
            f"ao_pos_fr_neg={ao_pos_fr_neg}, both_pos={both_pos} (n={n})")

    # --- C2R direction summary ---
    log(f"\n  --- C2R Direction Summary ---")
    log(f"  {'Layer':>6} {'h_in':>10} {'attn_out':>10} {'h_aft_attn':>12} "
        f"{'mlp_inp_rc':>12} {'mlp_out':>10} {'full_resid':>12}")
    log(f"  {'-'*80}")
    for li in target_layers:
        hi = summary[li].get("h_in_patch_c2r", {}).get("mean", 0)
        ao = summary[li].get("attn_out_patch_c2r", {}).get("mean", 0)
        ha = summary[li].get("h_after_attn_patch_c2r", {}).get("mean", 0)
        mi = summary[li].get("mlp_input_recompute_c2r", {}).get("mean", 0)
        mo = summary[li].get("mlp_out_patch_c2r", {}).get("mean", 0)
        fr = summary[li].get("full_resid_patch_c2r", {}).get("mean", 0)
        log(f"  L{li:>4} {hi:>+10.4f} {ao:>+10.4f} {ha:>+12.4f} "
            f"{mi:>+12.4f} {mo:>+10.4f} {fr:>+12.4f}")

    # ================================================================
    # Save
    # ================================================================
    output = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "361",
        "unified_notation": {
            "C2R_effect": "-Δgap / |base_gap| (positive = binding damaged)",
            "R2C_effect": "+Δgap / |base_gap| (positive = binding rescued)",
            "base_gap": "clean_gap - corrupt_gap",
        },
        "conditions": CONDITIONS,
        "condition_descriptions": {
            "h_in_patch": "Replace pre-layer residual at last token → layer recomputes",
            "attn_out_patch": "Replace attention output at last token → MLP computes from mixed context",
            "h_after_attn_patch": "Replace post-attn residual at last token → LayerNorm+MLP recompute",
            "mlp_input_recompute": "Replace MLP input (post-LayerNorm) at last token → MLP recomputes",
            "mlp_out_patch": "Replace MLP output at last token (= full_mlp)",
            "full_resid_patch": "Replace entire layer output at last token (= full_resid)",
        },
        "test_layers": target_layers,
        "n_pairs": n_test,
        "summary": {str(k): v for k, v in summary.items()},
        "decomposition": {str(k): v for k, v in decomp_results.items()},
        "per_condition_per_pair": {},
    }

    for li in target_layers:
        output["per_condition_per_pair"][str(li)] = {}
        for cond in CONDITIONS:
            output["per_condition_per_pair"][str(li)][cond] = {
                "c2r_effects": [v["effect"] for v in results[li][cond]["c2r"]],
                "r2c_effects": [v["effect"] for v in results[li][cond]["r2c"]],
                "c2r_delta_gap": [v["delta_gap"] for v in results[li][cond]["c2r"]],
                "r2c_delta_gap": [v["delta_gap"] for v in results[li][cond]["r2c"]],
            }

    os.makedirs("results/phase361_resid_decomposition", exist_ok=True)
    out_path = f"results/phase361_resid_decomposition/{model_name}_phase361.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    log(f"\n  Saved to {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Phase 361 complete for {model_name} in {time.time()-t0:.0f}s")
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
