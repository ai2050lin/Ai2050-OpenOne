"""
Phase 330b: Value Prior Corrected Layer Binding
================================================

Phase 330 found:
  - GLM4: binding forms at L25-L32 (rapid growth)
  - Qwen3: binding forms early (L1)
  - DS7B: binding forms at L4, NI=-2.679

But baseline_binding may be contaminated by value prior:
  "white" has higher prior than "black", inflating binding.

This script adds VALUE PRIOR CORRECTION:
  For each value v, compute prior(v) = logit(v | "The item")
  Then: corrected_binding = binding - (prior(t) - prior(c))

Wait — that's circular. Better approach:
  Use a NEUTRAL prompt like "The" (no object) to compute prior.

Actually, the Phase 329b design already uses "The item" as baseline.
The issue is that value prior still leaks through:
  - "red" vs "blue" prior difference affects baseline

New approach: COMPUTE VALUE PRIOR FROM UNCONDITIONAL MODEL
  prior(v) = logit(v | empty/generic prompt)

Then: corrected_binding = raw_binding - prior_advantage
  where prior_advantage = prior(target) - prior(competitor)

This removes the "red is just more common than blue" confound.

Usage:
  python tests/glm5/phase330b_prior_corrected.py qwen3
  python tests/glm5/phase330b_prior_corrected.py glm4
  python tests/glm5/phase330b_prior_corrected.py deepseek7b
"""
import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
    },
}

TEST_PAIRS = [
    # === COLOR - high_compatible (8) ===
    ("apple", "red", "blue", "color", "high_compatible"),
    ("banana", "yellow", "purple", "color", "high_compatible"),
    ("snow", "white", "black", "color", "high_compatible"),
    ("sky", "blue", "green", "color", "high_compatible"),
    ("cherry", "red", "blue", "color", "high_compatible"),
    ("leaf", "green", "red", "color", "high_compatible"),
    ("orange", "orange", "blue", "color", "high_compatible"),
    ("grass", "green", "red", "color", "high_compatible"),

    # === COLOR - near_incompatible (3) ===
    ("apple", "blue", "black", "color", "near_incompatible"),
    ("snow", "pink", "orange", "color", "near_incompatible"),
    ("banana", "white", "black", "color", "near_incompatible"),

    # === TEXTURE - high_compatible (4) ===
    ("stone", "rough", "soft", "texture", "high_compatible"),
    ("silk", "smooth", "rough", "texture", "high_compatible"),
    ("glass", "smooth", "rough", "texture", "high_compatible"),
    ("sand", "rough", "smooth", "texture", "high_compatible"),

    # === TEMPERATURE - high_compatible (4) ===
    ("ice", "cold", "hot", "temperature", "high_compatible"),
    ("fire", "hot", "cold", "temperature", "high_compatible"),
    ("oven", "hot", "cold", "temperature", "high_compatible"),
    ("snow", "cold", "hot", "temperature", "high_compatible"),

    # === COLOR - abstract_absurd (3) ===
    ("idea", "red", "blue", "color", "abstract_absurd"),
    ("concept", "green", "yellow", "color", "abstract_absurd"),
    ("justice", "blue", "red", "color", "abstract_absurd"),

    # === TEXTURE - abstract_absurd (1) ===
    ("theory", "rough", "smooth", "texture", "abstract_absurd"),

    # === TEMPERATURE - abstract_absurd (1) ===
    ("music", "hot", "cold", "temperature", "abstract_absurd"),
]


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=impl,
            )
            log(f"  Loaded {model_name} with attn_impl={impl}")
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Model: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def get_W_U(model, model_name):
    if hasattr(model, "lm_head"):
        w = model.lm_head.weight
        if not w.is_meta:
            return w.detach().cpu().float().numpy()
    import glob
    from safetensors import safe_open
    model_path = MODEL_CONFIGS[model_name]["path"]
    sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    for sf_file in sf_files:
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            if 'lm_head.weight' in sf.keys():
                w = sf.get_tensor('lm_head.weight')
                log(f"  Loaded lm_head from {os.path.basename(sf_file)}")
                return w.float().numpy()
    raise ValueError(f"Cannot load lm_head for {model_name}")


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return None
    if len(ids) > 1:
        log(f"    WARN: '{word}' tokenized to {len(ids)} tokens, using first")
    return ids[0]


def get_hidden_states(model, tokenizer, device, prompt, n_layers):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs_list = []
    for hs in out.hidden_states:
        hs_list.append(hs[0, -1].float().cpu().numpy())
    return hs_list


def run_all(model_name):
    log(f"Phase 330b: Prior-Corrected Layer Binding — {model_name}")
    log("=" * 60)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]

    log("  Loading W_U...")
    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")

    # ============================================================
    # STEP 1: Compute VALUE PRIORS at each layer
    # Using "The" as neutral prompt — no object, no attribute context
    # ============================================================
    log("\n=== STEP 1: Computing value priors ===")

    neutral_prompt = "The"
    hs_neutral = get_hidden_states(model, tokenizer, device, neutral_prompt, n_layers)

    all_values = sorted(set(p[1] for p in TEST_PAIRS) | set(p[2] for p in TEST_PAIRS))
    value_ids = {}
    for v in all_values:
        tid = get_token_id(tokenizer, v)
        if tid is not None:
            value_ids[v] = tid

    # Compute prior at each layer: prior(v, L) = W_U[tid_v] @ hs_neutral[L]
    value_priors = {}  # {value: [prior_at_L0, prior_at_L1, ...]}
    for v, tid in value_ids.items():
        priors_at_layer = [float(W_U[tid] @ hs_neutral[l]) for l in range(n_layers + 1)]
        value_priors[v] = priors_at_layer

    # Also compute priors using "The item" as alternative baseline
    hs_item = get_hidden_states(model, tokenizer, device, "The item", n_layers)
    value_priors_item = {}
    for v, tid in value_ids.items():
        priors_at_layer = [float(W_U[tid] @ hs_item[l]) for l in range(n_layers + 1)]
        value_priors_item[v] = priors_at_layer

    log(f"  Computed priors for {len(value_priors)} values at {n_layers+1} layers")

    # ============================================================
    # STEP 2: Compute per-layer binding with prior correction
    # ============================================================
    log(f"\n=== STEP 2: Per-layer binding with prior correction ===")

    results = {}
    level_order = ["high_compatible", "near_incompatible", "abstract_absurd"]
    layer_trajectories = {cl: [] for cl in level_order}

    # Collect all pair data first
    all_pair_data = []

    for idx, (obj, target_val, competitor_val, attr_type, compat_level) in enumerate(TEST_PAIRS):
        log(f"  [{idx+1}/{len(TEST_PAIRS)}] {obj}-{target_val} ({compat_level})")

        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)

        if tid_t is None or tid_c is None:
            log(f"    SKIP: token not found")
            continue

        # Get hidden states for object prompt
        hs_obj = get_hidden_states(model, tokenizer, device, f"The {obj}", n_layers)
        hs_itm = get_hidden_states(model, tokenizer, device, "The item", n_layers)

        # Compute binding at each layer
        layer_data = []
        for l in range(n_layers + 1):
            logit_t_obj = float(W_U[tid_t] @ hs_obj[l])
            logit_c_obj = float(W_U[tid_c] @ hs_obj[l])
            logit_t_item = float(W_U[tid_t] @ hs_itm[l])
            logit_c_item = float(W_U[tid_c] @ hs_itm[l])

            # Raw binding (Phase 330 style)
            advantage_obj = logit_t_obj - logit_c_obj
            advantage_item = logit_t_item - logit_c_item
            raw_binding = advantage_obj - advantage_item

            # Prior correction using "The" baseline
            prior_t = value_priors.get(target_val, [0]*(n_layers+1))[l]
            prior_c = value_priors.get(competitor_val, [0]*(n_layers+1))[l]
            prior_advantage = prior_t - prior_c
            corrected_binding_the = raw_binding - prior_advantage

            # Prior correction using "The item" baseline
            prior_t_item = value_priors_item.get(target_val, [0]*(n_layers+1))[l]
            prior_c_item = value_priors_item.get(competitor_val, [0]*(n_layers+1))[l]
            prior_adv_item = prior_t_item - prior_c_item
            corrected_binding_item = raw_binding - prior_adv_item

            layer_data.append({
                "layer": l,
                "raw_binding": round(raw_binding, 4),
                "prior_advantage_the": round(prior_advantage, 4),
                "corrected_binding_the": round(corrected_binding_the, 4),
                "prior_advantage_item": round(prior_adv_item, 4),
                "corrected_binding_item": round(corrected_binding_item, 4),
                "logit_t_obj": round(logit_t_obj, 4),
                "logit_c_obj": round(logit_c_obj, 4),
                "logit_t_item": round(logit_t_item, 4),
                "logit_c_item": round(logit_c_item, 4),
            })

        result = {
            "obj": obj,
            "target_val": target_val,
            "competitor_val": competitor_val,
            "attr_type": attr_type,
            "compat_level": compat_level,
            "layer_data": layer_data,
        }

        key = f"{obj}_{target_val}"
        results[key] = result
        all_pair_data.append(result)

        # Summary
        raw_final = layer_data[-1]["raw_binding"]
        corr_final = layer_data[-1]["corrected_binding_the"]
        corr_item_final = layer_data[-1]["corrected_binding_item"]
        log(f"    raw={raw_final:+.3f}, corr(The)={corr_final:+.3f}, corr(item)={corr_item_final:+.3f}")

        if (idx + 1) % 5 == 0 and torch.cuda.is_available():
            log(f"    GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB, elapsed={time.time()-t0:.0f}s")

    # ============================================================
    # STEP 3: Aggregate by compat_level
    # ============================================================
    log(f"\n=== STEP 3: Aggregation by compat_level ===")

    for cl in level_order:
        pairs_at_layer = defaultdict(list)
        pairs_corr_the = defaultdict(list)
        pairs_corr_item = defaultdict(list)

        for r in all_pair_data:
            if r["compat_level"] == cl:
                for ld in r["layer_data"]:
                    pairs_at_layer[ld["layer"]].append(ld["raw_binding"])
                    pairs_corr_the[ld["layer"]].append(ld["corrected_binding_the"])
                    pairs_corr_item[ld["layer"]].append(ld["corrected_binding_item"])

        trajectory = []
        for l in range(n_layers + 1):
            raw = pairs_at_layer.get(l, [])
            corr_t = pairs_corr_the.get(l, [])
            corr_i = pairs_corr_item.get(l, [])

            trajectory.append({
                "layer": l,
                "raw_mean": round(float(np.mean(raw)), 4) if raw else 0.0,
                "raw_pos_rate": round(float(np.mean([1 if x > 0 else 0 for x in raw])), 3) if raw else 0.0,
                "corrected_the_mean": round(float(np.mean(corr_t)), 4) if corr_t else 0.0,
                "corrected_the_pos_rate": round(float(np.mean([1 if x > 0 else 0 for x in corr_t])), 3) if corr_t else 0.0,
                "corrected_item_mean": round(float(np.mean(corr_i)), 4) if corr_i else 0.0,
                "corrected_item_pos_rate": round(float(np.mean([1 if x > 0 else 0 for x in corr_i])), 3) if corr_i else 0.0,
                "n": len(raw),
            })

        layer_trajectories[cl] = trajectory

        # Summary
        final_raw = trajectory[-1]["raw_mean"]
        final_corr_the = trajectory[-1]["corrected_the_mean"]
        final_corr_item = trajectory[-1]["corrected_item_mean"]
        log(f"  {cl}: raw={final_raw:+.3f}, corr(The)={final_corr_the:+.3f}, corr(item)={final_corr_item:+.3f}")

    # ============================================================
    # STEP 4: Layer-by-layer comparison table
    # ============================================================
    log(f"\n=== STEP 4: Layer-by-layer corrected binding ===")
    sample_layers = [0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 28, 30, 32, 35, n_layers-3, n_layers-2, n_layers-1, n_layers]
    sample_layers = sorted(set(l for l in sample_layers if l <= n_layers))

    header = f"  {'Layer':>6}"
    for cl in level_order:
        header += f"  {'raw':>8}  {'corr':>8}"
    log(header)
    log("  " + "-" * (6 + 18 * len(level_order)))

    for l in sample_layers:
        row = f"  L{l:>5}"
        for cl in level_order:
            if cl in layer_trajectories and l < len(layer_trajectories[cl]):
                t = layer_trajectories[cl][l]
                row += f"  {t['raw_mean']:>+8.3f}  {t['corrected_item_mean']:>+8.3f}"
            else:
                row += f"  {'N/A':>8}  {'N/A':>8}"
        log(row)

    # ============================================================
    # STEP 5: Binding gain analysis (corrected)
    # ============================================================
    log(f"\n=== STEP 5: Corrected binding gain ===")

    for cl in level_order:
        if cl not in layer_trajectories:
            continue
        corr_bindings = [t["corrected_item_mean"] for t in layer_trajectories[cl]]
        gains = [0.0] + [corr_bindings[i] - corr_bindings[i-1] for i in range(1, len(corr_bindings))]
        indexed_gains = [(i, g) for i, g in enumerate(gains)]
        top5 = sorted(indexed_gains, key=lambda x: x[1], reverse=True)[:5]
        log(f"  {cl} - Top 5 corrected binding gain layers:")
        for layer, gain in top5:
            log(f"    L{layer}: {gain:+.4f}")

    # ===== Release model =====
    del model, W_U
    gc.collect()
    torch.cuda.empty_cache()

    # ===== Save results =====
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "approach": "prior_corrected_layer_binding",
        "value_priors_sample": {v: [round(p, 4) for p in priors[:5]] + ["..."] + [round(p, 4) for p in priors[-3:]]
                                for v, priors in list(value_priors.items())[:10]},
        "level_trajectories": layer_trajectories,
        "details": results,
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    all_results = convert(all_results)

    os.makedirs("results/phase330b_prior_corrected", exist_ok=True)
    out_path = f"results/phase330b_prior_corrected/{model_name}_phase330b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")

    # ===== Final summary =====
    log("\n" + "=" * 60)
    log(f"SUMMARY — {model_name}")
    log("=" * 60)
    log(f"  {'Level':<22} {'raw':>8} {'corr(item)':>12} {'corr(The)':>12}")
    for cl in level_order:
        if cl in layer_trajectories:
            t = layer_trajectories[cl][-1]
            log(f"  {cl:<22} {t['raw_mean']:>+8.3f} {t['corrected_item_mean']:>+12.3f} {t['corrected_the_mean']:>+12.3f}")

    # HC vs AA comparison
    if "high_compatible" in layer_trajectories and "abstract_absurd" in layer_trajectories:
        hc_raw = layer_trajectories["high_compatible"][-1]["raw_mean"]
        aa_raw = layer_trajectories["abstract_absurd"][-1]["raw_mean"]
        hc_corr = layer_trajectories["high_compatible"][-1]["corrected_item_mean"]
        aa_corr = layer_trajectories["abstract_absurd"][-1]["corrected_item_mean"]
        hc_corr_the = layer_trajectories["high_compatible"][-1]["corrected_the_mean"]
        aa_corr_the = layer_trajectories["abstract_absurd"][-1]["corrected_the_mean"]

        log(f"\n  HC vs AA:")
        log(f"    Raw:         HC={hc_raw:+.3f} vs AA={aa_raw:+.3f}, HC>AA={hc_raw > aa_raw}")
        log(f"    Corrected(item): HC={hc_corr:+.3f} vs AA={aa_corr:+.3f}, HC>AA={hc_corr > aa_corr}")
        log(f"    Corrected(The):  HC={hc_corr_the:+.3f} vs AA={aa_corr_the:+.3f}, HC>AA={hc_corr_the > aa_corr_the}")

    log(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)

    run_all(model_name)
    log("Phase 330b complete!")
