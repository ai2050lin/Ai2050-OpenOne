"""
Phase 331: Formula Audit + Multi-Prior Baseline Binding
========================================================

CRITICAL FIX: Phase 330b had a double-correction bug.

Phase 330b computed:
  raw_binding = gap(object) - gap(item)
  corrected_binding_item = raw_binding - gap_item
  = gap(object) - 2 * gap(item)   <-- DOUBLE CORRECTION!

This script provides FOUR INDEPENDENT binding definitions, each with
a SINGLE baseline subtraction:

1. gap_obj = logit(target|obj) - logit(competitor|obj)
   Raw object advantage (no baseline subtraction at all)

2. binding_item = gap(obj) - gap(item)
   Object advantage minus "The item" baseline
   (This is the original Phase 330 definition, CORRECT)

3. binding_the = gap(obj) - gap(The)
   Object advantage minus "The" baseline

4. binding_thing = gap(obj) - gap(The thing)
   Object advantage minus "The thing" baseline

5. binding_multi = gap(obj) - mean(gap(item), gap(The), gap(The thing))
   Object advantage minus averaged multi-prior

Each definition subtracts gap(baseline) EXACTLY ONCE.
No double-correction.

This also expands the test pairs significantly to get more robust results.

Usage:
  python tests/glm5/phase331_formula_audit.py qwen3
  python tests/glm5/phase331_formula_audit.py glm4
  python tests/glm5/phase331_formula_audit.py deepseek7b
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

# ===== EXPANDED TEST PAIRS (more data for robust results) =====
# Format: (object, target_value, competitor_value, attr_type, compat_level)
TEST_PAIRS = [
    # === COLOR - high_compatible (12) ===
    ("apple", "red", "blue", "color", "high_compatible"),
    ("banana", "yellow", "purple", "color", "high_compatible"),
    ("snow", "white", "black", "color", "high_compatible"),
    ("sky", "blue", "green", "color", "high_compatible"),
    ("cherry", "red", "blue", "color", "high_compatible"),
    ("leaf", "green", "red", "color", "high_compatible"),
    ("orange", "orange", "blue", "color", "high_compatible"),
    ("grass", "green", "red", "color", "high_compatible"),
    ("tomato", "red", "blue", "color", "high_compatible"),
    ("lemon", "yellow", "purple", "color", "high_compatible"),
    ("carrot", "orange", "blue", "color", "high_compatible"),
    ("coal", "black", "white", "color", "high_compatible"),

    # === COLOR - near_incompatible (6) ===
    ("apple", "blue", "black", "color", "near_incompatible"),
    ("snow", "pink", "orange", "color", "near_incompatible"),
    ("banana", "white", "black", "color", "near_incompatible"),
    ("grass", "yellow", "purple", "color", "near_incompatible"),
    ("sky", "red", "brown", "color", "near_incompatible"),
    ("fire", "blue", "green", "color", "near_incompatible"),

    # === TEXTURE - high_compatible (6) ===
    ("stone", "rough", "soft", "texture", "high_compatible"),
    ("silk", "smooth", "rough", "texture", "high_compatible"),
    ("glass", "smooth", "rough", "texture", "high_compatible"),
    ("sand", "rough", "soft", "texture", "high_compatible"),
    ("velvet", "soft", "hard", "texture", "high_compatible"),
    ("metal", "smooth", "rough", "texture", "high_compatible"),

    # === TEMPERATURE - high_compatible (6) ===
    ("ice", "cold", "hot", "temperature", "high_compatible"),
    ("fire", "hot", "cold", "temperature", "high_compatible"),
    ("oven", "hot", "cold", "temperature", "high_compatible"),
    ("snow", "cold", "hot", "temperature", "high_compatible"),
    ("lava", "hot", "cold", "temperature", "high_compatible"),
    ("fridge", "cold", "hot", "temperature", "high_compatible"),

    # === COLOR - cross_type (4) ===
    ("apple", "sharp", "soft", "texture", "cross_type"),
    ("snow", "sweet", "bitter", "taste", "cross_type"),
    ("stone", "sweet", "sour", "taste", "cross_type"),
    ("fire", "quiet", "loud", "sound", "cross_type"),

    # === COLOR - abstract_absurd (5) ===
    ("idea", "red", "blue", "color", "abstract_absurd"),
    ("concept", "green", "yellow", "color", "abstract_absurd"),
    ("justice", "blue", "red", "color", "abstract_absurd"),
    ("democracy", "hot", "cold", "temperature", "abstract_absurd"),
    ("freedom", "rough", "smooth", "texture", "abstract_absurd"),
]

# Baseline prompts for multi-prior comparison
BASELINE_PROMPTS = [
    "The",           # Most neutral - no object context
    "The item",      # Generic object context
    "The thing",     # Alternative generic
    "It is",         # Different syntactic frame
    "Something",     # Truly generic
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
    log(f"Phase 331: Formula Audit + Multi-Prior Binding — {model_name}")
    log("=" * 60)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]

    log("  Loading W_U...")
    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")

    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ============================================================
    # STEP 0: Compute BASELINE hidden states for ALL prior prompts
    # ============================================================
    log(f"\n=== STEP 0: Computing baseline hidden states ===")

    baseline_hs = {}
    for bp in BASELINE_PROMPTS:
        log(f"  Baseline: '{bp}'")
        baseline_hs[bp] = get_hidden_states(model, tokenizer, device, bp, n_layers)

    log(f"  {len(baseline_hs)} baselines computed")

    # ============================================================
    # STEP 1: Compute per-pair, per-layer binding with ALL definitions
    # ============================================================
    log(f"\n=== STEP 1: Per-pair binding computation (5 definitions) ===")

    all_pair_data = []
    level_order = ["high_compatible", "near_incompatible", "cross_type", "abstract_absurd"]

    for idx, (obj, target_val, competitor_val, attr_type, compat_level) in enumerate(TEST_PAIRS):
        log(f"  [{idx+1}/{len(TEST_PAIRS)}] {obj}-{target_val} ({compat_level})")

        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)

        if tid_t is None or tid_c is None:
            log(f"    SKIP: token not found")
            continue

        # Get hidden states for object prompt
        hs_obj = get_hidden_states(model, tokenizer, device, f"The {obj}", n_layers)

        # Compute all 5 binding definitions at each layer
        layer_data = []
        for l in range(n_layers + 1):
            # Gap for object context
            logit_t_obj = float(W_U[tid_t] @ hs_obj[l])
            logit_c_obj = float(W_U[tid_c] @ hs_obj[l])
            gap_obj = logit_t_obj - logit_c_obj

            # Gap for each baseline
            gaps_baseline = {}
            for bp in BASELINE_PROMPTS:
                logit_t_bp = float(W_U[tid_t] @ baseline_hs[bp][l])
                logit_c_bp = float(W_U[tid_c] @ baseline_hs[bp][l])
                gaps_baseline[bp] = logit_t_bp - logit_c_bp

            # FIVE INDEPENDENT binding definitions (no double-correction!)
            # 1. Raw object advantage (no baseline)
            binding_raw = gap_obj

            # 2. binding_item = gap_obj - gap("The item")
            binding_item = gap_obj - gaps_baseline.get("The item", 0)

            # 3. binding_the = gap_obj - gap("The")
            binding_the = gap_obj - gaps_baseline.get("The", 0)

            # 4. binding_thing = gap_obj - gap("The thing")
            binding_thing = gap_obj - gaps_baseline.get("The thing", 0)

            # 5. binding_multi = gap_obj - mean(all baselines)
            mean_gap = float(np.mean(list(gaps_baseline.values())))
            binding_multi = gap_obj - mean_gap

            layer_data.append({
                "layer": l,
                "gap_obj": round(gap_obj, 4),
                "gap_the": round(gaps_baseline.get("The", 0), 4),
                "gap_item": round(gaps_baseline.get("The item", 0), 4),
                "gap_thing": round(gaps_baseline.get("The thing", 0), 4),
                "binding_raw": round(binding_raw, 4),
                "binding_item": round(binding_item, 4),
                "binding_the": round(binding_the, 4),
                "binding_thing": round(binding_thing, 4),
                "binding_multi": round(binding_multi, 4),
            })

        result = {
            "obj": obj,
            "target_val": target_val,
            "competitor_val": competitor_val,
            "attr_type": attr_type,
            "compat_level": compat_level,
            "layer_data": layer_data,
        }

        all_pair_data.append(result)

        # Summary at final layer
        ld = layer_data[-1]
        log(f"    raw={ld['binding_raw']:+.3f}, item={ld['binding_item']:+.3f}, "
            f"the={ld['binding_the']:+.3f}, thing={ld['binding_thing']:+.3f}, multi={ld['binding_multi']:+.3f}")

        if (idx + 1) % 5 == 0 and torch.cuda.is_available():
            log(f"    GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB, elapsed={time.time()-t0:.0f}s")

    # ============================================================
    # STEP 2: Aggregate by compat_level for each binding definition
    # ============================================================
    log(f"\n=== STEP 2: Aggregation by compat_level ===")

    binding_defs = ["binding_raw", "binding_item", "binding_the", "binding_thing", "binding_multi"]
    level_trajectories = {}

    for cl in level_order:
        # For each binding definition, collect values at each layer
        def_at_layer = {bd: defaultdict(list) for bd in binding_defs}

        for r in all_pair_data:
            if r["compat_level"] == cl:
                for ld in r["layer_data"]:
                    for bd in binding_defs:
                        def_at_layer[bd][ld["layer"]].append(ld[bd])

        trajectory = []
        for l in range(n_layers + 1):
            entry = {"layer": l}
            for bd in binding_defs:
                vals = def_at_layer[bd].get(l, [])
                if vals:
                    entry[bd + "_mean"] = round(float(np.mean(vals)), 4)
                    entry[bd + "_pos_rate"] = round(float(np.mean([1 if v > 0 else 0 for v in vals])), 3)
                    entry[bd + "_std"] = round(float(np.std(vals)), 4)
                else:
                    entry[bd + "_mean"] = 0.0
                    entry[bd + "_pos_rate"] = 0.0
                    entry[bd + "_std"] = 0.0
                entry[bd + "_n"] = len(vals) if vals else 0
            trajectory.append(entry)

        level_trajectories[cl] = trajectory

        # Summary for this compat_level
        final = trajectory[-1]
        log(f"  {cl}:")
        for bd in binding_defs:
            log(f"    {bd}: {final[bd+'_mean']:+.3f} (pos_rate={final[bd+'_pos_rate']:.2f})")

    # ============================================================
    # STEP 3: HC vs AA comparison across ALL binding definitions
    # ============================================================
    log(f"\n=== STEP 3: HC vs AA comparison (5 definitions) ===")

    hc_traj = level_trajectories.get("high_compatible", [])
    aa_traj = level_trajectories.get("abstract_absurd", [])

    if hc_traj and aa_traj:
        log(f"  {'Definition':<18} {'HC_final':>10} {'AA_final':>10} {'HC>AA':>8} {'Gap':>8}")
        log(f"  {'-'*54}")

        for bd in binding_defs:
            hc_final = hc_traj[-1][bd + "_mean"]
            aa_final = aa_traj[-1][bd + "_mean"]
            hc_gt_aa = hc_final > aa_final
            gap = hc_final - aa_final
            log(f"  {bd:<18} {hc_final:>+10.3f} {aa_final:>+10.3f} {str(hc_gt_aa):>8} {gap:>+8.3f}")

    # ============================================================
    # STEP 4: Layer-by-layer table (sampled layers, binding_multi)
    # ============================================================
    log(f"\n=== STEP 4: Layer-by-layer binding_multi (corrected definition) ===")

    sample_layers = [0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 28, 30, 32, 35, n_layers-3, n_layers-2, n_layers-1, n_layers]
    sample_layers = sorted(set(l for l in sample_layers if l <= n_layers))

    header = f"  {'Layer':>6}  {'RelDepth':>8}"
    for cl in level_order:
        header += f"  {cl:>14}"
    log(header)
    log("  " + "-" * (16 + 16 * len(level_order)))

    for l in sample_layers:
        rel_depth = f"{l/n_layers:.2f}"
        row = f"  L{l:>5}  {rel_depth:>8}"
        for cl in level_order:
            if cl in level_trajectories and l < len(level_trajectories[cl]):
                t = level_trajectories[cl][l]
                row += f"  {t['binding_multi_mean']:>+14.3f}"
            else:
                row += f"  {'N/A':>14}"
        log(row)

    # ============================================================
    # STEP 5: Critical comparison - binding_item vs Phase 330b's corrected
    # ============================================================
    log(f"\n=== STEP 5: Formula audit — Phase 330 vs 330b vs 331 ===")

    if "high_compatible" in level_trajectories:
        hc = level_trajectories["high_compatible"][-1]
        log(f"  Phase 331 binding_item (correct): {hc['binding_item_mean']:+.3f}")
        log(f"  Phase 331 binding_the:            {hc['binding_the_mean']:+.3f}")
        log(f"  Phase 331 binding_multi:           {hc['binding_multi_mean']:+.3f}")
        log(f"  Phase 331 binding_raw:             {hc['binding_raw_mean']:+.3f}")
        log(f"")
        log(f"  NOTE: Phase 330b's 'corrected_binding_item' = gap_obj - 2*gap_item")
        log(f"        Phase 331's 'binding_item' = gap_obj - gap_item (correct!)")
        log(f"        If gap_item > 0, Phase 330b OVER-corrected (binding too high)")
        log(f"        If gap_item < 0, Phase 330b UNDER-corrected (binding too low)")

        # Show gap_item at final layer for reference
        if "abstract_absurd" in level_trajectories:
            aa = level_trajectories["abstract_absurd"][-1]
            log(f"")
            log(f"  For AA pairs:")
            log(f"    binding_raw (gap_obj):   {aa['binding_raw_mean']:+.3f}")
            log(f"    binding_item:            {aa['binding_item_mean']:+.3f}")
            log(f"    binding_the:             {aa['binding_the_mean']:+.3f}")
            log(f"    binding_multi:           {aa['binding_multi_mean']:+.3f}")

    # ============================================================
    # STEP 6: Binding gain analysis (which layers contribute most)
    # ============================================================
    log(f"\n=== STEP 6: Binding gain analysis (binding_multi) ===")

    for cl in level_order:
        if cl not in level_trajectories:
            continue
        bindings = [t["binding_multi_mean"] for t in level_trajectories[cl]]
        gains = [0.0] + [bindings[i] - bindings[i-1] for i in range(1, len(bindings))]
        indexed_gains = [(i, g) for i, g in enumerate(gains)]
        top5 = sorted(indexed_gains, key=lambda x: x[1], reverse=True)[:5]
        log(f"  {cl} - Top 5 binding_multi gain layers:")
        for layer, gain in top5:
            rel = layer / n_layers
            log(f"    L{layer} (rel={rel:.2f}): {gain:+.4f}")

    # ============================================================
    # STEP 7: First HC > AA layer across all definitions
    # ============================================================
    log(f"\n=== STEP 7: First HC > AA layer across definitions ===")

    for bd in binding_defs:
        if "high_compatible" in level_trajectories and "abstract_absurd" in level_trajectories:
            hc_vals = [t[bd + "_mean"] for t in level_trajectories["high_compatible"]]
            aa_vals = [t[bd + "_mean"] for t in level_trajectories["abstract_absurd"]]
            first_sep = None
            for i in range(len(hc_vals)):
                if hc_vals[i] > aa_vals[i] and hc_vals[i] > 0.1:
                    first_sep = i
                    break
            if first_sep is not None:
                log(f"  {bd}: HC>AA at L{first_sep} (rel={first_sep/n_layers:.2f}), "
                    f"HC={hc_vals[first_sep]:+.3f}, AA={aa_vals[first_sep]:+.3f}")
            else:
                log(f"  {bd}: HC never exceeds AA with threshold 0.1")

    # ============================================================
    # STEP 8: Per-value analysis (check for tokenization/prior anomalies)
    # ============================================================
    log(f"\n=== STEP 8: Per-value baseline gap at final layer ===")

    # Collect all unique values
    all_values = sorted(set(p[1] for p in TEST_PAIRS) | set(p[2] for p in TEST_PAIRS))
    value_gap_info = {}

    for v in all_values:
        tid = get_token_id(tokenizer, v)
        if tid is None:
            continue
        # Gap at final layer for each baseline
        gaps = {}
        for bp in BASELINE_PROMPTS:
            logit_v = float(W_U[tid] @ baseline_hs[bp][-1])
            gaps[bp] = logit_v
        value_gap_info[v] = gaps

    # Sort by "The" baseline logit to see which values have highest prior
    sorted_by_the = sorted(value_gap_info.items(), key=lambda x: x[1]["The"], reverse=True)
    log(f"  {'Value':<12} {'The':>10} {'The item':>10} {'The thing':>10} {'It is':>10} {'Something':>10}")
    for v, gaps in sorted_by_the:
        log(f"  {v:<12} {gaps.get('The',0):>+10.3f} {gaps.get('The item',0):>+10.3f} "
            f"{gaps.get('The thing',0):>+10.3f} {gaps.get('It is',0):>+10.3f} {gaps.get('Something',0):>+10.3f}")

    # ===== Release model =====
    del model, W_U
    gc.collect()
    torch.cuda.empty_cache()

    # ===== Save results =====
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "approach": "formula_audit_multi_prior",
        "baseline_prompts": BASELINE_PROMPTS,
        "level_trajectories": level_trajectories,
        "value_gap_info": value_gap_info,
        "details": all_pair_data,
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

    os.makedirs("results/phase331_formula_audit", exist_ok=True)
    out_path = f"results/phase331_formula_audit/{model_name}_phase331.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")

    # ===== Final summary =====
    log("\n" + "=" * 60)
    log(f"SUMMARY — {model_name}")
    log("=" * 60)

    log(f"\n  HC vs AA across binding definitions:")
    if "high_compatible" in level_trajectories and "abstract_absurd" in level_trajectories:
        hc = level_trajectories["high_compatible"][-1]
        aa = level_trajectories["abstract_absurd"][-1]
        for bd in binding_defs:
            hc_v = hc[bd + "_mean"]
            aa_v = aa[bd + "_mean"]
            log(f"    {bd:<18} HC={hc_v:+.3f}  AA={aa_v:+.3f}  HC>AA={hc_v > aa_v}")

    log(f"\n  NI across binding definitions:")
    if "near_incompatible" in level_trajectories:
        ni = level_trajectories["near_incompatible"][-1]
        for bd in binding_defs:
            log(f"    {bd:<18} NI={ni[bd+'_mean']:+.3f}")

    # Cross-type
    if "cross_type" in level_trajectories:
        ct = level_trajectories["cross_type"][-1]
        log(f"\n  Cross-type across binding definitions:")
        for bd in binding_defs:
            log(f"    {bd:<18} CT={ct[bd+'_mean']:+.3f}")

    log(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)

    run_all(model_name)
    log("Phase 331 complete!")
