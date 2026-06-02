"""
Phase 329: Three-Way Interaction (I×S×V) Binding Test
=====================================================

Goal: From slot-value synergy (S×V) to object-conditioned binding (I×V, I×S×V).

Core question: Does object identity (I) interact with value direction (V)?
  - If I×V > 0 for high_compatible pairs, but ≤ 0 for absurd pairs → binding exists
  - If I×V ≈ 0 for all pairs → no binding, only S×V synergy

Factorial design (2^3 = 8 conditions):
  baseline, I, S, V, I+S, I+V, S+V, I+S+V

Interactions:
  I×V = Effect(I+V) - Effect(I) - Effect(V)         [KEY BINDING TERM]
  S×V = Effect(S+V) - Effect(S) - Effect(V)         [REPRODUCE 328b]
  I×S×V = Effect(I+S+V) - Effect(I+S) - Effect(I+V) - Effect(S+V)
          + Effect(I) + Effect(S) + Effect(V)        [THREE-WAY]

Negative examples stratified:
  high_compatible: apple-red, banana-yellow
  near_incompatible: apple-blue, snow-pink
  cross_type: apple-sharp, snow-sweet
  abstract_absurd: idea-red, concept-green

Direction computation (object-agnostic for S and V):
  I (object): "I see the {obj}" vs "I see the item"
  S (slot):   "It has a property" vs "It is an object"
  V (value):  "It is {val}" vs "It is an object"

Injection prompt: "The"
Target layer: opt_layer per model
Alpha: 1.0

Usage:
  python tests/glm5/phase329_three_way_binding.py qwen3
  python tests/glm5/phase329_three_way_binding.py glm4
  python tests/glm5/phase329_three_way_binding.py deepseek7b
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
        "n_layers": 36, "d_model": 2560, "opt_layer": 0,
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096, "opt_layer": 3,
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584, "opt_layer": 6,
    },
}

# ===== TEST DATA =====
# (object, target_value, competitor_value, attr_type, compat_level)
TEST_PAIRS = [
    # === COLOR - high_compatible (6) ===
    ("apple", "red", "blue", "color", "high_compatible"),
    ("banana", "yellow", "purple", "color", "high_compatible"),
    ("snow", "white", "black", "color", "high_compatible"),
    ("sky", "blue", "green", "color", "high_compatible"),
    ("cherry", "red", "blue", "color", "high_compatible"),
    ("leaf", "green", "red", "color", "high_compatible"),

    # === COLOR - near_incompatible (2) ===
    ("apple", "blue", "black", "color", "near_incompatible"),
    ("snow", "pink", "orange", "color", "near_incompatible"),

    # === COLOR - cross_type (2) ===
    ("apple", "sharp", "cold", "texture", "cross_type"),
    ("snow", "sweet", "loud", "taste", "cross_type"),

    # === COLOR - abstract_absurd (3) ===
    ("idea", "red", "blue", "color", "abstract_absurd"),
    ("concept", "green", "yellow", "color", "abstract_absurd"),
    ("justice", "blue", "red", "color", "abstract_absurd"),

    # === TEXTURE - high_compatible (3) ===
    ("stone", "rough", "soft", "texture", "high_compatible"),
    ("silk", "smooth", "rough", "texture", "high_compatible"),
    ("glass", "smooth", "rough", "texture", "high_compatible"),

    # === TEXTURE - near_incompatible (1) ===
    ("stone", "soft", "fluffy", "texture", "near_incompatible"),

    # === TEXTURE - abstract_absurd (1) ===
    ("theory", "rough", "smooth", "texture", "abstract_absurd"),

    # === TEMPERATURE - high_compatible (3) ===
    ("ice", "cold", "hot", "temperature", "high_compatible"),
    ("fire", "hot", "cold", "temperature", "high_compatible"),
    ("oven", "hot", "cold", "temperature", "high_compatible"),

    # === TEMPERATURE - near_incompatible (1) ===
    ("ice", "warm", "burning", "temperature", "near_incompatible"),

    # === TEMPERATURE - abstract_absurd (1) ===
    ("music", "hot", "cold", "temperature", "abstract_absurd"),
]

# ===== DIRECTION TEMPLATES =====
OBJ_BASE = "I see the item"
OBJ_TESTS = ["I see the {obj}", "The {obj} is here"]
SLOT_BASE = "It is an object"
SLOT_TESTS = ["It has a property", "It has some feature"]
VALUE_BASE = "It is an object"
VALUE_TESTS = ["It is {val}", "Something is {val}"]

INJECTION_PROMPT = "The"


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


def get_layers(model):
    if hasattr(model, 'model'):
        inner = model.model
    else:
        inner = model
    if hasattr(inner, 'layers'):
        return list(inner.layers)
    elif hasattr(inner, 'encoder') and hasattr(inner.encoder, 'layer'):
        return list(inner.encoder.layer)
    return []


def extract_rep(model, tokenizer, device, sentence, target_layer):
    layers_list = get_layers(model)
    captured = {}
    def hook_fn(module, input, output):
        captured['rep'] = (output[0] if isinstance(output, tuple) else output).detach().float().cpu()
    hook = layers_list[target_layer].register_forward_hook(hook_fn)
    inp = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
    try:
        with torch.no_grad():
            model(**inp)
        return captured['rep'][0, -1].numpy()
    finally:
        hook.remove()


def compute_direction(model, tokenizer, device, base_sent, test_sents, opt_layer):
    """Compute average direction from base to multiple test sentences."""
    h_b = extract_rep(model, tokenizer, device, base_sent, opt_layer)
    dirs = []
    for test in test_sents:
        h_t = extract_rep(model, tokenizer, device, test, opt_layer)
        d = h_t - h_b
        d = d / (np.linalg.norm(d) + 1e-8)
        dirs.append(d)
    avg_dir = np.mean(dirs, axis=0)
    avg_dir = avg_dir / (np.linalg.norm(avg_dir) + 1e-8)
    return avg_dir


def get_baseline_logits(model, tokenizer, device, prompt):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1].float().cpu().numpy()


def inject_multi_and_get_logits(model, tokenizer, device, prompt, directions_alphas, layer_idx):
    """Inject multiple directions and return logits."""
    layers_list = get_layers(model)
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_modified = hidden.clone()
        for direction, alpha in directions_alphas:
            d_tensor = torch.tensor(alpha * direction, dtype=hidden.dtype, device=hidden.device)
            hidden_modified[0, -1, :] += d_tensor
        return (hidden_modified,) + output[1:] if isinstance(output, tuple) else hidden_modified
    hook = layers_list[layer_idx].register_forward_hook(hook_fn)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    try:
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1].float().cpu().numpy()
    finally:
        hook.remove()
    return logits


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return None
    if len(ids) > 1:
        log(f"    WARN: '{word}' tokenized to {len(ids)} tokens, using first")
    return ids[0]


def run_all(model_name):
    log(f"Phase 329: Three-Way Interaction (I×S×V) — {model_name}")
    log("=" * 60)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    opt_layer = MODEL_CONFIGS[model_name]["opt_layer"]
    log(f"  opt_layer={opt_layer}")

    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ============================================================
    # STEP 1: Compute directions (with caching)
    # ============================================================
    log("\n=== STEP 1: Computing directions ===")

    obj_dirs = {}
    value_dirs = {}

    # Object directions (one per unique object)
    unique_objects = sorted(set(p[0] for p in TEST_PAIRS))
    log(f"  Computing {len(unique_objects)} object directions...")
    for i, obj in enumerate(unique_objects):
        base = OBJ_BASE
        tests = [t.format(obj=obj) for t in OBJ_TESTS]
        obj_dirs[obj] = compute_direction(model, tokenizer, device, base, tests, opt_layer)
        if (i + 1) % 5 == 0:
            log(f"    {i+1}/{len(unique_objects)} obj directions done, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    log(f"  Object directions: {len(obj_dirs)}")

    # Value directions (one per unique value)
    all_values = sorted(set(p[1] for p in TEST_PAIRS) | set(p[2] for p in TEST_PAIRS))
    log(f"  Computing {len(all_values)} value directions...")
    for i, val in enumerate(all_values):
        base = VALUE_BASE
        tests = [t.format(val=val) for t in VALUE_TESTS]
        value_dirs[val] = compute_direction(model, tokenizer, device, base, tests, opt_layer)
        if (i + 1) % 5 == 0:
            log(f"    {i+1}/{len(all_values)} value directions done, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    log(f"  Value directions: {len(value_dirs)}")

    # Slot direction (single, shared)
    log(f"  Computing slot direction...")
    slot_dir = compute_direction(model, tokenizer, device, SLOT_BASE, SLOT_TESTS, opt_layer)
    log(f"  Slot direction: norm={np.linalg.norm(slot_dir):.4f}")

    # ============================================================
    # STEP 2: Value priors (baseline logits on neutral prompt)
    # ============================================================
    log("\n=== STEP 2: Computing value priors ===")

    baseline_logits = get_baseline_logits(model, tokenizer, device, INJECTION_PROMPT)
    value_priors = {}
    for val in all_values:
        tid = get_token_id(tokenizer, val)
        if tid is not None:
            value_priors[val] = float(baseline_logits[tid])
        else:
            value_priors[val] = None
            log(f"  WARN: token '{val}' not found!")

    log(f"  Value priors computed for {sum(1 for v in value_priors.values() if v is not None)}/{len(all_values)} values")

    # ============================================================
    # STEP 3: Factorial injection experiment
    # ============================================================
    log(f"\n=== STEP 3: Factorial injection (8 conditions × {len(TEST_PAIRS)} pairs) ===")

    alpha = 1.0
    results = {}

    for idx, (obj, target_val, competitor_val, attr_type, compat_level) in enumerate(TEST_PAIRS):
        log(f"  [{idx+1}/{len(TEST_PAIRS)}] {obj}-{target_val} ({compat_level})")

        # Get token IDs
        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)

        if tid_t is None or tid_c is None:
            log(f"    SKIP: token not found")
            continue

        # Get directions
        I = obj_dirs[obj]
        S = slot_dir
        V = value_dirs[target_val]

        # 8 conditions: baseline, I, S, V, IS, IV, SV, ISV
        conditions = {
            "baseline": [],
            "I": [(I, alpha)],
            "S": [(S, alpha)],
            "V": [(V, alpha)],
            "IS": [(I, alpha), (S, alpha)],
            "IV": [(I, alpha), (V, alpha)],
            "SV": [(S, alpha), (V, alpha)],
            "ISV": [(I, alpha), (S, alpha), (V, alpha)],
        }

        # Run each condition
        cond_logits = {}
        for cond_name, dirs_alphas in conditions.items():
            if not dirs_alphas:
                logits = baseline_logits.copy()
            else:
                logits = inject_multi_and_get_logits(
                    model, tokenizer, device, INJECTION_PROMPT, dirs_alphas, opt_layer
                )
            cond_logits[cond_name] = logits

        # Extract effects (logit change from baseline)
        effects = {}
        for cond_name, logits in cond_logits.items():
            eff_t = float(logits[tid_t]) - float(baseline_logits[tid_t])
            eff_c = float(logits[tid_c]) - float(baseline_logits[tid_c])
            effects[cond_name] = {"target": round(eff_t, 4), "competitor": round(eff_c, 4)}

        # ============================================================
        # Compute interactions
        # ============================================================

        # I×V on target
        iv_target = effects["IV"]["target"] - effects["I"]["target"] - effects["V"]["target"]
        iv_compet = effects["IV"]["competitor"] - effects["I"]["competitor"] - effects["V"]["competitor"]
        iv_binding = iv_target - iv_compet

        # S×V on target
        sv_target = effects["SV"]["target"] - effects["S"]["target"] - effects["V"]["target"]
        sv_compet = effects["SV"]["competitor"] - effects["S"]["competitor"] - effects["V"]["competitor"]
        sv_binding = sv_target - sv_compet

        # I×S on target
        is_target = effects["IS"]["target"] - effects["I"]["target"] - effects["S"]["target"]
        is_compet = effects["IS"]["competitor"] - effects["I"]["competitor"] - effects["S"]["competitor"]

        # I×S×V (three-way interaction)
        # = Effect(ISV) - Effect(IS) - Effect(IV) - Effect(SV)
        #   + Effect(I) + Effect(S) + Effect(V)
        isv_target = (effects["ISV"]["target"] - effects["IS"]["target"] - effects["IV"]["target"]
                      - effects["SV"]["target"] + effects["I"]["target"] + effects["S"]["target"]
                      + effects["V"]["target"])
        isv_compet = (effects["ISV"]["competitor"] - effects["IS"]["competitor"] - effects["IV"]["competitor"]
                      - effects["SV"]["competitor"] + effects["I"]["competitor"] + effects["S"]["competitor"]
                      + effects["V"]["competitor"])
        isv_binding = isv_target - isv_compet

        # Value-prior-corrected I×V
        # adjusted_iv = I×V / (prior_target - prior_compet) if prior_diff > 0
        prior_t = value_priors.get(target_val, 0) or 0
        prior_c = value_priors.get(competitor_val, 0) or 0
        prior_diff = prior_t - prior_c
        if abs(prior_diff) > 0.1:
            iv_prior_corrected = iv_binding / (prior_diff / abs(prior_diff))
        else:
            iv_prior_corrected = iv_binding

        result = {
            "obj": obj,
            "target_val": target_val,
            "competitor_val": competitor_val,
            "attr_type": attr_type,
            "compat_level": compat_level,
            "effects": effects,
            "IxV_target": round(iv_target, 4),
            "IxV_compet": round(iv_compet, 4),
            "IxV_binding": round(iv_binding, 4),
            "SxV_target": round(sv_target, 4),
            "SxV_compet": round(sv_compet, 4),
            "SxV_binding": round(sv_binding, 4),
            "IxS_target": round(is_target, 4),
            "IxS_compet": round(is_compet, 4),
            "IxSxV_target": round(isv_target, 4),
            "IxSxV_compet": round(isv_compet, 4),
            "IxSxV_binding": round(isv_binding, 4),
            "prior_target": round(prior_t, 4),
            "prior_compet": round(prior_c, 4),
            "prior_diff": round(prior_diff, 4),
        }

        key = f"{obj}_{target_val}"
        results[key] = result

        log(f"    IxV={iv_binding:+.3f}, SxV={sv_binding:+.3f}, IxSxV={isv_binding:+.3f}")

        # Periodic GPU check
        if (idx + 1) % 5 == 0 and torch.cuda.is_available():
            log(f"    GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB, elapsed={time.time()-t0:.0f}s")

    # ============================================================
    # STEP 4: Aggregate by compat_level
    # ============================================================
    log("\n=== STEP 4: Aggregation by compat_level ===")

    level_order = ["high_compatible", "near_incompatible", "cross_type", "abstract_absurd"]
    level_stats = defaultdict(lambda: {
        "IxV_binding": [], "SxV_binding": [], "IxSxV_binding": [],
        "IxV_target": [], "IxSxV_target": [],
    })

    for key, r in results.items():
        cl = r["compat_level"]
        level_stats[cl]["IxV_binding"].append(r["IxV_binding"])
        level_stats[cl]["SxV_binding"].append(r["SxV_binding"])
        level_stats[cl]["IxSxV_binding"].append(r["IxSxV_binding"])
        level_stats[cl]["IxV_target"].append(r["IxV_target"])
        level_stats[cl]["IxSxV_target"].append(r["IxSxV_target"])

    level_summaries = {}
    for cl in level_order:
        if cl not in level_stats:
            continue
        stats = level_stats[cl]
        n = len(stats["IxV_binding"])
        level_summaries[cl] = {
            "n": n,
            "IxV_binding_mean": round(float(np.mean(stats["IxV_binding"])), 4),
            "IxV_binding_pos_rate": round(float(np.mean([1 if x > 0 else 0 for x in stats["IxV_binding"]])), 3),
            "SxV_binding_mean": round(float(np.mean(stats["SxV_binding"])), 4),
            "SxV_binding_pos_rate": round(float(np.mean([1 if x > 0 else 0 for x in stats["SxV_binding"]])), 3),
            "IxSxV_binding_mean": round(float(np.mean(stats["IxSxV_binding"])), 4),
            "IxSxV_binding_pos_rate": round(float(np.mean([1 if x > 0 else 0 for x in stats["IxSxV_binding"]])), 3),
            "IxV_target_mean": round(float(np.mean(stats["IxV_target"])), 4),
            "IxSxV_target_mean": round(float(np.mean(stats["IxSxV_target"])), 4),
        }
        s = level_summaries[cl]
        log(f"  {cl} (n={n}): IxV={s['IxV_binding_mean']:+.3f}({s['IxV_binding_pos_rate']:.2f}), "
            f"SxV={s['SxV_binding_mean']:+.3f}({s['SxV_binding_pos_rate']:.2f}), "
            f"IxSxV={s['IxSxV_binding_mean']:+.3f}({s['IxSxV_binding_pos_rate']:.2f})")

    # Check monotonicity: IxV should decrease from high_compatible to abstract_absurd
    if "high_compatible" in level_summaries and "abstract_absurd" in level_summaries:
        hc = level_summaries["high_compatible"]["IxV_binding_mean"]
        aa = level_summaries["abstract_absurd"]["IxV_binding_mean"]
        log(f"\n  IxV monotonicity: high_compatible={hc:+.3f} vs abstract_absurd={aa:+.3f}, "
            f"HC>AA={hc > aa}")

    # ============================================================
    # STEP 5: High_compatible by attr_type
    # ============================================================
    log("\n=== STEP 5: High_compatible by attr_type ===")

    attr_stats = defaultdict(lambda: {"IxV_binding": [], "SxV_binding": [], "IxSxV_binding": []})
    for key, r in results.items():
        if r["compat_level"] == "high_compatible":
            at = r["attr_type"]
            attr_stats[at]["IxV_binding"].append(r["IxV_binding"])
            attr_stats[at]["SxV_binding"].append(r["SxV_binding"])
            attr_stats[at]["IxSxV_binding"].append(r["IxSxV_binding"])

    attr_summaries = {}
    for at in ["color", "texture", "temperature"]:
        if at not in attr_stats:
            continue
        stats = attr_stats[at]
        n = len(stats["IxV_binding"])
        attr_summaries[at] = {
            "n": n,
            "IxV_binding_mean": round(float(np.mean(stats["IxV_binding"])), 4),
            "SxV_binding_mean": round(float(np.mean(stats["SxV_binding"])), 4),
            "IxSxV_binding_mean": round(float(np.mean(stats["IxSxV_binding"])), 4),
        }
        s = attr_summaries[at]
        log(f"  {at} (n={n}): IxV={s['IxV_binding_mean']:+.3f}, "
            f"SxV={s['SxV_binding_mean']:+.3f}, IxSxV={s['IxSxV_binding_mean']:+.3f}")

    # ============================================================
    # STEP 6: Per-object details for color high_compatible
    # ============================================================
    log("\n=== STEP 6: Per-object IxV for color high_compatible ===")
    for key, r in results.items():
        if r["compat_level"] == "high_compatible" and r["attr_type"] == "color":
            log(f"  {r['obj']}-{r['target_val']}: IxV={r['IxV_binding']:+.3f}, "
                f"SxV={r['SxV_binding']:+.3f}, IxSxV={r['IxSxV_binding']:+.3f}")

    # ===== Release model =====
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ===== Save results =====
    all_results = {
        "model": model_name,
        "opt_layer": opt_layer,
        "alpha": alpha,
        "direction_approach": "object_agnostic_SV",
        "level_summaries": level_summaries,
        "attr_summaries": attr_summaries,
        "details": results,
    }

    # Convert numpy types
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

    os.makedirs("results/phase329_three_way", exist_ok=True)
    out_path = f"results/phase329_three_way/{model_name}_phase329.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")

    # ===== Final summary =====
    log("\n" + "=" * 60)
    log(f"SUMMARY — {model_name}")
    log("=" * 60)
    log(f"  Direction approach: object-agnostic S and V")
    log(f"  Alpha: {alpha}")
    log(f"")
    log(f"  By compat_level:")
    for cl in level_order:
        if cl in level_summaries:
            s = level_summaries[cl]
            log(f"    {cl}: IxV={s['IxV_binding_mean']:+.3f}, SxV={s['SxV_binding_mean']:+.3f}, "
                f"IxSxV={s['IxSxV_binding_mean']:+.3f}")
    log(f"")
    log(f"  By attr_type (high_compatible only):")
    for at, s in attr_summaries.items():
        log(f"    {at}: IxV={s['IxV_binding_mean']:+.3f}, SxV={s['SxV_binding_mean']:+.3f}, "
            f"IxSxV={s['IxSxV_binding_mean']:+.3f}")

    log(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)

    run_all(model_name)
    log("Phase 329 complete!")
