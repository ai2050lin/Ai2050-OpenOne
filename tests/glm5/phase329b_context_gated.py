"""
Phase 329b: Context-Gated Binding Test
=======================================

Phase 329 found IxV interaction is noisy with direction injection.
Key issue: object-agnostic V direction doesn't capture object-value binding.

New approach: Put the OBJECT in the PROMPT (natural context), then inject V.
Test whether the object context makes the value direction more effective
for COMPATIBLE values vs INCOMPATIBLE values.

Design:
  For each (object, target_value, competitor_value) pair:
    Condition A: "The {obj}" baseline
    Condition B: "The {obj}" + inject V_target
    Condition C: "The {obj}" + inject S + V_target
    Condition D: "The item" baseline
    Condition E: "The item" + inject V_target
    Condition F: "The item" + inject S + V_target

  Metrics:
    Value boost with object    = logit(target|B) - logit(target|A)
    Value boost without object = logit(target|E) - logit(target|D)
    Object-gated V boost       = boost_with_obj - boost_without_obj

    Same for competitor token.

    Binding = Object-gated V boost (target) - Object-gated V boost (competitor)

  If binding exists:
    Binding > 0 for high_compatible
    Binding ≈ 0 for abstract_absurd

  Also test S+V combo:
    SV boost with object    = logit(target|C) - logit(target|A)
    SV boost without object = logit(target|F) - logit(target|D)
    Object-gated SV boost   = SV_with_obj - SV_without_obj

Usage:
  python tests/glm5/phase329b_context_gated.py qwen3
  python tests/glm5/phase329b_context_gated.py glm4
  python tests/glm5/phase329b_context_gated.py deepseek7b
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

    # === TEXTURE - abstract_absurd (1) ===
    ("theory", "rough", "smooth", "texture", "abstract_absurd"),

    # === TEMPERATURE - high_compatible (3) ===
    ("ice", "cold", "hot", "temperature", "high_compatible"),
    ("fire", "hot", "cold", "temperature", "high_compatible"),
    ("oven", "hot", "cold", "temperature", "high_compatible"),

    # === TEMPERATURE - abstract_absurd (1) ===
    ("music", "hot", "cold", "temperature", "abstract_absurd"),
]

# Direction templates (object-agnostic)
SLOT_BASE = "It is an object"
SLOT_TESTS = ["It has a property", "It has some feature"]
VALUE_BASE = "It is an object"
VALUE_TESTS = ["It is {val}", "Something is {val}"]


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


def get_logits(model, tokenizer, device, prompt):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1].float().cpu().numpy()


def inject_and_get_logits(model, tokenizer, device, prompt, directions_alphas, layer_idx):
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
    log(f"Phase 329b: Context-Gated Binding — {model_name}")
    log("=" * 60)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    opt_layer = MODEL_CONFIGS[model_name]["opt_layer"]
    log(f"  opt_layer={opt_layer}")

    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ============================================================
    # STEP 1: Compute directions (object-agnostic)
    # ============================================================
    log("\n=== STEP 1: Computing directions ===")

    # Value directions
    all_values = sorted(set(p[1] for p in TEST_PAIRS) | set(p[2] for p in TEST_PAIRS))
    value_dirs = {}
    for i, val in enumerate(all_values):
        base = VALUE_BASE
        tests = [t.format(val=val) for t in VALUE_TESTS]
        value_dirs[val] = compute_direction(model, tokenizer, device, base, tests, opt_layer)
        if (i + 1) % 5 == 0:
            log(f"    {i+1}/{len(all_values)} value directions done")

    log(f"  Value directions: {len(value_dirs)}")

    # Slot direction
    slot_dir = compute_direction(model, tokenizer, device, SLOT_BASE, SLOT_TESTS, opt_layer)
    log(f"  Slot direction computed")

    # ============================================================
    # STEP 2: Context-gated binding test
    # ============================================================
    log(f"\n=== STEP 2: Context-gated binding test ===")

    alpha = 1.0
    results = {}

    for idx, (obj, target_val, competitor_val, attr_type, compat_level) in enumerate(TEST_PAIRS):
        log(f"  [{idx+1}/{len(TEST_PAIRS)}] {obj}-{target_val} ({compat_level})")

        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)

        if tid_t is None or tid_c is None:
            log(f"    SKIP: token not found")
            continue

        V_t = value_dirs[target_val]

        # 6 conditions
        # A: "The {obj}" baseline
        logits_A = get_logits(model, tokenizer, device, f"The {obj}")
        # B: "The {obj}" + V_target
        logits_B = inject_and_get_logits(model, tokenizer, device, f"The {obj}", [(V_t, alpha)], opt_layer)
        # C: "The {obj}" + S + V_target
        logits_C = inject_and_get_logits(model, tokenizer, device, f"The {obj}", [(slot_dir, alpha), (V_t, alpha)], opt_layer)
        # D: "The item" baseline
        logits_D = get_logits(model, tokenizer, device, "The item")
        # E: "The item" + V_target
        logits_E = inject_and_get_logits(model, tokenizer, device, "The item", [(V_t, alpha)], opt_layer)
        # F: "The item" + S + V_target
        logits_F = inject_and_get_logits(model, tokenizer, device, "The item", [(slot_dir, alpha), (V_t, alpha)], opt_layer)

        # === Value boost ===
        # With object context
        vboost_t_obj = float(logits_B[tid_t]) - float(logits_A[tid_t])
        vboost_c_obj = float(logits_B[tid_c]) - float(logits_A[tid_c])
        # Without object context (generic "item")
        vboost_t_item = float(logits_E[tid_t]) - float(logits_D[tid_t])
        vboost_c_item = float(logits_E[tid_c]) - float(logits_D[tid_c])
        # Object-gated V boost
        gated_v_t = vboost_t_obj - vboost_t_item
        gated_v_c = vboost_c_obj - vboost_c_item
        binding_V = gated_v_t - gated_v_c

        # === SV boost ===
        svboost_t_obj = float(logits_C[tid_t]) - float(logits_A[tid_t])
        svboost_c_obj = float(logits_C[tid_c]) - float(logits_A[tid_c])
        svboost_t_item = float(logits_F[tid_t]) - float(logits_D[tid_t])
        svboost_c_item = float(logits_F[tid_c]) - float(logits_D[tid_c])
        gated_sv_t = svboost_t_obj - svboost_t_item
        gated_sv_c = svboost_c_obj - svboost_c_item
        binding_SV = gated_sv_t - gated_sv_c

        # === Baseline comparison (object vs item without injection) ===
        baseline_t_obj = float(logits_A[tid_t])
        baseline_c_obj = float(logits_A[tid_c])
        baseline_t_item = float(logits_D[tid_t])
        baseline_c_item = float(logits_D[tid_c])
        baseline_advantage_t = baseline_t_obj - baseline_t_item
        baseline_advantage_c = baseline_c_obj - baseline_c_item
        baseline_binding = baseline_advantage_t - baseline_advantage_c

        # === Ranking at baseline ===
        # Is target above competitor in the object context?
        rank_obj = 1 if baseline_t_obj > baseline_c_obj else 0
        rank_item = 1 if baseline_t_item > baseline_c_item else 0

        result = {
            "obj": obj,
            "target_val": target_val,
            "competitor_val": competitor_val,
            "attr_type": attr_type,
            "compat_level": compat_level,
            "vboost_t_obj": round(vboost_t_obj, 4),
            "vboost_c_obj": round(vboost_c_obj, 4),
            "vboost_t_item": round(vboost_t_item, 4),
            "vboost_c_item": round(vboost_c_item, 4),
            "gated_v_t": round(gated_v_t, 4),
            "gated_v_c": round(gated_v_c, 4),
            "binding_V": round(binding_V, 4),
            "svboost_t_obj": round(svboost_t_obj, 4),
            "svboost_c_obj": round(svboost_c_obj, 4),
            "svboost_t_item": round(svboost_t_item, 4),
            "svboost_c_item": round(svboost_c_item, 4),
            "gated_sv_t": round(gated_sv_t, 4),
            "gated_sv_c": round(gated_sv_c, 4),
            "binding_SV": round(binding_SV, 4),
            "baseline_t_obj": round(baseline_t_obj, 4),
            "baseline_c_obj": round(baseline_c_obj, 4),
            "baseline_binding": round(baseline_binding, 4),
            "rank_obj": rank_obj,
            "rank_item": rank_item,
        }

        key = f"{obj}_{target_val}"
        results[key] = result

        log(f"    binding_V={binding_V:+.3f}, binding_SV={binding_SV:+.3f}, "
            f"baseline_bind={baseline_binding:+.3f}, rank_obj={rank_obj}")

        if (idx + 1) % 5 == 0 and torch.cuda.is_available():
            log(f"    GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB, elapsed={time.time()-t0:.0f}s")

    # ============================================================
    # STEP 3: Aggregate by compat_level
    # ============================================================
    log("\n=== STEP 3: Aggregation by compat_level ===")

    level_order = ["high_compatible", "near_incompatible", "cross_type", "abstract_absurd"]
    level_stats = defaultdict(lambda: {
        "binding_V": [], "binding_SV": [], "baseline_binding": [],
        "gated_v_t": [], "gated_sv_t": [],
        "rank_obj": [],
    })

    for key, r in results.items():
        cl = r["compat_level"]
        level_stats[cl]["binding_V"].append(r["binding_V"])
        level_stats[cl]["binding_SV"].append(r["binding_SV"])
        level_stats[cl]["baseline_binding"].append(r["baseline_binding"])
        level_stats[cl]["gated_v_t"].append(r["gated_v_t"])
        level_stats[cl]["gated_sv_t"].append(r["gated_sv_t"])
        level_stats[cl]["rank_obj"].append(r["rank_obj"])

    level_summaries = {}
    for cl in level_order:
        if cl not in level_stats:
            continue
        stats = level_stats[cl]
        n = len(stats["binding_V"])
        level_summaries[cl] = {
            "n": n,
            "binding_V_mean": round(float(np.mean(stats["binding_V"])), 4),
            "binding_V_pos_rate": round(float(np.mean([1 if x > 0 else 0 for x in stats["binding_V"]])), 3),
            "binding_SV_mean": round(float(np.mean(stats["binding_SV"])), 4),
            "binding_SV_pos_rate": round(float(np.mean([1 if x > 0 else 0 for x in stats["binding_SV"]])), 3),
            "baseline_binding_mean": round(float(np.mean(stats["baseline_binding"])), 4),
            "baseline_binding_pos_rate": round(float(np.mean([1 if x > 0 else 0 for x in stats["baseline_binding"]])), 3),
            "gated_v_t_mean": round(float(np.mean(stats["gated_v_t"])), 4),
            "gated_sv_t_mean": round(float(np.mean(stats["gated_sv_t"])), 4),
            "rank_obj_rate": round(float(np.mean(stats["rank_obj"])), 3),
        }
        s = level_summaries[cl]
        log(f"  {cl} (n={n}): bind_V={s['binding_V_mean']:+.3f}({s['binding_V_pos_rate']:.2f}), "
            f"bind_SV={s['binding_SV_mean']:+.3f}({s['binding_SV_pos_rate']:.2f}), "
            f"baseline={s['baseline_binding_mean']:+.3f}({s['baseline_binding_pos_rate']:.2f}), "
            f"rank={s['rank_obj_rate']:.2f}")

    # Check monotonicity
    if "high_compatible" in level_summaries and "abstract_absurd" in level_summaries:
        hc_v = level_summaries["high_compatible"]["binding_V_mean"]
        aa_v = level_summaries["abstract_absurd"]["binding_V_mean"]
        hc_sv = level_summaries["high_compatible"]["binding_SV_mean"]
        aa_sv = level_summaries["abstract_absurd"]["binding_SV_mean"]
        hc_bl = level_summaries["high_compatible"]["baseline_binding_mean"]
        aa_bl = level_summaries["abstract_absurd"]["baseline_binding_mean"]
        log(f"\n  Monotonicity (HC > AA):")
        log(f"    binding_V:    HC={hc_v:+.3f} vs AA={aa_v:+.3f}, HC>AA={hc_v > aa_v}")
        log(f"    binding_SV:   HC={hc_sv:+.3f} vs AA={aa_sv:+.3f}, HC>AA={hc_sv > aa_sv}")
        log(f"    baseline:     HC={hc_bl:+.3f} vs AA={aa_bl:+.3f}, HC>AA={hc_bl > aa_bl}")

    # ============================================================
    # STEP 4: High_compatible by attr_type
    # ============================================================
    log("\n=== STEP 4: High_compatible by attr_type ===")

    attr_stats = defaultdict(lambda: {"binding_V": [], "binding_SV": [], "baseline_binding": []})
    for key, r in results.items():
        if r["compat_level"] == "high_compatible":
            at = r["attr_type"]
            attr_stats[at]["binding_V"].append(r["binding_V"])
            attr_stats[at]["binding_SV"].append(r["binding_SV"])
            attr_stats[at]["baseline_binding"].append(r["baseline_binding"])

    attr_summaries = {}
    for at in ["color", "texture", "temperature"]:
        if at not in attr_stats:
            continue
        stats = attr_stats[at]
        n = len(stats["binding_V"])
        attr_summaries[at] = {
            "n": n,
            "binding_V_mean": round(float(np.mean(stats["binding_V"])), 4),
            "binding_SV_mean": round(float(np.mean(stats["binding_SV"])), 4),
            "baseline_binding_mean": round(float(np.mean(stats["baseline_binding"])), 4),
        }
        s = attr_summaries[at]
        log(f"  {at} (n={n}): bind_V={s['binding_V_mean']:+.3f}, "
            f"bind_SV={s['binding_SV_mean']:+.3f}, baseline={s['baseline_binding_mean']:+.3f}")

    # ===== Release model =====
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ===== Save results =====
    all_results = {
        "model": model_name,
        "opt_layer": opt_layer,
        "alpha": alpha,
        "approach": "context_gated",
        "level_summaries": level_summaries,
        "attr_summaries": attr_summaries,
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

    os.makedirs("results/phase329b_context_gated", exist_ok=True)
    out_path = f"results/phase329b_context_gated/{model_name}_phase329b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")

    # ===== Final summary =====
    log("\n" + "=" * 60)
    log(f"SUMMARY — {model_name}")
    log("=" * 60)
    log(f"  Approach: context-gated (object in prompt)")
    for cl in level_order:
        if cl in level_summaries:
            s = level_summaries[cl]
            log(f"  {cl}: bind_V={s['binding_V_mean']:+.3f}, bind_SV={s['binding_SV_mean']:+.3f}, "
                f"baseline={s['baseline_binding_mean']:+.3f}, rank={s['rank_obj_rate']:.2f}")

    log(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)

    run_all(model_name)
    log("Phase 329b complete!")
