"""
Phase 330: Layer-Level Tracking of Contextual Binding
======================================================

Phase 329b found baseline_binding is the strongest binding signal:
  binding = [logit(t|obj) - logit(c|obj)] - [logit(t|item) - logit(c|item)]

But we don't know WHERE in the network this binding forms.

This script tracks binding score across ALL layers:
  For each layer L:
    binding(L) = [logit_from_L(t|obj) - logit_from_L(c|obj)]
              - [logit_from_L(t|item) - logit_from_L(c|item)]

  where logit_from_L(v|prompt) = W_U[v] @ hidden_state_L[-1]

This reveals:
  - Which layer first shows binding
  - How binding develops across layers
  - Whether binding is formed early (embedding) or late (deep layers)

Usage:
  python tests/glm5/phase330_layer_binding.py qwen3
  python tests/glm5/phase330_layer_binding.py glm4
  python tests/glm5/phase330_layer_binding.py deepseek7b
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

# ===== TEST DATA — expanded from Phase 329b =====
# Format: (object, target_value, competitor_value, attr_type, compat_level)
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
    """Get lm_head weight matrix [vocab_size, d_model]"""
    if hasattr(model, "lm_head"):
        w = model.lm_head.weight
        if not w.is_meta:
            return w.detach().cpu().float().numpy()
    # Fallback: load from safetensors
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
    """Get hidden states from all layers using output_hidden_states=True"""
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, d_model]
    # Index 0 = embedding output, Index 1..n_layers = after each transformer layer
    hs_list = []
    for i, hs in enumerate(out.hidden_states):
        # Take last token position
        hs_list.append(hs[0, -1].float().cpu().numpy())
    return hs_list  # list of [d_model], length = n_layers+1


def run_all(model_name):
    log(f"Phase 330: Layer-Level Binding Tracking — {model_name}")
    log("=" * 60)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]

    # Load W_U for logit projection
    log("  Loading W_U...")
    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")

    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ============================================================
    # STEP 1: Compute per-layer binding for each test pair
    # ============================================================
    log(f"\n=== STEP 1: Per-layer binding computation ===")

    # Layer indices: 0=embedding, 1..n_layers=after each transformer layer
    all_layer_indices = list(range(n_layers + 1))
    # Sample layers for efficiency (every layer is needed for layer tracking)
    # But we only need to output at sampled layers for readability

    results = {}
    per_layer_binding = defaultdict(lambda: {"HC": [], "NI": [], "AA": []})

    for idx, (obj, target_val, competitor_val, attr_type, compat_level) in enumerate(TEST_PAIRS):
        log(f"  [{idx+1}/{len(TEST_PAIRS)}] {obj}-{target_val} ({compat_level})")

        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)

        if tid_t is None or tid_c is None:
            log(f"    SKIP: token not found")
            continue

        # Get hidden states for both prompts
        hs_obj = get_hidden_states(model, tokenizer, device, f"The {obj}", n_layers)
        hs_item = get_hidden_states(model, tokenizer, device, "The item", n_layers)

        # Compute binding at each layer
        layer_bindings = []
        for layer_idx in range(n_layers + 1):
            # Project hidden state to logit space for target and competitor
            logit_t_obj = float(W_U[tid_t] @ hs_obj[layer_idx])
            logit_c_obj = float(W_U[tid_c] @ hs_obj[layer_idx])
            logit_t_item = float(W_U[tid_t] @ hs_item[layer_idx])
            logit_c_item = float(W_U[tid_c] @ hs_item[layer_idx])

            # Binding score at this layer
            advantage_obj = logit_t_obj - logit_c_obj
            advantage_item = logit_t_item - logit_c_item
            binding = advantage_obj - advantage_item

            layer_bindings.append({
                "layer": layer_idx,
                "logit_t_obj": round(logit_t_obj, 4),
                "logit_c_obj": round(logit_c_obj, 4),
                "logit_t_item": round(logit_t_item, 4),
                "logit_c_item": round(logit_c_item, 4),
                "advantage_obj": round(advantage_obj, 4),
                "advantage_item": round(advantage_item, 4),
                "binding": round(binding, 4),
            })

        # Find key layer info
        binding_values = [lb["binding"] for lb in layer_bindings]
        first_positive_layer = None
        for i, bv in enumerate(binding_values):
            if bv > 0:
                first_positive_layer = i
                break

        max_binding = max(binding_values)
        max_binding_layer = binding_values.index(max_binding)

        final_binding = binding_values[-1]

        # Also track the "binding gain" at each layer (derivative)
        binding_gains = [0.0] + [binding_values[i] - binding_values[i-1] for i in range(1, len(binding_values))]
        max_gain = max(binding_gains)
        max_gain_layer = binding_gains.index(max_gain)

        result = {
            "obj": obj,
            "target_val": target_val,
            "competitor_val": competitor_val,
            "attr_type": attr_type,
            "compat_level": compat_level,
            "target_tid": tid_t,
            "competitor_tid": tid_c,
            "first_positive_layer": first_positive_layer,
            "max_binding": round(max_binding, 4),
            "max_binding_layer": max_binding_layer,
            "final_binding": round(final_binding, 4),
            "max_binding_gain": round(max_gain, 4),
            "max_binding_gain_layer": max_gain_layer,
            "layer_bindings": layer_bindings,
        }

        key = f"{obj}_{target_val}"
        results[key] = result

        # Aggregate by compat_level
        cl = compat_level
        per_layer_binding[cl]["all_bindings"].append(binding_values) if "all_bindings" in per_layer_binding[cl] else None

        log(f"    first_pos=L{first_positive_layer}, max={max_binding:+.3f}@L{max_binding_layer}, "
            f"final={final_binding:+.3f}, max_gain={max_gain:+.3f}@L{max_gain_layer}")

        if (idx + 1) % 5 == 0 and torch.cuda.is_available():
            log(f"    GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB, elapsed={time.time()-t0:.0f}s")

    # ============================================================
    # STEP 2: Aggregate layer trajectories by compat_level
    # ============================================================
    log(f"\n=== STEP 2: Layer trajectory aggregation ===")

    level_order = ["high_compatible", "near_incompatible", "abstract_absurd"]
    layer_trajectories = {}

    for cl in level_order:
        # Collect all pair bindings at each layer
        pairs_at_layer = defaultdict(list)
        for key, r in results.items():
            if r["compat_level"] == cl:
                for lb in r["layer_bindings"]:
                    pairs_at_layer[lb["layer"]].append(lb["binding"])

        trajectory = []
        for layer_idx in range(n_layers + 1):
            bindings = pairs_at_layer.get(layer_idx, [])
            if bindings:
                trajectory.append({
                    "layer": layer_idx,
                    "mean_binding": round(float(np.mean(bindings)), 4),
                    "std_binding": round(float(np.std(bindings)), 4),
                    "n": len(bindings),
                    "pos_rate": round(float(np.mean([1 if b > 0 else 0 for b in bindings])), 3),
                })
            else:
                trajectory.append({
                    "layer": layer_idx,
                    "mean_binding": 0.0,
                    "std_binding": 0.0,
                    "n": 0,
                    "pos_rate": 0.0,
                })

        layer_trajectories[cl] = trajectory

        # Find critical layers
        mean_bindings = [t["mean_binding"] for t in trajectory]
        first_positive = None
        for i, mb in enumerate(mean_bindings):
            if mb > 0.1:  # threshold for "meaningful positive"
                first_positive = i
                break

        max_mean = max(mean_bindings)
        max_mean_layer = mean_bindings.index(max_mean)

        # Find layer of maximum binding gain (derivative)
        gains = [0.0] + [mean_bindings[i] - mean_bindings[i-1] for i in range(1, len(mean_bindings))]
        max_gain_val = max(gains)
        max_gain_layer = gains.index(max_gain_val)

        log(f"  {cl}:")
        log(f"    First positive (>0.1): L{first_positive}")
        log(f"    Max mean binding: {max_mean:+.3f} @ L{max_mean_layer}")
        log(f"    Max binding gain: {max_gain_val:+.3f} @ L{max_gain_layer}")

    # Print layer-by-layer summary for key layers
    log(f"\n=== Layer-by-layer binding (sampled) ===")
    sample_layers = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25, n_layers-3, n_layers-2, n_layers-1, n_layers]
    sample_layers = [l for l in sample_layers if l <= n_layers]

    header = f"  {'Layer':>6}"
    for cl in level_order:
        header += f"  {cl:>16}"
    log(header)
    log("  " + "-" * (6 + 18 * len(level_order)))

    for l in sample_layers:
        row = f"  L{l:>5}"
        for cl in level_order:
            if cl in layer_trajectories and l < len(layer_trajectories[cl]):
                t = layer_trajectories[cl][l]
                row += f"  {t['mean_binding']:>+8.3f}({t['pos_rate']:.2f})"
            else:
                row += f"  {'N/A':>16}"
        log(row)

    # ============================================================
    # STEP 3: Binding gain analysis (which layer contributes most)
    # ============================================================
    log(f"\n=== STEP 3: Binding gain analysis ===")

    for cl in level_order:
        if cl not in layer_trajectories:
            continue
        mean_bindings = [t["mean_binding"] for t in layer_trajectories[cl]]
        gains = [0.0] + [mean_bindings[i] - mean_bindings[i-1] for i in range(1, len(mean_bindings))]

        # Top 5 gain layers
        indexed_gains = [(i, g) for i, g in enumerate(gains)]
        top5 = sorted(indexed_gains, key=lambda x: x[1], reverse=True)[:5]

        log(f"  {cl} - Top 5 binding gain layers:")
        for layer, gain in top5:
            log(f"    L{layer}: {gain:+.4f}")

    # ============================================================
    # STEP 4: Individual pair analysis - key examples
    # ============================================================
    log(f"\n=== STEP 4: Key example trajectories ===")

    key_examples = ["apple_red", "banana_yellow", "snow_white", "idea_red", "justice_blue"]
    for key in key_examples:
        if key in results:
            r = results[key]
            lbs = r["layer_bindings"]
            log(f"  {key} ({r['compat_level']}):")
            # Show trajectory at sample layers
            for l in [0, 1, 2, 3, 5, 8, 12, 15, 20, n_layers-1]:
                if l < len(lbs):
                    lb = lbs[l]
                    log(f"    L{l}: bind={lb['binding']:+.3f}, "
                        f"adv_obj={lb['advantage_obj']:+.3f}, adv_item={lb['advantage_item']:+.3f}")

    # ===== Release model =====
    del model, W_U
    gc.collect()
    torch.cuda.empty_cache()

    # ===== Save results =====
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "approach": "layer_binding_tracking",
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

    os.makedirs("results/phase330_layer_binding", exist_ok=True)
    out_path = f"results/phase330_layer_binding/{model_name}_phase330.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")

    # ===== Final summary =====
    log("\n" + "=" * 60)
    log(f"SUMMARY — {model_name}")
    log("=" * 60)
    for cl in level_order:
        if cl in layer_trajectories:
            mean_bindings = [t["mean_binding"] for t in layer_trajectories[cl]]
            final_bind = mean_bindings[-1]
            max_bind = max(mean_bindings)
            max_layer = mean_bindings.index(max_bind)
            log(f"  {cl}: final_binding={final_bind:+.3f}, max_binding={max_bind:+.3f}@L{max_layer}")

    # HC vs AA comparison
    if "high_compatible" in layer_trajectories and "abstract_absurd" in layer_trajectories:
        hc_final = layer_trajectories["high_compatible"][-1]["mean_binding"]
        aa_final = layer_trajectories["abstract_absurd"][-1]["mean_binding"]
        log(f"\n  HC vs AA (final): HC={hc_final:+.3f}, AA={aa_final:+.3f}, HC>AA={hc_final > aa_final}")

        # Find where HC first exceeds AA
        hc_traj = [t["mean_binding"] for t in layer_trajectories["high_compatible"]]
        aa_traj = [t["mean_binding"] for t in layer_trajectories["abstract_absurd"]]
        for i in range(len(hc_traj)):
            if hc_traj[i] > aa_traj[i] and hc_traj[i] > 0.1:
                log(f"  HC first dominates AA at L{i}: HC={hc_traj[i]:+.3f} > AA={aa_traj[i]:+.3f}")
                break

    log(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)

    run_all(model_name)
    log("Phase 330 complete!")
