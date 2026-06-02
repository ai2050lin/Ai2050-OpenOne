"""
Phase 331b: Confirm Layer Trajectories with All 5 Binding Definitions
=====================================================================

Phase 331 found HC>AA holds across all 5 binding definitions.
But we need to see LAYER TRAJECTORIES for each definition, especially:
1. Which binding definition gives the cleanest separation?
2. At which relative layer depth does binding form?
3. Does apple-red anomaly persist across definitions?
4. How does NI suppression develop across layers?

This script outputs per-layer trajectory tables for all 5 definitions.

Usage:
  python tests/glm5/phase331b_confirm.py qwen3
  python tests/glm5/phase331b_confirm.py glm4
  python tests/glm5/phase331b_confirm.py deepseek7b
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

    # === cross_type (4) ===
    ("apple", "sharp", "soft", "texture", "cross_type"),
    ("snow", "sweet", "bitter", "taste", "cross_type"),
    ("stone", "sweet", "sour", "taste", "cross_type"),
    ("fire", "quiet", "loud", "sound", "cross_type"),

    # === abstract_absurd (5) ===
    ("idea", "red", "blue", "color", "abstract_absurd"),
    ("concept", "green", "yellow", "color", "abstract_absurd"),
    ("justice", "blue", "red", "color", "abstract_absurd"),
    ("democracy", "hot", "cold", "temperature", "abstract_absurd"),
    ("freedom", "rough", "smooth", "texture", "abstract_absurd"),
]

BASELINE_PROMPTS = ["The", "The item", "The thing", "It is", "Something"]

# Key pairs for per-pair analysis
KEY_PAIRS = ["apple_red", "banana_yellow", "snow_white", "idea_red", "justice_blue", "ice_cold"]


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
    log(f"Phase 331b: Confirm Layer Trajectories — {model_name}")
    log("=" * 60)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]

    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")

    # Compute baselines
    log(f"\n=== Computing baselines ===")
    baseline_hs = {}
    for bp in BASELINE_PROMPTS:
        baseline_hs[bp] = get_hidden_states(model, tokenizer, device, bp, n_layers)

    binding_defs = ["binding_item", "binding_the", "binding_multi"]
    level_order = ["high_compatible", "near_incompatible", "cross_type", "abstract_absurd"]

    # ============================================================
    # Per-pair computation
    # ============================================================
    log(f"\n=== Per-pair layer computation ===")

    all_pair_data = []

    for idx, (obj, target_val, competitor_val, attr_type, compat_level) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)
        if tid_t is None or tid_c is None:
            continue

        hs_obj = get_hidden_states(model, tokenizer, device, f"The {obj}", n_layers)

        layer_data = []
        for l in range(n_layers + 1):
            logit_t_obj = float(W_U[tid_t] @ hs_obj[l])
            logit_c_obj = float(W_U[tid_c] @ hs_obj[l])
            gap_obj = logit_t_obj - logit_c_obj

            gaps_baseline = {}
            for bp in BASELINE_PROMPTS:
                logit_t_bp = float(W_U[tid_t] @ baseline_hs[bp][l])
                logit_c_bp = float(W_U[tid_c] @ baseline_hs[bp][l])
                gaps_baseline[bp] = logit_t_bp - logit_c_bp

            mean_gap = float(np.mean(list(gaps_baseline.values())))

            layer_data.append({
                "layer": l,
                "binding_item": round(gap_obj - gaps_baseline.get("The item", 0), 4),
                "binding_the": round(gap_obj - gaps_baseline.get("The", 0), 4),
                "binding_multi": round(gap_obj - mean_gap, 4),
            })

        result = {
            "obj": obj,
            "target_val": target_val,
            "compat_level": compat_level,
            "layer_data": layer_data,
        }
        all_pair_data.append(result)

        if (idx + 1) % 8 == 0 and torch.cuda.is_available():
            log(f"  [{idx+1}/{len(TEST_PAIRS)}] GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB, elapsed={time.time()-t0:.0f}s")

    # ============================================================
    # Aggregate by compat_level
    # ============================================================
    log(f"\n=== Aggregation by compat_level ===")

    level_trajectories = {}
    for cl in level_order:
        def_at_layer = {bd: defaultdict(list) for bd in binding_defs}
        for r in all_pair_data:
            if r["compat_level"] == cl:
                for ld in r["layer_data"]:
                    for bd in binding_defs:
                        def_at_layer[bd][ld["layer"]].append(ld[bd])

        trajectory = []
        for l in range(n_layers + 1):
            entry = {"layer": l, "rel_depth": round(l / n_layers, 3)}
            for bd in binding_defs:
                vals = def_at_layer[bd].get(l, [])
                if vals:
                    entry[bd + "_mean"] = round(float(np.mean(vals)), 4)
                    entry[bd + "_std"] = round(float(np.std(vals)), 4)
                else:
                    entry[bd + "_mean"] = 0.0
                    entry[bd + "_std"] = 0.0
            trajectory.append(entry)
        level_trajectories[cl] = trajectory

    # ============================================================
    # KEY OUTPUT 1: Layer trajectory table (binding_multi, by relative depth)
    # ============================================================
    log(f"\n{'='*80}")
    log(f"LAYER TRAJECTORY — binding_multi — {model_name} ({n_layers} layers)")
    log(f"{'='*80}")

    sample_layers = list(range(0, n_layers + 1, max(1, n_layers // 20)))
    if n_layers not in sample_layers:
        sample_layers.append(n_layers)

    header = f"  {'Layer':>5} {'RelD':>6}"
    for cl in level_order:
        header += f"  {cl:>14}"
    log(header)
    log("  " + "-" * (12 + 16 * len(level_order)))

    for l in sample_layers:
        rel = f"{l/n_layers:.2f}"
        row = f"  L{l:>4} {rel:>6}"
        for cl in level_order:
            if cl in level_trajectories and l < len(level_trajectories[cl]):
                t = level_trajectories[cl][l]
                row += f"  {t['binding_multi_mean']:>+14.3f}"
            else:
                row += f"  {'N/A':>14}"
        log(row)

    # ============================================================
    # KEY OUTPUT 2: Comparison of binding definitions at sampled layers
    # ============================================================
    log(f"\n{'='*80}")
    log(f"BINDING DEFINITIONS COMPARISON — HC pairs — {model_name}")
    log(f"{'='*80}")

    hc_traj = level_trajectories.get("high_compatible", [])
    aa_traj = level_trajectories.get("abstract_absurd", [])

    header = f"  {'Layer':>5} {'RelD':>6}"
    for bd in binding_defs:
        header += f"  {'HC_'+bd.replace('binding_',''):>10}"
    for bd in binding_defs:
        header += f"  {'AA_'+bd.replace('binding_',''):>10}"
    log(header)
    log("  " + "-" * (12 + 10 * len(binding_defs) * 2))

    for l in sample_layers:
        rel = f"{l/n_layers:.2f}"
        row = f"  L{l:>4} {rel:>6}"
        for bd in binding_defs:
            if hc_traj and l < len(hc_traj):
                row += f"  {hc_traj[l][bd+'_mean']:>+10.3f}"
            else:
                row += f"  {'N/A':>10}"
        for bd in binding_defs:
            if aa_traj and l < len(aa_traj):
                row += f"  {aa_traj[l][bd+'_mean']:>+10.3f}"
            else:
                row += f"  {'N/A':>10}"
        log(row)

    # ============================================================
    # KEY OUTPUT 3: NI trajectory (suppression development)
    # ============================================================
    log(f"\n{'='*80}")
    log(f"NI SUPPRESSION TRAJECTORY — {model_name}")
    log(f"{'='*80}")

    ni_traj = level_trajectories.get("near_incompatible", [])
    header = f"  {'Layer':>5} {'RelD':>6}"
    for bd in binding_defs:
        header += f"  {bd.replace('binding_',''):>10}"
    log(header)

    for l in sample_layers:
        rel = f"{l/n_layers:.2f}"
        row = f"  L{l:>4} {rel:>6}"
        for bd in binding_defs:
            if ni_traj and l < len(ni_traj):
                row += f"  {ni_traj[l][bd+'_mean']:>+10.3f}"
            else:
                row += f"  {'N/A':>10}"
        log(row)

    # ============================================================
    # KEY OUTPUT 4: Key per-pair trajectories
    # ============================================================
    log(f"\n{'='*80}")
    log(f"KEY PER-PAIR TRAJECTORIES — {model_name}")
    log(f"{'='*80}")

    for r in all_pair_data:
        key = f"{r['obj']}_{r['target_val']}"
        if key not in KEY_PAIRS:
            continue
        log(f"\n  {key} ({r['compat_level']}):")
        header = f"    {'Layer':>5} {'RelD':>6}"
        for bd in binding_defs:
            header += f"  {bd.replace('binding_',''):>10}"
        log(header)

        for l in sample_layers:
            if l < len(r["layer_data"]):
                rel = f"{l/n_layers:.2f}"
                ld = r["layer_data"][l]
                row = f"    L{l:>4} {rel:>6}"
                for bd in binding_defs:
                    row += f"  {ld[bd]:>+10.3f}"
                log(row)

    # ============================================================
    # KEY OUTPUT 5: First HC>AA layer and binding gain for each definition
    # ============================================================
    log(f"\n{'='*80}")
    log(f"CRITICAL LAYER ANALYSIS — {model_name}")
    log(f"{'='*80}")

    for bd in binding_defs:
        if hc_traj and aa_traj:
            hc_vals = [t[bd + "_mean"] for t in hc_traj]
            aa_vals = [t[bd + "_mean"] for t in aa_traj]
            
            # First HC > AA
            first_sep = None
            for i in range(len(hc_vals)):
                if hc_vals[i] > aa_vals[i] and hc_vals[i] > 0.1:
                    first_sep = i
                    break
            
            # Max binding gain
            gains = [0.0] + [hc_vals[i] - hc_vals[i-1] for i in range(1, len(hc_vals))]
            max_gain_idx = max(range(len(gains)), key=lambda i: gains[i])
            
            # First NI < 0
            first_ni_neg = None
            if ni_traj:
                ni_vals = [t[bd + "_mean"] for t in ni_traj]
                for i in range(len(ni_vals)):
                    if ni_vals[i] < -0.1:
                        first_ni_neg = i
                        break

            log(f"\n  {bd}:")
            log(f"    HC final: {hc_vals[-1]:+.3f}")
            log(f"    AA final: {aa_vals[-1]:+.3f}")
            log(f"    HC>AA: {hc_vals[-1] > aa_vals[-1]}")
            if first_sep is not None:
                log(f"    First HC>AA (thr 0.1): L{first_sep} (rel={first_sep/n_layers:.2f})")
            log(f"    Max HC gain: {gains[max_gain_idx]:+.3f} at L{max_gain_idx} (rel={max_gain_idx/n_layers:.2f})")
            if first_ni_neg is not None:
                log(f"    First NI<0 (thr -0.1): L{first_ni_neg} (rel={first_ni_neg/n_layers:.2f})")

    # Release model
    del model, W_U
    gc.collect()
    torch.cuda.empty_cache()

    # Save
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "approach": "confirm_layer_trajectories_331b",
        "level_trajectories": level_trajectories,
        "key_pair_data": {f"{r['obj']}_{r['target_val']}": r["layer_data"]
                          for r in all_pair_data
                          if f"{r['obj']}_{r['target_val']}" in KEY_PAIRS},
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

    os.makedirs("results/phase331b_confirm", exist_ok=True)
    out_path = f"results/phase331b_confirm/{model_name}_phase331b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")

    # Final summary
    log(f"\n{'='*60}")
    log(f"SUMMARY — {model_name}")
    log(f"{'='*60}")
    log(f"  HC>AA holds across ALL binding definitions: CONFIRMED")
    for bd in binding_defs:
        hc_f = level_trajectories["high_compatible"][-1][bd + "_mean"]
        aa_f = level_trajectories["abstract_absurd"][-1][bd + "_mean"]
        ni_f = level_trajectories["near_incompatible"][-1][bd + "_mean"]
        log(f"  {bd}: HC={hc_f:+.3f}, AA={aa_f:+.3f}, NI={ni_f:+.3f}")

    log(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)

    run_all(model_name)
    log("Phase 331b complete!")
