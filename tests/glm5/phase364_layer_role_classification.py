"""
Phase 364: Layer Role Classification — Full-Layer MLP C2R + Position Migration
================================================================================

Goals:
1. Complete MLP C2R damage profile at EVERY layer
2. Position migration scan: how binding info shifts from object→last_token across layers
3. Logit lens binding trace (free with output_hidden_states)
4. Layer classification: writing / carrying / calibration / readout

Notation:
  C2R: effect = -Δgap / |base_gap| (positive = binding damaged)
  R2C: effect = +Δgap / |base_gap| (positive = binding rescued)

Estimated runtime per model:
  Qwen3: ~35 min | GLM4: ~80 min | DS7B: ~45 min
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
        "n_layers": 36, "d_model": 2560,
        "migration_layers": [0, 3, 9, 15, 23, 27, 35],
        "attn_c2r_layers": [0],
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
        "migration_layers": [0, 4, 12, 20, 28, 36, 38],
        "attn_c2r_layers": [0],
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
        "migration_layers": [0, 3, 6, 9, 15, 19, 21, 24],
        "attn_c2r_layers": [0, 3, 9],
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


# ===== Model Loading =====

def load_model_bf16(model_name):
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

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        log(f"  GPU={gpu_count} components, CPU={cpu_count} components, GPU mem={gpu_mem:.2f}GB")

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


def compute_effect(delta_gap, base_gap, direction):
    abs_base = abs(base_gap)
    if abs_base < 1e-10:
        return 0.0
    if direction == "c2r":
        return -delta_gap / abs_base
    else:
        return delta_gap / abs_base


def find_object_positions(tokenizer, prompt, obj_word):
    input_ids = tokenizer.encode(prompt)
    positions = []
    for i, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid]).strip().lower()
        if obj_word.lower() in decoded and decoded != '':
            positions.append(i)
    if not positions:
        positions = [1] if len(input_ids) > 1 else [0]
    return positions


# ===== Hook Helpers =====

def _make_output_patch_hook_last(replacement):
    def hook(module, input, output):
        val = output[0] if isinstance(output, tuple) else output
        modified = val.clone()
        rep_t = torch.tensor(replacement, dtype=modified.dtype, device=modified.device)
        modified[0, -1, :] = rep_t
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook


def _make_input_patch_hook_positions(replacement_dict):
    def pre_hook(module, args):
        hidden_states = args[0]
        modified = hidden_states.clone()
        for pos, rep in replacement_dict.items():
            rep_t = torch.tensor(rep, dtype=modified.dtype, device=modified.device)
            modified[0, pos, :] = rep_t
        return (modified,) + args[1:]
    return pre_hook


def _make_input_patch_hook_all(replacement_full):
    def pre_hook(module, args):
        hidden_states = args[0]
        modified = hidden_states.clone()
        rep_t = torch.tensor(replacement_full, dtype=modified.dtype, device=modified.device)
        n = min(modified.shape[1], rep_t.shape[0])
        modified[0, :n, :] = rep_t[:n, :]
        return (modified,) + args[1:]
    return pre_hook


# ===== Main Experiment =====

def run_phase364(model_name):
    log(f"\n{'='*60}")
    log(f"Phase 364: {model_name}")
    log(f"{'='*60}")

    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    migration_layers = cfg["migration_layers"]
    attn_c2r_layers = cfg["attn_c2r_layers"]

    # Load model
    t0_load = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    layers = get_layers(model)
    log(f"  Load time: {time.time()-t0_load:.0f}s, n_layers={n_layers}, d_model={cfg['d_model']}")

    # Results storage
    results = {
        "model": model_name,
        "phase": "364",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": len(TEST_PAIRS),
        "n_layers": n_layers,
        "migration_layers": migration_layers,
        "attn_c2r_layers": attn_c2r_layers,
        "logit_lens": {},       # layer -> {clean_gaps, corrupt_gaps, binding_signals}
        "mlp_c2r": {},          # layer -> [per_pair_effects]
        "attn_c2r": {},         # layer -> [per_pair_effects]
        "position_migration": {}, # "L{li}_{pos_type}" -> [per_pair_effects]
    }

    t0_total = time.time()
    n_pairs = len(TEST_PAIRS)

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue

        clean_prompt = f"The {obj}"

        # ===== Step 1: Clean forward pass =====
        # Capture: output_hidden_states (logit lens), h_in at migration layers
        clean_h_in_mig = {}
        def make_pre_hook_clean(key):
            def pre_hook(module, args):
                hidden = args[0]
                clean_h_in_mig[key] = hidden[0].detach().cpu().float().numpy()
            return pre_hook

        hooks_1 = []
        for li in migration_layers:
            hooks_1.append(layers[li].register_forward_pre_hook(make_pre_hook_clean(f"h_in_{li}")))

        inp_clean = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out_clean = model(**inp_clean, output_hidden_states=True)
        clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
        clean_hs = out_clean.hidden_states

        for h in hooks_1:
            h.remove()

        clean_gap = float(clean_logits[tid_t] - clean_logits[tid_c])
        clean_seq_len = inp_clean["input_ids"].shape[1]

        # Logit lens for clean — compute immediately, then free hidden states
        clean_ll_gaps = []
        for li in range(len(clean_hs)):
            hs_last = clean_hs[li][0, -1, :].float().cpu().numpy()
            ll_clean_gap = float(hs_last @ W_U[tid_t] - hs_last @ W_U[tid_c])
            clean_ll_gaps.append(ll_clean_gap)
            if str(li) not in results["logit_lens"]:
                results["logit_lens"][str(li)] = {"clean_gaps": [], "corrupt_gaps": [], "binding_signals": []}

        del clean_hs
        gc.collect()

        # ===== Step 2: Corrupt forward pass =====
        # Capture: output_hidden_states (logit lens), h_in at migration layers, mlp_out at ALL layers, attn_out at key layers
        corrupt_h_in_mig = {}
        corrupt_mlp_all = {}
        corrupt_attn_key = {}

        def make_pre_hook_corrupt(key):
            def pre_hook(module, args):
                hidden = args[0]
                corrupt_h_in_mig[key] = hidden[0].detach().cpu().float().numpy()
            return pre_hook

        def make_post_hook_mlp(key):
            def hook(module, input, output):
                val = output[0] if isinstance(output, tuple) else output
                corrupt_mlp_all[key] = val[0, -1, :].detach().cpu().float().numpy()
            return hook

        def make_post_hook_attn(key):
            def hook(module, input, output):
                val = output[0] if isinstance(output, tuple) else output
                corrupt_attn_key[key] = val[0, -1, :].detach().cpu().float().numpy()
            return hook

        hooks_2 = []
        for li in migration_layers:
            hooks_2.append(layers[li].register_forward_pre_hook(make_pre_hook_corrupt(f"h_in_{li}")))
        for li in range(n_layers):
            hooks_2.append(layers[li].mlp.register_forward_hook(make_post_hook_mlp(f"mlp_{li}")))
        for li in attn_c2r_layers:
            hooks_2.append(layers[li].self_attn.register_forward_hook(make_post_hook_attn(f"attn_{li}")))

        inp_corrupt = tokenizer(CORRUPTED_BASELINE, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out_corrupt = model(**inp_corrupt, output_hidden_states=True)
        corrupt_logits_val = out_corrupt.logits[0, -1].float().cpu().numpy()
        corrupt_hs = out_corrupt.hidden_states

        for h in hooks_2:
            h.remove()

        corrupt_gap = float(corrupt_logits_val[tid_t] - corrupt_logits_val[tid_c])
        corrupt_seq_len = inp_corrupt["input_ids"].shape[1]
        base_gap = clean_gap - corrupt_gap

        # Logit lens for corrupt + combine with stored clean values
        for li in range(len(corrupt_hs)):
            hs_last = corrupt_hs[li][0, -1, :].float().cpu().numpy()
            ll_corrupt_gap = float(hs_last @ W_U[tid_t] - hs_last @ W_U[tid_c])
            ll_clean_gap = clean_ll_gaps[li] if li < len(clean_ll_gaps) else 0
            results["logit_lens"][str(li)]["clean_gaps"].append(ll_clean_gap)
            results["logit_lens"][str(li)]["corrupt_gaps"].append(ll_corrupt_gap)
            results["logit_lens"][str(li)]["binding_signals"].append(ll_clean_gap - ll_corrupt_gap)

        del corrupt_hs, clean_ll_gaps
        gc.collect()

        if abs(base_gap) < 1e-10:
            del clean_h_in_mig, corrupt_h_in_mig, corrupt_mlp_all, corrupt_attn_key
            gc.collect(); torch.cuda.empty_cache()
            continue

        # ===== Step 3: MLP C2R at every layer =====
        for li in range(n_layers):
            corrupt_mlp = corrupt_mlp_all.get(f"mlp_{li}")
            if corrupt_mlp is None:
                continue

            hook = layers[li].mlp.register_forward_hook(_make_output_patch_hook_last(corrupt_mlp))
            with torch.no_grad():
                pout = model(**inp_clean, output_hidden_states=False)
            p_logits = pout.logits[0, -1].float().cpu().numpy()
            hook.remove()

            p_gap = float(p_logits[tid_t] - p_logits[tid_c])
            delta_gap = p_gap - clean_gap
            c2r_effect = compute_effect(delta_gap, base_gap, "c2r")

            if str(li) not in results["mlp_c2r"]:
                results["mlp_c2r"][str(li)] = []
            results["mlp_c2r"][str(li)].append(c2r_effect)

            del pout
            if li % 6 == 5:
                gc.collect(); torch.cuda.empty_cache()

        # ===== Step 4: Attn C2R at key layers =====
        for li in attn_c2r_layers:
            corrupt_attn = corrupt_attn_key.get(f"attn_{li}")
            if corrupt_attn is None:
                continue

            hook = layers[li].self_attn.register_forward_hook(_make_output_patch_hook_last(corrupt_attn))
            with torch.no_grad():
                pout = model(**inp_clean, output_hidden_states=False)
            p_logits = pout.logits[0, -1].float().cpu().numpy()
            hook.remove()

            p_gap = float(p_logits[tid_t] - p_logits[tid_c])
            delta_gap = p_gap - clean_gap
            c2r_effect = compute_effect(delta_gap, base_gap, "c2r")

            if str(li) not in results["attn_c2r"]:
                results["attn_c2r"][str(li)] = []
            results["attn_c2r"][str(li)].append(c2r_effect)

            del pout
            gc.collect(); torch.cuda.empty_cache()

        # ===== Step 5: Position migration at key layers =====
        obj_positions_corrupt = find_object_positions(tokenizer, CORRUPTED_BASELINE, "item")

        for li in migration_layers:
            clean_h = clean_h_in_mig.get(f"h_in_{li}")   # [seq_len_clean, d_model]
            corrupt_h = corrupt_h_in_mig.get(f"h_in_{li}")  # [seq_len_corrupt, d_model]

            if clean_h is None or corrupt_h is None:
                continue

            for pos_type in ["last_token", "object_token", "all_tokens"]:
                if pos_type == "last_token":
                    last_pos = corrupt_h.shape[0] - 1
                    repl = {last_pos: clean_h[-1]}
                    hook = layers[li].register_forward_pre_hook(
                        _make_input_patch_hook_positions(repl))

                elif pos_type == "object_token":
                    repl = {}
                    for p in obj_positions_corrupt:
                        if p < min(corrupt_h.shape[0], clean_h.shape[0]):
                            repl[p] = clean_h[p]
                    if not repl:
                        continue
                    hook = layers[li].register_forward_pre_hook(
                        _make_input_patch_hook_positions(repl))

                elif pos_type == "all_tokens":
                    min_len = min(clean_h.shape[0], corrupt_h.shape[0])
                    full_repl = np.copy(corrupt_h)
                    full_repl[:min_len] = clean_h[:min_len]
                    hook = layers[li].register_forward_pre_hook(
                        _make_input_patch_hook_all(full_repl))

                with torch.no_grad():
                    pout = model(**inp_corrupt, output_hidden_states=False)
                p_logits = pout.logits[0, -1].float().cpu().numpy()
                hook.remove()

                p_gap = float(p_logits[tid_t] - p_logits[tid_c])
                delta_gap = p_gap - corrupt_gap
                r2c_effect = compute_effect(delta_gap, base_gap, "r2c")

                key = f"L{li}_{pos_type}"
                if key not in results["position_migration"]:
                    results["position_migration"][key] = []
                results["position_migration"][key].append(r2c_effect)

                del pout
                gc.collect(); torch.cuda.empty_cache()

        # Cleanup per-pair
        del clean_h_in_mig, corrupt_h_in_mig, corrupt_mlp_all, corrupt_attn_key
        gc.collect(); torch.cuda.empty_cache()

        if (pidx + 1) % 3 == 0:
            elapsed = time.time() - t0_total
            gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            eta = elapsed / (pidx + 1) * (n_pairs - pidx - 1)
            log(f"  [{pidx+1}/{n_pairs}] elapsed={elapsed:.0f}s ETA={eta:.0f}s GPU={gpu_gb:.1f}GB")

    # ===== Post-processing: compute statistics =====
    log("\n  Computing statistics...")

    # MLP C2R summary
    mlp_c2r_summary = {}
    for li_str, effects in results["mlp_c2r"].items():
        arr = np.array(effects)
        mlp_c2r_summary[li_str] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "n": len(arr),
            "pos_rate": float(np.mean(arr > 0)),
            "neg_rate": float(np.mean(arr < 0)),
        }

    # Attn C2R summary
    attn_c2r_summary = {}
    for li_str, effects in results["attn_c2r"].items():
        arr = np.array(effects)
        attn_c2r_summary[li_str] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": len(arr),
        }

    # Position migration summary
    pos_mig_summary = {}
    for key, effects in results["position_migration"].items():
        arr = np.array(effects)
        pos_mig_summary[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": len(arr),
        }

    # Logit lens summary
    ll_summary = {}
    for li_str, data in results["logit_lens"].items():
        if len(data["clean_gaps"]) == 0:
            continue
        cg = np.array(data["clean_gaps"])
        crg = np.array(data["corrupt_gaps"])
        bs = np.array(data["binding_signals"])
        ll_summary[li_str] = {
            "clean_gap_mean": float(np.mean(cg)),
            "corrupt_gap_mean": float(np.mean(crg)),
            "binding_signal_mean": float(np.mean(bs)),
            "binding_signal_std": float(np.std(bs)),
        }

    # ===== Layer classification =====
    log("  Classifying layers...")
    layer_roles = {}
    for li in range(n_layers):
        li_str = str(li)
        mlp_c2r_mean = mlp_c2r_summary.get(li_str, {}).get("mean", 0)
        mlp_c2r_pos_rate = mlp_c2r_summary.get(li_str, {}).get("pos_rate", 0.5)
        ll_bs = ll_summary.get(li_str, {}).get("binding_signal_mean", 0)
        ll_bs_next = ll_summary.get(str(li+1), {}).get("binding_signal_mean", 0)
        ll_delta = ll_bs_next - ll_bs  # how much binding signal increases at this layer

        # Classification rules:
        # Writing: high C2R damage (>0.2) OR logit lens signal jumps significantly
        # Carrying: low C2R damage (<0.1) and logit lens signal stable
        # Calibration: negative C2R damage (< -0.1)
        # Readout: late layer with moderate C2R and high logit lens signal

        if mlp_c2r_mean < -0.1:
            role = "calibration"
        elif mlp_c2r_mean > 0.3 and li <= n_layers // 3:
            role = "writing"
        elif mlp_c2r_mean > 0.15 and li > n_layers * 2 // 3:
            role = "readout"
        elif mlp_c2r_mean > 0.15:
            role = "writing"  # mid-layer with significant contribution
        elif abs(mlp_c2r_mean) <= 0.1:
            role = "carrying"
        else:
            role = "carrying"  # default for small positive

        layer_roles[li_str] = {
            "role": role,
            "mlp_c2r_mean": mlp_c2r_mean,
            "logit_lens_binding": ll_bs,
            "logit_lens_delta": ll_delta,
        }

    # Bootstrap CI for key layers
    log("  Computing bootstrap CIs...")
    bootstrap_ci = {}
    key_layers_for_ci = [0] + migration_layers
    np.random.seed(42)
    for li in key_layers_for_ci:
        li_str = str(li)
        effects = results["mlp_c2r"].get(li_str, [])
        if len(effects) < 5:
            continue
        arr = np.array(effects)
        boots = []
        for _ in range(1000):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            boots.append(np.mean(sample))
        bootstrap_ci[li_str] = {
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
        }

    # ===== Save results =====
    output = {
        "model": model_name,
        "phase": "364",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": n_pairs,
        "n_layers": n_layers,
        "total_time_s": round(time.time() - t0_total, 1),
        "mlp_c2r_summary": mlp_c2r_summary,
        "attn_c2r_summary": attn_c2r_summary,
        "position_migration_summary": pos_mig_summary,
        "logit_lens_summary": ll_summary,
        "layer_roles": layer_roles,
        "bootstrap_ci": bootstrap_ci,
        "mlp_c2r_per_pair": {k: [round(x, 4) for x in v] for k, v in results["mlp_c2r"].items()},
        "position_migration_per_pair": {k: [round(x, 4) for x in v] for k, v in results["position_migration"].items()},
    }

    os.makedirs(f"results/phase364_layer_role", exist_ok=True)
    out_path = f"results/phase364_layer_role/{model_name}_phase364.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"  Saved to {out_path}")

    # ===== Print summary =====
    log(f"\n  === MLP C2R Summary ===")
    for li in range(n_layers):
        li_str = str(li)
        s = mlp_c2r_summary.get(li_str, {})
        mean = s.get("mean", 0)
        role = layer_roles.get(li_str, {}).get("role", "?")
        marker = ""
        if mean > 0.2:
            marker = " ***"
        elif mean < -0.1:
            marker = " (neg)"
        log(f"    L{li:2d}: MLP_C2R={mean:+.3f}  role={role}{marker}")

    log(f"\n  === Position Migration ===")
    for li in migration_layers:
        last = pos_mig_summary.get(f"L{li}_last_token", {}).get("mean", 0)
        obj = pos_mig_summary.get(f"L{li}_object_token", {}).get("mean", 0)
        all_t = pos_mig_summary.get(f"L{li}_all_tokens", {}).get("mean", 0)
        log(f"    L{li:2d}: last={last:+.3f}  obj={obj:+.3f}  all={all_t:+.3f}  (obj-last={obj-last:+.3f})")

    log(f"\n  === Layer Role Distribution ===")
    role_counts = defaultdict(int)
    for v in layer_roles.values():
        role_counts[v["role"]] += 1
    for role, count in sorted(role_counts.items()):
        log(f"    {role}: {count} layers")

    total_time = time.time() - t0_total
    log(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # Release model
    del model
    gc.collect(); torch.cuda.empty_cache()

    return output


# ===== Entry Point =====
if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    run_phase364(model_name)
