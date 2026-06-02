"""
Phase 336b: Round 2 — Granular Early Attention Analysis
========================================================

Round 1 key findings that need confirmation:
1. Qwen3 early attn (L0-8) recovers 69.7% — WHERE does this come from?
2. DS7B early attn (L0-8) recovers 88.5% — same question
3. GLM4 early attn (L0-10) recovers only 0.3% — confirmed minimal
4. DS7B L23 attn_reverse = 65.2% — anomalous, needs check

Round 2 tests:
- Granular early attention blocks: L0-2, L3-5, L6-8, L0-4, L5-8
- Early MLP blocks: L0-2, L3-5, L6-8 (to check if MLP also contributes)
- Full early block (attn+MLP): L0-8 (to see if MLP adds to attn)
- DS7B L21+L23 single-layer reverse (for anomaly check)

Usage:
  python tests/glm5/phase336b_early_attn_confirm.py qwen3
  python tests/glm5/phase336b_early_attn_confirm.py glm4
  python tests/glm5/phase336b_early_attn_confirm.py deepseek7b
"""
import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime

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

# Granular early blocks for Phase 338 confirmation
EARLY_BLOCK_CONFIGS = {
    "qwen3": {
        # Granular attention blocks
        "attn_blocks": [
            {"name": "L0-2_attn", "layers": list(range(0, 3)), "type": "attn"},
            {"name": "L3-5_attn", "layers": list(range(3, 6)), "type": "attn"},
            {"name": "L6-8_attn", "layers": list(range(6, 9)), "type": "attn"},
            {"name": "L0-4_attn", "layers": list(range(0, 5)), "type": "attn"},
            {"name": "L5-8_attn", "layers": list(range(5, 9)), "type": "attn"},
        ],
        # Early MLP blocks (control: does early MLP contribute?)
        "mlp_blocks": [
            {"name": "L0-2_mlp", "layers": list(range(0, 3)), "type": "mlp"},
            {"name": "L3-5_mlp", "layers": list(range(3, 6)), "type": "mlp"},
            {"name": "L6-8_mlp", "layers": list(range(6, 9)), "type": "mlp"},
        ],
        # Full early block (attn+MLP)
        "full_blocks": [
            {"name": "L0-8_full", "layers": list(range(0, 9)), "type": "full"},
        ],
        # Reverse destruction at key layers (for comparison)
        "reverse_layers": [25, 29],
    },
    "glm4": {
        "attn_blocks": [
            {"name": "L0-4_attn", "layers": list(range(0, 5)), "type": "attn"},
            {"name": "L5-10_attn", "layers": list(range(5, 11)), "type": "attn"},
        ],
        "mlp_blocks": [
            {"name": "L0-4_mlp", "layers": list(range(0, 5)), "type": "mlp"},
            {"name": "L5-10_mlp", "layers": list(range(5, 11)), "type": "mlp"},
        ],
        "full_blocks": [
            {"name": "L0-10_full", "layers": list(range(0, 11)), "type": "full"},
        ],
        "reverse_layers": [38],
    },
    "deepseek7b": {
        "attn_blocks": [
            {"name": "L0-2_attn", "layers": list(range(0, 3)), "type": "attn"},
            {"name": "L3-5_attn", "layers": list(range(3, 6)), "type": "attn"},
            {"name": "L6-8_attn", "layers": list(range(6, 9)), "type": "attn"},
            {"name": "L0-4_attn", "layers": list(range(0, 5)), "type": "attn"},
            {"name": "L5-8_attn", "layers": list(range(5, 9)), "type": "attn"},
        ],
        "mlp_blocks": [
            {"name": "L0-2_mlp", "layers": list(range(0, 3)), "type": "mlp"},
            {"name": "L3-5_mlp", "layers": list(range(3, 6)), "type": "mlp"},
            {"name": "L6-8_mlp", "layers": list(range(6, 9)), "type": "mlp"},
        ],
        "full_blocks": [
            {"name": "L0-8_full", "layers": list(range(0, 9)), "type": "full"},
        ],
        "reverse_layers": [21, 23],
    },
}

HC_PAIRS = [
    ("apple", "red", "blue"),
    ("banana", "yellow", "purple"),
    ("snow", "white", "black"),
    ("sky", "blue", "green"),
    ("cherry", "red", "blue"),
    ("leaf", "green", "red"),
    ("stone", "rough", "soft"),
    ("silk", "smooth", "rough"),
    ("ice", "cold", "hot"),
    ("fire", "hot", "cold"),
    ("oven", "hot", "cold"),
    ("fridge", "cold", "hot"),
    ("grass", "green", "red"),
    ("ocean", "blue", "yellow"),
    ("sun", "yellow", "purple"),
    ("blood", "red", "green"),
    ("coal", "black", "white"),
    ("milk", "white", "black"),
    ("rose", "red", "blue"),
    ("gold", "yellow", "gray"),
    ("silver", "gray", "red"),
    ("cloud", "white", "green"),
    ("rain", "wet", "dry"),
    ("desert", "hot", "cold"),
]

CORRUPTED_PROMPT = "The item"


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
                return w.float().numpy()
    raise ValueError(f"Cannot load lm_head for {model_name}")


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return None
    return ids[0]


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")


def run_and_capture(model, tokenizer, device, prompt, n_layers):
    captured = {}
    layers = get_layers(model)
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().cpu()
            else:
                captured[key] = output.detach().cpu()
        return hook
    hooks = []
    for li in range(n_layers):
        layer = layers[li]
        if hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(make_hook(f"attn_{li}")))
        if hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(make_hook(f"mlp_{li}")))
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    for h in hooks:
        h.remove()
    attn_outs = {}
    mlp_outs = {}
    for li in range(n_layers):
        if f"attn_{li}" in captured:
            attn_outs[li] = captured[f"attn_{li}"]
        if f"mlp_{li}" in captured:
            mlp_outs[li] = captured[f"mlp_{li}"]
    final_hidden = out.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    seq_len = inp["input_ids"].shape[1]
    return attn_outs, mlp_outs, final_hidden, seq_len


def run_patched_multilayer(model, tokenizer, device, base_prompt,
                           patch_specs, n_layers):
    layers = get_layers(model)
    hooks = []
    def make_patch_hook(replacement):
        def hook(module, input, output):
            if isinstance(output, tuple):
                target_device = output[0].device
                target_dtype = output[0].dtype
            else:
                target_device = output.device
                target_dtype = output.dtype
            rep = replacement.to(device=target_device, dtype=target_dtype)
            if isinstance(output, tuple):
                return (rep,) + output[1:]
            return rep
        return hook
    for layer_idx, comp_type, replacement in patch_specs:
        layer = layers[layer_idx]
        if comp_type == "attn" and hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(make_patch_hook(replacement)))
        elif comp_type == "mlp" and hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(make_patch_hook(replacement)))
    inp = tokenizer(base_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    for h in hooks:
        h.remove()
    final_hidden = out.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    return final_hidden


def build_block_specs(block, source_attn_outs, source_mlp_outs):
    """Build patch specs based on block type."""
    specs = []
    btype = block["type"]
    for li in block["layers"]:
        if btype in ("attn", "full"):
            if li in source_attn_outs:
                specs.append((li, "attn", source_attn_outs[li]))
        if btype in ("mlp", "full"):
            if li in source_mlp_outs:
                specs.append((li, "mlp", source_mlp_outs[li]))
    return specs


def run_round2(model_name):
    log(f"Phase 336b Round 2: Granular Early Attention — {model_name}")
    log("=" * 60)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    ebcfg = EARLY_BLOCK_CONFIGS[model_name]

    W_U = get_W_U(model, model_name)

    # Corrupted baseline
    corrupted_attn_outs, corrupted_mlp_outs, corrupted_hidden, corrupted_seq_len = \
        run_and_capture(model, tokenizer, device, CORRUPTED_PROMPT, n_layers)

    # Collect all test blocks
    all_blocks = ebcfg["attn_blocks"] + ebcfg["mlp_blocks"] + ebcfg["full_blocks"]

    # Per-pair experiments
    results = []
    filtered_count = 0

    for pidx, (obj, target_val, competitor_val) in enumerate(HC_PAIRS):
        pair_key = f"{obj}_{target_val}"
        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)
        if tid_t is None or tid_c is None:
            continue

        binding_dir = W_U[tid_t] - W_U[tid_c]
        clean_prompt = f"The {obj}"
        clean_attn_outs, clean_mlp_outs, clean_hidden, clean_seq_len = \
            run_and_capture(model, tokenizer, device, clean_prompt, n_layers)

        if clean_seq_len != corrupted_seq_len:
            del clean_attn_outs, clean_mlp_outs
            gc.collect()
            torch.cuda.empty_cache()
            continue

        binding_clean = float(binding_dir @ clean_hidden)
        binding_corrupted = float(binding_dir @ corrupted_hidden)
        binding_range = binding_clean - binding_corrupted

        if binding_range < 0.3:
            filtered_count += 1
            del clean_attn_outs, clean_mlp_outs
            gc.collect()
            torch.cuda.empty_cache()
            continue

        pair_result = {
            "obj": obj, "target_val": target_val, "competitor_val": competitor_val,
            "binding_clean": round(binding_clean, 4),
            "binding_corrupted": round(binding_corrupted, 4),
            "binding_range": round(binding_range, 4),
            "patches": {},
        }

        # ---- Block patching (corrupted → clean) ----
        for block in all_blocks:
            bname = block["name"]
            try:
                specs = build_block_specs(block, clean_attn_outs, clean_mlp_outs)
                if not specs:
                    continue
                patched_hidden = run_patched_multilayer(
                    model, tokenizer, device, CORRUPTED_PROMPT,
                    specs, n_layers,
                )
                binding_patched = float(binding_dir @ patched_hidden)
                recovery_pct = 100.0 * (binding_patched - binding_corrupted) / max(binding_range, 1e-10)
                pair_result["patches"][bname] = {
                    "binding": round(binding_patched, 4),
                    "recovery_pct": round(recovery_pct, 1),
                }
            except Exception as e:
                pair_result["patches"][bname] = {"error": str(e)}

        # ---- Reverse destruction at key layers ----
        for kl in ebcfg["reverse_layers"]:
            for rtype in ["mlp_reverse", "attn_reverse", "full_reverse"]:
                cond_name = f"L{kl}_{rtype}"
                try:
                    specs = []
                    if rtype in ("attn_reverse", "full_reverse") and kl in corrupted_attn_outs:
                        specs.append((kl, "attn", corrupted_attn_outs[kl]))
                    if rtype in ("mlp_reverse", "full_reverse") and kl in corrupted_mlp_outs:
                        specs.append((kl, "mlp", corrupted_mlp_outs[kl]))
                    if not specs:
                        continue
                    patched_hidden = run_patched_multilayer(
                        model, tokenizer, device, clean_prompt,
                        specs, n_layers,
                    )
                    binding_patched = float(binding_dir @ patched_hidden)
                    destruction_pct = 100.0 * (binding_clean - binding_patched) / max(binding_range, 1e-10)
                    pair_result["patches"][cond_name] = {
                        "binding": round(binding_patched, 4),
                        "destruction_pct": round(destruction_pct, 1),
                    }
                except Exception as e:
                    pair_result["patches"][cond_name] = {"error": str(e)}

        results.append(pair_result)
        del clean_attn_outs, clean_mlp_outs
        gc.collect()
        torch.cuda.empty_cache()

        if (pidx + 1) % 4 == 0 or pidx < 2:
            elapsed = time.time() - t0
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{pidx+1}/{len(HC_PAIRS)}] {pair_key}: "
                f"valid={len(results)}, filtered={filtered_count}, "
                f"elapsed={elapsed:.0f}s, GPU={gpu_mem:.2f}GB")

    # ============================================================
    # Aggregate results
    # ============================================================
    log(f"\n{'='*80}")
    log(f"AGGREGATE RESULTS — {model_name}")
    log(f"{'='*80}")

    if not results:
        log("  No valid pairs!")
        del model, W_U, corrupted_attn_outs, corrupted_mlp_outs
        gc.collect()
        torch.cuda.empty_cache()
        return

    n_valid = len(results)
    avg_range = float(np.mean([r["binding_range"] for r in results]))
    log(f"  Valid pairs: {n_valid}, filtered: {filtered_count}, avg_range={avg_range:+.4f}")

    # ---- Early attention results (granular) ----
    log(f"\n--- Granular Early Attention Recovery ---")
    log(f"  {'Block':>15} {'recovery%':>10} {'std%':>8} {'n':>4}")
    log("  " + "-" * 40)

    attn_results = {}
    for block in ebcfg["attn_blocks"]:
        bname = block["name"]
        recs = [r["patches"][bname]["recovery_pct"] for r in results
                if bname in r["patches"] and "recovery_pct" in r["patches"][bname]]
        if recs:
            avg_rec = float(np.mean(recs))
            std_rec = float(np.std(recs))
            attn_results[bname] = {"mean": round(avg_rec, 1), "std": round(std_rec, 1), "n": len(recs)}
            log(f"  {bname:>15} {avg_rec:>+10.1f} {std_rec:>8.1f} {len(recs):>4}")

    # ---- Early MLP results (control) ----
    log(f"\n--- Early MLP Recovery (control) ---")
    log(f"  {'Block':>15} {'recovery%':>10} {'std%':>8} {'n':>4}")
    log("  " + "-" * 40)

    mlp_results = {}
    for block in ebcfg["mlp_blocks"]:
        bname = block["name"]
        recs = [r["patches"][bname]["recovery_pct"] for r in results
                if bname in r["patches"] and "recovery_pct" in r["patches"][bname]]
        if recs:
            avg_rec = float(np.mean(recs))
            std_rec = float(np.std(recs))
            mlp_results[bname] = {"mean": round(avg_rec, 1), "std": round(std_rec, 1), "n": len(recs)}
            log(f"  {bname:>15} {avg_rec:>+10.1f} {std_rec:>8.1f} {len(recs):>4}")

    # ---- Full early block results ----
    log(f"\n--- Full Early Block Recovery (attn+MLP) ---")
    for block in ebcfg["full_blocks"]:
        bname = block["name"]
        recs = [r["patches"][bname]["recovery_pct"] for r in results
                if bname in r["patches"] and "recovery_pct" in r["patches"][bname]]
        if recs:
            avg_rec = float(np.mean(recs))
            std_rec = float(np.std(recs))
            log(f"  {bname}: recovery={avg_rec:+.1f}% (std={std_rec:.1f}%)")
            # Compare with attn-only
            attn_bname = bname.replace("_full", "_attn").replace(
                "L0-8", "L0-8_attn").replace("L0-10", "L0-10_attn")
            # Actually, let me just compare with the widest attn block
            widest_attn = ebcfg["attn_blocks"][-1]["name"]  # Last attn block is widest
            if widest_attn in attn_results:
                attn_rec = attn_results[widest_attn]["mean"]
                log(f"  vs {widest_attn}: {attn_rec:+.1f}% → full adds {avg_rec - attn_rec:+.1f}%")

    # ---- Reverse destruction ----
    log(f"\n--- Reverse Destruction ---")
    log(f"  {'Condition':>20} {'destruction%':>14} {'std%':>8}")
    log("  " + "-" * 46)

    reverse_results = {}
    for kl in ebcfg["reverse_layers"]:
        for rtype in ["mlp_reverse", "attn_reverse", "full_reverse"]:
            cond = f"L{kl}_{rtype}"
            dests = [r["patches"][cond]["destruction_pct"] for r in results
                     if cond in r["patches"] and "destruction_pct" in r["patches"][cond]]
            if dests:
                avg_dest = float(np.mean(dests))
                std_dest = float(np.std(dests))
                reverse_results[cond] = {"mean": round(avg_dest, 1), "std": round(std_dest, 1)}
                log(f"  {cond:>20} {avg_dest:>+14.1f} {std_dest:>8.1f}")

    # ---- Per-pair analysis for early attention ----
    log(f"\n--- Per-pair Early Attention Recovery (widest block) ---")
    widest_attn = ebcfg["attn_blocks"][-1]["name"]
    log(f"  {'Pair':>20} {'range':>8} {'attn_rec':>10}")
    log("  " + "-" * 42)
    for r in results:
        pk = f"{r['obj']}_{r['target_val']}"
        if widest_attn in r["patches"] and "recovery_pct" in r["patches"][widest_attn]:
            rec = r["patches"][widest_attn]["recovery_pct"]
            log(f"  {pk:>20} {r['binding_range']:>+8.3f} {rec:>+10.1f}")

    # Save
    save_data = {
        "model": model_name,
        "round": 2,
        "n_valid_pairs": n_valid,
        "n_filtered": filtered_count,
        "attn_results": attn_results,
        "mlp_results": mlp_results,
        "reverse_results": reverse_results,
        "details": results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

    save_data = convert(save_data)
    os.makedirs("results/phase336_multilayer", exist_ok=True)
    out_path = f"results/phase336_multilayer/{model_name}_phase336b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {out_path}")

    del model, W_U, corrupted_attn_outs, corrupted_mlp_outs
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)
    run_round2(model_name)
