"""
Phase 334b: Cross-model Analysis — Round 2 Confirmation
========================================================

Analyzes Phase 334 results across all three models.
If DS7B's attn/MLP split is real, we need more data to confirm.

For round 2, we test DS7B with expanded pairs focusing on the key layer L23,
using ONLY pairs with clear positive binding_range (filtering out anomalous pairs).
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

# Extended HC pairs — more color/attribute pairs for better statistics
EXTENDED_HC = [
    # Original 12
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
    # Additional 12 color pairs
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

# Key layers per model (reduced set for round 2)
PATCH_LAYERS = {
    "qwen3": [25, 29],
    "glm4": [30, 38],
    "deepseek7b": [21, 23, 25],
}

CORRUPTED_PROMPT = "The item"
PATCH_TYPES = ["attn", "mlp", "attn_direct_only", "full_block"]


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


def run_patched(model, tokenizer, device, corrupted_prompt,
                clean_attn_outs, clean_mlp_outs,
                corrupted_attn_outs, corrupted_mlp_outs,
                patch_type, patch_layer, n_layers):
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

    layer = layers[patch_layer]
    if patch_type == "attn":
        if patch_layer in clean_attn_outs and hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(
                make_patch_hook(clean_attn_outs[patch_layer])))
    elif patch_type == "mlp":
        if patch_layer in clean_mlp_outs and hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(
                make_patch_hook(clean_mlp_outs[patch_layer])))
    elif patch_type == "attn_direct_only":
        if patch_layer in clean_attn_outs and hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(
                make_patch_hook(clean_attn_outs[patch_layer])))
        if patch_layer in corrupted_mlp_outs and hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(
                make_patch_hook(corrupted_mlp_outs[patch_layer])))
    elif patch_type == "full_block":
        if patch_layer in clean_attn_outs and hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(
                make_patch_hook(clean_attn_outs[patch_layer])))
        if patch_layer in clean_mlp_outs and hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(
                make_patch_hook(clean_mlp_outs[patch_layer])))

    inp = tokenizer(corrupted_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    for h in hooks:
        h.remove()
    final_hidden = out.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    return final_hidden


def run_round2(model_name):
    log(f"Phase 334b Round 2: Confirmation — {model_name}")
    log("=" * 60)
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    patch_layers = PATCH_LAYERS[model_name]
    W_U = get_W_U(model, model_name)

    # Corrupted baseline
    corrupted_attn_outs, corrupted_mlp_outs, corrupted_hidden, corrupted_seq_len = \
        run_and_capture(model, tokenizer, device, CORRUPTED_PROMPT, n_layers)

    # Per-pair experiments
    results = []
    filtered_count = 0

    for pidx, (obj, target_val, competitor_val) in enumerate(EXTENDED_HC):
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

        # Filter: only keep pairs with positive binding_range (clean > corrupted)
        if binding_range < 0.3:
            filtered_count += 1
            log(f"  [{pidx+1}] {pair_key}: FILTERED (range={binding_range:+.3f})")
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

        for patch_layer in patch_layers:
            for patch_type in PATCH_TYPES:
                patch_key = f"L{patch_layer}_{patch_type}"
                try:
                    patched_hidden = run_patched(
                        model, tokenizer, device, CORRUPTED_PROMPT,
                        clean_attn_outs, clean_mlp_outs,
                        corrupted_attn_outs, corrupted_mlp_outs,
                        patch_type, patch_layer, n_layers,
                    )
                    binding_patched = float(binding_dir @ patched_hidden)
                    recovery_pct = 100.0 * (binding_patched - binding_corrupted) / max(binding_range, 1e-10)
                    pair_result["patches"][patch_key] = {
                        "binding": round(binding_patched, 4),
                        "recovery_pct": round(recovery_pct, 1),
                    }
                except Exception as e:
                    pair_result["patches"][patch_key] = {"error": str(e)}

        results.append(pair_result)
        del clean_attn_outs, clean_mlp_outs
        gc.collect()
        torch.cuda.empty_cache()

        if (pidx + 1) % 6 == 0:
            elapsed = time.time() - t0
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  Progress: {pidx+1}/{len(EXTENDED_HC)}, "
                f"valid={len(results)}, filtered={filtered_count}, "
                f"elapsed={elapsed:.0f}s, GPU={gpu_mem:.2f}GB")

    # Aggregate
    log(f"\n=== Results (filtered pairs only, n={len(results)}) ===")

    if not results:
        log("  No valid pairs after filtering!")
        del model, W_U, corrupted_attn_outs, corrupted_mlp_outs
        gc.collect()
        torch.cuda.empty_cache()
        return

    avg_clean = float(np.mean([r["binding_clean"] for r in results]))
    avg_corrupted = float(np.mean([r["binding_corrupted"] for r in results]))
    avg_range = float(np.mean([r["binding_range"] for r in results]))

    log(f"  Baselines: clean={avg_clean:+.4f}, corrupted={avg_corrupted:+.4f}, range={avg_range:+.4f}")
    log(f"  Filtered out: {filtered_count} pairs with binding_range < 0.3")

    log(f"\n  {'Layer':>5} {'attn':>8} {'mlp':>8} {'attn_dir':>9} {'full':>8} "
        f"{'mlp/attn':>9} {'indirect':>10}")
    log("  " + "-" * 62)

    for patch_layer in patch_layers:
        vals = {}
        stds = {}
        for pt in PATCH_TYPES:
            pk = f"L{patch_layer}_{pt}"
            recs = [r["patches"][pk]["recovery_pct"] for r in results
                    if pk in r["patches"] and "recovery_pct" in r["patches"][pk]]
            if recs:
                vals[pt] = float(np.mean(recs))
                stds[pt] = float(np.std(recs))
            else:
                vals[pt] = float('nan')
                stds[pt] = float('nan')

        mlp_attn_ratio = vals.get('mlp', 0) / max(abs(vals.get('attn', 0.1)), 0.1)
        indirect = vals.get('attn', 0) - vals.get('attn_direct_only', 0)

        log(f"  L{patch_layer:>4} {vals.get('attn', float('nan')):>+8.1f} "
            f"{vals.get('mlp', float('nan')):>+8.1f} "
            f"{vals.get('attn_direct_only', float('nan')):>+9.1f} "
            f"{vals.get('full_block', float('nan')):>+8.1f} "
            f"{mlp_attn_ratio:>8.1f}x {indirect:>+9.1f}%")

    # Save
    save_data = {
        "model": model_name,
        "round": 2,
        "n_valid_pairs": len(results),
        "n_filtered": filtered_count,
        "avg_binding_clean": round(avg_clean, 4),
        "avg_binding_corrupted": round(avg_corrupted, 4),
        "avg_binding_range": round(avg_range, 4),
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

    save_data = convert(save_data)
    os.makedirs("results/phase334_causal_patching", exist_ok=True)
    out_path = f"results/phase334_causal_patching/{model_name}_phase334b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")

    del model, W_U, corrupted_attn_outs, corrupted_mlp_outs
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)
    run_round2(model_name)
