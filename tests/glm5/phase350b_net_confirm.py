"""
Phase 350b: Confirmation Test — Top 1% Net Contribution Absolute Value + Extended Pairs
=======================================================================================

Phase 350 found that Top 1% channels have highest net/gross (0.04-0.16),
but we need to confirm:
1. Absolute net contribution of each band (which band contributes most total net?)
2. Extended pair set (add more pairs for statistical robustness)

This is a focused confirmation test.
"""
import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", end=end, flush=True)


MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
        "binding_layers": [21, 23, 25, 27, 29],
        "d_ff": 9728,
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
        "binding_layers": [30, 33, 36, 38],
        "d_ff": 13696,
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
        "binding_layers": [19, 21, 23, 24],
        "d_ff": 18944,
    },
}

# Extended pair set for confirmation
TEST_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("fire", "hot", "cold"), ("grass", "green", "red"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"), ("blood", "red", "green"),
    ("ice", "cold", "hot"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    # Extended pairs
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("rain", "wet", "dry"), ("desert", "hot", "cold"), ("moon", "white", "black"),
    ("smoke", "gray", "red"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("night", "dark", "bright"), ("steel", "gray", "gold"), ("ivory", "white", "black"),
]

CORRUPTED_BASELINE = "The item"

BANDS = [
    ("Top 1%", 0.00, 0.01),
    ("1-10%", 0.01, 0.10),
    ("10-30%", 0.10, 0.30),
    ("30-60%", 0.30, 0.60),
    ("Bottom 40%", 0.60, 1.00),
]


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
            log(f"  Failed with {impl}: {e}")
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


def safe_weight_to_numpy(w):
    if w.is_meta:
        return None
    try:
        return w.detach().cpu().float().numpy()
    except:
        return None


def get_mlp_weights_from_disk(model_name, layer_idx):
    import glob
    from safetensors import safe_open
    W_gate = W_up = W_down = None; d_ff = 0
    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
        try:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                keys = sf.keys()
                p = f"model.layers.{layer_idx}.mlp"
                guk = f"{p}.gate_up_proj.weight"
                if guk in keys:
                    w = sf.get_tensor(guk).float().numpy()
                    d_ff = w.shape[0] // 2; W_gate, W_up = w[:d_ff], w[d_ff:]
                gk = f"{p}.gate_proj.weight"
                if gk in keys and W_gate is None:
                    W_gate = sf.get_tensor(gk).float().numpy(); d_ff = W_gate.shape[0]
                uk = f"{p}.up_proj.weight"
                if uk in keys and W_up is None:
                    W_up = sf.get_tensor(uk).float().numpy()
                    if d_ff == 0: d_ff = W_up.shape[0]
                dk = f"{p}.down_proj.weight"
                if dk in keys and W_down is None:
                    W_down = sf.get_tensor(dk).float().numpy()
                if W_down is not None:
                    break
        except:
            continue
    return W_gate, W_up, W_down, d_ff


def get_mlp_weights(layer, model_name=None, model=None):
    mlp = layer.mlp
    W_gate = W_up = W_down = None; d_ff = 0
    if hasattr(mlp, 'gate_up_proj'):
        w = safe_weight_to_numpy(mlp.gate_up_proj.weight)
        if w is not None:
            d_ff = w.shape[0] // 2; W_gate, W_up = w[:d_ff], w[d_ff:]
    elif hasattr(mlp, 'gate_proj'):
        W_gate = safe_weight_to_numpy(mlp.gate_proj.weight)
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_gate is not None:
            d_ff = W_gate.shape[0]
        elif W_up is not None:
            d_ff = W_up.shape[0]
    elif hasattr(mlp, 'up_proj'):
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_up is not None:
            d_ff = W_up.shape[0]
    if hasattr(mlp, 'down_proj'):
        W_down = safe_weight_to_numpy(mlp.down_proj.weight)
    if W_down is None and model_name is not None:
        layers = get_layers(model)
        for i, l in enumerate(layers):
            if l is layer:
                W_gate, W_up, W_down, d_ff = get_mlp_weights_from_disk(model_name, i)
                break
    return W_gate, W_up, W_down, d_ff


def capture_mlp_internals(model, tokenizer, device, prompt, target_layers, n_layers):
    layers = get_layers(model)
    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            val = output[0] if isinstance(output, tuple) else output
            captured[key] = val[0, -1, :].detach().cpu().float().numpy()
        return hook
    hooks = []
    for li in target_layers:
        layer = layers[li]
        if hasattr(layer.mlp, 'gate_proj'):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(f"gate_{li}")))
        elif hasattr(layer.mlp, 'gate_up_proj'):
            def make_glm4_hook(idx):
                def hook(module, input, output):
                    val = output[0] if isinstance(output, tuple) else output
                    v = val[0, -1, :].detach().cpu().float().numpy()
                    d = v.shape[0] // 2
                    captured[f"gate_{idx}"] = v[:d]; captured[f"up_{idx}"] = v[d:]
                return hook
            hooks.append(layer.mlp.gate_up_proj.register_forward_hook(make_glm4_hook(li)))
        if hasattr(layer.mlp, 'up_proj'):
            hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook(f"up_{li}")))
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        model(**inp, output_hidden_states=True)
    for h in hooks:
        h.remove()
    return captured


def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


def run_experiment(model_name):
    log(f"Phase 350b: Confirmation — Absolute Net by Band ({model_name})")
    log("=" * 70)
    t0 = time.time()
    cfg = MODEL_CONFIGS[model_name]
    binding_layers = cfg["binding_layers"]
    n_layers = cfg["n_layers"]

    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    d_model = W_U.shape[1]

    layers = get_layers(model)
    mlp_weights = {}
    for li in binding_layers:
        W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_gate": W_gate, "W_up": W_up, "W_down": W_down, "d_ff": d_ff}

    # Collect per-band net and gross
    band_net = {band_name: 0.0 for band_name, _, _ in BANDS}
    band_gross = {band_name: 0.0 for band_name, _, _ in BANDS}
    band_net_from_gate = {band_name: 0.0 for band_name, _, _ in BANDS}
    band_net_from_up = {band_name: 0.0 for band_name, _, _ in BANDS}

    # Per-pair per-layer net for significance testing
    pair_layer_nets = defaultdict(lambda: defaultdict(list))

    valid_pairs = 0
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue

        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            continue
        direction_normed = direction / dir_norm

        clean_prompt = f"The {obj}"
        clean_caps = capture_mlp_internals(model, tokenizer, device, clean_prompt, binding_layers, n_layers)
        corrupt_caps = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, n_layers)

        valid_pair = False
        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None:
                continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue

            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]

            min_d = min(d_ff, W_down.shape[1], cg.shape[0])
            Wd = W_down[:, :min_d]

            gsc = silu_np(cg[:min_d]); gsr = silu_np(crg[:min_d])
            uc = cu[:min_d]; ur = cru[:min_d]

            channel_proj = Wd.T @ direction_normed
            dact = gsc * uc - gsr * ur
            gate_driven = (gsc - gsr) * ur
            up_driven = gsc * (uc - ur)

            abs_cproj = np.abs(channel_proj)
            sorted_indices = np.argsort(abs_cproj)[::-1]
            rank = np.zeros(min_d, dtype=int)
            rank[sorted_indices] = np.arange(min_d)
            frac_rank = rank / max(min_d - 1, 1)

            for band_name, lo, hi in BANDS:
                mask = (frac_rank >= lo) & (frac_rank < hi)
                if not np.any(mask):
                    continue

                bc = channel_proj[mask]
                bd = dact[mask]
                bg = gate_driven[mask]
                bu = up_driven[mask]

                net = float(np.sum(bc * bd))
                gross = float(np.sum(np.abs(bc * bd)))
                gate_net = float(np.sum(bc * bg))
                up_net = float(np.sum(bc * bu))

                band_net[band_name] += net
                band_gross[band_name] += gross
                band_net_from_gate[band_name] += gate_net
                band_net_from_up[band_name] += up_net

                pair_layer_nets[band_name][f"{obj}-{target}"].append(net)

            valid_pair = True

        del clean_caps, corrupt_caps
        gc.collect(); torch.cuda.empty_cache()
        if valid_pair:
            valid_pairs += 1
        if (pidx + 1) % 10 == 0:
            log(f"  [{pidx+1}/{len(TEST_PAIRS)}] elapsed={time.time()-t0:.0f}s")

    log(f"  Valid pairs: {valid_pairs}")

    # Print results
    total_net = sum(band_net.values())
    total_gross = sum(band_gross.values())

    log(f"\n  {'Band':<15} {'Gross%':>8} {'Net':>10} {'Net%':>8} {'Net/Gross':>10} {'Gate%Net':>10} {'Up%Net':>10}")
    log(f"  {'-'*73}")

    results = {}
    for band_name, _, _ in BANDS:
        gross_frac = band_gross[band_name] / max(total_gross, 1e-10)
        net_val = band_net[band_name]
        net_frac = net_val / max(total_net, 1e-10)
        net_gross = net_val / max(band_gross[band_name], 1e-10)
        gate_pct = abs(band_net_from_gate[band_name]) / max(abs(band_net_from_gate[band_name]) + abs(band_net_from_up[band_name]), 1e-10) * 100
        up_pct = abs(band_net_from_up[band_name]) / max(abs(band_net_from_gate[band_name]) + abs(band_net_from_up[band_name]), 1e-10) * 100

        log(f"  {band_name:<15} {gross_frac:>8.4f} {net_val:>+10.2f} {net_frac:>8.4f} "
            f"{net_gross:>10.4f} {gate_pct:>9.1f}% {up_pct:>9.1f}%")

        # Significance: fraction of pairs with net > 0
        positive_fracs = []
        for pair_key, nets in pair_layer_nets[band_name].items():
            if nets:
                positive_fracs.append(float(np.mean([n > 0 for n in nets])))

        results[band_name] = {
            "gross_frac": float(gross_frac),
            "net_total": float(net_val),
            "net_frac": float(net_frac),
            "net_gross": float(net_gross),
            "gate_pct": float(gate_pct),
            "up_pct": float(up_pct),
            "n_pairs": len(positive_fracs),
            "mean_positive_frac": float(np.mean(positive_fracs)) if positive_fracs else 0,
        }

    log(f"\n  Total net: {total_net:+.2f}, Total gross: {total_gross:.2f}, Overall net/gross: {total_net/max(total_gross,1e-10):.4f}")

    # Significance test
    log(f"\n  {'Band':<15} {'Mean%positive':>14} {'Interpretation':>30}")
    log(f"  {'-'*60}")
    for band_name, _, _ in BANDS:
        mpf = results[band_name]["mean_positive_frac"]
        interp = "SIGNIFICANT" if mpf > 0.7 else ("moderate" if mpf > 0.55 else "weak/noise")
        log(f"  {band_name:<15} {mpf:>14.3f} {interp:>30}")

    # Save
    os.makedirs(f"results/phase350b_net_confirm", exist_ok=True)
    out_path = f"results/phase350b_net_confirm/{model_name}_phase350b.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"model": model_name, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "n_pairs": valid_pairs, "bands": results}, f, indent=2)
    log(f"\n  Saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    log(f"Phase 350b complete for {model_name} in {time.time()-t0:.0f}s")
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
