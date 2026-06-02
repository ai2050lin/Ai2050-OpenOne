"""
Phase 343b: Round 2 Confirmation — Expanded Random Directions
==============================================================

Phase 343 showed balanced amplification is universal across direction types.
This Round 2 test:
1. Increases random directions from 10 to 50 for statistical robustness
2. Tests multiple prompts (not just apple) for direction generality
3. Adds more binding pairs for completeness

Key question: Is the net/gross ratio truly indistinguishable between binding and random?

Usage:
  python tests/glm5/phase343b_confirmation.py qwen3
  python tests/glm5/phase343b_confirmation.py deepseek7b
  python tests/glm5/phase343b_confirmation.py glm4
"""
import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", end=end, flush=True)


MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
        "binding_layers": [21, 23, 25, 27, 29],
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
        "binding_layers": [30, 33, 36, 38],
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
        "binding_layers": [19, 21, 23, 24],
    },
}

TEST_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("ice", "cold", "hot"), ("fire", "hot", "cold"), ("grass", "green", "red"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"), ("blood", "red", "green"),
    ("stone", "rough", "soft"), ("silk", "smooth", "rough"), ("oven", "hot", "cold"),
    ("fridge", "cold", "hot"), ("coal", "black", "white"), ("milk", "white", "black"),
]

CORRUPTED_BASELINE = "The item"
N_RANDOM = 50
N_BINDING_PAIRS = 10  # use more pairs


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
        except: continue
    if model is None: raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def get_W_U(model, model_name):
    if hasattr(model, "lm_head"):
        w = model.lm_head.weight
        if not w.is_meta: return w.detach().cpu().float().numpy()
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
    if hasattr(model, "model") and hasattr(model.model, "layers"): return model.model.layers
    raise ValueError(f"Cannot find layers")


def safe_weight_to_numpy(w):
    if w.is_meta: return None
    try: return w.detach().cpu().float().numpy()
    except: return None


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
                if dk in keys and W_down is None: W_down = sf.get_tensor(dk).float().numpy()
                if W_down is not None: break
        except: continue
    return W_gate, W_up, W_down, d_ff


def get_mlp_weights(layer, model_name=None, model=None):
    mlp = layer.mlp
    W_gate = W_up = W_down = None; d_ff = 0
    if hasattr(mlp, 'gate_up_proj'):
        w = safe_weight_to_numpy(mlp.gate_up_proj.weight)
        if w is not None: d_ff = w.shape[0] // 2; W_gate, W_up = w[:d_ff], w[d_ff:]
    elif hasattr(mlp, 'gate_proj'):
        W_gate = safe_weight_to_numpy(mlp.gate_proj.weight)
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_gate is not None: d_ff = W_gate.shape[0]
        elif W_up is not None: d_ff = W_up.shape[0]
    elif hasattr(mlp, 'up_proj'):
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_up is not None: d_ff = W_up.shape[0]
    if hasattr(mlp, 'down_proj'): W_down = safe_weight_to_numpy(mlp.down_proj.weight)
    if W_down is None and model_name is not None:
        layers = get_layers(model)
        for i, l in enumerate(layers):
            if l is layer: W_gate, W_up, W_down, d_ff = get_mlp_weights_from_disk(model_name, i); break
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
    with torch.no_grad(): out = model(**inp, output_hidden_states=True)
    for h in hooks: h.remove()
    return captured


def channel_decomposition(W_down, direction, gate_clean, up_clean, gate_corrupt, up_corrupt):
    d_ff = W_down.shape[1]
    min_d = min(gate_clean.shape[0], d_ff, W_down.shape[1])
    def silu(x): return x * (1.0 / (1.0 + np.exp(-x)))
    gsc = silu(gate_clean[:min_d]); gsr = silu(gate_corrupt[:min_d])
    uc = up_clean[:min_d]; ur = up_corrupt[:min_d]
    Wd = W_down[:, :min_d]
    dp = direction @ Wd  # direction projection per channel
    cc = dp * gsc * uc; cr = dp * gsr * ur
    delta = cc - cr
    pos_mask = dp > 0; neg_mask = dp < 0
    pos_gross = float(np.sum(np.abs(delta[pos_mask])))
    neg_gross = float(np.sum(np.abs(delta[neg_mask])))
    total_gross = pos_gross + neg_gross
    net = float(np.sum(delta))
    balance = neg_gross / max(pos_gross, 1e-10)
    net_gross = abs(net) / max(total_gross, 1e-10)
    return {"balance": balance, "net_gross_ratio": net_gross, "total_gross": total_gross, "net": net}


def run_experiment(model_name):
    log(f"Phase 343b: Confirmation with 50 random directions — {model_name}")
    log("=" * 70)
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    binding_layers = cfg["binding_layers"]
    W_U = get_W_U(model, model_name)
    d_model = W_U.shape[1]
    log(f"  W_U shape: {W_U.shape}")

    # Pre-extract MLP weights
    layers = get_layers(model)
    mlp_weights = {}
    for li in binding_layers:
        _, _, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_down": W_down, "d_ff": d_ff}

    # ===== Test 1: Binding directions with multiple pairs =====
    log(f"\n  Test 1: Binding directions ({N_BINDING_PAIRS} pairs)")
    binding_results = {li: {"balance": [], "net_gross": []} for li in binding_layers}

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS[:N_BINDING_PAIRS]):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue

        binding_dir = W_U[tid_t] - W_U[tid_c]
        binding_dir_normed = binding_dir / max(np.linalg.norm(binding_dir), 1e-10)
        clean_prompt = f"The {obj}"

        # Quick range check
        inp_c = tokenizer(CORRUPTED_BASELINE, return_tensors="pt", truncation=True, max_length=128).to(device)
        inp_cl = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out_c = model(**inp_c, output_hidden_states=True)
            out_cl = model(**inp_cl, output_hidden_states=True)
        final_c = out_c.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        final_cl = out_cl.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        br = float(binding_dir @ final_cl) - float(binding_dir @ final_c)
        del out_c, out_cl; gc.collect(); torch.cuda.empty_cache()
        if br < 0.3: continue

        clean_caps = capture_mlp_internals(model, tokenizer, device, clean_prompt, binding_layers, cfg["n_layers"])
        corrupt_caps = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, cfg["n_layers"])

        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None: continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps: continue
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]; cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]

            res = channel_decomposition(W_down, binding_dir_normed, cg, cu, crg, cru)
            binding_results[li]["balance"].append(res["balance"])
            binding_results[li]["net_gross"].append(res["net_gross_ratio"])

        del clean_caps, corrupt_caps; gc.collect(); torch.cuda.empty_cache()

        if (pidx + 1) % 5 == 0:
            log(f"    [{pidx+1}/{N_BINDING_PAIRS}] elapsed={time.time()-t0:.0f}s")

    # ===== Test 2: Random directions (50 samples) =====
    log(f"\n  Test 2: Random directions ({N_RANDOM} samples)")
    # Capture activations once for the same prompts
    clean_caps = capture_mlp_internals(model, tokenizer, device, "The apple", binding_layers, cfg["n_layers"])
    corrupt_caps = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, cfg["n_layers"])

    random_results = {li: {"balance": [], "net_gross": []} for li in binding_layers}

    np.random.seed(123)
    for ri in range(N_RANDOM):
        d = np.random.randn(d_model)
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10: continue
        d = d / d_norm

        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None: continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps: continue
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]; cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]

            res = channel_decomposition(W_down, d, cg, cu, crg, cru)
            random_results[li]["balance"].append(res["balance"])
            random_results[li]["net_gross"].append(res["net_gross_ratio"])

    del clean_caps, corrupt_caps; gc.collect(); torch.cuda.empty_cache()

    # ===== Test 3: Multiple prompt contexts for random directions =====
    # Test with 5 different objects as prompts, 10 random dirs each
    log(f"\n  Test 3: Random directions across multiple prompts")
    prompt_objects = ["apple", "banana", "snow", "fire", "ocean"]
    multi_prompt_random = {li: {"balance": [], "net_gross": []} for li in binding_layers}

    for obj in prompt_objects:
        cp = capture_mlp_internals(model, tokenizer, device, f"The {obj}", binding_layers, cfg["n_layers"])
        rp = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, cfg["n_layers"])

        np.random.seed(hash(obj) % 2**31)
        for _ in range(10):
            d = np.random.randn(d_model)
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-10: continue
            d = d / d_norm

            for li in binding_layers:
                mw = mlp_weights[li]
                W_down = mw["W_down"]; d_ff = mw["d_ff"]
                if W_down is None: continue
                gk = f"gate_{li}"; uk = f"up_{li}"
                if gk not in cp or gk not in rp: continue
                cg = cp[gk][:d_ff]; crg = rp[gk][:d_ff]
                cu = cp.get(uk, np.ones(d_ff))[:d_ff]; cru = rp.get(uk, np.ones(d_ff))[:d_ff]
                res = channel_decomposition(W_down, d, cg, cu, crg, cru)
                multi_prompt_random[li]["balance"].append(res["balance"])
                multi_prompt_random[li]["net_gross"].append(res["net_gross_ratio"])

        del cp, rp; gc.collect(); torch.cuda.empty_cache()

    # ===== Aggregate and compare =====
    log(f"\n{'='*70}")
    log(f"RESULTS: Binding vs Random Comparison")
    log(f"{'='*70}")

    log(f"\n  {'Layer':>6} {'Binding Bal':>12} {'Random Bal':>12} {'Diff':>8} "
        f"{'Binding N/G':>12} {'Random N/G':>12} {'N/G Diff':>10}")
    log("  " + "-" * 80)

    all_binding_bal = []; all_random_bal = []
    all_binding_ng = []; all_random_ng = []

    for li in binding_layers:
        bb = binding_results[li]["balance"]
        rb = random_results[li]["balance"]
        bng = binding_results[li]["net_gross"]
        rng = random_results[li]["net_gross"]

        bb_mean = np.mean(bb) if bb else 0
        rb_mean = np.mean(rb) if rb else 0
        bng_mean = np.mean(bng) if bng else 0
        rng_mean = np.mean(rng) if rng else 0

        all_binding_bal.extend(bb); all_random_bal.extend(rb)
        all_binding_ng.extend(bng); all_random_ng.extend(rng)

        log(f"  L{li:>5} {bb_mean:>12.4f} {rb_mean:>12.4f} {bb_mean-rb_mean:>+8.4f} "
            f"{bng_mean:>12.4f} {rng_mean:>12.4f} {bng_mean-rng_mean:>+10.4f}")

    # Overall comparison
    log(f"\n  OVERALL:")
    log(f"    Binding: balance={np.mean(all_binding_bal):.4f}±{np.std(all_binding_bal):.4f}, "
        f"net/gross={np.mean(all_binding_ng):.4f}±{np.std(all_binding_ng):.4f}, n={len(all_binding_bal)}")
    log(f"    Random:  balance={np.mean(all_random_bal):.4f}±{np.std(all_random_bal):.4f}, "
        f"net/gross={np.mean(all_random_ng):.4f}±{np.std(all_random_ng):.4f}, n={len(all_random_bal)}")

    # Multi-prompt random
    all_mpr_bal = []; all_mpr_ng = []
    for li in binding_layers:
        all_mpr_bal.extend(multi_prompt_random[li]["balance"])
        all_mpr_ng.extend(multi_prompt_random[li]["net_gross"])
    if all_mpr_bal:
        log(f"    Multi-prompt random: balance={np.mean(all_mpr_bal):.4f}±{np.std(all_mpr_bal):.4f}, "
            f"net/gross={np.mean(all_mpr_ng):.4f}±{np.std(all_mpr_ng):.4f}, n={len(all_mpr_bal)}")

    # Statistical test: are binding and random significantly different?
    from scipy import stats as scipy_stats
    if len(all_binding_bal) > 5 and len(all_random_bal) > 5:
        t_bal, p_bal = scipy_stats.ttest_ind(all_binding_bal, all_random_bal)
        t_ng, p_ng = scipy_stats.ttest_ind(all_binding_ng, all_random_ng)
        log(f"\n  STATISTICAL TEST (t-test):")
        log(f"    Balance: t={t_bal:.4f}, p={p_bal:.4f} {'***' if p_bal < 0.001 else '**' if p_bal < 0.01 else '*' if p_bal < 0.05 else 'ns'}")
        log(f"    Net/gross: t={t_ng:.4f}, p={p_ng:.4f} {'***' if p_ng < 0.001 else '**' if p_ng < 0.01 else '*' if p_ng < 0.05 else 'ns'}")
        
        if p_bal > 0.05 and p_ng > 0.05:
            log(f"    → No significant difference: balanced amplification is MLP's GENERAL property")
        else:
            log(f"    → Significant difference: binding direction may have special structure")

    # Distribution comparison
    log(f"\n  DISTRIBUTION COMPARISON:")
    log(f"    Binding net/gross percentiles: "
        f"P10={np.percentile(all_binding_ng, 10):.4f}, "
        f"P50={np.percentile(all_binding_ng, 50):.4f}, "
        f"P90={np.percentile(all_binding_ng, 90):.4f}")
    log(f"    Random net/gross percentiles:  "
        f"P10={np.percentile(all_random_ng, 10):.4f}, "
        f"P50={np.percentile(all_random_ng, 50):.4f}, "
        f"P90={np.percentile(all_random_ng, 90):.4f}")

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)): return [convert(v) for v in obj]
        return obj

    save_data = convert({
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "binding_balance_mean": float(np.mean(all_binding_bal)),
        "binding_balance_std": float(np.std(all_binding_bal)),
        "random_balance_mean": float(np.mean(all_random_bal)),
        "random_balance_std": float(np.std(all_random_bal)),
        "binding_net_gross_mean": float(np.mean(all_binding_ng)),
        "binding_net_gross_std": float(np.std(all_binding_ng)),
        "random_net_gross_mean": float(np.mean(all_random_ng)),
        "random_net_gross_std": float(np.std(all_random_ng)),
        "per_layer_binding": {str(li): {"balance": binding_results[li]["balance"],
                                        "net_gross": binding_results[li]["net_gross"]}
                              for li in binding_layers},
        "per_layer_random": {str(li): {"balance": random_results[li]["balance"],
                                       "net_gross": random_results[li]["net_gross"]}
                             for li in binding_layers},
    })

    os.makedirs("results/phase343_generality", exist_ok=True)
    out_path = f"results/phase343_generality/{model_name}_phase343b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {out_path}")

    del model, W_U, mlp_weights; gc.collect(); torch.cuda.empty_cache()
    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS: log(f"Unknown model: {model_name}"); sys.exit(1)
    run_experiment(model_name)
    log("Phase 343b complete!")
