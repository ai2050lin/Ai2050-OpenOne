"""
Phase 350: Channel Stratification + Gate/Up Mean-Shift Source Decomposition
=============================================================================

Phase 349 found that micro-bias comes from:
  - Positive-proj channels have Δact_mean slightly positive (+0.005)
  - Negative-proj channels have Δact_mean slightly negative (-0.002)
  - This ~2% shift produces all of the net binding signal

This script answers TWO critical questions:

Part A: Which channel intensity band produces the micro-bias?
  - Stratify by |channel_proj|: Top 1%, 1-10%, 10-30%, 30-60%, Bottom 40%
  - For each band: gross_frac, net/gross, Δact_mean_pos - Δact_mean_neg
  - This tells us WHERE in the channel spectrum the bias lives

Part B: Where does the Δact mean-shift come from? (gate vs up decomposition)
  - Decompose Δact = gate_driven + up_driven (exact decomposition)
  - For each component, check:
    (a) Correlation with channel_proj sign
    (b) Mean value in positive-proj vs negative-proj channels
    (c) Which component FIRST shows the pos/neg asymmetry?
  - This tells us WHETHER gate or up is the "path selector"

Part C: Net/gross by channel band, decomposed by gate/up
  - Combine A + B: for each band, what fraction of net comes from gate vs up?

Usage:
  python tests/glm5/phase350_channel_stratify_gateup.py qwen3
  python tests/glm5/phase350_channel_stratify_gateup.py deepseek7b
  python tests/glm5/phase350_channel_stratify_gateup.py glm4
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

TEST_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("fire", "hot", "cold"), ("grass", "green", "red"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"), ("blood", "red", "green"),
    ("ice", "cold", "hot"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
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
    log(f"Phase 350: Channel Stratification + Gate/Up Mean-Shift — {model_name}")
    log("=" * 70)
    t0 = time.time()
    cfg = MODEL_CONFIGS[model_name]
    binding_layers = cfg["binding_layers"]
    n_layers = cfg["n_layers"]

    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    d_model = W_U.shape[1]
    log(f"  W_U shape: {W_U.shape}")

    # Pre-extract MLP weights for all binding layers
    layers = get_layers(model)
    mlp_weights = {}
    for li in binding_layers:
        W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_gate": W_gate, "W_up": W_up, "W_down": W_down, "d_ff": d_ff}
        if W_down is not None:
            log(f"  L{li}: W_down={W_down.shape}, d_ff={d_ff}")
        else:
            log(f"  L{li}: W_down=None (will try disk)")

    # ======================================================================
    # Collect all activation data (18 pairs × binding layers)
    # ======================================================================
    log(f"\nCollecting activations for {len(TEST_PAIRS)} pairs...")

    # Store per-(pair, layer) data
    all_data = []  # list of dicts

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

            # Core quantities
            channel_proj = Wd.T @ direction_normed  # [min_d]
            dact = gsc * uc - gsr * ur               # [min_d]

            # Decompose Δact exactly:
            # Δact = (SiLU(g_c) - SiLU(g_r)) * u_r + SiLU(g_c) * (u_c - u_r)
            gate_diff = gsc - gsr     # SiLU(gate) difference
            up_diff = uc - ur         # up difference
            gate_driven = gate_diff * ur   # gate change at corrupt up level
            up_driven = gsc * up_diff      # up change at clean gate level
            # gate_driven + up_driven = dact (exact)

            # Also decompose at the pre-SiLU level
            raw_gate_diff = cg[:min_d] - crg[:min_d]  # raw gate pre-SiLU diff

            all_data.append({
                "pair": f"{obj}-{target}-{competitor}",
                "layer": li,
                "channel_proj": channel_proj,
                "dact": dact,
                "gate_diff": gate_diff,
                "up_diff": up_diff,
                "gate_driven": gate_driven,
                "up_driven": up_driven,
                "raw_gate_diff": raw_gate_diff,
                "n_channels": min_d,
            })

        del clean_caps, corrupt_caps
        gc.collect(); torch.cuda.empty_cache()
        if (pidx + 1) % 6 == 0:
            log(f"  [{pidx+1}/{len(TEST_PAIRS)}] elapsed={time.time()-t0:.0f}s")

    log(f"  Collected {len(all_data)} (pair, layer) combinations")

    # ======================================================================
    # Part A: Channel Stratification by |channel_proj|
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART A: Channel Stratification by |channel_proj|")
    log(f"{'='*70}")

    # Accumulate across all (pair, layer)
    band_results = {band_name: {
        "gross_total": 0.0, "net_total": 0.0,
        "dact_mean_pos": [], "dact_mean_neg": [],
        "n_channels_total": 0,
    } for band_name, _, _ in BANDS}

    for entry in all_data:
        cproj = entry["channel_proj"]
        dact = entry["dact"]
        n = entry["n_channels"]

        abs_cproj = np.abs(cproj)
        sorted_indices = np.argsort(abs_cproj)[::-1]  # descending by |cproj|
        rank = np.zeros(n, dtype=int)
        rank[sorted_indices] = np.arange(n)
        frac_rank = rank / max(n - 1, 1)  # 0=top, 1=bottom

        for band_name, lo, hi in BANDS:
            mask = (frac_rank >= lo) & (frac_rank < hi)
            if not np.any(mask):
                continue

            band_cproj = cproj[mask]
            band_dact = dact[mask]

            # gross and net in this band
            gross = float(np.sum(np.abs(band_cproj * band_dact)))
            net = float(np.sum(band_cproj * band_dact))

            # Positive vs negative proj channels within this band
            pos_mask_band = band_cproj > 0
            neg_mask_band = band_cproj < 0

            band_results[band_name]["gross_total"] += gross
            band_results[band_name]["net_total"] += net
            band_results[band_name]["n_channels_total"] += int(np.sum(mask))

            if np.sum(pos_mask_band) > 0:
                band_results[band_name]["dact_mean_pos"].append(
                    float(np.mean(band_dact[pos_mask_band])))
            if np.sum(neg_mask_band) > 0:
                band_results[band_name]["dact_mean_neg"].append(
                    float(np.mean(band_dact[neg_mask_band])))

    # Print Part A summary
    total_gross_all = sum(b["gross_total"] for b in band_results.values())
    log(f"\n  {'Band':<15} {'Gross%':>8} {'Net/Gross':>10} {'Δact_mean_pos':>14} {'Δact_mean_neg':>14} {'Pos-Neg':>10}")
    log(f"  {'-'*71}")

    part_a_data = {}
    for band_name, _, _ in BANDS:
        bd = band_results[band_name]
        gross_frac = bd["gross_total"] / max(total_gross_all, 1e-10)
        net_gross = bd["net_total"] / max(bd["gross_total"], 1e-10)
        mean_pos = np.mean(bd["dact_mean_pos"]) if bd["dact_mean_pos"] else 0
        mean_neg = np.mean(bd["dact_mean_neg"]) if bd["dact_mean_neg"] else 0
        pos_neg_diff = mean_pos - mean_neg

        log(f"  {band_name:<15} {gross_frac:>8.4f} {net_gross:>10.4f} "
            f"{mean_pos:>+14.6f} {mean_neg:>+14.6f} {pos_neg_diff:>+10.6f}")

        part_a_data[band_name] = {
            "gross_frac": float(gross_frac),
            "net_gross": float(net_gross),
            "net_total": float(bd["net_total"]),
            "gross_total": float(bd["gross_total"]),
            "dact_mean_pos": float(mean_pos),
            "dact_mean_neg": float(mean_neg),
            "pos_neg_diff": float(pos_neg_diff),
            "n_channels_total": bd["n_channels_total"],
        }

    # ======================================================================
    # Part B: Gate/Up Mean-Shift Source Decomposition
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART B: Gate/Up Mean-Shift Source Decomposition")
    log(f"{'='*70}")
    log(f"  Goal: Find WHICH component first shows pos/neg asymmetry")

    # For each component, compute:
    # (1) Correlation with channel_proj
    # (2) Mean in pos-proj channels vs neg-proj channels
    # (3) The pos-neg mean difference

    components = {
        "Δact": lambda e: e["dact"],
        "gate_diff (SiLU)": lambda e: e["gate_diff"],
        "up_diff": lambda e: e["up_diff"],
        "gate_driven": lambda e: e["gate_driven"],
        "up_driven": lambda e: e["up_driven"],
        "raw_gate_diff (pre-SiLU)": lambda e: e["raw_gate_diff"],
    }

    component_results = {}

    for comp_name, comp_fn in components.items():
        corrs = []
        pos_means = []
        neg_means = []
        pos_abs_means = []
        neg_abs_means = []

        for entry in all_data:
            cproj = entry["channel_proj"]
            comp = comp_fn(entry)

            if len(comp) < 10:
                continue

            # Correlation
            if np.std(comp) > 1e-10 and np.std(cproj) > 1e-10:
                corrs.append(float(np.corrcoef(comp, cproj)[0, 1]))

            # Positive vs negative proj channels
            pos_mask = cproj > 0
            neg_mask = cproj < 0

            if np.sum(pos_mask) > 0:
                pos_means.append(float(np.mean(comp[pos_mask])))
                pos_abs_means.append(float(np.mean(np.abs(comp[pos_mask]))))
            if np.sum(neg_mask) > 0:
                neg_means.append(float(np.mean(comp[neg_mask])))
                neg_abs_means.append(float(np.mean(np.abs(comp[neg_mask]))))

        mean_corr = float(np.mean(corrs)) if corrs else 0
        std_corr = float(np.std(corrs)) if corrs else 0
        mean_pos = float(np.mean(pos_means)) if pos_means else 0
        mean_neg = float(np.mean(neg_means)) if neg_means else 0
        mean_pos_abs = float(np.mean(pos_abs_means)) if pos_abs_means else 0
        mean_neg_abs = float(np.mean(neg_abs_means)) if neg_abs_means else 0

        # Key metric: pos-neg mean difference
        pos_neg_diff = mean_pos - mean_neg

        # Effect size: (pos_mean - neg_mean) / pooled_std
        # This tells us how large the asymmetry is relative to variance
        all_means = pos_means + neg_means
        pooled_std = float(np.std(all_means)) if len(all_means) > 1 else 1.0
        effect_size = pos_neg_diff / max(pooled_std, 1e-10)

        component_results[comp_name] = {
            "corr_with_cproj": mean_corr,
            "corr_std": std_corr,
            "pos_mean": mean_pos,
            "neg_mean": mean_neg,
            "pos_neg_diff": pos_neg_diff,
            "pos_abs_mean": mean_pos_abs,
            "neg_abs_mean": mean_neg_abs,
            "effect_size": effect_size,
        }

    # Print Part B summary
    log(f"\n  {'Component':<25} {'Corr(cproj)':>12} {'Pos_mean':>12} {'Neg_mean':>12} {'Pos-Neg':>12} {'Effect':>8}")
    log(f"  {'-'*81}")

    for comp_name in ["Δact", "gate_diff (SiLU)", "up_diff", "gate_driven", "up_driven", "raw_gate_diff (pre-SiLU)"]:
        cr = component_results[comp_name]
        log(f"  {comp_name:<25} {cr['corr_with_cproj']:>+12.4f} {cr['pos_mean']:>+12.6f} "
            f"{cr['neg_mean']:>+12.6f} {cr['pos_neg_diff']:>+12.6f} {cr['effect_size']:>+8.4f}")

    # ======================================================================
    # Part C: Channel Stratification × Gate/Up Decomposition
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART C: Channel Band × Gate/Up Net Contribution")
    log(f"{'='*70}")

    band_gate_net = {band_name: 0.0 for band_name, _, _ in BANDS}
    band_up_net = {band_name: 0.0 for band_name, _, _ in BANDS}
    band_gate_gross = {band_name: 0.0 for band_name, _, _ in BANDS}
    band_up_gross = {band_name: 0.0 for band_name, _, _ in BANDS}
    band_total_net = {band_name: 0.0 for band_name, _, _ in BANDS}
    band_total_gross = {band_name: 0.0 for band_name, _, _ in BANDS}

    # Per-band gate/up pos-neg mean diffs
    band_gate_posneg = {band_name: [] for band_name, _, _ in BANDS}
    band_up_posneg = {band_name: [] for band_name, _, _ in BANDS}

    for entry in all_data:
        cproj = entry["channel_proj"]
        gate_driven = entry["gate_driven"]
        up_driven = entry["up_driven"]
        dact = entry["dact"]
        n = entry["n_channels"]

        abs_cproj = np.abs(cproj)
        sorted_indices = np.argsort(abs_cproj)[::-1]
        rank = np.zeros(n, dtype=int)
        rank[sorted_indices] = np.arange(n)
        frac_rank = rank / max(n - 1, 1)

        for band_name, lo, hi in BANDS:
            mask = (frac_rank >= lo) & (frac_rank < hi)
            if not np.any(mask):
                continue

            bc = cproj[mask]
            bg = gate_driven[mask]
            bu = up_driven[mask]
            bd = dact[mask]

            # Net and gross for gate-driven and up-driven within this band
            gate_net = float(np.sum(bc * bg))
            up_net = float(np.sum(bc * bu))
            total_net = float(np.sum(bc * bd))
            gate_gross = float(np.sum(np.abs(bc * bg)))
            up_gross = float(np.sum(np.abs(bc * bu)))
            total_gross = float(np.sum(np.abs(bc * bd)))

            band_gate_net[band_name] += gate_net
            band_up_net[band_name] += up_net
            band_gate_gross[band_name] += gate_gross
            band_up_gross[band_name] += up_gross
            band_total_net[band_name] += total_net
            band_total_gross[band_name] += total_gross

            # Gate-driven and up-driven pos/neg mean diffs within this band
            pos_mask = bc > 0
            neg_mask = bc < 0
            if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
                band_gate_posneg[band_name].append(
                    float(np.mean(bg[pos_mask])) - float(np.mean(bg[neg_mask])))
                band_up_posneg[band_name].append(
                    float(np.mean(bu[pos_mask])) - float(np.mean(bu[neg_mask])))

    log(f"\n  {'Band':<15} {'Total Net':>10} {'Gate Net':>10} {'Up Net':>10} "
        f"{'Gate%':>8} {'Gate_PN':>10} {'Up_PN':>10}")
    log(f"  {'-'*73}")

    part_c_data = {}
    for band_name, _, _ in BANDS:
        tn = band_total_net[band_name]
        gn = band_gate_net[band_name]
        un = band_up_net[band_name]
        gate_pct = abs(gn) / max(abs(gn) + abs(un), 1e-10) * 100
        g_pn = float(np.mean(band_gate_posneg[band_name])) if band_gate_posneg[band_name] else 0
        u_pn = float(np.mean(band_up_posneg[band_name])) if band_up_posneg[band_name] else 0

        log(f"  {band_name:<15} {tn:>+10.4f} {gn:>+10.4f} {un:>+10.4f} "
            f"{gate_pct:>7.1f}% {g_pn:>+10.6f} {u_pn:>+10.6f}")

        part_c_data[band_name] = {
            "total_net": float(tn),
            "gate_net": float(gn),
            "up_net": float(un),
            "gate_pct": float(gate_pct),
            "gate_posneg_diff": float(g_pn),
            "up_posneg_diff": float(u_pn),
        }

    # ======================================================================
    # Part D: Verification — Is the pos-neg asymmetry in gate_diff 
    #         or up_diff BEFORE applying SiLU?
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART D: Raw Gate Diff (pre-SiLU) vs Post-SiLU Gate Diff Asymmetry")
    log(f"{'='*70}")
    log(f"  Goal: Check if the asymmetry enters at the raw gate level or only after SiLU")

    raw_gate_posneg_diffs = []
    silu_gate_posneg_diffs = []
    up_posneg_diffs = []

    for entry in all_data:
        cproj = entry["channel_proj"]
        raw_gd = entry["raw_gate_diff"]
        silu_gd = entry["gate_diff"]
        ud = entry["up_diff"]

        pos_mask = cproj > 0
        neg_mask = cproj < 0

        if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
            raw_gate_posneg_diffs.append(
                float(np.mean(raw_gd[pos_mask])) - float(np.mean(raw_gd[neg_mask])))
            silu_gate_posneg_diffs.append(
                float(np.mean(silu_gd[pos_mask])) - float(np.mean(silu_gd[neg_mask])))
            up_posneg_diffs.append(
                float(np.mean(ud[pos_mask])) - float(np.mean(ud[neg_mask])))

    log(f"\n  Raw gate_diff (pre-SiLU) pos-neg mean diff: {np.mean(raw_gate_posneg_diffs):>+.6f} ± {np.std(raw_gate_posneg_diffs):.6f}")
    log(f"  SiLU(gate)_diff (post-SiLU) pos-neg mean diff: {np.mean(silu_gate_posneg_diffs):>+.6f} ± {np.std(silu_gate_posneg_diffs):.6f}")
    log(f"  up_diff pos-neg mean diff: {np.mean(up_posneg_diffs):>+.6f} ± {np.std(up_posneg_diffs):.6f}")

    # Ratio: how much does SiLU amplify the raw gate asymmetry?
    raw_mean = np.mean(np.abs(raw_gate_posneg_diffs))
    silu_mean = np.mean(np.abs(silu_gate_posneg_diffs))
    silu_amplification = silu_mean / max(raw_mean, 1e-10)
    log(f"  SiLU amplification of gate asymmetry: {silu_amplification:.3f}x")

    # Which component has larger pos-neg asymmetry?
    gate_asym = np.mean(np.abs(silu_gate_posneg_diffs))
    up_asym = np.mean(np.abs(up_posneg_diffs))
    log(f"  |gate_diff asymmetry|: {gate_asym:.6f}")
    log(f"  |up_diff asymmetry|: {up_asym:.6f}")
    log(f"  Gate/Up asymmetry ratio: {gate_asym / max(up_asym, 1e-10):.3f}")

    part_d_data = {
        "raw_gate_posneg_diff_mean": float(np.mean(raw_gate_posneg_diffs)),
        "raw_gate_posneg_diff_std": float(np.std(raw_gate_posneg_diffs)),
        "silu_gate_posneg_diff_mean": float(np.mean(silu_gate_posneg_diffs)),
        "silu_gate_posneg_diff_std": float(np.std(silu_gate_posneg_diffs)),
        "up_posneg_diff_mean": float(np.mean(up_posneg_diffs)),
        "up_posneg_diff_std": float(np.std(up_posneg_diffs)),
        "silu_amplification": float(silu_amplification),
        "gate_asym": float(gate_asym),
        "up_asym": float(up_asym),
        "gate_up_asym_ratio": float(gate_asym / max(up_asym, 1e-10)),
    }

    # ======================================================================
    # Save results
    # ======================================================================
    results = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": len(TEST_PAIRS),
        "binding_layers": binding_layers,
        "part_a_channel_stratification": part_a_data,
        "part_b_gate_up_decomposition": component_results,
        "part_c_band_gate_up": part_c_data,
        "part_d_raw_vs_silu_gate": part_d_data,
    }

    os.makedirs(f"results/phase350_channel_stratify_gateup", exist_ok=True)
    out_path = f"results/phase350_channel_stratify_gateup/{model_name}_phase350.json"

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=convert)
    log(f"\n  Results saved to {out_path}")

    # Release model
    del model; gc.collect(); torch.cuda.empty_cache()
    log(f"  GPU memory released")

    elapsed = time.time() - t0
    log(f"\nPhase 350 complete for {model_name} in {elapsed:.0f}s")
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
