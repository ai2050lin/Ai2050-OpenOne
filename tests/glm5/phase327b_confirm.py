"""
Phase 327b: 确认Phase 327关键发现
=================================
1. slot+value增强是真实计算组合还是方向叠加的几何效应？
2. binding用value方向注入是否能单调？
3. "to touch"对sharp极强(5.78)是否稳定？

验证策略:
- Test 1: alpha对照——slot(alpha/2)+value(alpha/2) vs slot(alpha)+value(alpha) vs value(alpha)
  如果slot+value(alpha/2+alpha/2) > value(alpha)，说明是计算组合而非范数叠加
- Test 2: binding用value方向注入替代type方向
- Test 3: sharp "to touch"用更多对象确认

用法:
  python tests/glm5/phase327b_confirm.py qwen3
  python tests/glm5/phase327b_confirm.py glm4
  python tests/glm5/phase327b_confirm.py deepseek7b
"""
import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


MODEL_CONFIGS = {
    "qwen3": {"path": "", "n_layers": 36, "d_model": 2560, "opt_layer": 0},
    "glm4": {"path": "", "n_layers": 40, "d_model": 4096, "opt_layer": 3},
    "deepseek7b": {"path": "", "n_layers": 28, "d_model": 3584, "opt_layer": 6},
}
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from model_utils import MODEL_CONFIGS as _MU
    for k in MODEL_CONFIGS:
        if k in _MU:
            MODEL_CONFIGS[k]["path"] = _MU[k]["path"]
except ImportError:
    pass


# Test 1: Alpha对照组合
COMBO_OBJECTS = [
    ("apple", "red", "blue"),
    ("banana", "yellow", "purple"),
    ("snow", "white", "black"),
    ("sky", "blue", "red"),
    ("grass", "green", "yellow"),
    ("cherry", "red", "green"),
    ("orange", "orange", "blue"),
    ("leaf", "green", "red"),
    ("rose", "red", "blue"),
    ("sun", "yellow", "black"),
]

COMBO_TEMPLATES = {
    "slot": ["{obj} has some feature", "{obj} has a property"],
    "type": ["{obj} has a color"],
    "value": ["{obj} looks {val}", "{obj} is {val}"],
}
COMBO_BASE = "{obj} is an object"


# Test 2: Binding用value方向注入
# 只用high/absurd对照——如果value方向能让high>absurd，说明binding可区分
BINDING_PAIRS = [
    # High compatibility
    ("apple", "red", "blue", "color"),
    ("snow", "white", "green", "color"),
    ("sky", "blue", "red", "color"),
    ("grass", "green", "purple", "color"),
    ("cherry", "red", "yellow", "color"),
    ("stone", "rough", "soft", "texture"),
    ("silk", "smooth", "prickly", "texture"),
    ("ice", "cold", "hot", "temperature"),
    ("stove", "hot", "cold", "temperature"),
    ("lemon", "sour", "salty", "taste"),
    ("candy", "sweet", "bitter", "taste"),
    # Absurd
    ("apple", "invisible", "transparent", "color"),
    ("snow", "flaming", "electrical", "color"),
    ("sky", "checkerboard", "zigzag", "color"),
    ("music", "rough", "smooth", "texture"),
    ("idea", "grainy", "silky", "texture"),
    ("idea", "hot", "cold", "temperature"),
    ("color", "warm", "frozen", "temperature"),
    ("stone", "sweet", "sour", "taste"),
    ("iron", "salty", "bitter", "taste"),
]


# Test 3: Sharp "to touch" 用更多对象确认
SHARP_OBJECTS = [
    "knife", "needle", "razor", "scissors", "sword", "nail", "thorn", "splinter",
    "glass", "spike", "awl", "dagger",
]


# ============================================================
# 核心函数
# ============================================================

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


def inject_multi_and_get_logits(model, tokenizer, device, prompt, directions_alphas, layer_idx):
    """注入多个方向"""
    layers_list = get_layers(model)
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_modified = hidden.clone()
        for direction, alpha in directions_alphas:
            d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
            hidden_modified[0, -1, :] += (alpha * d_tensor).to(hidden.dtype)
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


def get_baseline_logits(model, tokenizer, device, prompt):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1].float().cpu().numpy()


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


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


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python phase327b_confirm.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    t0 = time.time()
    log(f"=== Phase 327b: Confirmation for {model_name} ===")

    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    opt_layer = cfg["opt_layer"]
    alpha = 2.0

    log(f"  n_layers={cfg['n_layers']}, d_model={cfg['d_model']}, opt_layer={opt_layer}, alpha={alpha}")

    results = {}

    # ============================================================
    # TEST 1: Alpha对照组合测试
    # ============================================================
    # 关键对比:
    # A: value(alpha=2.0) alone — baseline
    # B: slot(alpha=1.0) + value(alpha=1.0) — 总alpha=2.0但各半
    # C: slot(alpha=2.0) + value(alpha=2.0) — Phase 327原始
    # 如果B > A，说明slot参与计算（总alpha相同但分配不同）
    # 如果C > B，说明更多alpha也有效（符合预期）

    log("\n=== TEST 1: Alpha Control Combo (10 objects) ===")

    ALPHA_CONFIGS = {
        "value_a2": "value(alpha=2.0)",
        "slot_a1_value_a1": "slot(alpha=1.0)+value(alpha=1.0)",
        "slot_a2_value_a2": "slot(alpha=2.0)+value(alpha=2.0)",
        "type_a2": "type(alpha=2.0)",
        "slot_a1_type_a1": "slot(alpha=1.0)+type(alpha=1.0)",
        "slot_a2_type_a2": "slot(alpha=2.0)+type(alpha=2.0)",
    }

    alpha_results = {k: {"compat": [], "incompat": []} for k in ALPHA_CONFIGS}

    for i, (obj, compat, incompat) in enumerate(COMBO_OBJECTS):
        if (i+1) % 2 == 0:
            log(f"  alpha combo object {i+1}/10 done")

        base = COMBO_BASE.format(obj=obj)
        baseline = get_baseline_logits(model, tokenizer, device, base)

        compat_tid = get_token_id(tokenizer, compat)
        incompat_tid = get_token_id(tokenizer, incompat)

        # Compute directions
        slot_dir = compute_direction(
            model, tokenizer, device, base,
            [t.format(obj=obj) for t in COMBO_TEMPLATES["slot"]], opt_layer
        )
        type_dir = compute_direction(
            model, tokenizer, device, base,
            [t.format(obj=obj) for t in COMBO_TEMPLATES["type"]], opt_layer
        )
        value_dir = compute_direction(
            model, tokenizer, device, base,
            [t.format(obj=obj, val=compat) for t in COMBO_TEMPLATES["value"]], opt_layer
        )

        configs = {
            "value_a2": [(value_dir, 2.0)],
            "slot_a1_value_a1": [(slot_dir, 1.0), (value_dir, 1.0)],
            "slot_a2_value_a2": [(slot_dir, 2.0), (value_dir, 2.0)],
            "type_a2": [(type_dir, 2.0)],
            "slot_a1_type_a1": [(slot_dir, 1.0), (type_dir, 1.0)],
            "slot_a2_type_a2": [(slot_dir, 2.0), (type_dir, 2.0)],
        }

        for cfg_name, dir_alpha_list in configs.items():
            patched = inject_multi_and_get_logits(model, tokenizer, device, base, dir_alpha_list, opt_layer)
            cd = float(patched[compat_tid] - baseline[compat_tid]) if compat_tid else 0
            id_ = float(patched[incompat_tid] - baseline[incompat_tid]) if incompat_tid else 0
            alpha_results[cfg_name]["compat"].append(cd)
            alpha_results[cfg_name]["incompat"].append(id_)

    # Aggregate
    alpha_agg = {}
    log(f"\n  Alpha Control Results:")
    log(f"  {'Config':<25s} {'Compat':>8s} {'Incompat':>8s} {'C-I':>8s}")
    for cfg_name in ALPHA_CONFIGS:
        r = alpha_results[cfg_name]
        cm = float(np.mean(r["compat"]))
        im = float(np.mean(r["incompat"]))
        agg = {"compat_mean": round(cm, 4), "incompat_mean": round(im, 4), "binding": round(cm - im, 4)}
        alpha_agg[cfg_name] = agg
        log(f"  {cfg_name:<25s} {cm:>8.4f} {im:>8.4f} {cm-im:>8.4f}")

    # 关键对比
    v_a2 = alpha_agg["value_a2"]["compat_mean"]
    s1v1 = alpha_agg["slot_a1_value_a1"]["compat_mean"]
    s2v2 = alpha_agg["slot_a2_value_a2"]["compat_mean"]
    log(f"\n  Key comparisons:")
    log(f"  slot(1)+value(1) vs value(2): {s1v1:.4f} vs {v_a2:.4f}, diff={s1v1-v_a2:+.4f}")
    log(f"  slot(2)+value(2) vs value(2): {s2v2:.4f} vs {v_a2:.4f}, diff={s2v2-v_a2:+.4f}")
    log(f"  slot(1)+value(1) > value(2)? {s1v1 > v_a2}")

    t_a2 = alpha_agg["type_a2"]["compat_mean"]
    s1t1 = alpha_agg["slot_a1_type_a1"]["compat_mean"]
    s2t2 = alpha_agg["slot_a2_type_a2"]["compat_mean"]
    log(f"  slot(1)+type(1) vs type(2): {s1t1:.4f} vs {t_a2:.4f}, diff={s1t1-t_a2:+.4f}")
    log(f"  slot(2)+type(2) vs type(2): {s2t2:.4f} vs {t_a2:.4f}, diff={s2t2-t_a2:+.4f}")

    results["alpha_control"] = alpha_agg

    # ============================================================
    # TEST 2: Binding用value方向注入
    # ============================================================
    log(f"\n=== TEST 2: Binding with Value Direction (20 pairs) ===")

    high_bindings = []
    absurd_bindings = []

    for i, (obj, compat, incompat, attr_type) in enumerate(BINDING_PAIRS):
        if (i+1) % 5 == 0:
            log(f"  binding pair {i+1}/20 done")

        base = f"{obj} is an object"
        baseline = get_baseline_logits(model, tokenizer, device, base)

        compat_tid = get_token_id(tokenizer, compat)
        incompat_tid = get_token_id(tokenizer, incompat)
        if compat_tid is None or incompat_tid is None:
            continue

        # Value方向注入（使用compatible value的方向）
        value_templates = {
            "color": f"{obj} looks {compat}",
            "texture": f"{obj} feels {compat}",
            "temperature": f"{obj} is {compat} to touch",
            "taste": f"{obj} tastes {compat}",
        }
        value_test = value_templates[attr_type]

        h_b = extract_rep(model, tokenizer, device, base, opt_layer)
        h_v = extract_rep(model, tokenizer, device, value_test, opt_layer)
        v_dir = (h_v - h_b)
        v_dir = v_dir / (np.linalg.norm(v_dir) + 1e-8)

        patched = inject_multi_and_get_logits(model, tokenizer, device, base, [(v_dir, alpha)], opt_layer)
        cd = float(patched[compat_tid] - baseline[compat_tid])
        id_ = float(patched[incompat_tid] - baseline[incompat_tid])
        binding = cd - id_

        if i < 11:  # high
            high_bindings.append(binding)
        else:  # absurd
            absurd_bindings.append(binding)

    high_mean = round(float(np.mean(high_bindings)), 4) if high_bindings else 0
    absurd_mean = round(float(np.mean(absurd_bindings)), 4) if absurd_bindings else 0
    log(f"  Value-direction binding:")
    log(f"    High: mean={high_mean:.4f} (n={len(high_bindings)})")
    log(f"    Absurd: mean={absurd_mean:.4f} (n={len(absurd_bindings)})")
    log(f"    High > Absurd? {high_mean > absurd_mean}")

    results["value_binding"] = {
        "high": {"mean": high_mean, "n": len(high_bindings)},
        "absurd": {"mean": absurd_mean, "n": len(absurd_bindings)},
        "high_gt_absurd": high_mean > absurd_mean,
    }

    # ============================================================
    # TEST 3: Sharp "to touch" confirmation (12 objects)
    # ============================================================
    log(f"\n=== TEST 3: Sharp 'to touch' Confirmation (12 objects) ===")

    sharp_stats = {"touch": [], "state": []}

    for i, obj in enumerate(SHARP_OBJECTS):
        if (i+1) % 3 == 0:
            log(f"  sharp object {i+1}/12 done")

        base = f"{obj} is an object"
        baseline = get_baseline_logits(model, tokenizer, device, base)

        constructions = {
            "touch": f"{obj} is sharp to touch",
            "state": f"{obj} is sharp",
        }

        target_words = ["sharp", "pointed", "keen", "cutting"]
        for ch_name, ch_test in constructions.items():
            h_b = extract_rep(model, tokenizer, device, base, opt_layer)
            h_t = extract_rep(model, tokenizer, device, ch_test, opt_layer)
            d = (h_t - h_b)
            d = d / (np.linalg.norm(d) + 1e-8)

            patched = inject_multi_and_get_logits(model, tokenizer, device, base, [(d, alpha)], opt_layer)
            word_deltas = []
            for w in target_words:
                tid = get_token_id(tokenizer, w)
                if tid is not None:
                    word_deltas.append(float(patched[tid] - baseline[tid]))
            if word_deltas:
                sharp_stats[ch_name].append(float(np.mean(word_deltas)))

    sharp_agg = {}
    for ch in ["touch", "state"]:
        vals = sharp_stats[ch]
        sharp_agg[ch] = {"mean": round(float(np.mean(vals)), 4) if vals else 0, "n": len(vals)}
        log(f"    sharp_{ch}: mean={sharp_agg[ch]['mean']:.4f} (n={sharp_agg[ch]['n']})")

    results["sharp_confirm"] = sharp_agg

    # ============================================================
    # Save
    # ============================================================
    out = {
        "model": model_name,
        "n_layers": cfg["n_layers"],
        "d_model": cfg["d_model"],
        "opt_layer": opt_layer,
        "alpha": alpha,
        "results": results,
    }

    out_dir = "results/phase327b_confirm"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_name}_phase327b.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"\n  Saved to {out_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"Done. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
