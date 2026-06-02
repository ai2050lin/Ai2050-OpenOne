"""
Phase 327: Slot组合因果 + Temperature构式分解 + Binding大矩阵
===========================================================
核心问题:
1. slot是否真正参与属性计算？slot+type/value是否比type/value alone更强？
2. temperature "to touch"是温度专用还是一般物理感知构式？
3. binding是否随兼容等级单调变化？

测试设计:
- Test 1: 7种组合注入(slot, type, value, slot+type, slot+value, type+value, slot+type+value)
- Test 2: temperature构式分解(to touch vs feels vs is vs has)
- Test 3: binding兼容等级矩阵(60对: high/medium/low/absurd)

用法:
  python tests/glm5/phase327_slot_combo_binding_matrix.py qwen3
  python tests/glm5/phase327_slot_combo_binding_matrix.py glm4
  python tests/glm5/phase327_slot_combo_binding_matrix.py deepseek7b
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


# ============================================================
# Test 1: Slot组合因果测试
# ============================================================
# 对5个color对象测试7种组合注入

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

# 输出词测量
SPECIFIC_VALUES = ["red", "blue", "green", "yellow", "orange", "white", "black"]
TYPE_WORDS = ["color"]
GENERIC_PROPERTY = ["property", "feature", "quality", "characteristic", "attribute", "trait"]


# ============================================================
# Test 2: Temperature构式分解
# ============================================================
TEMP_CONSTRUCTION_OBJECTS = [
    ("tea", "hot"), ("ice", "cold"), ("stove", "hot"), ("snow", "cold"),
    ("oven", "hot"), ("fridge", "cold"), ("coffee", "hot"), ("freezer", "cold"),
    ("fireplace", "hot"), ("glacier", "cold"), ("desert", "hot"), ("tundra", "cold"),
    ("sauna", "hot"), ("frost", "cold"), ("magma", "hot"), ("arctic", "cold"),
    ("volcano", "hot"), ("iceberg", "cold"), ("lava", "hot"), ("permafrost", "cold"),
]

# temperature专用模板
TEMP_CONSTRUCTIONS = {
    "contact":  "{obj} is {val} to touch",       # is hot to touch
    "tactile":  "{obj} feels {val}",               # feels hot
    "state":    "{obj} is {val}",                   # is hot
    "type":     "{obj} has a temperature quality",  # type only
}

# 非温度对照：检查"to touch"是否是通用构式
NONTTEMP_CONSTRUCTIONS = {
    "rough_touch":   "{obj} is rough to touch",    # texture + to touch
    "rough_state":   "{obj} is rough",              # texture bare
    "sharp_touch":   "{obj} is sharp to touch",     # shape + to touch
    "sharp_state":   "{obj} is sharp",              # shape bare
    "heavy_lift":    "{obj} is heavy to lift",      # size + to lift
    "heavy_state":   "{obj} is heavy",              # size bare
}

NONTTEMP_OBJECTS = [
    "stone", "sandpaper", "knife", "needle", "boulder", "anvil",
    "steel", "concrete", "glass", "wood", "brick", "iron",
]


# ============================================================
# Test 3: Binding兼容等级矩阵
# ============================================================
# 每对: (object, compat_value, incompat_value, attr_type, compat_level)
# compat_level: 4=high, 3=medium, 2=low, 1=absurd

BINDING_MATRIX = [
    # === Color: 20对 ===
    # High compatibility (4)
    ("apple", "red", "blue", "color", 4),
    ("snow", "white", "green", "color", 4),
    ("sky", "blue", "red", "color", 4),
    ("grass", "green", "purple", "color", 4),
    ("cherry", "red", "yellow", "color", 4),
    # Medium compatibility (3)
    ("banana", "green", "blue", "color", 3),   # unripe banana is green
    ("leaf", "yellow", "purple", "color", 3),   # autumn leaf
    ("sun", "orange", "blue", "color", 3),       # sunset sun
    # Low compatibility (2)
    ("snow", "pink", "green", "color", 2),      # pink snow is rare
    ("apple", "blue", "purple", "color", 2),     # blue apple
    ("sky", "green", "purple", "color", 2),      # green sky
    # Absurd (1)
    ("apple", "invisible", "transparent", "color", 1),
    ("snow", "flaming", "electrical", "color", 1),
    ("sky", "checkerboard", "zigzag", "color", 1),

    # === Texture: 15对 ===
    # High (4)
    ("stone", "rough", "soft", "texture", 4),
    ("silk", "smooth", "prickly", "texture", 4),
    ("sandpaper", "coarse", "silky", "texture", 4),
    ("velvet", "soft", "grainy", "texture", 4),
    ("cactus", "spiky", "smooth", "texture", 4),
    # Medium (3)
    ("wood", "rough", "slippery", "texture", 3),
    ("glass", "smooth", "fuzzy", "texture", 3),
    # Low (2)
    ("water", "rough", "grainy", "texture", 2),
    ("cloud", "solid", "spiky", "texture", 2),
    # Absurd (1)
    ("music", "rough", "smooth", "texture", 1),
    ("idea", "grainy", "silky", "texture", 1),

    # === Temperature: 15对 ===
    # High (4)
    ("ice", "cold", "hot", "temperature", 4),
    ("stove", "hot", "cold", "temperature", 4),
    ("oven", "hot", "freezing", "temperature", 4),
    ("glacier", "cold", "warm", "temperature", 4),
    ("fire", "hot", "cold", "temperature", 4),
    # Medium (3)
    ("tea", "warm", "freezing", "temperature", 3),
    ("room", "comfortable", "blazing", "temperature", 3),
    # Low (2)
    ("stone", "hot", "cold", "temperature", 2),
    ("book", "warm", "frozen", "temperature", 2),
    # Absurd (1)
    ("idea", "hot", "cold", "temperature", 1),
    ("color", "warm", "frozen", "temperature", 1),

    # === Taste: 10对 ===
    # High (4)
    ("lemon", "sour", "salty", "taste", 4),
    ("candy", "sweet", "bitter", "taste", 4),
    ("salt", "salty", "sweet", "taste", 4),
    # Medium (3)
    ("coffee", "bitter", "sour", "taste", 3),
    # Low (2)
    ("water", "sweet", "sour", "taste", 2),
    # Absurd (1)
    ("stone", "sweet", "sour", "taste", 1),
    ("iron", "salty", "bitter", "taste", 1),
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


def inject_and_get_logits(model, tokenizer, device, prompt, direction, layer_idx, alpha):
    layers_list = get_layers(model)
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
        hidden_modified = hidden.clone()
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


def inject_multi_and_get_logits(model, tokenizer, device, prompt, directions_alphas, layer_idx):
    """注入多个方向: directions_alphas = [(dir1, alpha1), (dir2, alpha2), ...]"""
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
    """从多个测试句计算平均方向（相对base）"""
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
        print("Usage: python phase327_slot_combo_binding_matrix.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    t0 = time.time()
    log(f"=== Phase 327: Slot Combo + Temp Construction + Binding Matrix for {model_name} ===")

    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    opt_layer = cfg["opt_layer"]
    alpha = 2.0

    log(f"  n_layers={cfg['n_layers']}, d_model={cfg['d_model']}, opt_layer={opt_layer}, alpha={alpha}")

    results = {}

    # ============================================================
    # TEST 1: Slot组合因果测试 (10 color objects × 7 injection combos)
    # ============================================================
    log("\n=== TEST 1: Slot Combination Causal Test (10 objects) ===")

    combo_results = {}  # combo_name -> list of per-object dicts

    COMBO_NAMES = ["slot", "type", "value", "slot+type", "slot+value", "type+value", "slot+type+value"]

    for combo_name in COMBO_NAMES:
        combo_results[combo_name] = {
            "compat_deltas": [],    # compat value logit delta
            "incompat_deltas": [],  # incompat value logit delta
            "generic_deltas": [],   # generic property words delta
            "type_deltas": [],      # type word delta
            "specific_deltas": [],  # all specific values delta
        }

    for i, (obj, compat, incompat) in enumerate(COMBO_OBJECTS):
        if (i+1) % 2 == 0:
            log(f"  combo object {i+1}/10 done")

        base = COMBO_BASE.format(obj=obj)
        baseline = get_baseline_logits(model, tokenizer, device, base)

        # Compute directions
        slot_dir = compute_direction(
            model, tokenizer, device, base,
            [t.format(obj=obj) for t in COMBO_TEMPLATES["slot"]],
            opt_layer
        )
        type_dir = compute_direction(
            model, tokenizer, device, base,
            [t.format(obj=obj) for t in COMBO_TEMPLATES["type"]],
            opt_layer
        )
        value_dir = compute_direction(
            model, tokenizer, device, base,
            [t.format(obj=obj, val=compat) for t in COMBO_TEMPLATES["value"]],
            opt_layer
        )

        # 7种组合注入
        combos = {
            "slot": [(slot_dir, alpha)],
            "type": [(type_dir, alpha)],
            "value": [(value_dir, alpha)],
            "slot+type": [(slot_dir, alpha), (type_dir, alpha)],
            "slot+value": [(slot_dir, alpha), (value_dir, alpha)],
            "type+value": [(type_dir, alpha), (value_dir, alpha)],
            "slot+type+value": [(slot_dir, alpha), (type_dir, alpha), (value_dir, alpha)],
        }

        for combo_name, dir_alpha_list in combos.items():
            if len(dir_alpha_list) == 1:
                patched = inject_and_get_logits(
                    model, tokenizer, device, base, dir_alpha_list[0][0], opt_layer, dir_alpha_list[0][1]
                )
            else:
                patched = inject_multi_and_get_logits(
                    model, tokenizer, device, base, dir_alpha_list, opt_layer
                )

            # 测量输出
            compat_tid = get_token_id(tokenizer, compat)
            incompat_tid = get_token_id(tokenizer, incompat)

            compat_d = float(patched[compat_tid] - baseline[compat_tid]) if compat_tid else 0
            incompat_d = float(patched[incompat_tid] - baseline[incompat_tid]) if incompat_tid else 0

            generic_ds = [float(patched[get_token_id(tokenizer, w)] - baseline[get_token_id(tokenizer, w)])
                         for w in GENERIC_PROPERTY if get_token_id(tokenizer, w) is not None]
            type_ds = [float(patched[get_token_id(tokenizer, w)] - baseline[get_token_id(tokenizer, w)])
                      for w in TYPE_WORDS if get_token_id(tokenizer, w) is not None]
            spec_ds = [float(patched[get_token_id(tokenizer, w)] - baseline[get_token_id(tokenizer, w)])
                      for w in SPECIFIC_VALUES if get_token_id(tokenizer, w) is not None]

            combo_results[combo_name]["compat_deltas"].append(compat_d)
            combo_results[combo_name]["incompat_deltas"].append(incompat_d)
            combo_results[combo_name]["generic_deltas"].append(float(np.mean(generic_ds)) if generic_ds else 0)
            combo_results[combo_name]["type_deltas"].append(float(np.mean(type_ds)) if type_ds else 0)
            combo_results[combo_name]["specific_deltas"].append(float(np.mean(spec_ds)) if spec_ds else 0)

    # Aggregate
    combo_agg = {}
    log(f"\n  === Combo Results (10 color objects) ===")
    log(f"  {'Combo':<20s} {'Compat':>8s} {'Incompat':>8s} {'Generic':>8s} {'Type':>8s} {'Specific':>8s}")
    for combo_name in COMBO_NAMES:
        r = combo_results[combo_name]
        agg = {
            "compat_mean": round(float(np.mean(r["compat_deltas"])), 4),
            "incompat_mean": round(float(np.mean(r["incompat_deltas"])), 4),
            "generic_mean": round(float(np.mean(r["generic_deltas"])), 4),
            "type_mean": round(float(np.mean(r["type_deltas"])), 4),
            "specific_mean": round(float(np.mean(r["specific_deltas"])), 4),
            "compat_pos_rate": round(sum(1 for v in r["compat_deltas"] if v > 0) / max(len(r["compat_deltas"]), 1), 4),
        }
        combo_agg[combo_name] = agg
        log(f"  {combo_name:<20s} {agg['compat_mean']:>8.4f} {agg['incompat_mean']:>8.4f} "
            f"{agg['generic_mean']:>8.4f} {agg['type_mean']:>8.4f} {agg['specific_mean']:>8.4f}")

    # 关键对比: slot+type vs type, slot+value vs value
    log(f"\n  === Key Comparisons ===")
    for base_name in ["type", "value"]:
        combo_name = f"slot+{base_name}"
        base_agg = combo_agg[base_name]
        combo_agg_data = combo_agg[combo_name]
        compat_diff = combo_agg_data["compat_mean"] - base_agg["compat_mean"]
        generic_diff = combo_agg_data["generic_mean"] - base_agg["generic_mean"]
        log(f"  {combo_name} vs {base_name}: compat_diff={compat_diff:+.4f}, generic_diff={generic_diff:+.4f}")

    results["combo_test"] = combo_agg

    # ============================================================
    # TEST 2: Temperature构式分解 (20 pairs)
    # ============================================================
    log(f"\n=== TEST 2: Temperature Construction Decomposition (20 pairs) ===")

    # 2a: Temperature专用模板
    temp_channel_stats = {ch: [] for ch in TEMP_CONSTRUCTIONS}

    for i, (obj, val) in enumerate(TEMP_CONSTRUCTION_OBJECTS):
        if (i+1) % 5 == 0:
            log(f"  temperature pair {i+1}/20 done")

        base = f"{obj} is an object"
        baseline = get_baseline_logits(model, tokenizer, device, base)
        tgt_tid = get_token_id(tokenizer, val)

        for ch_name, ch_tpl in TEMP_CONSTRUCTIONS.items():
            ch_test = ch_tpl.format(obj=obj, val=val)
            h_b = extract_rep(model, tokenizer, device, base, opt_layer)
            h_t = extract_rep(model, tokenizer, device, ch_test, opt_layer)
            d = (h_t - h_b)
            d = d / (np.linalg.norm(d) + 1e-8)

            patched = inject_and_get_logits(model, tokenizer, device, base, d, opt_layer, alpha)
            if tgt_tid is not None:
                tgt_delta = float(patched[tgt_tid] - baseline[tgt_tid])
                temp_channel_stats[ch_name].append(tgt_delta)

    temp_agg = {}
    log(f"  Temperature constructions:")
    for ch in TEMP_CONSTRUCTIONS:
        vals = temp_channel_stats[ch]
        temp_agg[ch] = {
            "tgt_mean": round(float(np.mean(vals)), 4) if vals else 0,
            "std": round(float(np.std(vals)), 4) if vals else 0,
            "n": len(vals),
        }
        log(f"    {ch}: tgt_mean={temp_agg[ch]['tgt_mean']:.4f} (n={temp_agg[ch]['n']})")

    # 2b: 非温度"to touch"对照
    log(f"\n  Non-temperature 'to touch' controls:")
    nontemp_stats = {ch: [] for ch in NONTTEMP_CONSTRUCTIONS}

    for i, obj in enumerate(NONTTEMP_OBJECTS):
        if (i+1) % 3 == 0:
            log(f"  nontemp object {i+1}/{len(NONTTEMP_OBJECTS)} done")

        base = f"{obj} is an object"
        baseline = get_baseline_logits(model, tokenizer, device, base)

        for ch_name, ch_tpl in NONTTEMP_CONSTRUCTIONS.items():
            ch_test = ch_tpl.format(obj=obj)
            h_b = extract_rep(model, tokenizer, device, base, opt_layer)
            h_t = extract_rep(model, tokenizer, device, ch_test, opt_layer)
            d = (h_t - h_b)
            d = d / (np.linalg.norm(d) + 1e-8)

            patched = inject_and_get_logits(model, tokenizer, device, base, d, opt_layer, alpha)

            # 测量对应属性词
            if "rough" in ch_name:
                target_words = ["rough", "coarse", "grainy"]
            elif "sharp" in ch_name:
                target_words = ["sharp", "pointed"]
            elif "heavy" in ch_name:
                target_words = ["heavy", "massive"]
            else:
                target_words = []

            word_deltas = []
            for w in target_words:
                tid = get_token_id(tokenizer, w)
                if tid is not None:
                    word_deltas.append(float(patched[tid] - baseline[tid]))
            if word_deltas:
                nontemp_stats[ch_name].append(float(np.mean(word_deltas)))

    nontemp_agg = {}
    for ch in NONTTEMP_CONSTRUCTIONS:
        vals = nontemp_stats[ch]
        nontemp_agg[ch] = {
            "mean": round(float(np.mean(vals)), 4) if vals else 0,
            "std": round(float(np.std(vals)), 4) if vals else 0,
            "n": len(vals),
        }
        log(f"    {ch}: mean={nontemp_agg[ch]['mean']:.4f} (n={nontemp_agg[ch]['n']})")

    results["temperature_construction"] = temp_agg
    results["nontemp_construction"] = nontemp_agg

    # ============================================================
    # TEST 3: Binding兼容等级矩阵 (52 pairs)
    # ============================================================
    log(f"\n=== TEST 3: Binding Compatibility Matrix (52 pairs) ===")

    level_stats = {1: [], 2: [], 3: [], 4: []}  # absurd, low, medium, high
    type_binding = {"color": {1: [], 2: [], 3: [], 4: []},
                    "texture": {1: [], 2: [], 3: [], 4: []},
                    "temperature": {1: [], 2: [], 3: [], 4: []},
                    "taste": {1: [], 2: [], 3: [], 4: []}}

    for i, (obj, compat, incompat, attr_type, level) in enumerate(BINDING_MATRIX):
        if (i+1) % 5 == 0:
            log(f"  binding pair {i+1}/{len(BINDING_MATRIX)} done")

        base = f"{obj} is an object"
        baseline = get_baseline_logits(model, tokenizer, device, base)

        compat_tid = get_token_id(tokenizer, compat)
        incompat_tid = get_token_id(tokenizer, incompat)
        if compat_tid is None or incompat_tid is None:
            continue

        # Type方向注入
        type_templates = {
            "color": f"{obj} has a color",
            "texture": f"{obj} has a surface feel",
            "temperature": f"{obj} has a temperature quality",
            "taste": f"{obj} has a flavor",
        }
        type_test = type_templates[attr_type]

        h_b = extract_rep(model, tokenizer, device, base, opt_layer)
        h_t = extract_rep(model, tokenizer, device, type_test, opt_layer)
        type_dir = (h_t - h_b)
        type_dir = type_dir / (np.linalg.norm(type_dir) + 1e-8)

        patched = inject_and_get_logits(model, tokenizer, device, base, type_dir, opt_layer, alpha)
        compat_d = float(patched[compat_tid] - baseline[compat_tid])
        incompat_d = float(patched[incompat_tid] - baseline[incompat_tid])
        binding = compat_d - incompat_d

        level_stats[level].append(binding)
        if attr_type in type_binding:
            type_binding[attr_type][level].append(binding)

    # Aggregate by level
    level_agg = {}
    log(f"  Binding by compatibility level (type direction):")
    level_names = {4: "high", 3: "medium", 2: "low", 1: "absurd"}
    for level in [4, 3, 2, 1]:
        vals = level_stats[level]
        agg = {
            "mean_binding": round(float(np.mean(vals)), 4) if vals else 0,
            "std": round(float(np.std(vals)), 4) if vals else 0,
            "positive_rate": round(sum(1 for v in vals if v > 0) / max(len(vals), 1), 4) if vals else 0,
            "n": len(vals),
        }
        level_agg[level_names[level]] = agg
        log(f"    {level_names[level]} (n={agg['n']}): mean={agg['mean_binding']:.4f}, "
            f"pos_rate={agg['positive_rate']:.2f}")

    # 是否单调递减? high > medium > low > absurd
    level_means = [level_agg[level_names[l]]["mean_binding"] for l in [4, 3, 2, 1]]
    is_monotone = all(level_means[i] >= level_means[i+1] for i in range(len(level_means)-1))
    log(f"  Monotone decreasing (high>=med>=low>=absurd): {is_monotone}")
    log(f"  Level means: {level_means}")

    # Aggregate by type
    type_agg = {}
    log(f"  Binding by attribute type:")
    for attr_type in ["color", "texture", "temperature", "taste"]:
        type_vals = []
        for level in [4, 3, 2, 1]:
            type_vals.extend(type_binding[attr_type][level])
        if type_vals:
            agg = {
                "mean_binding": round(float(np.mean(type_vals)), 4),
                "positive_rate": round(sum(1 for v in type_vals if v > 0) / max(len(type_vals), 1), 4),
                "n": len(type_vals),
            }
            type_agg[attr_type] = agg
            log(f"    {attr_type}: mean={agg['mean_binding']:.4f}, pos_rate={agg['positive_rate']:.2f} (n={agg['n']})")

    results["binding_matrix"] = {
        "by_level": level_agg,
        "by_type": type_agg,
        "is_monotone": is_monotone,
        "level_means": level_means,
    }

    # ============================================================
    # Save results
    # ============================================================
    out = {
        "model": model_name,
        "n_layers": cfg["n_layers"],
        "d_model": cfg["d_model"],
        "opt_layer": opt_layer,
        "alpha": alpha,
        "results": results,
    }

    out_dir = "results/phase327_combo_binding"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_name}_phase327.json"
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
