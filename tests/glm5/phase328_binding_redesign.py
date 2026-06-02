"""
Phase 328: Binding指标重新设计
==============================
Phase 327b暴露了binding指标的根本问题：absurd对象的binding最高，
说明测到的不是"对象-属性兼容性"，而是"对象语义空间的可推动度"。

三种新指标：
1. Rank Gain (排名增益) —— 注入后兼容属性值排名是否上升，不兼容是否下降
2. Baseline-Corrected Binding —— 减去随机对象的兼容优势，消除"易推动"偏差
3. Interaction Term —— Effect(obj+val) - Effect(obj) - Effect(val)，超叠加=binding

用法:
  python tests/glm5/phase328_binding_redesign.py qwen3
  python tests/glm5/phase328_binding_redesign.py glm4
  python tests/glm5/phase328_binding_redesign.py deepseek7b
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
        "n_layers": 36, "d_model": 2560, "opt_layer": 0,
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096, "opt_layer": 3,
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584, "opt_layer": 6,
    },
}


# ===== 对象-属性数据集 =====

# Test 1: Rank Gain - 每个对象5个候选值，按兼容度排序
RANK_DATA = {
    "color": [
        ("apple", ["red", "green", "yellow", "blue", "black"], [0,1,2]),
        ("banana", ["yellow", "green", "brown", "blue", "red"], [0,1,2]),
        ("snow", ["white", "blue", "gray", "black", "red"], [0,1,2]),
        ("sky", ["blue", "gray", "red", "green", "black"], [0,1,2]),
        ("grass", ["green", "yellow", "brown", "blue", "red"], [0,1,2]),
        ("cherry", ["red", "pink", "dark", "blue", "white"], [0,1,2]),
        ("orange", ["orange", "yellow", "green", "blue", "black"], [0,1,2]),
        ("fire", ["red", "orange", "yellow", "blue", "green"], [0,1,2]),
        ("rose", ["red", "pink", "white", "blue", "black"], [0,1,2]),
        ("leaf", ["green", "yellow", "brown", "blue", "red"], [0,1,2]),
    ],
    "texture": [
        ("stone", ["rough", "hard", "smooth", "soft", "fluffy"], [0,1,2]),
        ("silk", ["smooth", "soft", "shiny", "rough", "hard"], [0,1,2]),
        ("sand", ["grainy", "rough", "dry", "smooth", "wet"], [0,1,2]),
        ("glass", ["smooth", "hard", "cold", "rough", "soft"], [0,1,2]),
        ("velvet", ["soft", "smooth", "warm", "rough", "hard"], [0,1,2]),
        ("concrete", ["rough", "hard", "cold", "soft", "fluffy"], [0,1,2]),
        ("leather", ["smooth", "tough", "warm", "rough", "soft"], [0,1,2]),
        ("cotton", ["soft", "light", "warm", "rough", "hard"], [0,1,2]),
    ],
    "temperature": [
        ("ice", ["cold", "freezing", "cool", "hot", "warm"], [0,1,2]),
        ("fire", ["hot", "burning", "warm", "cold", "cool"], [0,1,2]),
        ("tea", ["warm", "hot", "cool", "cold", "freezing"], [0,1,2]),
        ("snow", ["cold", "freezing", "cool", "hot", "warm"], [0,1,2]),
        ("oven", ["hot", "warm", "cool", "cold", "freezing"], [0,1,2]),
        ("lava", ["hot", "burning", "molten", "cold", "cool"], [0,1,2]),
        ("breeze", ["cool", "refreshing", "warm", "hot", "freezing"], [0,1,2]),
        ("refrigerator", ["cold", "cool", "warm", "hot", "burning"], [0,1,2]),
    ],
}

# Test 2 & 3: Interaction/Baseline - 兼容/不兼容属性对
INTERACTION_PAIRS = [
    ("apple", "red", "blue", "color"),
    ("banana", "yellow", "purple", "color"),
    ("snow", "white", "black", "color"),
    ("sky", "blue", "green", "color"),
    ("ice", "cold", "hot", "temperature"),
    ("fire", "hot", "cold", "temperature"),
    ("tea", "warm", "freezing", "temperature"),
    ("stone", "rough", "soft", "texture"),
    ("silk", "smooth", "rough", "texture"),
    ("sand", "grainy", "smooth", "texture"),
    ("glass", "smooth", "rough", "texture"),
    ("lemon", "sour", "sweet", "taste"),
    ("candy", "sweet", "sour", "taste"),
    ("knife", "sharp", "soft", "texture"),
    ("pillow", "soft", "hard", "texture"),
]

# Baseline correction用的随机对照对象
BASELINE_OBJS = ["idea", "music", "theory", "number", "concept", "system", "event", "moment"]

# 方向模板
DIR_TEMPLATES = {
    "slot": ["{obj} has some feature", "{obj} has a property"],
    "color": ["{obj} has a color", "The color of {obj}"],
    "texture": ["{obj} has a texture", "The texture of {obj}"],
    "temperature": ["{obj} has a temperature", "The temperature of {obj}"],
    "taste": ["{obj} has a taste", "The taste of {obj}"],
}
BASE_TEMPLATE = "{obj} is an object"


# ===== 核心函数 =====

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


def inject_and_get_logits(model, tokenizer, device, prompt, direction, alpha, layer_idx):
    """单方向注入，返回logits"""
    layers_list = get_layers(model)
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_modified = hidden.clone()
        d_tensor = torch.tensor(direction * alpha, dtype=hidden.dtype, device=hidden.device)
        hidden_modified[0, -1, :] += d_tensor
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
    """多方向注入"""
    layers_list = get_layers(model)
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_modified = hidden.clone()
        for direction, alpha in directions_alphas:
            d_tensor = torch.tensor(alpha * direction, dtype=hidden.dtype, device=hidden.device)
            hidden_modified[0, -1, :] += d_tensor
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


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


# ===== Test 1: Rank Gain =====
def test_rank_gain(model, tokenizer, device, opt_layer, alpha=2.0):
    """
    对每个对象注入type方向，计算5个候选值在注入前后的排名变化。
    如果binding成立：compat值排名上升(rank下降)，incompat排名下降(rank上升)。
    rank_gain = baseline_rank - injected_rank (正=排名上升=好)
    """
    log("=== TEST 1: Rank Gain ===")
    results = {}
    
    for attr_type, obj_list in RANK_DATA.items():
        log(f"  {attr_type}: {len(obj_list)} objects")
        type_details = []
        
        for obj, values, compat_idx in obj_list:
            # 提取type方向
            base = BASE_TEMPLATE.format(obj=obj)
            templates = DIR_TEMPLATES.get(attr_type, DIR_TEMPLATES["color"])
            test_sents = [t.format(obj=obj) for t in templates]
            direction = compute_direction(model, tokenizer, device, base, test_sents, opt_layer)
            
            # 基线logits
            prompt = f"The {obj} is"
            baseline = get_baseline_logits(model, tokenizer, device, prompt)
            
            # 注入后logits
            injected = inject_and_get_logits(model, tokenizer, device, prompt, direction, alpha, opt_layer)
            
            # 各值在5个候选中的排名 (0=最高)
            value_ids = [get_token_id(tokenizer, v) for v in values]
            baseline_scores = [float(baseline[vid]) for vid in value_ids]
            injected_scores = [float(injected[vid]) for vid in value_ids]
            
            baseline_ranks = [int(x) for x in np.argsort(np.argsort(-np.array(baseline_scores)))]
            injected_ranks = [int(x) for x in np.argsort(np.argsort(-np.array(injected_scores)))]
            
            rank_gains = [baseline_ranks[i] - injected_ranks[i] for i in range(len(values))]
            
            compat_gain = np.mean([rank_gains[i] for i in compat_idx])
            incompat_gain = np.mean([rank_gains[i] for i in range(len(values)) if i not in compat_idx])
            
            type_details.append({
                "obj": obj,
                "values": values,
                "baseline_ranks": baseline_ranks,
                "injected_ranks": injected_ranks,
                "rank_gains": rank_gains,
                "compat_gain": round(float(compat_gain), 4),
                "incompat_gain": round(float(incompat_gain), 4),
                "net_gain": round(float(compat_gain - incompat_gain), 4),
            })
        
        compat_gains = [d["compat_gain"] for d in type_details]
        incompat_gains = [d["incompat_gain"] for d in type_details]
        net_gains = [d["net_gain"] for d in type_details]
        
        results[attr_type] = {
            "details": type_details,
            "mean_compat_gain": round(float(np.mean(compat_gains)), 4),
            "mean_incompat_gain": round(float(np.mean(incompat_gains)), 4),
            "mean_net_gain": round(float(np.mean(net_gains)), 4),
            "positive_net_rate": round(float(np.mean([1 if g > 0 else 0 for g in net_gains])), 3),
        }
        log(f"    net_gain={results[attr_type]['mean_net_gain']:.3f}, "
            f"compat={results[attr_type]['mean_compat_gain']:.3f}, "
            f"incompat={results[attr_type]['mean_incompat_gain']:.3f}, "
            f"pos_rate={results[attr_type]['positive_net_rate']:.2f}")
    
    return results


# ===== Test 2: Baseline-Corrected Binding =====
def test_baseline_corrected(model, tokenizer, device, opt_layer, alpha=2.0):
    """
    corrected_binding = binding(real_obj) - mean(binding(random_objs))
    
    binding(obj) = delta(compat) - delta(incompat)，其中delta是type方向注入后的logit变化
    
    如果corrected_binding > 0且单调(high>medium>low)，说明binding真实存在。
    """
    log("=== TEST 2: Baseline-Corrected Binding ===")
    details = {}
    
    for obj, compat_val, incompat_val, attr_type in INTERACTION_PAIRS:
        compat_id = get_token_id(tokenizer, compat_val)
        incompat_id = get_token_id(tokenizer, incompat_val)
        
        # 真实对象的type方向
        base = BASE_TEMPLATE.format(obj=obj)
        templates = DIR_TEMPLATES.get(attr_type, DIR_TEMPLATES["color"])
        test_sents = [t.format(obj=obj) for t in templates]
        direction = compute_direction(model, tokenizer, device, base, test_sents, opt_layer)
        
        prompt = f"The {obj} is"
        baseline = get_baseline_logits(model, tokenizer, device, prompt)
        injected = inject_and_get_logits(model, tokenizer, device, prompt, direction, alpha, opt_layer)
        
        delta_compat = float(injected[compat_id]) - float(baseline[compat_id])
        delta_incompat = float(injected[incompat_id]) - float(baseline[incompat_id])
        raw_binding = delta_compat - delta_incompat
        
        # Baseline: 用4个random对象的binding做校正
        baseline_bindings = []
        for rand_obj in BASELINE_OBJS[:4]:
            rand_base = BASE_TEMPLATE.format(obj=rand_obj)
            rand_test = [t.format(obj=rand_obj) for t in templates]
            rand_dir = compute_direction(model, tokenizer, device, rand_base, rand_test, opt_layer)
            
            rand_prompt = f"The {rand_obj} is"
            rand_bl = get_baseline_logits(model, tokenizer, device, rand_prompt)
            rand_inj = inject_and_get_logits(model, tokenizer, device, rand_prompt, rand_dir, alpha, opt_layer)
            
            dc = float(rand_inj[compat_id]) - float(rand_bl[compat_id])
            di = float(rand_inj[incompat_id]) - float(rand_bl[incompat_id])
            baseline_bindings.append(dc - di)
        
        avg_baseline = float(np.mean(baseline_bindings))
        corrected = raw_binding - avg_baseline
        
        details[obj] = {
            "attr_type": attr_type,
            "compat_val": compat_val,
            "incompat_val": incompat_val,
            "delta_compat": round(delta_compat, 4),
            "delta_incompat": round(delta_incompat, 4),
            "raw_binding": round(raw_binding, 4),
            "avg_baseline_binding": round(avg_baseline, 4),
            "corrected_binding": round(corrected, 4),
        }
        
        log(f"    {obj}({attr_type}): raw={raw_binding:.3f}, baseline={avg_baseline:.3f}, corrected={corrected:.3f}")
    
    # 按类型汇总
    type_summary = defaultdict(lambda: {"corrected": [], "raw": []})
    for obj, r in details.items():
        type_summary[r["attr_type"]]["corrected"].append(r["corrected_binding"])
        type_summary[r["attr_type"]]["raw"].append(r["raw_binding"])
    
    summary = {}
    for atype, vals in type_summary.items():
        summary[atype] = {
            "mean_corrected": round(float(np.mean(vals["corrected"])), 4),
            "mean_raw": round(float(np.mean(vals["raw"])), 4),
            "corrected_positive_rate": round(float(np.mean([1 if v > 0 else 0 for v in vals["corrected"]])), 3),
            "n": len(vals["corrected"]),
        }
        log(f"  {atype}: corrected={summary[atype]['mean_corrected']:.3f}, "
            f"raw={summary[atype]['mean_raw']:.3f}, "
            f"pos_rate={summary[atype]['corrected_positive_rate']:.2f}")
    
    return {"details": details, "type_summary": summary}


# ===== Test 3: Interaction Term =====
def test_interaction_term(model, tokenizer, device, opt_layer, alpha=1.0):
    """
    Interaction = Effect(obj_dir + val_dir) - Effect(obj_dir) - Effect(val_dir)
    
    Effect = 注入某方向后目标词logit - 基线logit
    目标词 = compat_val
    
    如果interaction > 0: obj和val有超叠加 = binding存在
    如果interaction ≈ 0: obj和val独立 = 无binding
    """
    log("=== TEST 3: Interaction Term ===")
    details = {}
    
    prompt = "The"  # 中性提示
    
    for obj, compat_val, incompat_val, attr_type in INTERACTION_PAIRS:
        compat_id = get_token_id(tokenizer, compat_val)
        incompat_id = get_token_id(tokenizer, incompat_val)
        
        # obj方向: slot方向（泛属性空间）
        obj_base = BASE_TEMPLATE.format(obj=obj)
        obj_test = [t.format(obj=obj) for t in DIR_TEMPLATES["slot"]]
        obj_dir = compute_direction(model, tokenizer, device, obj_base, obj_test, opt_layer)
        
        # val方向: 具体属性值
        val_base = f"{obj} is an object"
        val_test = [f"{obj} is {compat_val}", f"The {compat_val} {obj}"]
        val_dir = compute_direction(model, tokenizer, device, val_base, val_test, opt_layer)
        
        # 基线logits
        baseline = get_baseline_logits(model, tokenizer, device, prompt)
        
        # Effect(obj_dir only)
        inj_obj = inject_and_get_logits(model, tokenizer, device, prompt, obj_dir, alpha, opt_layer)
        effect_obj_c = float(inj_obj[compat_id]) - float(baseline[compat_id])
        effect_obj_i = float(inj_obj[incompat_id]) - float(baseline[incompat_id])
        
        # Effect(val_dir only)
        inj_val = inject_and_get_logits(model, tokenizer, device, prompt, val_dir, alpha, opt_layer)
        effect_val_c = float(inj_val[compat_id]) - float(baseline[compat_id])
        effect_val_i = float(inj_val[incompat_id]) - float(baseline[incompat_id])
        
        # Effect(obj_dir + val_dir)
        inj_both = inject_multi_and_get_logits(
            model, tokenizer, device, prompt, [(obj_dir, alpha), (val_dir, alpha)], opt_layer
        )
        effect_both_c = float(inj_both[compat_id]) - float(baseline[compat_id])
        effect_both_i = float(inj_both[incompat_id]) - float(baseline[incompat_id])
        
        # Interaction term
        interaction_c = effect_both_c - effect_obj_c - effect_val_c
        interaction_i = effect_both_i - effect_obj_i - effect_val_i
        binding_interaction = interaction_c - interaction_i
        
        details[obj] = {
            "attr_type": attr_type,
            "compat_val": compat_val,
            "incompat_val": incompat_val,
            "effect_obj_c": round(effect_obj_c, 4),
            "effect_obj_i": round(effect_obj_i, 4),
            "effect_val_c": round(effect_val_c, 4),
            "effect_val_i": round(effect_val_i, 4),
            "effect_both_c": round(effect_both_c, 4),
            "effect_both_i": round(effect_both_i, 4),
            "interaction_c": round(interaction_c, 4),
            "interaction_i": round(interaction_i, 4),
            "binding_interaction": round(binding_interaction, 4),
        }
        
        log(f"    {obj}({attr_type}): interaction_c={interaction_c:.3f}, binding_inter={binding_interaction:.3f}")
    
    # 按类型汇总
    type_summary = defaultdict(lambda: {"binding_interactions": [], "interaction_cs": []})
    for obj, r in details.items():
        type_summary[r["attr_type"]]["binding_interactions"].append(r["binding_interaction"])
        type_summary[r["attr_type"]]["interaction_cs"].append(r["interaction_c"])
    
    summary = {}
    for atype, vals in type_summary.items():
        summary[atype] = {
            "mean_binding_interaction": round(float(np.mean(vals["binding_interactions"])), 4),
            "mean_interaction_c": round(float(np.mean(vals["interaction_cs"])), 4),
            "positive_rate": round(float(np.mean([1 if v > 0 else 0 for v in vals["binding_interactions"]])), 3),
            "n": len(vals["binding_interactions"]),
        }
        log(f"  {atype}: binding_interaction={summary[atype]['mean_binding_interaction']:.3f}, "
            f"pos_rate={summary[atype]['positive_rate']:.2f}")
    
    return {"details": details, "type_summary": summary}


# ===== Main =====

def run_all(model_name):
    log(f"Phase 328: Binding Redesign — {model_name}")
    log("=" * 60)
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    opt_layer = MODEL_CONFIGS[model_name]["opt_layer"]
    log(f"  opt_layer={opt_layer}, load_time={time.time()-t0:.1f}s")
    
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # Test 1: Rank Gain
    t1 = time.time()
    rank_results = test_rank_gain(model, tokenizer, device, opt_layer, alpha=2.0)
    log(f"Test 1 done in {time.time()-t1:.1f}s")
    
    if torch.cuda.is_available():
        log(f"  GPU after T1: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # Test 2: Baseline-Corrected Binding
    t2 = time.time()
    corrected_results = test_baseline_corrected(model, tokenizer, device, opt_layer, alpha=2.0)
    log(f"Test 2 done in {time.time()-t2:.1f}s")
    
    if torch.cuda.is_available():
        log(f"  GPU after T2: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # Test 3: Interaction Term
    t3 = time.time()
    interaction_results = test_interaction_term(model, tokenizer, device, opt_layer, alpha=1.0)
    log(f"Test 3 done in {time.time()-t3:.1f}s")
    
    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 保存
    all_results = {
        "model": model_name,
        "opt_layer": opt_layer,
        "rank_gain": rank_results,
        "baseline_corrected": corrected_results,
        "interaction_term": interaction_results,
    }
    
    # 转换numpy类型为Python原生类型
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
    
    os.makedirs("results/phase328_binding_redesign", exist_ok=True)
    out_path = f"results/phase328_binding_redesign/{model_name}_phase328.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")
    
    # 摘要
    log("\n" + "=" * 60)
    log("SUMMARY — " + model_name)
    log("=" * 60)
    
    log("\n[1] Rank Gain (type方向注入, alpha=2.0)")
    for atype, r in rank_results.items():
        log(f"  {atype}: net_gain={r['mean_net_gain']:.3f}, "
            f"compat={r['mean_compat_gain']:.3f}, incompat={r['mean_incompat_gain']:.3f}, "
            f"pos_rate={r['positive_net_rate']:.2f}")
    
    log("\n[2] Baseline-Corrected Binding (type方向注入, alpha=2.0)")
    for atype, r in corrected_results["type_summary"].items():
        log(f"  {atype}: corrected={r['mean_corrected']:.3f}, raw={r['mean_raw']:.3f}, "
            f"pos_rate={r['corrected_positive_rate']:.2f}, n={r['n']}")
    
    log("\n[3] Interaction Term (alpha=1.0)")
    for atype, r in interaction_results["type_summary"].items():
        log(f"  {atype}: binding_interaction={r['mean_binding_interaction']:.3f}, "
            f"pos_rate={r['positive_rate']:.2f}, n={r['n']}")
    
    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.1f}s")
    
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}. Use: qwen3, glm4, deepseek7b")
        sys.exit(1)
    
    run_all(model_name)
    log("Phase 328 complete!")
