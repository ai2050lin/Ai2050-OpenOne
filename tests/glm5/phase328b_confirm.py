"""
Phase 328b: Interaction Term确认
=================================
Phase 328发现GLM4 color interaction term = 1.328(slot+value)，但Rank Gain为负。
需要确认：

1. Interaction Term是真正的binding，还是slot+value的几何交互？
   - 如果是binding，interaction应只在compat值上为正，incompat上为负或零
   - 如果是几何交互，interaction应在所有值上均匀

2. Absurd对象的interaction term是什么？
   - 如果binding真实，absurd对象的interaction应为负或弱
   - 如果只是几何效应，absurd对象也可能有强interaction

3. Value方向+slot方向的interaction vs Value方向+type方向
   - 如果只有slot+value有超叠加，说明slot是关键调制器
   - 如果type+value也有超叠加，说明只是方向叠加效应

用法:
  python tests/glm5/phase328b_confirm.py qwen3
  python tests/glm5/phase328b_confirm.py glm4
  python tests/glm5/phase328b_confirm.py deepseek7b
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

# 测试对象——包含normal和absurd
TEST_PAIRS = [
    # Normal pairs (object, compat_val, incompat_val, attr_type)
    ("apple", "red", "blue", "color"),
    ("snow", "white", "black", "color"),
    ("sky", "blue", "green", "color"),
    ("banana", "yellow", "purple", "color"),
    ("ice", "cold", "hot", "temperature"),
    ("fire", "hot", "cold", "temperature"),
    ("stone", "rough", "soft", "texture"),
    ("silk", "smooth", "rough", "texture"),
    # Absurd pairs
    ("idea", "red", "blue", "color"),
    ("music", "hot", "cold", "temperature"),
    ("theory", "rough", "smooth", "texture"),
    ("number", "green", "yellow", "color"),
    ("concept", "cold", "hot", "temperature"),
    ("event", "smooth", "rough", "texture"),
]

DIR_TEMPLATES = {
    "slot": ["{obj} has some feature", "{obj} has a property"],
    "color": ["{obj} has a color", "The color of {obj}"],
    "texture": ["{obj} has a texture", "The texture of {obj}"],
    "temperature": ["{obj} has a temperature", "The temperature of {obj}"],
}
BASE_TEMPLATE = "{obj} is an object"


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


def compute_interaction(model, tokenizer, device, opt_layer, obj, compat_val, incompat_val,
                         attr_type, dir_type1, dir_type2, alpha=1.0):
    """
    计算两种方向的interaction term。
    dir_type1, dir_type2: "slot", "type", "value"
    """
    compat_id = get_token_id(tokenizer, compat_val)
    incompat_id = get_token_id(tokenizer, incompat_val)
    prompt = "The"
    
    # 方向1
    base1 = BASE_TEMPLATE.format(obj=obj)
    if dir_type1 == "slot":
        test1 = [t.format(obj=obj) for t in DIR_TEMPLATES["slot"]]
    elif dir_type1 == "type":
        test1 = [t.format(obj=obj) for t in DIR_TEMPLATES[attr_type]]
    else:  # value
        test1 = [f"{obj} is {compat_val}", f"The {compat_val} {obj}"]
    dir1 = compute_direction(model, tokenizer, device, base1, test1, opt_layer)
    
    # 方向2
    base2 = BASE_TEMPLATE.format(obj=obj)
    if dir_type2 == "slot":
        test2 = [t.format(obj=obj) for t in DIR_TEMPLATES["slot"]]
    elif dir_type2 == "type":
        test2 = [t.format(obj=obj) for t in DIR_TEMPLATES[attr_type]]
    else:  # value
        test2 = [f"{obj} is {compat_val}", f"The {compat_val} {obj}"]
    dir2 = compute_direction(model, tokenizer, device, base2, test2, opt_layer)
    
    # 基线
    baseline = get_baseline_logits(model, tokenizer, device, prompt)
    
    # Effect(dir1 only)
    inj1 = inject_and_get_logits(model, tokenizer, device, prompt, dir1, alpha, opt_layer)
    eff1_c = float(inj1[compat_id]) - float(baseline[compat_id])
    eff1_i = float(inj1[incompat_id]) - float(baseline[incompat_id])
    
    # Effect(dir2 only)
    inj2 = inject_and_get_logits(model, tokenizer, device, prompt, dir2, alpha, opt_layer)
    eff2_c = float(inj2[compat_id]) - float(baseline[compat_id])
    eff2_i = float(inj2[incompat_id]) - float(baseline[incompat_id])
    
    # Effect(dir1 + dir2)
    inj_both = inject_multi_and_get_logits(
        model, tokenizer, device, prompt, [(dir1, alpha), (dir2, alpha)], opt_layer
    )
    eff_both_c = float(inj_both[compat_id]) - float(baseline[compat_id])
    eff_both_i = float(inj_both[incompat_id]) - float(baseline[incompat_id])
    
    # Interaction
    inter_c = eff_both_c - eff1_c - eff2_c
    inter_i = eff_both_i - eff1_i - eff2_i
    binding_inter = inter_c - inter_i
    
    return {
        "eff1_c": round(eff1_c, 4),
        "eff1_i": round(eff1_i, 4),
        "eff2_c": round(eff2_c, 4),
        "eff2_i": round(eff2_i, 4),
        "eff_both_c": round(eff_both_c, 4),
        "eff_both_i": round(eff_both_i, 4),
        "interaction_c": round(inter_c, 4),
        "interaction_i": round(inter_i, 4),
        "binding_interaction": round(binding_inter, 4),
    }


def run_all(model_name):
    log(f"Phase 328b: Interaction Term Confirmation — {model_name}")
    log("=" * 60)
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    opt_layer = MODEL_CONFIGS[model_name]["opt_layer"]
    log(f"  opt_layer={opt_layer}")
    
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    results = {}
    
    # ============================================================
    # TEST 1: slot+value interaction for normal vs absurd objects
    # ============================================================
    log("\n=== TEST 1: slot+value interaction — normal vs absurd ===")
    
    normal_interactions = []
    absurd_interactions = []
    
    for item in TEST_PAIRS:
        obj, compat, incompat, attr_type = item
        is_absurd = obj in ("idea", "music", "theory", "number", "concept", "event")
        
        r = compute_interaction(model, tokenizer, device, opt_layer, obj, compat, incompat,
                                 attr_type, "slot", "value", alpha=1.0)
        
        results[f"slot+value_{obj}"] = {
            "obj": obj, "compat": compat, "incompat": incompat,
            "attr_type": attr_type, "is_absurd": is_absurd,
            **r,
        }
        
        if is_absurd:
            absurd_interactions.append(r["binding_interaction"])
        else:
            normal_interactions.append(r["binding_interaction"])
        
        log(f"  {obj}({attr_type},{'absurd' if is_absurd else 'normal'}): "
            f"binding_inter={r['binding_interaction']:.3f}, inter_c={r['interaction_c']:.3f}")
    
    normal_mean = round(float(np.mean(normal_interactions)), 4) if normal_interactions else 0
    absurd_mean = round(float(np.mean(absurd_interactions)), 4) if absurd_interactions else 0
    
    log(f"\n  Normal objects: mean_binding_interaction = {normal_mean:.3f}, "
        f"pos_rate = {np.mean([1 if x > 0 else 0 for x in normal_interactions]):.2f}")
    log(f"  Absurd objects: mean_binding_interaction = {absurd_mean:.3f}, "
        f"pos_rate = {np.mean([1 if x > 0 else 0 for x in absurd_interactions]):.2f}")
    
    test1_summary = {
        "normal_mean": normal_mean,
        "absurd_mean": absurd_mean,
        "normal_positive_rate": round(float(np.mean([1 if x > 0 else 0 for x in normal_interactions])), 3),
        "absurd_positive_rate": round(float(np.mean([1 if x > 0 else 0 for x in absurd_interactions])), 3),
        "normal_gt_absurd": normal_mean > absurd_mean,
    }
    
    # ============================================================
    # TEST 2: type+value interaction (对比slot+value)
    # ============================================================
    log("\n=== TEST 2: type+value interaction (compare with slot+value) ===")
    
    # 只用normal color对象
    color_normal = [("apple", "red", "blue", "color"), ("snow", "white", "black", "color"),
                    ("sky", "blue", "green", "color"), ("banana", "yellow", "purple", "color")]
    
    type_value_interactions = []
    slot_value_interactions_t2 = []
    
    for obj, compat, incompat, attr_type in color_normal:
        # type+value
        r_tv = compute_interaction(model, tokenizer, device, opt_layer, obj, compat, incompat,
                                    attr_type, "type", "value", alpha=1.0)
        type_value_interactions.append(r_tv["binding_interaction"])
        
        # slot+value (重复以确认)
        r_sv = compute_interaction(model, tokenizer, device, opt_layer, obj, compat, incompat,
                                    attr_type, "slot", "value", alpha=1.0)
        slot_value_interactions_t2.append(r_sv["binding_interaction"])
        
        log(f"  {obj}: slot+value={r_sv['binding_interaction']:.3f}, type+value={r_tv['binding_interaction']:.3f}")
    
    tv_mean = round(float(np.mean(type_value_interactions)), 4)
    sv_mean = round(float(np.mean(slot_value_interactions_t2)), 4)
    
    log(f"  slot+value mean: {sv_mean:.3f}")
    log(f"  type+value mean: {tv_mean:.3f}")
    
    test2_summary = {
        "slot_value_mean": sv_mean,
        "type_value_mean": tv_mean,
        "slot_value_gt_type_value": sv_mean > tv_mean,
    }
    
    # ============================================================
    # TEST 3: interaction on compat vs incompat value separately
    # ============================================================
    log("\n=== TEST 3: interaction_c vs interaction_i breakdown ===")
    
    # 如果binding真实，interaction_c应为正（超叠加在compat值上更强）
    # 如果只是几何效应，interaction_c和interaction_i应该相近
    
    all_inter_c = []
    all_inter_i = []
    
    for obj, compat, incompat, attr_type in color_normal:
        r = results.get(f"slot+value_{obj}")
        if r:
            all_inter_c.append(r["interaction_c"])
            all_inter_i.append(r["interaction_i"])
        else:
            # 重新计算
            r_new = compute_interaction(model, tokenizer, device, opt_layer, obj, compat, incompat,
                                         attr_type, "slot", "value", alpha=1.0)
            all_inter_c.append(r_new["interaction_c"])
            all_inter_i.append(r_new["interaction_i"])
    
    mean_c = round(float(np.mean(all_inter_c)), 4) if all_inter_c else 0
    mean_i = round(float(np.mean(all_inter_i)), 4) if all_inter_i else 0
    
    log(f"  mean interaction_c (compat values): {mean_c:.3f}")
    log(f"  mean interaction_i (incompat values): {mean_i:.3f}")
    log(f"  binding = inter_c - inter_i: {mean_c - mean_i:.3f}")
    
    test3_summary = {
        "mean_interaction_c": mean_c,
        "mean_interaction_i": mean_i,
        "binding_asymmetry": round(mean_c - mean_i, 4),
        "c_positive": all(ic > 0 for ic in all_inter_c) if all_inter_c else False,
    }
    
    # 释放
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 汇总
    all_results = {
        "model": model_name,
        "opt_layer": opt_layer,
        "test1_normal_vs_absurd": test1_summary,
        "test2_slot_vs_type": test2_summary,
        "test3_asymmetry": test3_summary,
        "details": results,
    }
    
    # 转换numpy
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
    
    os.makedirs("results/phase328b_confirm", exist_ok=True)
    out_path = f"results/phase328b_confirm/{model_name}_phase328b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")
    
    # 摘要
    log("\n" + "=" * 60)
    log(f"SUMMARY — {model_name}")
    log("=" * 60)
    log(f"  Test 1: normal={test1_summary['normal_mean']:.3f}, absurd={test1_summary['absurd_mean']:.3f}, "
        f"normal>absurd={test1_summary['normal_gt_absurd']}")
    log(f"  Test 2: slot+value={test2_summary['slot_value_mean']:.3f}, type+value={test2_summary['type_value_mean']:.3f}")
    log(f"  Test 3: inter_c={test3_summary['mean_interaction_c']:.3f}, inter_i={test3_summary['mean_interaction_i']:.3f}, "
        f"asymmetry={test3_summary['binding_asymmetry']:.3f}")
    
    log(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)
    
    run_all(model_name)
    log("Phase 328b complete!")
