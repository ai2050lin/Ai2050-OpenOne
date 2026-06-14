"""
Phase 496: 真正因果MLP/Attn干预 + MLP子模块分解 + 多Token位置
================================================================

Phase 495遗留核心瓶颈:
1. Exp2是静态分解(只消融shared分量)，不是真正因果干预(阻断整个MLP/Attn)
2. MLP翻转是"守门模块"但不知道MLP内部哪个子模块执行翻转
3. 不知道跨层释放主要发生在哪个token位置
4. DS7B animal异常未解释

Phase 496核心实验:
- Exp1: 真正因果MLP/Attn干预 — 在L(n-1)完全阻断MLP或Attn输出后forward
  → 直接验证"MLP翻转"是否是因果机制
- Exp2: MLP子模块分解 — 分别干预gate_proj/up_proj/down_proj输出
  → 找到执行符号翻转的MLP子模块
- Exp3: 多Token位置干预 — 在object/relation/last/all位置干预
  → 找到跨层释放的关键token位置
- Exp4: DS7B animal异常机制 — 竞争类别+生命属性+动作方向分析

用法:
  python tests/glm5/phase496_causal_mlp_attn_intervention.py qwen3 1
  python tests/glm5/phase496_causal_mlp_attn_intervention.py glm4 1
  python tests/glm5/phase496_causal_mlp_attn_intervention.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)


def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES_16 = {
    "fruit":     ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "plum",
                  "cherry", "lemon", "lime", "kiwi", "melon", "fig", "date", "guava"],
    "animal":    ["dog", "cat", "horse", "lion", "bear", "rabbit", "eagle", "fish",
                  "tiger", "deer", "wolf", "fox", "owl", "hawk", "crab", "frog"],
    "clothing":  ["shirt", "dress", "hat", "coat", "jacket", "skirt", "scarf", "boot",
                  "pants", "vest", "glove", "sock", "belt", "tie", "cap", "robe"],
    "food":      ["bread", "rice", "cheese", "pasta", "soup", "cake", "salad", "meat",
                  "pizza", "taco", "stew", "pie", "roll", "ham", "corn", "bean"],
    "vehicle":   ["car", "bus", "bicycle", "truck", "train", "plane", "boat", "motorcycle",
                  "van", "scooter", "taxi", "tram", "ship", "jet", "subway", "sled"],
    "plant":     ["tree", "flower", "grass", "bush", "fern", "moss", "vine", "shrub",
                  "palm", "oak", "pine", "rose", "lily", "weed", "reed", "cactus"],
    "tool":      ["hammer", "saw", "drill", "wrench", "pliers", "chisel", "ruler", "knife",
                  "screwdriver", "axe", "shovel", "mallet", "clamp", "vise", "file", "level"],
    "furniture": ["chair", "table", "desk", "bed", "sofa", "shelf", "cabinet", "bench",
                  "stool", "dresser", "wardrobe", "couch", "hutch", "ottoman", "cot", "armoire"],
    "body_part":   ["hand", "foot", "head", "arm", "leg", "eye", "ear", "nose",
                    "mouth", "finger", "toe", "knee", "elbow", "wrist", "ankle", "hip"],
    "building":    ["house", "church", "school", "store", "factory", "tower", "bridge", "castle",
                    "barn", "temple", "museum", "hotel", "cabin", "palace", "fort", "shrine"],
    "container":   ["box", "bottle", "cup", "jar", "bowl", "basket", "bucket", "pot",
                    "can", "tub", "mug", "flask", "vase", "crate", "barrel", "jug"],
    "device":      ["phone", "radio", "clock", "lamp", "camera", "speaker", "monitor", "printer",
                    "fan", "heater", "scale", "timer", "scanner", "router", "switch", "pager"],
    "place":       ["park", "beach", "forest", "lake", "mountain", "river", "desert", "island",
                    "valley", "cliff", "cave", "swamp", "meadow", "canyon", "bay", "glacier"],
    "material":    ["wood", "stone", "metal", "glass", "paper", "cloth", "rubber", "leather",
                    "plastic", "cotton", "silk", "wool", "clay", "sand", "gold", "iron"],
    "emotion":     ["joy", "fear", "anger", "sadness", "hope", "love", "pride", "shame",
                    "guilt", "envy", "pity", "awe", "grief", "bliss", "rage", "dread"],
    "action":      ["run", "walk", "jump", "swim", "climb", "throw", "catch", "push",
                    "pull", "lift", "carry", "drag", "drop", "kick", "hit", "hold"],
}

CAT_NAMES_16 = list(CATEGORIES_16.keys())


def get_model_and_tokenizer(model_name):
    """BF16加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        plog(f"{model_name}: GPU={gpu_count}, CPU={cpu_count}")

    device = next(model.parameters()).device
    return model, tokenizer, device


def encode_prompts(tokenizer, device, objects, template="The {obj} is a kind of"):
    texts = [template.format(obj=obj) for obj in objects]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }


def compute_target_logit(h, W_U, cat_name, cat_names):
    logit = W_U @ h
    return float(logit[cat_names.index(cat_name)])


def get_shared_direction(h_target, h_others):
    """计算shared_semantic方向"""
    others_mean = np.mean(h_others, axis=0)
    B_c = h_target - others_mean
    Bc_norm_dir = B_c / (np.linalg.norm(B_c) + 1e-10)

    all_mean = np.mean(np.vstack([h_target[np.newaxis], h_others]), axis=0)
    shared_dir = all_mean / (np.linalg.norm(all_mean) + 1e-10)
    shared_dir = shared_dir - np.dot(shared_dir, Bc_norm_dir) * Bc_norm_dir
    if np.linalg.norm(shared_dir) > 1e-6:
        shared_dir = shared_dir / np.linalg.norm(shared_dir)

    return shared_dir, Bc_norm_dir


def get_layer_hidden(model, tokenizer, device, objects, layer_idx,
                     template="The {obj} is a kind of"):
    """获取某层所有样本hidden states"""
    layers = get_layers(model)
    inputs = encode_prompts(tokenizer, device, objects, template)

    captured = {}
    def make_hook(key):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook

    hook = layers[layer_idx].register_forward_hook(make_hook("h"))
    with torch.no_grad():
        model(**inputs)
    hook.remove()

    h = captured["h"]
    mask = inputs["attention_mask"].bool()
    last_idx = mask.sum(dim=1) - 1
    h_last = h[torch.arange(h.size(0)), last_idx.cpu()]
    return h_last.numpy(), inputs, last_idx


def forward_with_intervention(model, tokenizer, device, single_obj,
                              intervene_layer, intervene_fn,
                              capture_layers=None,
                              template="The {obj} is a kind of"):
    """
    通用因果干预forward
    
    intervene_fn: (module, input, output) -> modified_output
    capture_layers: list of layer indices to capture
    """
    layers = get_layers(model)
    inputs = encode_prompts(tokenizer, device, [single_obj], template)
    mask = inputs["attention_mask"].bool()
    seq_pos = int(mask[0].sum() - 1)

    captured = {}
    hooks = []

    # 干预hook
    hooks.append(layers[intervene_layer].register_forward_hook(intervene_fn))

    # 捕获hooks
    if capture_layers:
        for li in capture_layers:
            def make_cap_hook(key):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook
            hooks.append(layers[li].register_forward_hook(make_cap_hook(f"L{li}")))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return captured, seq_pos, inputs


def make_zero_output_hook():
    """创建将模块输出归零的hook（保留tuple结构）"""
    def hook(module, inp, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        else:
            return torch.zeros_like(output)
    return hook


def make_scale_output_hook(scale):
    """创建缩放模块输出的hook"""
    def hook(module, inp, output):
        if isinstance(output, tuple):
            return (output[0] * scale,) + output[1:]
        else:
            return output * scale
    return hook


# ==================== Exp1: 真正因果MLP/Attn干预 ====================
def exp1_causal_mlp_attn(model, tokenizer, device, model_name, W_U):
    """
    核心实验: 在L(n-1)完全阻断MLP或Attn输出后forward
    
    对比三种条件:
    1. Baseline: 自然forward
    2. Zero MLP: 阻断L(n-1)的MLP输出
    3. Zero Attn: 阻断L(n-1)的Attn输出
    
    如果MLP执行符号翻转:
    - Zero MLP → 释放类不再释放(shared方向对目标logit回到刹车方向)
    - Zero Attn → 释放类仍然释放(或略有减弱)
    """
    plog(f"=== Exp1: True Causal MLP/Attn Intervention at L(n-1) ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n1 = n_layers - 1
    L_n2 = n_layers - 2
    layers = get_layers(model)

    # 选关键类别: 2个释放类 + 2个刹车类/异常类
    test_cats = ["fruit", "clothing", "emotion", "action"]

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()

        cat_objs = CATEGORIES_16[cat_name][:10]  # 10样本
        other_cats = [c for c in CAT_NAMES_16 if c != cat_name][:7]

        # Step 1: Baseline - 获取L(n-1)和L(n-2)的hidden
        h_n1_all, _, _ = get_layer_hidden(model, tokenizer, device, cat_objs, L_n1)
        h_n2_all, _, _ = get_layer_hidden(model, tokenizer, device, cat_objs, L_n2)
        h_n1_mean = np.mean(h_n1_all, axis=0)

        # 获取other categories的hidden
        other_h_n1 = {}
        for oc in other_cats:
            oc_h, _, _ = get_layer_hidden(model, tokenizer, device, CATEGORIES_16[oc][:8], L_n1)
            other_h_n1[oc] = oc_h
        h_others_n1 = np.array([np.mean(other_h_n1[oc], axis=0) for oc in other_cats])
        shared_dir_n1, _ = get_shared_direction(h_n1_mean, h_others_n1)

        # Step 2: 对每个样本做三种forward
        sample_results = []
        for s_idx, obj in enumerate(cat_objs):
            plog(f"    Sample {s_idx}/{len(cat_objs)}: {obj}")
            # 2a. Baseline forward with capture at L(n-1)
            captured_base, seq_pos, inputs = forward_with_intervention(
                model, tokenizer, device, obj,
                intervene_layer=L_n2,  # 无干预，只捕获
                intervene_fn=lambda m, i, o: o,  # pass-through
                capture_layers=[L_n1, L_n2],
            )

            if f"L{L_n1}" not in captured_base or f"L{L_n2}" not in captured_base:
                plog(f"    WARNING: capture failed for sample {s_idx}")
                continue

            h_n1_base = captured_base[f"L{L_n1}"][0, seq_pos, :].numpy()
            h_n2_base = captured_base[f"L{L_n2}"][0, seq_pos, :].numpy()

            D_base = compute_target_logit(h_n1_base, W_U, cat_name, CAT_NAMES_16)
            proj_shared_base = float(np.dot(h_n1_base, shared_dir_n1))
            contrib_shared_base = float(proj_shared_base * np.dot(shared_dir_n1, W_U[CAT_NAMES_16.index(cat_name)]))

            # 2b. Zero MLP forward
            # 需要找到MLP模块在layer中的位置
            layer = layers[L_n1]
            if hasattr(layer, 'mlp'):
                mlp_mod = layer.mlp
            elif hasattr(layer, 'feed_forward'):
                mlp_mod = layer.feed_forward
            else:
                plog(f"    WARNING: no MLP found at L{L_n1}")
                continue

            zero_mlp_hook = make_zero_output_hook()
            captured_zmlp, _, _ = forward_with_intervention(
                model, tokenizer, device, obj,
                intervene_layer=L_n2,
                intervene_fn=lambda m, i, o: o,
                capture_layers=[L_n1],
            )
            # 需要在MLP上注册hook而不是layer上
            # 重新实现：直接在MLP和Attn上注册hook
            inputs_single = encode_prompts(tokenizer, device, [obj])
            mask_single = inputs_single["attention_mask"].bool()
            sp = int(mask_single[0].sum() - 1)

            # Baseline re-capture (同时捕获L(n-1))
            captured = {}
            def make_cap(key):
                def hook(mod, inp, out):
                    if isinstance(out, tuple):
                        captured[key] = out[0].detach().float().cpu()
                    else:
                        captured[key] = out.detach().float().cpu()
                return hook

            h_cap = layers[L_n1].register_forward_hook(make_cap("Ln1"))
            with torch.no_grad():
                model(**inputs_single)
            h_cap.remove()

            if "Ln1" not in captured:
                continue
            h_n1_baseline = captured["Ln1"][0, sp, :].numpy()
            D_baseline = compute_target_logit(h_n1_baseline, W_U, cat_name, CAT_NAMES_16)

            # Zero MLP forward
            captured_mlp = {}
            mlp_hook = mlp_mod.register_forward_hook(make_zero_output_hook())
            cap_mlp = layers[L_n1].register_forward_hook(make_cap("Ln1"))
            with torch.no_grad():
                model(**inputs_single)
            mlp_hook.remove()
            cap_mlp.remove()

            if "Ln1" not in captured_mlp:
                # captured在lambda中共享，需要重新组织
                pass
            # Actually, the captured dict is shared. Let me restructure.

            # Let me redo this more carefully
            break  # exit sample loop, restructure

        # Re-structured approach: batch per sample with explicit captures
        sample_data = []
        for s_idx, obj in enumerate(cat_objs):
            plog(f"    Sample {s_idx}/{len(cat_objs)}: {obj}")
            inputs_single = encode_prompts(tokenizer, device, [obj])
            mask_single = inputs_single["attention_mask"].bool()
            sp = int(mask_single[0].sum() - 1)

            layer_n1 = layers[L_n1]

            # Identify sub-modules
            mlp_mod = getattr(layer_n1, 'mlp', None) or getattr(layer_n1, 'feed_forward', None)
            attn_mod = getattr(layer_n1, 'self_attn', None)

            if mlp_mod is None or attn_mod is None:
                plog(f"    WARNING: MLP or Attn not found at L{L_n1}")
                continue

            # --- Baseline ---
            captured = {}
            hooks = []
            hooks.append(layers[L_n1].register_forward_hook(make_cap_hook("Ln1")))
            with torch.no_grad():
                model(**inputs_single)
            for h in hooks:
                h.remove()
            if "Ln1" not in captured:
                continue
            h_n1_bl = captured["Ln1"][0, sp, :].numpy()
            D_bl = compute_target_logit(h_n1_bl, W_U, cat_name, CAT_NAMES_16)
            proj_bl = float(np.dot(h_n1_bl, shared_dir_n1))

            # --- Zero MLP ---
            captured_z = {}
            hooks_z = []
            hooks_z.append(mlp_mod.register_forward_hook(make_zero_output_hook()))
            hooks_z.append(layers[L_n1].register_forward_hook(make_cap_hook("Ln1")))
            with torch.no_grad():
                model(**inputs_single)
            for h in hooks_z:
                h.remove()
            if "Ln1" not in captured_z:
                continue
            h_n1_zmlp = captured_z["Ln1"][0, sp, :].numpy()
            D_zmlp = compute_target_logit(h_n1_zmlp, W_U, cat_name, CAT_NAMES_16)
            proj_zmlp = float(np.dot(h_n1_zmlp, shared_dir_n1))

            # --- Zero Attn ---
            captured_a = {}
            hooks_a = []
            hooks_a.append(attn_mod.register_forward_hook(make_zero_output_hook()))
            hooks_a.append(layers[L_n1].register_forward_hook(make_cap_hook("Ln1")))
            with torch.no_grad():
                model(**inputs_single)
            for h in hooks_a:
                h.remove()
            if "Ln1" not in captured_a:
                continue
            h_n1_zattn = captured_a["Ln1"][0, sp, :].numpy()
            D_zattn = compute_target_logit(h_n1_zattn, W_U, cat_name, CAT_NAMES_16)
            proj_zattn = float(np.dot(h_n1_zattn, shared_dir_n1))

            sample_data.append({
                "obj": obj,
                "D_baseline": D_bl,
                "D_zero_mlp": D_zmlp,
                "D_zero_attn": D_zattn,
                "delta_D_zero_mlp": D_zmlp - D_bl,
                "delta_D_zero_attn": D_zattn - D_bl,
                "proj_shared_baseline": proj_bl,
                "proj_shared_zero_mlp": proj_zmlp,
                "proj_shared_zero_attn": proj_zattn,
            })

        if not sample_data:
            results[cat_name] = {"error": "no valid samples"}
            continue

        # Aggregate
        mean_delta_zmlp = np.mean([s["delta_D_zero_mlp"] for s in sample_data])
        mean_delta_zattn = np.mean([s["delta_D_zero_attn"] for s in sample_data])
        mean_proj_bl = np.mean([s["proj_shared_baseline"] for s in sample_data])
        mean_proj_zmlp = np.mean([s["proj_shared_zero_mlp"] for s in sample_data])
        mean_proj_zattn = np.mean([s["proj_shared_zero_attn"] for s in sample_data])

        results[cat_name] = {
            "n_samples": len(sample_data),
            "mean_delta_D_zero_mlp": mean_delta_zmlp,
            "mean_delta_D_zero_attn": mean_delta_zattn,
            "mean_proj_shared_baseline": mean_proj_bl,
            "mean_proj_shared_zero_mlp": mean_proj_zmlp,
            "mean_proj_shared_zero_attn": mean_proj_zattn,
            "mlp_flip_verdict": "MLP produces release" if mean_delta_zmlp < -5 else "MLP produces brake" if mean_delta_zmlp > 5 else "MLP neutral",
            "attn_flip_verdict": "Attn produces release" if mean_delta_zattn < -5 else "Attn produces brake" if mean_delta_zattn > 5 else "Attn neutral",
            "sample_details": sample_data,
            "elapsed": time.time() - t0,
        }
        plog(f"  {cat_name}: ΔD(zeroMLP)={mean_delta_zmlp:.2f}, ΔD(zeroAttn)={mean_delta_zattn:.2f}, "
             f"proj(bl/mlp/attn)={mean_proj_bl:.1f}/{mean_proj_zmlp:.1f}/{mean_proj_zattn:.1f}")

    return results


def make_cap_hook(key):
    """创建捕获输出的hook"""
    _captured = {}
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            _captured[key] = out[0].detach().float().cpu()
        else:
            _captured[key] = out.detach().float().cpu()
    hook.captured = _captured
    return hook


# 修正Exp1: 使用闭包捕获字典
def exp1_causal_mlp_attn_v2(model, tokenizer, device, model_name, W_U):
    """
    核心实验: 在L(n-1)完全阻断MLP或Attn输出后forward
    
    对比三种条件:
    1. Baseline: 自然forward
    2. Zero MLP: 阻断L(n-1)的MLP输出
    3. Zero Attn: 阻断L(n-1)的Attn输出
    
    如果MLP执行符号翻转:
    - Zero MLP → 目标logit应该显著变化（回到L(n-2)的方向）
    - Zero Attn → 目标logit变化较小
    """
    plog(f"=== Exp1: True Causal MLP/Attn Intervention at L(n-1) ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n1 = n_layers - 1
    L_n2 = n_layers - 2
    layers = get_layers(model)

    test_cats = ["fruit", "clothing", "emotion", "action"]

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()

        cat_objs = CATEGORIES_16[cat_name][:10]
        other_cats = [c for c in CAT_NAMES_16 if c != cat_name][:7]

        # 获取shared方向
        h_n1_all, _, _ = get_layer_hidden(model, tokenizer, device, cat_objs, L_n1)
        h_n1_mean = np.mean(h_n1_all, axis=0)
        other_h_n1 = {}
        for oc in other_cats:
            oc_h, _, _ = get_layer_hidden(model, tokenizer, device, CATEGORIES_16[oc][:8], L_n1)
            other_h_n1[oc] = oc_h
        h_others_n1 = np.array([np.mean(other_h_n1[oc], axis=0) for oc in other_cats])
        shared_dir_n1, _ = get_shared_direction(h_n1_mean, h_others_n1)

        layer_n1 = layers[L_n1]
        mlp_mod = getattr(layer_n1, 'mlp', None) or getattr(layer_n1, 'feed_forward', None)
        attn_mod = getattr(layer_n1, 'self_attn', None)

        if mlp_mod is None or attn_mod is None:
            plog(f"    WARNING: MLP or Attn not found at L{L_n1}")
            continue

        sample_data = []
        for s_idx, obj in enumerate(cat_objs):
            plog(f"    Sample {s_idx}/{len(cat_objs)}: {obj}")
            inputs_single = encode_prompts(tokenizer, device, [obj])
            mask_single = inputs_single["attention_mask"].bool()
            sp = int(mask_single[0].sum() - 1)

            # --- Baseline ---
            cap_dict = {}
            def make_capture(key, store):
                def hook(mod, inp, out):
                    if isinstance(out, tuple):
                        store[key] = out[0].detach().float().cpu()
                    else:
                        store[key] = out.detach().float().cpu()
                return hook

            hooks = [layers[L_n1].register_forward_hook(make_capture("Ln1", cap_dict))]
            with torch.no_grad():
                model(**inputs_single)
            for h in hooks:
                h.remove()

            if "Ln1" not in cap_dict:
                continue
            h_n1_bl = cap_dict["Ln1"][0, sp, :].numpy()
            D_bl = compute_target_logit(h_n1_bl, W_U, cat_name, CAT_NAMES_16)
            proj_bl = float(np.dot(h_n1_bl, shared_dir_n1))

            # --- Zero MLP ---
            cap_dict_mlp = {}
            hooks_mlp = [
                mlp_mod.register_forward_hook(make_zero_output_hook()),
                layers[L_n1].register_forward_hook(make_capture("Ln1", cap_dict_mlp)),
            ]
            with torch.no_grad():
                model(**inputs_single)
            for h in hooks_mlp:
                h.remove()

            if "Ln1" not in cap_dict_mlp:
                continue
            h_n1_zmlp = cap_dict_mlp["Ln1"][0, sp, :].numpy()
            D_zmlp = compute_target_logit(h_n1_zmlp, W_U, cat_name, CAT_NAMES_16)
            proj_zmlp = float(np.dot(h_n1_zmlp, shared_dir_n1))

            # --- Zero Attn ---
            cap_dict_attn = {}
            hooks_attn = [
                attn_mod.register_forward_hook(make_zero_output_hook()),
                layers[L_n1].register_forward_hook(make_capture("Ln1", cap_dict_attn)),
            ]
            with torch.no_grad():
                model(**inputs_single)
            for h in hooks_attn:
                h.remove()

            if "Ln1" not in cap_dict_attn:
                continue
            h_n1_zattn = cap_dict_attn["Ln1"][0, sp, :].numpy()
            D_zattn = compute_target_logit(h_n1_zattn, W_U, cat_name, CAT_NAMES_16)
            proj_zattn = float(np.dot(h_n1_zattn, shared_dir_n1))

            sample_data.append({
                "obj": obj,
                "D_baseline": round(D_bl, 4),
                "D_zero_mlp": round(D_zmlp, 4),
                "D_zero_attn": round(D_zattn, 4),
                "delta_D_zero_mlp": round(D_zmlp - D_bl, 4),
                "delta_D_zero_attn": round(D_zattn - D_bl, 4),
                "proj_shared_baseline": round(proj_bl, 2),
                "proj_shared_zero_mlp": round(proj_zmlp, 2),
                "proj_shared_zero_attn": round(proj_zattn, 2),
            })

        if not sample_data:
            results[cat_name] = {"error": "no valid samples"}
            continue

        mean_delta_zmlp = np.mean([s["delta_D_zero_mlp"] for s in sample_data])
        mean_delta_zattn = np.mean([s["delta_D_zero_attn"] for s in sample_data])
        mean_proj_bl = np.mean([s["proj_shared_baseline"] for s in sample_data])
        mean_proj_zmlp = np.mean([s["proj_shared_zero_mlp"] for s in sample_data])
        mean_proj_zattn = np.mean([s["proj_shared_zero_attn"] for s in sample_data])

        results[cat_name] = {
            "n_samples": len(sample_data),
            "mean_delta_D_zero_mlp": round(float(mean_delta_zmlp), 4),
            "mean_delta_D_zero_attn": round(float(mean_delta_zattn), 4),
            "mean_proj_shared_baseline": round(float(mean_proj_bl), 2),
            "mean_proj_shared_zero_mlp": round(float(mean_proj_zmlp), 2),
            "mean_proj_shared_zero_attn": round(float(mean_proj_zattn), 2),
            "mlp_role": "release" if mean_delta_zmlp < -5 else "brake" if mean_delta_zmlp > 5 else "neutral",
            "attn_role": "release" if mean_delta_zattn < -5 else "brake" if mean_delta_zattn > 5 else "neutral",
            "sample_details": sample_data,
            "elapsed": round(time.time() - t0, 2),
        }
        plog(f"  {cat_name}: ΔD(zeroMLP)={mean_delta_zmlp:.2f}, ΔD(zeroAttn)={mean_delta_zattn:.2f}")

    return results


# ==================== Exp2: MLP子模块分解 ====================
def exp2_mlp_submodule(model, tokenizer, device, model_name, W_U):
    """
    在L(n-1)分别干预MLP的gate_proj/up_proj/down_proj输出
    
    对于split_gate_up(Qwen3/DS7B): gate_proj, up_proj, down_proj
    对于merged_gate_up(GLM4): gate_up_proj, down_proj
    
    方法: 在down_proj的输入端注入/消融特定方向的激活
    """
    plog(f"=== Exp2: MLP Sub-module Decomposition ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n1 = n_layers - 1
    layers = get_layers(model)
    mlp_type = MODEL_CONFIGS[model_name]["mlp_type"]

    test_cats = ["fruit", "clothing", "emotion"]
    results = {}

    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()
        cat_objs = CATEGORIES_16[cat_name][:8]

        # 获取shared方向
        other_cats = [c for c in CAT_NAMES_16 if c != cat_name][:7]
        h_n1_all, _, _ = get_layer_hidden(model, tokenizer, device, cat_objs, L_n1)
        h_n1_mean = np.mean(h_n1_all, axis=0)
        other_h_n1 = {}
        for oc in other_cats:
            oc_h, _, _ = get_layer_hidden(model, tokenizer, device, CATEGORIES_16[oc][:8], L_n1)
            other_h_n1[oc] = oc_h
        h_others_n1 = np.array([np.mean(other_h_n1[oc], axis=0) for oc in other_cats])
        shared_dir_n1, _ = get_shared_direction(h_n1_mean, h_others_n1)

        layer_n1 = layers[L_n1]
        mlp_mod = getattr(layer_n1, 'mlp', None) or getattr(layer_n1, 'feed_forward', None)
        if mlp_mod is None:
            results[cat_name] = {"error": "no MLP"}
            continue

        # 识别子模块
        down_proj = getattr(mlp_mod, 'down_proj', None)
        if down_proj is None:
            results[cat_name] = {"error": "no down_proj"}
            continue

        # 对每个样本: 获取MLP各阶段的中间激活
        sample_data = []
        for s_idx, obj in enumerate(cat_objs):
            plog(f"    Sample {s_idx}/{len(cat_objs)}: {obj}")

            # 使用hook捕获MLP内部激活
            # MLP结构: x -> LN -> up/gate -> act -> down -> output
            # 我们需要捕获:
            # 1. MLP input (after LN)
            # 2. After activation (before down_proj)
            # 3. MLP output (after down_proj)

            mlp_activations = {}

            def capture_act_hook(key, store):
                def hook(mod, inp, out):
                    if isinstance(out, tuple):
                        store[key] = out[0].detach().float().cpu()
                    else:
                        store[key] = out.detach().float().cpu()
                    # Also store input
                    if isinstance(inp, tuple) and len(inp) > 0:
                        store[key + "_input"] = inp[0].detach().float().cpu()
                return hook

            # 方法: 获取down_proj的权重，计算它对shared方向的贡献
            # 注意: device_map="auto"可能导致权重在meta device上，需要安全获取
            try:
                w = down_proj.weight
                if w.device.type == 'meta':
                    # 权重在meta device，跳过权重分析
                    W_down = None
                else:
                    W_down = w.detach().float().cpu().numpy()  # [d_model, intermediate]
            except NotImplementedError:
                W_down = None

            # 获取MLP output在shared方向上的投影
            inputs_single = encode_prompts(tokenizer, device, [obj])
            mask_single = inputs_single["attention_mask"].bool()
            sp = int(mask_single[0].sum() - 1)

            # 捕获MLP output和down_proj input
            cap_mlp = {}
            hooks_mlp = [
                mlp_mod.register_forward_hook(capture_act_hook("mlp_out", cap_mlp)),
                down_proj.register_forward_hook(capture_act_hook("down_out", cap_mlp)),
                layers[L_n1].register_forward_hook(capture_act_hook("layer_out", cap_mlp)),
            ]
            with torch.no_grad():
                model(**inputs_single)
            for h in hooks_mlp:
                h.remove()

            if "mlp_out" not in cap_mlp or "down_out" not in cap_mlp:
                continue

            mlp_out = cap_mlp["mlp_out"][0, sp, :].numpy()
            down_out = cap_mlp["down_out"][0, sp, :].numpy()
            layer_out = cap_mlp["layer_out"][0, sp, :].numpy()

            # MLP output对shared方向的贡献
            proj_shared_mlp = float(np.dot(mlp_out, shared_dir_n1))
            # down_proj output对shared方向的贡献 (should be same as mlp_out for last layer)
            proj_shared_down = float(np.dot(down_out, shared_dir_n1))

            # down_proj权重对目标类别logit的贡献
            # mlp_out = down_act @ W_down^T (where down_act is the intermediate activation)
            # contribution to target logit = <mlp_out, W_U[target]> = <down_act, W_down @ W_U[target]>
            target_idx = CAT_NAMES_16.index(cat_name)
            readout_dir = W_U[target_idx]  # [d_model]

            # down_proj: W_down shape = [d_model, intermediate]
            # mlp_out = activation @ W_down.T  →  <mlp_out, shared_dir> = <activation, W_down.T @ shared_dir>
            # shared_readout: 中间层维度中哪些对shared方向有贡献
            if W_down is not None:
                shared_readout = W_down.T @ shared_dir_n1  # [intermediate]
            else:
                shared_readout = None

            # 对目标logit的贡献
            mlp_target_contrib = float(np.dot(mlp_out, readout_dir))
            down_target_contrib = float(np.dot(down_out, readout_dir))

            # 分析gate_proj和up_proj (如果存在)
            gate_proj_mod = getattr(mlp_mod, 'gate_proj', None)
            up_proj_mod = getattr(mlp_mod, 'up_proj', None)
            gate_up_proj_mod = getattr(mlp_mod, 'gate_up_proj', None)

            gate_analysis = {}
            if gate_proj_mod is not None and up_proj_mod is not None:
                # Split gate/up模式
                try:
                    wg = gate_proj_mod.weight
                    wu = up_proj_mod.weight
                    if wg.device.type == 'meta' or wu.device.type == 'meta':
                        raise NotImplementedError("meta device")
                    W_gate = wg.detach().float().cpu().numpy()  # [intermediate, d_model]
                    W_up = wu.detach().float().cpu().numpy()  # [intermediate, d_model]

                    gate_shared = W_gate @ shared_dir_n1
                    up_shared = W_up @ shared_dir_n1

                    gate_analysis = {
                        "W_gate_shared_norm": float(np.linalg.norm(gate_shared)),
                        "W_up_shared_norm": float(np.linalg.norm(up_shared)),
                        "W_gate_shared_readout": float(np.dot(gate_shared, shared_readout)) if W_down is not None else None,
                        "W_up_shared_readout": float(np.dot(up_shared, shared_readout)) if W_down is not None else None,
                        "W_gate_shared_mean": float(np.mean(np.abs(gate_shared))),
                        "W_up_shared_mean": float(np.mean(np.abs(up_shared))),
                    }
                except (NotImplementedError, RuntimeError):
                    gate_analysis = {"error": "weights on meta device"}

            elif gate_up_proj_mod is not None:
                # Merged gate_up模式 (GLM4)
                try:
                    wgu = gate_up_proj_mod.weight
                    if wgu.device.type == 'meta':
                        raise NotImplementedError("meta device")
                    W_gate_up = wgu.detach().float().cpu().numpy()
                    inter_size = W_gate_up.shape[0] // 2
                    W_gate_m = W_gate_up[:inter_size, :]
                    W_up_m = W_gate_up[inter_size:, :]

                    gate_shared = W_gate_m @ shared_dir_n1
                    up_shared = W_up_m @ shared_dir_n1

                    gate_analysis = {
                        "mode": "merged_gate_up",
                        "W_gate_shared_norm": float(np.linalg.norm(gate_shared)),
                        "W_up_shared_norm": float(np.linalg.norm(up_shared)),
                        "W_gate_shared_readout": float(np.dot(gate_shared, shared_readout)) if W_down is not None else None,
                        "W_up_shared_readout": float(np.dot(up_shared, shared_readout)) if W_down is not None else None,
                    }
                except (NotImplementedError, RuntimeError):
                    gate_analysis = {"error": "weights on meta device"}

            sample_data.append({
                "obj": obj,
                "proj_shared_mlp": round(proj_shared_mlp, 2),
                "proj_shared_down": round(proj_shared_down, 2),
                "mlp_target_contrib": round(mlp_target_contrib, 4),
                "down_target_contrib": round(down_target_contrib, 4),
                "gate_analysis": gate_analysis,
            })

        results[cat_name] = {
            "n_samples": len(sample_data),
            "mlp_type": mlp_type,
            "samples": sample_data,
            "elapsed": round(time.time() - t0, 2),
        }

        # 汇总
        if sample_data:
            mean_mlp_contrib = np.mean([s["mlp_target_contrib"] for s in sample_data])
            plog(f"  {cat_name}: mean MLP target contrib = {mean_mlp_contrib:.4f}")

    return results


# ==================== Exp3: 多Token位置干预 ====================
def exp3_multi_token_position(model, tokenizer, device, model_name, W_U):
    """
    在L(n-2)的不同token位置分别干预shared分量
    
    位置:
    - object_token: 对象词元(如"apple")
    - relation_token: 关系词元(如"a", "kind")
    - last_token: 最后词元(模型读出位置)
    - all_tokens: 所有位置
    """
    plog(f"=== Exp3: Multi-Token Position Intervention ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n2 = n_layers - 2
    L_n1 = n_layers - 1
    layers = get_layers(model)

    test_cats = ["fruit", "emotion"]
    results = {}

    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()
        cat_objs = CATEGORIES_16[cat_name][:8]
        other_cats = [c for c in CAT_NAMES_16 if c != cat_name][:7]

        # 获取shared方向
        h_n2_all, _, _ = get_layer_hidden(model, tokenizer, device, cat_objs, L_n2)
        h_n2_mean = np.mean(h_n2_all, axis=0)
        other_h_n2 = {}
        for oc in other_cats:
            oc_h, _, _ = get_layer_hidden(model, tokenizer, device, CATEGORIES_16[oc][:8], L_n2)
            other_h_n2[oc] = oc_h
        h_others_n2 = np.array([np.mean(other_h_n2[oc], axis=0) for oc in other_cats])
        shared_dir_n2, _ = get_shared_direction(h_n2_mean, h_others_n2)

        # 获取L(n-1)基线
        h_n1_all, _, _ = get_layer_hidden(model, tokenizer, device, cat_objs, L_n1)
        h_n1_mean = np.mean(h_n1_all, axis=0)
        other_h_n1 = {}
        for oc in other_cats:
            oc_h, _, _ = get_layer_hidden(model, tokenizer, device, CATEGORIES_16[oc][:8], L_n1)
            other_h_n1[oc] = oc_h
        h_others_n1 = np.array([np.mean(other_h_n1[oc], axis=0) for oc in other_cats])
        shared_dir_n1, _ = get_shared_direction(h_n1_mean, h_others_n1)

        position_results = {}

        for s_idx, obj in enumerate(cat_objs):
            plog(f"    Sample {s_idx}/{len(cat_objs)}: {obj}")
            inputs_single = encode_prompts(tokenizer, device, [obj])
            input_ids = inputs_single["input_ids"]
            mask_single = inputs_single["attention_mask"].bool()
            sp = int(mask_single[0].sum() - 1)

            # 解码token位置
            tokens = [f"T{i}" for i in range(sp + 1)]
            # 对象词元通常是第1或第2个位置
            # "The apple is a kind of" -> tokens: [The, apple, is, a, kind, of]
            # 对象是位置1, 关系是3-5, last是5
            n_tokens = sp + 1
            object_pos = 1  # 通常对象在第2个位置
            last_pos = sp
            # 关系位置: "a", "kind", "of" 通常是倒数3个
            relation_start = max(2, sp - 2)

            # 基线
            cap_dict = {}
            hooks = [layers[L_n1].register_forward_hook(
                lambda m, i, o, store=cap_dict: store.update({"Ln1": o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu()})
            )]
            with torch.no_grad():
                model(**inputs_single)
            for h in hooks:
                h.remove()
            if "Ln1" not in cap_dict:
                continue
            h_n1_bl = cap_dict["Ln1"][0, sp, :].numpy()
            D_bl = compute_target_logit(h_n1_bl, W_U, cat_name, CAT_NAMES_16)

            # 逐位置干预
            for pos_name, positions in [
                ("object_only", [object_pos]),
                ("relation_only", list(range(relation_start, sp))),
                ("last_only", [last_pos]),
                ("all_semantic", list(range(1, sp + 1))),
            ]:
                cap_dict_int = {}
                # 在L(n-2)的指定位置消融shared分量
                def make_positional_replace_hook(pos_list, shared_dir, d_model):
                    def hook(module, inp, out):
                        if isinstance(out, tuple):
                            modified = out[0].clone()
                        else:
                            modified = out.clone()
                        for p in pos_list:
                            if p < modified.size(1):
                                h_p = modified[0, p, :].float().cpu().numpy()
                                proj = np.dot(h_p, shared_dir)
                                h_p_new = h_p - proj * shared_dir
                                modified[0, p, :] = torch.tensor(
                                    h_p_new, dtype=modified.dtype, device=modified.device
                                )
                        if isinstance(out, tuple):
                            return (modified,) + out[1:]
                        return modified
                    return hook

                hooks_int = [
                    layers[L_n2].register_forward_hook(
                        make_positional_replace_hook(positions, shared_dir_n2, None)
                    ),
                    layers[L_n1].register_forward_hook(
                        lambda m, i, o, store=cap_dict_int: store.update({"Ln1": o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu()})
                    ),
                ]
                with torch.no_grad():
                    model(**inputs_single)
                for h in hooks_int:
                    h.remove()

                if "Ln1" not in cap_dict_int:
                    continue
                h_n1_int = cap_dict_int["Ln1"][0, sp, :].numpy()
                D_int = compute_target_logit(h_n1_int, W_U, cat_name, CAT_NAMES_16)
                proj_int = float(np.dot(h_n1_int, shared_dir_n1))

                if pos_name not in position_results:
                    position_results[pos_name] = []
                position_results[pos_name].append({
                    "obj": obj,
                    "delta_D": round(D_int - D_bl, 4),
                    "proj_shared_n1": round(proj_int, 2),
                })

        # 汇总
        summary = {}
        for pos_name, data in position_results.items():
            if data:
                mean_delta = np.mean([d["delta_D"] for d in data])
                summary[pos_name] = {
                    "n_samples": len(data),
                    "mean_delta_D": round(float(mean_delta), 4),
                    "details": data,
                }
                plog(f"  {cat_name} {pos_name}: mean ΔD = {mean_delta:.2f}")

        results[cat_name] = {
            "positions": summary,
            "token_layout": f"n_tokens={n_tokens}, object_pos={object_pos}, last_pos={last_pos}",
            "elapsed": round(time.time() - t0, 2),
        }

    return results


# ==================== Exp4: DS7B Animal异常分析 ====================
def exp4_ds7b_animal(model, tokenizer, device, model_name, W_U):
    """
    DS7B animal异常: ΔD_n1=+90.51（与fruit/clothing/emotion方向相反）
    
    检查:
    1. animal类别的竞争类别释放模式
    2. animal shared方向的生命/主体/动作属性
    3. animal目标token的读出结构
    """
    plog(f"=== Exp4: DS7B Animal Anomaly Analysis ===")
    if model_name != "deepseek7b":
        plog("  Skipping: only for deepseek7b")
        return {"skipped": True, "reason": "only for deepseek7b"}

    n_layers = get_model_info(model, model_name).n_layers
    L_n1 = n_layers - 1
    L_n2 = n_layers - 2
    layers = get_layers(model)

    # 对比animal vs fruit
    target_cats = ["animal", "fruit"]
    results = {}

    for cat_name in target_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()
        cat_objs = CATEGORIES_16[cat_name][:10]
        other_cats = [c for c in CAT_NAMES_16 if c != cat_name][:7]

        # 获取L(n-1) hidden
        h_n1_all, _, _ = get_layer_hidden(model, tokenizer, device, cat_objs, L_n1)
        h_n1_mean = np.mean(h_n1_all, axis=0)

        other_h_n1 = {}
        for oc in other_cats:
            oc_h, _, _ = get_layer_hidden(model, tokenizer, device, CATEGORIES_16[oc][:8], L_n1)
            other_h_n1[oc] = oc_h
        h_others_n1 = np.array([np.mean(other_h_n1[oc], axis=0) for oc in other_cats])
        shared_dir_n1, Bc_dir = get_shared_direction(h_n1_mean, h_others_n1)

        # 分析: shared方向对所有类别的logit贡献
        all_logit_contribs = {}
        for i, cn in enumerate(CAT_NAMES_16):
            contrib = float(np.dot(shared_dir_n1, W_U[i]) * np.dot(h_n1_mean, shared_dir_n1))
            all_logit_contribs[cn] = round(contrib, 4)

        # 分析: Bc方向对所有类别的logit贡献
        Bc_logit_contribs = {}
        for i, cn in enumerate(CAT_NAMES_16):
            contrib = float(np.dot(Bc_dir, W_U[i]) * np.dot(h_n1_mean, Bc_dir))
            Bc_logit_contribs[cn] = round(contrib, 4)

        # 获取MLP的shared贡献
        layer_n1 = layers[L_n1]
        mlp_mod = getattr(layer_n1, 'mlp', None)

        # Zero MLP test for this category
        sample_delta_zmlp = []
        for s_idx, obj in enumerate(cat_objs[:5]):
            inputs_single = encode_prompts(tokenizer, device, [obj])
            mask_single = inputs_single["attention_mask"].bool()
            sp = int(mask_single[0].sum() - 1)

            # Baseline
            cap_bl = {}
            hooks_bl = [layers[L_n1].register_forward_hook(
                lambda m, i, o, s=cap_bl: s.update({"Ln1": o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu()})
            )]
            with torch.no_grad():
                model(**inputs_single)
            for h in hooks_bl:
                h.remove()
            if "Ln1" not in cap_bl:
                continue
            D_bl = compute_target_logit(cap_bl["Ln1"][0, sp, :].numpy(), W_U, cat_name, CAT_NAMES_16)

            # Zero MLP
            if mlp_mod:
                cap_zm = {}
                hooks_zm = [
                    mlp_mod.register_forward_hook(make_zero_output_hook()),
                    layers[L_n1].register_forward_hook(
                        lambda m, i, o, s=cap_zm: s.update({"Ln1": o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu()})
                    ),
                ]
                with torch.no_grad():
                    model(**inputs_single)
                for h in hooks_zm:
                    h.remove()
                if "Ln1" not in cap_zm:
                    continue
                D_zm = compute_target_logit(cap_zm["Ln1"][0, sp, :].numpy(), W_U, cat_name, CAT_NAMES_16)
                sample_delta_zmlp.append(D_zm - D_bl)

        results[cat_name] = {
            "all_logit_contribs_shared": all_logit_contribs,
            "Bc_logit_contribs": Bc_logit_contribs,
            "zero_mlp_delta_D": [round(float(d), 4) for d in sample_delta_zmlp],
            "mean_zero_mlp_delta_D": round(float(np.mean(sample_delta_zmlp)), 4) if sample_delta_zmlp else None,
            "elapsed": round(time.time() - t0, 2),
        }
        plog(f"  {cat_name}: zero_mlp ΔD = {np.mean(sample_delta_zmlp):.2f}" if sample_delta_zmlp else f"  {cat_name}: no zero_mlp data")

    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    plog(f"Phase 496 R{round_num}: {model_name}")
    plog(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        plog(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # 加载模型
    model, tokenizer, device = get_model_and_tokenizer(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, {info.n_layers} layers, d_model={info.d_model}")

    # 获取W_U
    W_U = get_W_U(model, model_name)
    plog(f"W_U shape: {W_U.shape}")

    results = {
        "phase": 496,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: 真正因果MLP/Attn干预
    plog("\n" + "="*60)
    results["exp1_causal_mlp_attn"] = exp1_causal_mlp_attn_v2(
        model, tokenizer, device, model_name, W_U
    )

    # Exp2: MLP子模块分解
    plog("\n" + "="*60)
    results["exp2_mlp_submodule"] = exp2_mlp_submodule(
        model, tokenizer, device, model_name, W_U
    )

    # Exp3: 多Token位置干预
    plog("\n" + "="*60)
    results["exp3_multi_token_position"] = exp3_multi_token_position(
        model, tokenizer, device, model_name, W_U
    )

    # Exp4: DS7B Animal异常
    plog("\n" + "="*60)
    results["exp4_ds7b_animal"] = exp4_ds7b_animal(
        model, tokenizer, device, model_name, W_U
    )

    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase496_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"Results saved to {out_path}")

    # 释放模型
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    plog("Done!")


if __name__ == "__main__":
    main()
