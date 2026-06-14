"""
Phase 495: 逐样本因果验证、末层Attn/MLP分解、剂量曲线与异常机制
====================================================================

Phase 494遗留核心问题:
1. 均值替换是否夸大跨层因果？→ 需要逐样本干预验证
2. 末层符号翻转由哪个模块(Attn/MLP/LN)执行？→ 需要层内分解
3. ablate/double比例不对称说明传递非线性→ 需要剂量曲线
4. DS7B animal异常方向相反→ 需要解释

Phase 495核心实验:
- Exp1: 逐样本跨层因果干预 — 对每个样本单独消融shared再forward，与均值结果对比
- Exp2: 末层Attn/MLP分解 — 在L(n-1)分别消融Attn输出和MLP输出，看谁执行符号翻转
- Exp3: 剂量曲线 — 对L(n-2) shared做连续强度干预(-2.0到+2.0)，观察L(n-1)响应
- Exp4: 异常类别专项 — DS7B animal为什么方向相反 + action为什么弱释放

用法:
  python tests/glm5/phase495_samplewise_attn_mlp_dose.py qwen3 1
  python tests/glm5/phase495_samplewise_attn_mlp_dose.py glm4 1
  python tests/glm5/phase495_samplewise_attn_mlp_dose.py deepseek7b 1
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


def ablate_direction(h, direction):
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    proj = np.dot(h, d_norm) * d_norm
    return h - proj


def get_category_hidden_single(model, tokenizer, device, objects, layer_idx,
                                template="The {obj} is a kind of"):
    """获取某类在某层的所有样本hidden states（逐样本）"""
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


# ==================== Exp1: 逐样本跨层因果干预 ====================
def exp1_samplewise_causal(model, tokenizer, device, model_name, W_U):
    """
    核心验证: 对每个样本单独在L(n-2)消融shared分量，然后forward到L(n-1)
    与Phase 494的均值替换结果对比
    
    这是最关键的验证——如果逐样本结果的平均值与均值替换一致，
    则Phase 494的结论可靠；否则需要修正。
    """
    plog(f"=== Exp1: Sample-wise Causal Cross-layer Intervention ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n2 = n_layers - 2
    L_n1 = n_layers - 1
    layers = get_layers(model)

    # 选4个关键类别: 2个正常反转 + 2个异常
    test_cats = ["fruit", "clothing", "emotion", "action"]
    other_cats_pool = CAT_NAMES_16.copy()

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()
        cat_objs = CATEGORIES_16[cat_name][:12]
        other_cats = [c for c in other_cats_pool if c != cat_name][:7]

        # Step 1: 获取其他类别在L(n-2)的hidden（用于计算shared方向）
        other_h_n2 = {}
        for oc in other_cats:
            oc_h_all, _, _ = get_category_hidden_single(
                model, tokenizer, device, CATEGORIES_16[oc][:8], L_n2
            )
            other_h_n2[oc] = oc_h_all

        h_others_n2 = np.array([np.mean(other_h_n2[oc], axis=0) for oc in other_cats])

        # Step 2: 获取目标类别每个样本在L(n-2)的hidden
        h_target_all, inputs_n2, last_idx_n2 = get_category_hidden_single(
            model, tokenizer, device, cat_objs, L_n2
        )
        h_target_mean = np.mean(h_target_all, axis=0)

        # 计算shared方向（基于类别均值）
        shared_dir_n2, Bc_n2 = get_shared_direction(h_target_mean, h_others_n2)

        # Step 3: 获取自然状态下L(n-1)的hidden（基线）
        h_n1_natural_all, _, _ = get_category_hidden_single(
            model, tokenizer, device, cat_objs, L_n1
        )
        h_n1_natural_mean = np.mean(h_n1_natural_all, axis=0)

        # 获取L(n-1)的shared方向
        other_h_n1 = {}
        for oc in other_cats:
            oc_h_all, _, _ = get_category_hidden_single(
                model, tokenizer, device, CATEGORIES_16[oc][:8], L_n1
            )
            other_h_n1[oc] = oc_h_all
        h_others_n1 = np.array([np.mean(other_h_n1[oc], axis=0) for oc in other_cats])
        shared_dir_n1, _ = get_shared_direction(h_n1_natural_mean, h_others_n1)

        # Step 4: 基线DCF
        baseline_D_n1_mean = compute_target_logit(h_n1_natural_mean, W_U, cat_name, CAT_NAMES_16)

        # Step 5: 逐样本因果干预
        # 对每个样本：在L(n-2)消融其shared分量 → forward到L(n-1) → 记录L(n-1)变化
        sample_deltas = []
        sample_baseline_Ds = []

        for s_idx in range(len(cat_objs)):
            # 获取单个样本的L(n-2) hidden
            h_n2_sample = h_target_all[s_idx]

            # 消融shared分量
            proj_shared = np.dot(h_n2_sample, shared_dir_n2)
            h_n2_modified = h_n2_sample - proj_shared * shared_dir_n2

            # 注入修改后的值到L(n-2)，forward到L(n-1)
            modified_tensor = torch.tensor(
                h_n2_modified.reshape(1, 1, -1),
                dtype=torch.bfloat16, device=device
            )

            captured_causal = {}

            def make_replace_hook(replacement_val, seq_pos):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        out[0, seq_pos, :] = replacement_val[0, 0, :]
                        return (out,) + output[1:]
                    else:
                        out = output.clone()
                        out[0, seq_pos, :] = replacement_val[0, 0, :]
                        return out
                return hook

            def make_capture_hook(key):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        captured_causal[key] = output[0].detach().float().cpu()
                    else:
                        captured_causal[key] = output.detach().float().cpu()
                return hook

            # 单样本输入
            single_inputs = encode_prompts(tokenizer, device, [cat_objs[s_idx]])
            single_mask = single_inputs["attention_mask"].bool()
            seq_pos = int(single_mask[0].sum() - 1)

            hook_replace = layers[L_n2].register_forward_hook(
                make_replace_hook(modified_tensor, seq_pos)
            )
            hook_capture = layers[L_n1].register_forward_hook(
                make_capture_hook(f"L{L_n1}")
            )

            with torch.no_grad():
                model(**single_inputs)

            hook_replace.remove()
            hook_capture.remove()

            if f"L{L_n1}" in captured_causal:
                h_n1_causal = captured_causal[f"L{L_n1}"]
                h_n1_causal_last = h_n1_causal[0, seq_pos, :].numpy()

                causal_D = compute_target_logit(h_n1_causal_last, W_U, cat_name, CAT_NAMES_16)
                baseline_D = compute_target_logit(h_n1_natural_all[s_idx], W_U, cat_name, CAT_NAMES_16)
                delta_D = causal_D - baseline_D
                sample_deltas.append(delta_D)
                sample_baseline_Ds.append(baseline_D)

        # Step 6: 统计
        mean_delta = float(np.mean(sample_deltas))
        std_delta = float(np.std(sample_deltas))
        median_delta = float(np.median(sample_deltas))

        # 逐样本的符号一致性
        positive_count = sum(1 for d in sample_deltas if d > 0)
        negative_count = sum(1 for d in sample_deltas if d < 0)

        # 均值替换的delta（Phase 494方法的重现）
        h_n2_mean_modified = ablate_direction(h_target_mean, shared_dir_n2)
        # 这里不能直接forward均值，但可以比较DCF变化
        mean_ablate_D_n2 = compute_target_logit(h_n2_mean_modified, W_U, cat_name, CAT_NAMES_16)
        mean_baseline_D_n2 = compute_target_logit(h_target_mean, W_U, cat_name, CAT_NAMES_16)
        mean_ablate_delta_n2 = mean_ablate_D_n2 - mean_baseline_D_n2

        results[cat_name] = {
            "n_samples": len(sample_deltas),
            "sample_deltas": [float(d) for d in sample_deltas],
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "median_delta": median_delta,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "sign_consistency": max(positive_count, negative_count) / len(sample_deltas) if sample_deltas else 0,
            "mean_ablate_delta_n2": float(mean_ablate_delta_n2),
            "baseline_D_n1_mean": float(baseline_D_n1_mean),
            "elapsed": time.time() - t0,
        }

        plog(f"    mean_ΔD_n1={mean_delta:+.2f} ± {std_delta:.2f}, "
             f"sign={negative_count}/{len(sample_deltas)} neg, "
             f"consistency={results[cat_name]['sign_consistency']:.1%}")

    # 汇总
    plog(f"\n  === Exp1 Sample-wise Summary ===")
    for cat_name, data in results.items():
        plog(f"  {cat_name}: mean_ΔD={data['mean_delta']:+.2f}, "
             f"consistency={data['sign_consistency']:.1%}")

    return results


# ==================== Exp2: 末层Attn/MLP分解 ====================
def exp2_attn_mlp_decomposition(model, tokenizer, device, model_name, W_U):
    """
    核心问题: L(n-1)的符号翻转由哪个模块执行？
    
    方法:
    1. 正常forward，在L(n-1)捕获:
       - 整层输出 (layer_output)
       - Attention输出 (attn_output) — hook在self_attn
       - MLP输出 (mlp_output) — hook在mlp
    2. 对每个子模块输出分别消融shared分量，看DCF变化
    3. 看Attn和MLP哪个对shared方向的logit贡献发生了符号翻转
    """
    plog(f"=== Exp2: Last Layer Attn/MLP Decomposition ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n2 = n_layers - 2
    L_n1 = n_layers - 1
    layers = get_layers(model)

    test_cats = ["fruit", "clothing", "animal", "emotion", "action"]
    other_cats_pool = CAT_NAMES_16.copy()

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()
        cat_objs = CATEGORIES_16[cat_name][:10]
        other_cats = [c for c in other_cats_pool if c != cat_name][:7]

        # 收集三个hook: 整层输出, Attn输出, MLP输出
        for li in [L_n2, L_n1]:
            captured = {}

            def make_hook(key):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook

            layer = layers[li]

            # 获取Attn和MLP子模块
            attn_module = layer.self_attn
            mlp_module = layer.mlp

            hook_layer = layer.register_forward_hook(make_hook("layer"))
            hook_attn = attn_module.register_forward_hook(make_hook("attn"))
            hook_mlp = mlp_module.register_forward_hook(make_hook("mlp"))

            inputs = encode_prompts(tokenizer, device, cat_objs)
            with torch.no_grad():
                model(**inputs)

            hook_layer.remove()
            hook_attn.remove()
            hook_mlp.remove()

            mask = inputs["attention_mask"].bool()
            last_idx = mask.sum(dim=1) - 1

            # 提取最后token
            h_layer = captured["layer"][torch.arange(captured["layer"].size(0)), last_idx.cpu()].numpy()
            h_attn = captured["attn"][torch.arange(captured["attn"].size(0)), last_idx.cpu()].numpy()
            h_mlp = captured["mlp"][torch.arange(captured["mlp"].size(0)), last_idx.cpu()].numpy()

            h_layer_mean = np.mean(h_layer, axis=0)
            h_attn_mean = np.mean(h_attn, axis=0)
            h_mlp_mean = np.mean(h_mlp, axis=0)

            # 获取其他类别的hidden
            other_h_layer = {}
            for oc in other_cats:
                oc_captured = {}

                def make_oc_hook(key):
                    def hook(module, inp, output):
                        if isinstance(output, tuple):
                            oc_captured[key] = output[0].detach().float().cpu()
                        else:
                            oc_captured[key] = output.detach().float().cpu()
                    return hook

                oc_inputs = encode_prompts(tokenizer, device, CATEGORIES_16[oc][:8])
                oc_layer = layers[li]
                hook_oc = oc_layer.register_forward_hook(make_oc_hook("layer"))
                with torch.no_grad():
                    model(**oc_inputs)
                hook_oc.remove()

                oc_mask = oc_inputs["attention_mask"].bool()
                oc_last_idx = oc_mask.sum(dim=1) - 1
                other_h_layer[oc] = oc_captured["layer"][
                    torch.arange(oc_captured["layer"].size(0)), oc_last_idx.cpu()
                ].numpy()

            h_others = np.array([np.mean(other_h_layer[oc], axis=0) for oc in other_cats])

            # shared方向（基于整层输出）
            shared_dir, Bc_norm = get_shared_direction(h_layer_mean, h_others)

            # 整层: shared方向对target logit的贡献
            layer_target_contrib = float(compute_target_logit(shared_dir, W_U, cat_name, CAT_NAMES_16))

            # Attn输出: shared方向投影
            attn_proj_shared = float(np.dot(h_attn_mean, shared_dir))

            # MLP输出: shared方向投影
            mlp_proj_shared = float(np.dot(h_mlp_mean, shared_dir))

            # Attn输出: 对target logit的shared贡献（attn输出中shared分量的logit贡献）
            attn_shared_component = np.dot(h_attn_mean, shared_dir) * shared_dir
            attn_target_contrib = float(compute_target_logit(attn_shared_component, W_U, cat_name, CAT_NAMES_16))

            # MLP输出: 对target logit的shared贡献
            mlp_shared_component = np.dot(h_mlp_mean, shared_dir) * shared_dir
            mlp_target_contrib = float(compute_target_logit(mlp_shared_component, W_U, cat_name, CAT_NAMES_16))

            # 消融实验: 分别消融Attn和MLP的shared分量，看对DCF的影响
            # Attn消融: layer_output - attn_shared → 重新计算DCF
            # 注意: 这不是真正消融后forward，而是计算消融shared后的DCF
            # 更精确的方法是: 消融h_attn中的shared分量，然后重建h_layer
            # h_layer ≈ h_residual_before_attn + h_attn + h_mlp (残差连接)
            # 消融attn中的shared: h_layer_new = h_layer - attn_shared_component
            h_layer_no_attn_shared = h_layer_mean - attn_shared_component
            D_no_attn_shared = compute_target_logit(h_layer_no_attn_shared, W_U, cat_name, CAT_NAMES_16)

            h_layer_no_mlp_shared = h_layer_mean - mlp_shared_component
            D_no_mlp_shared = compute_target_logit(h_layer_no_mlp_shared, W_U, cat_name, CAT_NAMES_16)

            D_baseline = compute_target_logit(h_layer_mean, W_U, cat_name, CAT_NAMES_16)

            key = f"L{li}"
            if cat_name not in results:
                results[cat_name] = {}

            results[cat_name][key] = {
                "layer_target_contrib": layer_target_contrib,
                "attn_proj_shared": attn_proj_shared,
                "mlp_proj_shared": mlp_proj_shared,
                "attn_target_contrib": attn_target_contrib,
                "mlp_target_contrib": mlp_target_contrib,
                "D_baseline": float(D_baseline),
                "delta_D_no_attn_shared": float(D_no_attn_shared - D_baseline),
                "delta_D_no_mlp_shared": float(D_no_mlp_shared - D_baseline),
                "attn_norm": float(np.linalg.norm(h_attn_mean)),
                "mlp_norm": float(np.linalg.norm(h_mlp_mean)),
            }

            plog(f"    {key}: layer_contrib={layer_target_contrib:+.4f}, "
                 f"attn_contrib={attn_target_contrib:+.4f}, mlp_contrib={mlp_target_contrib:+.4f}, "
                 f"ΔD_no_attn={D_no_attn_shared - D_baseline:+.2f}, ΔD_no_mlp={D_no_mlp_shared - D_baseline:+.2f}")

        # 跨层变化
        ln2 = results[cat_name][f"L{L_n2}"]
        ln1 = results[cat_name][f"L{L_n1}"]

        results[cat_name]["cross_layer"] = {
            "layer_contrib_flip": ln2["layer_target_contrib"] * ln1["layer_target_contrib"] < 0,
            "attn_contrib_flip": ln2["attn_target_contrib"] * ln1["attn_target_contrib"] < 0,
            "mlp_contrib_flip": ln2["mlp_target_contrib"] * ln1["mlp_target_contrib"] < 0,
            "attn_delta_D_change": ln1["delta_D_no_attn_shared"] - ln2["delta_D_no_attn_shared"],
            "mlp_delta_D_change": ln1["delta_D_no_mlp_shared"] - ln2["delta_D_no_mlp_shared"],
        }

        flip_info = results[cat_name]["cross_layer"]
        plog(f"    Flip: layer={flip_info['layer_contrib_flip']}, "
             f"attn={flip_info['attn_contrib_flip']}, mlp={flip_info['mlp_contrib_flip']}")

        results[cat_name]["elapsed"] = time.time() - t0

    return results


# ==================== Exp3: 剂量曲线 ====================
def exp3_dose_response(model, tokenizer, device, model_name, W_U):
    """
    对L(n-2) shared做连续强度干预，观察L(n-1)的非线性响应
    
    看ablate/double不对称的原因:
    - 线性? 阈值? 饱和? 门控?
    """
    plog(f"=== Exp3: Dose-Response Curve ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n2 = n_layers - 2
    L_n1 = n_layers - 1
    layers = get_layers(model)

    # 只测2个类别: 1个释放型 + 1个保守型参照
    test_cats = ["fruit"]
    other_cats_pool = CAT_NAMES_16.copy()

    # 剂量: scale factor for shared direction
    # scale=0: 完全消融; scale=1: 自然; scale=2: 加倍; scale=-1: 反转
    scales = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()
        cat_objs = CATEGORIES_16[cat_name][:8]
        other_cats = [c for c in other_cats_pool if c != cat_name][:7]

        # Step 1: 获取自然L(n-2)和L(n-1)的hidden
        inputs = encode_prompts(tokenizer, device, cat_objs)

        captured_natural = {}
        def make_hook_natural(key):
            def hook(module, inp, output):
                if isinstance(output, tuple):
                    captured_natural[key] = output[0].detach().float().cpu()
                else:
                    captured_natural[key] = output.detach().float().cpu()
            return hook

        hook_n2 = layers[L_n2].register_forward_hook(make_hook_natural(f"L{L_n2}"))
        hook_n1 = layers[L_n1].register_forward_hook(make_hook_natural(f"L{L_n1}"))
        with torch.no_grad():
            model(**inputs)
        hook_n2.remove()
        hook_n1.remove()

        mask = inputs["attention_mask"].bool()
        last_idx = mask.sum(dim=1) - 1
        h_n2_natural = captured_natural[f"L{L_n2}"]
        h_n1_natural = captured_natural[f"L{L_n1}"]
        h_n2_mean = h_n2_natural[0, last_idx[0].item(), :].numpy()
        h_n1_mean = h_n1_natural[0, last_idx[0].item(), :].numpy()

        # 其他类别在L(n-2)的hidden
        other_h_n2 = {}
        for oc in other_cats:
            oc_captured = {}
            def make_oc_hook(key):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        oc_captured[key] = output[0].detach().float().cpu()
                    else:
                        oc_captured[key] = output.detach().float().cpu()
                return hook
            oc_inputs = encode_prompts(tokenizer, device, CATEGORIES_16[oc][:4])
            h_n2_oc = layers[L_n2].register_forward_hook(make_oc_hook(f"L{L_n2}"))
            with torch.no_grad():
                model(**oc_inputs)
            h_n2_oc.remove()
            oc_mask = oc_inputs["attention_mask"].bool()
            oc_last_idx = oc_mask.sum(dim=1) - 1
            other_h_n2[oc] = oc_captured[f"L{L_n2}"][0, oc_last_idx[0].item(), :].numpy()

        h_others_n2 = np.array([other_h_n2[oc] for oc in other_cats])
        shared_dir_n2, _ = get_shared_direction(h_n2_mean, h_others_n2)

        # 自然基线
        proj_shared_natural = float(np.dot(h_n2_mean, shared_dir_n2))
        baseline_D_n1 = compute_target_logit(h_n1_mean, W_U, cat_name, CAT_NAMES_16)

        # Step 2: 剂量曲线
        dose_results = []
        for scale in scales:
            # 修改shared分量: h_new = h + (scale-1) * proj_shared * shared_dir
            # scale=0: 消融; scale=1: 自然; scale=2: 加倍
            h_n2_modified = h_n2_mean + (scale - 1.0) * proj_shared_natural * shared_dir_n2

            modified_tensor = torch.tensor(
                h_n2_modified.reshape(1, 1, -1),
                dtype=torch.bfloat16, device=device
            )

            captured_causal = {}

            def make_replace_hook(replacement_val, seq_pos):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        out[0, seq_pos, :] = replacement_val[0, 0, :]
                        return (out,) + output[1:]
                    else:
                        out = output.clone()
                        out[0, seq_pos, :] = replacement_val[0, 0, :]
                        return out
                return hook

            def make_capture_hook(key):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        captured_causal[key] = output[0].detach().float().cpu()
                    else:
                        captured_causal[key] = output.detach().float().cpu()
                return hook

            seq_pos = int(mask[0].sum() - 1)

            hook_replace = layers[L_n2].register_forward_hook(
                make_replace_hook(modified_tensor, seq_pos)
            )
            hook_capture = layers[L_n1].register_forward_hook(
                make_capture_hook(f"L{L_n1}")
            )

            with torch.no_grad():
                model(**inputs)

            hook_replace.remove()
            hook_capture.remove()

            if f"L{L_n1}" in captured_causal:
                h_n1_causal = captured_causal[f"L{L_n1}"]
                h_n1_causal_last = h_n1_causal[0, seq_pos, :].numpy()

                causal_D = compute_target_logit(h_n1_causal_last, W_U, cat_name, CAT_NAMES_16)
                delta_D = causal_D - baseline_D_n1

                # L(n-1)的shared投影
                proj_shared_n1 = float(np.dot(h_n1_causal_last, shared_dir_n2))
                norm_n1 = float(np.linalg.norm(h_n1_causal_last))

                dose_results.append({
                    "scale": scale,
                    "delta_D_n1": float(delta_D),
                    "proj_shared_n1": proj_shared_n1,
                    "norm_n1": norm_n1,
                })

                plog(f"    scale={scale:+.1f}: ΔD_n1={delta_D:+.2f}, "
                     f"proj_n1={proj_shared_n1:.1f}, norm={norm_n1:.1f}")

        # 线性度检查: scale=1是基线(ΔD=0)
        # 如果传递是线性的: ΔD(scale) 应该正比于 (scale-1)*proj_shared_natural
        # 非线性度 = ΔD(ablate)/Δproj(ablate) vs ΔD(double)/Δproj(double)
        ablate_data = next((d for d in dose_results if d["scale"] == 0.0), None)
        double_data = next((d for d in dose_results if d["scale"] == 2.0), None)

        nonlinearity = {}
        if ablate_data and double_data:
            # 线性预测: ablate减少proj_shared_natural, double增加等量
            # 如果线性: delta_D应该与delta_proj成正比
            ablate_ratio = ablate_data["delta_D_n1"] / (ablate_data["proj_shared_n1"] - proj_shared_natural) if abs(ablate_data["proj_shared_n1"] - proj_shared_natural) > 1e-6 else 0
            double_ratio = double_data["delta_D_n1"] / (double_data["proj_shared_n1"] - proj_shared_natural) if abs(double_data["proj_shared_n1"] - proj_shared_natural) > 1e-6 else 0

            nonlinearity = {
                "ablate_sensitivity": float(ablate_ratio),
                "double_sensitivity": float(double_ratio),
                "sensitivity_ratio": float(ablate_ratio / double_ratio) if abs(double_ratio) > 1e-6 else float('inf'),
            }

        results[cat_name] = {
            "dose_curve": dose_results,
            "nonlinearity": nonlinearity,
            "baseline_D_n1": float(baseline_D_n1),
            "proj_shared_natural": float(proj_shared_natural),
            "elapsed": time.time() - t0,
        }

    return results


# ==================== Exp4: DS7B animal异常 + action弱释放 ====================
def exp4_anomaly_analysis(model, tokenizer, device, model_name, W_U):
    """
    1. DS7B animal: 为什么ablate L(n-2) shared后ΔD_n1=+90.51（方向相反）?
    2. action: 为什么Qwen3弱释放、GLM4/DS7B不释放?
    
    方法:
    - 对每个异常类别，检查shared方向在logit空间中的详细分解
    - 看shared方向对哪些竞争类别有正/负贡献
    - 检查animal在DS7B中是否被不同的竞争类别"拖拽"
    """
    plog(f"=== Exp4: Anomaly Category Analysis ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n2 = n_layers - 2
    L_n1 = n_layers - 1

    test_cats = ["fruit", "animal", "action"]  # fruit作为参照
    other_cats_pool = CAT_NAMES_16.copy()

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()
        cat_objs = CATEGORIES_16[cat_name][:10]

        for li in [L_n2, L_n1]:
            h_target_all, _, _ = get_category_hidden_single(
                model, tokenizer, device, cat_objs, li
            )
            h_target_mean = np.mean(h_target_all, axis=0)

            other_hiddens = {}
            for oc in other_cats_pool:
                if oc == cat_name:
                    continue
                oc_h_all, _, _ = get_category_hidden_single(
                    model, tokenizer, device, CATEGORIES_16[oc][:4], li
                )
                other_hiddens[oc] = np.mean(oc_h_all, axis=0)

            h_others = np.array([other_hiddens[oc] for oc in other_cats_pool if oc != cat_name])
            shared_dir, Bc_norm = get_shared_direction(h_target_mean, h_others)

            # shared方向对所有16个类别的logit贡献
            all_logit_contribs = {}
            for c in CAT_NAMES_16:
                all_logit_contribs[c] = float(compute_target_logit(shared_dir, W_U, c, CAT_NAMES_16))

            # Bc方向对所有16个类别的logit贡献
            Bc_logit_contribs = {}
            for c in CAT_NAMES_16:
                Bc_logit_contribs[c] = float(compute_target_logit(Bc_norm, W_U, c, CAT_NAMES_16))

            # 投影系数
            proj_shared = float(np.dot(h_target_mean, shared_dir))
            proj_Bc = float(np.dot(h_target_mean, Bc_norm))

            # 消融实验: 消融shared/Bc后的DCF变化
            baseline_D = compute_target_logit(h_target_mean, W_U, cat_name, CAT_NAMES_16)
            h_no_shared = ablate_direction(h_target_mean, shared_dir)
            h_no_Bc = ablate_direction(h_target_mean, Bc_norm)
            D_no_shared = compute_target_logit(h_no_shared, W_U, cat_name, CAT_NAMES_16)
            D_no_Bc = compute_target_logit(h_no_Bc, W_U, cat_name, CAT_NAMES_16)

            key = f"L{li}"
            if cat_name not in results:
                results[cat_name] = {}

            results[cat_name][key] = {
                "all_logit_contribs": all_logit_contribs,
                "Bc_logit_contribs": Bc_logit_contribs,
                "proj_shared": proj_shared,
                "proj_Bc": proj_Bc,
                "baseline_D": float(baseline_D),
                "delta_D_no_shared": float(D_no_shared - baseline_D),
                "delta_D_no_Bc": float(D_no_Bc - baseline_D),
            }

            # 找shared方向中贡献最大的类别（正和负）
            sorted_contribs = sorted(all_logit_contribs.items(), key=lambda x: x[1], reverse=True)
            top3_pos = sorted_contribs[:3]
            top3_neg = sorted_contribs[-3:]

            plog(f"    {key}: target_contrib={all_logit_contribs[cat_name]:+.4f}, "
                 f"ΔD_no_shared={D_no_shared - baseline_D:+.2f}")
            plog(f"      Top3+: {[(c, f'{v:+.4f}') for c, v in top3_pos]}")
            plog(f"      Top3-: {[(c, f'{v:+.4f}') for c, v in top3_neg]}")

        # 跨层对比
        ln2 = results[cat_name][f"L{L_n2}"]
        ln1 = results[cat_name][f"L{L_n1}"]
        results[cat_name]["cross_layer"] = {
            "target_contrib_flip": ln2["all_logit_contribs"][cat_name] * ln1["all_logit_contribs"][cat_name] < 0,
            "shared_target_Ln2": ln2["all_logit_contribs"][cat_name],
            "shared_target_Ln1": ln1["all_logit_contribs"][cat_name],
        }

        results[cat_name]["elapsed"] = time.time() - t0

    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    plog(f"Phase 495: {model_name}, round={round_num}")

    # 加载模型
    model, tokenizer, device = get_model_and_tokenizer(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # 获取W_U
    W_U = get_W_U(model, model_name)
    plog(f"W_U: shape={W_U.shape}")

    all_results = {
        "phase": 495,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        }
    }

    # Exp1: 逐样本跨层因果干预
    plog(f"\n{'='*60}")
    all_results["exp1_samplewise_causal"] = exp1_samplewise_causal(
        model, tokenizer, device, model_name, W_U
    )

    # Exp2: 末层Attn/MLP分解
    plog(f"\n{'='*60}")
    all_results["exp2_attn_mlp_decomposition"] = exp2_attn_mlp_decomposition(
        model, tokenizer, device, model_name, W_U
    )

    # Exp3: 剂量曲线
    plog(f"\n{'='*60}")
    all_results["exp3_dose_response"] = exp3_dose_response(
        model, tokenizer, device, model_name, W_U
    )

    # Exp4: 异常类别分析
    plog(f"\n{'='*60}")
    all_results["exp4_anomaly_analysis"] = exp4_anomaly_analysis(
        model, tokenizer, device, model_name, W_U
    )

    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase495_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")

    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Model released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    plog(f"Phase 495 {model_name} complete!")


if __name__ == "__main__":
    main()
