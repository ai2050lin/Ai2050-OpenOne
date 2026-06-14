"""
Phase 494: 跨层因果干预、shared主轴logit分解与异常类别机制
===========================================================

Phase 493遗留核心问题:
1. 跨层传递只测了相关性(slope/corr)，不是因果性 — 需要在L(n-2)真正修改hidden state继续前向
2. Qwen3的shared_semantic主轴在末层支撑边界的机制未分解
3. vehicle/emotion/action等例外类别的机制未解释

Phase 494核心实验:
- Exp1: 真正跨层因果干预 — 在L(n-2)修改hidden state → 继续forward到L(n-1) → 追踪L(n-1)实际变化
- Exp2: shared_semantic主轴logit分解 — 将shared方向直接分解为对各类别logit的贡献
- Exp3: 异常类别深度分析 — vehicle/emotion/container/action的多层轨迹
- Exp4: 语义类型分组统计 — 实体/人工物/抽象/动作/材料的释放模式

用法:
  python tests/glm5/phase494_causal_transfer_shared_decomp.py qwen3 1
  python tests/glm5/phase494_causal_transfer_shared_decomp.py glm4 1
  python tests/glm5/phase494_causal_transfer_shared_decomp.py deepseek7b 1
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

# 语义类型分组
SEMANTIC_GROUPS = {
    "natural_entity": ["fruit", "animal", "plant", "body_part"],
    "artifact":       ["clothing", "vehicle", "tool", "furniture", "building", "container", "device"],
    "abstract":       ["emotion"],
    "action":         ["action"],
    "substance":      ["food", "material"],
    "location":       ["place"],
}


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


def get_category_hidden(model, tokenizer, device, objects, layer_idx,
                        template="The {obj} is a kind of"):
    """获取某类在某层的所有hidden states"""
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
    return h_last.numpy()


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


# ==================== Exp1: 真正跨层因果干预 ====================
def exp1_causal_cross_layer(model, tokenizer, device, model_name, W_U):
    """
    核心突破: 在L(n-2)修改hidden state → 继续forward到L(n-1) → 追踪L(n-1)实际变化
    
    方法:
    1. 正常前向传播, 捕获L(n-2)的hidden state
    2. 在L(n-2)修改hidden state (ablate/double/reverse shared)
    3. 从L(n-2)开始继续前向传播到L(n-1)
    4. 捕获L(n-1)的hidden state, 观察shared/DCF变化
    
    这是真正的因果测试!
    """
    plog(f"=== Exp1: True Causal Cross-layer Intervention ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n2 = n_layers - 2
    L_n1 = n_layers - 1
    layers = get_layers(model)

    # 选6个关键类别: 3个正常反转 + 3个异常
    test_cats = ["fruit", "clothing", "animal", "vehicle", "container", "emotion"]
    other_cats_pool = CAT_NAMES_16.copy()

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()
        cat_objs = CATEGORIES_16[cat_name][:12]
        other_cats = [c for c in other_cats_pool if c != cat_name][:7]

        # Step 1: 获取自然状态下L(n-2)和L(n-1)的hidden states
        # 使用output_hidden_states方式获取所有层
        inputs = encode_prompts(tokenizer, device, cat_objs)

        # 自然前向: 收集L(n-2)和L(n-1)的输出
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

        # 取每个样本最后token
        h_n2_all = h_n2_natural[torch.arange(h_n2_natural.size(0)), last_idx.cpu()].numpy()
        h_n1_all = h_n1_natural[torch.arange(h_n1_natural.size(0)), last_idx.cpu()].numpy()
        h_n2_mean = np.mean(h_n2_all, axis=0)
        h_n1_mean = np.mean(h_n1_all, axis=0)

        # 获取其他类别在L(n-2)的hidden
        other_h_n2 = {}
        other_h_n1 = {}
        for oc in other_cats:
            oc_inputs = encode_prompts(tokenizer, device, CATEGORIES_16[oc][:8])
            oc_captured = {}
            def make_oc_hook(key):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        oc_captured[key] = output[0].detach().float().cpu()
                    else:
                        oc_captured[key] = output.detach().float().cpu()
                return hook
            h_n2_oc = layers[L_n2].register_forward_hook(make_oc_hook(f"L{L_n2}"))
            h_n1_oc = layers[L_n1].register_forward_hook(make_oc_hook(f"L{L_n1}"))
            with torch.no_grad():
                model(**oc_inputs)
            h_n2_oc.remove()
            h_n1_oc.remove()

            oc_mask = oc_inputs["attention_mask"].bool()
            oc_last_idx = oc_mask.sum(dim=1) - 1
            other_h_n2[oc] = oc_captured[f"L{L_n2}"][torch.arange(oc_captured[f"L{L_n2}"].size(0)), oc_last_idx.cpu()].numpy()
            other_h_n1[oc] = oc_captured[f"L{L_n1}"][torch.arange(oc_captured[f"L{L_n1}"].size(0)), oc_last_idx.cpu()].numpy()

        h_others_n2 = np.array([np.mean(other_h_n2[oc], axis=0) for oc in other_cats])
        h_others_n1 = np.array([np.mean(other_h_n1[oc], axis=0) for oc in other_cats])

        # Step 2: 计算shared方向
        shared_dir_n2, Bc_n2 = get_shared_direction(h_n2_mean, h_others_n2)
        shared_dir_n1, Bc_n1 = get_shared_direction(h_n1_mean, h_others_n1)

        # 自然状态基线
        baseline_D_n1 = compute_target_logit(h_n1_mean, W_U, cat_name, CAT_NAMES_16)
        baseline_D_n2 = compute_target_logit(h_n2_mean, W_U, cat_name, CAT_NAMES_16)
        h_ablate_n1 = ablate_direction(h_n1_mean, shared_dir_n1)
        ablate_shared_n1_delta = compute_target_logit(h_ablate_n1, W_U, cat_name, CAT_NAMES_16) - baseline_D_n1

        # Step 3: 真正的跨层因果干预
        # 在L(n-2)修改hidden state → 用hook注入修改后的值 → 继续forward → 捕获L(n-1)
        interventions = {
            "ablate_shared": lambda h, d: h - np.dot(h, d) * d,  # 消融shared分量
            "double_shared": lambda h, d: h + np.dot(h, d) * d,  # 加倍shared分量
            "reverse_shared": lambda h, d: h - 2 * np.dot(h, d) * d,  # 反转shared分量
        }

        intervention_results = {}
        for intv_name, intv_fn in interventions.items():
            # 修改L(n-2)的hidden state (使用均值)
            h_n2_modified = intv_fn(h_n2_mean, shared_dir_n2)

            # 将修改后的值注入: 使用hook替换L(n-2)的输出
            # 然后继续forward, 捕获L(n-1)的输出
            modified_tensor = torch.tensor(
                h_n2_modified.reshape(1, 1, -1),
                dtype=torch.bfloat16, device=device
            )

            captured_causal = {}
            causal_n1_delta = None

            # 方法: 在L(n-2)的hook中替换输出, 在L(n-1)的hook中捕获
            def make_replace_hook(replacement_val, seq_pos):
                """替换指定位置的hidden state"""
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        # 只替换最后一个token位置
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

            # 确定seq_pos (最后一个token的位置)
            # inputs中的attention_mask告诉我们最后一个token位置
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

                # L(n-1)的DCF变化
                causal_D_n1 = compute_target_logit(h_n1_causal_last, W_U, cat_name, CAT_NAMES_16)
                delta_D_n1 = causal_D_n1 - baseline_D_n1

                # L(n-1)的shared投影变化
                proj_shared_n1_natural = float(np.dot(h_n1_mean, shared_dir_n1))
                proj_shared_n1_causal = float(np.dot(h_n1_causal_last, shared_dir_n1))
                delta_proj_shared_n1 = proj_shared_n1_causal - proj_shared_n1_natural

                # L(n-2)的shared投影变化
                proj_shared_n2_natural = float(np.dot(h_n2_mean, shared_dir_n2))
                proj_shared_n2_causal = float(np.dot(h_n2_modified, shared_dir_n2))
                delta_proj_shared_n2 = proj_shared_n2_causal - proj_shared_n2_natural

                # 因果传递率: L(n-1)shared变化 / L(n-2)shared变化
                if abs(delta_proj_shared_n2) > 1e-6:
                    causal_transfer_ratio = delta_proj_shared_n1 / delta_proj_shared_n2
                else:
                    causal_transfer_ratio = 0.0

                # L(n-1)残差范数变化
                norm_n1_natural = float(np.linalg.norm(h_n1_mean))
                norm_n1_causal = float(np.linalg.norm(h_n1_causal_last))

                intervention_results[intv_name] = {
                    "delta_proj_shared_n2": float(delta_proj_shared_n2),
                    "delta_proj_shared_n1": float(delta_proj_shared_n1),
                    "causal_transfer_ratio": float(causal_transfer_ratio),
                    "delta_D_n1": float(delta_D_n1),
                    "delta_D_n2": float(compute_target_logit(h_n2_modified, W_U, cat_name, CAT_NAMES_16) - baseline_D_n2),
                    "norm_n1_natural": float(norm_n1_natural),
                    "norm_n1_causal": float(norm_n1_causal),
                }

                plog(f"    {intv_name}: Δproj_n2={delta_proj_shared_n2:+.1f}, "
                     f"Δproj_n1={delta_proj_shared_n1:+.1f}, "
                     f"ratio={causal_transfer_ratio:.3f}, "
                     f"ΔD_n1={delta_D_n1:+.2f}")
            else:
                plog(f"    {intv_name}: FAILED - L(n-1) not captured")
                intervention_results[intv_name] = {"error": "L(n-1) not captured"}

        cat_results = {
            "baseline_D_n1": float(baseline_D_n1),
            "baseline_D_n2": float(baseline_D_n2),
            "ablate_shared_n1_delta": float(ablate_shared_n1_delta),
            "proj_shared_n2": float(np.dot(h_n2_mean, shared_dir_n2)),
            "proj_shared_n1": float(np.dot(h_n1_mean, shared_dir_n1)),
            "interventions": intervention_results,
            "elapsed": time.time() - t0,
        }
        results[cat_name] = cat_results

    # 汇总
    plog(f"\n  === Exp1 Causal Transfer Summary ===")
    for cat_name, data in results.items():
        intv = data["interventions"]
        if "ablate_shared" in intv and "error" not in intv["ablate_shared"]:
            ablate = intv["ablate_shared"]
            plog(f"  {cat_name}: ablate ratio={ablate['causal_transfer_ratio']:.3f}, "
                 f"ΔD_n1={ablate['delta_D_n1']:+.2f}")

    return results


# ==================== Exp2: shared_semantic主轴logit分解 ====================
def exp2_shared_logit_decomposition(model, tokenizer, device, model_name, W_U):
    """
    直接分解shared_semantic方向对各类别logit的贡献
    
    方法:
    1. 在L(n-1), 计算shared方向 → W_U @ shared_dir 给出每个类别的logit贡献
    2. 分析shared方向主要支撑哪些类别、抑制哪些类别
    3. 对比L(n-2)和L(n-1), 看shared方向的logit贡献如何跨层变化
    """
    plog(f"=== Exp2: Shared Semantic Logit Decomposition ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n2 = n_layers - 2
    L_n1 = n_layers - 1

    test_cats = CAT_NAMES_16
    other_cats_pool = CAT_NAMES_16.copy()

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()
        cat_objs = CATEGORIES_16[cat_name][:12]
        other_cats = [c for c in other_cats_pool if c != cat_name][:7]

        for li in [L_n2, L_n1]:
            h_target_all = get_category_hidden(model, tokenizer, device, cat_objs, li)
            h_target = np.mean(h_target_all, axis=0)

            other_hiddens = {}
            for oc in other_cats:
                other_hiddens[oc] = get_category_hidden(
                    model, tokenizer, device, CATEGORIES_16[oc][:8], li
                )
            h_others = np.array([np.mean(other_hiddens[oc], axis=0) for oc in other_cats])

            shared_dir, Bc_norm = get_shared_direction(h_target, h_others)

            # shared方向对所有类别的logit贡献
            shared_logit_contribution = W_U @ shared_dir  # [vocab_size]

            # 对目标类别和竞争类别的贡献
            target_logit_contrib = float(compute_target_logit(shared_dir, W_U, cat_name, CAT_NAMES_16))

            # 竞争类别贡献
            competitor_contribs = {}
            for oc in other_cats:
                oc_contrib = float(compute_target_logit(shared_dir, W_U, oc, CAT_NAMES_16))
                competitor_contribs[oc] = oc_contrib

            # shared方向在logit空间的主成分
            # 取top-k类别(按绝对贡献排序)
            abs_contrib = np.abs(shared_logit_contribution)
            top_k = 10
            top_indices = np.argsort(abs_contrib)[-top_k:][::-1]

            # shared方向的自然投影系数
            proj_coeff = float(np.dot(h_target, shared_dir))

            # shared对总logit的贡献比例
            total_logits = W_U @ h_target
            shared_logits = proj_coeff * shared_logit_contribution
            shared_variance_ratio = float(
                np.sum(shared_logits**2) / (np.sum(total_logits**2) + 1e-10)
            )

            key = f"L{li}"
            if cat_name not in results:
                results[cat_name] = {}

            results[cat_name][key] = {
                "target_logit_contrib": float(target_logit_contrib),
                "competitor_contribs": competitor_contribs,
                "proj_coeff": float(proj_coeff),
                "shared_variance_ratio": float(shared_variance_ratio),
                "top10_logit_categories": [
                    {
                        "idx": int(idx),
                        "category": CAT_NAMES_16[idx] if idx < len(CAT_NAMES_16) else f"cat_{idx}",
                        "contribution": float(shared_logit_contribution[idx]),
                    }
                    for idx in top_indices if idx < len(CAT_NAMES_16)
                ],
            }

        elapsed = time.time() - t0
        results[cat_name]["elapsed"] = elapsed

        # 跨层变化: L(n-2) → L(n-1)的shared方向logit贡献变化
        ln2_data = results[cat_name][f"L{L_n2}"]
        ln1_data = results[cat_name][f"L{L_n1}"]
        results[cat_name]["cross_layer_change"] = {
            "target_contrib_delta": ln1_data["target_logit_contrib"] - ln2_data["target_logit_contrib"],
            "proj_coeff_ratio": ln1_data["proj_coeff"] / (ln2_data["proj_coeff"] + 1e-10),
            "shared_variance_delta": ln1_data["shared_variance_ratio"] - ln2_data["shared_variance_ratio"],
        }

        plog(f"    L{n_layers-2}: target_contrib={ln2_data['target_logit_contrib']:+.4f}, "
             f"proj={ln2_data['proj_coeff']:.1f}")
        plog(f"    L{n_layers-1}: target_contrib={ln1_data['target_logit_contrib']:+.4f}, "
             f"proj={ln1_data['proj_coeff']:.1f}")
        plog(f"    Δtarget={results[cat_name]['cross_layer_change']['target_contrib_delta']:+.4f}")

    return results


# ==================== Exp3: 异常类别多层轨迹 ====================
def exp3_exception_category_trajectory(model, tokenizer, device, model_name, W_U):
    """
    对vehicle/emotion/container/action做深层轨迹分析:
    - 最后4层的shared方向DCF
    - 每层的ablate_shared_delta
    - 每层的shared方向logit贡献
    判断边界是否在更早层释放,或根本不需要释放
    """
    plog(f"=== Exp3: Exception Category Multi-layer Trajectory ===")
    n_layers = get_model_info(model, model_name).n_layers

    # 异常类别 + 对照组
    test_cats = ["fruit", "vehicle", "container", "emotion", "action"]
    other_cats_pool = CAT_NAMES_16.copy()

    # 测最后6层
    test_layers = list(range(max(0, n_layers - 6), n_layers))

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()
        cat_objs = CATEGORIES_16[cat_name][:12]
        other_cats = [c for c in other_cats_pool if c != cat_name][:7]

        layer_results = {}
        for li in test_layers:
            h_target_all = get_category_hidden(model, tokenizer, device, cat_objs, li)
            h_target = np.mean(h_target_all, axis=0)

            other_hiddens = {}
            for oc in other_cats:
                other_hiddens[oc] = get_category_hidden(
                    model, tokenizer, device, CATEGORIES_16[oc][:8], li
                )
            h_others = np.array([np.mean(other_hiddens[oc], axis=0) for oc in other_cats])

            shared_dir, Bc_norm = get_shared_direction(h_target, h_others)

            # ablate_shared
            baseline_D = compute_target_logit(h_target, W_U, cat_name, CAT_NAMES_16)
            h_ablate = ablate_direction(h_target, shared_dir)
            ablate_shared_delta = compute_target_logit(h_ablate, W_U, cat_name, CAT_NAMES_16) - baseline_D

            # shared方向的logit贡献
            target_logit_contrib = float(compute_target_logit(shared_dir, W_U, cat_name, CAT_NAMES_16))

            # 投影系数
            proj_coeff = float(np.dot(h_target, shared_dir))
            residual_norm = float(np.linalg.norm(h_target))

            layer_results[f"L{li}"] = {
                "ablate_shared_delta": float(ablate_shared_delta),
                "target_logit_contrib": float(target_logit_contrib),
                "proj_coeff": float(proj_coeff),
                "residual_norm": float(residual_norm),
            }

            plog(f"    L{li}: ablate_shared={ablate_shared_delta:+.2f}, "
                 f"target_contrib={target_logit_contrib:+.4f}, "
                 f"proj={proj_coeff:.1f}")

        # 找反转点(如果有的话)
        reversal_layer = None
        for i in range(len(test_layers) - 1):
            li_curr = test_layers[i]
            li_next = test_layers[i + 1]
            delta_curr = layer_results[f"L{li_curr}"]["ablate_shared_delta"]
            delta_next = layer_results[f"L{li_next}"]["ablate_shared_delta"]
            if delta_curr > 0 and delta_next < 0:
                reversal_layer = li_next
                break

        results[cat_name] = {
            "layers": layer_results,
            "reversal_layer": reversal_layer,
            "elapsed": time.time() - t0,
        }

        if reversal_layer:
            plog(f"  → {cat_name} reversal at L{reversal_layer}")
        else:
            last_delta = layer_results[f"L{test_layers[-1]}"]["ablate_shared_delta"]
            plog(f"  → {cat_name} NO reversal (last delta={last_delta:+.2f})")

    return results


# ==================== Exp4: 语义类型分组统计 ====================
def exp4_semantic_group_statistics(model, tokenizer, device, model_name, W_U):
    """
    按语义类型分组统计释放模式
    """
    plog(f"=== Exp4: Semantic Group Statistics ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n2 = n_layers - 2
    L_n1 = n_layers - 1

    results = {}
    for group_name, group_cats in SEMANTIC_GROUPS.items():
        plog(f"  Group: {group_name}")
        group_reversals = 0
        group_data = {}

        for cat_name in group_cats:
            cat_objs = CATEGORIES_16[cat_name][:12]
            other_cats = [c for c in CAT_NAMES_16 if c != cat_name][:7]

            # L(n-2)和L(n-1)的ablate_shared
            deltas = {}
            for li in [L_n2, L_n1]:
                h_target_all = get_category_hidden(model, tokenizer, device, cat_objs, li)
                h_target = np.mean(h_target_all, axis=0)

                other_hiddens = {}
                for oc in other_cats:
                    other_hiddens[oc] = get_category_hidden(
                        model, tokenizer, device, CATEGORIES_16[oc][:8], li
                    )
                h_others = np.array([np.mean(other_hiddens[oc], axis=0) for oc in other_cats])

                shared_dir, _ = get_shared_direction(h_target, h_others)
                baseline_D = compute_target_logit(h_target, W_U, cat_name, CAT_NAMES_16)
                h_ablate = ablate_direction(h_target, shared_dir)
                deltas[f"L{li}"] = compute_target_logit(h_ablate, W_U, cat_name, CAT_NAMES_16) - baseline_D

            is_reversal = deltas[f"L{L_n2}"] > 0 and deltas[f"L{L_n1}"] < 0
            if is_reversal:
                group_reversals += 1

            group_data[cat_name] = {
                "Ln2_delta": float(deltas[f"L{L_n2}"]),
                "Ln1_delta": float(deltas[f"L{L_n1}"]),
                "is_reversal": is_reversal,
            }

        results[group_name] = {
            "categories": group_data,
            "reversal_count": group_reversals,
            "total_count": len(group_cats),
            "reversal_rate": group_reversals / len(group_cats),
        }

        plog(f"    Reversal: {group_reversals}/{len(group_cats)} "
             f"({group_reversals/len(group_cats)*100:.0f}%)")

    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    plog(f"Phase 494: {model_name}, round={round_num}")

    # 加载模型
    model, tokenizer, device = get_model_and_tokenizer(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # 获取W_U
    W_U = get_W_U(model, model_name)
    plog(f"W_U: shape={W_U.shape}")

    all_results = {
        "phase": 494,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        }
    }

    # Exp1: 真正跨层因果干预
    plog(f"\n{'='*60}")
    all_results["exp1_causal_cross_layer"] = exp1_causal_cross_layer(
        model, tokenizer, device, model_name, W_U
    )

    # Exp2: shared主轴logit分解
    plog(f"\n{'='*60}")
    all_results["exp2_shared_logit_decomposition"] = exp2_shared_logit_decomposition(
        model, tokenizer, device, model_name, W_U
    )

    # Exp3: 异常类别多层轨迹
    plog(f"\n{'='*60}")
    all_results["exp3_exception_trajectory"] = exp3_exception_category_trajectory(
        model, tokenizer, device, model_name, W_U
    )

    # Exp4: 语义类型分组统计
    plog(f"\n{'='*60}")
    all_results["exp4_semantic_groups"] = exp4_semantic_group_statistics(
        model, tokenizer, device, model_name, W_U
    )

    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase494_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")

    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Model released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    plog(f"Phase 494 {model_name} complete!")


if __name__ == "__main__":
    main()
