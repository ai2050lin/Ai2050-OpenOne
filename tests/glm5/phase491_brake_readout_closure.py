"""
Phase 491: 末层释放机制、倒数第二层刹车因果闭环与关系槽位调制
================================================================

Phase 490核心发现:
1. shared_semantic消融效应随层位形成倒U型曲线, 末层剧烈反转
2. 末层-1 orth_bc抑制成分主导, 末层支撑成分主导
3. DS7B末层极端放大(+93.5→-121.1)
4. GLM4末层刹车减弱但未完全反转
5. ablate_shared ≈ ablate_orth_bc (几乎完全重合!)
6. 竞争类别控制在末层集中生效

Phase 491核心目标:
- Exp1: R2确认L(n-2)/L(n-1)反转, 增加对象数(16个/类)
- Exp2: L(n-2)刹车闭环: ablate/inject/double/reverse抑制方向
- Exp3: L(n-1)读出闭环: ablate/inject/double/reverse支撑方向
- Exp4: 关系槽位调制: kind_of/used_for/found_in下测L(n-2)/L(n-1)

用法:
  python tests/glm5/phase491_brake_readout_closure.py qwen3 1
  python tests/glm5/phase491_brake_readout_closure.py glm4 1
  python tests/glm5/phase491_brake_readout_closure.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS, safe_decode)


def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES = {
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
}

CAT_NAMES = list(CATEGORIES.keys())

# 关系模板
RELATION_TEMPLATES = {
    "kind_of": "The {obj} is a kind of",
    "used_for": "The {obj} is used for",
    "found_in": "The {obj} is found in",
}


def get_model_and_tokenizer(model_name):
    """BF16加载模型, 参考model_demo_bf16.py"""
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
    """编码提示"""
    texts = [template.format(obj=obj) for obj in objects]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }


def get_category_centers(model, tokenizer, device, categories, layer_idx, W_U, template="The {obj} is a kind of"):
    """获取每个类别在某层的hidden state中心"""
    layers = get_layers(model)
    cat_centers = {}
    cat_hidden = {}

    for cat_name, objects in categories.items():
        # 根据round调整训练集大小
        train_objs = objects[:12]  # R2用更多对象
        inputs = encode_prompts(tokenizer, device, train_objs, template)

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
        center = h_last.mean(dim=0)

        cat_centers[cat_name] = center.numpy()
        cat_hidden[cat_name] = h_last.numpy()

    return cat_centers, cat_hidden


def compute_dcf_from_center(h_center, W_U, cat_name, cat_names):
    """计算DCF"""
    logit = W_U @ h_center
    target_idx = cat_names.index(cat_name)
    target_D = logit[target_idx]
    all_D = {cn: logit[cat_names.index(cn)] for cn in cat_names}
    return target_D, all_D


def ablate_direction(h, direction):
    """消融h在direction方向的投影"""
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    proj = np.dot(h, d_norm) * d_norm
    return h - proj


def inject_direction(h, direction, scale=1.0):
    """注入direction到h"""
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    return h + scale * d_norm


def reverse_direction(h, direction):
    """反转h在direction方向的投影"""
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    proj = np.dot(h, d_norm) * d_norm
    return h - 2 * proj


def double_direction(h, direction):
    """加倍h在direction方向的投影"""
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    proj = np.dot(h, d_norm) * d_norm
    return h + proj


def get_bc_and_orth_dirs(h_target, h_others, n_dirs=8):
    """获取B_c方向和orth子空间分解"""
    others_mean = np.mean(h_others, axis=0)
    B_c = h_target - others_mean
    Bc_norm = B_c / (np.linalg.norm(B_c) + 1e-10)

    # 构造对比矩阵(正交于B_c)
    contrast_matrix = []
    for h_c in np.vstack([h_target[np.newaxis], h_others]):
        diff = h_c - others_mean
        diff_orth = diff - np.dot(diff, Bc_norm) * Bc_norm
        if np.linalg.norm(diff_orth) > 1e-6:
            contrast_matrix.append(diff_orth)

    if len(contrast_matrix) > 0:
        M = np.stack(contrast_matrix)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        orth_subspace = Vt[:n_dirs]
        singular_values = S[:n_dirs]
    else:
        orth_subspace = np.eye(len(B_c))[:n_dirs]
        singular_values = np.ones(n_dirs)

    return Bc_norm, orth_subspace, singular_values


def classify_orth_dirs(h_target, orth_subspace, W_U, cat_name, threshold=0.05):
    """将orth子空间方向分类为support/inhibit/neutral"""
    baseline_target_D, _ = compute_dcf_from_center(h_target, W_U, cat_name, CAT_NAMES)

    support_dirs = []
    inhibit_dirs = []
    neutral_dirs = []

    for i, d in enumerate(orth_subspace):
        h_mod = ablate_direction(h_target, d)
        target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
        delta = target_D_after - baseline_target_D

        info = {
            "dir_idx": i,
            "ablate_delta": float(delta),
            "direction": d,
        }

        if delta < -threshold:
            support_dirs.append(info)
        elif delta > threshold:
            inhibit_dirs.append(info)
        else:
            neutral_dirs.append(info)

    return support_dirs, inhibit_dirs, neutral_dirs


# ==================== Exp1: R2确认L(n-2)/L(n-1)反转 ====================
def exp1_r2_confirmation(model, tokenizer, device, model_name, W_U, round_num):
    """R2确认: 用更多对象和更多类别验证末层反转"""
    plog(f"=== Exp1: R2 Confirmation of L(n-2)/L(n-1) Reversal ===")
    n_layers = get_model_info(model, model_name).n_layers

    # 所有模型测3个类别
    test_cats = ["fruit", "clothing", "food"]

    # 关键层位: 早中晚+末3层
    test_layers = list(range(0, n_layers, max(1, n_layers // 8))) + list(range(max(0, n_layers-3), n_layers))
    test_layers = sorted(set(test_layers))

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        cat_results = {}

        for li in test_layers:
            t0 = time.time()
            test_cats_dict = {cn: CATEGORIES[cn] for cn in [cat_name] + [c for c in CAT_NAMES if c != cat_name][:7]}
            cat_centers, cat_hidden = get_category_centers(
                model, tokenizer, device, test_cats_dict, li, W_U
            )

            h_target = cat_centers[cat_name]
            other_cats = [c for c in test_cats_dict if c != cat_name]
            h_others = np.array([cat_centers[c] for c in other_cats])

            others_mean = np.mean(h_others, axis=0)
            B_c = h_target - others_mean
            Bc_norm = B_c / (np.linalg.norm(B_c) + 1e-10)

            # shared_semantic方向(所有类别中心的共享方向, 正交于B_c)
            all_mean = np.mean(np.vstack([h_target[np.newaxis], h_others]), axis=0)
            shared_dir = all_mean / (np.linalg.norm(all_mean) + 1e-10)
            shared_dir = shared_dir - np.dot(shared_dir, Bc_norm) * Bc_norm
            if np.linalg.norm(shared_dir) > 1e-6:
                shared_dir = shared_dir / np.linalg.norm(shared_dir)

            # Baseline
            baseline_target_D, _ = compute_dcf_from_center(h_target, W_U, cat_name, CAT_NAMES)

            # 1. ablate_shared
            h_mod = ablate_direction(h_target, shared_dir)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            shared_delta = float(target_D_after - baseline_target_D)

            # 2. inject_shared (注入1倍shared方向)
            h_mod = inject_direction(h_target, shared_dir, scale=1.0)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            inject_shared_delta = float(target_D_after - baseline_target_D)

            # 3. double_shared (加倍shared方向的投影)
            h_mod = double_direction(h_target, shared_dir)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            double_shared_delta = float(target_D_after - baseline_target_D)

            # 4. ablate_orth_bc (只保留B_c投影)
            proj_bc = np.dot(h_target, Bc_norm) * Bc_norm
            target_D_after, _ = compute_dcf_from_center(proj_bc, W_U, cat_name, CAT_NAMES)
            orth_bc_delta = float(target_D_after - baseline_target_D)

            elapsed = time.time() - t0
            layer_res = {
                "ablate_shared": shared_delta,
                "inject_shared": inject_shared_delta,
                "double_shared": double_shared_delta,
                "ablate_orth_bc": orth_bc_delta,
                "baseline": float(baseline_target_D),
                "elapsed": elapsed,
            }
            cat_results[f"L{li}"] = layer_res

            if li % 6 == 0 or li >= n_layers - 3:
                plog(f"    L{li}: ablate_shared={shared_delta:+.3f}, inject_shared={inject_shared_delta:+.3f}, "
                     f"double_shared={double_shared_delta:+.3f}, ablate_orth={orth_bc_delta:+.3f}")

        results[cat_name] = cat_results

    return results


# ==================== Exp2: L(n-2)刹车闭环 ====================
def exp2_brake_closure(model, tokenizer, device, model_name, W_U, round_num):
    """L(n-2)刹车因果闭环: ablate/inject/double/reverse抑制方向"""
    plog(f"=== Exp2: L(n-2) Brake Closure ===")
    n_layers = get_model_info(model, model_name).n_layers

    test_cats = ["fruit", "clothing"] if model_name != "deepseek7b" else ["fruit", "food"]

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        cat_results = {}

        # 在L(n-2)做闭环
        li = n_layers - 2
        plog(f"  Testing L{li} (n_layers={n_layers})")

        test_cats_dict = {cn: CATEGORIES[cn] for cn in [cat_name] + [c for c in CAT_NAMES if c != cat_name][:7]}
        cat_centers, cat_hidden = get_category_centers(
            model, tokenizer, device, test_cats_dict, li, W_U
        )

        h_target = cat_centers[cat_name]
        other_cats = [c for c in test_cats_dict if c != cat_name]
        h_others = np.array([cat_centers[c] for c in other_cats])

        Bc_norm, orth_subspace, singular_values = get_bc_and_orth_dirs(h_target, h_others)

        # 分类orth方向
        support_dirs, inhibit_dirs, neutral_dirs = classify_orth_dirs(
            h_target, orth_subspace, W_U, cat_name, threshold=0.05
        )

        plog(f"    L{li} dirs: support={len(support_dirs)}, inhibit={len(inhibit_dirs)}, neutral={len(neutral_dirs)}")

        baseline_target_D, _ = compute_dcf_from_center(h_target, W_U, cat_name, CAT_NAMES)

        # 对抑制方向组做4种操作
        # 1. 合并所有抑制方向为一个方向(它们的加权平均)
        if len(inhibit_dirs) > 0:
            inhibit_directions = [d["direction"] for d in inhibit_dirs]
            # 主抑制方向: 加权平均
            combined_inhibit = np.zeros_like(h_target)
            for d_info in inhibit_dirs:
                combined_inhibit += d_info["ablate_delta"] * d_info["direction"]
            if np.linalg.norm(combined_inhibit) > 1e-6:
                combined_inhibit = combined_inhibit / np.linalg.norm(combined_inhibit)
            else:
                combined_inhibit = inhibit_directions[0]
        else:
            combined_inhibit = None

        operations = {}

        if combined_inhibit is not None:
            # ablate: 移除抑制方向
            h_mod = ablate_direction(h_target, combined_inhibit)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            operations["ablate_inhibit"] = {
                "target_delta": float(target_D_after - baseline_target_D),
                "desc": "消融抑制方向 → 若边界增强, 证明抑制方向在压制边界"
            }

            # inject: 注入抑制方向
            h_mod = inject_direction(h_target, combined_inhibit, scale=1.0)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            operations["inject_inhibit"] = {
                "target_delta": float(target_D_after - baseline_target_D),
                "desc": "注入抑制方向 → 若边界削弱, 证明抑制方向在压制边界"
            }

            # double: 加倍抑制方向
            h_mod = double_direction(h_target, combined_inhibit)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            operations["double_inhibit"] = {
                "target_delta": float(target_D_after - baseline_target_D),
                "desc": "加倍抑制方向 → 若边界更弱, 证明抑制方向在压制边界"
            }

            # reverse: 反转抑制方向
            h_mod = reverse_direction(h_target, combined_inhibit)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            operations["reverse_inhibit"] = {
                "target_delta": float(target_D_after - baseline_target_D),
                "desc": "反转抑制方向 → 若边界增强, 证明抑制方向在压制边界"
            }

        # 对单个最强抑制方向也测试
        if len(inhibit_dirs) > 0:
            strongest_inhibit = max(inhibit_dirs, key=lambda x: x["ablate_delta"])
            d_strongest = orth_subspace[strongest_inhibit["dir_idx"]]

            for op_name, op_func in [("ablate", ablate_direction), ("inject", inject_direction),
                                      ("double", double_direction), ("reverse", reverse_direction)]:
                if op_name == "inject":
                    h_mod = op_func(h_target, d_strongest, scale=1.0)
                else:
                    h_mod = op_func(h_target, d_strongest)
                target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
                operations[f"strongest_inhibit_{op_name}"] = {
                    "target_delta": float(target_D_after - baseline_target_D),
                    "dir_idx": strongest_inhibit["dir_idx"],
                    "dir_ablate_delta": strongest_inhibit["ablate_delta"],
                }

        cat_results[f"L{li}"] = {
            "n_support": len(support_dirs),
            "n_inhibit": len(inhibit_dirs),
            "n_neutral": len(neutral_dirs),
            "operations": operations,
            "baseline": float(baseline_target_D),
        }

        ops_str = ', '.join(f'{k}={v["target_delta"]:+.3f}' for k,v in operations.items() if isinstance(v, dict) and 'target_delta' in v)
        plog(f"    L{li} results: {ops_str}")

        # 也在L(n-1)做对比测试(验证支撑方向)
        li2 = n_layers - 1
        cat_centers2, cat_hidden2 = get_category_centers(
            model, tokenizer, device, test_cats_dict, li2, W_U
        )
        h_target2 = cat_centers2[cat_name]
        h_others2 = np.array([cat_centers2[c] for c in other_cats])

        Bc_norm2, orth_subspace2, sv2 = get_bc_and_orth_dirs(h_target2, h_others2)
        support2, inhibit2, neutral2 = classify_orth_dirs(h_target2, orth_subspace2, W_U, cat_name, 0.05)

        baseline2, _ = compute_dcf_from_center(h_target2, W_U, cat_name, CAT_NAMES)

        # 对支撑方向做闭环
        if len(support2) > 0:
            combined_support = np.zeros_like(h_target2)
            for d_info in support2:
                combined_support += abs(d_info["ablate_delta"]) * d_info["direction"]
            if np.linalg.norm(combined_support) > 1e-6:
                combined_support = combined_support / np.linalg.norm(combined_support)
            else:
                combined_support = support2[0]["direction"]
        else:
            combined_support = None

        ops2 = {}
        if combined_support is not None:
            # ablate
            h_mod = ablate_direction(h_target2, combined_support)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            ops2["ablate_support"] = {
                "target_delta": float(target_D_after - baseline2),
                "desc": "消融支撑方向 → 若边界削弱, 证明支撑方向在维持边界"
            }

            # inject
            h_mod = inject_direction(h_target2, combined_support, scale=1.0)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            ops2["inject_support"] = {
                "target_delta": float(target_D_after - baseline2),
                "desc": "注入支撑方向 → 若边界增强, 证明支撑方向在维持边界"
            }

            # double
            h_mod = double_direction(h_target2, combined_support)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            ops2["double_support"] = {
                "target_delta": float(target_D_after - baseline2),
                "desc": "加倍支撑方向 → 若边界更强, 证明支撑方向在维持边界"
            }

            # reverse
            h_mod = reverse_direction(h_target2, combined_support)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            ops2["reverse_support"] = {
                "target_delta": float(target_D_after - baseline2),
                "desc": "反转支撑方向 → 若边界削弱, 证明支撑方向在维持边界"
            }

        cat_results[f"L{li2}"] = {
            "n_support": len(support2),
            "n_inhibit": len(inhibit2),
            "n_neutral": len(neutral2),
            "operations": ops2,
            "baseline": float(baseline2),
        }

        ops2_str = ', '.join(f'{k}={v["target_delta"]:+.3f}' for k,v in ops2.items() if isinstance(v, dict) and 'target_delta' in v)
        plog(f"    L{li2} results: {ops2_str}")

        results[cat_name] = cat_results

    return results


# ==================== Exp3: 末层读出支撑闭环 ====================
def exp3_readout_closure(model, tokenizer, device, model_name, W_U, round_num):
    """L(n-1)读出支撑闭环: ablate/inject/double/reverse支撑方向(更多类别)"""
    plog(f"=== Exp3: L(n-1) Readout Support Closure ===")
    n_layers = get_model_info(model, model_name).n_layers

    # 扩展到4个类别
    if model_name == "deepseek7b":
        test_cats = ["fruit", "food", "animal", "tool"]
    else:
        test_cats = ["fruit", "clothing", "food", "animal"]

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")

        li = n_layers - 1
        test_cats_dict = {cn: CATEGORIES[cn] for cn in [cat_name] + [c for c in CAT_NAMES if c != cat_name][:7]}
        cat_centers, cat_hidden = get_category_centers(
            model, tokenizer, device, test_cats_dict, li, W_U
        )

        h_target = cat_centers[cat_name]
        other_cats = [c for c in test_cats_dict if c != cat_name]
        h_others = np.array([cat_centers[c] for c in other_cats])

        Bc_norm, orth_subspace, sv = get_bc_and_orth_dirs(h_target, h_others)
        support_dirs, inhibit_dirs, neutral_dirs = classify_orth_dirs(
            h_target, orth_subspace, W_U, cat_name, threshold=0.05
        )

        baseline_target_D, _ = compute_dcf_from_center(h_target, W_U, cat_name, CAT_NAMES)

        plog(f"    L{li}: support={len(support_dirs)}, inhibit={len(inhibit_dirs)}, neutral={len(neutral_dirs)}")

        # 对每个支撑方向单独测试
        dir_tests = []
        for sd in support_dirs:
            d = orth_subspace[sd["dir_idx"]]
            ops = {}
            for op_name, op_func in [("ablate", ablate_direction), ("inject", inject_direction),
                                      ("double", double_direction), ("reverse", reverse_direction)]:
                if op_name == "inject":
                    h_mod = op_func(h_target, d, scale=1.0)
                else:
                    h_mod = op_func(h_target, d)
                target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
                ops[op_name] = float(target_D_after - baseline_target_D)
            dir_tests.append({
                "dir_idx": sd["dir_idx"],
                "original_ablate_delta": sd["ablate_delta"],
                "operations": ops,
            })

        # 对每个抑制方向也测试
        for id_info in inhibit_dirs:
            d = orth_subspace[id_info["dir_idx"]]
            ops = {}
            for op_name, op_func in [("ablate", ablate_direction), ("inject", inject_direction),
                                      ("double", double_direction), ("reverse", reverse_direction)]:
                if op_name == "inject":
                    h_mod = op_func(h_target, d, scale=1.0)
                else:
                    h_mod = op_func(h_target, d)
                target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
                ops[op_name] = float(target_D_after - baseline_target_D)
            dir_tests.append({
                "dir_idx": id_info["dir_idx"],
                "original_ablate_delta": id_info["ablate_delta"],
                "operations": ops,
                "is_inhibit": True,
            })

        results[cat_name] = {
            "layer": li,
            "n_support": len(support_dirs),
            "n_inhibit": len(inhibit_dirs),
            "dir_tests": dir_tests,
            "baseline": float(baseline_target_D),
        }

    return results


# ==================== Exp4: 关系槽位调制 ====================
def exp4_relation_modulation(model, tokenizer, device, model_name, W_U, round_num):
    """测试不同关系模板下L(n-2)/L(n-1)的shared_semantic效应"""
    plog(f"=== Exp4: Relation Template Modulation ===")
    n_layers = get_model_info(model, model_name).n_layers

    if model_name == "deepseek7b":
        test_cats = ["fruit", "food"]
    else:
        test_cats = ["fruit", "clothing"]

    test_relations = list(RELATION_TEMPLATES.keys())
    test_layers = [n_layers - 2, n_layers - 1]

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        cat_results = {}

        for relation, template in RELATION_TEMPLATES.items():
            plog(f"    Relation: {relation}")
            rel_results = {}

            for li in test_layers:
                test_cats_dict = {cn: CATEGORIES[cn] for cn in [cat_name] + [c for c in CAT_NAMES if c != cat_name][:7]}

                # 使用不同模板获取中心
                cat_centers, cat_hidden = get_category_centers(
                    model, tokenizer, device, test_cats_dict, li, W_U, template=template
                )

                h_target = cat_centers[cat_name]
                other_cats = [c for c in test_cats_dict if c != cat_name]
                h_others = np.array([cat_centers[c] for c in other_cats])

                others_mean = np.mean(h_others, axis=0)
                B_c = h_target - others_mean
                Bc_norm = B_c / (np.linalg.norm(B_c) + 1e-10)

                # shared_semantic方向
                all_mean = np.mean(np.vstack([h_target[np.newaxis], h_others]), axis=0)
                shared_dir = all_mean / (np.linalg.norm(all_mean) + 1e-10)
                shared_dir = shared_dir - np.dot(shared_dir, Bc_norm) * Bc_norm
                if np.linalg.norm(shared_dir) > 1e-6:
                    shared_dir = shared_dir / np.linalg.norm(shared_dir)

                baseline_target_D, _ = compute_dcf_from_center(h_target, W_U, cat_name, CAT_NAMES)

                # ablate_shared
                h_mod = ablate_direction(h_target, shared_dir)
                target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
                ablate_delta = float(target_D_after - baseline_target_D)

                # inject_shared
                h_mod = inject_direction(h_target, shared_dir, scale=1.0)
                target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
                inject_delta = float(target_D_after - baseline_target_D)

                # ablate_orth_bc
                proj_bc = np.dot(h_target, Bc_norm) * Bc_norm
                target_D_after, _ = compute_dcf_from_center(proj_bc, W_U, cat_name, CAT_NAMES)
                orth_delta = float(target_D_after - baseline_target_D)

                rel_results[f"L{li}"] = {
                    "ablate_shared": ablate_delta,
                    "inject_shared": inject_delta,
                    "ablate_orth_bc": orth_delta,
                    "baseline": float(baseline_target_D),
                }

                plog(f"      L{li} ({relation}): ablate_shared={ablate_delta:+.3f}, "
                     f"inject_shared={inject_delta:+.3f}, ablate_orth={orth_delta:+.3f}")

            cat_results[relation] = rel_results

        results[cat_name] = cat_results

    return results


# ==================== 主函数 ====================
def run_phase491(model_name, round_num):
    """运行Phase 491完整实验"""
    plog(f"========== Phase 491: {model_name} Round {round_num} ==========")

    # 加载模型
    model, tokenizer, device = get_model_and_tokenizer(model_name)

    # 获取W_U
    W_U_raw = get_W_U(model, model_name)
    W_U = W_U_raw.numpy() if hasattr(W_U_raw, 'numpy') else W_U_raw

    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    all_results = {
        "phase": 491,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: R2确认
    try:
        plog("Starting Exp1: R2 Confirmation...")
        exp1_results = exp1_r2_confirmation(model, tokenizer, device, model_name, W_U, round_num)
        all_results["exp1_r2_confirmation"] = exp1_results
        plog("Exp1 complete.")
    except Exception as e:
        plog(f"Exp1 ERROR: {e}")
        import traceback; traceback.print_exc()

    # Exp2: 刹车闭环
    try:
        plog("Starting Exp2: Brake Closure...")
        exp2_results = exp2_brake_closure(model, tokenizer, device, model_name, W_U, round_num)
        all_results["exp2_brake_closure"] = exp2_results
        plog("Exp2 complete.")
    except Exception as e:
        plog(f"Exp2 ERROR: {e}")
        import traceback; traceback.print_exc()

    # Exp3: 读出支撑闭环
    try:
        plog("Starting Exp3: Readout Closure...")
        exp3_results = exp3_readout_closure(model, tokenizer, device, model_name, W_U, round_num)
        all_results["exp3_readout_closure"] = exp3_results
        plog("Exp3 complete.")
    except Exception as e:
        plog(f"Exp3 ERROR: {e}")
        import traceback; traceback.print_exc()

    # Exp4: 关系槽位调制
    try:
        plog("Starting Exp4: Relation Modulation...")
        exp4_results = exp4_relation_modulation(model, tokenizer, device, model_name, W_U, round_num)
        all_results["exp4_relation_modulation"] = exp4_results
        plog("Exp4 complete.")
    except Exception as e:
        plog(f"Exp4 ERROR: {e}")
        import traceback; traceback.print_exc()

    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase491_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"Results saved to {out_path}")

    # 释放模型
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Model released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    valid_models = ["qwen3", "glm4", "deepseek7b"]
    if model_name not in valid_models:
        plog(f"Invalid model: {model_name}. Must be one of {valid_models}")
        sys.exit(1)

    run_phase491(model_name, round_num)
    plog("Phase 491 complete!")
