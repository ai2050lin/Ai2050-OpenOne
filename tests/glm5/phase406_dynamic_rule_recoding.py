"""
Phase 406: Dynamic Rule Recoding
=================================

核心问题: 模型的TYPE×SPEED几何是否随上下文规则动态重编码?

如果模型只是存储了静态知识表(cheetah=fast, snail=slow), 那么无论规则如何改变,
内部TYPE×SPEED几何应该不变. 但如果模型真正理解语义关系, 那么当规则反转时,
TYPE×SPEED几何应该跟着重编码.

测试设计:
- World A (默认规则): 小动物慢, 大动物快; 大车快, 小车慢
  - snail→slow, cheetah→fast, bicycle→slow, rocket→fast

- World B (反转规则): 小动物快, 大动物慢; 小车快, 大车慢
  - snail→fast, cheetah→slow, bicycle→fast, rocket→slow

- World C (无关规则): 动物颜色规则 (与速度无关, 作为控制条件)

对每个世界, 测量:
1. 候选分布: 8个速度候选词的entropy, variance, speed_gradient, rank_corr
2. 层级方向效应: 在early/mid/deep层, 注入速度方向后的odd/even
3. TYPE聚类: 同TYPE对象在残差空间的聚类
4. 跨世界比较: World A vs World B 的分布差异是否显著

关键指标:
- Δentropy = entropy_B - entropy_A (规则反转后分布变化)
- Δspeed_gradient = gradient_B - gradient_A (速度梯度是否反转)
- Δrank_corr = corr_B - corr_A (排序相关是否反转)
- Δlogit_odd = odd_B - odd_A (方向效应是否反转)

Usage:
  python tests/glm5/phase406_dynamic_rule_recoding.py qwen3
  python tests/glm5/phase406_dynamic_rule_recoding.py deepseek7b
  python tests/glm5/phase406_dynamic_rule_recoding.py glm4
"""

import sys
import os
import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, OrderedDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model, get_W_U

# ===== 配置 =====

SPEED_CANDIDATES = OrderedDict([
    ("sluggish", 1), ("slow", 2), ("steady", 3), ("moderate", 4),
    ("quick", 5), ("fast", 6), ("rapid", 7), ("swift", 8),
])

# 6 objects for Round 1
SPEED_OBJECTS = OrderedDict([
    ("snail",     {"type": "animal",     "speed_level": 1, "default_fast": False}),
    ("cheetah",   {"type": "animal",     "speed_level": 5, "default_fast": True}),
    ("bicycle",   {"type": "vehicle",    "speed_level": 2, "default_fast": False}),
    ("rocket",    {"type": "vehicle",    "speed_level": 5, "default_fast": True}),
    ("glacier",   {"type": "phenomenon", "speed_level": 1, "default_fast": False}),
    ("lightning", {"type": "phenomenon", "speed_level": 5, "default_fast": True}),
])

# 12 objects for Round 2 (confirmation)
SPEED_OBJECTS_EXTENDED = OrderedDict([
    ("snail",     {"type": "animal",     "speed_level": 1, "default_fast": False}),
    ("turtle",    {"type": "animal",     "speed_level": 1, "default_fast": False}),
    ("cheetah",   {"type": "animal",     "speed_level": 5, "default_fast": True}),
    ("falcon",    {"type": "animal",     "speed_level": 5, "default_fast": True}),
    ("bicycle",   {"type": "vehicle",    "speed_level": 2, "default_fast": False}),
    ("cart",      {"type": "vehicle",    "speed_level": 1, "default_fast": False}),
    ("rocket",    {"type": "vehicle",    "speed_level": 5, "default_fast": True}),
    ("jet",       {"type": "vehicle",    "speed_level": 5, "default_fast": True}),
    ("glacier",   {"type": "phenomenon", "speed_level": 1, "default_fast": False}),
    ("erosion",   {"type": "phenomenon", "speed_level": 1, "default_fast": False}),
    ("lightning", {"type": "phenomenon", "speed_level": 5, "default_fast": True}),
    ("explosion", {"type": "phenomenon", "speed_level": 5, "default_fast": True}),
])

# ===== 规则世界定义 =====
# 每个世界用一组规则描述prompt + 分类映射

WORLD_CONFIGS = OrderedDict({
    "A_default": {
        "description": "默认世界 - 大小/速度正常映射",
        # 规则prompt: 告诉模型规则
        "rule_prompts": [
            "In this world, large animals are fast and small animals are slow. Large vehicles are fast and small vehicles are slow. Fast phenomena move quickly and slow phenomena move gradually.",
            "In this world, the rule is: bigger means faster for both animals and vehicles. Natural phenomena follow their normal speed.",
        ],
        # 每个对象的fast/slow标签 (在这个世界下的期望)
        "obj_speed": {
            "snail": "slow", "turtle": "slow", "cheetah": "fast", "falcon": "fast",
            "bicycle": "slow", "cart": "slow", "rocket": "fast", "jet": "fast",
            "glacier": "slow", "erosion": "slow", "lightning": "fast", "explosion": "fast",
        },
    },
    "B_reversed": {
        "description": "反转世界 - 小动物快, 大动物慢; 小车快, 大车慢",
        "rule_prompts": [
            "In this world, small animals are fast and large animals are slow. Small vehicles are fast and large vehicles are slow. Slow phenomena move quickly and fast phenomena move gradually.",
            "In this world, the rule is: smaller means faster for both animals and vehicles. The speed of natural phenomena is reversed from normal.",
        ],
        "obj_speed": {
            "snail": "fast", "turtle": "fast", "cheetah": "slow", "falcon": "slow",
            "bicycle": "fast", "cart": "fast", "rocket": "slow", "jet": "slow",
            "glacier": "fast", "erosion": "fast", "lightning": "slow", "explosion": "slow",
        },
    },
    "C_control": {
        "description": "控制世界 - 颜色规则, 与速度无关",
        "rule_prompts": [
            "In this world, animals are classified by color: bright animals are common, dark animals are rare. Vehicles are classified by price: expensive ones are luxury, cheap ones are utility. Phenomena are classified by frequency: common ones are daily, rare ones are extraordinary.",
            "In this world, the rule is: things are organized by appearance and cost rather than speed. Animals by color, vehicles by price, phenomena by frequency.",
        ],
        "obj_speed": {
            # 在控制世界, 速度标签应该回归默认
            "snail": "slow", "turtle": "slow", "cheetah": "fast", "falcon": "fast",
            "bicycle": "slow", "cart": "slow", "rocket": "fast", "jet": "fast",
            "glacier": "slow", "erosion": "slow", "lightning": "fast", "explosion": "fast",
        },
    },
})

# 采样层配置
SAMPLE_LAYERS = {
    "qwen3": [0, 4, 12, 20, 28, 35],
    "deepseek7b": [0, 4, 10, 16, 20, 27],
    "glm4": [0, 5, 15, 25, 35, 39],
}

# 方向注入层 (early + deep)
INJECT_LAYERS = {
    "qwen3": [4, 28],
    "deepseek7b": [4, 20],
    "glm4": [5, 35],
}


def log_memory():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.2f}GB alloc, {reserved:.2f}GB reserved"
    return "GPU not available"


def load_model_bf16_safe(model_name):
    """BF16 + device_map=auto 加载模型, 使用flash attention"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} (BF16+auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl
            )
            print(f"  Loaded with attn_implementation={impl}")
            break
        except Exception as e:
            print(f"  attn_implementation={impl} failed: {e}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    print(f"  Loaded. {log_memory()}")
    return model, tokenizer


def compute_distribution_metrics(logits, candidate_ids, speed_levels):
    """计算候选分布的多个指标"""
    cand_logits = np.array([logits[cid] if cid is not None else float('-inf') for cid in candidate_ids])
    valid_mask = np.array([cid is not None for cid in candidate_ids])

    if valid_mask.sum() < 2:
        return {"entropy": 0, "variance": 0, "top_gap": 0, "rank_corr": 0,
                "speed_gradient": 0, "cand_probs": []}

    max_logit = np.max(cand_logits[valid_mask])
    exp_logits = np.exp(cand_logits - max_logit)
    exp_logits[~valid_mask] = 0
    total = np.sum(exp_logits)
    probs = exp_logits / total if total > 0 else np.zeros_like(exp_logits)

    valid_probs = probs[valid_mask]
    valid_probs_pos = valid_probs[valid_probs > 0]
    entropy = -np.sum(valid_probs_pos * np.log(valid_probs_pos)) if len(valid_probs_pos) > 0 else 0

    variance = float(np.var(cand_logits[valid_mask])) if valid_mask.sum() > 1 else 0

    sorted_logits = np.sort(cand_logits[valid_mask])[::-1]
    top_gap = float(sorted_logits[0] - sorted_logits[1]) if len(sorted_logits) > 1 else 0

    valid_levels = np.array(speed_levels)[valid_mask]
    valid_cand_logits = cand_logits[valid_mask]
    if len(valid_levels) > 2:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(valid_levels, valid_cand_logits)
        rank_corr = float(corr) if not np.isnan(corr) else 0
    else:
        rank_corr = 0

    if len(valid_levels) > 1:
        slope = np.polyfit(valid_levels, valid_cand_logits, 1)[0]
        speed_gradient = float(slope)
    else:
        speed_gradient = 0

    # Candidate probabilities
    cand_probs = [float(p) for p in probs]

    return {
        "entropy": float(entropy),
        "variance": float(variance),
        "top_gap": float(top_gap),
        "rank_corr": float(rank_corr),
        "speed_gradient": float(speed_gradient),
        "cand_probs": cand_probs,
    }


def build_world_prompt(world_config, obj_name, obj_data, rule_idx=0):
    """
    构建世界规则prompt + 对象查询
    
    格式:
    [规则描述] The {obj} is ___.
    """
    rule_text = world_config["rule_prompts"][rule_idx % len(world_config["rule_prompts"])]
    query = f" The {obj_name} is"
    return rule_text + query


def build_simple_prompt(obj_name, speed_label):
    """简单的无规则prompt (baseline)"""
    return f"The {obj_name} is {speed_label}."


def run_forward_with_rule(model, tokenizer, device, prompt, W_U_np, candidate_ids, speed_levels,
                          capture_layers=None, layers_list=None):
    """
    运行前向传播, 返回候选分布指标和层级轨迹
    
    Args:
        capture_layers: 要捕获残差流输出的层索引列表
        layers_list: 层列表 (如果capture_layers不为None)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Hook for capturing layer outputs
    captured_layers = {}
    handles = []
    if capture_layers and layers_list:
        def make_hook(li):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured_layers[li] = output[0].detach().float().cpu()
                else:
                    captured_layers[li] = output.detach().float().cpu()
            return hook_fn
        for li in capture_layers:
            if li < len(layers_list):
                handles.append(layers_list[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)

    for h in handles:
        h.remove()

    # Final logits metrics
    final_logits = out.logits[0, -1].float().cpu().numpy()
    final_metrics = compute_distribution_metrics(final_logits, candidate_ids, speed_levels)

    # Layer trajectory (from hidden states)
    hs = out.hidden_states
    layer_metrics = {}
    for li in capture_layers or []:
        if li < len(hs):
            h_vec = hs[li][0, -1].float().cpu().numpy()
            layer_logits = W_U_np @ h_vec
            lm = compute_distribution_metrics(layer_logits, candidate_ids, speed_levels)
            layer_metrics[str(li)] = {
                "entropy": lm["entropy"],
                "variance": lm["variance"],
                "speed_gradient": lm["speed_gradient"],
                "rank_corr": lm["rank_corr"],
            }

    return {
        "final_metrics": final_metrics,
        "layer_metrics": layer_metrics,
    }


def compute_speed_direction_at_layer(model, tokenizer, device, layers_list, li,
                                      obj_name, speed_label):
    """
    计算速度方向: clean(corrrect) vs corrupt的残差差
    
    clean: "The {obj} is {speed_label}."
    corrupt: "The item is {speed_label}."
    """
    target = speed_label

    captured = {}
    def make_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook_fn

    handle = layers_list[li].register_forward_hook(make_hook('h'))

    # Clean
    clean_prompt = f"The {obj_name} is {target}."
    captured.clear()
    inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        model(input_ids=inputs["input_ids"].to(device),
              attention_mask=inputs["attention_mask"].to(device))
    h_clean = captured['h'][0, -1].numpy().copy()

    # Corrupt
    corrupt_prompt = f"The item is {target}."
    captured.clear()
    inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        model(input_ids=inputs["input_ids"].to(device),
              attention_mask=inputs["attention_mask"].to(device))
    h_corrupt = captured['h'][0, -1].numpy().copy()

    handle.remove()

    dh = h_clean - h_corrupt
    return dh


def inject_direction_at_layer(model, tokenizer, device, layers_list, li,
                              direction_np, prompt, candidate_ids, speed_levels, W_U_np,
                              beta=8.0):
    """
    在指定层注入方向, 返回候选分布指标
    
    Returns:
        dict with odd/even decomposition
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    results = {}

    for sign_name, sign in [("plus", 1.0), ("minus", -1.0)]:
        delta = torch.tensor(sign * direction_np, dtype=torch.bfloat16, device=device)

        def make_add_hook(delta_vec):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    hs[0, -1, :] += delta_vec
                    return (hs,) + output[1:]
                else:
                    hs = output.clone()
                    hs[0, -1, :] += delta_vec
                    return hs
            return hook_fn

        handle = layers_list[li].register_forward_hook(make_add_hook(delta))

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)

        handle.remove()

        final_logits = out.logits[0, -1].float().cpu().numpy()
        metrics = compute_distribution_metrics(final_logits, candidate_ids, speed_levels)

        results[sign_name] = {
            "logits": final_logits,
            "metrics": metrics,
        }

    # Baseline (no injection)
    with torch.no_grad():
        out_base = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
    base_logits = out_base.logits[0, -1].float().cpu().numpy()
    base_metrics = compute_distribution_metrics(base_logits, candidate_ids, speed_levels)

    # Compute odd/even
    # For each metric
    metrics_names = ["entropy", "variance", "speed_gradient", "rank_corr", "top_gap"]
    odd_even = {}
    for mn in metrics_names:
        val_plus = results["plus"]["metrics"][mn]
        val_minus = results["minus"]["metrics"][mn]
        val_base = base_metrics[mn]

        eff_plus = val_plus - val_base
        eff_minus = val_minus - val_base
        odd = (eff_plus - eff_minus) / 2
        even = (eff_plus + eff_minus) / 2
        odd_even[f"{mn}_odd"] = float(odd)
        odd_even[f"{mn}_even"] = float(even)
        odd_even[f"{mn}_base"] = float(val_base)

    # Logit odd/even for speed candidates
    cand_plus = results["plus"]["logits"]
    cand_minus = results["minus"]["logits"]
    cand_base = base_logits

    # Average logit difference across candidates
    cand_logit_diff_plus = float(np.mean([cand_plus[cid] - cand_base[cid]
                                           for cid in candidate_ids if cid is not None]))
    cand_logit_diff_minus = float(np.mean([cand_minus[cid] - cand_base[cid]
                                            for cid in candidate_ids if cid is not None]))
    logit_odd = (cand_logit_diff_plus - cand_logit_diff_minus) / 2
    logit_even = (cand_logit_diff_plus + cand_logit_diff_minus) / 2
    odd_even["logit_odd"] = float(logit_odd)
    odd_even["logit_even"] = float(logit_even)

    return odd_even


def compute_type_clustering(model, tokenizer, device, W_U_np, objects, world_config,
                            layer_indices, layers_list):
    """
    计算TYPE聚类: 同TYPE对象在残差空间的聚类程度
    
    对每个世界规则, 对每个层:
    1. 获取每个对象的残差向量
    2. 计算within-type距离 vs across-type距离
    3. 聚类比 = mean(within-type dist) / mean(across-type dist)
    """
    obj_names = list(objects.keys())
    types = set(obj_data["type"] for obj_data in objects.values())

    results = {}

    for li in layer_indices:
        if li >= len(layers_list):
            continue

        captured = {}
        def make_hook(key):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook_fn

        handle = layers_list[li].register_forward_hook(make_hook('h'))

        obj_vectors = {}
        for obj_name in obj_names:
            obj_data = objects[obj_name]
            # Use the first rule prompt
            prompt = build_world_prompt(world_config, obj_name, obj_data, rule_idx=0)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            captured.clear()
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            if 'h' in captured:
                obj_vectors[obj_name] = captured['h'][0, -1].numpy().copy()

        handle.remove()

        if len(obj_vectors) < 2:
            continue

        # Compute clustering
        within_dists = []
        across_dists = []

        for i, n1 in enumerate(obj_names):
            if n1 not in obj_vectors:
                continue
            for j, n2 in enumerate(obj_names):
                if j <= i or n2 not in obj_vectors:
                    continue
                dist = np.linalg.norm(obj_vectors[n1] - obj_vectors[n2])
                if objects[n1]["type"] == objects[n2]["type"]:
                    within_dists.append(dist)
                else:
                    across_dists.append(dist)

        if within_dists and across_dists:
            mean_within = float(np.mean(within_dists))
            mean_across = float(np.mean(across_dists))
            cluster_ratio = mean_within / mean_across if mean_across > 0 else 0
        else:
            mean_within = 0
            mean_across = 0
            cluster_ratio = 0

        results[str(li)] = {
            "mean_within_type_dist": mean_within,
            "mean_across_type_dist": mean_across,
            "cluster_ratio": cluster_ratio,
            "n_within": len(within_dists),
            "n_across": len(across_dists),
        }

    return results


def run_phase406(model_name, use_extended=False):
    """
    Phase 406主函数: 动态规则重编码测试
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    objects = SPEED_OBJECTS_EXTENDED if use_extended else SPEED_OBJECTS
    obj_names = sorted(objects.keys())

    print(f"\n{'='*80}")
    print(f"=== Phase 406: Dynamic Rule Recoding ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")
    print(f"  Objects: {len(obj_names)}, Extended: {use_extended}")
    print(f"  Worlds: {list(WORLD_CONFIGS.keys())}")

    # Load model
    model, tokenizer = load_model_bf16_safe(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device

    # Get W_U
    W_U_np = get_W_U(model, model_name)
    print(f"  W_U: shape={W_U_np.shape}, n_layers={info.n_layers}")

    # Resolve token IDs for speed candidates
    candidate_ids = []
    speed_levels = []
    cand_names = []
    for cand_name, level in SPEED_CANDIDATES.items():
        ids = tokenizer.encode(cand_name, add_special_tokens=False)
        tid = ids[0] if ids else None
        candidate_ids.append(tid)
        speed_levels.append(level)
        cand_names.append(cand_name)

    print(f"  Candidates: {dict(zip(cand_names, candidate_ids))}")

    # Layer config
    sample_layers = SAMPLE_LAYERS.get(model_name, [0, info.n_layers//2, info.n_layers-1])
    inject_layers = INJECT_LAYERS.get(model_name, [4])
    # Ensure layers within bounds
    sample_layers = [li for li in sample_layers if li < info.n_layers]
    inject_layers = [li for li in inject_layers if li < info.n_layers]

    all_results = {
        "model": model_name,
        "timestamp": timestamp,
        "extended": use_extended,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "candidate_ids": {n: int(tid) if tid is not None else None for n, tid in zip(cand_names, candidate_ids)},
        "world_results": {},
    }

    # ====== Part A: Per-world candidate distribution ======
    print(f"\n{'='*70}")
    print(f"=== Part A: Candidate Distribution per World ===")

    for world_name, world_config in WORLD_CONFIGS.items():
        print(f"\n--- World {world_name}: {world_config['description']} ---")
        world_results = {
            "per_object": {},
            "layer_trajectory": {},
        }

        for obj_name in obj_names:
            obj_data = objects[obj_name]
            obj_speed_in_world = world_config["obj_speed"].get(obj_name, "slow")

            # Test with both rule prompts
            for rule_idx in range(len(world_config["rule_prompts"])):
                prompt = build_world_prompt(world_config, obj_name, obj_data, rule_idx=rule_idx)

                result = run_forward_with_rule(
                    model, tokenizer, device, prompt, W_U_np, candidate_ids, speed_levels,
                    capture_layers=sample_layers, layers_list=layers_list
                )

                fm = result["final_metrics"]

                key = f"rule{rule_idx}"
                if obj_name not in world_results["per_object"]:
                    world_results["per_object"][obj_name] = {}
                world_results["per_object"][obj_name][key] = {
                    "entropy": fm["entropy"],
                    "variance": fm["variance"],
                    "top_gap": fm["top_gap"],
                    "rank_corr": fm["rank_corr"],
                    "speed_gradient": fm["speed_gradient"],
                    "speed_label_in_world": obj_speed_in_world,
                }

                # Store layer trajectory for first rule only
                if rule_idx == 0:
                    for li_str, lm in result["layer_metrics"].items():
                        if li_str not in world_results["layer_trajectory"]:
                            world_results["layer_trajectory"][li_str] = {}
                        if obj_name not in world_results["layer_trajectory"][li_str]:
                            world_results["layer_trajectory"][li_str][obj_name] = lm

            # Print summary for this object
            obj_data_r = world_results["per_object"].get(obj_name, {})
            if "rule0" in obj_data_r:
                r0 = obj_data_r["rule0"]
                print(f"  {obj_name} ({obj_speed_in_world}): "
                      f"entropy={r0['entropy']:.3f}, gradient={r0['speed_gradient']:.3f}, "
                      f"rank_corr={r0['rank_corr']:.3f}")

        # Aggregate by type
        type_agg = defaultdict(list)
        for obj_name, obj_data_r in world_results["per_object"].items():
            if "rule0" not in obj_data_r:
                continue
            obj_type = objects[obj_name]["type"]
            r0 = obj_data_r["rule0"]
            type_agg[obj_type].append(r0)

        world_results["type_aggregate"] = {}
        for obj_type, metrics_list in type_agg.items():
            agg = {}
            for mn in ["entropy", "variance", "speed_gradient", "rank_corr"]:
                vals = [m[mn] for m in metrics_list]
                agg[mn] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)) if len(vals) > 1 else 0,
                }
            world_results["type_aggregate"][obj_type] = agg
            print(f"\n  Type {obj_type} aggregate:")
            for mn, mv in agg.items():
                print(f"    {mn}: mean={mv['mean']:.4f}, std={mv['std']:.4f}")

        all_results["world_results"][world_name] = world_results
        print(f"  {log_memory()}")

    # ====== Part B: Cross-world comparison ======
    print(f"\n{'='*70}")
    print(f"=== Part B: Cross-World Comparison ===")

    world_names = list(WORLD_CONFIGS.keys())
    for obj_name in obj_names:
        obj_data = objects[obj_name]
        print(f"\n  {obj_name} ({obj_data['type']}, speed_level={obj_data['speed_level']}):")

        for world_name in world_names:
            wr = all_results["world_results"][world_name]["per_object"].get(obj_name, {})
            if "rule0" in wr:
                r0 = wr["rule0"]
                speed_label = WORLD_CONFIGS[world_name]["obj_speed"].get(obj_name, "?")
                print(f"    World {world_name} (expect:{speed_label}): "
                      f"grad={r0['speed_gradient']:.3f}, corr={r0['rank_corr']:.3f}, "
                      f"entropy={r0['entropy']:.3f}")

    # Compute Δmetrics between worlds
    cross_world = {}
    for obj_name in obj_names:
        obj_cross = {}
        for mn in ["entropy", "speed_gradient", "rank_corr", "variance"]:
            vals = {}
            for world_name in world_names:
                wr = all_results["world_results"][world_name]["per_object"].get(obj_name, {})
                if "rule0" in wr:
                    vals[world_name] = wr["rule0"][mn]
            if len(vals) >= 2:
                # Δ between B and A
                if "A_default" in vals and "B_reversed" in vals:
                    obj_cross[f"delta_BA_{mn}"] = float(vals["B_reversed"] - vals["A_default"])
                if "A_default" in vals and "C_control" in vals:
                    obj_cross[f"delta_CA_{mn}"] = float(vals["C_control"] - vals["A_default"])
        cross_world[obj_name] = obj_cross

    # Aggregate deltas by type
    type_delta = defaultdict(lambda: defaultdict(list))
    for obj_name, obj_cross in cross_world.items():
        obj_type = objects[obj_name]["type"]
        for key, val in obj_cross.items():
            type_delta[obj_type][key].append(val)

    print(f"\n  Cross-world delta summary (B-A):")
    for obj_type, deltas in type_delta.items():
        print(f"    {obj_type}:")
        for key, vals in deltas.items():
            if key.startswith("delta_BA"):
                print(f"      {key}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

    all_results["cross_world"] = cross_world
    all_results["cross_world_type_agg"] = {
        k: {kk: {"mean": float(np.mean(vv)), "std": float(np.std(vv))}
             for kk, vv in v.items()}
        for k, v in type_delta.items()
    }

    # ====== Part C: Direction injection under different worlds ======
    print(f"\n{'='*70}")
    print(f"=== Part C: Direction Injection under Different Worlds ===")

    # For each world, compute speed direction at inject_layers, then inject
    dir_results = {}

    for world_name, world_config in WORLD_CONFIGS.items():
        print(f"\n  World {world_name}...")
        world_dir = {}

        for li in inject_layers:
            print(f"    Layer {li}...")
            layer_dir = {}

            # Compute direction using default speed labels
            # Use 2 representative objects per type
            rep_objects = [n for n in ["cheetah", "rocket", "lightning", "snail", "bicycle", "glacier"]
                          if n in objects]
            if not rep_objects:
                rep_objects = obj_names[:min(6, len(obj_names))]

            # Compute directions for each object
            obj_directions = {}
            for obj_name in rep_objects:
                obj_data = objects[obj_name]
                default_speed = "fast" if obj_data["default_fast"] else "slow"
                dh = compute_speed_direction_at_layer(
                    model, tokenizer, device, layers_list, li, obj_name, default_speed
                )
                obj_directions[obj_name] = dh
                print(f"      {obj_name} dir: |d|={np.linalg.norm(dh):.4f}")

            # Average direction for fast and slow
            fast_dirs = [obj_directions[n] for n in rep_objects if objects[n]["default_fast"]]
            slow_dirs = [obj_directions[n] for n in rep_objects if not objects[n]["default_fast"]]

            if fast_dirs and slow_dirs:
                fast_dir = np.mean(fast_dirs, axis=0)
                slow_dir = np.mean(slow_dirs, axis=0)
                speed_direction = fast_dir - slow_dir  # "fastness" direction
                dir_norm = np.linalg.norm(speed_direction)
                if dir_norm > 0:
                    speed_direction = speed_direction / dir_norm
            else:
                speed_direction = np.zeros(info.d_model)

            # For each object, inject speed direction and measure effect
            for obj_name in rep_objects:
                prompt = build_world_prompt(world_config, obj_name, objects[obj_name], rule_idx=0)

                odd_even = inject_direction_at_layer(
                    model, tokenizer, device, layers_list, li,
                    speed_direction, prompt, candidate_ids, speed_levels, W_U_np,
                    beta=8.0
                )
                layer_dir[obj_name] = odd_even

            # Aggregate
            agg = defaultdict(list)
            for obj_name, oe in layer_dir.items():
                for key, val in oe.items():
                    agg[key].append(val)

            agg_means = {k: float(np.mean(v)) for k, v in agg.items()}
            world_dir[str(li)] = {
                "per_object": layer_dir,
                "aggregate": agg_means,
            }

            print(f"      Agg: logit_odd={agg_means.get('logit_odd', 0):+.4f}, "
                  f"logit_even={agg_means.get('logit_even', 0):+.4f}, "
                  f"sg_odd={agg_means.get('speed_gradient_odd', 0):+.4f}, "
                  f"sg_even={agg_means.get('speed_gradient_even', 0):+.4f}")

        dir_results[world_name] = world_dir
        print(f"    {log_memory()}")

    all_results["direction_injection"] = dir_results

    # ====== Part D: TYPE clustering under different worlds ======
    print(f"\n{'='*70}")
    print(f"=== Part D: TYPE Clustering under Different Worlds ===")

    cluster_layers = [sample_layers[0], sample_layers[len(sample_layers)//2], sample_layers[-1]]
    cluster_layers = [li for li in cluster_layers if li < info.n_layers]

    cluster_results = {}
    for world_name, world_config in WORLD_CONFIGS.items():
        print(f"\n  World {world_name}...")
        cr = compute_type_clustering(
            model, tokenizer, device, W_U_np, objects, world_config,
            cluster_layers, layers_list
        )
        cluster_results[world_name] = cr

        for li_str, cr_data in cr.items():
            print(f"    L{li_str}: cluster_ratio={cr_data['cluster_ratio']:.4f} "
                  f"(within={cr_data['mean_within_type_dist']:.2f}, "
                  f"across={cr_data['mean_across_type_dist']:.2f})")

    all_results["type_clustering"] = cluster_results

    # ====== Save results ======
    results_dir = ROOT / "results" / "phase406_dynamic_rule_recoding"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{model_name}_phase406.json"
    # Convert all numpy types to Python types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj

    all_results = convert(all_results)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    # ====== Print key summary ======
    print(f"\n{'='*80}")
    print(f"=== Phase 406 Summary ({model_name}) ===")
    print(f"{'='*80}")

    # 1. Cross-world speed gradient delta
    print(f"\n1. Speed Gradient by World (rule0):")
    for obj_name in obj_names:
        vals = {}
        for world_name in world_names:
            wr = all_results["world_results"][world_name]["per_object"].get(obj_name, {})
            if "rule0" in wr:
                vals[world_name] = wr["rule0"]["speed_gradient"]
        if vals:
            print(f"  {obj_name}: " + ", ".join(f"{wn}={v:.3f}" for wn, v in vals.items()))

    # 2. Direction injection odd/even across worlds
    print(f"\n2. Direction Injection Speed Gradient Odd/Even:")
    for world_name in world_names:
        for li_str in [str(li) for li in inject_layers]:
            dr = all_results["direction_injection"].get(world_name, {}).get(li_str, {})
            agg = dr.get("aggregate", {})
            print(f"  World {world_name} L{li_str}: "
                  f"sg_odd={agg.get('speed_gradient_odd', 0):+.4f}, "
                  f"sg_even={agg.get('speed_gradient_even', 0):+.4f}, "
                  f"entropy_even={agg.get('entropy_even', 0):+.4f}")

    # 3. Type clustering
    print(f"\n3. TYPE Clustering Ratio:")
    for world_name in world_names:
        cr = all_results["type_clustering"].get(world_name, {})
        mid_li = str(cluster_layers[len(cluster_layers)//2]) if cluster_layers else "?"
        if mid_li in cr:
            print(f"  World {world_name} L{mid_li}: ratio={cr[mid_li]['cluster_ratio']:.4f}")

    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    return all_results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    use_extended = "--extended" in sys.argv or "--confirm" in sys.argv

    run_phase406(model_name, use_extended=use_extended)


if __name__ == "__main__":
    main()
