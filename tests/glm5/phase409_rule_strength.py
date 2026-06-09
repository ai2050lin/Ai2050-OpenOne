"""
Phase 409: Rule Strength Gradient Test
=======================================

Phase 406发现: 自然语言反转规则没有改写速度几何, 只增加不确定性。
关键问题: 这是因为规则太弱, 还是静态知识几何确实难以被规则覆盖?

本实验设计4档规则强度, 测试规则强度与几何重编码的梯度关系:
- Level 1: 温和描述式规则 (普通陈述)
- Level 2: 定义式规则 (明确绑定对象-属性)
- Level 3: 多例示范 (in-context learning, 4个示例)
- Level 4: 强制问答式 (Q&A训练式, 6个问答对)

测试属性: temperature (最强信号) + speed (参照)
对象数: 每属性8个对象 (比406更多, 提高统计可靠性)

核心指标:
1. attribute_gradient 是否随规则强度反转
2. entropy 是否随规则强度持续变化
3. level-gradient correlation 是否随规则强度反转
4. 候选分布是否系统性偏移

Usage:
  python tests/glm5/phase409_rule_strength.py qwen3
  python tests/glm5/phase409_rule_strength.py glm4
  python tests/glm5/phase409_rule_strength.py deepseek7b
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

# ===== 属性和对象定义 =====
ATTRIBUTE_CONFIGS = OrderedDict({
    "temperature": {
        "description": "温度属性 (Phase 408最强信号)",
        "candidates": OrderedDict([
            ("freezing", 1), ("cold", 2), ("cool", 3), ("warm", 4), ("hot", 5), ("scorching", 6),
        ]),
        "objects": OrderedDict([
            ("ice",         {"type": "substance", "level": 1, "default_label": "cold"}),
            ("snow",        {"type": "substance", "level": 1, "default_label": "cold"}),
            ("refrigerator",{"type": "object",    "level": 2, "default_label": "cold"}),
            ("oven",        {"type": "object",    "level": 5, "default_label": "hot"}),
            ("desert",      {"type": "place",     "level": 5, "default_label": "hot"}),
            ("volcano",     {"type": "place",     "level": 6, "default_label": "scorching"}),
            ("lava",        {"type": "substance", "level": 6, "default_label": "scorching"}),
            ("furnace",     {"type": "object",    "level": 5, "default_label": "hot"}),
        ]),
        "prompt_template": "The {obj} is",
        # 反转规则: 高温对象变低温, 低温对象变高温
        "reverse_high": ["ice", "snow", "refrigerator"],
        "reverse_low":  ["oven", "desert", "volcano", "lava", "furnace"],
    },
    "speed": {
        "description": "速度属性 (Phase 406参照)",
        "candidates": OrderedDict([
            ("sluggish", 1), ("slow", 2), ("steady", 3), ("moderate", 4),
            ("quick", 5), ("fast", 6), ("rapid", 7), ("swift", 8),
        ]),
        "objects": OrderedDict([
            ("snail",      {"type": "animal",     "level": 1, "default_label": "slow"}),
            ("turtle",     {"type": "animal",     "level": 2, "default_label": "slow"}),
            ("sloth",      {"type": "animal",     "level": 1, "default_label": "sluggish"}),
            ("cheetah",    {"type": "animal",     "level": 6, "default_label": "fast"}),
            ("falcon",     {"type": "animal",     "level": 8, "default_label": "swift"}),
            ("bicycle",    {"type": "vehicle",    "level": 2, "default_label": "slow"}),
            ("rocket",     {"type": "vehicle",    "level": 7, "default_label": "rapid"}),
            ("glacier",    {"type": "phenomenon", "level": 1, "default_label": "sluggish"}),
        ]),
        "prompt_template": "The {obj} is",
        "reverse_high": ["snail", "turtle", "sloth", "bicycle", "glacier"],
        "reverse_low":  ["cheetah", "falcon", "rocket"],
    },
})

# ===== 规则强度定义 =====
# 每个属性有4档规则, 从弱到强
RULE_LEVELS = OrderedDict({
    "L0_baseline": {
        "description": "无规则基线",
        "get_prompt": lambda attr: "",
    },
    "L1_mild": {
        "description": "温和描述式规则",
        "get_prompt": lambda attr: {
            "temperature": "In this world, ice and snow are very hot, while volcanoes and deserts are freezing cold.",
            "speed": "In this world, snails and turtles are extremely fast, while cheetahs and rockets are very slow.",
        }[attr],
    },
    "L2_definition": {
        "description": "定义式规则 (对象-属性明确绑定)",
        "get_prompt": lambda attr: {
            "temperature": (
                "By definition in this world: ice is scorching, snow is hot, "
                "volcanoes are freezing, and deserts are cold."
            ),
            "speed": (
                "By definition in this world: snails are swift, turtles are rapid, "
                "cheetahs are sluggish, and rockets are slow."
            ),
        }[attr],
    },
    "L3_examples": {
        "description": "多例示范 (in-context learning, 4个示例)",
        "get_prompt": lambda attr: {
            "temperature": (
                "In this world, the temperature rules are reversed:\n"
                "ice → scorching\n"
                "snow → hot\n"
                "volcano → freezing\n"
                "desert → cold\n"
            ),
            "speed": (
                "In this world, the speed rules are reversed:\n"
                "snail → swift\n"
                "turtle → rapid\n"
                "cheetah → sluggish\n"
                "rocket → slow\n"
            ),
        }[attr],
    },
    "L4_qa_forced": {
        "description": "强制问答式 (Q&A训练, 6个问答对)",
        "get_prompt": lambda attr: {
            "temperature": (
                "Q: Is ice hot or cold in this world? A: Ice is scorching hot.\n"
                "Q: Is snow hot or cold in this world? A: Snow is hot.\n"
                "Q: Is a volcano hot or cold in this world? A: A volcano is freezing cold.\n"
                "Q: Is a desert hot or cold in this world? A: A desert is cold.\n"
                "Q: Is lava hot or cold in this world? A: Lava is freezing cold.\n"
                "Q: Is an oven hot or cold in this world? A: An oven is cold.\n"
            ),
            "speed": (
                "Q: Is a snail fast or slow in this world? A: A snail is extremely fast.\n"
                "Q: Is a turtle fast or slow in this world? A: A turtle is rapid.\n"
                "Q: Is a cheetah fast or slow in this world? A: A cheetah is sluggish.\n"
                "Q: Is a rocket fast or slow in this world? A: A rocket is very slow.\n"
                "Q: Is a falcon fast or slow in this world? A: A falcon is sluggish.\n"
                "Q: Is a glacier fast or slow in this world? A: A glacier is swift.\n"
            ),
        }[attr],
    },
})

# 采样层配置
SAMPLE_LAYERS = {
    "qwen3": [0, 8, 16, 24, 32, 35],
    "deepseek7b": [0, 7, 14, 20, 24, 27],
    "glm4": [0, 8, 16, 24, 32, 39],
}


def log_memory():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.2f}GB alloc, {reserved:.2f}GB reserved"
    return "GPU not available"


def load_model_bf16_safe(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} (BF16+auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    # 尝试flash_attention_2优先(省内存), 再sdpa, 最后eager
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


def compute_distribution_metrics(logits, candidate_ids, levels):
    """计算候选分布的多个指标"""
    cand_logits = np.array([logits[cid] if cid is not None else float('-inf') for cid in candidate_ids])
    valid_mask = np.array([cid is not None for cid in candidate_ids])

    if valid_mask.sum() < 2:
        return {"entropy": 0, "variance": 0, "top_gap": 0, "rank_corr": 0,
                "gradient": 0, "top_candidate": "", "top_level": 0}

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

    valid_levels = np.array(levels)[valid_mask]
    valid_cand_logits = cand_logits[valid_mask]
    if len(valid_levels) > 2:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(valid_levels, valid_cand_logits)
        rank_corr = float(corr) if not np.isnan(corr) else 0
    else:
        rank_corr = 0

    if len(valid_levels) > 1:
        slope = np.polyfit(valid_levels, valid_cand_logits, 1)[0]
        gradient = float(slope)
    else:
        gradient = 0

    # Top candidate info
    top_idx = np.argmax(cand_logits)
    top_candidate = ""
    cand_names_list = list(ATTRIBUTE_CONFIGS.get("temperature", ATTRIBUTE_CONFIGS.get("speed", {})).get("candidates", {}).keys())
    if top_idx < len(cand_names_list):
        top_candidate = cand_names_list[top_idx]
    top_level = int(valid_levels[np.argmax(valid_cand_logits)]) if len(valid_cand_logits) > 0 else 0

    return {
        "entropy": float(entropy),
        "variance": float(variance),
        "top_gap": float(top_gap),
        "rank_corr": float(rank_corr),
        "gradient": float(gradient),
        "top_level": top_level,
    }


def compute_distribution_metrics_with_names(logits, candidate_ids, levels, cand_names):
    """计算候选分布指标(带候选名)"""
    cand_logits = np.array([logits[cid] if cid is not None else float('-inf') for cid in candidate_ids])
    valid_mask = np.array([cid is not None for cid in candidate_ids])

    if valid_mask.sum() < 2:
        return {"entropy": 0, "variance": 0, "top_gap": 0, "rank_corr": 0,
                "gradient": 0, "top_candidate": "", "top_level": 0,
                "prob_distribution": {}}

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

    valid_levels = np.array(levels)[valid_mask]
    valid_cand_logits = cand_logits[valid_mask]
    if len(valid_levels) > 2:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(valid_levels, valid_cand_logits)
        rank_corr = float(corr) if not np.isnan(corr) else 0
    else:
        rank_corr = 0

    if len(valid_levels) > 1:
        slope = np.polyfit(valid_levels, valid_cand_logits, 1)[0]
        gradient = float(slope)
    else:
        gradient = 0

    # Top candidate
    top_valid_idx = np.argmax(valid_cand_logits)
    top_level = int(valid_levels[top_valid_idx])
    top_candidate = ""
    valid_cand_names = [cand_names[i] for i in range(len(cand_names)) if valid_mask[i]]
    if top_valid_idx < len(valid_cand_names):
        top_candidate = valid_cand_names[top_valid_idx]

    # Probability distribution
    prob_dist = {}
    for i, cn in enumerate(cand_names):
        if valid_mask[i]:
            prob_dist[cn] = float(probs[i])

    return {
        "entropy": float(entropy),
        "variance": float(variance),
        "top_gap": float(top_gap),
        "rank_corr": float(rank_corr),
        "gradient": float(gradient),
        "top_candidate": top_candidate,
        "top_level": top_level,
        "prob_distribution": prob_dist,
    }


def run_rule_strength_test(model, tokenizer, device, W_U_np, attr_name, attr_config,
                           candidate_ids, levels, cand_names, capture_layers=None):
    """
    对单个属性运行所有规则强度级别的测试

    Returns:
        dict: {rule_level: {object_name: metrics, ...}, aggregate: {...}}
    """
    objects = attr_config["objects"]
    obj_names = sorted(objects.keys())

    all_rule_results = {}

    for rule_key, rule_config in RULE_LEVELS.items():
        rule_prompt = rule_config["get_prompt"](attr_name)

        print(f"\n  --- Rule Level: {rule_key} ({rule_config['description']}) ---")
        if rule_prompt:
            print(f"      Prompt: {rule_prompt[:80]}...")

        rule_result = {"per_object": {}, "layer_trajectory": {}}

        for obj_name in obj_names:
            obj_data = objects[obj_name]

            # 构造完整prompt
            if rule_prompt:
                full_prompt = f"{rule_prompt}\nThe {obj_name} is"
            else:
                full_prompt = f"The {obj_name} is"

            # Forward pass
            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=256)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)

            # Final logits (最后一个token的预测)
            final_logits = out.logits[0, -1].float().cpu().numpy()
            final_metrics = compute_distribution_metrics_with_names(
                final_logits, candidate_ids, levels, cand_names
            )

            # Layer trajectory (用W_U投影)
            layer_metrics = {}
            hs = out.hidden_states
            for li in capture_layers or []:
                if li < len(hs):
                    h_vec = hs[li][0, -1].float().cpu().numpy()
                    layer_logits = W_U_np @ h_vec
                    lm = compute_distribution_metrics(layer_logits, candidate_ids, levels)
                    layer_metrics[str(li)] = {
                        "entropy": lm["entropy"],
                        "gradient": lm["gradient"],
                        "rank_corr": lm["rank_corr"],
                    }

            rule_result["per_object"][obj_name] = {
                "entropy": final_metrics["entropy"],
                "variance": final_metrics["variance"],
                "gradient": final_metrics["gradient"],
                "rank_corr": final_metrics["rank_corr"],
                "top_gap": final_metrics["top_gap"],
                "top_candidate": final_metrics["top_candidate"],
                "top_level": final_metrics["top_level"],
                "level": obj_data["level"],
                "type": obj_data["type"],
                "prob_distribution": final_metrics["prob_distribution"],
            }
            rule_result["layer_trajectory"][obj_name] = layer_metrics

            print(f"    {obj_name} (L{obj_data['level']}): "
                  f"grad={final_metrics['gradient']:.4f}, "
                  f"corr={final_metrics['rank_corr']:.4f}, "
                  f"entropy={final_metrics['entropy']:.4f}, "
                  f"top={final_metrics['top_candidate']}")

        # Aggregate for this rule level
        all_gradients = [rule_result["per_object"][n]["gradient"] for n in obj_names]
        all_levels = [objects[n]["level"] for n in obj_names]

        # Level-gradient correlation
        if len(all_gradients) > 2:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(all_levels, all_gradients)
            level_gradient_corr = float(corr) if not np.isnan(corr) else 0
        else:
            level_gradient_corr = 0

        # High-level vs low-level gradient
        low_obj_gradients = [rule_result["per_object"][n]["gradient"]
                            for n in obj_names if objects[n]["level"] <= 2]
        high_obj_gradients = [rule_result["per_object"][n]["gradient"]
                             for n in obj_names if objects[n]["level"] >= 5]

        # Mean entropy
        all_entropies = [rule_result["per_object"][n]["entropy"] for n in obj_names]

        # Type-level analysis
        type_agg = defaultdict(list)
        for obj_name in obj_names:
            obj_type = objects[obj_name]["type"]
            type_agg[obj_type].append(rule_result["per_object"][obj_name]["gradient"])

        rule_result["aggregate"] = {
            "level_gradient_corr": level_gradient_corr,
            "mean_gradient": float(np.mean(all_gradients)),
            "mean_entropy": float(np.mean(all_entropies)),
            "low_level_mean_gradient": float(np.mean(low_obj_gradients)) if low_obj_gradients else 0,
            "high_level_mean_gradient": float(np.mean(high_obj_gradients)) if high_obj_gradients else 0,
            "high_low_gradient_delta": (float(np.mean(high_obj_gradients)) - float(np.mean(low_obj_gradients))) if (high_obj_gradients and low_obj_gradients) else 0,
            "n_objects": len(obj_names),
            "type_mean_gradient": {k: float(np.mean(v)) for k, v in type_agg.items()},
        }

        agg = rule_result["aggregate"]
        print(f"  >>> Aggregate: corr={agg['level_gradient_corr']:.4f}, "
              f"grad_delta(H-L)={agg['high_low_gradient_delta']:+.4f}, "
              f"entropy={agg['mean_entropy']:.4f}")

        all_rule_results[rule_key] = rule_result

        # 中间日志: 内存和时间
        print(f"    {log_memory()}")

    return all_rule_results


def run_phase409(model_name):
    """Phase 409主函数"""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 409: Rule Strength Gradient Test ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")
    print(f"Core question: Can stronger rules overwrite static knowledge geometry?")

    # Load model
    model, tokenizer = load_model_bf16_safe(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device

    # Get W_U
    W_U_np = get_W_U(model, model_name)
    print(f"  W_U: shape={W_U_np.shape}, n_layers={info.n_layers}")

    # Layer config
    sample_layers = SAMPLE_LAYERS.get(model_name, [0, info.n_layers//2, info.n_layers-1])
    sample_layers = [li for li in sample_layers if li < info.n_layers]

    all_results = {
        "model": model_name,
        "timestamp": timestamp,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "phase": 409,
        "description": "Rule Strength Gradient Test",
        "attributes": {},
    }

    # ===== Test each attribute =====
    for attr_name, attr_config in ATTRIBUTE_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"=== Attribute: {attr_name} ({attr_config['description']}) ===")
        print(f"{'='*70}")

        # Resolve token IDs
        candidate_ids = []
        levels = []
        cand_names = []
        for cand_name, level in attr_config["candidates"].items():
            ids = tokenizer.encode(cand_name, add_special_tokens=False)
            tid = ids[0] if ids else None
            candidate_ids.append(tid)
            levels.append(level)
            cand_names.append(cand_name)

        print(f"  Candidates: {dict(zip(cand_names, candidate_ids))}")
        print(f"  Objects: {sorted(attr_config['objects'].keys())}")

        attr_result = run_rule_strength_test(
            model, tokenizer, device, W_U_np, attr_name, attr_config,
            candidate_ids, levels, cand_names,
            capture_layers=sample_layers
        )

        all_results["attributes"][attr_name] = attr_result
        print(f"  {log_memory()}")

    # ===== Cross-rule-level comparison =====
    print(f"\n{'='*80}")
    print(f"=== Cross-Rule-Level Comparison ({model_name}) ===")
    print(f"{'='*80}")

    for attr_name, attr_result in all_results["attributes"].items():
        print(f"\n--- {attr_name} ---")
        print(f"  {'Rule Level':<20} {'Grad Corr':>10} {'H-L Delta':>10} "
              f"{'Mean Entropy':>12} {'Mean Grad':>10}")
        print(f"  {'-'*65}")
        for rule_key in RULE_LEVELS.keys():
            if rule_key in attr_result:
                agg = attr_result[rule_key]["aggregate"]
                print(f"  {rule_key:<20} {agg['level_gradient_corr']:>+10.4f} "
                      f"{agg['high_low_gradient_delta']:>+10.4f} "
                      f"{agg['mean_entropy']:>12.4f} "
                      f"{agg['mean_gradient']:>+10.4f}")

    # ===== Gradient reversal analysis =====
    print(f"\n{'='*80}")
    print(f"=== Gradient Reversal Analysis ===")
    print(f"{'='*80}")

    for attr_name, attr_result in all_results["attributes"].items():
        print(f"\n--- {attr_name} ---")
        baseline_corr = attr_result.get("L0_baseline", {}).get("aggregate", {}).get("level_gradient_corr", 0)
        baseline_delta = attr_result.get("L0_baseline", {}).get("aggregate", {}).get("high_low_gradient_delta", 0)

        for rule_key in ["L1_mild", "L2_definition", "L3_examples", "L4_qa_forced"]:
            if rule_key in attr_result:
                agg = attr_result[rule_key]["aggregate"]
                corr_change = agg["level_gradient_corr"] - baseline_corr
                delta_change = agg["high_low_gradient_delta"] - baseline_delta
                entropy_change = agg["mean_entropy"] - attr_result.get("L0_baseline", {}).get("aggregate", {}).get("mean_entropy", 0)

                # 是否反转: sign变化
                reversed_corr = (baseline_corr > 0 and agg["level_gradient_corr"] < 0) or \
                               (baseline_corr < 0 and agg["level_gradient_corr"] > 0)
                reversed_delta = (baseline_delta > 0 and agg["high_low_gradient_delta"] < 0) or \
                                (baseline_delta < 0 and agg["high_low_gradient_delta"] > 0)

                print(f"  {rule_key}: corr_change={corr_change:+.4f}, "
                      f"delta_change={delta_change:+.4f}, "
                      f"entropy_change={entropy_change:+.4f}, "
                      f"REVERSED_CORR={reversed_corr}, REVERSED_DELTA={reversed_delta}")

    # ===== Per-object detailed analysis =====
    print(f"\n{'='*80}")
    print(f"=== Per-Object Gradient by Rule Level ===")
    print(f"{'='*80}")

    for attr_name, attr_result in all_results["attributes"].items():
        print(f"\n--- {attr_name} ---")
        objects = ATTRIBUTE_CONFIGS[attr_name]["objects"]
        obj_names = sorted(objects.keys())

        header = f"  {'Object':<15} {'Level':>5}"
        for rk in RULE_LEVELS.keys():
            header += f" {rk:>10}"
        print(header)
        print(f"  {'-'*80}")

        for on in obj_names:
            line = f"  {on:<15} {objects[on]['level']:>5}"
            for rk in RULE_LEVELS.keys():
                if rk in attr_result and on in attr_result[rk]["per_object"]:
                    grad = attr_result[rk]["per_object"][on]["gradient"]
                    line += f" {grad:>+10.4f}"
                else:
                    line += f" {'N/A':>10}"
            print(line)

    # ===== Save results =====
    results_dir = ROOT / "results" / "phase409_rule_strength"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{model_name}_phase409.json"

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

    run_phase409(model_name)


if __name__ == "__main__":
    main()
