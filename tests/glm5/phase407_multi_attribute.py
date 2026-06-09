"""
Phase 407: Multi-Attribute Continuous Property Encoding
=======================================================

Phase 406发现: 速度(speed)的gradient不随规则反转而动态重编码, 是静态知识编码.
Phase 407验证: 这种"静态知识编码"是否是连续属性编码的一般特征?

测试4个连续属性维度:
1. temperature (温度): cold / cool / warm / hot
2. brightness (亮度): dark / dim / bright / brilliant
3. size (大小): tiny / small / medium / large / huge
4. speed (速度): 作为参照, 已在406测试

对每个属性, 测量:
1. 候选分布: entropy, speed_gradient(或temperature_gradient等), rank_corr
2. 层级轨迹: 各层的gradient变化
3. TYPE特异性: 不同TYPE的gradient是否有差异

如果temperature/brightness/size的gradient也具有固定的方向性(不随上下文变化),
则说明"静态知识编码"是连续属性的一般机制.

Usage:
  python tests/glm5/phase407_multi_attribute.py qwen3
  python tests/glm5/phase407_multi_attribute.py glm4
  python tests/glm5/phase407_multi_attribute.py deepseek7b
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

# ===== 属性定义 =====
ATTRIBUTE_CONFIGS = OrderedDict({
    "temperature": {
        "description": "温度属性",
        "candidates": OrderedDict([
            ("freezing", 1), ("cold", 2), ("cool", 3), ("warm", 4), ("hot", 5), ("scorching", 6),
        ]),
        "objects": OrderedDict([
            ("ice",      {"type": "substance", "level": 1, "label": "freezing"}),
            ("snow",     {"type": "substance", "level": 1, "label": "cold"}),
            ("desert",   {"type": "place",     "level": 5, "label": "hot"}),
            ("volcano",  {"type": "place",     "level": 6, "label": "scorching"}),
            ("oven",     {"type": "object",    "level": 5, "label": "hot"}),
            ("refrigerator", {"type": "object", "level": 1, "label": "cold"}),
        ]),
        "prompt_template": "The {obj} is",
    },
    "brightness": {
        "description": "亮度属性",
        "candidates": OrderedDict([
            ("dark", 1), ("dim", 2), ("glowing", 3), ("bright", 4), ("brilliant", 5), ("dazzling", 6),
        ]),
        "objects": OrderedDict([
            ("cave",     {"type": "place",  "level": 1, "label": "dark"}),
            ("shadow",   {"type": "concept", "level": 1, "label": "dark"}),
            ("candle",   {"type": "object", "level": 3, "label": "glowing"}),
            ("flashlight", {"type": "object", "level": 4, "label": "bright"}),
            ("sun",      {"type": "celestial", "level": 5, "label": "brilliant"}),
            ("star",     {"type": "celestial", "level": 4, "label": "bright"}),
        ]),
        "prompt_template": "The {obj} is",
    },
    "size": {
        "description": "大小属性",
        "candidates": OrderedDict([
            ("microscopic", 1), ("tiny", 2), ("small", 3), ("medium", 4),
            ("large", 5), ("huge", 6), ("massive", 7),
        ]),
        "objects": OrderedDict([
            ("ant",      {"type": "animal", "level": 1, "label": "microscopic"}),
            ("mouse",    {"type": "animal", "level": 2, "label": "tiny"}),
            ("elephant", {"type": "animal", "level": 6, "label": "huge"}),
            ("pebble",   {"type": "object", "level": 2, "label": "tiny"}),
            ("mountain", {"type": "object", "level": 7, "label": "massive"}),
            ("ocean",    {"type": "place",  "level": 7, "label": "massive"}),
        ]),
        "prompt_template": "The {obj} is",
    },
    "speed": {
        "description": "速度属性(参照)",
        "candidates": OrderedDict([
            ("sluggish", 1), ("slow", 2), ("steady", 3), ("moderate", 4),
            ("quick", 5), ("fast", 6), ("rapid", 7), ("swift", 8),
        ]),
        "objects": OrderedDict([
            ("snail",     {"type": "animal",     "level": 1, "label": "slow"}),
            ("cheetah",   {"type": "animal",     "level": 6, "label": "fast"}),
            ("bicycle",   {"type": "vehicle",    "level": 2, "label": "slow"}),
            ("rocket",    {"type": "vehicle",    "level": 7, "label": "swift"}),
            ("glacier",   {"type": "phenomenon", "level": 1, "label": "slow"}),
            ("lightning", {"type": "phenomenon", "level": 8, "label": "swift"}),
        ]),
        "prompt_template": "The {obj} is",
    },
})

# 采样层配置
SAMPLE_LAYERS = {
    "qwen3": [0, 4, 12, 20, 28, 35],
    "deepseek7b": [0, 4, 10, 16, 20, 27],
    "glm4": [0, 5, 15, 25, 35, 39],
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
                "gradient": 0, "cand_probs": []}

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

    cand_probs = [float(p) for p in probs]

    return {
        "entropy": float(entropy),
        "variance": float(variance),
        "top_gap": float(top_gap),
        "rank_corr": float(rank_corr),
        "gradient": float(gradient),
        "cand_probs": cand_probs,
    }


def run_attribute_test(model, tokenizer, device, W_U_np, attr_config, candidate_ids, levels,
                       capture_layers=None, layers_list=None):
    """对单个属性运行完整测试"""
    objects = attr_config["objects"]
    obj_names = sorted(objects.keys())

    results = {"per_object": {}, "layer_trajectory": {}}

    for obj_name in obj_names:
        obj_data = objects[obj_name]
        prompt = attr_config["prompt_template"].format(obj=obj_name)

        # Forward pass
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)

        # Final logits
        final_logits = out.logits[0, -1].float().cpu().numpy()
        final_metrics = compute_distribution_metrics(final_logits, candidate_ids, levels)

        # Layer trajectory
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

        results["per_object"][obj_name] = {
            "entropy": final_metrics["entropy"],
            "variance": final_metrics["variance"],
            "gradient": final_metrics["gradient"],
            "rank_corr": final_metrics["rank_corr"],
            "top_gap": final_metrics["top_gap"],
            "level": obj_data["level"],
            "type": obj_data["type"],
        }
        results["layer_trajectory"][obj_name] = layer_metrics

        print(f"    {obj_name} (L{obj_data['level']}, {obj_data['type']}): "
              f"grad={final_metrics['gradient']:.3f}, corr={final_metrics['rank_corr']:.3f}, "
              f"entropy={final_metrics['entropy']:.3f}")

    # Aggregate
    all_gradients = [results["per_object"][n]["gradient"] for n in obj_names]
    all_levels = [results["per_object"][n]["level"] for n in obj_names]

    # Check if gradient direction correlates with level
    if len(all_gradients) > 2:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(all_levels, all_gradients)
        level_gradient_corr = float(corr) if not np.isnan(corr) else 0
    else:
        level_gradient_corr = 0

    # Type-level analysis
    type_agg = defaultdict(list)
    for obj_name in obj_names:
        obj_type = objects[obj_name]["type"]
        type_agg[obj_type].append(results["per_object"][obj_name]["gradient"])

    results["aggregate"] = {
        "level_gradient_corr": level_gradient_corr,
        "mean_gradient": float(np.mean(all_gradients)),
        "n_objects": len(obj_names),
        "type_mean_gradient": {k: float(np.mean(v)) for k, v in type_agg.items()},
    }

    return results


def run_phase407(model_name):
    """Phase 407主函数"""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 407: Multi-Attribute Continuous Property Encoding ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")

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
        "attributes": {},
    }

    # ===== Test each attribute =====
    for attr_name, attr_config in ATTRIBUTE_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"=== Attribute: {attr_name} ({attr_config['description']}) ===")

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

        attr_result = run_attribute_test(
            model, tokenizer, device, W_U_np, attr_config,
            candidate_ids, levels,
            capture_layers=sample_layers, layers_list=layers_list
        )

        # Print summary
        agg = attr_result["aggregate"]
        print(f"\n  Summary for {attr_name}:")
        print(f"    level_gradient_corr = {agg['level_gradient_corr']:.4f}")
        print(f"    mean_gradient = {agg['mean_gradient']:.4f}")
        for t, g in agg["type_mean_gradient"].items():
            print(f"    {t} mean_gradient = {g:.4f}")

        all_results["attributes"][attr_name] = attr_result
        print(f"  {log_memory()}")

    # ===== Cross-attribute comparison =====
    print(f"\n{'='*80}")
    print(f"=== Cross-Attribute Comparison ({model_name}) ===")
    print(f"{'='*80}")

    print(f"\n1. Level-Gradient Correlation (是否level越高gradient越正):")
    for attr_name, attr_result in all_results["attributes"].items():
        corr = attr_result["aggregate"]["level_gradient_corr"]
        print(f"  {attr_name}: corr = {corr:.4f}")

    print(f"\n2. Gradient Direction by Object Level (高level vs 低level):")
    for attr_name, attr_result in all_results["attributes"].items():
        obj_data = attr_result["per_object"]
        low_level = [v["gradient"] for v in obj_data.values() if v["level"] <= 2]
        high_level = [v["gradient"] for v in obj_data.values() if v["level"] >= 5]
        low_mean = float(np.mean(low_level)) if low_level else 0
        high_mean = float(np.mean(high_level)) if high_level else 0
        print(f"  {attr_name}: low_level(L1-2) grad={low_mean:+.4f}, "
              f"high_level(L5+) grad={high_mean:+.4f}, "
              f"delta={high_mean - low_mean:+.4f}")

    print(f"\n3. Entropy Comparison:")
    for attr_name, attr_result in all_results["attributes"].items():
        obj_data = attr_result["per_object"]
        entropies = [v["entropy"] for v in obj_data.values()]
        print(f"  {attr_name}: mean_entropy = {np.mean(entropies):.4f}, "
              f"std = {np.std(entropies):.4f}")

    # ===== Save results =====
    results_dir = ROOT / "results" / "phase407_multi_attribute"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{model_name}_phase407.json"

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

    run_phase407(model_name)


if __name__ == "__main__":
    main()
