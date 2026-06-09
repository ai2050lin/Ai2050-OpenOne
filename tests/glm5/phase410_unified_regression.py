"""
Phase 410: Cross-Attribute Unified Regression Model
====================================================

Phase 409发现: 规则强度可以非对称地改写静态知识几何。
- 低等级对象(ice/snow)容易被推向高等级方向(cold→hot)
- 高等级对象(desert/volcano)难以被推回低等级方向(hot→cold)

Phase 410目标: 建立统一回归模型, 量化各因子对gradient的贡献
  gradient = α·level + β·TYPE + γ·attribute + δ·rule_strength
            + η·token_frequency + κ·reversal_direction + ε

如果R²高, 说明编码机制有统一结构;
如果R²低, 说明不同属性/对象的编码机制有本质差异。

额外关注: 反转方向的非对称性
- cold→hot (up-reversal) vs hot→cold (down-reversal)

Usage:
  python tests/glm5/phase410_unified_regression.py qwen3
  python tests/glm5/phase410_unified_regression.py glm4
  python tests/glm5/phase410_unified_regression.py deepseek7b
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

# ===== 扩展属性和对象定义 (更多对象 = 更可靠的回归) =====
ATTRIBUTE_CONFIGS = OrderedDict({
    "temperature": {
        "description": "温度属性",
        "candidates": OrderedDict([
            ("freezing", 1), ("cold", 2), ("cool", 3), ("warm", 4), ("hot", 5), ("scorching", 6),
        ]),
        "objects": OrderedDict([
            # Low level (1-2): 应该被推向hot方向
            ("ice",         {"type": "substance",  "level": 1, "freq_rank": 3}),
            ("snow",        {"type": "substance",  "level": 1, "freq_rank": 2}),
            ("frost",       {"type": "substance",  "level": 1, "freq_rank": 5}),
            ("refrigerator",{"type": "object",     "level": 2, "freq_rank": 4}),
            ("freezer",     {"type": "object",     "level": 1, "freq_rank": 6}),
            # Mid level (3-4)
            ("spring",      {"type": "season",     "level": 3, "freq_rank": 2}),
            ("autumn",      {"type": "season",     "level": 3, "freq_rank": 3}),
            # High level (5-6): 应该被推向cold方向
            ("desert",      {"type": "place",      "level": 5, "freq_rank": 2}),
            ("volcano",     {"type": "place",      "level": 6, "freq_rank": 3}),
            ("oven",        {"type": "object",     "level": 5, "freq_rank": 3}),
            ("furnace",     {"type": "object",     "level": 5, "freq_rank": 5}),
            ("lava",        {"type": "substance",  "level": 6, "freq_rank": 4}),
            ("fire",        {"type": "substance",  "level": 5, "freq_rank": 1}),
        ]),
        "prompt_template": "The {obj} is",
    },
    "speed": {
        "description": "速度属性",
        "candidates": OrderedDict([
            ("sluggish", 1), ("slow", 2), ("steady", 3), ("moderate", 4),
            ("quick", 5), ("fast", 6), ("rapid", 7), ("swift", 8),
        ]),
        "objects": OrderedDict([
            # Low level (1-2)
            ("snail",      {"type": "animal",     "level": 1, "freq_rank": 3}),
            ("sloth",      {"type": "animal",     "level": 1, "freq_rank": 4}),
            ("turtle",     {"type": "animal",     "level": 2, "freq_rank": 2}),
            ("bicycle",    {"type": "vehicle",    "level": 2, "freq_rank": 2}),
            ("glacier",    {"type": "phenomenon", "level": 1, "freq_rank": 4}),
            # Mid level
            ("jogger",     {"type": "person",     "level": 4, "freq_rank": 5}),
            # High level (5-8)
            ("cheetah",    {"type": "animal",     "level": 6, "freq_rank": 2}),
            ("falcon",     {"type": "animal",     "level": 8, "freq_rank": 3}),
            ("rocket",     {"type": "vehicle",    "level": 7, "freq_rank": 2}),
            ("missile",    {"type": "vehicle",    "level": 7, "freq_rank": 4}),
            ("lightning",  {"type": "phenomenon", "level": 8, "freq_rank": 2}),
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
            # Low level (1-2)
            ("ant",        {"type": "animal",  "level": 1, "freq_rank": 2}),
            ("grain",      {"type": "object",  "level": 1, "freq_rank": 3}),
            ("mouse",      {"type": "animal",  "level": 2, "freq_rank": 2}),
            ("pebble",     {"type": "object",  "level": 2, "freq_rank": 4}),
            # Mid level
            ("cat",        {"type": "animal",  "level": 3, "freq_rank": 1}),
            ("table",      {"type": "object",  "level": 4, "freq_rank": 1}),
            # High level (5-7)
            ("elephant",   {"type": "animal",  "level": 6, "freq_rank": 2}),
            ("mountain",   {"type": "place",   "level": 7, "freq_rank": 1}),
            ("ocean",      {"type": "place",   "level": 7, "freq_rank": 1}),
            ("planet",     {"type": "celestial","level": 7, "freq_rank": 2}),
        ]),
        "prompt_template": "The {obj} is",
    },
})

# 规则强度定义 (精简版: baseline + strongest rule)
RULE_CONFIGS = OrderedDict({
    "L0_baseline": {
        "description": "无规则基线",
        "get_prompt": lambda attr: "",
    },
    "L4_qa_forced": {
        "description": "强制问答式 (最强规则)",
        "get_prompt": lambda attr: {
            "temperature": (
                "Q: Is ice hot or cold in this world? A: Ice is scorching hot.\n"
                "Q: Is snow hot or cold in this world? A: Snow is hot.\n"
                "Q: Is frost hot or cold in this world? A: Frost is hot.\n"
                "Q: Is a refrigerator hot or cold in this world? A: A refrigerator is warm.\n"
                "Q: Is a freezer hot or cold in this world? A: A freezer is hot.\n"
                "Q: Is a volcano hot or cold in this world? A: A volcano is freezing cold.\n"
                "Q: Is a desert hot or cold in this world? A: A desert is cold.\n"
                "Q: Is an oven hot or cold in this world? A: An oven is cold.\n"
                "Q: Is a furnace hot or cold in this world? A: A furnace is freezing cold.\n"
                "Q: Is lava hot or cold in this world? A: Lava is freezing cold.\n"
                "Q: Is fire hot or cold in this world? A: Fire is cold.\n"
            ),
            "speed": (
                "Q: Is a snail fast or slow in this world? A: A snail is extremely fast.\n"
                "Q: Is a sloth fast or slow in this world? A: A sloth is swift.\n"
                "Q: Is a turtle fast or slow in this world? A: A turtle is rapid.\n"
                "Q: Is a bicycle fast or slow in this world? A: A bicycle is swift.\n"
                "Q: Is a glacier fast or slow in this world? A: A glacier is swift.\n"
                "Q: Is a cheetah fast or slow in this world? A: A cheetah is sluggish.\n"
                "Q: Is a falcon fast or slow in this world? A: A falcon is sluggish.\n"
                "Q: Is a rocket fast or slow in this world? A: A rocket is very slow.\n"
                "Q: Is a missile fast or slow in this world? A: A missile is slow.\n"
                "Q: Is lightning fast or slow in this world? A: Lightning is sluggish.\n"
            ),
            "size": (
                "Q: Is an ant big or small in this world? A: An ant is massive.\n"
                "Q: Is a grain big or small in this world? A: A grain is huge.\n"
                "Q: Is a mouse big or small in this world? A: A mouse is huge.\n"
                "Q: Is a pebble big or small in this world? A: A pebble is massive.\n"
                "Q: Is an elephant big or small in this world? A: An elephant is tiny.\n"
                "Q: Is a mountain big or small in this world? A: A mountain is microscopic.\n"
                "Q: Is an ocean big or small in this world? A: An ocean is tiny.\n"
                "Q: Is a planet big or small in this world? A: A planet is microscopic.\n"
            ),
        }[attr],
    },
})

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
        return {"entropy": 0, "variance": 0, "top_gap": 0, "rank_corr": 0, "gradient": 0}

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

    return {
        "entropy": float(entropy),
        "variance": float(variance),
        "top_gap": float(top_gap),
        "rank_corr": float(rank_corr),
        "gradient": float(gradient),
    }


def run_unified_test(model, tokenizer, device, W_U_np, attr_name, attr_config,
                     candidate_ids, levels):
    """对单个属性运行baseline和最强规则测试"""
    objects = attr_config["objects"]
    obj_names = sorted(objects.keys())

    results = {}

    for rule_key, rule_config in RULE_CONFIGS.items():
        rule_prompt = rule_config["get_prompt"](attr_name)

        print(f"\n  --- {rule_key} ({rule_config['description']}) ---")

        rule_result = {"per_object": {}}

        for obj_name in obj_names:
            obj_data = objects[obj_name]

            if rule_prompt:
                full_prompt = f"{rule_prompt}\nThe {obj_name} is"
            else:
                full_prompt = f"The {obj_name} is"

            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=256)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)

            final_logits = out.logits[0, -1].float().cpu().numpy()
            metrics = compute_distribution_metrics(final_logits, candidate_ids, levels)

            rule_result["per_object"][obj_name] = {
                "gradient": metrics["gradient"],
                "entropy": metrics["entropy"],
                "rank_corr": metrics["rank_corr"],
                "level": obj_data["level"],
                "type": obj_data["type"],
                "freq_rank": obj_data["freq_rank"],
            }

            print(f"    {obj_name} (L{obj_data['level']}, {obj_data['type']}): "
                  f"grad={metrics['gradient']:+.4f}, corr={metrics['rank_corr']:+.4f}")

        results[rule_key] = rule_result
        print(f"    {log_memory()}")

    return results


def run_regression_analysis(all_data):
    """运行统一回归分析"""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder

    # 收集所有数据点
    rows = []
    for attr_name, attr_data in all_data.items():
        for rule_key, rule_data in attr_data.items():
            for obj_name, obj_metrics in rule_data["per_object"].items():
                rows.append({
                    "gradient": obj_metrics["gradient"],
                    "level": obj_metrics["level"],
                    "type": obj_metrics["type"],
                    "attribute": attr_name,
                    "rule_strength": 0 if "L0" in rule_key else 4,
                    "freq_rank": obj_metrics["freq_rank"],
                })

    if len(rows) < 10:
        print("  Not enough data for regression")
        return None

    # 构建特征矩阵
    # Ensure UTF-8 output
    import sys
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    X_list = []
    y_list = []

    le_type = LabelEncoder()
    le_attr = LabelEncoder()

    all_types = list(set(r["type"] for r in rows))
    all_attrs = list(set(r["attribute"] for r in rows))
    le_type.fit(all_types)
    le_attr.fit(all_attrs)

    for r in rows:
        # 特征: level, type_encoded, attr_encoded, rule_strength, freq_rank
        # 交互项: level × rule_strength (测试规则是否改变level效应)
        level = r["level"]
        rule = r["rule_strength"]
        type_enc = le_type.transform([r["type"]])[0]
        attr_enc = le_attr.transform([r["attribute"]])[0]
        freq = r["freq_rank"]
        interaction = level * rule

        X_list.append([level, type_enc, attr_enc, rule, freq, interaction])
        y_list.append(r["gradient"])

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)

    # 回归
    reg = LinearRegression()
    reg.fit(X, y)
    r2 = reg.score(X, y)
    y_pred = reg.predict(X)
    residuals = y - y_pred
    rmse = np.sqrt(np.mean(residuals**2))

    feature_names = ["level", "type", "attribute", "rule_strength", "freq_rank", "level×rule"]
    coef_dict = {name: float(coef) for name, coef in zip(feature_names, reg.coef_)}
    coef_dict["intercept"] = float(reg.intercept_)

    # 分组分析: 各属性各规则的R²
    group_r2 = {}
    for attr_name in all_attrs:
        for rule_key in ["L0_baseline", "L4_qa_forced"]:
            mask = np.array([r["attribute"] == attr_name and
                           ((rule_key == "L0_baseline" and r["rule_strength"] == 0) or
                            (rule_key == "L4_qa_forced" and r["rule_strength"] == 4))
                           for r in rows])
            if mask.sum() > 3:
                X_sub = X[mask]
                y_sub = y[mask]
                # 简单回归: gradient ~ level
                X_level = X_sub[:, 0:1]
                reg_sub = LinearRegression()
                reg_sub.fit(X_level, y_sub)
                r2_sub = reg_sub.score(X_level, y_sub)
                group_r2[f"{attr_name}_{rule_key}"] = {
                    "r2": float(r2_sub),
                    "slope": float(reg_sub.coef_[0]),
                    "intercept": float(reg_sub.intercept_),
                    "n": int(mask.sum()),
                }

    # 非对称性分析: up-reversal vs down-reversal
    # low-level对象在规则下gradient应该变正 (up-reversal)
    # high-level对象在规则下gradient应该变负 (down-reversal)
    asymmetry = {}
    for attr_name in all_attrs:
        attr_rows = [r for r in rows if r["attribute"] == attr_name]
        low_baseline = [r["gradient"] for r in attr_rows
                       if r["level"] <= 2 and r["rule_strength"] == 0]
        low_rule = [r["gradient"] for r in attr_rows
                   if r["level"] <= 2 and r["rule_strength"] == 4]
        high_baseline = [r["gradient"] for r in attr_rows
                        if r["level"] >= 5 and r["rule_strength"] == 0]
        high_rule = [r["gradient"] for r in attr_rows
                    if r["level"] >= 5 and r["rule_strength"] == 4]

        up_shift = (np.mean(low_rule) - np.mean(low_baseline)) if (low_rule and low_baseline) else 0
        down_shift = (np.mean(high_rule) - np.mean(high_baseline)) if (high_rule and high_baseline) else 0

        asymmetry[attr_name] = {
            "up_reversal_shift": float(up_shift),    # low-level gradient change
            "down_reversal_shift": float(down_shift),  # high-level gradient change
            "asymmetry_ratio": float(up_shift / abs(down_shift)) if abs(down_shift) > 0.01 else float('inf'),
            "low_baseline_mean": float(np.mean(low_baseline)) if low_baseline else 0,
            "low_rule_mean": float(np.mean(low_rule)) if low_rule else 0,
            "high_baseline_mean": float(np.mean(high_baseline)) if high_baseline else 0,
            "high_rule_mean": float(np.mean(high_rule)) if high_rule else 0,
        }

    return {
        "overall_r2": float(r2),
        "rmse": float(rmse),
        "coefficients": coef_dict,
        "n_data_points": len(rows),
        "group_r2": group_r2,
        "asymmetry": asymmetry,
    }


def run_phase410(model_name):
    """Phase 410主函数"""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 410: Cross-Attribute Unified Regression ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")

    # Load model
    model, tokenizer = load_model_bf16_safe(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device

    # Get W_U
    W_U_np = get_W_U(model, model_name)
    print(f"  W_U: shape={W_U_np.shape}, n_layers={info.n_layers}")

    all_results = {
        "model": model_name,
        "timestamp": timestamp,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "phase": 410,
        "description": "Cross-Attribute Unified Regression Model",
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
        print(f"  Objects ({len(attr_config['objects'])}): {sorted(attr_config['objects'].keys())}")

        attr_result = run_unified_test(
            model, tokenizer, device, W_U_np, attr_name, attr_config,
            candidate_ids, levels
        )

        all_results["attributes"][attr_name] = attr_result

    # ===== Regression Analysis =====
    print(f"\n{'='*80}")
    print(f"=== Unified Regression Analysis ===")
    print(f"{'='*80}")

    regression_result = run_regression_analysis(all_results["attributes"])
    all_results["regression"] = regression_result

    if regression_result:
        print(f"\n  Overall R2 = {regression_result['overall_r2']:.4f}")
        print(f"  RMSE = {regression_result['rmse']:.4f}")
        print(f"  N data points = {regression_result['n_data_points']}")
        print(f"\n  Coefficients:")
        for name, coef in regression_result["coefficients"].items():
            print(f"    {name}: {coef:+.6f}")

        print(f"\n  Group R2 (gradient ~ level):")
        for key, val in regression_result["group_r2"].items():
            print(f"    {key}: R2={val['r2']:.4f}, slope={val['slope']:+.4f}")

        print(f"\n  Asymmetry Analysis (up-reversal vs down-reversal):")
        for attr_name, asym in regression_result["asymmetry"].items():
            print(f"    {attr_name}:")
            print(f"      up_shift (low→high): {asym['up_reversal_shift']:+.4f}")
            print(f"      down_shift (high→low): {asym['down_reversal_shift']:+.4f}")
            print(f"      asymmetry_ratio: {asym['asymmetry_ratio']:.4f}")
            print(f"      low: {asym['low_baseline_mean']:+.4f} → {asym['low_rule_mean']:+.4f}")
            print(f"      high: {asym['high_baseline_mean']:+.4f} → {asym['high_rule_mean']:+.4f}")

    # ===== Save results =====
    results_dir = ROOT / "results" / "phase410_unified_regression"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{model_name}_phase410.json"

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

    run_phase410(model_name)


if __name__ == "__main__":
    main()
