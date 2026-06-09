"""
Phase 415: Fictional Object Rule Reversal Test
================================================

Phase 414结论: W_U方向范数不能解释非对称反转。
修正假说: 非对称反转来自对象知识锚定深度 - 对象-属性关联在训练中的强度。

本实验检验这个假说:
1. 用虚构词(glorp, zarple等)定义对象-属性关系
2. 用规则反转虚构对象的属性, 测试是否存在非对称反转
3. 与真实对象(ice/desert等)的反转非对称性对比

核心预测:
- 如果虚构对象(无先验知识, 锚定深度=0)没有非对称反转 → 非对称性来自知识锚定 ✓
- 如果虚构对象仍有非对称反转 → 非对称性来自W_U方向结构或其他因素

设计:
- 属性: temperature, speed, size (3个属性)
- 虚构对象: 每属性6个(3个定义HIGH, 3个定义LOW)
- 规则强度: L0基线, L2定义式, L4强制QA
- 对比: 同样的规则强度下, 虚构对象 vs 真实对象的反转非对称性

Usage:
  python tests/glm5/phase415_fictional_objects.py qwen3
  python tests/glm5/phase415_fictional_objects.py glm4
  python tests/glm5/phase415_fictional_objects.py deepseek7b
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

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model, get_W_U, load_model

# ===== 属性和候选词定义 =====
ATTRIBUTE_CONFIGS = OrderedDict({
    "temperature": {
        "candidates": OrderedDict([
            ("freezing", 1), ("cold", 2), ("cool", 3), ("warm", 4), ("hot", 5), ("scorching", 6),
        ]),
        # 真实对象 (有先验知识)
        "real_objects": OrderedDict([
            ("ice",    {"level": 1, "default_polarity": "low"}),
            ("snow",   {"level": 1, "default_polarity": "low"}),
            ("frost",  {"level": 1, "default_polarity": "low"}),
            ("desert", {"level": 5, "default_polarity": "high"}),
            ("volcano",{"level": 6, "default_polarity": "high"}),
            ("lava",   {"level": 6, "default_polarity": "high"}),
        ]),
        # 虚构对象 (无先验知识, 通过规则定义)
        # 格式: (虚构词, 定义极性, 定义等级)
        # LOW极性定义: glorp被定义为冷的 → 用规则反转成热的 (up-reversal)
        # HIGH极性定义: zindle被定义为热的 → 用规则反转成冷的 (down-reversal)
        "fictional_objects": OrderedDict([
            ("glorp",   {"defined_polarity": "low",  "defined_level": 2}),
            ("snarvel", {"defined_polarity": "low",  "defined_level": 1}),
            ("frelk",   {"defined_polarity": "low",  "defined_level": 2}),
            ("zindle",  {"defined_polarity": "high", "defined_level": 5}),
            ("plaxum",  {"defined_polarity": "high", "defined_level": 6}),
            ("gronick", {"defined_polarity": "high", "defined_level": 5}),
        ]),
        "prompt_template": "The {obj} is",
    },
    "speed": {
        "candidates": OrderedDict([
            ("sluggish", 1), ("slow", 2), ("steady", 3), ("moderate", 4),
            ("quick", 5), ("fast", 6), ("rapid", 7), ("swift", 8),
        ]),
        "real_objects": OrderedDict([
            ("snail",   {"level": 1, "default_polarity": "low"}),
            ("sloth",   {"level": 1, "default_polarity": "low"}),
            ("turtle",  {"level": 2, "default_polarity": "low"}),
            ("cheetah", {"level": 6, "default_polarity": "high"}),
            ("falcon",  {"level": 8, "default_polarity": "high"}),
            ("rocket",  {"level": 7, "default_polarity": "high"}),
        ]),
        "fictional_objects": OrderedDict([
            ("glorp",   {"defined_polarity": "low",  "defined_level": 2}),
            ("snarvel", {"defined_polarity": "low",  "defined_level": 1}),
            ("frelk",   {"defined_polarity": "low",  "defined_level": 2}),
            ("zindle",  {"defined_polarity": "high", "defined_level": 6}),
            ("plaxum",  {"defined_polarity": "high", "defined_level": 7}),
            ("gronick", {"defined_polarity": "high", "defined_level": 6}),
        ]),
        "prompt_template": "The {obj} is",
    },
    "size": {
        "candidates": OrderedDict([
            ("microscopic", 1), ("tiny", 2), ("small", 3), ("medium", 4),
            ("large", 5), ("huge", 6), ("massive", 7),
        ]),
        "real_objects": OrderedDict([
            ("ant",     {"level": 1, "default_polarity": "low"}),
            ("grain",   {"level": 1, "default_polarity": "low"}),
            ("pebble",  {"level": 2, "default_polarity": "low"}),
            ("mountain",{"level": 6, "default_polarity": "high"}),
            ("elephant",{"level": 5, "default_polarity": "high"}),
            ("whale",   {"level": 6, "default_polarity": "high"}),
        ]),
        "fictional_objects": OrderedDict([
            ("glorp",   {"defined_polarity": "low",  "defined_level": 2}),
            ("snarvel", {"defined_polarity": "low",  "defined_level": 1}),
            ("frelk",   {"defined_polarity": "low",  "defined_level": 2}),
            ("zindle",  {"defined_polarity": "high", "defined_level": 6}),
            ("plaxum",  {"defined_polarity": "high", "defined_level": 5}),
            ("gronick", {"defined_polarity": "high", "defined_level": 6}),
        ]),
        "prompt_template": "The {obj} is",
    },
})

# ===== 规则定义 =====
# 对虚构对象: 先定义属性, 再反转
# 对真实对象: 直接反转

def get_fictional_definition_prompt(attr_name, obj_name, obj_config):
    """生成虚构对象的属性定义prompt"""
    if attr_name == "temperature":
        level_to_word = {1: "freezing", 2: "cold", 3: "cool", 4: "warm", 5: "hot", 6: "scorching"}
        attr_word = "temperature"
    elif attr_name == "speed":
        level_to_word = {1: "sluggish", 2: "slow", 3: "steady", 4: "moderate", 5: "quick", 6: "fast", 7: "rapid", 8: "swift"}
        attr_word = "speed"
    else:  # size
        level_to_word = {1: "microscopic", 2: "tiny", 3: "small", 4: "medium", 5: "large", 6: "huge", 7: "massive"}
        attr_word = "size"

    level = obj_config["defined_level"]
    word = level_to_word.get(level, "moderate")
    return f"A {obj_name} is a thing whose {attr_word} is {word}."


def get_rule_prompts(attr_name, obj_type="fictional"):
    """生成反转规则prompt

    对虚构对象: 先定义再反转 (定义 → 反转)
    对真实对象: 直接反转
    """
    rules = OrderedDict()

    if attr_name == "temperature":
        level_to_word = {1: "freezing", 2: "cold", 3: "cool", 4: "warm", 5: "hot", 6: "scorching"}
        low_word, high_word = "cold", "hot"
        low_rev, high_rev = "scorching", "freezing"
    elif attr_name == "speed":
        level_to_word = {1: "sluggish", 2: "slow", 3: "steady", 4: "moderate", 5: "quick", 6: "fast", 7: "rapid", 8: "swift"}
        low_word, high_word = "slow", "fast"
        low_rev, high_rev = "swift", "sluggish"
    else:  # size
        level_to_word = {1: "microscopic", 2: "tiny", 3: "small", 4: "medium", 5: "large", 6: "huge", 7: "massive"}
        low_word, high_word = "small", "huge"
        low_rev, high_rev = "massive", "microscopic"

    if obj_type == "fictional":
        # 虚构对象: 定义+反转
        low_objs = ["glorp", "snarvel", "frelk"]
        high_objs = ["zindle", "plaxum", "gronick"]

        # L1: 温和反转
        rules["L1_mild"] = (
            f"In this world, {low_objs[0]} and {low_objs[1]} are very {high_word}, "
            f"while {high_objs[0]} and {high_objs[1]} are very {low_word}."
        )
        # L2: 定义式反转
        rules["L2_definition"] = (
            f"By definition in this world: "
            f"{low_objs[0]} is {high_rev}, {low_objs[1]} is {high_word}, "
            f"{high_objs[0]} is {low_rev}, and {high_objs[1]} is {low_word}."
        )
        # L4: QA强制反转
        qa_pairs = []
        for obj in low_objs[:3]:
            qa_pairs.append(f"Q: Is {obj} {high_word} or {low_word} in this world? A: {obj} is {high_rev if obj == low_objs[0] else high_word}.")
        for obj in high_objs[:3]:
            qa_pairs.append(f"Q: Is {obj} {high_word} or {low_word} in this world? A: {obj} is {low_rev if obj == high_objs[0] else low_word}.")
        rules["L4_qa"] = "\n".join(qa_pairs)
    else:
        # 真实对象: 直接反转
        if attr_name == "temperature":
            low_objs = ["ice", "snow", "frost"]
            high_objs = ["desert", "volcano", "lava"]
        elif attr_name == "speed":
            low_objs = ["snail", "sloth", "turtle"]
            high_objs = ["cheetah", "falcon", "rocket"]
        else:
            low_objs = ["ant", "grain", "pebble"]
            high_objs = ["mountain", "elephant", "whale"]

        rules["L1_mild"] = (
            f"In this world, {low_objs[0]} and {low_objs[1]} are very {high_word}, "
            f"while {high_objs[0]} and {high_objs[1]} are very {low_word}."
        )
        rules["L2_definition"] = (
            f"By definition in this world: "
            f"{low_objs[0]} is {high_rev}, {low_objs[1]} is {high_word}, "
            f"{high_objs[0]} is {low_rev}, and {high_objs[1]} is {low_word}."
        )
        qa_pairs = []
        for obj in low_objs[:3]:
            qa_pairs.append(f"Q: Is {obj} {high_word} or {low_word} in this world? A: {obj} is {high_rev if obj == low_objs[0] else high_word}.")
        for obj in high_objs[:3]:
            qa_pairs.append(f"Q: Is {obj} {high_word} or {low_word} in this world? A: {obj} is {low_rev if obj == high_objs[0] else low_word}.")
        rules["L4_qa"] = "\n".join(qa_pairs)

    return rules


def compute_level_from_probs(probs, candidates_levels):
    """从概率分布计算期望等级"""
    level = 0.0
    for cand_name, cand_level in candidates_levels.items():
        level += cand_level * probs.get(cand_name, 0.0)
    return level


def compute_entropy(probs, candidates_list):
    """计算分布熵"""
    ent = 0.0
    for cand in candidates_list:
        p = probs.get(cand, 0.0)
        if p > 1e-10:
            ent -= p * np.log2(p)
    return ent


def get_candidate_probs(logits, tokenizer, candidates_list):
    """从logits获取候选词概率"""
    # 获取候选词token ids
    cand_ids = {}
    for cand in candidates_list:
        ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        if ids:
            cand_ids[cand] = ids[-1]  # 取最后一个token
        else:
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if ids:
                cand_ids[cand] = ids[-1]

    # softmax
    probs = torch.softmax(logits, dim=-1)
    result = {}
    for cand, tid in cand_ids.items():
        if tid < probs.shape[-1]:
            result[cand] = float(probs[tid].item())

    # 归一化(只看候选词)
    total = sum(result.values())
    if total > 0:
        for k in result:
            result[k] /= total
    return result


def log_memory():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.2f}GB alloc, {reserved:.2f}GB reserved"
    return "GPU not available"


def test_object(model, tokenizer, device, attr_name, attr_config, obj_name, obj_config,
                rule_prompt="", is_fictional=False):
    """测试单个对象在给定规则下的候选词分布"""

    # 构建输入prompt
    prompt_template = attr_config["prompt_template"]

    # 对虚构对象, 先加定义
    if is_fictional:
        definition = get_fictional_definition_prompt(attr_name, obj_name, obj_config)
        if rule_prompt:
            full_prompt = definition + "\n" + rule_prompt + "\n" + prompt_template.format(obj=obj_name)
        else:
            full_prompt = definition + "\n" + prompt_template.format(obj=obj_name)
    else:
        if rule_prompt:
            full_prompt = rule_prompt + "\n" + prompt_template.format(obj=obj_name)
        else:
            full_prompt = prompt_template.format(obj=obj_name)

    # tokenize
    input_ids = tokenizer.encode(full_prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]

    # 获取候选词概率
    candidates = attr_config["candidates"]
    cand_list = list(candidates.keys())
    probs = get_candidate_probs(next_token_logits.cpu(), tokenizer, cand_list)

    # 计算指标
    expected_level = compute_level_from_probs(probs, candidates)
    entropy = compute_entropy(probs, cand_list)

    # 低/高极性概率
    low_cands = [c for c in cand_list if candidates[c] <= 3]
    high_cands = [c for c in cand_list if candidates[c] >= 5]
    low_prob = sum(probs.get(c, 0.0) for c in low_cands)
    high_prob = sum(probs.get(c, 0.0) for c in high_cands)

    # 最大概率词
    max_cand = max(probs, key=probs.get) if probs else "N/A"
    max_prob = probs.get(max_cand, 0.0)

    return {
        "probs": probs,
        "expected_level": expected_level,
        "entropy": entropy,
        "low_polarity_prob": low_prob,
        "high_polarity_prob": high_prob,
        "max_candidate": max_cand,
        "max_prob": max_prob,
        "prompt": full_prompt,
    }


def run_phase415(model_name):
    """Phase 415主函数"""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 415: Fictional Object Rule Reversal Test ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")

    # 加载模型 (不用8bit, 避免logits NaN问题)
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
    device = next(model.parameters()).device
    print(f"  Loaded. device={device}, {log_memory()}")

    all_results = {
        "model": model_name,
        "timestamp": timestamp,
        "phase": 415,
        "attributes": {},
    }

    for attr_name, attr_config in ATTRIBUTE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"=== Attribute: {attr_name} ===")
        print(f"{'='*60}")

        attr_results = {
            "real_objects": {},
            "fictional_objects": {},
            "asymmetry_analysis": {},
        }

        candidates = attr_config["candidates"]

        # ========== 测试真实对象 ==========
        print(f"\n--- Real Objects (with prior knowledge) ---")
        real_rules = get_rule_prompts(attr_name, "real")
        rule_levels = ["L0_baseline", "L1_mild", "L2_definition", "L4_qa"]

        for obj_name, obj_config in attr_config["real_objects"].items():
            print(f"\n  Object: {obj_name} (default_level={obj_config['level']}, polarity={obj_config['default_polarity']})")
            obj_results = {}

            for rule_level in rule_levels:
                if rule_level == "L0_baseline":
                    rule_prompt = ""
                else:
                    rule_prompt = real_rules.get(rule_level, "")

                result = test_object(model, tokenizer, device, attr_name, attr_config,
                                     obj_name, obj_config, rule_prompt, is_fictional=False)
                obj_results[rule_level] = {
                    "expected_level": result["expected_level"],
                    "entropy": result["entropy"],
                    "low_polarity_prob": result["low_polarity_prob"],
                    "high_polarity_prob": result["high_polarity_prob"],
                    "max_candidate": result["max_candidate"],
                    "max_prob": result["max_prob"],
                    "probs": result["probs"],
                }
                print(f"    {rule_level}: level={result['expected_level']:.3f} "
                      f"entropy={result['entropy']:.3f} "
                      f"max={result['max_candidate']}({result['max_prob']:.3f}) "
                      f"low_prob={result['low_polarity_prob']:.3f} "
                      f"high_prob={result['high_polarity_prob']:.3f}")

            attr_results["real_objects"][obj_name] = obj_results

        # ========== 测试虚构对象 ==========
        print(f"\n--- Fictional Objects (no prior knowledge) ---")
        fictional_rules = get_rule_prompts(attr_name, "fictional")

        for obj_name, obj_config in attr_config["fictional_objects"].items():
            print(f"\n  Object: {obj_name} (defined_polarity={obj_config['defined_polarity']}, "
                  f"defined_level={obj_config['defined_level']})")
            obj_results = {}

            for rule_level in rule_levels:
                if rule_level == "L0_baseline":
                    rule_prompt = ""
                else:
                    rule_prompt = fictional_rules.get(rule_level, "")

                result = test_object(model, tokenizer, device, attr_name, attr_config,
                                     obj_name, obj_config, rule_prompt, is_fictional=True)
                obj_results[rule_level] = {
                    "expected_level": result["expected_level"],
                    "entropy": result["entropy"],
                    "low_polarity_prob": result["low_polarity_prob"],
                    "high_polarity_prob": result["high_polarity_prob"],
                    "max_candidate": result["max_candidate"],
                    "max_prob": result["max_prob"],
                    "probs": result["probs"],
                }
                print(f"    {rule_level}: level={result['expected_level']:.3f} "
                      f"entropy={result['entropy']:.3f} "
                      f"max={result['max_candidate']}({result['max_prob']:.3f}) "
                      f"low_prob={result['low_polarity_prob']:.3f} "
                      f"high_prob={result['high_polarity_prob']:.3f}")

            attr_results["fictional_objects"][obj_name] = obj_results

        # ========== 非对称性分析 ==========
        print(f"\n{'='*60}")
        print(f"=== Asymmetry Analysis: {attr_name} ===")
        print(f"{'='*60}")

        # 计算真实对象的反转非对称性
        # up-reversal: LOW polarity对象被规则推向HIGH方向
        # down-reversal: HIGH polarity对象被规则推向LOW方向
        # 非对称性 = up-reversal效果 - down-reversal效果

        asymmetry_results = {}

        for rule_level in ["L1_mild", "L2_definition", "L4_qa"]:
            # === 真实对象 ===
            real_low_deltas = []
            real_high_deltas = []

            for obj_name, obj_config in attr_config["real_objects"].items():
                obj_data = attr_results["real_objects"][obj_name]
                baseline_level = obj_data["L0_baseline"]["expected_level"]
                rule_level_val = obj_data[rule_level]["expected_level"]
                delta = rule_level_val - baseline_level

                if obj_config["default_polarity"] == "low":
                    # LOW对象: 规则应推向HIGH → 期望delta > 0 (up-reversal)
                    real_low_deltas.append(delta)
                else:
                    # HIGH对象: 规则应推向LOW → 期望delta < 0 (down-reversal)
                    real_high_deltas.append(delta)

            real_up_effect = float(np.mean(real_low_deltas)) if real_low_deltas else 0
            real_down_effect = float(np.mean([abs(d) for d in real_high_deltas])) if real_high_deltas else 0
            real_asymmetry = real_up_effect - real_down_effect  # 正 = up更容易

            # === 虚构对象 ===
            fict_low_deltas = []
            fict_high_deltas = []

            for obj_name, obj_config in attr_config["fictional_objects"].items():
                obj_data = attr_results["fictional_objects"][obj_name]
                baseline_level = obj_data["L0_baseline"]["expected_level"]
                rule_level_val = obj_data[rule_level]["expected_level"]
                delta = rule_level_val - baseline_level

                if obj_config["defined_polarity"] == "low":
                    fict_low_deltas.append(delta)
                else:
                    fict_high_deltas.append(delta)

            fict_up_effect = float(np.mean(fict_low_deltas)) if fict_low_deltas else 0
            fict_down_effect = float(np.mean([abs(d) for d in fict_high_deltas])) if fict_high_deltas else 0
            fict_asymmetry = fict_up_effect - fict_down_effect

            print(f"\n  Rule Level: {rule_level}")
            print(f"    Real objects:    up-effect={real_up_effect:+.3f}  down-effect={real_down_effect:+.3f}  "
                  f"asymmetry={real_asymmetry:+.3f}")
            print(f"    Fictional objects: up-effect={fict_up_effect:+.3f}  down-effect={fict_down_effect:+.3f}  "
                  f"asymmetry={fict_asymmetry:+.3f}")
            print(f"    Asymmetry difference (real - fictional) = {real_asymmetry - fict_asymmetry:+.3f}")

            asymmetry_results[rule_level] = {
                "real_up_effect": real_up_effect,
                "real_down_effect": real_down_effect,
                "real_asymmetry": real_asymmetry,
                "fictional_up_effect": fict_up_effect,
                "fictional_down_effect": fict_down_effect,
                "fictional_asymmetry": fict_asymmetry,
                "asymmetry_difference": real_asymmetry - fict_asymmetry,
            }

        attr_results["asymmetry_analysis"] = asymmetry_results
        all_results["attributes"][attr_name] = attr_results

    # ========== 跨属性综合分析 ==========
    print(f"\n{'='*80}")
    print(f"=== Cross-Attribute Summary ({model_name}) ===")
    print(f"{'='*80}")

    for attr_name, attr_data in all_results["attributes"].items():
        print(f"\n  {attr_name}:")
        for rule_level, asym in attr_data["asymmetry_analysis"].items():
            real_asym = asym["real_asymmetry"]
            fict_asym = asym["fictional_asymmetry"]
            diff = asym["asymmetry_difference"]
            verdict = ""
            if abs(diff) > 0.1:
                verdict = "REAL >> FICTIONAL (knowledge anchoring confirmed)"
            elif abs(diff) < 0.05:
                verdict = "SIMILAR (knowledge anchoring NOT the cause)"
            else:
                verdict = "MODERATE difference"
            print(f"    {rule_level}: real_asym={real_asym:+.3f} fict_asym={fict_asym:+.3f} "
                  f"diff={diff:+.3f} → {verdict}")

    # ========== 保存结果 ==========
    results_dir = ROOT / "results" / "phase415_fictional_objects"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{model_name}_phase415.json"

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

    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Model released. {log_memory()}")

    return all_results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase415(model_name)


if __name__ == "__main__":
    main()
