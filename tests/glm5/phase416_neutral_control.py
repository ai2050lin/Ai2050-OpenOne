"""
Phase 416: Neutral Object Control - Separating Knowledge Anchoring from Embedding Bias
======================================================================================

Phase 415发现: 虚构词(glorp/zindle)有token embedding偏见, 不是真正的零先验对象。
本实验目标: 构造3种中性程度递增的对象, 精确分离各因素贡献。

三种对象类型:
1. 真实对象(ice/desert) - 训练知识锚定 + embedding先验
2. 虚构词+定义(glorp/zindle + definition) - 上下文锚定 + embedding偏见
3. 随机token ID对象(用单token占位符) - 仅规则调制, 无知识/嵌入偏见

关键对照:
- 真实对象 vs 虚构词 → 知识锚定贡献
- 虚构词 vs 随机token → embedding偏见贡献
- 随机token有无规则 → 纯规则调制贡献

如果随机token对象的反转没有非对称性 → 知识锚定+嵌入偏见是唯一来源
如果随机token仍有非对称性 → W_U方向或其他结构因素也有贡献

属性: temperature(最强信号) + speed(参照), 共2属性
规则: L0基线, L4强制QA

Usage:
  python tests/glm5/phase416_neutral_control.py qwen3
  python tests/glm5/phase416_neutral_control.py glm4
  python tests/glm5/phase416_neutral_control.py deepseek7b
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

from model_utils import MODEL_CONFIGS, get_model_info, release_model, get_W_U


def load_model_bf16(model_name):
    """BF16 + device_map=auto 加载, 参考 model_demo_bf16.py"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[bf16] Loading {model_name} (bfloat16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试flash_attention_2优先(省内存), 再sdpa, 最后eager
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
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        print(f"[bf16] {model_name}: GPU={gpu_count} components, CPU={cpu_count} components, "
              f"class={type(model).__name__}, GPU mem={gpu_mem:.2f}GB")
    else:
        print(f"[bf16] {model_name}: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, device


# ===== 属性和候选词定义 =====
ATTRIBUTE_CONFIGS = OrderedDict({
    "temperature": {
        "candidates": OrderedDict([
            ("freezing", 1), ("cold", 2), ("cool", 3), ("warm", 4), ("hot", 5), ("scorching", 6),
        ]),
        # 真实对象
        "real_objects": OrderedDict([
            ("ice",    {"level": 1, "default_polarity": "low"}),
            ("snow",   {"level": 1, "default_polarity": "low"}),
            ("frost",  {"level": 1, "default_polarity": "low"}),
            ("desert", {"level": 5, "default_polarity": "high"}),
            ("volcano",{"level": 6, "default_polarity": "high"}),
            ("lava",   {"level": 6, "default_polarity": "high"}),
        ]),
        # 虚构对象(Phase 415发现它们有embedding偏见)
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
})


def get_candidate_ids(tokenizer, candidates_list):
    """获取候选词的token IDs"""
    cand_ids = {}
    for cand in candidates_list:
        # 尝试多种编码方式
        for prefix in [" ", ""]:
            ids = tokenizer.encode(prefix + cand, add_special_tokens=False)
            if ids:
                cand_ids[cand] = ids[-1]
                break
    return cand_ids


def get_candidate_probs_from_logits(logits_tensor, cand_ids):
    """从logits tensor获取候选词概率"""
    probs = torch.softmax(logits_tensor.float(), dim=-1)
    result = {}
    for cand, tid in cand_ids.items():
        if tid < probs.shape[-1]:
            result[cand] = float(probs[tid].item())
    # 归一化
    total = sum(result.values())
    if total > 0:
        for k in result:
            result[k] /= total
    return result


def compute_metrics(probs, candidates):
    """计算期望level, 熵, 极性概率"""
    expected_level = sum(candidates[c] * probs.get(c, 0.0) for c in candidates)
    entropy = 0.0
    for c in candidates:
        p = probs.get(c, 0.0)
        if p > 1e-10:
            entropy -= p * np.log2(p)

    low_cands = [c for c in candidates if candidates[c] <= 3]
    high_cands = [c for c in candidates if candidates[c] >= 5]
    low_prob = sum(probs.get(c, 0.0) for c in low_cands)
    high_prob = sum(probs.get(c, 0.0) for c in high_cands)

    max_cand = max(probs, key=probs.get) if probs else "N/A"
    max_prob = probs.get(max_cand, 0.0)

    return {
        "expected_level": expected_level,
        "entropy": entropy,
        "low_prob": low_prob,
        "high_prob": high_prob,
        "max_cand": max_cand,
        "max_prob": max_prob,
        "probs": {k: round(v, 6) for k, v in probs.items()},
    }


def find_single_token_placeholder(tokenizer, candidates, vocab_size):
    """找到不与任何候选词重叠的单token占位符

    策略: 选取vocab中低频、不常用的token作为随机对象名
    """
    # 获取候选词的token IDs
    cand_id_set = set()
    for cand in candidates:
        ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        cand_id_set.update(ids)
        ids = tokenizer.encode(cand, add_special_tokens=False)
        cand_id_set.update(ids)

    # 选取一些特殊token作为占位符 (避开头部的特殊token和候选词)
    placeholders = []
    # 选取词汇表中间偏后的token (通常是低频词)
    for tid in range(vocab_size // 2, vocab_size):
        if tid not in cand_id_set:
            decoded = tokenizer.decode([tid]).strip()
            # 过滤掉太长或含特殊字符的
            if decoded and len(decoded) <= 8 and decoded.isalpha():
                placeholders.append((tid, decoded))
                if len(placeholders) >= 20:
                    break

    return placeholders


def test_prompt(model, tokenizer, device, prompt, cand_ids, candidates):
    """测试单个prompt, 返回候选词分布"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        next_logits = outputs.logits[0, -1, :]

    probs = get_candidate_probs_from_logits(next_logits.cpu(), cand_ids)
    metrics = compute_metrics(probs, candidates)
    return metrics


def run_phase416(model_name):
    """Phase 416主函数"""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 416: Neutral Object Control ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)

    all_results = {
        "model": model_name,
        "timestamp": timestamp,
        "phase": 416,
        "model_info": {
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "vocab_size": info.vocab_size,
        },
        "attributes": {},
    }

    for attr_name, attr_config in ATTRIBUTE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"=== Attribute: {attr_name} ===")
        print(f"{'='*60}")

        candidates = attr_config["candidates"]
        cand_list = list(candidates.keys())
        cand_ids = get_candidate_ids(tokenizer, cand_list)

        attr_results = {"real": {}, "fictional": {}, "random_token": {}}

        # =============================================
        # 条件1: 真实对象 - 3种规则强度
        # =============================================
        print(f"\n--- Condition 1: Real Objects ---")

        for obj_name, obj_config in attr_config["real_objects"].items():
            obj_data = {}
            prompt_template = attr_config["prompt_template"]

            # L0基线
            prompt = prompt_template.format(obj=obj_name)
            m = test_prompt(model, tokenizer, device, prompt, cand_ids, candidates)
            obj_data["L0"] = m
            print(f"  {obj_name} L0: level={m['expected_level']:.3f} "
                  f"max={m['max_cand']}({m['max_prob']:.3f})")

            # L4反转规则
            if obj_config["default_polarity"] == "low":
                if attr_name == "temperature":
                    rule = (f"Q: Is {obj_name} hot or cold in this world? A: {obj_name} is scorching hot.\n"
                            f"Q: Is snow hot or cold in this world? A: Snow is hot.\n"
                            f"Q: Is a desert hot or cold in this world? A: A desert is freezing cold.\n"
                            f"Q: Is a volcano hot or cold in this world? A: A volcano is freezing cold.\n")
                else:
                    rule = (f"Q: Is {obj_name} fast or slow in this world? A: {obj_name} is extremely fast.\n"
                            f"Q: Is a snail fast or slow in this world? A: A snail is extremely fast.\n"
                            f"Q: Is a cheetah fast or slow in this world? A: A cheetah is very slow.\n"
                            f"Q: Is a rocket fast or slow in this world? A: A rocket is very slow.\n")
            else:
                if attr_name == "temperature":
                    rule = (f"Q: Is {obj_name} hot or cold in this world? A: {obj_name} is freezing cold.\n"
                            f"Q: Is a desert hot or cold in this world? A: A desert is freezing cold.\n"
                            f"Q: Is ice hot or cold in this world? A: Ice is scorching hot.\n"
                            f"Q: Is snow hot or cold in this world? A: Snow is hot.\n")
                else:
                    rule = (f"Q: Is {obj_name} fast or slow in this world? A: {obj_name} is very slow.\n"
                            f"Q: Is a cheetah fast or slow in this world? A: A cheetah is very slow.\n"
                            f"Q: Is a snail fast or slow in this world? A: A snail is extremely fast.\n"
                            f"Q: Is a turtle fast or slow in this world? A: A turtle is rapid.\n")

            prompt = rule + prompt_template.format(obj=obj_name)
            m = test_prompt(model, tokenizer, device, prompt, cand_ids, candidates)
            obj_data["L4_reverse"] = m
            print(f"  {obj_name} L4: level={m['expected_level']:.3f} "
                  f"max={m['max_cand']}({m['max_prob']:.3f})")

            # delta
            delta = obj_data["L4_reverse"]["expected_level"] - obj_data["L0"]["expected_level"]
            obj_data["delta_L4"] = delta
            print(f"  {obj_name} delta={delta:+.3f}")

            attr_results["real"][obj_name] = obj_data

        # =============================================
        # 条件2: 虚构词+定义 - 同样3种规则强度
        # =============================================
        print(f"\n--- Condition 2: Fictional Objects (with definition) ---")

        for obj_name, obj_config in attr_config["fictional_objects"].items():
            obj_data = {}
            prompt_template = attr_config["prompt_template"]

            # 定义
            if attr_name == "temperature":
                level_to_word = {1: "freezing", 2: "cold", 3: "cool", 4: "warm", 5: "hot", 6: "scorching"}
                attr_word = "temperature"
            else:
                level_to_word = {1: "sluggish", 2: "slow", 3: "steady", 4: "moderate", 5: "quick", 6: "fast", 7: "rapid", 8: "swift"}
                attr_word = "speed"
            defined_word = level_to_word.get(obj_config["defined_level"], "moderate")
            definition = f"A {obj_name} is a thing whose {attr_word} is {defined_word}."

            # L0(只有定义, 无反转规则)
            prompt = definition + "\n" + prompt_template.format(obj=obj_name)
            m = test_prompt(model, tokenizer, device, prompt, cand_ids, candidates)
            obj_data["L0"] = m
            print(f"  {obj_name} L0: level={m['expected_level']:.3f} "
                  f"max={m['max_cand']}({m['max_prob']:.3f})")

            # L4反转规则
            if obj_config["defined_polarity"] == "low":
                if attr_name == "temperature":
                    rule = (f"Q: Is {obj_name} hot or cold in this world? A: {obj_name} is scorching hot.\n"
                            f"Q: Is glorp hot or cold in this world? A: Glorp is scorching hot.\n"
                            f"Q: Is zindle hot or cold in this world? A: Zindle is freezing cold.\n"
                            f"Q: Is plaxum hot or cold in this world? A: Plaxum is freezing cold.\n")
                else:
                    rule = (f"Q: Is {obj_name} fast or slow in this world? A: {obj_name} is extremely fast.\n"
                            f"Q: Is glorp fast or slow in this world? A: Glorp is extremely fast.\n"
                            f"Q: Is zindle fast or slow in this world? A: Zindle is very slow.\n"
                            f"Q: Is plaxum fast or slow in this world? A: Plaxum is very slow.\n")
            else:
                if attr_name == "temperature":
                    rule = (f"Q: Is {obj_name} hot or cold in this world? A: {obj_name} is freezing cold.\n"
                            f"Q: Is zindle hot or cold in this world? A: Zindle is freezing cold.\n"
                            f"Q: Is glorp hot or cold in this world? A: Glorp is scorching hot.\n"
                            f"Q: Is snarvel hot or cold in this world? A: Snarvel is hot.\n")
                else:
                    rule = (f"Q: Is {obj_name} fast or slow in this world? A: {obj_name} is very slow.\n"
                            f"Q: Is zindle fast or slow in this world? A: Zindle is very slow.\n"
                            f"Q: Is glorp fast or slow in this world? A: Glorp is extremely fast.\n"
                            f"Q: Is snarvel fast or slow in this world? A: Snarvel is rapid.\n")

            prompt = definition + "\n" + rule + prompt_template.format(obj=obj_name)
            m = test_prompt(model, tokenizer, device, prompt, cand_ids, candidates)
            obj_data["L4_reverse"] = m
            print(f"  {obj_name} L4: level={m['expected_level']:.3f} "
                  f"max={m['max_cand']}({m['max_prob']:.3f})")

            delta = obj_data["L4_reverse"]["expected_level"] - obj_data["L0"]["expected_level"]
            obj_data["delta_L4"] = delta
            print(f"  {obj_name} delta={delta:+.3f}")

            attr_results["fictional"][obj_name] = obj_data

        # =============================================
        # 条件3: 随机token ID对象 - 纯规则调制
        # =============================================
        print(f"\n--- Condition 3: Random Token Objects (no knowledge, no embedding bias) ---")

        placeholders = find_single_token_placeholder(tokenizer, candidates, info.vocab_size)
        print(f"  Found {len(placeholders)} placeholder tokens")

        # 选6个占位符: 3个作为LOW, 3个作为HIGH
        n_placeholders = min(6, len(placeholders))
        selected = placeholders[:n_placeholders]

        for i, (tid, decoded) in enumerate(selected):
            obj_name = decoded
            polarity = "low" if i < n_placeholders // 2 else "high"
            defined_level = 2 if polarity == "low" else 5

            obj_data = {"token_id": tid, "decoded": decoded, "polarity": polarity}
            prompt_template = attr_config["prompt_template"]

            # 定义
            if attr_name == "temperature":
                level_to_word = {1: "freezing", 2: "cold", 3: "cool", 4: "warm", 5: "hot", 6: "scorching"}
                attr_word = "temperature"
            else:
                level_to_word = {1: "sluggish", 2: "slow", 3: "steady", 4: "moderate", 5: "quick", 6: "fast", 7: "rapid", 8: "swift"}
                attr_word = "speed"
            defined_word = level_to_word.get(defined_level, "moderate")
            definition = f"A {obj_name} is a thing whose {attr_word} is {defined_word}."

            # L0(只有定义)
            prompt = definition + "\n" + prompt_template.format(obj=obj_name)
            m = test_prompt(model, tokenizer, device, prompt, cand_ids, candidates)
            obj_data["L0"] = m
            print(f"  {obj_name}(tid={tid}) L0: level={m['expected_level']:.3f} "
                  f"max={m['max_cand']}({m['max_prob']:.3f})")

            # L4反转规则
            if polarity == "low":
                if attr_name == "temperature":
                    rule = (f"Q: Is {obj_name} hot or cold in this world? A: {obj_name} is scorching hot.\n"
                            f"Q: Is {selected[0][1]} hot or cold in this world? A: {selected[0][1]} is scorching hot.\n"
                            f"Q: Is {selected[n_placeholders//2][1]} hot or cold in this world? A: {selected[n_placeholders//2][1]} is freezing cold.\n")
                else:
                    rule = (f"Q: Is {obj_name} fast or slow in this world? A: {obj_name} is extremely fast.\n"
                            f"Q: Is {selected[0][1]} fast or slow in this world? A: {selected[0][1]} is extremely fast.\n"
                            f"Q: Is {selected[n_placeholders//2][1]} fast or slow in this world? A: {selected[n_placeholders//2][1]} is very slow.\n")
            else:
                if attr_name == "temperature":
                    rule = (f"Q: Is {obj_name} hot or cold in this world? A: {obj_name} is freezing cold.\n"
                            f"Q: Is {selected[n_placeholders//2][1]} hot or cold in this world? A: {selected[n_placeholders//2][1]} is freezing cold.\n"
                            f"Q: Is {selected[0][1]} hot or cold in this world? A: {selected[0][1]} is scorching hot.\n")
                else:
                    rule = (f"Q: Is {obj_name} fast or slow in this world? A: {obj_name} is very slow.\n"
                            f"Q: Is {selected[n_placeholders//2][1]} fast or slow in this world? A: {selected[n_placeholders//2][1]} is very slow.\n"
                            f"Q: Is {selected[0][1]} fast or slow in this world? A: {selected[0][1]} is extremely fast.\n")

            prompt = definition + "\n" + rule + prompt_template.format(obj=obj_name)
            m = test_prompt(model, tokenizer, device, prompt, cand_ids, candidates)
            obj_data["L4_reverse"] = m
            print(f"  {obj_name}(tid={tid}) L4: level={m['expected_level']:.3f} "
                  f"max={m['max_cand']}({m['max_prob']:.3f})")

            delta = obj_data["L4_reverse"]["expected_level"] - obj_data["L0"]["expected_level"]
            obj_data["delta_L4"] = delta
            print(f"  {obj_name}(tid={tid}) delta={delta:+.3f}")

            attr_results["random_token"][obj_name] = obj_data

        # =============================================
        # 非对称性分析
        # =============================================
        print(f"\n{'='*60}")
        print(f"=== Asymmetry Analysis: {attr_name} ===")
        print(f"{'='*60}")

        def compute_asymmetry(obj_dict, polarity_key="default_polarity"):
            """计算up-reversal vs down-reversal的平均delta"""
            up_deltas = []  # LOW对象被推向HIGH
            down_deltas = []  # HIGH对象被推向LOW
            for obj_name, obj_data in obj_dict.items():
                if "delta_L4" not in obj_data:
                    continue
                # 获取极性
                if polarity_key == "default_polarity":
                    pol = attr_config["real_objects"].get(obj_name, {}).get(polarity_key)
                elif polarity_key == "defined_polarity":
                    pol = attr_config["fictional_objects"].get(obj_name, {}).get(polarity_key)
                else:
                    pol = obj_data.get("polarity")

                if pol == "low":
                    up_deltas.append(obj_data["delta_L4"])
                elif pol == "high":
                    down_deltas.append(obj_data["delta_L4"])

            up_mean = float(np.mean(up_deltas)) if up_deltas else 0
            down_mean = float(np.mean([abs(d) for d in down_deltas])) if down_deltas else 0
            asymmetry = up_mean - down_mean
            return up_mean, down_mean, asymmetry

        real_up, real_down, real_asym = compute_asymmetry(attr_results["real"], "default_polarity")
        fict_up, fict_down, fict_asym = compute_asymmetry(attr_results["fictional"], "defined_polarity")
        rand_up, rand_down, rand_asym = compute_asymmetry(attr_results["random_token"], "polarity")

        print(f"\n  Real objects:      up={real_up:+.3f}  down={real_down:+.3f}  asymmetry={real_asym:+.3f}")
        print(f"  Fictional objects: up={fict_up:+.3f}  down={fict_down:+.3f}  asymmetry={fict_asym:+.3f}")
        print(f"  Random tokens:     up={rand_up:+.3f}  down={rand_down:+.3f}  asymmetry={rand_asym:+.3f}")

        print(f"\n  === Factor Decomposition ===")
        print(f"  Knowledge anchoring (real - fictional): {real_asym - fict_asym:+.3f}")
        print(f"  Embedding bias (fictional - random):    {fict_asym - rand_asym:+.3f}")
        print(f"  Base asymmetry (random token):          {rand_asym:+.3f}")

        attr_results["asymmetry"] = {
            "real": {"up": real_up, "down": real_down, "asymmetry": real_asym},
            "fictional": {"up": fict_up, "down": fict_down, "asymmetry": fict_asym},
            "random_token": {"up": rand_up, "down": rand_down, "asymmetry": rand_asym},
            "knowledge_anchoring_effect": real_asym - fict_asym,
            "embedding_bias_effect": fict_asym - rand_asym,
            "base_asymmetry": rand_asym,
        }

        all_results["attributes"][attr_name] = attr_results

    # ========== 跨属性综合 ==========
    print(f"\n{'='*80}")
    print(f"=== Cross-Attribute Summary ({model_name}) ===")
    print(f"{'='*80}")

    for attr_name, attr_data in all_results["attributes"].items():
        asym = attr_data["asymmetry"]
        print(f"\n  {attr_name}:")
        print(f"    Real:      asymmetry={asym['real']['asymmetry']:+.3f}")
        print(f"    Fictional: asymmetry={asym['fictional']['asymmetry']:+.3f}")
        print(f"    Random:    asymmetry={asym['random_token']['asymmetry']:+.3f}")
        print(f"    Knowledge anchoring: {asym['knowledge_anchoring_effect']:+.3f}")
        print(f"    Embedding bias:      {asym['embedding_bias_effect']:+.3f}")
        print(f"    Base asymmetry:      {asym['base_asymmetry']:+.3f}")

        if abs(asym['base_asymmetry']) < 0.1:
            print(f"    --> Random token has near-zero asymmetry → W_U is NOT the cause")
        else:
            print(f"    --> Random token still has asymmetry → other structural factors exist")

    # ========== 保存 ==========
    results_dir = ROOT / "results" / "phase416_neutral_control"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name}_phase416.json"

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
    gpu_after = torch.cuda.memory_allocated() / 1e9
    print(f"  Model released. GPU: {gpu_after:.2f}GB")

    return all_results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase416(model_name)


if __name__ == "__main__":
    main()
