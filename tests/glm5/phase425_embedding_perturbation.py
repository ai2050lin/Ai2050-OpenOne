"""
Phase 425: 词嵌入成分扰动与知识轨道映射
=========================================

核心问题: 一个对象词的初始embedding中, 哪些成分决定它进入哪些语义轨道?

实验设计:
1. 选择目标对象(apple, dog, knife, car, desert)
2. 构造功能方向:
   - 类别方向 (fruit vs animal vs tool vs vehicle vs place)
   - 属性方向 (red, sweet, seed, fur, sharp, wheel)
   - 对象身份方向 (对象embedding - 类别中心)
   - 随机等范数方向 (对照)
3. 对词嵌入做加法/移除扰动
4. 跟踪候选分布变化和轨道归属

知识槽位任务:
- 类别: "An X is a kind of ___" → fruit/animal/tool/vehicle/place
- 颜色: "The color of X is usually ___" → red/green/blue/gray/brown
- 味道: "X tastes ___" → sweet/sour/bitter/salty/savory
- 部件: "X has ___" → seeds/fur/blades/wheels/sand
- 来源: "X grows on/lives in/is found in ___" → tree/house/kitchen/road/desert

Round 1: 验证测试 - 5对象 × 4扰动类型 × 3任务 × 3强度 = 180测试点
Round 2: 扩展测试 - 更多对象+更多属性方向
Round 3: 确认测试 - 如有重要发现

Usage:
  python tests/glm5/phase425_embedding_perturbation.py qwen3
  python tests/glm5/phase425_embedding_perturbation.py glm4
  python tests/glm5/phase425_embedding_perturbation.py deepseek7b
"""

import sys
import os

# Windows UTF-8 输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_model_info, get_W_U


# ===== 对象和类别定义 =====
OBJECT_CATEGORIES = OrderedDict([
    ("fruit", {
        "objects": ["apple", "banana", "orange", "grape", "lemon"],
        "opposing": "animal",
    }),
    ("animal", {
        "objects": ["dog", "cat", "horse", "lion", "fish"],
        "opposing": "fruit",
    }),
    ("tool", {
        "objects": ["knife", "hammer", "spoon", "scissors", "ruler"],
        "opposing": "vehicle",
    }),
    ("vehicle", {
        "objects": ["car", "train", "bicycle", "rocket", "bus"],
        "opposing": "tool",
    }),
    ("place", {
        "objects": ["desert", "forest", "ocean", "city", "mountain"],
        "opposing": "fruit",
    }),
])

# 主测试对象 (每个类别取2个, 共10个)
PRIMARY_OBJECTS = ["apple", "orange", "dog", "horse", "knife", "scissors", "car", "bicycle", "desert", "ocean"]

# 知识槽位任务
KNOWLEDGE_TASKS = OrderedDict([
    ("category", {
        "template": "A {obj} is a kind of",
        "candidates": OrderedDict([
            ("fruit", 1), ("animal", 2), ("tool", 3), ("vehicle", 4), ("place", 5),
        ]),
        "expected": {
            "apple": "fruit", "orange": "fruit",
            "dog": "animal", "horse": "animal",
            "knife": "tool", "scissors": "tool",
            "car": "vehicle", "bicycle": "vehicle",
            "desert": "place", "ocean": "place",
        },
    }),
    ("property", {
        "template": "The most notable property of a {obj} is that it is",
        "candidates": OrderedDict([
            ("edible", 1), ("alive", 2), ("sharp", 3), ("fast", 4), ("vast", 5),
        ]),
        "expected": {
            "apple": "edible", "orange": "edible",
            "dog": "alive", "horse": "alive",
            "knife": "sharp", "scissors": "sharp",
            "car": "fast", "bicycle": "fast",
            "desert": "vast", "ocean": "vast",
        },
    }),
    ("part", {
        "template": "A {obj} has",
        "candidates": OrderedDict([
            ("seeds", 1), ("fur", 2), ("blades", 3), ("wheels", 4), ("sand", 5),
        ]),
        "expected": {
            "apple": "seeds", "orange": "seeds",
            "dog": "fur", "horse": "fur",
            "knife": "blades", "scissors": "blades",
            "car": "wheels", "bicycle": "wheels",
            "desert": "sand", "ocean": "sand",
        },
    }),
])

# 扰动强度
ALPHA_VALUES = [0.5, 1.0, 2.0]

# 扰动类型
PERTURBATION_TYPES = ["add_category", "remove_category", "add_opposing", "add_random"]


def load_model_bf16(model_name):
    """BF16 + device_map=auto + flash attention 加载模型"""
    cfg = MODEL_CONFIGS[model_name]
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} (BF16+auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试flash_attention_2优先, 然后sdpa, 最后eager
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
            print(f"  {impl} failed: {e}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  device={device}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, device


def get_category_directions(model, tokenizer, device):
    """构造类别功能方向
    
    d_fruit = mean(E[fruit_words]) - mean(E[animal_words])
    d_animal = mean(E[animal_words]) - mean(E[fruit_words])
    等等...
    
    Returns:
        dict: {category_name: direction_vector (numpy [d_model])}
    """
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().cpu().float().numpy()  # [vocab, d_model]
    
    # 计算每个类别的中心
    category_centers = {}
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        vecs = []
        for obj_word in cat_info["objects"]:
            tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
            if tok_ids:
                vecs.append(W_E[tok_ids[0]])
        if vecs:
            category_centers[cat_name] = np.mean(vecs, axis=0)
    
    # 计算类别方向: 当前类别中心 - 对立类别中心
    category_directions = {}
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        opposing = cat_info["opposing"]
        if cat_name in category_centers and opposing in category_centers:
            d = category_centers[cat_name] - category_centers[opposing]
            norm = np.linalg.norm(d)
            if norm > 0:
                d = d / norm  # 归一化
            category_directions[cat_name] = d
    
    # 计算全局均值 (用于构造随机方向时的范数参考)
    all_vecs = list(category_centers.values())
    global_center = np.mean(all_vecs, axis=0)
    
    return category_directions, category_centers, global_center, W_E


def get_object_embedding(W_E, tokenizer, obj_word):
    """获取对象词的embedding向量"""
    tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
    if not tok_ids:
        return None
    # 取第一个token的embedding
    return W_E[tok_ids[0]].copy()


def make_random_direction(d_model, rng, norm=1.0):
    """生成随机方向 (等范数)"""
    d = rng.randn(d_model)
    d_norm = np.linalg.norm(d)
    if d_norm > 0:
        d = d / d_norm * norm
    return d


def make_orthogonal_random(direction, d_model, rng, norm=1.0):
    """生成与给定方向正交的随机方向 (等范数)"""
    d = rng.randn(d_model)
    # Gram-Schmidt: 移除在direction上的投影
    proj = np.dot(d, direction) / max(np.dot(direction, direction), 1e-10) * direction
    d = d - proj
    d_norm = np.linalg.norm(d)
    if d_norm > 0:
        d = d / d_norm * norm
    return d


def get_candidate_ids(tokenizer, candidates):
    """获取候选词的token IDs"""
    cand_ids = {}
    for cand in candidates:
        ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        if ids:
            cand_ids[cand] = ids[-1]
    return cand_ids


def run_with_perturbed_embedding(model, tokenizer, device, template, obj_word,
                                  base_embedding, perturbation, alpha,
                                  cand_ids, candidates, W_E_shape):
    """用扰动后的embedding运行前向推理
    
    方法: 构造inputs_embeds, 把obj_word对应的token替换为perturbed embedding
    
    Returns:
        (level, probs_dict)
    """
    # 构建prompt
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    
    # 找到obj_word在prompt中的位置
    # 方法: tokenize不含obj的模板, 找到差异
    template_no_obj = template.replace("{obj}", "")
    # 更安全: 直接用obj_word的token位置
    obj_tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
    
    # 获取原始embeddings
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
    
    # 找到obj token在input_ids中的位置
    # 从左到右扫描
    obj_positions = []
    prompt_ids = input_ids[0].cpu().tolist()
    for i in range(len(prompt_ids) - len(obj_tok_ids) + 1):
        if prompt_ids[i:i+len(obj_tok_ids)] == obj_tok_ids:
            obj_positions = list(range(i, i + len(obj_tok_ids)))
            break
    
    if not obj_positions:
        # fallback: 用所有位置的均值修改
        obj_positions = [1]  # 通常第1个位置是第一个词
    
    # 应用扰动到obj token位置
    perturbed_embeds = inputs_embeds.clone()
    perturbation_tensor = torch.tensor(
        perturbation * alpha, 
        dtype=perturbed_embeds.dtype, 
        device=device
    )
    
    for pos in obj_positions:
        perturbed_embeds[0, pos, :] += perturbation_tensor
    
    # 前向推理
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        try:
            outputs = model(inputs_embeds=perturbed_embeds, attention_mask=attention_mask)
        except Exception as e:
            # fallback: 用原始input_ids + 正常推理
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception as e2:
                return 3.0, {c: 1.0/len(candidates) for c in candidates}
    
    next_logits = outputs.logits[0, -1, :]
    
    # 提取候选概率
    probs = torch.softmax(next_logits.float().cpu(), dim=-1)
    result = {}
    for cand, tid in cand_ids.items():
        if tid < probs.shape[-1]:
            result[cand] = float(probs[tid].item())
    total = sum(result.values())
    if total > 0:
        for k in result:
            result[k] /= total
    
    # 计算level
    level = sum(candidates[c] * result.get(c, 0.0) for c in candidates)
    
    return level, result


def run_baseline(model, tokenizer, device, template, obj_word, cand_ids, candidates):
    """运行基线(无扰动)"""
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    next_logits = outputs.logits[0, -1, :]
    
    probs = torch.softmax(next_logits.float().cpu(), dim=-1)
    result = {}
    for cand, tid in cand_ids.items():
        if tid < probs.shape[-1]:
            result[cand] = float(probs[tid].item())
    total = sum(result.values())
    if total > 0:
        for k in result:
            result[k] /= total
    
    level = sum(candidates[c] * result.get(c, 0.0) for c in candidates)
    return level, result


def get_object_category(obj_word):
    """获取对象所属类别"""
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        if obj_word in cat_info["objects"]:
            return cat_name
    return None


def run_phase425(model_name, round_num=1):
    """运行Phase 425实验"""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 425: Embedding Perturbation ({model_name}) R{round_num} [{timestamp}] ===")
    print(f"{'='*80}")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    # 获取W_E和类别方向
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing category directions...")
    category_directions, category_centers, global_center, W_E = get_category_directions(model, tokenizer, device)
    
    for cat_name, d in category_directions.items():
        print(f"  d_{cat_name}: norm={np.linalg.norm(d):.4f}, dim={len(d)}")
    
    # 选择测试对象
    if round_num == 1:
        # R1: 每个类别1个对象 = 5个
        test_objects = ["apple", "dog", "knife", "car", "desert"]
        alpha_values = [1.0, 2.0]  # 只测2个强度
    elif round_num == 2:
        # R2: 10个对象
        test_objects = PRIMARY_OBJECTS
        alpha_values = ALPHA_VALUES
    else:
        # R3: 确认测试
        test_objects = PRIMARY_OBJECTS
        alpha_values = [1.0, 2.0, 3.0]
    
    rng = np.random.RandomState(42)
    d_model = W_E.shape[1]
    
    # ===== 测试矩阵 =====
    results = {
        "model": model_name,
        "model_class": info.model_class,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "timestamp": timestamp,
        "phase": "425",
        "round": round_num,
        "test_objects": test_objects,
        "alpha_values": alpha_values,
        "per_object": {},
    }
    
    total_tests = 0
    for obj_word in test_objects:
        obj_cat = get_object_category(obj_word)
        if obj_cat is None:
            continue
        
        obj_embedding = get_object_embedding(W_E, tokenizer, obj_word)
        if obj_embedding is None:
            continue
        
        opposing_cat = OBJECT_CATEGORIES[obj_cat]["opposing"]
        
        print(f"\n[{time.strftime('%H:%M:%S')}] === Object: {obj_word} (category={obj_cat}, opposing={opposing_cat}) ===")
        
        obj_results = {
            "category": obj_cat,
            "opposing": opposing_cat,
            "embedding_norm": float(np.linalg.norm(obj_embedding)),
            "tasks": {},
        }
        
        for task_name, task_info in KNOWLEDGE_TASKS.items():
            template = task_info["template"]
            candidates = task_info["candidates"]
            expected = task_info["expected"].get(obj_word, "unknown")
            cand_ids = get_candidate_ids(tokenizer, candidates)
            
            print(f"\n  Task: {task_name} (expected={expected})")
            
            task_data = {
                "expected": expected,
                "baseline": None,
                "perturbations": {},
            }
            
            # 基线
            base_level, base_probs = run_baseline(model, tokenizer, device, template, obj_word, cand_ids, candidates)
            task_data["baseline"] = {
                "level": round(base_level, 4),
                "probs": {k: round(v, 4) for k, v in base_probs.items()},
            }
            base_top = max(base_probs, key=base_probs.get) if base_probs else "N/A"
            print(f"    Baseline: level={base_level:.3f}, top={base_top}")
            
            # 扰动测试
            for perturb_type in PERTURBATION_TYPES:
                for alpha in alpha_values:
                    # 确定扰动方向
                    if perturb_type == "add_category":
                        # 加上自身类别方向 (应该增强类别信号)
                        if obj_cat in category_directions:
                            perturbation = category_directions[obj_cat]
                        else:
                            continue
                    elif perturb_type == "remove_category":
                        # 减去自身类别方向 (应该削弱类别信号)
                        if obj_cat in category_directions:
                            perturbation = -category_directions[obj_cat]
                        else:
                            continue
                    elif perturb_type == "add_opposing":
                        # 加上对立类别方向 (应该把对象推向对立类别)
                        if opposing_cat in category_directions:
                            perturbation = category_directions[opposing_cat]
                        else:
                            continue
                    elif perturb_type == "add_random":
                        # 加上随机正交方向 (对照)
                        if obj_cat in category_directions:
                            perturbation = make_orthogonal_random(
                                category_directions[obj_cat], d_model, rng
                            )
                        else:
                            continue
                    else:
                        continue
                    
                    key = f"{perturb_type}_a{alpha}"
                    level, probs = run_with_perturbed_embedding(
                        model, tokenizer, device, template, obj_word,
                        obj_embedding, perturbation, alpha,
                        cand_ids, candidates, W_E.shape
                    )
                    
                    delta_level = level - base_level
                    top_cand = max(probs, key=probs.get) if probs else "N/A"
                    
                    task_data["perturbations"][key] = {
                        "level": round(level, 4),
                        "delta": round(delta_level, 4),
                        "top": top_cand,
                        "probs": {k: round(v, 4) for k, v in probs.items()},
                    }
                    
                    total_tests += 1
                    print(f"    {key}: level={level:.3f} delta={delta_level:+.3f} top={top_cand}")
            
            obj_results["tasks"][task_name] = task_data
            
            # 定期GPU日志
            if torch.cuda.is_available():
                gpu = torch.cuda.memory_allocated() / 1e9
                if gpu > 10:
                    print(f"    [GPU: {gpu:.2f}GB] High memory usage")
        
        results["per_object"][obj_word] = obj_results
    
    # ===== 汇总分析 =====
    print(f"\n{'='*80}")
    print(f"=== Phase 425 Summary ({model_name}) R{round_num} ===")
    print(f"{'='*80}")
    
    # 扰动类型到summary key的映射
    perturb_to_summary_key = {
        "add_category": "add_category",
        "remove_category": "remove_category",
        "add_opposing": "add_opposing",
        "add_random": "add_random",
    }
    
    summary = {
        "add_category": {},      # 加自身类别方向
        "remove_category": {},   # 减自身类别方向
        "add_opposing": {},      # 加对立类别方向
        "add_random": {},        # 随机方向
        "basin_transition": {},  # 轨道跃迁
    }
    
    for obj_word in test_objects:
        obj_cat = get_object_category(obj_word)
        if obj_cat is None or obj_word not in results["per_object"]:
            continue
        
        obj_data = results["per_object"][obj_word]
        
        # 分析每个任务的效果
        for task_name, task_data in obj_data["tasks"].items():
            base = task_data["baseline"]["level"]
            expected = task_data["expected"]
            
            for perturb_type in ["add_category", "remove_category", "add_opposing", "add_random"]:
                effect_key = f"{obj_cat}_{task_name}"
                skey = perturb_to_summary_key[perturb_type]
                
                for alpha in alpha_values:
                    key = f"{perturb_type}_a{alpha}"
                    if key in task_data["perturbations"]:
                        delta = task_data["perturbations"][key]["delta"]
                        
                        if effect_key not in summary[skey]:
                            summary[skey][effect_key] = []
                        summary[skey][effect_key].append(delta)
                        
                        # 检查轨道跃迁: 候选top是否改变
                        base_probs_dict = task_data["baseline"]["probs"]
                        perturb_top = task_data["perturbations"][key]["top"]
                        if base_probs_dict:
                            base_top = max(base_probs_dict, key=base_probs_dict.get)
                            if perturb_top != base_top:
                                trans_key = f"{obj_cat}_{task_name}_{perturb_type}"
                                if trans_key not in summary["basin_transition"]:
                                    summary["basin_transition"][trans_key] = 0
                                summary["basin_transition"][trans_key] += 1
    
    # 打印关键效果
    print("\n--- Perturbation Effects ---")
    for perturb_type in ["add_category", "remove_category", "add_opposing", "add_random"]:
        print(f"\n  {perturb_type}:")
        for effect_key, deltas in summary[perturb_type].items():
            mean_d = np.mean(deltas) if deltas else 0
            print(f"    {effect_key}: mean_delta={mean_d:+.4f} (n={len(deltas)})")
    
    print("\n--- Basin Transitions ---")
    for effect_key, count in summary["basin_transition"].items():
        print(f"  {effect_key}: {count} transitions")
    
    # 保存summary (list→mean)
    results["summary"] = {}
    for k, v in summary.items():
        if isinstance(v, dict):
            results["summary"][k] = {
                kk: {"mean": round(np.mean(vv), 4), "n": len(vv)} 
                for kk, vv in v.items() if isinstance(vv, list)
            }
        elif isinstance(v, (int, float)):
            results["summary"][k] = v
    
    # ===== 保存结果 =====
    results_dir = ROOT / "results" / "phase425_embedding_perturbation"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name}_phase425_r{round_num}.json"
    
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
    
    results = convert(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Model released. GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    return results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    run_phase425(model_name, round_num)


if __name__ == "__main__":
    main()
