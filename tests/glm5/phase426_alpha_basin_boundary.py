"""
Phase 426: 精细Alpha轨道边界扫描
=================================

基于Phase 425的发现:
1. Qwen3: alpha=0.5无效果, alpha=1.0直接跳轨道 → 存在临界阈值
2. GLM4: 任何扰动都敏感 → 需要更小alpha区分语义特异性和脆弱性
3. DS7B: 嵌入扰动几乎无效 → 需要确认是否真无效

本实验目标:
1. 精细alpha扫描: 0.02 ~ 2.0, 共20+个采样点
2. 只选single-token对象(解决多token问题)
3. 记录: 每个alpha的候选概率, entropy, top-1, 轨道归属
4. 定位: 临界跃迁阈值(basin boundary)

关键改进:
- 所有对象均为single-token(验证后确认)
- 更精细的alpha网格
- 记录entropy变化曲线
- 记录每个候选词的概率曲线(不只是level)
- 加对象身份残差扰动作为对照

Usage:
  python tests/glm5/phase426_alpha_basin_boundary.py qwen3 1
  python tests/glm5/phase426_alpha_basin_boundary.py glm4 1
  python tests/glm5/phase426_alpha_basin_boundary.py deepseek7b 1
"""

import sys
import os

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

# ===== 精细Alpha网格 =====
# R1用较粗的网格验证, R2用全网格
ALPHA_GRID_FINE = [
    0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.75, 0.90, 1.00, 1.25, 1.50, 1.75, 2.00
]
ALPHA_GRID_COARSE = [
    0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00
]

# ===== 对象和类别定义 =====
# 只选single-token对象, 避免多token污染
OBJECT_CATEGORIES = OrderedDict([
    ("fruit", {
        "objects": ["apple", "banana", "orange", "grape", "lemon", "mango", "peach"],
        "opposing": "animal",
    }),
    ("animal", {
        "objects": ["dog", "cat", "horse", "lion", "fish", "bird", "bear"],
        "opposing": "fruit",
    }),
    ("tool", {
        "objects": ["knife", "hammer", "spoon", "ruler", "nail", "chisel"],
        "opposing": "vehicle",
    }),
    ("vehicle", {
        "objects": ["car", "train", "bus", "truck", "boat", "ship"],
        "opposing": "tool",
    }),
    ("place", {
        "objects": ["desert", "forest", "ocean", "city", "island", "valley"],
        "opposing": "fruit",
    }),
])

# R1: 每个类别2个对象, R2: 每个类别3个
R1_OBJECTS = ["apple", "orange", "dog", "horse", "knife", "hammer", "car", "bus", "desert", "ocean"]
R2_OBJECTS = ["apple", "orange", "grape", "dog", "horse", "lion", "knife", "hammer", "spoon",
              "car", "bus", "train", "desert", "ocean", "forest"]

# 知识槽位任务 (同Phase 425)
KNOWLEDGE_TASKS = OrderedDict([
    ("category", {
        "template": "A {obj} is a kind of",
        "candidates": OrderedDict([
            ("fruit", 1), ("animal", 2), ("tool", 3), ("vehicle", 4), ("place", 5),
        ]),
    }),
    ("property", {
        "template": "The most notable property of a {obj} is that it is",
        "candidates": OrderedDict([
            ("edible", 1), ("alive", 2), ("sharp", 3), ("fast", 4), ("vast", 5),
        ]),
    }),
    ("part", {
        "template": "A {obj} has",
        "candidates": OrderedDict([
            ("seeds", 1), ("fur", 2), ("blades", 3), ("wheels", 4), ("sand", 5),
        ]),
    }),
])

# 扰动类型
PERTURBATION_TYPES = ["remove_category", "add_opposing", "add_random", "remove_identity"]


def load_model_bf16(model_name):
    """BF16 + device_map=auto + flash attention 加载模型"""
    cfg = MODEL_CONFIGS[model_name]
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} (BF16+auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # flash_attention_2 > sdpa > eager
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


def verify_single_token(tokenizer, obj_word):
    """验证对象词是否为single-token"""
    tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
    return len(tok_ids) == 1, tok_ids


def get_category_directions(model, tokenizer, device):
    """构造类别功能方向(同Phase 425)"""
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().cpu().float().numpy()

    # 类别中心
    category_centers = {}
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        vecs = []
        for obj_word in cat_info["objects"]:
            tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
            if tok_ids:
                vecs.append(W_E[tok_ids[0]])
        if vecs:
            category_centers[cat_name] = np.mean(vecs, axis=0)

    # 类别方向
    category_directions = {}
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        opposing = cat_info["opposing"]
        if cat_name in category_centers and opposing in category_centers:
            d = category_centers[cat_name] - category_centers[opposing]
            norm = np.linalg.norm(d)
            if norm > 0:
                d = d / norm
            category_directions[cat_name] = d

    # 全局中心
    all_vecs = list(category_centers.values())
    global_center = np.mean(all_vecs, axis=0)

    return category_directions, category_centers, global_center, W_E


def get_candidate_ids(tokenizer, candidates):
    """获取候选词的token IDs"""
    cand_ids = {}
    for cand in candidates:
        ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        if ids:
            cand_ids[cand] = ids[-1]
    return cand_ids


def compute_entropy(probs_dict):
    """计算概率分布的entropy"""
    probs = np.array(list(probs_dict.values()))
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def run_with_perturbed_embedding(model, tokenizer, device, template, obj_word,
                                  perturbation, alpha, cand_ids, candidates):
    """用扰动后的embedding运行前向推理, 返回(level, probs_dict, entropy)"""
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)

    obj_tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)

    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)

    # 找到obj token在input_ids中的位置
    obj_positions = []
    prompt_ids = input_ids[0].cpu().tolist()
    for i in range(len(prompt_ids) - len(obj_tok_ids) + 1):
        if prompt_ids[i:i+len(obj_tok_ids)] == obj_tok_ids:
            obj_positions = list(range(i, i + len(obj_tok_ids)))
            break

    if not obj_positions:
        obj_positions = [1]

    # 应用扰动到所有obj token位置
    perturbed_embeds = inputs_embeds.clone()
    perturbation_tensor = torch.tensor(
        perturbation * alpha, dtype=perturbed_embeds.dtype, device=device
    )
    for pos in obj_positions:
        perturbed_embeds[0, pos, :] += perturbation_tensor

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        try:
            outputs = model(inputs_embeds=perturbed_embeds, attention_mask=attention_mask)
        except Exception:
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            except:
                empty = {c: 1.0/len(candidates) for c in candidates}
                return sum(candidates[c]*empty[c] for c in candidates), empty, compute_entropy(empty)

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
    entropy = compute_entropy(result)

    return level, result, entropy


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
    entropy = compute_entropy(result)
    top_cand = max(result, key=result.get) if result else "N/A"

    return level, result, entropy, top_cand


def get_object_category(obj_word):
    """获取对象所属类别"""
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        if obj_word in cat_info["objects"]:
            return cat_name
    return None


def make_orthogonal_random(direction, d_model, rng, norm=1.0):
    """生成与给定方向正交的随机方向"""
    d = rng.randn(d_model)
    proj = np.dot(d, direction) / max(np.dot(direction, direction), 1e-10) * direction
    d = d - proj
    d_norm = np.linalg.norm(d)
    if d_norm > 0:
        d = d / d_norm * norm
    return d


def find_critical_alpha(alpha_curve, level_curve, base_level, threshold=0.5):
    """找到临界alpha: level变化超过threshold的alpha值"""
    for i in range(1, len(alpha_curve)):
        delta = abs(level_curve[i] - base_level)
        if delta > threshold:
            # 线性插值
            alpha_prev = alpha_curve[i-1]
            alpha_curr = alpha_curve[i]
            level_prev = level_curve[i-1]
            level_curr = level_curve[i]
            delta_prev = abs(level_prev - base_level)
            delta_curr = abs(level_curr - base_level)
            if delta_curr - delta_prev > 1e-10:
                frac = (threshold - delta_prev) / (delta_curr - delta_prev)
                return alpha_prev + frac * (alpha_curr - alpha_prev)
            return alpha_curr
    return None


def run_phase426(model_name, round_num=1):
    """运行Phase 426实验"""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 426: Alpha Basin Boundary Scan ({model_name}) R{round_num} [{timestamp}] ===")
    print(f"{'='*80}")

    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # 获取W_E和类别方向
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing category directions...")
    category_directions, category_centers, global_center, W_E = get_category_directions(model, tokenizer, device)

    for cat_name, d in category_directions.items():
        print(f"  d_{cat_name}: norm={np.linalg.norm(d):.4f}")

    # 选择测试对象
    if round_num == 1:
        test_objects = R1_OBJECTS
        alpha_grid = ALPHA_GRID_COARSE
    else:
        test_objects = R2_OBJECTS
        alpha_grid = ALPHA_GRID_FINE

    # 过滤: 只保留single-token对象
    single_token_objects = []
    for obj_word in test_objects:
        is_single, tok_ids = verify_single_token(tokenizer, obj_word)
        if is_single:
            single_token_objects.append(obj_word)
        else:
            print(f"  WARNING: '{obj_word}' is multi-token ({tok_ids}), skipping")

    test_objects = single_token_objects
    print(f"\n  Testing {len(test_objects)} single-token objects with {len(alpha_grid)} alpha values")

    rng = np.random.RandomState(42)
    d_model = W_E.shape[1]

    # ===== 测试矩阵 =====
    results = {
        "model": model_name,
        "model_class": info.model_class,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "timestamp": timestamp,
        "phase": "426",
        "round": round_num,
        "alpha_grid": alpha_grid,
        "test_objects": test_objects,
        "per_object": {},
        "critical_alphas": {},
    }

    total_tests = 0
    t_start = time.time()

    for obj_idx, obj_word in enumerate(test_objects):
        obj_cat = get_object_category(obj_word)
        if obj_cat is None:
            continue

        opposing_cat = OBJECT_CATEGORIES[obj_cat]["opposing"]

        # 对象身份残差: e_apple - center(fruit)
        obj_tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
        obj_embedding = W_E[obj_tok_ids[0]].copy()
        identity_residual = obj_embedding - category_centers.get(obj_cat, np.zeros(d_model))
        id_norm = np.linalg.norm(identity_residual)
        if id_norm > 0:
            identity_direction = identity_residual / id_norm
        else:
            identity_direction = np.zeros(d_model)

        print(f"\n[{time.strftime('%H:%M:%S')}] === Object {obj_idx+1}/{len(test_objects)}: "
              f"{obj_word} (cat={obj_cat}, opp={opposing_cat}) ===")

        obj_results = {
            "category": obj_cat,
            "opposing": opposing_cat,
            "embedding_norm": float(np.linalg.norm(obj_embedding)),
            "identity_norm": float(id_norm),
            "tasks": {},
        }

        for task_name, task_info in KNOWLEDGE_TASKS.items():
            template = task_info["template"]
            candidates = task_info["candidates"]
            cand_ids = get_candidate_ids(tokenizer, candidates)

            print(f"\n  Task: {task_name}")

            task_data = {
                "expected_category": obj_cat,
                "baselines": {},  # {alpha: 0.0: {level, probs, entropy, top}}
                "perturbation_curves": {},  # {perturb_type: {alpha: {level, delta, entropy, top, probs}}}
            }

            # ===== 1. 基线(alpha=0) =====
            base_level, base_probs, base_entropy, base_top = run_baseline(
                model, tokenizer, device, template, obj_word, cand_ids, candidates
            )
            task_data["baselines"]["0.0"] = {
                "level": round(base_level, 4),
                "entropy": round(base_entropy, 4),
                "top": base_top,
                "probs": {k: round(v, 4) for k, v in base_probs.items()},
            }
            print(f"    Baseline: level={base_level:.3f}, entropy={base_entropy:.3f}, top={base_top}")

            # ===== 2. 扰动曲线 =====
            for perturb_type in PERTURBATION_TYPES:
                # 确定扰动方向
                if perturb_type == "remove_category":
                    if obj_cat in category_directions:
                        direction = -category_directions[obj_cat]
                    else:
                        continue
                elif perturb_type == "add_opposing":
                    if opposing_cat in category_directions:
                        direction = category_directions[opposing_cat]
                    else:
                        continue
                elif perturb_type == "add_random":
                    if obj_cat in category_directions:
                        direction = make_orthogonal_random(
                            category_directions[obj_cat], d_model, rng
                        )
                    else:
                        continue
                elif perturb_type == "remove_identity":
                    direction = -identity_direction
                else:
                    continue

                curve_data = {}

                for alpha in alpha_grid:
                    if alpha == 0.0:
                        # 已有基线
                        curve_data[str(alpha)] = {
                            "level": round(base_level, 4),
                            "delta": 0.0,
                            "entropy": round(base_entropy, 4),
                            "top": base_top,
                        }
                        continue

                    level, probs, entropy = run_with_perturbed_embedding(
                        model, tokenizer, device, template, obj_word,
                        direction, alpha, cand_ids, candidates
                    )
                    delta = level - base_level
                    top_cand = max(probs, key=probs.get) if probs else "N/A"

                    curve_data[str(alpha)] = {
                        "level": round(level, 4),
                        "delta": round(delta, 4),
                        "entropy": round(entropy, 4),
                        "top": top_cand,
                    }

                    total_tests += 1

                # 保存曲线数据
                task_data["perturbation_curves"][perturb_type] = curve_data

                # 打印关键alpha点
                key_alphas = ["0.0", "0.1", "0.3", "0.5", "1.0", "2.0"]
                print(f"    {perturb_type}: ", end="")
                for ka in key_alphas:
                    if ka in curve_data:
                        d = curve_data[ka]
                        print(f"a={ka}→{d['level']:.2f}(Δ{d['delta']:+.2f},{d['top'][:3]}) ", end="")
                print()

            obj_results["tasks"][task_name] = task_data

            # GPU日志
            if torch.cuda.is_available():
                gpu = torch.cuda.memory_allocated() / 1e9
                if gpu > 10:
                    print(f"    [GPU: {gpu:.2f}GB]")

        # ===== 3. 计算临界alpha =====
        critical_alphas = {}
        for task_name, task_data in obj_results["tasks"].items():
            base_level = task_data["baselines"]["0.0"]["level"]
            task_crits = {}
            for perturb_type, curve in task_data["perturbation_curves"].items():
                alphas = sorted([float(a) for a in curve.keys()])
                levels = [curve[str(a)]["level"] for a in alphas]
                crit = find_critical_alpha(alphas, levels, base_level, threshold=0.5)
                if crit is not None:
                    task_crits[perturb_type] = round(crit, 3)
            if task_crits:
                critical_alphas[task_name] = task_crits

        results["critical_alphas"][obj_word] = critical_alphas
        print(f"  Critical alphas: {critical_alphas}")

        results["per_object"][obj_word] = obj_results

        # 进度
        elapsed = time.time() - t_start
        est_total = elapsed / (obj_idx + 1) * len(test_objects)
        print(f"  [{time.strftime('%H:%M:%S')}] Progress: {obj_idx+1}/{len(test_objects)}, "
              f"elapsed={elapsed/60:.1f}min, est_total={est_total/60:.1f}min")

    # ===== 汇总分析 =====
    print(f"\n{'='*80}")
    print(f"=== Phase 426 Summary ({model_name}) R{round_num} ===")
    print(f"{'='*80}")

    # 1. 临界alpha汇总
    print("\n--- Critical Alpha (|delta| > 0.5) ---")
    for obj_word, crits in results["critical_alphas"].items():
        print(f"  {obj_word}:")
        for task_name, task_crits in crits.items():
            for perturb_type, alpha_val in task_crits.items():
                print(f"    {task_name}/{perturb_type}: α_c = {alpha_val}")

    # 2. 语义特异性比: 不同alpha下category vs random
    print("\n--- Semantic Specificity Ratio (category_effect / random_effect) ---")
    specificity_ratios = {}
    for obj_word in test_objects:
        if obj_word not in results["per_object"]:
            continue
        obj_data = results["per_object"][obj_word]
        obj_cat = get_object_category(obj_word)
        if obj_cat is None:
            continue

        for task_name, task_data in obj_data["tasks"].items():
            cat_curve = task_data["perturbation_curves"].get("remove_category", {})
            rand_curve = task_data["perturbation_curves"].get("add_random", {})

            for alpha_str in ["0.5", "1.0"]:
                cat_delta = abs(cat_curve.get(alpha_str, {}).get("delta", 0))
                rand_delta = abs(rand_curve.get(alpha_str, {}).get("delta", 0))

                ratio = cat_delta / rand_delta if rand_delta > 0.01 else float('inf')
                key = f"{obj_word}_{task_name}_a{alpha_str}"
                specificity_ratios[key] = round(ratio, 2)

                if alpha_str == "1.0":
                    print(f"  {key}: category|Δ|={cat_delta:.3f}, random|Δ|={rand_delta:.3f}, ratio={ratio:.2f}")

    results["specificity_ratios"] = specificity_ratios

    # 3. 轨道跃迁alpha (top-1变化点)
    print("\n--- Top-1 Switch Points ---")
    top1_switches = {}
    for obj_word in test_objects:
        if obj_word not in results["per_object"]:
            continue
        obj_data = results["per_object"][obj_word]

        for task_name, task_data in obj_data["tasks"].items():
            base_top = task_data["baselines"]["0.0"]["top"]

            for perturb_type, curve in task_data["perturbation_curves"].items():
                prev_top = base_top
                switch_alpha = None
                for alpha_str in sorted([float(a) for a in curve.keys()]):
                    curr_top = curve[str(alpha_str)]["top"]
                    if curr_top != prev_top and alpha_str > 0:
                        switch_alpha = alpha_str
                        break
                    prev_top = curr_top

                if switch_alpha is not None:
                    key = f"{obj_word}_{task_name}_{perturb_type}"
                    top1_switches[key] = switch_alpha
                    if perturb_type in ["remove_category", "add_opposing"]:
                        new_top = curve[str(switch_alpha)]["top"]
                        print(f"  {key}: switch at α={switch_alpha}, {base_top}→{new_top}")

    results["top1_switches"] = top1_switches

    # ===== 保存结果 =====
    results_dir = ROOT / "results" / "phase426_alpha_basin_boundary"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name}_phase426_r{round_num}.json"

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

    run_phase426(model_name, round_num)


if __name__ == "__main__":
    main()
