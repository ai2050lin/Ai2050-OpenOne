"""
Phase 94b: 语义结构图谱 — 概率轨迹深度分析
==========================================
基于Phase 94a的发现，进行更深入的分析:
  1. 翻译对齐的完整概率轨迹 (每层)
  2. 跨结构类型的涌现层对比
  3. 架构先验控制 (随机模型)
  4. 跨模型对比 (Qwen3 vs DS7B)

Run:
  python tests/glm5/ccml_phase94b_trajectory.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase94b_trajectory.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase94b_trajectory.py --model deepseek7b --exp 1
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F_torch
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict

from model_utils import load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS


# ============================================================
# 翻译测试用例 (最稳定的结构)
# ============================================================
TRANSLATION_TESTS = [
    ("苹果的英文是", "apple", "苹果"),
    ("猫的英文是", "cat", "猫"),
    ("狗的英文是", "dog", "狗"),
    ("书的英文是", "book", "书"),
    ("水的英文是", "water", "水"),
    ("火的英文是", "fire", "火"),
    ("花的英文是", "flower", "花"),
    ("鱼的英文是", "fish", "鱼"),
    ("太阳的英文是", "sun", "太阳"),
    ("月亮的英文是", "moon", "月亮"),
    ("红色的英文是", "red", "红色"),
    ("蓝色的英文是", "blue", "蓝色"),
    ("绿色的英文是", "green", "绿色"),
    ("白色的英文是", "white", "白色"),
    ("黑色的英文是", "black", "黑色"),
]

# ============================================================
# 类比测试用例
# ============================================================
ANALOGY_TESTS = [
    ("苹果属于水果，狗属于什么？答案是", "动物"),
    ("医生在医院工作，教师在哪里工作？答案是", "学校"),
    ("眼睛负责视觉，耳朵负责什么？答案是", "听觉"),
    ("北京是中国的首都，东京是哪个国家的首都？答案是", "日本"),
    ("猫属于哺乳动物，蛇属于什么？答案是", "爬行动物"),
]

# ============================================================
# 简单事实测试 (作为对照组)
# ============================================================
FACT_TESTS = [
    ("法国的首都是", "巴黎"),
    ("日本的首都是", "东京"),
    ("中国的首都是", "北京"),
    ("英国的首都是", "伦敦"),
    ("德国的首都是", "柏林"),
    ("水的化学式是", "H2O"),
    ("铁的化学符号是", "Fe"),
    ("金的颜色是", "金色"),
    ("雪的颜色是", "白色"),
    ("草的颜色是", "绿色"),
]


def get_full_probability_trajectory(model, tokenizer, device, prompt, target_str, n_layers):
    """
    获取目标token在每层的完整概率轨迹
    返回: {
        layer_idx: {
            "prob": float, "rank": int, "top1_token": str, "top1_prob": float,
            "top5": [{"token": str, "prob": float}]
        }
    }
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Get target token ids - try multiple formats
    target_variants = [target_str, f" {target_str}", f'"{target_str}', f"\"{target_str}"]
    target_ids = set()
    for variant in target_variants:
        try:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            target_ids.update(ids)
        except:
            pass
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    
    W_U = model.lm_head.weight.data.float()
    
    trajectory = {}
    for layer_idx in range(n_layers + 1):
        h = outputs.hidden_states[layer_idx][0, -1, :].float()
        logits = h @ W_U.T
        probs = torch.softmax(logits, dim=-1)
        
        # Find target prob (best variant)
        best_target_prob = 0
        best_target_rank = probs.shape[0]
        best_target_token = ""
        
        for tid in target_ids:
            if 0 <= tid < probs.shape[0]:
                p = probs[tid].item()
                r = (probs > p).sum().item() + 1
                if p > best_target_prob:
                    best_target_prob = p
                    best_target_rank = r
                    best_target_token = tokenizer.decode([tid])
        
        # Top-5 tokens
        top5_vals, top5_ids = torch.topk(probs, 5)
        top5 = []
        for i in range(5):
            try:
                tok_str = tokenizer.decode([top5_ids[i].item()])
            except:
                tok_str = f"<id:{top5_ids[i].item()}>"
            top5.append({
                "token": tok_str,
                "prob": top5_vals[i].item()
            })
        
        trajectory[layer_idx] = {
            "prob": best_target_prob,
            "rank": best_target_rank,
            "token": best_target_token,
            "top1_token": top5[0]["token"],
            "top1_prob": top5[0]["prob"],
            "top5": top5
        }
    
    return trajectory


def analyze_trajectory(trajectory, n_layers):
    """分析概率轨迹的关键特征"""
    # 涌现层: 目标首次成为top-1
    emergence_layer = None
    # 信息出现层: 目标prob > 0.01
    appearance_layer = None
    # 最大概率层
    max_prob = 0
    max_prob_layer = None
    
    for l in range(n_layers + 1):
        t = trajectory[l]
        if t["rank"] == 1 and emergence_layer is None:
            emergence_layer = l
        if t["prob"] > 0.01 and appearance_layer is None:
            appearance_layer = l
        if t["prob"] > max_prob:
            max_prob = t["prob"]
            max_prob_layer = l
    
    # 计算概率增长率 (信息涌现速率)
    growth_rates = {}
    for l in range(1, n_layers + 1):
        if trajectory[l-1]["prob"] > 1e-8:
            growth_rates[l] = trajectory[l]["prob"] / max(trajectory[l-1]["prob"], 1e-8)
        else:
            growth_rates[l] = 0
    
    # 找到最大增长层
    max_growth_layer = None
    max_growth_rate = 0
    for l, rate in growth_rates.items():
        if rate > max_growth_rate and rate < 1e6:  # exclude extreme outliers
            max_growth_rate = rate
            max_growth_layer = l
    
    return {
        "emergence_layer": emergence_layer,
        "appearance_layer": appearance_layer,
        "max_prob_layer": max_prob_layer,
        "max_prob": max_prob,
        "final_prob": trajectory[n_layers]["prob"],
        "max_growth_layer": max_growth_layer,
        "max_growth_rate": max_growth_rate,
    }


# ============================================================
# Experiment 1: 翻译对齐完整轨迹
# ============================================================
def exp1_translation_trajectory(model_name):
    """翻译对齐的完整概率轨迹分析"""
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"\n{'='*70}")
    print(f"实验1: 翻译对齐完整概率轨迹")
    print(f"模型: {model_name}, 层数: {n_layers}")
    print(f"{'='*70}")
    
    all_results = {"structure": "translation_trajectory", "model": model_name, "cases": []}
    
    for idx, (prompt, target, chinese) in enumerate(TRANSLATION_TESTS):
        print(f"\n--- 案例 {idx+1}: '{prompt}' → '{target}' ---")
        
        trajectory = get_full_probability_trajectory(
            model, tokenizer, device, prompt, target, n_layers)
        
        analysis = analyze_trajectory(trajectory, n_layers)
        
        print(f"  信息出现层: {analysis['appearance_layer']}")
        print(f"  涌现层(top-1): {analysis['emergence_layer']}")
        print(f"  最大概率层: {analysis['max_prob_layer']} (prob={analysis['max_prob']:.4f})")
        print(f"  最大增长层: {analysis['max_growth_layer']} (rate={analysis['max_growth_rate']:.1f}x)")
        print(f"  最终概率: {analysis['final_prob']:.4f}")
        
        # Print key layer trajectory
        key_layers = list(range(0, n_layers + 1, max(1, n_layers // 12)))
        if n_layers not in key_layers:
            key_layers.append(n_layers)
        if analysis['appearance_layer'] and analysis['appearance_layer'] not in key_layers:
            key_layers.append(analysis['appearance_layer'])
        if analysis['emergence_layer'] and analysis['emergence_layer'] not in key_layers:
            key_layers.append(analysis['emergence_layer'])
        key_layers.sort()
        
        print(f"  概率轨迹:")
        for l in key_layers:
            t = trajectory[l]
            marker = ""
            if l == analysis['appearance_layer']:
                marker = " [APPEAR]"
            if l == analysis['emergence_layer']:
                marker = " [EMERGE]"
            if l == analysis['max_growth_layer']:
                marker = " [GROWTH]"
            print(f"    L{l:2d}: prob={t['prob']:.6f}, rank={t['rank']:5d}, "
                  f"top1='{t['top1_token']}'({t['top1_prob']:.4f}){marker}")
        
        # Save trajectory data (compact)
        compact_trajectory = {}
        for l in range(n_layers + 1):
            compact_trajectory[l] = {
                "prob": trajectory[l]["prob"],
                "rank": trajectory[l]["rank"],
                "top1": trajectory[l]["top1_token"]
            }
        
        all_results["cases"].append({
            "prompt": prompt, "target": target, "chinese": chinese,
            "analysis": analysis, "trajectory": compact_trajectory
        })
    
    # Aggregate statistics
    print(f"\n{'='*50}")
    print(f"翻译对齐结构汇总统计")
    print(f"{'='*50}")
    
    appear_layers = [c["analysis"]["appearance_layer"] for c in all_results["cases"] 
                     if c["analysis"]["appearance_layer"] is not None]
    emerge_layers = [c["analysis"]["emergence_layer"] for c in all_results["cases"]
                     if c["analysis"]["emergence_layer"] is not None]
    growth_layers = [c["analysis"]["max_growth_layer"] for c in all_results["cases"]
                     if c["analysis"]["max_growth_layer"] is not None]
    
    if appear_layers:
        print(f"  信息出现层: mean={np.mean(appear_layers):.1f} ± {np.std(appear_layers):.1f}, "
              f"range=[{min(appear_layers)}, {max(appear_layers)}]")
    if emerge_layers:
        print(f"  涌现层(top-1): mean={np.mean(emerge_layers):.1f} ± {np.std(emerge_layers):.1f}, "
              f"range=[{min(emerge_layers)}, {max(emerge_layers)}]")
    if growth_layers:
        print(f"  最大增长层: mean={np.mean(growth_layers):.1f} ± {np.std(growth_layers):.1f}, "
              f"range=[{min(growth_layers)}, {max(growth_layers)}]")
    
    # Per-layer average probability trajectory
    print(f"\n  平均概率轨迹:")
    for l in range(0, n_layers + 1, max(1, n_layers // 12)):
        probs = [c["trajectory"][l]["prob"] for c in all_results["cases"] if l in c["trajectory"]]
        if probs:
            print(f"    L{l:2d}: mean_prob={np.mean(probs):.6f}, max_prob={np.max(probs):.6f}")
    
    all_results["aggregate"] = {
        "n_cases": len(all_results["cases"]),
        "appearance_mean": float(np.mean(appear_layers)) if appear_layers else None,
        "emergence_mean": float(np.mean(emerge_layers)) if emerge_layers else None,
        "growth_mean": float(np.mean(growth_layers)) if growth_layers else None,
    }
    
    # Save
    output_path = f"tests/glm5_temp/phase94b_{model_name}_translation_trajectory.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")
    
    release_model(model)
    return all_results


# ============================================================
# Experiment 2: 跨结构类型对比
# ============================================================
def exp2_cross_structure_comparison(model_name):
    """对比不同结构类型的涌现模式"""
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"\n{'='*70}")
    print(f"实验2: 跨结构类型对比")
    print(f"模型: {model_name}, 层数: {n_layers}")
    print(f"{'='*70}")
    
    structures = {
        "translation": TRANSLATION_TESTS[:10],
        "analogy": ANALOGY_TESTS,
        "fact": FACT_TESTS,
    }
    
    all_results = {"model": model_name, "structures": {}}
    
    for struct_name, tests in structures.items():
        print(f"\n--- 结构类型: {struct_name} ---")
        
        struct_results = []
        for idx, test in enumerate(tests):
            prompt = test[0]
            target = test[1]
            
            trajectory = get_full_probability_trajectory(
                model, tokenizer, device, prompt, target, n_layers)
            analysis = analyze_trajectory(trajectory, n_layers)
            
            print(f"  案例 {idx+1}: '{prompt[:25]}...' → '{target}'")
            print(f"    出现层={analysis['appearance_layer']}, "
                  f"涌现层={analysis['emergence_layer']}, "
                  f"最终prob={analysis['final_prob']:.4f}")
            
            struct_results.append({
                "prompt": prompt, "target": target, "analysis": analysis
            })
        
        # Aggregate for this structure
        appear = [r["analysis"]["appearance_layer"] for r in struct_results 
                  if r["analysis"]["appearance_layer"] is not None]
        emerge = [r["analysis"]["emergence_layer"] for r in struct_results
                  if r["analysis"]["emergence_layer"] is not None]
        final_probs = [r["analysis"]["final_prob"] for r in struct_results]
        
        all_results["structures"][struct_name] = {
            "cases": struct_results,
            "n_success": len(emerge),
            "n_total": len(struct_results),
            "success_rate": len(emerge) / len(struct_results),
            "appearance_mean": float(np.mean(appear)) if appear else None,
            "emergence_mean": float(np.mean(emerge)) if emerge else None,
            "mean_final_prob": float(np.mean(final_probs)),
        }
        
        if emerge:
            print(f"  汇总: 成功率={len(emerge)}/{len(struct_results)}, "
                  f"涌现层={np.mean(emerge):.1f}±{np.std(emerge):.1f}, "
                  f"平均最终prob={np.mean(final_probs):.4f}")
        else:
            print(f"  汇总: 成功率=0/{len(struct_results)}, "
                  f"平均最终prob={np.mean(final_probs):.4f}")
    
    # Cross-structure comparison
    print(f"\n{'='*50}")
    print(f"跨结构类型对比")
    print(f"{'='*50}")
    print(f"  {'结构类型':>12s} | {'成功率':>6s} | {'涌现层':>12s} | {'平均最终prob':>12s}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*12}-+-{'-'*12}")
    for struct_name, data in all_results["structures"].items():
        em = data["emergence_mean"]
        em_str = f"{em:.1f}" if em else "N/A"
        print(f"  {struct_name:>12s} | {data['success_rate']:5.1%} | {em_str:>12s} | {data['mean_final_prob']:12.4f}")
    
    # Save
    output_path = f"tests/glm5_temp/phase94b_{model_name}_cross_structure.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")
    
    release_model(model)
    return all_results


# ============================================================
# Experiment 3: 架构先验控制 (随机模型)
# ============================================================
def exp3_architecture_control(model_name):
    """架构先验控制: 对比训练模型和随机模型"""
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"\n{'='*70}")
    print(f"实验3: 架构先验控制 — 翻译对齐")
    print(f"模型: {model_name}, 层数: {n_layers}")
    print(f"{'='*70}")
    
    # Trained model trajectory
    print(f"\n--- 训练模型 ---")
    trained_results = []
    for idx, (prompt, target, chinese) in enumerate(TRANSLATION_TESTS[:8]):
        trajectory = get_full_probability_trajectory(
            model, tokenizer, device, prompt, target, n_layers)
        analysis = analyze_trajectory(trajectory, n_layers)
        trained_results.append({
            "prompt": prompt, "target": target,
            "final_prob": analysis["final_prob"],
            "emergence_layer": analysis["emergence_layer"],
            "trajectory": {l: trajectory[l]["prob"] for l in range(n_layers + 1)}
        })
        print(f"  '{chinese}→{target}': final_prob={analysis['final_prob']:.4f}, "
              f"emergence={analysis['emergence_layer']}")
    
    # Create random model
    print(f"\n--- 创建随机模型 ---")
    import copy
    random_model = copy.deepcopy(model)
    
    # Randomize weights while keeping architecture
    for name, param in random_model.named_parameters():
        if 'embed' in name or 'lm_head' in name:
            # Keep embedding and lm_head structure but randomize
            param.data.normal_(0, param.data.std() if param.data.std() > 0 else 0.02)
        else:
            param.data.normal_(0, param.data.std() if param.data.std() > 0 else 0.02)
    
    random_model.eval()
    
    # Random model trajectory
    print(f"\n--- 随机模型 ---")
    random_results = []
    for idx, (prompt, target, chinese) in enumerate(TRANSLATION_TESTS[:8]):
        trajectory = get_full_probability_trajectory(
            random_model, tokenizer, device, prompt, target, n_layers)
        analysis = analyze_trajectory(trajectory, n_layers)
        random_results.append({
            "prompt": prompt, "target": target,
            "final_prob": analysis["final_prob"],
            "emergence_layer": analysis["emergence_layer"],
            "trajectory": {l: trajectory[l]["prob"] for l in range(n_layers + 1)}
        })
        print(f"  '{chinese}→{target}': final_prob={analysis['final_prob']:.4f}, "
              f"emergence={analysis['emergence_layer']}")
    
    # Comparison
    print(f"\n{'='*50}")
    print(f"训练 vs 随机对比")
    print(f"{'='*50}")
    
    trained_final = [r["final_prob"] for r in trained_results]
    random_final = [r["final_prob"] for r in random_results]
    trained_emerge = [r["emergence_layer"] for r in trained_results if r["emergence_layer"] is not None]
    random_emerge = [r["emergence_layer"] for r in random_results if r["emergence_layer"] is not None]
    
    print(f"  训练模型: mean_final_prob={np.mean(trained_final):.4f}, "
          f"n_emerged={len(trained_emerge)}/{len(trained_results)}")
    print(f"  随机模型: mean_final_prob={np.mean(random_final):.4f}, "
          f"n_emerged={len(random_emerge)}/{len(random_results)}")
    print(f"  训练/随机概率比: {np.mean(trained_final)/max(np.mean(random_final), 1e-8):.1f}x")
    
    # Per-layer trajectory comparison
    print(f"\n  逐层平均概率对比:")
    for l in range(0, n_layers + 1, max(1, n_layers // 12)):
        t_probs = [r["trajectory"][l] for r in trained_results]
        r_probs = [r["trajectory"][l] for r in random_results]
        ratio = np.mean(t_probs) / max(np.mean(r_probs), 1e-8)
        print(f"    L{l:2d}: trained={np.mean(t_probs):.6f}, random={np.mean(r_probs):.6f}, "
              f"ratio={ratio:.1f}x")
    
    # Save
    results = {
        "model": model_name,
        "trained": trained_results,
        "random": random_results,
        "trained_mean_prob": float(np.mean(trained_final)),
        "random_mean_prob": float(np.mean(random_final)),
    }
    
    output_path = f"tests/glm5_temp/phase94b_{model_name}_arch_control.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")
    
    del random_model
    gc.collect()
    torch.cuda.empty_cache()
    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--exp", type=str, default="1",
                       help="Experiment number (1-3) or 'all'")
    args = parser.parse_args()
    
    exp_map = {
        "1": exp1_translation_trajectory,
        "2": exp2_cross_structure_comparison,
        "3": exp3_architecture_control,
    }
    
    if args.exp == "all":
        for exp_num in ["1", "2", "3"]:
            print(f"\n\n{'#'*70}")
            print(f"# Running Experiment {exp_num}")
            print(f"{'#'*70}")
            exp_map[exp_num](args.model)
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
    else:
        exp_map[args.exp](args.model)
