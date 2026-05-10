"""
Phase 96: 语义电路图谱 — 从"观察表示"到"观察计算"
====================================================
核心转折: Phase 95的最大失败是仍在看hidden state(表示)而非circuit(计算)
本Phase正式进入电路分析，回答: 哪些attention head/MLP真正负责翻译/检索/类比?

方法:
  Exp 1: Attention Head Ablation — 逐个关闭head，测量功能崩塌
  Exp 2: Residual Stream Patching — 跨prompt激活注入，真正的因果追踪
  Exp 3: 跨结构电路重叠 — 不同语言能力是否共享计算原语

关键方法论修正:
  - 不用mean(h_A-h_B)作为"翻译方向" — 这只是相关方向
  - 不用线性probe — Phase 95已证明R²为负不代表信息不存在
  - 直接做因果干预: ablate某个head → 功能是否崩塌

Run:
  python tests/glm5/ccml_phase96_circuit_atlas.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase96_circuit_atlas.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase96_circuit_atlas.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase96_circuit_atlas.py --model glm4 --exp 1
  python tests/glm5/ccml_phase96_circuit_atlas.py --model deepseek7b --exp 1
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict
from copy import deepcopy

from model_utils import load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS


# ============================================================
# 统一测试用例 — 每种结构50个样本
# ============================================================
TRANSLATION_PAIRS = [
    ("苹果的英文是", "apple"), ("猫的英文是", "cat"), ("狗的英文是", "dog"),
    ("书的英文是", "book"), ("水的英文是", "water"), ("火的英文是", "fire"),
    ("花的英文是", "flower"), ("鱼的英文是", "fish"), ("太阳的英文是", "sun"),
    ("月亮的英文是", "moon"), ("山的英文是", "mountain"), ("河的英文是", "river"),
    ("树的英文是", "tree"), ("天空的英文是", "sky"), ("风的英文是", "wind"),
    ("雨的英文是", "rain"), ("雪的英文是", "snow"), ("星的英文是", "star"),
    ("海的英文是", "sea"), ("草的英文是", "grass"), ("石的英文是", "stone"),
    ("路的英文是", "road"), ("门的英文是", "door"), ("窗的英文是", "window"),
    ("鸟的英文是", "bird"), ("马的英文是", "horse"), ("牛的英文是", "cow"),
    ("羊的英文是", "sheep"), ("猪的英文是", "pig"), ("鸡的英文是", "chicken"),
    ("米的英文是", "rice"), ("茶的英文是", "tea"), ("糖的英文是", "sugar"),
    ("盐的英文是", "salt"), ("油的英文是", "oil"), ("铁的英文是", "iron"),
    ("金的英文是", "gold"), ("银的英文是", "silver"), ("铜的英文是", "copper"),
    ("木的英文是", "wood"), ("纸的英文是", "paper"), ("布的英文是", "cloth"),
    ("血的英文是", "blood"), ("骨的英文是", "bone"), ("皮的英文是", "skin"),
    ("毛的英文是", "hair"), ("眼的英文是", "eye"), ("耳的英文是", "ear"),
    ("手的英文是", "hand"), ("脚的英文是", "foot"),
]

FACT_RETRIEVAL_PAIRS = [
    ("法国的首都是", "Paris"), ("日本的首都是", "Tokyo"), ("中国的首都是", "Beijing"),
    ("英国的首都是", "London"), ("德国的首都是", "Berlin"), ("意大利的首都是", "Rome"),
    ("西班牙的首都是", "Madrid"), ("韩国的首都是", "Seoul"), ("印度的首都是", "Delhi"),
    ("巴西的首都是", "Brasilia"), ("加拿大的首都是", "Ottawa"), ("澳大利亚的首都是", "Canberra"),
    ("俄罗斯的首都是", "Moscow"), ("墨西哥的首都是", "Mexico City"), ("埃及的首都是", "Cairo"),
    ("泰国的首都是", "Bangkok"), ("越南的首都是", "Hanoi"), ("土耳其的首都是", "Ankara"),
    ("阿根廷的首都是", "Buenos Aires"), ("荷兰的首都是", "Amsterdam"),
    ("瑞典的首都是", "Stockholm"), ("挪威的首都是", "Oslo"), ("芬兰的首都是", "Helsinki"),
    ("丹麦的首都是", "Copenhagen"), ("波兰的首都是", "Warsaw"),
    ("葡萄牙的首都是", "Lisbon"), ("希腊的首都是", "Athens"), ("瑞士的首都是", "Bern"),
    ("奥地利的首都是", "Vienna"), ("比利时的首都是", "Brussels"),
    ("爱尔兰的首都是", "Dublin"), ("新西兰的首都是", "Wellington"),
    ("菲律宾的首都是", "Manila"), ("马来西亚的首都是", "Kuala Lumpur"),
    ("印度尼西亚的首都是", "Jakarta"), ("新加坡的首都是", "Singapore"),
    ("巴基斯坦的首都是", "Islamabad"), ("孟加拉的首都是", "Dhaka"),
    ("以色列的首都是", "Jerusalem"), ("沙特的首都是", "Riyadh"),
    ("伊朗的首都是", "Tehran"), ("伊拉克的首都是", "Baghdad"),
    ("哥伦比亚的首都是", "Bogota"), ("智利的首都是", "Santiago"),
    ("秘鲁的首都是", "Lima"), ("委内瑞拉的首都是", "Caracas"),
    ("古巴的首都是", "Havana"), ("南非的首都是", "Pretoria"),
    ("尼日利亚的首都是", "Abuja"), ("肯尼亚的首都是", "Nairobi"),
]

ANALOGY_PAIRS = [
    ("king之于queen相当于man之于", "woman"),
    ("big之于bigger相当于small之于", "smaller"),
    ("hot之于cold相当于up之于", "down"),
    ("dog之于puppy相当于cat之于", "kitten"),
    ("run之于ran相当于eat之于", "ate"),
    ("good之于better相当于bad之于", "worse"),
    ("happy之于sad相当于light之于", "dark"),
    ("fast之于faster相当于slow之于", "slower"),
    ("tall之于short相当于wide之于", "narrow"),
    ("young之于old相当于new之于", "old"),
    ("man之于woman相当于boy之于", "girl"),
    ("hand之于finger相当于foot之于", "toe"),
    ("car之于road相当于ship之于", "sea"),
    ("bird之于fly相当于fish之于", "swim"),
    ("sun之于day相当于moon之于", "night"),
    ("pen之于write相当于knife之于", "cut"),
    ("doctor之于hospital相当于teacher之于", "school"),
    ("bread之于baker相当于shoes之于", "cobbler"),
    ("water之于drink相当于food之于", "eat"),
    ("eye之于see相当于ear之于", "hear"),
    ("walk之于legs相当于talk之于", "mouth"),
    ("book之于read相当于song之于", "sing"),
    ("fire之于hot相当于ice之于", "cold"),
    ("winter之于cold相当于summer之于", "hot"),
    ("morning之于breakfast相当于evening之于", "dinner"),
    ("seed之于tree相当于egg之于", "bird"),
    ("question之于answer相当于problem之于", "solution"),
    ("begin之于end相当于start之于", "finish"),
    ("open之于close相当于enter之于", "exit"),
    ("create之于destroy相当于build之于", "destroy"),
    ("love之于hate相当于friend之于", "enemy"),
    ("peace之于war相当于health之于", "disease"),
    ("rich之于poor相当于strong之于", "weak"),
    ("buy之于sell相当于borrow之于", "lend"),
    ("teach之于learn相当于give之于", "receive"),
    ("win之于lose相当于success之于", "failure"),
    ("clean之于dirty相当于safe之于", "dangerous"),
    ("easy之于hard相当于simple之于", "complex"),
    ("loud之于quiet相当于bright之于", "dim"),
    ("sharp之于dull相当于smooth之于", "rough"),
    ("ancient之于modern相当于past之于", "present"),
    ("single之于double相当于half之于", "whole"),
    ("part之于whole相当于piece之于", "entire"),
    ("singular之于plural相当于one之于", "many"),
    ("add之于subtract相当于multiply之于", "divide"),
    ("solid之于liquid相当于liquid之于", "gas"),
    ("north之于south相当于east之于", "west"),
    ("circle之于sphere相当于square之于", "cube"),
    ("line之于plane相当于point之于", "line"),
    ("speed之于distance相当于weight之于", "mass"),
]


def json_serialize(obj):
    """递归转换numpy类型为python原生类型"""
    if isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serialize(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_target_prob(model, tokenizer, device, prompt, target, W_U=None):
    """获取目标token的概率"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # last token

    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        return 0.0

    probs = F.softmax(logits, dim=-1)
    return probs[target_ids[0]].item()


def get_n_heads(model, model_name):
    """获取模型的注意力头数"""
    layers = get_layers(model)
    layer0 = layers[0]
    sa = layer0.self_attn
    # 从q_proj推断head数
    W_q = sa.q_proj.weight
    d_model = W_q.shape[1]
    # head_dim通常是64或128
    # 尝试从config获取
    if hasattr(model.config, 'num_attention_heads'):
        return model.config.num_attention_heads
    # 回退: 从权重推断
    if hasattr(model.config, 'head_dim'):
        return d_model // model.config.head_dim
    # 默认假设head_dim=128
    return d_model // 128


# ============================================================
# Exp 1: Attention Head Ablation
# ============================================================
def exp1_head_ablation(model_name):
    """逐个关闭attention head，测量翻译/检索/类比概率的变化"""
    print(f"\n{'='*60}")
    print(f"Exp 1: Attention Head Ablation — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    n_heads = get_n_heads(model, model_name)
    d_model = info.d_model
    head_dim = d_model // n_heads
    print(f"  模型: {model_name}, 层数: {n_layers}, 头数: {n_heads}, d_model: {d_model}, head_dim: {head_dim}")

    # ---- Step 1: 基线概率 ----
    print("\n[Step 1] 计算基线概率...")

    structures = {
        "translation": TRANSLATION_PAIRS[:30],  # 30个样本
        "retrieval": FACT_RETRIEVAL_PAIRS[:30],
        "analogy": ANALOGY_PAIRS[:20],  # 类比样本少一些
    }

    baseline_probs = {}
    for struct_name, pairs in structures.items():
        probs = []
        for prompt, target in pairs:
            p = get_target_prob(model, tokenizer, device, prompt, target)
            probs.append(p)
        baseline_probs[struct_name] = np.mean(probs)
        print(f"  {struct_name}: baseline prob = {np.mean(probs):.4f}")

    # ---- Step 2: 逐层逐头消融 ----
    print("\n[Step 2] 逐层逐头消融 (这会花较长时间)...")

    # 采样层以节省时间 (关键层+均匀采样)
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        # 前1/4, 中1/2, 后1/4 各采样几层
        q1 = n_layers // 4
        q3 = 3 * n_layers // 4
        sample_layers = sorted(set(
            [0, 1] +
            list(range(q1-1, q1+2)) +
            list(range(n_layers//2-1, n_layers//2+2)) +
            list(range(q3-1, q3+2)) +
            [n_layers-3, n_layers-2, n_layers-1]
        ))
    print(f"  采样层: {sample_layers}")

    # 对每个结构，选5个代表样本
    ablation_results = {struct: {} for struct in structures}

    for struct_name, pairs in structures.items():
        # 选择5个样本
        test_pairs = pairs[:5]
        print(f"\n  === 结构: {struct_name} ===")
        print(f"  测试样本数: {len(test_pairs)}")

        for layer_idx in sample_layers:
            layer = get_layers(model)[layer_idx]
            sa = layer.self_attn

            for head_idx in range(n_heads):
                hook_key = (layer_idx, head_idx)

                # 用pre-hook消融特定head — 修改o_proj的输入(即concatenated head outputs)
                def make_ablation_prehook(hi, hd):
                    def prehook_fn(module, input):
                        # input是tuple, input[0]是concatenated heads [batch, seq, n_heads*head_dim]
                        modified = input[0].clone()
                        start = hi * hd
                        end = (hi + 1) * hd
                        modified[:, :, start:end] = 0.0
                        return (modified,) + input[1:]
                    return prehook_fn

                hook_handle = sa.o_proj.register_forward_pre_hook(
                    make_ablation_prehook(head_idx, head_dim)
                )

                # 测量消融后的概率
                ablated_probs = []
                for prompt, target in test_pairs:
                    p = get_target_prob(model, tokenizer, device, prompt, target)
                    ablated_probs.append(p)

                hook_handle.remove()

                mean_ablated = np.mean(ablated_probs)
                drop = baseline_probs[struct_name] - mean_ablated
                rel_drop = drop / max(baseline_probs[struct_name], 1e-6)

                ablation_results[struct_name][hook_key] = {
                    "baseline": baseline_probs[struct_name],
                    "ablated": mean_ablated,
                    "drop": drop,
                    "rel_drop": rel_drop,
                }

            print(f"    L{layer_idx}: 完成 {n_heads} heads")

    # ---- Step 3: 分析关键head ----
    print("\n[Step 3] 分析关键head...")

    summary = {s: {"critical_heads": [], "top_heads": []} for s in structures}

    for struct_name in structures:
        results = ablation_results[struct_name]
        # 按相对降幅排序
        sorted_heads = sorted(results.items(), key=lambda x: x[1]["rel_drop"], reverse=True)

        # 关键head: 相对降幅 > 10%
        critical = [(k, v) for k, v in sorted_heads if v["rel_drop"] > 0.10]
        # 前10个最敏感head
        top10 = sorted_heads[:10]

        summary[struct_name]["critical_heads"] = [(f"L{k[0]}H{k[1]}", v) for k, v in critical]
        summary[struct_name]["top_heads"] = [(f"L{k[0]}H{k[1]}", v) for k, v in top10]

        print(f"\n  {struct_name} 关键head (rel_drop > 10%):")
        for name, v in critical[:10]:
            print(f"    {name}: drop={v['drop']:.4f} ({v['rel_drop']*100:.1f}%)")

        print(f"  {struct_name} Top-10 最敏感head:")
        for name, v in top10:
            print(f"    {name}: drop={v['drop']:.4f} ({v['rel_drop']*100:.1f}%)")

    # ---- Step 4: 跨结构head重叠分析 ----
    print("\n[Step 4] 跨结构head重叠分析...")

    # 每个结构取top-20最敏感head
    top_heads_per_struct = {}
    for struct_name in structures:
        results = ablation_results[struct_name]
        sorted_heads = sorted(results.items(), key=lambda x: x[1]["rel_drop"], reverse=True)
        top_heads_per_struct[struct_name] = set(k for k, v in sorted_heads[:20])

    # 重叠矩阵
    struct_names = list(structures.keys())
    overlap_matrix = {}
    for i, s1 in enumerate(struct_names):
        for j, s2 in enumerate(struct_names):
            if i < j:
                overlap = len(top_heads_per_struct[s1] & top_heads_per_struct[s2])
                union = len(top_heads_per_struct[s1] | top_heads_per_struct[s2])
                jaccard = overlap / max(union, 1)
                overlap_matrix[f"{s1}_vs_{s2}"] = {
                    "overlap": overlap, "union": union, "jaccard": jaccard
                }
                print(f"  {s1} vs {s2}: overlap={overlap}/20, Jaccard={jaccard:.3f}")

    # ---- 保存结果 ----
    # 序列化ablation_results (tuple key → string key)
    serializable = {}
    for struct_name in structures:
        serializable[struct_name] = {}
        for (li, hi), v in ablation_results[struct_name].items():
            serializable[struct_name][f"L{li}H{hi}"] = v

    output = {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "baseline_probs": baseline_probs,
        "sample_layers": sample_layers,
        "ablation_results": serializable,
        "summary": {
            s: {
                "critical_heads": [(n, v) for n, v in summary[s]["critical_heads"][:20]],
                "top_heads": [(n, v) for n, v in summary[s]["top_heads"]],
            } for s in structures
        },
        "overlap_matrix": overlap_matrix,
    }

    outpath = f"tests/glm5_temp/phase96_exp1_{model_name}_head_ablation.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_serialize(output), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")

    release_model(model)
    return output


# ============================================================
# Exp 2: Residual Stream Patching
# ============================================================
def exp2_residual_patching(model_name):
    """把翻译prompt的层激活注入另一翻译prompt，观察输出变化"""
    print(f"\n{'='*60}")
    print(f"Exp 2: Residual Stream Patching — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  模型: {model_name}, 层数: {n_layers}")

    # ---- Step 1: 生成patching对 ----
    # 用翻译任务: "猫的英文是" → "cat" 和 "狗的英文是" → "dog"
    # patch: 把猫prompt的L层激活注入狗prompt
    # 预期: 如果翻译信息在L层已经编码，注入后"狗"的输出应该偏向"cat"

    patch_pairs = TRANSLATION_PAIRS[:20]  # 20个翻译对

    # 采样层
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        sample_layers = sorted(set(
            [0, 1] +
            list(range(n_layers//4-1, n_layers//4+2)) +
            list(range(n_layers//2-1, n_layers//2+2)) +
            list(range(3*n_layers//4-1, 3*n_layers//4+2)) +
            [n_layers-3, n_layers-2, n_layers-1]
        ))
    print(f"  采样层: {sample_layers}")

    # ---- Step 2: 收集各层hidden states ----
    print("\n[Step 2] 收集各层hidden states...")

    all_hiddens = {}  # {prompt_idx: {layer: hidden_state}}
    all_baseline_probs = {}  # {prompt_idx: {target: prob}}

    for idx, (prompt, target) in enumerate(patch_pairs):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hiddens = {}
        for l in range(n_layers + 1):
            h = outputs.hidden_states[l][0, -1, :].detach().clone()
            hiddens[l] = h

        all_hiddens[idx] = hiddens

        # 基线概率
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if target_ids:
            all_baseline_probs[idx] = probs[target_ids[0]].item()
        else:
            all_baseline_probs[idx] = 0.0

        del outputs
        if idx % 5 == 0:
            print(f"  已收集 {idx+1}/{len(patch_pairs)} prompts")

    # ---- Step 3: 跨prompt patching ----
    print("\n[Step 3] 跨prompt patching...")

    patching_results = []

    # 对每对(i, j)，把i的L层激活注入j
    n_pairs = min(10, len(patch_pairs))  # 10对

    for i in range(n_pairs):
        for j in range(n_pairs):
            if i == j:
                continue

            source_prompt, source_target = patch_pairs[i]
            target_prompt, target_target = patch_pairs[j]

            source_target_ids = tokenizer.encode(source_target, add_special_tokens=False)
            target_target_ids = tokenizer.encode(target_target, add_special_tokens=False)
            if not source_target_ids or not target_target_ids:
                continue

            # 基线: target prompt的概率
            baseline_target_prob = all_baseline_probs[j]

            for layer_l in sample_layers:
                # 用hook在层L注入source的hidden state
                source_h = all_hiddens[i][layer_l]

                def make_patch_hook(src_h):
                    def hook_fn(module, input, output):
                        # 只patch最后一个token位置
                        if isinstance(output, tuple):
                            hidden = output[0]
                        else:
                            hidden = output

                        patched = hidden.clone()
                        patched[0, -1, :] = src_h.to(patched.device).to(patched.dtype)

                        if isinstance(output, tuple):
                            return (patched,) + output[1:]
                        return patched
                    return hook_fn

                # 在layer_l的输出后patch
                # 注意: 需要在layer_l的layernorm之后注入
                # 对Qwen2/Qwen3架构: model.model.layers[l]的输出
                layers = get_layers(model)

                # 找到合适的hook点: layer的forward输出
                hook_handle = layers[layer_l].register_forward_hook(
                    make_patch_hook(source_h)
                )

                # forward with patch
                inputs = tokenizer(target_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits, dim=-1)

                patched_source_prob = probs[source_target_ids[0]].item()
                patched_target_prob = probs[target_target_ids[0]].item()

                hook_handle.remove()
                del outputs

                patching_results.append({
                    "source_idx": i,
                    "target_idx": j,
                    "source": source_prompt,
                    "target": target_prompt,
                    "source_target": source_target,
                    "target_target": target_target,
                    "layer": layer_l,
                    "baseline_target_prob": baseline_target_prob,
                    "patched_source_prob": patched_source_prob,
                    "patched_target_prob": patched_target_prob,
                    "source_leak": patched_source_prob,  # source的target泄漏到target prompt
                    "target_suppress": baseline_target_prob - patched_target_prob,
                })

        print(f"  已完成source {i+1}/{n_pairs}")

    # ---- Step 4: 分析patching效果 ----
    print("\n[Step 4] 分析patching效果...")

    # 按层聚合
    layer_effects = defaultdict(list)
    for r in patching_results:
        layer_effects[r["layer"]].append(r)

    print("\n  各层patching效果:")
    print(f"  {'Layer':>6} | {'Source Leak':>12} | {'Target Suppress':>16} | {'N pairs':>8}")
    print("  " + "-" * 55)

    for layer_l in sorted(layer_effects.keys()):
        results = layer_effects[layer_l]
        mean_leak = np.mean([r["source_leak"] for r in results])
        mean_suppress = np.mean([r["target_suppress"] for r in results])
        print(f"  L{layer_l:>4} | {mean_leak:>12.4f} | {mean_suppress:>16.4f} | {len(results):>8}")

    # 找到信息最集中的层 (source leak最大的层)
    layer_leak_means = {}
    for layer_l, results in layer_effects.items():
        layer_leak_means[layer_l] = np.mean([r["source_leak"] for r in results])

    best_layer = max(layer_leak_means, key=layer_leak_means.get)
    print(f"\n  翻译信息最集中的层: L{best_layer} (mean leak = {layer_leak_means[best_layer]:.4f})")

    # ---- 保存结果 ----
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "sample_layers": sample_layers,
        "n_patch_pairs": n_pairs,
        "layer_effects": {
            str(l): {
                "mean_source_leak": np.mean([r["source_leak"] for r in rs]),
                "mean_target_suppress": np.mean([r["target_suppress"] for r in rs]),
                "n_pairs": len(rs),
            } for l, rs in layer_effects.items()
        },
        "best_leak_layer": best_layer,
        "best_leak_value": layer_leak_means[best_layer],
        "patching_details": patching_results[:100],  # 保存前100条详情
    }

    outpath = f"tests/glm5_temp/phase96_exp2_{model_name}_residual_patching.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_serialize(output), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")

    release_model(model)
    return output


# ============================================================
# Exp 3: 跨结构电路重叠 — 计算原语发现
# ============================================================
def exp3_cross_structure_circuit(model_name):
    """比较翻译/检索/类比各自需要的MLP层，发现共享计算原语"""
    print(f"\n{'='*60}")
    print(f"Exp 3: 跨结构电路重叠 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    n_heads = get_n_heads(model, model_name)
    d_model = info.d_model
    head_dim = d_model // n_heads
    print(f"  模型: {model_name}, 层数: {n_layers}, 头数: {n_heads}")

    structures = {
        "translation": TRANSLATION_PAIRS[:20],
        "retrieval": FACT_RETRIEVAL_PAIRS[:20],
        "analogy": ANALOGY_PAIRS[:15],
    }

    # ---- Step 1: MLP逐层消融 ----
    print("\n[Step 1] MLP逐层消融...")

    # 先计算基线
    baseline_probs = {}
    for struct_name, pairs in structures.items():
        probs = []
        for prompt, target in pairs[:5]:
            p = get_target_prob(model, tokenizer, device, prompt, target)
            probs.append(p)
        baseline_probs[struct_name] = np.mean(probs)
        print(f"  {struct_name} baseline: {baseline_probs[struct_name]:.4f}")

    # 逐层消融MLP
    mlp_ablation = {struct: {} for struct in structures}

    # 采样层
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        sample_layers = sorted(set(
            list(range(0, n_layers, 3)) + [n_layers-1]
        ))
    print(f"  采样层: {sample_layers}")

    for layer_idx in sample_layers:
        layers = get_layers(model)
        mlp = layers[layer_idx].mlp

        # hook: zero out MLP output
        def make_mlp_ablation_hook():
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            return hook_fn

        hook_handle = mlp.register_forward_hook(make_mlp_ablation_hook())

        for struct_name, pairs in structures.items():
            probs = []
            for prompt, target in pairs[:5]:
                p = get_target_prob(model, tokenizer, device, prompt, target)
                probs.append(p)
            mean_p = np.mean(probs)
            drop = baseline_probs[struct_name] - mean_p
            rel_drop = drop / max(baseline_probs[struct_name], 1e-6)
            mlp_ablation[struct_name][layer_idx] = {
                "baseline": baseline_probs[struct_name],
                "ablated": mean_p,
                "drop": drop,
                "rel_drop": rel_drop,
            }

        hook_handle.remove()
        print(f"  L{layer_idx}: " + ", ".join(
            f"{s}={mlp_ablation[s][layer_idx]['rel_drop']*100:.1f}%"
            for s in structures
        ))

    # ---- Step 2: 逐层注意力整体消融 ----
    print("\n[Step 2] 逐层注意力整体消融...")

    attn_ablation = {struct: {} for struct in structures}

    for layer_idx in sample_layers:
        layers = get_layers(model)
        sa = layers[layer_idx].self_attn

        # hook: zero out attention output
        def make_attn_ablation_hook():
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            return hook_fn

        hook_handle = sa.o_proj.register_forward_hook(make_attn_ablation_hook())

        for struct_name, pairs in structures.items():
            probs = []
            for prompt, target in pairs[:5]:
                p = get_target_prob(model, tokenizer, device, prompt, target)
                probs.append(p)
            mean_p = np.mean(probs)
            drop = baseline_probs[struct_name] - mean_p
            rel_drop = drop / max(baseline_probs[struct_name], 1e-6)
            attn_ablation[struct_name][layer_idx] = {
                "baseline": baseline_probs[struct_name],
                "ablated": mean_p,
                "drop": drop,
                "rel_drop": rel_drop,
            }

        hook_handle.remove()
        print(f"  L{layer_idx}: " + ", ".join(
            f"{s}={attn_ablation[s][layer_idx]['rel_drop']*100:.1f}%"
            for s in structures
        ))

    # ---- Step 3: 分析Attn vs MLP贡献比 ----
    print("\n[Step 3] Attn vs MLP贡献比...")

    for struct_name in structures:
        print(f"\n  {struct_name}:")
        print(f"  {'Layer':>6} | {'MLP drop':>10} | {'Attn drop':>10} | {'Attn/MLP':>10}")
        print("  " + "-" * 45)

        for layer_idx in sample_layers:
            mlp_drop = mlp_ablation[struct_name].get(layer_idx, {}).get("rel_drop", 0)
            attn_drop = attn_ablation[struct_name].get(layer_idx, {}).get("rel_drop", 0)
            ratio = attn_drop / max(mlp_drop, 1e-6)
            print(f"  L{layer_idx:>4} | {mlp_drop*100:>9.1f}% | {attn_drop*100:>9.1f}% | {ratio:>10.2f}")

    # ---- Step 4: 发现共享计算原语 ----
    print("\n[Step 4] 发现共享计算原语...")

    # MLP关键层: 降幅 > 5% 的层
    mlp_critical_layers = {}
    attn_critical_layers = {}
    for struct_name in structures:
        mlp_crit = [l for l, v in mlp_ablation[struct_name].items() if v["rel_drop"] > 0.05]
        attn_crit = [l for l, v in attn_ablation[struct_name].items() if v["rel_drop"] > 0.05]
        mlp_critical_layers[struct_name] = set(mlp_crit)
        attn_critical_layers[struct_name] = set(attn_crit)

    # 共享的MLP关键层
    all_mlp_crit = [mlp_critical_layers[s] for s in structures]
    shared_mlp = all_mlp_crit[0]
    for s in all_mlp_crit[1:]:
        shared_mlp = shared_mlp & s

    all_attn_crit = [attn_critical_layers[s] for s in structures]
    shared_attn = all_attn_crit[0]
    for s in all_attn_crit[1:]:
        shared_attn = shared_attn & s

    print(f"\n  各结构的MLP关键层:")
    for s in structures:
        print(f"    {s}: {sorted(mlp_critical_layers[s])}")
    print(f"  共享MLP关键层: {sorted(shared_mlp)}")

    print(f"\n  各结构的Attn关键层:")
    for s in structures:
        print(f"    {s}: {sorted(attn_critical_layers[s])}")
    print(f"  共享Attn关键层: {sorted(shared_attn)}")

    # ---- 保存结果 ----
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "sample_layers": sample_layers,
        "baseline_probs": baseline_probs,
        "mlp_ablation": {s: {str(l): v for l, v in d.items()} for s, d in mlp_ablation.items()},
        "attn_ablation": {s: {str(l): v for l, v in d.items()} for s, d in attn_ablation.items()},
        "mlp_critical_layers": {s: sorted(v) for s, v in mlp_critical_layers.items()},
        "attn_critical_layers": {s: sorted(v) for s, v in attn_critical_layers.items()},
        "shared_mlp_layers": sorted(shared_mlp),
        "shared_attn_layers": sorted(shared_attn),
    }

    outpath = f"tests/glm5_temp/phase96_exp3_{model_name}_cross_structure.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_serialize(output), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")

    release_model(model)
    return output


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                        choices=[1, 2, 3])
    args = parser.parse_args()

    if args.exp == 1:
        exp1_head_ablation(args.model)
    elif args.exp == 2:
        exp2_residual_patching(args.model)
    elif args.exp == 3:
        exp3_cross_structure_circuit(args.model)
