"""
Phase 111: Causal Circuit Extraction — 从统计相关到因果验证
=============================================================

Phase 110的硬伤 (用户批判):
  1. "attention ≠ computation"
     Attention是数据选择(memory fetch), MLP才是非线性变换(computation)
     Phase 110把二者混为一谈, 用attention差异作为"routing topology"
     但真正重要的计算发生在MLP中

  2. "overlap≈0%可能是高维稀疏伪结构"
     9728维取top-1%(97个neuron), 微小噪声就打乱排名
     overlap≈0%可能是统计必然, 不一定意味着"每层用不同neuron"
     需要零假设检验

  3. "缺少因果验证"
     Phase 110只做了统计相关分析, 没有验证功能重要性
     "翻译差分neuron"可能只是probe illusion
     需要扰动(ablation)来验证因果关系

  4. "neuron不是符号特征检测器"
     单个neuron可能没有稳定语义
     需要研究neuron的共激活模式(co-activation), 而非单个neuron

Phase 111核心升级:
  从"统计相关"到"因果验证"
  关键问题: 那些"翻译差分neuron"真的功能重要吗?

关键实验:
  Exp 1: Causal Neuron Ablation — 零化翻译差分neuron, 测量logit变化
    核心: Phase 110发现某些neuron对翻译vs中文有差分激活
    但这些neuron真的功能重要吗? 还是只是统计假象?
    方法: 对每层的top-k翻译差分neuron进行ablation(零化), 测量对输出logit的影响
    对照: 对同数量的随机neuron做ablation, 比较影响差异
    如果翻译差分neuron的ablation影响 >> 随机neuron → 因果验证通过
    如果翻译差分neuron的ablation影响 ≈ 随机neuron → probe illusion

  Exp 2: Null Hypothesis for Sparse Overlap — 验证overlap≈0是否是统计假象
    核心: 9728维的top-1% overlap≈0%可能只是高维稀疏排名的不稳定性
    方法: 用bootstrap重采样, 生成"随机排名"的overlap分布
    对比真实overlap是否显著低于随机排名的overlap
    如果真实overlap ≈ 随机排名overlap → overlap≈0%是统计假象
    如果真实overlap < 随机排名overlap → 真正的计算重编码

  Exp 3: MLP Circuit Tracing — 从MLP输入到输出的因果链
    核心: 只关注MLP(非线性计算), 不关注attention(数据选择)
    追踪: MLP gate activation → down_proj output → residual stream变化 → logit变化
    方法: 在翻译prompt中, 对每层MLP的输出进行分解
    分解: MLP输出 = Σ (gate_i * up_i * down_i) — 每个neuron的贡献
    找出对翻译输出logit贡献最大的neuron

  Exp 4: Neuron Co-activation Graph — 共激活拓扑
    核心: 不看单个neuron, 看neuron之间的共激活关系
    方法: 在翻译vs中文中, 分别计算neuron间的相关矩阵
    比较两种任务下的共激活图结构
    关键: 翻译任务是否形成了不同的"计算子图"?

Run:
  python tests/glm5/ccml_phase111_causal_circuit.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase111_causal_circuit.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase111_causal_circuit.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase111_causal_circuit.py --model qwen3 --exp 4
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U


# ============================================================
# 测试数据 — 扩大到40个词对以增加统计效力
# ============================================================
ANIMAL_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("鱼", "fish"), ("鸟", "bird"),
    ("马", "horse"), ("牛", "cow"), ("羊", "sheep"), ("猪", "pig"),
    ("鸡", "chicken"), ("鸭", "duck"),
]

NATURE_PAIRS = [
    ("水", "water"), ("火", "fire"), ("风", "wind"), ("雨", "rain"),
    ("雪", "snow"), ("冰", "ice"), ("雷", "thunder"), ("雾", "fog"),
    ("霜", "frost"), ("云", "cloud"),
]

OBJECT_PAIRS = [
    ("花", "flower"), ("树", "tree"), ("石", "stone"), ("铁", "iron"),
    ("金", "gold"), ("茶", "tea"), ("沙", "sand"), ("草", "grass"),
    ("血", "blood"), ("光", "light"),
]

CELESTIAL_PAIRS = [
    ("月", "moon"), ("日", "sun"), ("星", "star"), ("河", "river"),
    ("山", "mountain"), ("海", "sea"), ("天", "sky"), ("地", "earth"),
    ("夜", "night"), ("昼", "day"),
]

ALL_PAIRS = ANIMAL_PAIRS + NATURE_PAIRS + OBJECT_PAIRS + CELESTIAL_PAIRS  # 40词对


def get_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


# ============================================================
# 核心工具: 收集MLP gate activation (复用Phase 110)
# ============================================================
def compute_mlp_gate_activations(model, tokenizer, device, prompt, n_layers):
    """收集MLP gate activation (SwiGLU gate_proj → SiLU)"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    layers = get_layers(model)

    gate_activations = {}
    hooks = []

    def make_gate_hook(l):
        def hook_fn(module, input, output):
            gate_act = torch.nn.functional.silu(output)
            gate_activations[l] = gate_act[0, -1, :].detach().float().cpu().numpy()
        return hook_fn

    for l, layer in enumerate(layers):
        if hasattr(layer.mlp, 'gate_proj'):
            h = layer.mlp.gate_proj.register_forward_hook(make_gate_hook(l))
            hooks.append(h)

    with torch.no_grad():
        outputs = model(inputs["input_ids"])

    for h in hooks:
        h.remove()

    del outputs, inputs
    gc.collect()
    torch.cuda.empty_cache()

    return gate_activations


def compute_mlp_down_output(model, tokenizer, device, prompt, n_layers):
    """收集MLP down_proj的输出 (这是写入residual stream的量)"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    layers = get_layers(model)

    mlp_outputs = {}
    hooks = []

    def make_mlp_hook(l):
        def hook_fn(module, input, output):
            # MLP最终输出 = down_proj(SiLU(gate) * up)
            # 取最后一个token
            if isinstance(output, tuple):
                mlp_outputs[l] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                mlp_outputs[l] = output[0, -1, :].detach().float().cpu().numpy()
        return hook_fn

    for l, layer in enumerate(layers):
        h = layer.mlp.register_forward_hook(make_mlp_hook(l))
        hooks.append(h)

    with torch.no_grad():
        outputs = model(inputs["input_ids"])

    for h in hooks:
        h.remove()

    del outputs, inputs
    gc.collect()
    torch.cuda.empty_cache()

    return mlp_outputs


def get_translation_logit(model, tokenizer, device, prompt, target_token):
    """获取翻译目标token的logit值"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    target_id = get_token_id(tokenizer, target_token)
    if target_id is None:
        return None

    with torch.no_grad():
        outputs = model(inputs["input_ids"])
        logits = outputs.logits[0, -1, :]  # 最后一个token的logits
        return logits[target_id].item()


def get_top_logits(model, tokenizer, device, prompt, k=10):
    """获取top-k logit的token和值"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(inputs["input_ids"])
        logits = outputs.logits[0, -1, :]
        top_k = torch.topk(logits, k)
        results = []
        for val, idx in zip(top_k.values, top_k.indices):
            tok = tokenizer.decode([idx.item()])
            results.append((tok, val.item(), idx.item()))
        return results


# ============================================================
# Exp 1: Causal Neuron Ablation — 零化翻译差分neuron的因果测试
# ============================================================
def exp1_causal_ablation(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 1: Causal Neuron Ablation — 因果神经元消融测试")
    print(f"{'='*60}")
    print(f"  核心问题: '翻译差分neuron'真的功能重要吗? 还是probe illusion?")
    print(f"  方法: 零化top-k差分neuron → 测量logit变化 → 对比随机neuron")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    intermediate_size = model_info.intermediate_size
    layers = get_layers(model)

    # ========================================
    # Step 1: 先收集翻译差分neuron排名
    # ========================================
    print(f"\n  Step 1: 收集翻译差分neuron排名...")

    test_pairs = ALL_PAIRS[:30]  # 30个词对, 较大样本量

    zh_gate = defaultdict(list)
    trans_gate = defaultdict(list)

    for i, (zh, en) in enumerate(test_pairs):
        zh_prompt = f"{zh}是一种"
        gate_zh = compute_mlp_gate_activations(model, tokenizer, device, zh_prompt, n_layers)

        trans_prompt = f'"{zh}"的英文是'
        gate_trans = compute_mlp_gate_activations(model, tokenizer, device, trans_prompt, n_layers)

        for l in range(n_layers):
            if l in gate_zh:
                zh_gate[l].append(gate_zh[l])
            if l in gate_trans:
                trans_gate[l].append(gate_trans[l])

        if (i + 1) % 10 == 0:
            print(f"    已处理 {i+1}/{len(test_pairs)} 个词对")

    # 计算每层的差分neuron排名
    diff_rankings = {}
    for l in range(n_layers):
        if l not in zh_gate or l not in trans_gate:
            continue
        zh_data = np.array(zh_gate[l])
        trans_data = np.array(trans_gate[l])
        zh_mean = np.mean(zh_data, axis=0)
        trans_mean = np.mean(trans_data, axis=0)
        diff = trans_mean - zh_mean
        # 按差分绝对值排名 (翻译vs中文的最大差异neuron)
        ranking = np.argsort(np.abs(diff))[::-1]  # 降序
        diff_rankings[l] = ranking

    print(f"    差分neuron排名完成, 共{len(diff_rankings)}层")

    # ========================================
    # Step 2: Causal Ablation — 零化top-k差分neuron
    # ========================================
    print(f"\n  Step 2: Causal Ablation测试...")
    print(f"  对每层, 零化top-k翻译差分neuron, 测量对翻译logit的影响")
    print(f"  对照: 零化同数量随机neuron")

    # 用5个翻译词对做ablation测试 (每个都测量因果效应)
    ablation_pairs = [
        ("猫", "cat"), ("水", "water"), ("花", "flower"),
        ("月", "moon"), ("红", "red"),
    ]

    k_values = [10, 50, 97]  # top-10, top-50, top-1%(97)
    sample_layers = [0, 6, 12, 18, 24, 27, 30, 33, 35]

    results_by_layer = {}

    for l in sample_layers:
        if l not in diff_rankings:
            continue

        ranking = diff_rankings[l]
        layer_result = {"layer": l}

        for k in k_values:
            top_k_neurons = ranking[:k]  # 翻译差分最大的k个neuron

            diff_ablation_impacts = []
            random_ablation_impacts = []
            baseline_logits = []

            for zh, en in ablation_pairs:
                trans_prompt = f'"{zh}"的英文是'
                target_id = get_token_id(tokenizer, en)
                if target_id is None:
                    continue

                # === Baseline logit ===
                inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    base_outputs = model(inputs["input_ids"])
                    base_logits_all = base_outputs.logits[0, -1, :]
                    base_logit = base_logits_all[target_id].item()
                    baseline_logits.append(base_logit)

                # === Ablation: 零化top-k差分neuron ===
                # 在gate_proj的输出上, 将top-k neuron的输出设为0
                ablated_logits_list = []

                def make_ablation_hook(neurons_to_ablate, layer_idx):
                    # 转为tensor避免numpy negative stride问题
                    neuron_idx = torch.tensor(list(neurons_to_ablate), dtype=torch.long, device=device)
                    def hook_fn(module, input, output):
                        # output: gate_proj的输出, shape [batch, seq, intermediate]
                        output_modified = output.clone()
                        # 只修改最后一个token
                        output_modified[0, -1, neuron_idx] = 0.0
                        return output_modified
                    return hook_fn

                hook = layers[l].mlp.gate_proj.register_forward_hook(
                    make_ablation_hook(top_k_neurons, l)
                )

                with torch.no_grad():
                    abl_outputs = model(inputs["input_ids"])
                    abl_logits_all = abl_outputs.logits[0, -1, :]
                    abl_logit = abl_logits_all[target_id].item()
                    ablated_logits_list.append(abl_logit)

                hook.remove()

                # Ablation影响 = |baseline - ablated|
                diff_impact = abs(base_logit - abl_logit)
                diff_ablation_impacts.append(diff_impact)

                # === Ablation: 零化k个随机neuron (3次平均) ===
                random_impacts = []
                for _ in range(3):  # 3次随机, 减少方差
                    random_neurons = np.random.choice(intermediate_size, k, replace=False)

                    rand_idx = torch.tensor(list(random_neurons), dtype=torch.long, device=device)
                    def make_random_ablation_hook(neuron_idx):
                        def hook_fn(module, input, output):
                            output_modified = output.clone()
                            output_modified[0, -1, neuron_idx] = 0.0
                            return output_modified
                        return hook_fn

                    hook_rand = layers[l].mlp.gate_proj.register_forward_hook(
                        make_random_ablation_hook(rand_idx)
                    )

                    with torch.no_grad():
                        rand_outputs = model(inputs["input_ids"])
                        rand_logits_all = rand_outputs.logits[0, -1, :]
                        rand_logit = rand_logits_all[target_id].item()

                    hook_rand.remove()

                    random_impacts.append(abs(base_logit - rand_logit))

                random_ablation_impacts.append(np.mean(random_impacts))

            # 统计: 差分neuron ablation vs 随机neuron ablation
            if diff_ablation_impacts and random_ablation_impacts:
                mean_diff_impact = np.mean(diff_ablation_impacts)
                mean_random_impact = np.mean(random_ablation_impacts)
                ratio = mean_diff_impact / max(mean_random_impact, 1e-10)

                layer_result[f"top{k}_diff_impact"] = float(mean_diff_impact)
                layer_result[f"top{k}_random_impact"] = float(mean_random_impact)
                layer_result[f"top{k}_causal_ratio"] = float(ratio)

                # 统计显著性: 简单的配对t检验
                from scipy import stats as scipy_stats
                if len(diff_ablation_impacts) >= 3:
                    t_stat, p_value = scipy_stats.ttest_rel(diff_ablation_impacts, random_ablation_impacts)
                    layer_result[f"top{k}_p_value"] = float(p_value)
                    layer_result[f"top{k}_t_stat"] = float(t_stat)

                sig = "***" if ratio > 3 else "**" if ratio > 2 else "*" if ratio > 1.5 else "ns"
                print(f"    L{l}: top-{k} → diff_impact={mean_diff_impact:.4f}, "
                      f"random_impact={mean_random_impact:.4f}, "
                      f"ratio={ratio:.2f}{sig}")

        results_by_layer[l] = layer_result

    # ========================================
    # Step 3: 汇总 — 哪些层的差分neuron是因果重要的?
    # ========================================
    print(f"\n  === Causal Ablation汇总 ===")
    print(f"  causal_ratio > 2 = 差分neuron的ablation影响是随机的2倍以上")
    print(f"  p < 0.05 = 统计显著")

    for l in sample_layers:
        if l not in results_by_layer:
            continue
        r = results_by_layer[l]
        ratios = []
        for k in k_values:
            key = f"top{k}_causal_ratio"
            if key in r:
                ratios.append(f"top{k}={r[key]:.2f}")
        p_vals = []
        for k in k_values:
            key = f"top{k}_p_value"
            if key in r:
                p_vals.append(f"p={r[key]:.3f}")
        print(f"    L{l}: {', '.join(ratios)} | {', '.join(p_vals)}")

    results = {
        "ablation_pairs": [(zh, en) for zh, en in ablation_pairs],
        "k_values": k_values,
        "sample_layers": sample_layers,
        "results_by_layer": {str(k): v for k, v in results_by_layer.items()},
    }

    out_path = f"tests/glm5_temp/phase111_exp1_{model_name}_causal_ablation.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 2: Null Hypothesis for Sparse Overlap — 验证overlap≈0
# ============================================================
def exp2_null_overlap(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 2: Null Hypothesis for Sparse Overlap — 验证overlap≈0是否是统计假象")
    print(f"{'='*60}")
    print(f"  核心问题: 9728维取top-1%, overlap≈0%是真实的还是高维稀疏噪声?")
    print(f"  方法: bootstrap重采样, 比较真实overlap与随机排名overlap")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    intermediate_size = model_info.intermediate_size
    layers = get_layers(model)

    # ========================================
    # Step 1: 收集gate activations
    # ========================================
    print(f"\n  Step 1: 收集gate activations (40个词对)...")

    zh_gate = defaultdict(list)
    trans_gate = defaultdict(list)

    for i, (zh, en) in enumerate(ALL_PAIRS):
        zh_prompt = f"{zh}是一种"
        gate_zh = compute_mlp_gate_activations(model, tokenizer, device, zh_prompt, n_layers)

        trans_prompt = f'"{zh}"的英文是'
        gate_trans = compute_mlp_gate_activations(model, tokenizer, device, trans_prompt, n_layers)

        for l in range(n_layers):
            if l in gate_zh:
                zh_gate[l].append(gate_zh[l])
            if l in gate_trans:
                trans_gate[l].append(gate_trans[l])

        if (i + 1) % 10 == 0:
            print(f"    已处理 {i+1}/{len(ALL_PAIRS)} 个词对")

    # ========================================
    # Step 2: 真实overlap
    # ========================================
    print(f"\n  Step 2: 计算真实overlap...")

    top_pcts = [0.01, 0.02, 0.05]  # top-1%, top-2%, top-5%

    for top_pct in top_pcts:
        top_n = max(1, int(intermediate_size * top_pct))

        print(f"\n  --- top-{top_pct*100:.0f}% (n={top_n}) ---")

        # 计算每层的翻译差分neuron排名
        layer_rankings = {}
        for l in range(n_layers):
            if l not in zh_gate or l not in trans_gate:
                continue
            zh_data = np.array(zh_gate[l])
            trans_data = np.array(trans_gate[l])
            zh_mean = np.mean(zh_data, axis=0)
            trans_mean = np.mean(trans_data, axis=0)
            diff = trans_mean - zh_mean
            ranking = np.argsort(np.abs(diff))[::-1]
            layer_rankings[l] = set(ranking[:top_n])

        # 真实overlap (相邻层)
        print(f"  真实overlap (相邻层):")
        real_overlaps = {}
        for l in range(0, n_layers - 1):
            if l in layer_rankings and l + 1 in layer_rankings:
                overlap = len(layer_rankings[l] & layer_rankings[l + 1]) / top_n
                real_overlaps[l] = overlap
                if l % 6 == 0 or l >= n_layers - 3:
                    print(f"    L{l}→L{l+1}: overlap={overlap:.4f}")

        # ========================================
        # Step 3: Null hypothesis — 随机排名的overlap
        # ========================================
        print(f"\n  Null hypothesis: bootstrap随机排名的overlap...")

        n_bootstrap = 100
        null_overlaps = defaultdict(list)

        for b in range(n_bootstrap):
            # 随机生成每层的"排名" (随机排列)
            random_rankings = {}
            for l in range(n_layers):
                random_rankings[l] = set(np.random.choice(intermediate_size, top_n, replace=False))

            for l in range(0, n_layers - 1):
                overlap = len(random_rankings[l] & random_rankings[l + 1]) / top_n
                null_overlaps[l].append(overlap)

        # 对比真实 vs null
        print(f"\n  真实overlap vs null hypothesis overlap:")
        print(f"  如果真实 << null → 真正的计算重编码 (neuron被重新分配)")
        print(f"  如果真实 ≈ null → overlap≈0只是高维稀疏的统计必然")

        significant_layers = []
        for l in sorted(real_overlaps.keys()):
            if l in null_overlaps:
                null_mean = np.mean(null_overlaps[l])
                null_std = np.std(null_overlaps[l])
                real_val = real_overlaps[l]
                z_score = (real_val - null_mean) / max(null_std, 1e-10)

                # 真实值是否在null分布的哪个分位?
                percentile = np.mean(np.array(null_overlaps[l]) <= real_val) * 100

                if l % 3 == 0 or l >= n_layers - 3 or abs(z_score) > 2:
                    direction = "↓" if real_val < null_mean else "↑"
                    print(f"    L{l}→L{l+1}: real={real_val:.4f}, null_mean={null_mean:.4f}, "
                          f"z={z_score:+.2f}{direction}, percentile={percentile:.1f}%")

                if abs(z_score) > 2:
                    significant_layers.append((l, z_score, real_val, null_mean))

        if significant_layers:
            print(f"\n  显著偏离null的层 (|z|>2):")
            for l, z, real, null_m in significant_layers:
                direction = "低于" if z < 0 else "高于"
                print(f"    L{l}→L{l+1}: z={z:+.2f} (真实overlap{direction}随机期望)")
        else:
            print(f"\n  ❌ 没有层显著偏离null! overlap≈0%可能只是高维稀疏噪声!")

    # ========================================
    # Step 4: 理论期望 — 纯随机的overlap应该是多少?
    # ========================================
    print(f"\n  === 理论期望 vs 实际 ===")
    for top_pct in top_pcts:
        top_n = max(1, int(intermediate_size * top_pct))
        # 两次独立随机选top_n个, 期望overlap = top_n * (top_n / intermediate_size)
        expected_overlap = top_n * top_n / intermediate_size
        expected_pct = expected_overlap / top_n
        print(f"    top-{top_pct*100:.0f}% (n={top_n}): 期望随机overlap = {expected_pct:.4f} "
              f"({expected_overlap:.1f}/{top_n} neurons)")

    release_model(model)
    return {"status": "complete"}


# ============================================================
# Exp 3: MLP Circuit Tracing — MLP输出对logit的因果贡献
# ============================================================
def exp3_mlp_circuit_tracing(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 3: MLP Circuit Tracing — MLP→logit因果链追踪")
    print(f"{'='*60}")
    print(f"  核心: 只关注MLP(非线性计算), 追踪neuron→residual→logit的因果链")
    print(f"  方法: 分解MLP输出为各neuron的贡献, 计算每个neuron对目标logit的影响")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    intermediate_size = model_info.intermediate_size
    layers = get_layers(model)

    # ========================================
    # Step 1: 收集MLP down_proj权重
    # ========================================
    print(f"\n  Step 1: 提取MLP down_proj权重...")
    print(f"  W_down shape: [{d_model}, {intermediate_size}]")
    print(f"  每个neuron i 对residual stream的贡献 = W_down[:, i] * activation_i")

    W_down_by_layer = {}
    for l, layer in enumerate(layers):
        if hasattr(layer.mlp, 'down_proj'):
            W_down = layer.mlp.down_proj.weight.detach().float().cpu().numpy()  # [d_model, intermediate]
            W_down_by_layer[l] = W_down

    # W_U (lm_head权重)
    W_U = get_W_U(model)  # [vocab_size, d_model]

    # ========================================
    # Step 2: 对每个翻译词对, 分解MLP输出对logit的贡献
    # ========================================
    print(f"\n  Step 2: 分解MLP输出对翻译logit的neuron级贡献...")

    test_pairs = [("猫", "cat"), ("水", "water"), ("月", "moon"), ("火", "fire"), ("红", "red")]
    sample_layers = [0, 6, 12, 18, 24, 27, 30, 33, 35]

    results_by_pair = {}

    for zh, en in test_pairs:
        print(f"\n  --- {zh} → {en} ---")

        trans_prompt = f'"{zh}"的英文是'
        zh_prompt = f"{zh}是一种"
        target_id = get_token_id(tokenizer, en)
        if target_id is None:
            print(f"    无法找到token: {en}")
            continue

        # 获取W_U的target行 (d_model维)
        w_target = W_U[target_id]  # [d_model]

        # 收集gate activation
        gate_trans = compute_mlp_gate_activations(model, tokenizer, device, trans_prompt, n_layers)
        gate_zh = compute_mlp_gate_activations(model, tokenizer, device, zh_prompt, n_layers)

        # 收集up_proj activation (需要在up_proj后加hook)
        def compute_up_activations_for_prompt(prompt):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            up_activations = {}
            hooks = []

            def make_up_hook(l):
                def hook_fn(module, input, output):
                    up_activations[l] = output[0, -1, :].detach().float().cpu().numpy()
                return hook_fn

            for l, layer in enumerate(layers):
                if hasattr(layer.mlp, 'up_proj'):
                    h = layer.mlp.up_proj.register_forward_hook(make_up_hook(l))
                    hooks.append(h)

            with torch.no_grad():
                _ = model(inputs["input_ids"])

            for h in hooks:
                h.remove()

            return up_activations

        up_trans = compute_up_activations_for_prompt(trans_prompt)
        up_zh = compute_up_activations_for_prompt(zh_prompt)

        pair_results = {}

        for l in sample_layers:
            if l not in gate_trans or l not in gate_zh:
                continue
            if l not in up_trans or l not in up_zh:
                continue
            if l not in W_down_by_layer:
                continue

            W_down = W_down_by_layer[l]  # [d_model, intermediate]

            # MLP输出 = W_down @ (gate * up)  (每个neuron的贡献)
            # gate * up: [intermediate]
            # 每个neuron i 的贡献 = W_down[:, i] * (gate[i] * up[i])
            # 对target logit的贡献 = w_target @ W_down[:, i] * (gate[i] * up[i])

            # 预计算: w_target @ W_down = [intermediate] — 每个neuron对target logit的"路由权重"
            neuron_logit_weight = w_target @ W_down  # [intermediate]

            # 翻译prompt中每个neuron对target logit的贡献
            gate_up_trans = gate_trans[l] * up_trans[l]  # [intermediate]
            neuron_contribution_trans = neuron_logit_weight * gate_up_trans  # [intermediate]

            # 中文prompt
            gate_up_zh = gate_zh[l] * up_zh[l]
            neuron_contribution_zh = neuron_logit_weight * gate_up_zh

            # 差分贡献 (翻译 - 中文)
            diff_contribution = neuron_contribution_trans - neuron_contribution_zh

            # 排名: 对翻译logit贡献最大的neuron
            top_contrib_idx = np.argsort(neuron_contribution_trans)[::-1]
            top_diff_idx = np.argsort(np.abs(diff_contribution))[::-1]

            # 统计
            total_trans_contrib = np.sum(neuron_contribution_trans)
            total_zh_contrib = np.sum(neuron_contribution_zh)
            total_diff = np.sum(diff_contribution)

            # top-10 neuron的贡献占比
            top10_trans_contrib = np.sum(neuron_contribution_trans[top_contrib_idx[:10]])
            top10_trans_pct = top10_trans_contrib / max(abs(total_trans_contrib), 1e-10)

            # top-10差分neuron的差分贡献占比
            top10_diff_sum = np.sum(np.abs(diff_contribution[top_diff_idx[:10]]))
            total_diff_abs = np.sum(np.abs(diff_contribution))
            top10_diff_pct = top10_diff_sum / max(total_diff_abs, 1e-10)

            # 集中度: 少数neuron是否贡献了大部分差分?
            # Gini系数
            sorted_abs = np.sort(np.abs(diff_contribution))
            n = len(sorted_abs)
            cumsum = np.cumsum(sorted_abs)
            gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_abs)) / (n * cumsum[-1]) - (n + 1) / n if cumsum[-1] > 0 else 0

            pair_results[l] = {
                "total_trans_contrib": float(total_trans_contrib),
                "total_zh_contrib": float(total_zh_contrib),
                "total_diff": float(total_diff),
                "top10_trans_pct": float(top10_trans_pct),
                "top10_diff_pct": float(top10_diff_pct),
                "gini": float(gini),
                "top5_diff_neurons": top_diff_idx[:5].tolist(),
                "top5_diff_values": diff_contribution[top_diff_idx[:5]].tolist(),
            }

            if l % 6 == 0 or l >= n_layers - 3:
                print(f"    L{l}: total_diff={total_diff:.4f}, top10_diff_pct={top10_diff_pct:.2%}, "
                      f"gini={gini:.4f}, top5_neuron_ids={top_diff_idx[:5].tolist()}")

        results_by_pair[f"{zh}→{en}"] = pair_results

    # ========================================
    # Step 3: 跨词对的一致性 — 同一neuron是否在不同词对中都重要?
    # ========================================
    print(f"\n  === 跨词对的一致性 ===")
    print(f"  如果某neuron在多个词对的翻译中都贡献大 → 可能是通用翻译计算neuron")
    print(f"  如果不同词对用不同neuron → 支持分布式编码")

    for l in sample_layers:
        # 收集所有词对在该层的top-10差分neuron
        all_top_neurons = []
        for pair_name, pair_res in results_by_pair.items():
            if l in pair_res:
                top5 = pair_res[l]["top5_diff_neurons"]
                all_top_neurons.extend(top5)

        if all_top_neurons:
            # 计算跨词对的overlap
            from collections import Counter
            neuron_counts = Counter(all_top_neurons)
            # 出现在2+个词对中的neuron
            shared_neurons = sum(1 for c in neuron_counts.values() if c >= 2)
            total_unique = len(neuron_counts)
            print(f"    L{l}: {total_unique} unique neurons in top-5 across pairs, "
                  f"{shared_neurons} shared (≥2 pairs)")

    results = {
        "test_pairs": [(zh, en) for zh, en in test_pairs],
        "sample_layers": sample_layers,
        "results_by_pair": {k: {str(lk): lv for lk, lv in v.items()} for k, v in results_by_pair.items()},
    }

    out_path = f"tests/glm5_temp/phase111_exp3_{model_name}_mlp_circuit.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 4: Neuron Co-activation Graph — 共激活拓扑
# ============================================================
def exp4_coactivation_graph(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 4: Neuron Co-activation Graph — 神经元共激活拓扑")
    print(f"{'='*60}")
    print(f"  核心: 不看单个neuron, 看neuron之间的共激活关系")
    print(f"  翻译任务是否形成了不同的'计算子图'?")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    intermediate_size = model_info.intermediate_size
    layers = get_layers(model)

    # ========================================
    # Step 1: 收集gate activations (40个词对)
    # ========================================
    print(f"\n  Step 1: 收集gate activations...")

    zh_gate = defaultdict(list)
    trans_gate = defaultdict(list)

    for i, (zh, en) in enumerate(ALL_PAIRS):
        zh_prompt = f"{zh}是一种"
        gate_zh = compute_mlp_gate_activations(model, tokenizer, device, zh_prompt, n_layers)

        trans_prompt = f'"{zh}"的英文是'
        gate_trans = compute_mlp_gate_activations(model, tokenizer, device, trans_prompt, n_layers)

        for l in range(n_layers):
            if l in gate_zh:
                zh_gate[l].append(gate_zh[l])
            if l in gate_trans:
                trans_gate[l].append(gate_trans[l])

        if (i + 1) % 10 == 0:
            print(f"    已处理 {i+1}/{len(ALL_PAIRS)} 个词对")

    # ========================================
    # Step 2: 在每层, 只取active neuron做共激活分析
    # ========================================
    print(f"\n  Step 2: 构建neuron共激活图...")

    sample_layers = [0, 6, 12, 18, 24, 27, 30, 33, 35]
    # 只取激活值>0.1的neuron (active), 减少维度
    activation_threshold = 0.1

    results_by_layer = {}

    for l in sample_layers:
        if l not in zh_gate or l not in trans_gate:
            continue

        zh_data = np.array(zh_gate[l])    # (n_samples, intermediate)
        trans_data = np.array(trans_gate[l])

        # 找到至少在一个样本中激活的neuron
        zh_active_mask = np.any(zh_data > activation_threshold, axis=0)
        trans_active_mask = np.any(trans_data > activation_threshold, axis=0)
        active_mask = zh_active_mask | trans_active_mask
        active_indices = np.where(active_mask)[0]

        n_active = len(active_indices)
        print(f"\n    L{l}: {n_active} active neurons (of {intermediate_size})")

        if n_active < 10 or n_active > 500:
            # 太少或太多都不适合做共激活分析
            # 如果太多, 只取top激活的
            if n_active > 500:
                # 按平均激活值排序, 取top-500
                all_mean = np.mean(np.concatenate([zh_data, trans_data], axis=0), axis=0)
                active_indices = np.argsort(all_mean)[::-1][:500]
                n_active = 500
                print(f"      截取top-500活跃neuron")

        # 提取active neuron的激活值
        zh_active_data = zh_data[:, active_indices]  # (n_samples, n_active)
        trans_active_data = trans_data[:, active_indices]

        # ========================================
        # Step 3: 计算共激活相关矩阵
        # ========================================
        # 对于每个样本, 看neuron之间是否共激活
        # 用Pearson相关系数

        # 翻译prompt的共激活
        if zh_active_data.shape[0] >= 3 and trans_active_data.shape[0] >= 3:
            zh_corr = np.corrcoef(zh_active_data.T)   # (n_active, n_active)
            trans_corr = np.corrcoef(trans_active_data.T)

            # 只取上三角 (排除对角线)
            mask_upper = np.triu(np.ones((n_active, n_active), dtype=bool), k=1)

            zh_corr_vals = zh_corr[mask_upper]
            trans_corr_vals = trans_corr[mask_upper]

            # 去掉NaN
            zh_corr_vals = zh_corr_vals[~np.isnan(zh_corr_vals)]
            trans_corr_vals = trans_corr_vals[~np.isnan(trans_corr_vals)]

            # 统计共激活强度
            zh_mean_corr = np.mean(np.abs(zh_corr_vals))
            trans_mean_corr = np.mean(np.abs(trans_corr_vals))

            # 高共激活对 (|corr| > 0.5)
            zh_strong = np.sum(np.abs(zh_corr_vals) > 0.5)
            trans_strong = np.sum(np.abs(trans_corr_vals) > 0.5)
            total_pairs = len(zh_corr_vals)

            # 正相关 vs 负相关 (共激活 vs 互斥)
            zh_positive = np.sum(zh_corr_vals > 0.3)
            zh_negative = np.sum(zh_corr_vals < -0.3)
            trans_positive = np.sum(trans_corr_vals > 0.3)
            trans_negative = np.sum(trans_corr_vals < -0.3)

            # 差分: 翻译vs中文的共激活模式差异
            # 计算相关矩阵的差异
            if zh_corr.shape == trans_corr.shape:
                corr_diff = trans_corr - zh_corr
                diff_vals = corr_diff[mask_upper]
                diff_vals = diff_vals[~np.isnan(diff_vals)]

                # 差分最大的neuron对
                top_diff_pairs = np.argsort(np.abs(diff_vals))[::-1][:10]

                # 差分共激活的统计
                mean_diff = np.mean(np.abs(diff_vals))
                max_diff = np.max(np.abs(diff_vals)) if len(diff_vals) > 0 else 0
            else:
                mean_diff = 0
                max_diff = 0
                top_diff_pairs = []

            layer_result = {
                "n_active": int(n_active),
                "zh_mean_abs_corr": float(zh_mean_corr),
                "trans_mean_abs_corr": float(trans_mean_corr),
                "zh_strong_pairs": int(zh_strong),
                "trans_strong_pairs": int(trans_strong),
                "total_pairs": int(total_pairs),
                "zh_positive_corr": int(zh_positive),
                "zh_negative_corr": int(zh_negative),
                "trans_positive_corr": int(trans_positive),
                "trans_negative_corr": int(trans_negative),
                "mean_corr_diff": float(mean_diff),
                "max_corr_diff": float(max_diff),
            }
            results_by_layer[l] = layer_result

            print(f"      zh: mean|corr|={zh_mean_corr:.4f}, strong={zh_strong}/{total_pairs} "
                  f"(pos={zh_positive}, neg={zh_negative})")
            print(f"      trans: mean|corr|={trans_mean_corr:.4f}, strong={trans_strong}/{total_pairs} "
                  f"(pos={trans_positive}, neg={trans_negative})")
            print(f"      diff: mean|Δcorr|={mean_diff:.4f}, max|Δcorr|={max_diff:.4f}")

            # ========================================
            # Step 4: 图拓扑指标 — 翻译vs中文
            # ========================================
            # 构建邻接矩阵 (只保留|corr|>0.3的边)
            corr_threshold = 0.3

            zh_adj = (np.abs(zh_corr) > corr_threshold).astype(float)
            np.fill_diagonal(zh_adj, 0)
            trans_adj = (np.abs(trans_corr) > corr_threshold).astype(float)
            np.fill_diagonal(trans_adj, 0)

            # 图的度分布
            zh_degrees = np.sum(zh_adj, axis=1)
            trans_degrees = np.sum(trans_adj, axis=1)

            # Hub neurons (度最高的neuron)
            zh_hub_idx = np.argsort(zh_degrees)[::-1][:5]
            trans_hub_idx = np.argsort(trans_degrees)[::-1][:5]

            zh_hub_neurons = active_indices[zh_hub_idx].tolist()
            trans_hub_neurons = active_indices[trans_hub_idx].tolist()

            # Hub overlap: 翻译和中文的hub neuron是否相同?
            hub_overlap = len(set(zh_hub_idx) & set(trans_hub_idx))
            print(f"      Hub overlap: {hub_overlap}/5 (翻译和中文共享的hub neuron数)")

            # 连通分量数 (简单估计)
            def count_components(adj, n):
                visited = set()
                components = 0
                for start in range(n):
                    if start in visited:
                        continue
                    components += 1
                    stack = [start]
                    while stack:
                        node = stack.pop()
                        if node in visited:
                            continue
                        visited.add(node)
                        neighbors = np.where(adj[node] > 0)[0]
                        stack.extend(neighbors)
                return components

            zh_components = count_components(zh_adj, n_active)
            trans_components = count_components(trans_adj, n_active)

            print(f"      图连通分量: zh={zh_components}, trans={trans_components}")
            print(f"      Hub neurons (zh): {zh_hub_neurons[:3]}")
            print(f"      Hub neurons (trans): {trans_hub_neurons[:3]}")

            layer_result["zh_hub_neurons"] = zh_hub_neurons
            layer_result["trans_hub_neurons"] = trans_hub_neurons
            layer_result["hub_overlap"] = int(hub_overlap)
            layer_result["zh_components"] = int(zh_components)
            layer_result["trans_components"] = int(trans_components)
            layer_result["zh_mean_degree"] = float(np.mean(zh_degrees))
            layer_result["trans_mean_degree"] = float(np.mean(trans_degrees))

    results = {
        "sample_layers": sample_layers,
        "activation_threshold": activation_threshold,
        "results_by_layer": {str(k): v for k, v in results_by_layer.items()},
    }

    out_path = f"tests/glm5_temp/phase111_exp4_{model_name}_coactivation_graph.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3")
    parser.add_argument("--exp", type=int, default=1)
    args = parser.parse_args()

    if args.exp == 1:
        exp1_causal_ablation(args)
    elif args.exp == 2:
        exp2_null_overlap(args)
    elif args.exp == 3:
        exp3_mlp_circuit_tracing(args)
    elif args.exp == 4:
        exp4_coactivation_graph(args)
    else:
        print(f"Unknown exp: {args.exp}")
