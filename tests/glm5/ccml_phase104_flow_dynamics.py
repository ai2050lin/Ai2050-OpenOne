"""
Phase 104: 严格流动力学分析 — 从隐喻到数学
=============================================

Phase 103的硬伤 (用户批判):
  1. "过度离散化电路" — 不同prompt格式不是独立电路，而是同一连续动力系统的不同轨道区域
     错误: "3个独立翻译电路" → 正确: 同一流场的不同basin/trajectory region
  2. "最终LN把44维压到1维" — LayerNorm只做recenter+rescale，不消灭维度
     L35的rank=1是decoder bottleneck(W_unembed只读少数方向)，不是LN collapse
  3. "Jacobian仍然太粗糙" — 50个各向同性随机探针淹没稀疏结构
     需要: token-conditioned Jacobian, attention-mediated Jacobian
  4. "translation signal"误导 — 翻译不是feature，而是constrained decoding process
     错误: "翻译状态"→"翻译特征" → 正确: 不断约束allowable token manifolds的过程

核心方法论升级:
  从"隐喻式动力系统语言"到"严格动力学机制可解释性"
  关键: 建立"什么叫动力学机制"的严格定义

严格化的核心量:
  1. 局部Lyapunov指数 — 哪些层对微扰极敏感
  2. 条件化Jacobian谱 — 不同prompt下的真实动力学差异
  3. 轨迹束分析 — 多个相似prompt的轨迹收敛/分叉
  4. 最小控制能量 — 把系统从中文attractor推到英文attractor的最小扰动
  5. 特征值跃迁 — 不看rank，看特征值是否越过|λ|=1

实验设计:
  Exp 1: 条件化Jacobian — 不同prompt格式的Jacobian谱结构差异
    不用随机探针！用来自实际hidden state的有结构探针
    对比: 翻译prompt / 中文续写 / 英文续写 的Jacobian

  Exp 2: 局部Lyapunov分析 — 层间稳定性的精确测量
    对每层，计算最大Lyapunov指数(最大奇异值-1)
    对比不同上下文下的Lyapunov谱
    寻找Lyapunov spike → 路由相变的标志

  Exp 3: 轨迹束分析 — 从离散电路到连续流场
    不再离散化电路！
    构造prompt连续插值族(20步)，测量hidden state轨迹
    分析轨迹束的: 收敛、分叉、曲率
    看是否是连续变形而非离散切换

  Exp 4: 最小控制能量 — 翻译的真正代价
    不问"翻译方向存在吗"(错误问题)
    问: 多小的扰动能把中文→英文attractor?
    在不同层、不同位置、不同方向上搜索最小控制能量
    这才是"翻译的数学本质"

  Exp 5: 解码瓶颈分析 — 区分LN效果和W_unembed效果
    分离: LayerNorm对hidden state的变换 vs W_unembed的投影
    看L35的rank=1到底是LN造成的还是unembedding造成的

Run:
  python tests/glm5/ccml_phase104_flow_dynamics.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase104_flow_dynamics.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase104_flow_dynamics.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase104_flow_dynamics.py --model qwen3 --exp 4
  python tests/glm5/ccml_phase104_flow_dynamics.py --model qwen3 --exp 5
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

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U


# ============================================================
# 测试数据
# ============================================================
TRANSLATION_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("水", "water"), ("火", "fire"),
    ("树", "tree"), ("花", "flower"), ("鱼", "fish"), ("鸟", "bird"),
    ("铁", "iron"), ("茶", "tea"), ("马", "horse"), ("金", "gold"),
    ("雪", "snow"), ("月", "moon"), ("光", "light"), ("梦", "dream"),
    ("龙", "dragon"), ("云", "cloud"), ("冰", "ice"), ("星", "star"),
]


def get_token_prob_from_logits(logits, tokenizer, text):
    tok_ids = tokenizer.encode(text, add_special_tokens=False)
    if not tok_ids:
        return 0.0
    probs = torch.softmax(logits, dim=-1)
    return probs[tok_ids[0]].item()


def get_top_k_from_logits(logits, tokenizer, k=10):
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, k)
    results = []
    for i in range(k):
        tok_id = topk.indices[i].item()
        prob = topk.values[i].item()
        tok_str = tokenizer.decode([tok_id])
        results.append({"token": tok_str, "token_id": tok_id, "prob": prob})
    return results


# ============================================================
# Exp 1: 条件化Jacobian谱 — 用有结构探针替代随机探针
# ============================================================
def exp1_conditioned_jacobian(model_name):
    """
    条件化Jacobian谱分析

    Phase 103的问题: 用50个各向同性随机探针，淹没稀疏结构

    本实验改进:
    - 探针来自实际hidden state的差分方向
    - 对比3种上下文的Jacobian:
      (a) 中文续写 "X是一种"
      (b) 翻译(简洁) "X的英文是"
      (c) 翻译(指令) "请把X翻译成英文："

    关键问题: 这些上下文的Jacobian谱是"离散切换"还是"连续变形"?
    - 如果离散切换 → 不同子电路
    - 如果连续变形 → 同一流场的不同区域
    """
    print(f"\n{'='*70}")
    print(f"Exp 1: 条件化Jacobian谱 — {model_name}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    results = {}

    test_pairs = [("猫", "cat"), ("水", "water"), ("火", "fire"), ("树", "tree"),
                  ("龙", "dragon"), ("光", "light"), ("冰", "ice"), ("梦", "dream")]

    # 3种上下文模板
    context_templates = {
        "zh_continue": lambda zh, en: f"{zh}是一种",
        "trans_short": lambda zh, en: f"{zh}的英文是",
        "trans_instruction": lambda zh, en: f'请把"{zh}"翻译成英文：',
    }

    # 采样层
    sample_layers = [0, 6, 12, 18, 21, 24, 27, 30, 33]
    sample_layers = [l for l in sample_layers if l < n_layers]

    n_probes = 30  # 减少探针数，但用有结构探针
    eps_base = 1.0

    jacobian_results = {}

    for zh, en in test_pairs:
        print(f"\n  === 处理: {zh}({en}) ===")
        word_jacobian = {}

        # 先收集所有上下文的hidden states，用于构建有结构探针
        all_hiddens = {}
        for ctx_name, template in context_templates.items():
            prompt = template(zh, en)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)
            all_hiddens[ctx_name] = {}
            for l in range(n_layers + 1):
                all_hiddens[ctx_name][l] = outputs.hidden_states[l][0, -1, :].detach().clone().float()

        # 构建有结构探针
        # 探针来源: 不同上下文间的差分方向 + 同一上下文的层间差分方向
        structured_probes = {}

        # 1. 上下文间差分: trans_short - zh_continue
        for l in sample_layers:
            diff = all_hiddens["trans_short"][l] - all_hiddens["zh_continue"][l]
            norm = diff.norm()
            if norm > 1e-6:
                structured_probes[f"ctx_diff_L{l}"] = diff / norm

        # 2. 上下文间差分: trans_instruction - zh_continue
        for l in sample_layers:
            diff = all_hiddens["trans_instruction"][l] - all_hiddens["zh_continue"][l]
            norm = diff.norm()
            if norm > 1e-6:
                structured_probes[f"ctx_diff2_L{l}"] = diff / norm

        # 3. 层间差分方向 (来自各上下文)
        for ctx_name in ["zh_continue", "trans_short"]:
            for l in sample_layers:
                if l < n_layers:
                    diff = all_hiddens[ctx_name][l + 1] - all_hiddens[ctx_name][l]
                    norm = diff.norm()
                    if norm > 1e-6:
                        structured_probes[f"layer_diff_{ctx_name}_L{l}"] = diff / norm

        # 4. 随机方向作为对照 (5个)
        torch.manual_seed(42)
        for i in range(5):
            random_dir = torch.randn(d_model)
            structured_probes[f"random_{i}"] = F.normalize(random_dir, dim=0)

        print(f"    构建了 {len(structured_probes)} 个有结构探针")

        # 对每种上下文计算Jacobian
        for ctx_name, template in context_templates.items():
            prompt = template(zh, en)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]

            # Baseline hidden states
            with torch.no_grad():
                base_outputs = model(input_ids, output_hidden_states=True)

            base_hiddens = {}
            for l in range(n_layers + 1):
                base_hiddens[l] = base_outputs.hidden_states[l][0, -1, :].detach().clone().float()

            ctx_jacobian = {}

            for l in sample_layers:
                if l >= n_layers:
                    continue

                print(f"      {ctx_name} L{l}...", end="", flush=True)

                h_l_norm = float(base_hiddens[l].norm())
                eps_scaled = max(0.01 * h_l_norm, eps_base)

                # 用有结构探针计算Jacobian响应
                probe_responses = {}  # probe_name → (delta_h_l1, delta_logits)

                layers_list = get_layers(model)

                for probe_name, probe_dir in structured_probes.items():
                    probe_dir = probe_dir.to(device)

                    # Hook: 扰动层l的输出
                    perturbed = [None, None]  # [h_{l+1}_perturbed, logits_perturbed]
                    intervened = [False]

                    def make_hook(eps_val, p_dir, captured, flag):
                        def hook_fn(module, input, output):
                            if not flag[0]:
                                if isinstance(output, tuple):
                                    hs = output[0].clone()
                                    hs[:, -1, :] += eps_val * p_dir.to(hs.dtype).to(device)
                                    captured[0] = hs[0, -1, :].detach().float().cpu().numpy()
                                    output = (hs,) + output[1:]
                                flag[0] = True
                            return output
                        return hook_fn

                    handle = layers_list[l].register_forward_hook(
                        make_hook(eps_scaled, probe_dir, perturbed, intervened))

                    with torch.no_grad():
                        perturbed_outputs = model(input_ids, output_hidden_states=True)

                    handle.remove()

                    # 记录h_{l+1}的变化
                    if perturbed[0] is not None:
                        delta_h_l1 = perturbed[0] - base_hiddens[l + 1].cpu().numpy()
                        probe_responses[probe_name] = {
                            "delta_h_norm": float(np.linalg.norm(delta_h_l1)),
                            "delta_h_direction": delta_h_l1 / max(np.linalg.norm(delta_h_l1), 1e-10),
                        }

                    # logits变化
                    perturbed_logits = perturbed_outputs.logits[0, -1, :].float().cpu().numpy()
                    base_logits_np = base_outputs.logits[0, -1, :].float().cpu().numpy()
                    delta_logits = perturbed_logits - base_logits_np

                    # 特定token的logit变化
                    en_tok_ids = tokenizer.encode(en, add_special_tokens=False)
                    zh_tok_ids = tokenizer.encode(zh, add_special_tokens=False)
                    en_logit_change = float(delta_logits[en_tok_ids[0]]) if en_tok_ids else 0
                    zh_logit_change = float(delta_logits[zh_tok_ids[0]]) if zh_tok_ids else 0

                    if probe_name in probe_responses:
                        probe_responses[probe_name]["en_logit_change"] = en_logit_change
                        probe_responses[probe_name]["zh_logit_change"] = zh_logit_change
                        probe_responses[probe_name]["logit_change_norm"] = float(np.linalg.norm(delta_logits))

                # 分析结果
                # 分组统计: 上下文差分探针 vs 层间差分探针 vs 随机探针
                groups = {
                    "ctx_diff": [k for k in probe_responses if k.startswith("ctx_diff")],
                    "layer_diff": [k for k in probe_responses if k.startswith("layer_diff")],
                    "random": [k for k in probe_responses if k.startswith("random")],
                }

                group_stats = {}
                for gname, gkeys in groups.items():
                    if not gkeys:
                        continue
                    delta_h_norms = [probe_responses[k]["delta_h_norm"] for k in gkeys]
                    en_logit_changes = [probe_responses[k]["en_logit_change"] for k in gkeys]
                    zh_logit_changes = [probe_responses[k]["zh_logit_change"] for k in gkeys]
                    logit_norms = [probe_responses[k]["logit_change_norm"] for k in gkeys]

                    group_stats[gname] = {
                        "mean_delta_h_norm": float(np.mean(delta_h_norms)),
                        "mean_en_logit_change": float(np.mean(np.abs(en_logit_changes))),
                        "mean_zh_logit_change": float(np.mean(np.abs(zh_logit_changes))),
                        "mean_logit_change_norm": float(np.mean(logit_norms)),
                        "n_probes": len(gkeys),
                    }

                ctx_jacobian[str(l)] = {
                    "group_stats": group_stats,
                    "top_en_impact_probes": sorted(
                        [(k, probe_responses[k]["en_logit_change"]) for k in probe_responses],
                        key=lambda x: abs(x[1]), reverse=True
                    )[:5],
                }

                # 简洁打印
                stats_str = " | ".join(
                    [f"{g}: Δh={group_stats[g]['mean_delta_h_norm']:.4f}, "
                     f"en_Δlogit={group_stats[g]['mean_en_logit_change']:.6f}"
                     for g in group_stats]
                )
                print(f" {stats_str}")

            word_jacobian[ctx_name] = ctx_jacobian

        jacobian_results[f"{zh}_{en}"] = word_jacobian

    # ---- 跨词聚合: 3种上下文的Jacobian差异 ----
    print(f"\n\n  === 跨词聚合: 条件化Jacobian差异 ===")

    # 核心问题: 3种上下文的Jacobian是离散切换还是连续变形?
    # 如果是连续变形: zh_continue和trans_short的Jacobian差异 < trans_short和trans_instruction的差异
    # 如果是离散切换: 差异模式不连续

    aggregate_by_context = {}
    for ctx_name in context_templates:
        ctx_stats = {}
        for l in sample_layers:
            l_str = str(l)
            # 收集所有词的group_stats
            for gname in ["ctx_diff", "layer_diff", "random"]:
                vals_h = []
                vals_en = []
                for word_key, word_data in jacobian_results.items():
                    if ctx_name in word_data and l_str in word_data[ctx_name]:
                        gs = word_data[ctx_name][l_str].get("group_stats", {})
                        if gname in gs:
                            vals_h.append(gs[gname]["mean_delta_h_norm"])
                            vals_en.append(gs[gname]["mean_en_logit_change"])
                if vals_h:
                    if gname not in ctx_stats:
                        ctx_stats[gname] = {}
                    ctx_stats[gname][l_str] = {
                        "mean_delta_h": float(np.mean(vals_h)),
                        "mean_en_logit": float(np.mean(vals_en)),
                    }
        aggregate_by_context[ctx_name] = ctx_stats

    # 打印
    print(f"\n  层  | zh_continue(ctx_diff) | trans_short(ctx_diff) | trans_instruction(ctx_diff)")
    print(f"  {'─'*80}")
    for l in sample_layers:
        l_str = str(l)
        row = f"  L{l:2d} |"
        for ctx_name in context_templates:
            gs = aggregate_by_context[ctx_name].get("ctx_diff", {}).get(l_str, {})
            if gs:
                row += f" Δh={gs['mean_delta_h']:.4f}, en={gs['mean_en_logit']:.6f} |"
            else:
                row += f" N/A |"
        print(row)

    results["jacobian_by_word"] = jacobian_results
    results["aggregate"] = aggregate_by_context

    save_path = f"tests/glm5_temp/phase104_exp1_{model_name}_conditioned_jacobian.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 2: 局部Lyapunov分析
# ============================================================
def exp2_local_lyapunov(model_name):
    """
    局部Lyapunov分析 — 层间稳定性的精确测量

    核心概念:
    - Lyapunov指数 = log(σ_max), 其中σ_max是Jacobian最大奇异值
    - λ > 0: 不稳定(微扰被放大) — 路由相变候选区
    - λ ≈ 0: 中性 — 信息传递
    - λ < 0: 稳定(微扰被抑制) — 信息压缩

    关键改进:
    - 不只看最大Lyapunov指数，看整个谱
    - 对比不同上下文的Lyapunov谱
    - 寻找"Lyapunov spike": 某层突然不稳定 → 路由相变点

    严格化:
    - Phase 103的"放大率≈1.0"是各向同性平均，掩盖了极端方向
    - 本实验: 精确测量每个探针方向的Lyapunov指数
    """
    print(f"\n{'='*70}")
    print(f"Exp 2: 局部Lyapunov分析 — {model_name}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    results = {}

    test_pairs = [("猫", "cat"), ("水", "water"), ("火", "fire"), ("树", "tree"),
                  ("龙", "dragon"), ("光", "light"), ("冰", "ice"), ("梦", "dream")]

    n_probes = 100  # 增加到100个以精确估计谱
    eps_base = 1.0

    context_templates = {
        "zh_continue": lambda zh, en: f"{zh}是一种",
        "trans_short": lambda zh, en: f"{zh}的英文是",
    }

    lyapunov_results = {}

    for zh, en in test_pairs:
        print(f"\n  === 处理: {zh}({en}) ===")
        word_lyapunov = {}

        for ctx_name, template in context_templates.items():
            prompt = template(zh, en)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                base_outputs = model(input_ids, output_hidden_states=True)

            base_hiddens = {}
            for l in range(n_layers + 1):
                base_hiddens[l] = base_outputs.hidden_states[l][0, -1, :].detach().clone().float()

            ctx_lyapunov = {}
            layers_list = get_layers(model)

            # 每层都测！Lyapunov分析需要全层覆盖
            all_layers = list(range(n_layers))

            for l in all_layers:
                h_l_norm = float(base_hiddens[l].norm())
                eps_scaled = max(0.01 * h_l_norm, eps_base)

                # 生成随机探针
                torch.manual_seed(42)
                probes = torch.randn(n_probes, d_model, device=device)
                probes = F.normalize(probes, dim=1)

                # 计算每个探针的响应
                response_norms = []
                en_logit_changes = []

                for p_idx in range(n_probes):
                    probe = probes[p_idx]

                    intervened = [False]
                    perturbed_h = [None]

                    def make_hook(eps_val, p_vec, captured, flag):
                        def hook_fn(module, input, output):
                            if not flag[0]:
                                if isinstance(output, tuple):
                                    hs = output[0].clone()
                                    hs[:, -1, :] += eps_val * p_vec.to(hs.dtype).to(device)
                                    captured[0] = hs[0, -1, :].detach().float().cpu().numpy()
                                    output = (hs,) + output[1:]
                                flag[0] = True
                            return output
                        return hook_fn

                    handle = layers_list[l].register_forward_hook(
                        make_hook(eps_scaled, probe, perturbed_h, intervened))

                    with torch.no_grad():
                        perturbed_outputs = model(input_ids, output_hidden_states=True)

                    handle.remove()

                    if perturbed_h[0] is not None:
                        delta_h = perturbed_h[0] - base_hiddens[l + 1].cpu().numpy()
                        response_norm = np.linalg.norm(delta_h) / eps_scaled
                        response_norms.append(response_norm)

                    # logits变化
                    perturbed_logits = perturbed_outputs.logits[0, -1, :].float().cpu().numpy()
                    base_logits_np = base_outputs.logits[0, -1, :].float().cpu().numpy()
                    delta_logits = perturbed_logits - base_logits_np

                    en_tok_ids = tokenizer.encode(en, add_special_tokens=False)
                    if en_tok_ids:
                        en_logit_changes.append(float(delta_logits[en_tok_ids[0]]))

                # Lyapunov分析
                response_norms = np.array(response_norms)
                lyapunov_exponents = np.log(response_norms + 1e-10)

                # 谱统计
                max_lyapunov = float(np.max(lyapunov_exponents))
                min_lyapunov = float(np.min(lyapunov_exponents))
                mean_lyapunov = float(np.mean(lyapunov_exponents))

                # 不稳定模式比例 (λ > 0)
                unstable_fraction = float(np.mean(lyapunov_exponents > 0))

                # 强不稳定模式 (λ > 0.01)
                strongly_unstable_fraction = float(np.mean(lyapunov_exponents > 0.01))

                ctx_lyapunov[str(l)] = {
                    "max_lyapunov": max_lyapunov,
                    "min_lyapunov": min_lyapunov,
                    "mean_lyapunov": mean_lyapunov,
                    "unstable_fraction": unstable_fraction,
                    "strongly_unstable_fraction": strongly_unstable_fraction,
                    "en_logit_sensitivity": float(np.mean(np.abs(en_logit_changes))) if en_logit_changes else 0,
                    "percentile_90": float(np.percentile(lyapunov_exponents, 90)),
                    "percentile_10": float(np.percentile(lyapunov_exponents, 10)),
                }

                # 只在关键层打印
                if l % 6 == 0 or l in [5, 6, 20, 21, 26, 27, 33, 34]:
                    print(f"      {ctx_name} L{l}: max_λ={max_lyapunov:.4f}, "
                          f"mean_λ={mean_lyapunov:.4f}, "
                          f"unstable%={unstable_fraction:.3f}, "
                          f"en_sens={ctx_lyapunov[str(l)]['en_logit_sensitivity']:.6f}")

            word_lyapunov[ctx_name] = ctx_lyapunov

        lyapunov_results[f"{zh}_{en}"] = word_lyapunov

    # ---- 跨词聚合 ----
    print(f"\n\n  === 跨词聚合: Lyapunov谱 ===")
    print(f"\n  层  | zh_max_λ | trans_max_λ | zh_mean_λ | trans_mean_λ | zh_unstable% | trans_unstable% | Δ_max_λ")
    print(f"  {'─'*100}")

    aggregate = {}
    for l in range(n_layers):
        l_str = str(l)
        row_data = {}
        for ctx_name in ["zh_continue", "trans_short"]:
            max_l = [lyapunov_results[w][ctx_name][l_str]["max_lyapunov"]
                     for w in lyapunov_results if l_str in lyapunov_results[w].get(ctx_name, {})]
            mean_l = [lyapunov_results[w][ctx_name][l_str]["mean_lyapunov"]
                      for w in lyapunov_results if l_str in lyapunov_results[w].get(ctx_name, {})]
            uns = [lyapunov_results[w][ctx_name][l_str]["unstable_fraction"]
                   for w in lyapunov_results if l_str in lyapunov_results[w].get(ctx_name, {})]
            en_s = [lyapunov_results[w][ctx_name][l_str]["en_logit_sensitivity"]
                    for w in lyapunov_results if l_str in lyapunov_results[w].get(ctx_name, {})]

            if max_l:
                row_data[ctx_name] = {
                    "max_lyapunov": float(np.mean(max_l)),
                    "mean_lyapunov": float(np.mean(mean_l)),
                    "unstable_fraction": float(np.mean(uns)),
                    "en_sensitivity": float(np.mean(en_s)),
                }

        aggregate[l_str] = row_data

        if "zh_continue" in row_data and "trans_short" in row_data:
            zh_d = row_data["zh_continue"]
            tr_d = row_data["trans_short"]
            delta_max = tr_d["max_lyapunov"] - zh_d["max_lyapunov"]
            print(f"  L{l:2d} | {zh_d['max_lyapunov']:8.4f} | {tr_d['max_lyapunov']:11.4f} | "
                  f"{zh_d['mean_lyapunov']:9.4f} | {tr_d['mean_lyapunov']:12.4f} | "
                  f"{zh_d['unstable_fraction']:11.3f} | {tr_d['unstable_fraction']:14.3f} | "
                  f"{delta_max:+.4f}")

    # ---- 找Lyapunov spike ----
    print(f"\n  === Lyapunov Spike 检测 ===")
    spikes = []
    for l in range(n_layers):
        l_str = str(l)
        if "zh_continue" in aggregate.get(l_str, {}) and "trans_short" in aggregate.get(l_str, {}):
            zh_max = aggregate[l_str]["zh_continue"]["max_lyapunov"]
            tr_max = aggregate[l_str]["trans_short"]["max_lyapunov"]
            delta = tr_max - zh_max
            spikes.append((l, zh_max, tr_max, delta))

    # 按翻译上下文的max Lyapunov排序
    spikes_sorted = sorted(spikes, key=lambda x: x[2], reverse=True)
    print(f"  按trans_short max_λ排序 (最不稳定层):")
    for l, zh_m, tr_m, delta in spikes_sorted[:10]:
        print(f"    L{l}: trans_max_λ={tr_m:.4f}, zh_max_λ={zh_m:.4f}, Δ={delta:+.4f}")

    # 按Δ排序 (翻译特异性)
    spikes_delta = sorted(spikes, key=lambda x: abs(x[3]), reverse=True)
    print(f"\n  按|Δ_max_λ|排序 (翻译特异不稳定层):")
    for l, zh_m, tr_m, delta in spikes_delta[:10]:
        print(f"    L{l}: Δ={delta:+.4f}, trans_max_λ={tr_m:.4f}, zh_max_λ={zh_m:.4f}")

    results["lyapunov_by_word"] = lyapunov_results
    results["aggregate"] = aggregate

    save_path = f"tests/glm5_temp/phase104_exp2_{model_name}_lyapunov.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 3: 轨迹束分析 — 从离散电路到连续流场
# ============================================================
def exp3_trajectory_bundle(model_name):
    """
    轨迹束分析 — 检验"离散电路"vs"连续变形"

    Phase 103的错误: 把不同prompt格式解释为"独立电路"
    正确理解: 可能是同一连续动力系统的不同轨道区域

    本实验:
    - 构造prompt连续插值族 (从中文到翻译，20步)
    - 在每一步，记录所有层的hidden state轨迹
    - 分析轨迹束的:
      (a) 连续性: 是否平滑变化? 还是突然跳变?
      (b) 分叉点: 轨迹在哪些层/哪些步骤分叉?
      (c) 曲率: 轨迹弯曲最大的地方 = 关键计算层

    这直接检验"离散电路"vs"连续变形"
    """
    print(f"\n{'='*70}")
    print(f"Exp 3: 轨迹束分析 — {model_name}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    results = {}

    test_pairs = [("猫", "cat"), ("水", "water"), ("火", "fire"), ("树", "tree"),
                  ("龙", "dragon"), ("光", "light")]

    # 构造连续插值prompt族
    # 关键: 不是离散的3种格式，而是连续过渡
    def make_continuous_prompts(zh, en, n_steps=20):
        """
        从纯中文到纯翻译的连续插值
        方法: 在embedding空间做插值(而不是token空间)
        但token空间无法插值，所以用语义渐变的prompt序列
        """
        prompts = []
        # 语义渐变序列: 中文描述 → 接近翻译 → 翻译
        semantic_steps = [
            f"{zh}是一种",
            f"{zh}是一种动物",
            f"{zh}是一种常见的动物",
            f"关于{zh}，",
            f"关于{zh}，我想说",
            f"关于{zh}，我们知道",
            f"{zh}在中文里表示",
            f"{zh}在中文学里意思是",
            f"{zh}在中文里叫",
            f"{zh}在中文学中的含义是",
            f"{zh}的语言学含义是",
            f"{zh}在其他语言中",
            f"{zh}在英语中",
            f"{zh}在英文中叫做",
            f"{zh}在英文中称为",
            f"{zh}的英文名称是",
            f"{zh}的英文翻译是",
            f"{zh}的英文是",
            f'请把"{zh}"翻译成英文：',
            f"Translate {zh} to English:",
        ]
        return semantic_steps[:n_steps]

    trajectory_results = {}

    for zh, en in test_pairs:
        print(f"\n  === 处理: {zh}({en}) ===")
        prompts = make_continuous_prompts(zh, en, n_steps=20)

        # 收集每步的完整轨迹
        all_trajectories = []  # [n_steps, n_layers+1, d_model]

        for p_idx, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)

            trajectory = []
            for l in range(n_layers + 1):
                h = outputs.hidden_states[l][0, -1, :].detach().float().cpu().numpy()
                trajectory.append(h)

            all_trajectories.append(trajectory)

            # 输出分布
            logits = outputs.logits[0, -1, :].float().cpu()
            probs = torch.softmax(logits, dim=-1)

            en_tok_ids = tokenizer.encode(en, add_special_tokens=False)
            en_prob = probs[en_tok_ids[0]].item() if en_tok_ids else 0

            zh_tok_ids = tokenizer.encode(zh, add_special_tokens=False)
            zh_prob = probs[zh_tok_ids[0]].item() if zh_tok_ids else 0

            top5 = get_top_k_from_logits(logits, tokenizer, 5)
            top1 = top5[0]["token"] if top5 else ""

            if p_idx % 5 == 0 or en_prob > 0.01:
                print(f"    [{p_idx:2d}] {prompt:35s} → P({en})={en_prob:.6f}, "
                      f"P({zh})={zh_prob:.6f}, top1='{top1}'")

        # ---- 分析轨迹束 ----
        all_trajectories = np.array(all_trajectories)  # [n_steps, n_layers+1, d_model]

        # 1. 轨迹连续性: 相邻prompt的hidden state距离
        print(f"\n    === 轨迹连续性分析 ===")
        continuity = {}
        for l in [0, 9, 18, 27, 33]:
            if l >= n_layers + 1:
                continue
            step_distances = []
            for s in range(len(prompts) - 1):
                dist = np.linalg.norm(all_trajectories[s + 1, l] - all_trajectories[s, l])
                step_distances.append(dist)
            continuity[str(l)] = {
                "mean_step_dist": float(np.mean(step_distances)),
                "max_step_dist": float(np.max(step_distances)),
                "step_distances": [float(d) for d in step_distances],
            }
            # 找最大跳变
            max_jump_idx = int(np.argmax(step_distances))
            print(f"      L{l}: mean_step_dist={np.mean(step_distances):.2f}, "
                  f"max_jump at step[{max_jump_idx}→{max_jump_idx+1}]: "
                  f"'{prompts[max_jump_idx]}' → '{prompts[max_jump_idx+1]}' "
                  f"(dist={max(step_distances):.2f})")

        # 2. 轨迹分叉: 某层开始，不同prompt的轨迹发散
        print(f"\n    === 轨迹分叉分析 ===")
        # 用第一个prompt和最后一个prompt的轨迹距离来衡量分叉
        first_traj = all_trajectories[0]   # 中文
        last_traj = all_trajectories[-1]   # 翻译

        # 标准化: 用第一个prompt的L0 hidden state范数归一化
        norm_factor = np.linalg.norm(first_traj[0])

        layer_divergence = []
        for l in range(n_layers + 1):
            dist = np.linalg.norm(first_traj[l] - last_traj[l]) / norm_factor
            layer_divergence.append(dist)

        # 找分叉最快的层
        divergence_accel = np.diff(layer_divergence, 2) if len(layer_divergence) > 2 else []
        if len(divergence_accel) > 0:
            max_accel_layer = int(np.argmax(divergence_accel))
            print(f"      最大轨迹发散加速层: L{max_accel_layer} "
                  f"(accel={divergence_accel[max_accel_layer]:.4f})")

        # 3. 轨迹曲率: 每层处的"弯曲度"
        print(f"\n    === 轨迹曲率分析 ===")
        # 曲率 ≈ 二阶差分的范数 / 一阶差分的范数
        for s_idx in [0, 10, 19]:  # 中文/中间/翻译
            if s_idx >= len(prompts):
                continue
            curvatures = []
            for l in range(1, n_layers):
                dh1 = all_trajectories[s_idx, l] - all_trajectories[s_idx, l - 1]
                dh2 = all_trajectories[s_idx, l + 1] - all_trajectories[s_idx, l]
                # 二阶差分
                d2h = dh2 - dh1
                curvature = np.linalg.norm(d2h) / max(np.linalg.norm(dh1), 1e-6)
                curvatures.append((l, float(curvature)))

            # 找最大曲率层
            curvatures_sorted = sorted(curvatures, key=lambda x: x[1], reverse=True)
            top3 = curvatures_sorted[:3]
            print(f"      prompt[{s_idx}] 最大曲率层: " +
                  ", ".join([f"L{l}(κ={c:.4f})" for l, c in top3]))

        # 4. 关键检验: 连续变形 vs 离散跳变
        print(f"\n    === 连续变形 vs 离散跳变检验 ===")
        # 检验: 输出概率分布的变化是否连续
        en_probs_trajectory = []
        for s_idx in range(len(prompts)):
            inputs = tokenizer(prompts[s_idx], return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(inputs["input_ids"]).logits[0, -1, :].float().cpu()
            probs = torch.softmax(logits, dim=-1)
            en_tok_ids = tokenizer.encode(en, add_special_tokens=False)
            en_prob = probs[en_tok_ids[0]].item() if en_tok_ids else 0
            en_probs_trajectory.append(en_prob)

        # 检测不连续跳变
        en_probs_arr = np.array(en_probs_trajectory)
        jumps = np.diff(en_probs_arr)
        max_jump = float(np.max(np.abs(jumps)))
        max_jump_idx = int(np.argmax(np.abs(jumps)))

        # 连续变形的标志: 最大跳变 < 10%的总变化
        total_change = float(np.max(en_probs_arr) - np.min(en_probs_arr))
        relative_jump = max_jump / max(total_change, 1e-10)

        verdict = "CONTINUOUS" if relative_jump < 0.3 else "DISCRETE JUMP"
        print(f"      P({en})变化: min={np.min(en_probs_arr):.6f}, "
              f"max={np.max(en_probs_arr):.6f}, total_change={total_change:.6f}")
        print(f"      最大跳变: step[{max_jump_idx}→{max_jump_idx+1}], "
              f"|ΔP|={max_jump:.6f}, relative={relative_jump:.3f}")
        print(f"      判定: {verdict}")

        trajectory_results[f"{zh}_{en}"] = {
            "en_probs_trajectory": [float(p) for p in en_probs_trajectory],
            "continuity": continuity,
            "layer_divergence": [float(d) for d in layer_divergence],
            "max_jump_relative": relative_jump,
            "deformation_verdict": verdict,
            "prompts": prompts,
        }

    # ---- 跨词聚合 ----
    print(f"\n\n  === 跨词聚合: 连续变形 vs 离散跳变 ===")
    verdicts = [trajectory_results[w]["deformation_verdict"] for w in trajectory_results]
    n_continuous = sum(1 for v in verdicts if v == "CONTINUOUS")
    n_discrete = sum(1 for v in verdicts if v == "DISCRETE JUMP")
    print(f"  连续变形: {n_continuous}/{len(verdicts)}")
    print(f"  离散跳变: {n_discrete}/{len(verdicts)}")

    rel_jumps = [trajectory_results[w]["max_jump_relative"] for w in trajectory_results]
    print(f"  平均相对跳变: {np.mean(rel_jumps):.3f}")
    print(f"  最大相对跳变: {np.max(rel_jumps):.3f}")

    results["trajectory"] = trajectory_results
    results["verdict_summary"] = {
        "n_continuous": n_continuous,
        "n_discrete": n_discrete,
        "mean_relative_jump": float(np.mean(rel_jumps)),
    }

    save_path = f"tests/glm5_temp/phase104_exp3_{model_name}_trajectory_bundle.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 4: 最小控制能量
# ============================================================
def exp4_minimal_control_energy(model_name):
    """
    最小控制能量 — 翻译的真正代价

    核心问题(修正Phase 103):
    Phase 103问: "翻译方向存在吗?" (错误问题)
    正确问题: "多小的扰动能把系统从中文attractor推到英文attractor?"

    这才是"翻译的数学本质":
    - 不是找一个"翻译方向"
    - 而是找最小控制能量 min||δ|| s.t. 中文prompt+δ → 英文输出

    方法:
    - 在不同层注入不同大小的扰动
    - 沿不同方向(Δh SVD方向、随机方向、翻译差分方向)
    - 测量: 多大的扰动才能让P(英文翻译词)>阈值
    - 这给出"翻译的最小控制能量"

    严格化:
    - 不是"方向存在否" (二元问题)
    - 而是"控制能量多大" (连续量)
    """
    print(f"\n{'='*70}")
    print(f"Exp 4: 最小控制能量 — {model_name}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    results = {}

    test_pairs = [("猫", "cat"), ("水", "water"), ("火", "fire"), ("树", "tree"),
                  ("龙", "dragon"), ("光", "light"), ("冰", "ice"), ("梦", "dream")]

    # 先收集两种方向: Δh SVD方向 + 翻译差分方向
    print(f"\n  === Step 1: 收集方向 ===")

    # 1. Δh SVD方向
    all_delta_h = []
    for zh, en in test_pairs:
        trans_prompt = f'请把"{zh}"翻译成英文：'
        inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        for l in range(n_layers):
            h_l = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
            h_l1 = outputs.hidden_states[l + 1][0, -1, :].float().cpu().numpy()
            all_delta_h.append(h_l1 - h_l)

    all_delta_h = np.array(all_delta_h)
    U_dh, S_dh, Vt_dh = np.linalg.svd(all_delta_h, full_matrices=False)

    # 2. 翻译差分方向 (每层独立，来自不同prompt的hidden state差)
    print(f"    收集翻译差分方向...")
    translation_diff_dirs = {}  # l → 方向
    train_pairs = [("猫", "cat"), ("狗", "dog"), ("书", "book"), ("火", "fire"),
                   ("花", "flower"), ("鱼", "fish"), ("树", "tree"), ("鸟", "bird"),
                   ("马", "horse"), ("铁", "iron"), ("金", "gold"), ("茶", "tea")]
    for l in [9, 15, 21, 27, 33]:
        if l >= n_layers:
            continue
        deltas = []
        for zh_t, en_t in train_pairs:
            trans_prompt = f'请把"{zh_t}"翻译成英文：'
            zh_prompt = f"{zh_t}是一种"
            with torch.no_grad():
                out_trans = model(tokenizer(trans_prompt, return_tensors="pt").to(device)["input_ids"],
                                output_hidden_states=True)
                out_zh = model(tokenizer(zh_prompt, return_tensors="pt").to(device)["input_ids"],
                              output_hidden_states=True)
            h_trans = out_trans.hidden_states[l][0, -1, :].float().cpu().numpy()
            h_zh = out_zh.hidden_states[l][0, -1, :].float().cpu().numpy()
            deltas.append(h_trans - h_zh)
        mean_delta = np.mean(deltas, axis=0)
        mean_norm = np.linalg.norm(mean_delta)
        if mean_norm > 1e-6:
            translation_diff_dirs[l] = mean_delta / mean_norm
        print(f"      L{l}: ||trans_diff_dir||={mean_norm:.1f}")

    # 测试层 (包含有翻译差分方向的层)
    test_layers = sorted(set([6, 12, 18, 21, 24, 27, 30, 33] + list(translation_diff_dirs.keys())))
    test_layers = [l for l in test_layers if l < n_layers]

    # 方向类别 — 增加翻译差分方向!
    direction_sources = {
        "svd_top1": 0,
        "svd_top5_mean": list(range(5)),
        "svd_mid20_mean": list(range(18, 23)),
        "trans_diff": "special",  # 特殊处理: 每层用该层的翻译差分方向
        "random": None,
    }

    # 扰动幅度搜索 (更大范围!)
    alpha_search = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

    control_energy_results = {}
    layers_list = get_layers(model)

    for zh, en in test_pairs:
        print(f"\n  === 处理: {zh}({en}) ===")
        zh_prompt = f"{zh}是一种"
        inputs = tokenizer(zh_prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Baseline
        with torch.no_grad():
            base_logits = model(input_ids).logits[0, -1, :].float().cpu()
        base_en_prob = get_token_prob_from_logits(base_logits, tokenizer, en)
        print(f"    Baseline P({en})={base_en_prob:.8f}")

        word_energy = {}

        for l in test_layers:
            layer_energy = {}

            for dir_name, dir_idx in direction_sources.items():
                if dir_idx is None:
                    # 随机方向
                    torch.manual_seed(42 + l)
                    direction = torch.randn(d_model, device=device)
                    direction = F.normalize(direction, dim=0)
                elif dir_idx == "special":
                    # 翻译差分方向 (层特定)
                    if l not in translation_diff_dirs:
                        continue
                    direction = torch.tensor(translation_diff_dirs[l], dtype=torch.float32, device=device)
                    direction = F.normalize(direction, dim=0)
                elif isinstance(dir_idx, list):
                    # 多个方向的平均
                    if max(dir_idx) >= len(S_dh):
                        continue
                    dirs = Vt_dh[dir_idx]
                    mean_dir = np.mean(dirs, axis=0)
                    direction = torch.tensor(mean_dir, dtype=torch.float32, device=device)
                    direction = F.normalize(direction, dim=0)
                else:
                    if dir_idx >= len(S_dh):
                        continue
                    direction = torch.tensor(Vt_dh[dir_idx], dtype=torch.float32, device=device)
                    direction = F.normalize(direction, dim=0)

                # 搜索最小控制能量
                # 定义"成功": P(en) > 0.01 (从基线0.00005提升200倍)
                threshold = 0.01
                min_alpha_for_success = None

                for alpha in alpha_search:
                    intervened = [False]

                    def make_hook(alpha_val, d, flag):
                        def hook_fn(module, input, output):
                            if not flag[0]:
                                if isinstance(output, tuple):
                                    hs = output[0].clone()
                                    hs[:, -1, :] += alpha_val * d.to(hs.dtype).to(device)
                                    output = (hs,) + output[1:]
                                flag[0] = True
                            return output
                        return hook_fn

                    handle = layers_list[l].register_forward_hook(
                        make_hook(alpha, direction, intervened))

                    with torch.no_grad():
                        int_logits = model(input_ids).logits[0, -1, :].float().cpu()

                    handle.remove()

                    en_prob = get_token_prob_from_logits(int_logits, tokenizer, en)

                    if en_prob >= threshold:
                        min_alpha_for_success = alpha
                        break  # 找到最小alpha

                # 计算控制能量 = alpha * ||direction|| * ||h_l||
                with torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)
                h_l_norm = float(outputs.hidden_states[l][0, -1, :].norm())

                if min_alpha_for_success is not None:
                    energy = min_alpha_for_success  # 相对能量 (alpha值)
                    print(f"    L{l} {dir_name:20s}: min_α={min_alpha_for_success:.1f}, "
                          f"energy_α={energy:.1f}, reached P({en})={en_prob:.6f}")
                else:
                    energy = float('inf')
                    print(f"    L{l} {dir_name:20s}: FAILED (max α=100 insufficient)")

                layer_energy[dir_name] = {
                    "min_alpha": min_alpha_for_success,
                    "control_energy": energy,
                    "h_l_norm": h_l_norm,
                }

            word_energy[str(l)] = layer_energy

        control_energy_results[f"{zh}_{en}"] = word_energy

    # ---- 跨词聚合 ----
    print(f"\n\n  === 跨词聚合: 最小控制能量 ===")
    print(f"\n  层  | svd_top1 | svd_top5 | svd_mid20 | random | 最优方向")
    print(f"  {'─'*80}")

    for l in test_layers:
        l_str = str(l)
        agg = {}
        for dir_name in direction_sources:
            energies = []
            for word_key, word_data in control_energy_results.items():
                if l_str in word_data and dir_name in word_data[l_str]:
                    e = word_data[l_str][dir_name]["control_energy"]
                    if e != float('inf'):
                        energies.append(e)
            if energies:
                agg[dir_name] = float(np.mean(energies))
            else:
                agg[dir_name] = float('inf')

        best_dir = min(agg, key=lambda k: agg[k] if agg[k] != float('inf') else 1e10)
        print(f"  L{l:2d} | {agg['svd_top1']:8.1f} | {agg['svd_top5_mean']:8.1f} | "
              f"{agg['svd_mid20_mean']:9.1f} | {agg['random']:6.1f} | {best_dir}")

    results["control_energy"] = control_energy_results

    save_path = f"tests/glm5_temp/phase104_exp4_{model_name}_control_energy.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 5: 解码瓶颈分析 — LN vs W_unembed
# ============================================================
def exp5_decoder_bottleneck(model_name):
    """
    解码瓶颈分析 — 区分LN效果和W_unembed效果

    Phase 103硬伤: "最终LN把44维压到1维"
    修正: LayerNorm只做recenter+rescale，不消灭维度
    L35的rank=1更可能是decoder bottleneck

    本实验分离:
    1. LayerNorm对hidden state的变换 → 看LN后的rank
    2. W_unembed投影 → 看投影后的rank
    3. 两者联合效果 → 看最终logits的rank

    严格化:
    - Phase 103的Jacobian实际上包含了LN+unembed
    - 本实验: 逐步分离每个组件的贡献
    """
    print(f"\n{'='*70}")
    print(f"Exp 5: 解码瓶颈分析 — {model_name}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    results = {}

    test_pairs = [("猫", "cat"), ("水", "water"), ("火", "fire"), ("树", "tree"),
                  ("龙", "dragon"), ("光", "light"), ("冰", "ice"), ("梦", "dream")]

    W_U = get_W_U(model)  # [vocab_size, d_model]
    print(f"  W_U shape: {W_U.shape}")

    # 分析W_U的秩
    W_U_np = np.array(W_U, dtype=np.float64) if not isinstance(W_U, np.ndarray) else W_U.astype(np.float64)
    print(f"    W_U SVD (float64 for precision)...", end="", flush=True)
    U_wu, S_wu, Vt_wu = np.linalg.svd(W_U_np, full_matrices=False)
    print(f" done, rank={np.sum(S_wu > 1e-5)}")

    total_var_wu = np.sum(S_wu**2)
    cumvar_wu = np.cumsum(S_wu**2) / total_var_wu
    rank_90_wu = int(np.searchsorted(cumvar_wu, 0.9)) + 1
    rank_99_wu = int(np.searchsorted(cumvar_wu, 0.99)) + 1

    print(f"\n  === W_unembed 秩分析 ===")
    print(f"    W_U 有效秩(90%方差): {rank_90_wu}")
    print(f"    W_U 有效秩(99%方差): {rank_99_wu}")
    print(f"    Top10 奇异值: {[f'{s:.1f}' for s in S_wu[:10]]}")

    # 分析: LN之前 vs LN之后 vs unembed之后的秩
    bottleneck_results = {}

    for zh, en in test_pairs:
        print(f"\n  === 处理: {zh}({en}) ===")
        word_bottleneck = {}

        for ctx_name, prompt in [
            ("zh_continue", f"{zh}是一种"),
            ("trans_short", f"{zh}的英文是"),
        ]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)

            # 收集每层的hidden state
            hiddens = {}
            for l in range(n_layers + 1):
                hiddens[l] = outputs.hidden_states[l][0, -1, :].detach().float().cpu().numpy()

            ctx_analysis = {}

            # 分析最后3层的维度结构
            for l in [n_layers - 3, n_layers - 2, n_layers - 1]:
                if l < 0:
                    continue

                h_l = hiddens[l]  # LN之前的hidden state (实际是层l的输出)

                # Step 1: h_l本身的秩分析
                # 单个向量的"秩"没有意义，但我们看h_l的范数分布
                h_l_norm = np.linalg.norm(h_l)

                # Step 2: 模拟LN的效果
                # LN(h) = (h - mean(h)) / std(h) * gamma + beta
                # 简化: 只看recenter+rescale的效果
                h_l_centered = h_l - np.mean(h_l)
                h_l_std = np.std(h_l)
                if h_l_std > 1e-6:
                    h_l_normalized = h_l_centered / h_l_std
                else:
                    h_l_normalized = h_l_centered

                # Step 3: W_unembed投影
                # logits = h @ W_U^T
                logits_from_h = h_l.astype(np.float64) @ W_U_np.T
                logits_from_h_norm = h_l_normalized.astype(np.float64) @ W_U_np.T

                # Step 4: 分析logits的秩
                # logits是vocab_size维向量，但我们关注的是:
                # 多少个主成分解释了90%的logits方差?
                # 这需要多个样本，但只有1个样本。
                # 替代: 看logits的稀疏性(top-k概率占多少)

                logits_probs_h = np.exp(logits_from_h - np.max(logits_from_h))
                logits_probs_h = logits_probs_h / np.sum(logits_probs_h)

                logits_probs_h_norm = np.exp(logits_from_h_norm - np.max(logits_from_h_norm))
                logits_probs_h_norm = logits_probs_h_norm / np.sum(logits_probs_h_norm)

                # 稀疏性度量: top-10 token的概率和
                top10_prob_h = float(np.sum(np.sort(logits_probs_h)[-10:]))
                top10_prob_h_norm = float(np.sum(np.sort(logits_probs_h_norm)[-10:]))

                # top-100 token的概率和
                top100_prob_h = float(np.sum(np.sort(logits_probs_h)[-100:]))
                top100_prob_h_norm = float(np.sum(np.sort(logits_probs_h_norm)[-100:]))

                # 特定token概率
                en_tok_ids = tokenizer.encode(en, add_special_tokens=False)
                zh_tok_ids = tokenizer.encode(zh, add_special_tokens=False)

                en_prob_from_h = float(logits_probs_h[en_tok_ids[0]]) if en_tok_ids else 0
                en_prob_from_h_norm = float(logits_probs_h_norm[en_tok_ids[0]]) if en_tok_ids else 0

                ctx_analysis[str(l)] = {
                    "h_l_norm": float(h_l_norm),
                    "h_l_normalized_norm": float(np.linalg.norm(h_l_normalized)),
                    "top10_prob_from_h": top10_prob_h,
                    "top10_prob_from_h_norm": top10_prob_h_norm,
                    "top100_prob_from_h": top100_prob_h,
                    "top100_prob_from_h_norm": top100_prob_h_norm,
                    "en_prob_from_h": en_prob_from_h,
                    "en_prob_from_h_norm": en_prob_from_h_norm,
                }

                print(f"    {ctx_name} L{l}: ||h||={h_l_norm:.1f}, "
                      f"||h_norm||={np.linalg.norm(h_l_normalized):.2f}, "
                      f"top10_raw={top10_prob_h:.4f}, top10_LN={top10_prob_h_norm:.4f}, "
                      f"P({en})_raw={en_prob_from_h:.6f}, P({en})_LN={en_prob_from_h_norm:.6f}")

            # Step 5: 分析W_U的"解码敏感方向"
            # W_U的行是vocab中每个token的解码方向
            # W_U的SVD给出: 最容易解码的方向 (大奇异值) vs 难解码的方向 (小奇异值)
            # 问题: W_U是否能区分中英文token?

            en_tok_ids = tokenizer.encode(en, add_special_tokens=False)
            zh_tok_ids = tokenizer.encode(zh, add_special_tokens=False)

            # 中文和英文token的W_U行在SVD空间中的投影
            if en_tok_ids and zh_tok_ids:
                en_wu = W_U_np[en_tok_ids[0]]  # [d_model]
                zh_wu = W_U_np[zh_tok_ids[0]]  # [d_model]

                # 在W_U的SVD空间中的坐标
                en_svd_coords = en_wu @ Vt_wu[:10].T  # 前10个SVD方向
                zh_svd_coords = zh_wu @ Vt_wu[:10].T

                # 中英文token在SVD空间的距离
                svd_distance = np.linalg.norm(en_svd_coords - zh_svd_coords)
                cos_sim = float(np.dot(en_wu, zh_wu) / (np.linalg.norm(en_wu) * np.linalg.norm(zh_wu) + 1e-10))

                ctx_analysis["wu_analysis"] = {
                    "en_zh_cos_sim": cos_sim,
                    "en_zh_svd_distance": float(svd_distance),
                    "en_svd_coords": [float(c) for c in en_svd_coords],
                    "zh_svd_coords": [float(c) for c in zh_svd_coords],
                }

                print(f"    {ctx_name} W_U: en-zh cos_sim={cos_sim:.4f}, "
                      f"svd_dist={svd_distance:.4f}")

            word_bottleneck[ctx_name] = ctx_analysis

        bottleneck_results[f"{zh}_{en}"] = word_bottleneck

    # ---- W_U的秩和语言分离性 ----
    print(f"\n\n  === W_U: 语言分离性分析 ===")

    # 收集中英文token的W_U行
    en_wu_rows = []
    zh_wu_rows = []
    for zh, en in test_pairs:
        en_tok_ids = tokenizer.encode(en, add_special_tokens=False)
        zh_tok_ids = tokenizer.encode(zh, add_special_tokens=False)
        if en_tok_ids:
            en_wu_rows.append(W_U_np[en_tok_ids[0]])
        if zh_tok_ids:
            zh_wu_rows.append(W_U_np[zh_tok_ids[0]])

    en_wu_matrix = np.array(en_wu_rows)  # [n_words, d_model]
    zh_wu_matrix = np.array(zh_wu_rows)  # [n_words, d_model]

    # 中英文W_U行的SVD
    en_U, en_S, en_Vt = np.linalg.svd(en_wu_matrix, full_matrices=False)
    zh_U, zh_S, zh_Vt = np.linalg.svd(zh_wu_matrix, full_matrices=False)

    print(f"    英文token W_U行 top5 奇异值: {[f'{s:.2f}' for s in en_S[:5]]}")
    print(f"    中文token W_U行 top5 奇异值: {[f'{s:.2f}' for s in zh_S[:5]]}")

    # CCA分析: 中英文W_U行的子空间对齐度
    # 简化: 看前k个主方向的重叠
    k = min(5, en_wu_matrix.shape[0], zh_wu_matrix.shape[0])
    en_subspace = en_Vt[:k]  # [k, d_model]
    zh_subspace = zh_Vt[:k]  # [k, d_model]

    # 子空间重叠: ||en_Vt @ zh_Vt^T||_F / k
    overlap = np.linalg.norm(en_subspace @ zh_subspace.T, 'fro') / k
    print(f"    W_U 中英文子空间重叠度 (k={k}): {overlap:.4f}")

    results["wu_rank"] = {
        "rank_90": rank_90_wu,
        "rank_99": rank_99_wu,
        "top10_sv": [float(s) for s in S_wu[:10]],
        "en_zh_overlap": float(overlap),
    }
    results["bottleneck"] = bottleneck_results

    save_path = f"tests/glm5_temp/phase104_exp5_{model_name}_decoder_bottleneck.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=1, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    if args.exp == 1:
        exp1_conditioned_jacobian(args.model)
    elif args.exp == 2:
        exp2_local_lyapunov(args.model)
    elif args.exp == 3:
        exp3_trajectory_bundle(args.model)
    elif args.exp == 4:
        exp4_minimal_control_energy(args.model)
    elif args.exp == 5:
        exp5_decoder_bottleneck(args.model)
