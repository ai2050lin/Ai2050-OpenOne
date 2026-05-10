"""
Phase 97: 语义计算原语库 — 从"电路图谱"到"计算原语"
=====================================================
核心转折: Phase 96批判的核心洞察:
  1. Residual Patching ≠ 计算完成 (信息可移植 ≠ 计算完成)
  2. Jaccard≈0 不代表完全不同电路 (对连续系统太粗糙)
  3. "超冗余编码"结论不成立 (补偿/下游恢复/残差绕过都可能)
  4. Prompt confound 污染 (translation/retrieval prompt结构不同)
  5. "L6事实知识层"证据不足 (需跨任务/跨entity验证)

本Phase的方法论修正:
  - 修正术语: "portable information首次形成" 而非 "计算完成"
  - 结构匹配控制: 同格式prompt消除token/syntax混杂
  - Head贡献向量: 替代Jaccard，用连续相似度
  - 激活补丁矩阵: 2D source×target层patching，看信息流
  - 因果追踪: 哪条路径真正驱动最终token

目标: 识别语言模型的最小计算原语
  - Entity retrieval (实体检索)
  - Relation binding (关系绑定)
  - Cross-lingual alignment (跨语言对齐)
  - Candidate suppression (候选压制)
  - Constraint enforcement (约束满足)

Run:
  python tests/glm5/ccml_phase97_computational_primitives.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase97_computational_primitives.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase97_computational_primitives.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase97_computational_primitives.py --model glm4 --exp 2
  python tests/glm5/ccml_phase97_computational_primitives.py --model deepseek7b --exp 2
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

from model_utils import load_model, get_layers, get_model_info, release_model


# ============================================================
# 结构匹配控制用例 — 消除prompt格式混杂
# ============================================================
# 核心思想: 同一中文词，用不同指令格式引导不同任务
# 这样token结构/长度/punctuation完全一致

# 格式1: 翻译 "X的英文是" → 翻译任务
# 格式2: 补全 "X是一种" → 检索/分类任务
# 格式3: 类比 "X之于Y相当于" → 类比任务
# 格式4: 否定 "不是X而是" → 否定任务

STRUCTURE_MATCHED_PAIRS = [
    # (翻译prompt, 翻译target, 补全prompt, 补全target)
    # 同一个实体，不同任务方向
    ("猫的英文是", "cat", "猫是一种", "动物"),
    ("狗的英文是", "dog", "狗是一种", "动物"),
    ("书的英文是", "book", "书是一种", "物品"),
    ("水的英文是", "water", "水是一种", "液体"),
    ("火的英文是", "fire", "火是一种", "现象"),
    ("花的英文是", "flower", "花是一种", "植物"),
    ("鱼的英文是", "fish", "鱼是一种", "动物"),
    ("树的英文是", "tree", "树是一种", "植物"),
    ("山的英文是", "mountain", "山是一种", "地形"),
    ("河的英文是", "river", "河是一种", "地形"),
    ("鸟的英文是", "bird", "鸟是一种", "动物"),
    ("马的英文是", "horse", "马是一种", "动物"),
    ("铁的英文是", "iron", "铁是一种", "金属"),
    ("金的英文是", "gold", "金是一种", "金属"),
    ("茶的英文是", "tea", "茶是一种", "饮料"),
    ("米的英文是", "rice", "米是一种", "食物"),
    ("血的英文是", "blood", "血是一种", "液体"),
    ("眼的英文是", "eye", "眼是一种", "器官"),
    ("手的英文是", "hand", "手是一种", "器官"),
    ("风的英文是", "wind", "风是一种", "现象"),
    ("雪的英文是", "snow", "雪是一种", "现象"),
    ("星的英文是", "star", "星是一种", "天体"),
    ("海的英文是", "sea", "海是一种", "地形"),
    ("石的英文是", "stone", "石是一种", "物质"),
    ("草的英文是", "grass", "草是一种", "植物"),
    ("门的英文是", "door", "门是一种", "物品"),
    ("路的英文是", "road", "路是一种", "设施"),
    ("雨的英文是", "rain", "雨是一种", "现象"),
    ("糖的英文是", "sugar", "糖是一种", "食物"),
    ("盐的英文是", "salt", "盐是一种", "物质"),
]

# 补全对（格式匹配但语义不同）— 更多分类补全
COMPLETION_CATEGORY = [
    ("苹果是一种", "水果"), ("香蕉是一种", "水果"), ("葡萄是一种", "水果"),
    ("老虎是一种", "动物"), ("狮子是一种", "动物"), ("熊猫是一种", "动物"),
    ("桌子是一种", "家具"), ("椅子是一种", "家具"), ("柜子是一种", "家具"),
    ("汽车是一种", "交通工具"), ("火车是一种", "交通工具"), ("飞机是一种", "交通工具"),
    ("红色是一种", "颜色"), ("蓝色是一种", "颜色"), ("绿色是一种", "颜色"),
    ("钢是一种", "金属"), ("铜是一种", "金属"), ("银是一种", "金属"),
    ("牛奶是一种", "饮料"), ("果汁是一种", "饮料"), ("咖啡是一种", "饮料"),
    ("北京是一种", "城市"), ("上海是一种", "城市"), ("东京是一种", "城市"),
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


def get_target_prob(model, tokenizer, device, prompt, target):
    """获取目标token的概率"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        return 0.0
    probs = F.softmax(logits, dim=-1)
    return probs[target_ids[0]].item()


def get_n_heads(model, model_name):
    """获取模型的注意力头数"""
    if hasattr(model.config, 'num_attention_heads'):
        return model.config.num_attention_heads
    layers = get_layers(model)
    sa = layers[0].self_attn
    W_q = sa.q_proj.weight
    d_model = W_q.shape[1]
    if hasattr(model.config, 'head_dim'):
        return d_model // model.config.head_dim
    return d_model // 128


# ============================================================
# Exp 1: 结构匹配控制 — Head贡献向量对比
# ============================================================
def exp1_structure_matched_head_contribution(model_name):
    """
    用结构匹配的prompt对，测量每个head对翻译vs补全的贡献向量，
    然后用连续相似度(cosine/CCA)替代Jaccard。
    
    关键修正:
    - 翻译: "猫的英文是" → cat
    - 补全: "猫是一种" → 动物
    两者token结构完全一致，只有任务语义不同
    """
    print(f"\n{'='*60}")
    print(f"Exp 1: 结构匹配控制 — Head贡献向量 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    n_heads = get_n_heads(model, model_name)
    d_model = info.d_model
    head_dim = d_model // n_heads
    print(f"  模型: {model_name}, 层数: {n_layers}, 头数: {n_heads}, head_dim: {head_dim}")

    # 采样层
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
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

    # 结构匹配对
    matched_pairs = STRUCTURE_MATCHED_PAIRS[:20]  # 20对

    # ---- Step 1: 基线概率 ----
    print("\n[Step 1] 计算基线概率...")
    
    baseline_translation = []
    baseline_completion = []
    
    for trans_p, trans_t, comp_p, comp_t in matched_pairs:
        p_trans = get_target_prob(model, tokenizer, device, trans_p, trans_t)
        p_comp = get_target_prob(model, tokenizer, device, comp_p, comp_t)
        baseline_translation.append(p_trans)
        baseline_completion.append(p_comp)
    
    mean_base_trans = np.mean(baseline_translation)
    mean_base_comp = np.mean(baseline_completion)
    print(f"  翻译baseline: {mean_base_trans:.4f}")
    print(f"  补全baseline: {mean_base_comp:.4f}")

    # ---- Step 2: 逐层逐头消融 → 贡献向量 ----
    print("\n[Step 2] 逐层逐头消融 → 贡献向量...")

    # 对每个head，记录其在翻译和补全任务上的贡献(drop)
    head_contributions = {}  # {(layer, head): {"translation": drop, "completion": drop}}

    # 用5对结构匹配样本
    test_pairs = matched_pairs[:5]

    for layer_idx in sample_layers:
        layer = get_layers(model)[layer_idx]
        sa = layer.self_attn

        for head_idx in range(n_heads):
            # 注册pre-hook消融该head
            def make_ablation_prehook(hi, hd):
                def prehook_fn(module, input):
                    modified = input[0].clone()
                    start = hi * hd
                    end = (hi + 1) * hd
                    modified[:, :, start:end] = 0.0
                    return (modified,) + input[1:]
                return prehook_fn

            hook_handle = sa.o_proj.register_forward_pre_hook(
                make_ablation_prehook(head_idx, head_dim)
            )

            # 测量翻译和补全的消融后概率
            ablated_trans = []
            ablated_comp = []
            for trans_p, trans_t, comp_p, comp_t in test_pairs:
                p_trans = get_target_prob(model, tokenizer, device, trans_p, trans_t)
                p_comp = get_target_prob(model, tokenizer, device, comp_p, comp_t)
                ablated_trans.append(p_trans)
                ablated_comp.append(p_comp)

            hook_handle.remove()

            # 贡献 = 基线 - 消融后 (正=该head帮助该任务，负=该head干扰该任务)
            mean_ablated_trans = np.mean(ablated_trans)
            mean_ablated_comp = np.mean(ablated_comp)
            
            contrib_trans = mean_base_trans - mean_ablated_trans
            contrib_comp = mean_base_comp - mean_ablated_comp

            head_contributions[(layer_idx, head_idx)] = {
                "translation_drop": contrib_trans,
                "completion_drop": contrib_comp,
                "translation_ablated": mean_ablated_trans,
                "completion_ablated": mean_ablated_comp,
            }

        print(f"    L{layer_idx}: 完成 {n_heads} heads")

    # ---- Step 3: 构建贡献向量并计算相似度 ----
    print("\n[Step 3] 构建贡献向量并计算相似度...")

    # 贡献向量: 每个head一个2维向量 [translation_contribution, completion_contribution]
    trans_contrib_vec = []
    comp_contrib_vec = []
    head_labels = []
    
    for (li, hi), v in sorted(head_contributions.items()):
        trans_contrib_vec.append(v["translation_drop"])
        comp_contrib_vec.append(v["completion_drop"])
        head_labels.append(f"L{li}H{hi}")

    trans_vec = np.array(trans_contrib_vec)
    comp_vec = np.array(comp_contrib_vec)

    # Cosine similarity
    norm_t = np.linalg.norm(trans_vec)
    norm_c = np.linalg.norm(comp_vec)
    if norm_t > 0 and norm_c > 0:
        cosine_sim = np.dot(trans_vec, comp_vec) / (norm_t * norm_c)
    else:
        cosine_sim = 0.0
    
    # Pearson correlation
    if len(trans_vec) > 1:
        pearson_r = np.corrcoef(trans_vec, comp_vec)[0, 1]
    else:
        pearson_r = 0.0

    print(f"  贡献向量Cosine相似度: {cosine_sim:.4f}")
    print(f"  贡献向量Pearson相关: {pearson_r:.4f}")

    # ---- Step 4: 分层分析 ----
    print("\n[Step 4] 分层贡献向量分析...")

    layer_similarity = {}
    for layer_idx in sample_layers:
        layer_trans = []
        layer_comp = []
        for (li, hi), v in head_contributions.items():
            if li == layer_idx:
                layer_trans.append(v["translation_drop"])
                layer_comp.append(v["completion_drop"])
        
        if len(layer_trans) > 1:
            lt = np.array(layer_trans)
            lc = np.array(layer_comp)
            r = np.corrcoef(lt, lc)[0, 1] if np.std(lt) > 0 and np.std(lc) > 0 else 0.0
            
            # 还看: 该层head是偏向翻译还是补全?
            mean_trans = np.mean(lt)
            mean_comp = np.mean(lc)
            specialization = mean_trans - mean_comp  # 正=偏向翻译，负=偏向补全
            
            layer_similarity[layer_idx] = {
                "pearson_r": float(r),
                "mean_trans_contribution": float(mean_trans),
                "mean_comp_contribution": float(mean_comp),
                "specialization": float(specialization),
                "n_heads": len(layer_trans),
            }
            print(f"  L{layer_idx}: Pearson={r:.3f}, trans_contrib={mean_trans:.4f}, "
                  f"comp_contrib={mean_comp:.4f}, spec={specialization:.4f}")

    # ---- Step 5: 找出分化最严重的层 ----
    print("\n[Step 5] 翻译-补全分化最严重的层...")

    sorted_by_spec = sorted(layer_similarity.items(), key=lambda x: abs(x[1]["specialization"]), reverse=True)
    for li, v in sorted_by_spec[:5]:
        direction = "翻译" if v["specialization"] > 0 else "补全"
        print(f"  L{li}: 最偏向{direction} (spec={v['specialization']:.4f})")

    # ---- 保存结果 ----
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "sample_layers": sample_layers,
        "baseline_translation": float(mean_base_trans),
        "baseline_completion": float(mean_base_comp),
        "n_test_pairs": len(test_pairs),
        "global_cosine_sim": float(cosine_sim),
        "global_pearson_r": float(pearson_r),
        "head_contributions": {f"L{li}H{hi}": v for (li, hi), v in head_contributions.items()},
        "layer_similarity": {str(k): v for k, v in layer_similarity.items()},
        "sorted_by_specialization": [(f"L{li}", v) for li, v in sorted_by_spec],
    }

    outpath = f"tests/glm5_temp/phase97_exp1_{model_name}_matched_head_contribution.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_serialize(output), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")

    release_model(model)
    return output


# ============================================================
# Exp 2: 激活补丁矩阵 — 2D source×target层patching
# ============================================================
def exp2_activation_patch_matrix(model_name):
    """
    二维patching矩阵: source层×target层
    不再只看"注入L层"，而是:
    - 从source prompt的L_s层取激活
    - 注入target prompt的L_t层
    看2D矩阵中的信息流模式
    
    这能区分:
    1. 信息在哪里首次形成 (哪些source层有portable信息)
    2. 信息在哪里被使用 (哪些target层能接受并利用信息)
    3. 信息流是"一次写入"还是"逐步传播"
    """
    print(f"\n{'='*60}")
    print(f"Exp 2: 激活补丁矩阵 — 2D Source×Target — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  模型: {model_name}, 层数: {n_layers}")

    # 用翻译对
    pairs = [
        ("猫的英文是", "cat"), ("狗的英文是", "dog"),
        ("书的英文是", "book"), ("水的英文是", "water"),
        ("火的英文是", "fire"), ("花的英文是", "flower"),
        ("鱼的英文是", "fish"), ("树的英文是", "tree"),
        ("鸟的英文是", "bird"), ("马的英文是", "horse"),
    ]

    # 采样层 — 比Phase 96更密集
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        # 每3层采样
        sample_layers = list(range(0, n_layers, 3))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)
        sample_layers = sorted(sample_layers)
    print(f"  采样层: {sample_layers}")

    # ---- Step 1: 收集所有层hidden states ----
    print("\n[Step 1] 收集所有层hidden states...")

    all_hiddens = {}  # {idx: {layer: hidden}}
    all_baseline = {}

    for idx, (prompt, target) in enumerate(pairs):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hiddens = {}
        for l in range(n_layers + 1):
            h = outputs.hidden_states[l][0, -1, :].detach().clone()
            hiddens[l] = h
        all_hiddens[idx] = hiddens

        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if target_ids:
            all_baseline[idx] = {
                "target_prob": probs[target_ids[0]].item(),
                "target_id": target_ids[0],
            }
        del outputs

    # ---- Step 2: 2D Patching矩阵 ----
    print("\n[Step 2] 构建2D Patching矩阵...")

    # 对每对(i,j): i=source, j=target
    # 从source的L_s层取h → 注入target的L_t层
    # 测量source_target的概率变化

    n_test_pairs = 5  # 用5对

    patch_matrix = {}  # {(L_s, L_t): [source_leak_values]}

    for i in range(n_test_pairs):
        source_ids = tokenizer.encode(pairs[i][1], add_special_tokens=False)
        if not source_ids:
            continue

        for j in range(n_test_pairs):
            if i == j:
                continue

            target_baseline = all_baseline[j]["target_prob"]
            target_id = all_baseline[j]["target_id"]

            for l_s in sample_layers:
                source_h = all_hiddens[i][l_s]

                for l_t in sample_layers:
                    # 只有当L_s的层 <= L_t时才做patch（信息只能从过去到未来）
                    # 但我们也测试反向，看是否有"后层信息被前层使用"
                    
                    layers = get_layers(model)

                    # 在target的L_t层输出后注入source的L_s层hidden
                    def make_2d_patch_hook(src_h):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                hidden = output[0]
                            else:
                                hidden = output
                            patched = hidden.clone()
                            # 只patch最后一个token位置
                            patched[0, -1, :] = src_h.to(patched.device).to(patched.dtype)
                            if isinstance(output, tuple):
                                return (patched,) + output[1:]
                            return patched
                        return hook_fn

                    hook_handle = layers[l_t].register_forward_hook(
                        make_2d_patch_hook(source_h)
                    )

                    # Forward with patch
                    inputs = tokenizer(pairs[j][0], return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)

                    logits = outputs.logits[0, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    source_leak = probs[source_ids[0]].item()

                    hook_handle.remove()
                    del outputs

                    key = (l_s, l_t)
                    if key not in patch_matrix:
                        patch_matrix[key] = []
                    patch_matrix[key].append(source_leak)

        print(f"  完成source {i+1}/{n_test_pairs}")

    # ---- Step 3: 聚合矩阵 ----
    print("\n[Step 3] 聚合2D矩阵...")

    # 计算每个(L_s, L_t)的平均source leak
    avg_matrix = {}
    for (l_s, l_t), vals in patch_matrix.items():
        avg_matrix[(l_s, l_t)] = np.mean(vals)

    # 打印矩阵
    print(f"\n  2D Source×Target Patching Matrix (Source Leak):")
    print(f"  {'':>6}", end="")
    for l_t in sample_layers:
        print(f"  T{l_t:>2}", end="")
    print()
    
    for l_s in sample_layers:
        print(f"  S{l_s:>3}", end="")
        for l_t in sample_layers:
            val = avg_matrix.get((l_s, l_t), 0.0)
            if val > 0.1:
                print(f"  {val:>.3f}", end="")
            else:
                print(f"    .  ", end="")
        print()

    # ---- Step 4: 分析信息流模式 ----
    print("\n[Step 4] 分析信息流模式...")

    # 1. 信息形成层: 哪些source层的leak最高(不管target层)
    source_quality = {}
    for l_s in sample_layers:
        leaks = [avg_matrix.get((l_s, l_t), 0.0) for l_t in sample_layers if l_t >= l_s]
        source_quality[l_s] = np.mean(leaks) if leaks else 0.0
    
    best_source = max(source_quality, key=source_quality.get)
    print(f"  最佳信息源层: L{best_source} (mean leak = {source_quality[best_source]:.4f})")

    # 2. 信息接收层: 哪些target层最能利用注入的信息
    target_receptivity = {}
    for l_t in sample_layers:
        leaks = [avg_matrix.get((l_s, l_t), 0.0) for l_s in sample_layers if l_s <= l_t]
        target_receptivity[l_t] = np.mean(leaks) if leaks else 0.0
    
    best_target = max(target_receptivity, key=target_receptivity.get)
    print(f"  最佳信息接收层: L{best_target} (mean leak = {target_receptivity[best_target]:.4f})")

    # 3. 对角线模式: source=target时（自注入恢复）vs source≠target
    diagonal_leaks = [avg_matrix.get((l, l), 0.0) for l in sample_layers]
    off_diagonal_leaks = []
    for l_s in sample_layers:
        for l_t in sample_layers:
            if l_s != l_t:
                off_diagonal_leaks.append(avg_matrix.get((l_s, l_t), 0.0))
    
    print(f"  对角线平均leak: {np.mean(diagonal_leaks):.4f}")
    print(f"  非对角线平均leak: {np.mean(off_diagonal_leaks):.4f}")

    # 4. 关键观察: 信息是否在特定source层"一次性形成"?
    # 如果L28的source对所有target层都有高leak → 一次性形成
    # 如果需要source≈target → 逐步传播
    peak_source_layers = sorted(source_quality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Top-5 信息源层:")
    for l, q in peak_source_layers:
        print(f"    L{l}: quality={q:.4f}")

    # ---- 保存 ----
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "sample_layers": sample_layers,
        "n_test_pairs": n_test_pairs,
        "patch_matrix": {f"S{l_s}_T{l_t}": float(v) for (l_s, l_t), v in avg_matrix.items()},
        "source_quality": {str(k): float(v) for k, v in source_quality.items()},
        "target_receptivity": {str(k): float(v) for k, v in target_receptivity.items()},
        "diagonal_mean_leak": float(np.mean(diagonal_leaks)),
        "off_diagonal_mean_leak": float(np.mean(off_diagonal_leaks)),
        "best_source_layer": int(best_source),
        "best_target_layer": int(best_target),
    }

    outpath = f"tests/glm5_temp/phase97_exp2_{model_name}_patch_matrix.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_serialize(output), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")

    release_model(model)
    return output


# ============================================================
# Exp 3: 计算原语分离 — 翻译=对齐+压制+解码
# ============================================================
def exp3_primitive_decomposition(model_name):
    """
    尝试将翻译任务分解为3个计算原语:
    1. Cross-lingual alignment (跨语言对齐): "猫"的概念 → language-independent representation
    2. Candidate suppression (候选压制): suppress "猫"/"狗"/"动物"等中文token
    3. Output decoding (输出解码): 放大"cat"的logit
    
    方法:
    - 分析logits轨迹: 候选token的概率如何随层变化
    - 区分"对齐完成"和"压制完成"和"解码完成"
    - 用消融验证: 消融不同层分别破坏哪个原语?
    """
    print(f"\n{'='*60}")
    print(f"Exp 3: 计算原语分解 — 翻译=对齐+压制+解码 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  模型: {model_name}, 层数: {n_layers}")

    # 翻译对
    pairs = [
        ("猫的英文是", "cat", "猫"),  # (prompt, en_target, zh_source)
        ("狗的英文是", "dog", "狗"),
        ("书的英文是", "book", "书"),
        ("水的英文是", "water", "水"),
        ("火的英文是", "fire", "火"),
        ("花的英文是", "flower", "花"),
        ("鱼的英文是", "fish", "鱼"),
        ("树的英文是", "tree", "树"),
        ("鸟的英文是", "bird", "鸟"),
        ("马的英文是", "horse", "马"),
        ("山的英文是", "mountain", "山"),
        ("河的英文是", "river", "河"),
        ("铁的英文是", "iron", "铁"),
        ("金的英文是", "gold", "金"),
        ("茶的英文是", "tea", "茶"),
    ]

    # 对每个翻译对，追踪3类token的概率随层变化:
    # 1. en_target (cat) — 目标英文token
    # 2. zh_source (猫) — 源中文token
    # 3. category (动物) — 类别token (代表概念对齐)
    
    # 采样层
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        sample_layers = sorted(set(
            [0, 1, 2] +
            list(range(0, n_layers, 2)) +
            [n_layers-3, n_layers-2, n_layers-1]
        ))
    print(f"  采样层: {len(sample_layers)} 层")

    # ---- Step 1: 各层logits追踪 ----
    print("\n[Step 1] 各层logits追踪 (需要逐层推理)...")

    all_trajectories = []

    for pair_idx, (prompt, en_target, zh_source) in enumerate(pairs):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 获取target token ids
        en_ids = tokenizer.encode(en_target, add_special_tokens=False)
        zh_ids = tokenizer.encode(zh_source, add_special_tokens=False)
        
        if not en_ids or not zh_ids:
            continue
        
        en_id = en_ids[0]
        zh_id = zh_ids[0]

        # 逐层收集logits
        # 方法: 在每层后注入hook收集hidden state，然后手动unembed
        # 但更简单的方法: 用output_hidden_states=True

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 获取unembedding matrix
        W_U = model.lm_head.weight  # [vocab_size, d_model]
        
        layer_trajectory = {
            "prompt": prompt,
            "en_target": en_target,
            "zh_source": zh_source,
            "layers": {},
        }

        for l in sample_layers:
            h = outputs.hidden_states[l][0, -1, :]  # [d_model]
            
            # 通过LM head获取logits
            logits = F.linear(h.to(W_U.device).to(W_U.dtype), W_U)
            probs = F.softmax(logits, dim=-1)
            
            en_prob = probs[en_id].item()
            zh_prob = probs[zh_id].item()
            
            # Top-5 tokens
            top5_vals, top5_ids = torch.topk(probs, 5)
            top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids.tolist()]
            
            layer_trajectory["layers"][str(l)] = {
                "en_target_prob": en_prob,
                "zh_source_prob": zh_prob,
                "top5_tokens": top5_tokens,
                "top5_probs": top5_vals.tolist(),
            }

        del outputs
        all_trajectories.append(layer_trajectory)

        if (pair_idx + 1) % 5 == 0:
            print(f"  已完成 {pair_idx+1}/{len(pairs)} prompts")

    # ---- Step 2: 分析3个原语的涌现时间 ----
    print("\n[Step 2] 分析3个原语的涌现时间...")

    # 对齐: zh_source概率开始下降的层（概念从"中文形式"脱离）
    # 压制: zh_source概率降到极低（<0.01）的层
    # 解码: en_target概率成为top-1的层

    for traj in all_trajectories:
        layers_data = traj["layers"]
        
        # 找对齐起始层: zh_source概率首次下降超过50%的层
        zh_probs = [(int(l), layers_data[l]["zh_source_prob"]) for l in sorted(layers_data.keys(), key=int)]
        en_probs = [(int(l), layers_data[l]["en_target_prob"]) for l in sorted(layers_data.keys(), key=int)]
        
        # 找解码层: en_target首次成为top-1的层
        decode_layer = None
        for l in sorted(layers_data.keys(), key=int):
            if layers_data[l]["top5_tokens"][0] == traj["en_target"]:
                decode_layer = int(l)
                break
        
        # 找压制层: zh_source < 0.01
        suppress_layer = None
        for l, p in zh_probs:
            if p < 0.01:
                suppress_layer = l
                break
        
        # 找对齐起始层: zh_source首次从峰值下降>50%
        if len(zh_probs) > 1:
            peak_zh = max(zh_probs, key=lambda x: x[1])
            align_start = None
            for l, p in zh_probs:
                if l >= peak_zh[0] and p < peak_zh[1] * 0.5:
                    align_start = l
                    break
        else:
            align_start = None

        traj["primitive_layers"] = {
            "align_start": align_start,
            "suppress_layer": suppress_layer,
            "decode_layer": decode_layer,
        }

        print(f"  {traj['prompt']} → {traj['en_target']}: "
              f"align≈L{align_start}, suppress≈L{suppress_layer}, decode≈L{decode_layer}")

    # ---- Step 3: 统计3个原语的典型深度 ----
    print("\n[Step 3] 统计3个原语的典型涌现深度...")

    align_layers = [t["primitive_layers"]["align_start"] for t in all_trajectories if t["primitive_layers"]["align_start"] is not None]
    suppress_layers = [t["primitive_layers"]["suppress_layer"] for t in all_trajectories if t["primitive_layers"]["suppress_layer"] is not None]
    decode_layers = [t["primitive_layers"]["decode_layer"] for t in all_trajectories if t["primitive_layers"]["decode_layer"] is not None]

    print(f"  对齐起始层: mean={np.mean(align_layers):.1f}, std={np.std(align_layers):.1f}, n={len(align_layers)}")
    print(f"  压制完成层: mean={np.mean(suppress_layers):.1f}, std={np.std(suppress_layers):.1f}, n={len(suppress_layers)}")
    print(f"  解码完成层: mean={np.mean(decode_layers):.1f}, std={np.std(decode_layers):.1f}, n={len(decode_layers)}")

    # ---- 保存 ----
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "sample_layers": sample_layers,
        "n_pairs": len(all_trajectories),
        "trajectories": all_trajectories,
        "primitive_stats": {
            "align_start": {"mean": float(np.mean(align_layers)) if align_layers else None,
                           "std": float(np.std(align_layers)) if align_layers else None,
                           "n": len(align_layers)},
            "suppress": {"mean": float(np.mean(suppress_layers)) if suppress_layers else None,
                        "std": float(np.std(suppress_layers)) if suppress_layers else None,
                        "n": len(suppress_layers)},
            "decode": {"mean": float(np.mean(decode_layers)) if decode_layers else None,
                      "std": float(np.std(decode_layers)) if decode_layers else None,
                      "n": len(decode_layers)},
        },
    }

    outpath = f"tests/glm5_temp/phase97_exp3_{model_name}_primitive_decomposition.json"
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
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()

    if args.exp == 1:
        exp1_structure_matched_head_contribution(args.model)
    elif args.exp == 2:
        exp2_activation_patch_matrix(args.model)
    elif args.exp == 3:
        exp3_primitive_decomposition(args.model)
