"""
Phase 99: 因果隔离与计算原语检测
====================================
Phase 98批判的核心升级:

  之前的方法论缺陷:
    1. "L26 attn=语言切换路由" → 只测了impact，没测necessity
    2. "振荡=语义切换" → 在token层看，没在hidden state层看
    3. "MLP主导翻译" → impact大≠执行核心计算
    4. 所有结论都是correlation，不是causation

  Phase 99核心方法论升级:
    **因果隔离(Causal Isolation)**: 证明"没有X→没有Y"，不是"X和Y同时出现"

  实验设计:
    Exp1: 因果必要性测试 — zero-ablate关键层，看翻译是否崩塌
    Exp2: Hidden State语义子空间 — 在表示层分析zh/en子空间切换
    Exp3: Head级因果中介 — 找到翻译的必要head
    Exp4: 跨任务原语检测 — 同一电路是否处理不同计算

Run:
  python tests/glm5/ccml_phase99_causal_primitives.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase99_causal_primitives.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase99_causal_primitives.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase99_causal_primitives.py --model qwen3 --exp 4
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
from itertools import combinations

from model_utils import load_model, get_layers, get_model_info, release_model


# ============================================================
# 标准任务Benchmark — 不再随机试prompt
# ============================================================
TASK_BENCHMARK = {
    "translation": {
        "qwen3": "{zh}的英文是",
        "glm4": "Translate {zh} to English:",
        "deepseek7b": "Translate {zh} to English:",
        "test_pairs": [
            ("猫", "cat"), ("狗", "dog"), ("书", "book"),
            ("水", "water"), ("火", "fire"), ("花", "flower"),
            ("鱼", "fish"), ("树", "tree"), ("鸟", "bird"),
            ("马", "horse"), ("铁", "iron"), ("金", "gold"),
            ("茶", "tea"), ("米", "rice"), ("血", "blood"),
            ("眼", "eye"), ("手", "hand"), ("风", "wind"),
            ("雪", "snow"), ("星", "star"),
        ],
        "control_pairs": [  # 补全任务(同prompt前缀)
            ("猫", "咪"), ("狗", "的"), ("书", "本"),
        ],
    },
    "retrieval": {
        "qwen3": "{subject}的{attribute}是",
        "glm4": "The {attribute} of {subject} is",
        "deepseek7b": "The {attribute} of {subject} is",
        "test_pairs": [
            (("法国", "首都"), "巴黎"),
            (("日本", "首都"), "东京"),
            (("中国", "首都"), "北京"),
            (("美国", "首都"), "华盛顿"),
            (("英国", "首都"), "伦敦"),
            (("地球", "卫星"), "月球"),
            (("水", "化学式"), "H2O"),
            (("太阳", "行星"), "八大"),
        ],
    },
    "constraint": {
        "qwen3": "{word}的第二个字是",
        "glm4": "The second character of {word} is",
        "deepseek7b": "The second character of {word} is",
        "test_pairs": [
            ("苹果", "果"), ("香蕉", "蕉"), ("葡萄", "萄"),
            ("老虎", "虎"), ("大象", "象"), ("熊猫", "猫"),
        ],
    },
}


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
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_token_id(tokenizer, text):
    """安全获取token ID"""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


def compute_translation_score(model, tokenizer, device, prompt, en_target):
    """计算翻译得分: en_target在logits中的概率"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)

    # 尝试多种编码形式
    en_candidates = [en_target, f" {en_target}", f"_{en_target}",
                     en_target.lower(), en_target.capitalize()]
    best_prob = 0.0
    for cand in en_candidates:
        cid = get_token_id(tokenizer, cand)
        if cid is not None:
            p = probs[cid].item()
            if p > best_prob:
                best_prob = p

    # 语义簇概率
    en_cluster = [en_target, f" {en_target}", en_target.capitalize(),
                  f" {en_target.capitalize()}", en_target.upper()]
    cluster_prob = 0.0
    for cand in en_cluster:
        cid = get_token_id(tokenizer, cand)
        if cid is not None:
            cluster_prob += probs[cid].item()

    del outputs
    return best_prob, cluster_prob


def compute_retrieval_score(model, tokenizer, device, prompt, answer):
    """计算检索得分"""
    return compute_translation_score(model, tokenizer, device, prompt, answer)


# ============================================================
# Exp 1: 因果必要性测试
# ============================================================
def exp1_causal_necessity(model_name):
    """
    核心方法论升级: 因果隔离

    之前的path patching只测了"添加source信息有多大影响"(sufficiency)
    本实验测"移除某组件→翻译是否崩塌"(necessity)

    如果 zero-ablate L26 attn → 翻译概率大幅下降
    而 zero-ablate L25 attn → 翻译概率不受影响
    则证明 L26 attn 因果必要

    这直接回答批判: "L26 attn=语言切换路由"到底是执行还是传播?
    """
    print(f"\n{'='*60}")
    print(f"Exp 1: 因果必要性测试 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    # 从model config获取n_heads
    if hasattr(model.config, 'num_attention_heads'):
        n_heads = model.config.num_attention_heads
    elif hasattr(model.config, 'n_heads'):
        n_heads = model.config.n_heads
    else:
        n_heads = info.d_model // 128  # 假设head_dim=128
    head_dim = info.d_model // n_heads
    print(f"  模型: {model_name}, 层数: {n_layers}, Heads: {n_heads}")

    task = TASK_BENCHMARK["translation"]
    prompt_template = task[model_name]
    test_pairs = task["test_pairs"][:10]  # 10个测试对

    # Step 1: Baseline翻译概率
    print("\n  Step 1: 计算baseline翻译概率...")
    baseline_probs = []
    baseline_clusters = []
    for zh, en in test_pairs:
        prompt = prompt_template.format(zh=zh)
        prob, cluster = compute_translation_score(model, tokenizer, device, prompt, en)
        baseline_probs.append(prob)
        baseline_clusters.append(cluster)
    baseline_mean = np.mean(baseline_probs)
    baseline_cluster_mean = np.mean(baseline_clusters)
    print(f"    Baseline: en_target_prob={baseline_mean:.4f}, cluster_prob={baseline_cluster_mean:.4f}")

    # Step 2: 逐层zero-ablate Attn
    print("\n  Step 2: 逐层zero-ablate Attn output...")
    attn_effects = {}

    # 采样关键层 (全层太慢，采样关键区间)
    key_layers = list(range(0, n_layers, 3))  # 每3层
    # 加上Phase 98发现的关键层附近
    if model_name == "qwen3":
        for l in [24, 25, 26, 27, 28, 29, 30, 31]:
            if l not in key_layers:
                key_layers.append(l)
    key_layers = sorted(set(key_layers))

    for layer_idx in key_layers:
        # Hook: zero out attention output
        def make_zero_hook(li):
            def zero_hook(module, input, output):
                # output是tuple, 第一个元素是attn output
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            return zero_hook

        # 找到attn layer
        layers = get_layers(model)
        attn_layer = layers[layer_idx].self_attn

        handle = attn_layer.register_forward_hook(make_zero_hook(layer_idx))

        # 测试翻译
        ablated_probs = []
        for zh, en in test_pairs:
            prompt = prompt_template.format(zh=zh)
            prob, _ = compute_translation_score(model, tokenizer, device, prompt, en)
            ablated_probs.append(prob)

        handle.remove()

        # 效果 = ablated后的概率 / baseline
        effect = np.mean(ablated_probs)
        relative_effect = effect / baseline_mean if baseline_mean > 0 else 0
        drop = 1 - relative_effect

        attn_effects[layer_idx] = {
            "ablated_prob": float(effect),
            "relative_to_baseline": float(relative_effect),
            "drop_fraction": float(drop),
        }

        if abs(drop) > 0.1:  # 只打印显著的
            print(f"    L{layer_idx} Attn zero-ablate: prob={effect:.4f} (baseline={baseline_mean:.4f}), drop={drop:.1%}")

    # Step 3: 逐层zero-ablate MLP
    print("\n  Step 3: 逐层zero-ablate MLP output...")
    mlp_effects = {}

    for layer_idx in key_layers:
        def make_zero_hook(li):
            def zero_hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            return zero_hook

        layers = get_layers(model)
        mlp_layer = layers[layer_idx].mlp

        handle = mlp_layer.register_forward_hook(make_zero_hook(layer_idx))

        ablated_probs = []
        for zh, en in test_pairs:
            prompt = prompt_template.format(zh=zh)
            prob, _ = compute_translation_score(model, tokenizer, device, prompt, en)
            ablated_probs.append(prob)

        handle.remove()

        effect = np.mean(ablated_probs)
        relative_effect = effect / baseline_mean if baseline_mean > 0 else 0
        drop = 1 - relative_effect

        mlp_effects[layer_idx] = {
            "ablated_prob": float(effect),
            "relative_to_baseline": float(relative_effect),
            "drop_fraction": float(drop),
        }

        if abs(drop) > 0.1:
            print(f"    L{layer_idx} MLP zero-ablate: prob={effect:.4f} (baseline={baseline_mean:.4f}), drop={drop:.1%}")

    # Step 4: 分析因果必要性
    print("\n  Step 4: 因果必要性分析...")

    # Attn: 哪些层zero-ablate导致翻译崩塌>50%?
    attn_necessary = {l: e for l, e in attn_effects.items() if e["drop_fraction"] > 0.5}
    mlp_necessary = {l: e for l, e in mlp_effects.items() if e["drop_fraction"] > 0.5}

    print(f"    Attn因果必要层(drop>50%): {sorted(attn_necessary.keys())}")
    print(f"    MLP因果必要层(drop>50%): {sorted(mlp_necessary.keys())}")

    # 找最关键的层
    if attn_effects:
        attn_max_layer = max(attn_effects, key=lambda l: attn_effects[l]["drop_fraction"])
        print(f"    Attn最关键层: L{attn_max_layer} (drop={attn_effects[attn_max_layer]['drop_fraction']:.1%})")
    if mlp_effects:
        mlp_max_layer = max(mlp_effects, key=lambda l: mlp_effects[l]["drop_fraction"])
        print(f"    MLP最关键层: L{mlp_max_layer} (drop={mlp_effects[mlp_max_layer]['drop_fraction']:.1%})")

    # ---- 保存 ----
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "baseline_en_prob": float(baseline_mean),
        "baseline_cluster_prob": float(baseline_cluster_mean),
        "attn_ablation_effects": {str(k): v for k, v in attn_effects.items()},
        "mlp_ablation_effects": {str(k): v for k, v in mlp_effects.items()},
        "attn_necessary_layers": {str(k): v for k, v in attn_necessary.items()},
        "mlp_necessary_layers": {str(k): v for k, v in mlp_necessary.items()},
        "key_conclusion": {
            "attn_max_drop_layer": int(attn_max_layer) if attn_effects else -1,
            "attn_max_drop": float(attn_effects[attn_max_layer]["drop_fraction"]) if attn_effects else 0,
            "mlp_max_drop_layer": int(mlp_max_layer) if mlp_effects else -1,
            "mlp_max_drop": float(mlp_effects[mlp_max_layer]["drop_fraction"]) if mlp_effects else 0,
        }
    }

    outpath = f"tests/glm5_temp/phase99_exp1_{model_name}_causal_necessity.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_serialize(output), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")

    release_model(model)
    return output


# ============================================================
# Exp 2: Hidden State语义子空间分析
# ============================================================
def exp2_hidden_state_semantic_subspace(model_name):
    """
    核心方法论升级: 从token层到表示层

    之前的振荡分析在logits层看top-token变化
    批判: "振荡可能只是logit几何，不是语义切换"

    本实验:
    1. 收集大量中文词和英文词在各层的hidden state
    2. 在每层训练zh/en线性分类器
    3. 对翻译prompt的hidden state，看它被分类为zh还是en
    4. 这直接回答: "hidden state层面是否存在语言切换"

    如果:
    - 早期层hidden state被分类为"中文"
    - 晚期层被分类为"英文"
    - 中间层有切换
    则: 语言切换是表示层面的，不是logit层面的

    如果:
    - 所有层hidden state都被分类为"中文"
    - 但晚期层logits输出英文
    则: 语言切换是decoder投影的结果，不是表示层面的
    """
    print(f"\n{'='*60}")
    print(f"Exp 2: Hidden State语义子空间 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  模型: {model_name}, 层数: {n_layers}")

    # Step 1: 收集中文和英文的hidden state
    print("\n  Step 1: 收集中/英文词的hidden state...")

    # 中文词 (单字/双字)
    zh_words = ["猫", "狗", "书", "水", "火", "花", "鱼", "树", "鸟", "马",
                "山", "河", "铁", "金", "茶", "米", "血", "眼", "手", "风",
                "雪", "星", "海", "石", "草", "人", "天", "地", "日", "月",
                "红", "蓝", "大", "小", "高", "低", "快", "慢", "好", "坏"]

    # 英文词
    en_words = ["cat", "dog", "book", "water", "fire", "flower", "fish", "tree",
                "bird", "horse", "mountain", "river", "iron", "gold", "tea",
                "rice", "blood", "eye", "hand", "wind", "snow", "star", "sea",
                "stone", "grass", "person", "sky", "earth", "sun", "moon",
                "red", "blue", "big", "small", "tall", "low", "fast", "slow",
                "good", "bad"]

    # 采样层
    sample_layers = sorted(set(list(range(0, n_layers, 2)) + [n_layers-1]))

    # 收集hidden states
    zh_hiddens = {l: [] for l in sample_layers}  # [n_words, d_model]
    en_hiddens = {l: [] for l in sample_layers}

    def collect_hiddens(words, label, storage):
        for word in words:
            inputs = tokenizer(word, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            for l in sample_layers:
                h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
                storage[l].append(h)

            del outputs

    print(f"    收集{len(zh_words)}个中文词...")
    collect_hiddens(zh_words, "zh", zh_hiddens)

    print(f"    收集{len(en_words)}个英文词...")
    collect_hiddens(en_words, "en", en_hiddens)

    # Step 2: 训练zh/en线性分类器
    print("\n  Step 2: 训练zh/en线性分类器...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    classifiers = {}
    scalers = {}
    classifier_accuracies = {}

    for l in sample_layers:
        X_zh = np.array(zh_hiddens[l])
        X_en = np.array(en_hiddens[l])
        X = np.vstack([X_zh, X_en])
        y = np.array([0]*len(X_zh) + [1]*len(X_en))  # 0=zh, 1=en

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_scaled, y)

        acc = clf.score(X_scaled, y)
        classifiers[l] = clf
        scalers[l] = scaler
        classifier_accuracies[l] = acc

        if l % 6 == 0 or l == n_layers - 1:
            print(f"    L{l}: 分类准确率={acc:.3f}")

    # Step 3: 对翻译prompt的hidden state做分类
    print("\n  Step 3: 对翻译prompt做zh/en分类...")

    task = TASK_BENCHMARK["translation"]
    prompt_template = task[model_name]
    test_pairs = task["test_pairs"][:10]

    translation_classifications = {l: [] for l in sample_layers}

    for zh, en in test_pairs:
        prompt = prompt_template.format(zh=zh)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for l in sample_layers:
            h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy().reshape(1, -1)
            h_scaled = scalers[l].transform(h)
            en_prob = classifiers[l].predict_proba(h_scaled)[0, 1]  # P(en)
            translation_classifications[l].append(en_prob)

        del outputs

    # Step 4: 分析语言切换
    print("\n  Step 4: Hidden state语言切换分析...")

    layer_en_probs = {}
    for l in sample_layers:
        mean_en = np.mean(translation_classifications[l])
        layer_en_probs[l] = mean_en
        if l % 6 == 0 or l == n_layers - 1 or abs(mean_en - 0.5) < 0.15:
            print(f"    L{l}: P(en)={mean_en:.3f} {'← 切换点!' if abs(mean_en - 0.5) < 0.15 else ''}")

    # 找切换点
    switch_layer = None
    for i in range(len(sample_layers) - 1):
        l1, l2 = sample_layers[i], sample_layers[i+1]
        p1, p2 = layer_en_probs[l1], layer_en_probs[l2]
        if p1 < 0.5 and p2 >= 0.5:
            # 线性插值找切换点
            frac = (0.5 - p1) / (p2 - p1) if p2 > p1 else 0
            switch_layer = l1 + frac * (l2 - l1)
            break

    if switch_layer is not None:
        switch_depth = switch_layer / n_layers
        print(f"\n  *** Hidden state切换层: L{switch_layer:.1f} (深度{switch_depth:.1%}) ***")
    else:
        # 可能一直<0.5或一直>0.5
        final_en = layer_en_probs[sample_layers[-1]]
        initial_en = layer_en_probs[sample_layers[0]]
        print(f"\n  无明确切换点. 初始P(en)={initial_en:.3f}, 最终P(en)={final_en:.3f}")

    # Step 5: 对比logits层切换和hidden state层切换
    print("\n  Step 5: 对比logits切换 vs hidden state切换...")

    # Logits层切换 (Phase 98的方法)
    logits_en_probs = {}
    for zh, en in test_pairs:
        prompt = prompt_template.format(zh=zh)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for l in sample_layers:
            # 用l层的hidden state做lm_head投影
            h = outputs.hidden_states[l][0, -1, :]
            # 需要找到lm_head
            if hasattr(model, 'lm_head'):
                logits = model.lm_head(h)
            elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                # 共享embedding
                logits = F.linear(h, model.model.embed_tokens.weight)
            else:
                continue

            probs = F.softmax(logits, dim=-1)

            # en cluster概率
            en_cluster_prob = 0.0
            en_candidates = [en, f" {en}", en.lower(), en.capitalize()]
            for cand in en_candidates:
                cid = get_token_id(tokenizer, cand)
                if cid is not None:
                    en_cluster_prob += probs[cid].item()

            if l not in logits_en_probs:
                logits_en_probs[l] = []
            logits_en_probs[l].append(en_cluster_prob)

        del outputs

    logits_layer_means = {l: np.mean(probs) for l, probs in logits_en_probs.items()}

    # 找logits层切换点
    logits_switch = None
    for i in range(len(sample_layers) - 1):
        l1, l2 = sample_layers[i], sample_layers[i+1]
        if l1 in logits_layer_means and l2 in logits_layer_means:
            if logits_layer_means[l1] < 0.3 and logits_layer_means[l2] >= 0.3:
                logits_switch = l2
                break

    print(f"    Hidden state切换层: L{switch_layer:.1f}" if switch_layer else "    Hidden state: 无切换")
    print(f"    Logits切换层: L{logits_switch}" if logits_switch else "    Logits: 无切换")

    if switch_layer and logits_switch:
        gap = logits_switch - switch_layer
        print(f"    差距: {gap:.1f}层")
        if gap > 3:
            print(f"    → 表示层切换早于logits层切换{gap:.0f}层！表示层先完成切换，logits层后体现")
        elif gap < -3:
            print(f"    → Logits层切换早于表示层！语言切换可能是decoder投影效应")
        else:
            print(f"    → 表示层和logits层基本同步切换")

    # ---- 保存 ----
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "n_zh_words": len(zh_words),
        "n_en_words": len(en_words),
        "classifier_accuracies": {str(l): float(a) for l, a in classifier_accuracies.items()},
        "hidden_state_en_prob": {str(l): float(p) for l, p in layer_en_probs.items()},
        "logits_en_prob": {str(l): float(p) for l, p in logits_layer_means.items()},
        "hidden_state_switch_layer": float(switch_layer) if switch_layer else None,
        "logits_switch_layer": int(logits_switch) if logits_switch else None,
        "key_conclusion": {
            "representation_level_switch": switch_layer is not None,
            "logits_level_switch": logits_switch is not None,
            "gap_layers": float(logits_switch - switch_layer) if (switch_layer and logits_switch) else None,
        }
    }

    outpath = f"tests/glm5_temp/phase99_exp2_{model_name}_semantic_subspace.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_serialize(output), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")

    release_model(model)
    return output


# ============================================================
# Exp 3: Head级因果中介
# ============================================================
def exp3_head_causal_mediation(model_name):
    """
    核心方法论升级: 从层级到head级因果分析

    之前只分析了层级的attn/MLP影响
    现在要找到: 哪些具体head是翻译的必要条件

    方法:
    1. 对关键层(Phase 98的L26, L31)的每个head做zero-ablate
    2. 测翻译概率的变化
    3. 如果某个head zero-ablate→翻译崩塌 → 这个head因果必要
    4. 同时测补全任务: 如果同一个head对补全也必要 → 不是翻译专用的

    这直接回答: "是否存在翻译专用的head"?
    """
    print(f"\n{'='*60}")
    print(f"Exp 3: Head级因果中介 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    # 从model config获取n_heads
    if hasattr(model.config, 'num_attention_heads'):
        n_heads = model.config.num_attention_heads
    elif hasattr(model.config, 'n_heads'):
        n_heads = model.config.n_heads
    else:
        n_heads = info.d_model // 128
    head_dim = info.d_model // n_heads
    print(f"  模型: {model_name}, 层数: {n_layers}, Heads: {n_heads}")

    task = TASK_BENCHMARK["translation"]
    prompt_template = task[model_name]
    test_pairs = task["test_pairs"][:8]  # 8个测试对

    # 翻译prompt
    trans_prompts = [prompt_template.format(zh=zh) for zh, en in test_pairs]
    # 补全prompt (不加"的英文是"，只输入中文词)
    comp_prompts = [zh for zh, en in test_pairs]

    # 要分析的层 — Phase 98的关键层
    target_layers = []
    if model_name == "qwen3":
        target_layers = [24, 26, 28, 30, 31, 34]
    elif model_name == "glm4":
        target_layers = [20, 25, 30, 35, 37, 39]
    else:
        target_layers = [10, 15, 20, 25, 30, 31]

    # Baseline
    print("\n  计算baseline...")
    baseline_trans = []
    baseline_comp = []
    for prompt in trans_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        # 翻译得分: top-1概率
        top1_prob = probs.max().item()
        baseline_trans.append(top1_prob)
        del outputs

    for prompt in comp_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        top1_prob = probs.max().item()
        baseline_comp.append(top1_prob)
        del outputs

    baseline_trans_mean = np.mean(baseline_trans)
    baseline_comp_mean = np.mean(baseline_comp)
    print(f"    Baseline翻译: {baseline_trans_mean:.4f}, 补全: {baseline_comp_mean:.4f}")

    # 逐层逐head ablate
    print("\n  逐层逐head zero-ablate...")
    head_effects = {}

    for layer_idx in target_layers:
        print(f"\n  Layer {layer_idx}...")
        layers = get_layers(model)
        attn = layers[layer_idx].self_attn

        head_effects[layer_idx] = {}

        for h in range(n_heads):
            # Hook: zero out specific head's output
            def make_head_zero_hook(li, hi, nh, hd):
                def hook(module, input, output):
                    # output shape: [batch, seq, d_model]
                    if isinstance(output, tuple):
                        out = output[0].clone()
                    else:
                        out = output.clone()

                    # Zero out this head's contribution
                    # Attention output = sum of head outputs
                    # Each head contributes head_dim columns
                    start = hi * hd
                    end = (hi + 1) * hd
                    out[:, :, start:end] = 0

                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out
                return hook

            handle = attn.register_forward_hook(make_head_zero_hook(layer_idx, h, n_heads, head_dim))

            # 测试翻译
            trans_probs = []
            for i, (zh, en) in enumerate(test_pairs):
                prompt = trans_prompts[i]
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits, dim=-1)

                # 翻译得分
                en_candidates = [en, f" {en}", en.lower(), en.capitalize()]
                best_prob = 0.0
                for cand in en_candidates:
                    cid = get_token_id(tokenizer, cand)
                    if cid is not None:
                        p = probs[cid].item()
                        if p > best_prob:
                            best_prob = p
                trans_probs.append(best_prob)
                del outputs

            # 测试补全
            comp_probs = []
            for prompt in comp_prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits, dim=-1)
                top1_prob = probs.max().item()
                comp_probs.append(top1_prob)
                del outputs

            handle.remove()

            trans_mean = np.mean(trans_probs)
            comp_mean = np.mean(comp_probs)

            trans_drop = (baseline_trans_mean - trans_mean) / baseline_trans_mean
            comp_drop = (baseline_comp_mean - comp_mean) / baseline_comp_mean

            # 翻译特异性 = 翻译drop - 补全drop
            specificity = trans_drop - comp_drop

            head_effects[layer_idx][h] = {
                "trans_drop": float(trans_drop),
                "comp_drop": float(comp_drop),
                "specificity": float(specificity),
                "trans_ablated_prob": float(trans_mean),
                "comp_ablated_prob": float(comp_mean),
            }

        # 报告该层的显著head
        significant = {h: e for h, e in head_effects[layer_idx].items()
                       if e["trans_drop"] > 0.1 or e["specificity"] > 0.05}
        if significant:
            top_heads = sorted(significant.items(), key=lambda x: x[1]["specificity"], reverse=True)[:3]
            for h, e in top_heads:
                print(f"    Head {h}: trans_drop={e['trans_drop']:.1%}, comp_drop={e['comp_drop']:.1%}, specificity={e['specificity']:.1%}")
        else:
            print(f"    无显著head")

    # Step 2: 汇总分析
    print("\n  汇总分析...")
    all_heads = []
    for l, heads in head_effects.items():
        for h, e in heads.items():
            all_heads.append({"layer": l, "head": h, **e})

    # 按翻译特异性排序
    by_specificity = sorted(all_heads, key=lambda x: x["specificity"], reverse=True)
    print(f"\n  Top-10翻译特异性head:")
    for i, h in enumerate(by_specificity[:10]):
        print(f"    {i+1}. L{h['layer']}:H{h['head']} — trans_drop={h['trans_drop']:.1%}, comp_drop={h['comp_drop']:.1%}, specificity={h['specificity']:.1%}")

    # 按翻译drop排序
    by_trans_drop = sorted(all_heads, key=lambda x: x["trans_drop"], reverse=True)
    print(f"\n  Top-10翻译必要性head:")
    for i, h in enumerate(by_trans_drop[:10]):
        print(f"    {i+1}. L{h['layer']}:H{h['head']} — trans_drop={h['trans_drop']:.1%}, comp_drop={h['comp_drop']:.1%}, specificity={h['specificity']:.1%}")

    # ---- 保存 ----
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "baseline_trans_prob": float(baseline_trans_mean),
        "baseline_comp_prob": float(baseline_comp_mean),
        "head_effects": {str(l): {str(h): e for h, e in heads.items()} for l, heads in head_effects.items()},
        "top_specificity_heads": [{"layer": h["layer"], "head": h["head"],
                                    "specificity": h["specificity"],
                                    "trans_drop": h["trans_drop"],
                                    "comp_drop": h["comp_drop"]}
                                   for h in by_specificity[:20]],
        "top_necessity_heads": [{"layer": h["layer"], "head": h["head"],
                                  "trans_drop": h["trans_drop"],
                                  "comp_drop": h["comp_drop"],
                                  "specificity": h["specificity"]}
                                 for h in by_trans_drop[:20]],
    }

    outpath = f"tests/glm5_temp/phase99_exp3_{model_name}_head_mediation.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_serialize(output), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")

    release_model(model)
    return output


# ============================================================
# Exp 4: 跨任务原语检测
# ============================================================
def exp4_cross_task_primitives(model_name):
    """
    核心方法论升级: 从单任务到跨任务原语检测

    批判指出: 需要区分"通用计算原语"和"任务专用电路"

    本实验:
    1. 对3种任务(翻译/检索/约束)做逐层zero-ablate
    2. 如果同一层对多种任务都必要 → 可能是通用原语
    3. 如果某层只对翻译必要 → 可能是翻译专用电路

    这直接回答: "是否存在有限计算原语集"?
    """
    print(f"\n{'='*60}")
    print(f"Exp 4: 跨任务原语检测 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  模型: {model_name}, 层数: {n_layers}")

    # 采样层
    sample_layers = sorted(set(list(range(0, n_layers, 3)) + [n_layers-1]))
    if model_name == "qwen3":
        for l in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]:
            if l not in sample_layers:
                sample_layers.append(l)
    sample_layers = sorted(set(sample_layers))

    # Step 1: 计算各任务baseline
    print("\n  Step 1: 计算各任务baseline...")

    # 翻译
    trans_task = TASK_BENCHMARK["translation"]
    trans_template = trans_task[model_name]
    trans_pairs = trans_task["test_pairs"][:8]

    trans_baselines = []
    for zh, en in trans_pairs:
        prompt = trans_template.format(zh=zh)
        prob, _ = compute_translation_score(model, tokenizer, device, prompt, en)
        trans_baselines.append(prob)
    trans_baseline = np.mean(trans_baselines)
    print(f"    翻译baseline: {trans_baseline:.4f}")

    # 检索
    retr_task = TASK_BENCHMARK["retrieval"]
    retr_template = retr_task[model_name]
    retr_pairs = retr_task["test_pairs"][:6]

    retr_baselines = []
    for (subject, attr), answer in retr_pairs:
        prompt = retr_template.format(subject=subject, attribute=attr)
        prob, _ = compute_retrieval_score(model, tokenizer, device, prompt, answer)
        retr_baselines.append(prob)
    retr_baseline = np.mean(retr_baselines)
    print(f"    检索baseline: {retr_baseline:.4f}")

    # 约束
    cons_task = TASK_BENCHMARK["constraint"]
    cons_template = cons_task[model_name]
    cons_pairs = cons_task["test_pairs"][:6]

    cons_baselines = []
    for word, char in cons_pairs:
        prompt = cons_template.format(word=word)
        prob, _ = compute_translation_score(model, tokenizer, device, prompt, char)
        cons_baselines.append(prob)
    cons_baseline = np.mean(cons_baselines)
    print(f"    约束baseline: {cons_baseline:.4f}")

    # Step 2: 逐层ablate MLP (之前的exp1已做过attn, 这里focus MLP)
    print("\n  Step 2: 逐层zero-ablate MLP，测三种任务...")
    task_effects = {"translation": {}, "retrieval": {}, "constraint": {}}

    for layer_idx in sample_layers:
        layers = get_layers(model)
        mlp = layers[layer_idx].mlp

        def make_zero_hook():
            def hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            return hook

        handle = mlp.register_forward_hook(make_zero_hook())

        # 翻译
        trans_probs = []
        for zh, en in trans_pairs:
            prompt = trans_template.format(zh=zh)
            prob, _ = compute_translation_score(model, tokenizer, device, prompt, en)
            trans_probs.append(prob)

        # 检索
        retr_probs = []
        for (subject, attr), answer in retr_pairs:
            prompt = retr_template.format(subject=subject, attribute=attr)
            prob, _ = compute_retrieval_score(model, tokenizer, device, prompt, answer)
            retr_probs.append(prob)

        # 约束
        cons_probs = []
        for word, char in cons_pairs:
            prompt = cons_template.format(word=word)
            prob, _ = compute_translation_score(model, tokenizer, device, prompt, char)
            cons_probs.append(prob)

        handle.remove()

        task_effects["translation"][layer_idx] = {
            "ablated_prob": float(np.mean(trans_probs)),
            "drop": float(1 - np.mean(trans_probs)/trans_baseline) if trans_baseline > 0 else 0,
        }
        task_effects["retrieval"][layer_idx] = {
            "ablated_prob": float(np.mean(retr_probs)),
            "drop": float(1 - np.mean(retr_probs)/retr_baseline) if retr_baseline > 0 else 0,
        }
        task_effects["constraint"][layer_idx] = {
            "ablated_prob": float(np.mean(cons_probs)),
            "drop": float(1 - np.mean(cons_probs)/cons_baseline) if cons_baseline > 0 else 0,
        }

        # 打印显著的
        any_significant = any(task_effects[t][layer_idx]["drop"] > 0.15 for t in task_effects)
        if any_significant:
            td = task_effects["translation"][layer_idx]["drop"]
            rd = task_effects["retrieval"][layer_idx]["drop"]
            cd = task_effects["constraint"][layer_idx]["drop"]
            print(f"    L{layer_idx}: trans_drop={td:.1%}, retr_drop={rd:.1%}, cons_drop={cd:.1%}")

    # Step 3: 分析原语vs专用电路
    print("\n  Step 3: 原语分析...")

    # 通用层: 对所有任务都必要(drop>20%)
    universal_layers = []
    for l in sample_layers:
        drops = [task_effects[t][l]["drop"] for t in task_effects]
        if all(d > 0.2 for d in drops):
            universal_layers.append(l)

    # 翻译专用层: 只对翻译必要
    translation_specific = []
    for l in sample_layers:
        td = task_effects["translation"][l]["drop"]
        other_drops = [task_effects[t][l]["drop"] for t in task_effects if t != "translation"]
        if td > 0.2 and all(d < 0.1 for d in other_drops):
            translation_specific.append(l)

    # 检索专用层
    retrieval_specific = []
    for l in sample_layers:
        rd = task_effects["retrieval"][l]["drop"]
        other_drops = [task_effects[t][l]["drop"] for t in task_effects if t != "retrieval"]
        if rd > 0.2 and all(d < 0.1 for d in other_drops):
            retrieval_specific.append(l)

    print(f"    通用原语层(drop>20% on all): {universal_layers}")
    print(f"    翻译专用层: {translation_specific}")
    print(f"    检索专用层: {retrieval_specific}")

    if universal_layers:
        print(f"\n    → 存在通用计算原语! MLP L{universal_layers}对多种任务都必要")
    if translation_specific:
        print(f"    → 存在翻译专用电路! L{translation_specific}只对翻译必要")

    # ---- 保存 ----
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "baselines": {
            "translation": float(trans_baseline),
            "retrieval": float(retr_baseline),
            "constraint": float(cons_baseline),
        },
        "task_effects": {
            task: {str(l): e for l, e in effects.items()}
            for task, effects in task_effects.items()
        },
        "universal_primitive_layers": universal_layers,
        "translation_specific_layers": translation_specific,
        "retrieval_specific_layers": retrieval_specific,
    }

    outpath = f"tests/glm5_temp/phase99_exp4_{model_name}_cross_task_primitives.json"
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
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    if args.exp == 1:
        exp1_causal_necessity(args.model)
    elif args.exp == 2:
        exp2_hidden_state_semantic_subspace(args.model)
    elif args.exp == 3:
        exp3_head_causal_mediation(args.model)
    elif args.exp == 4:
        exp4_cross_task_primitives(args.model)
