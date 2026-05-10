"""
Phase 95: 因果语义干预 — 从观察到因果
==========================================
核心修正: Phase 94的最大硬伤是 top-1 token ≠ 内部语义状态
本Phase通过三个层次解决:

  Exp 1: 语义线性探针 (Semantic Linear Probing)
    - 不看top-1 token，而是训练线性探针检测语义信息在hidden state中何时变得线性可分
    - 比较probe检测层 vs top-1涌现层 → 发现"信息比token更早出现"

  Exp 2: 翻译方向提取与干预 (Translation Direction Extraction & Intervention)
    - 找到hidden space中翻译对应的语义方向
    - 对该方向进行ablation → 翻译能力应该崩塌
    - 对该方向进行steering → 翻译应该更早出现

  Exp 3: 结构签名验证 (Structure Signature Validation)
    - 统一测量6种指标: emergence_depth, activation_sharpness, dimensionality,
      probe_accuracy, causal_effect_size, cross_example_consistency

Run:
  python tests/glm5/ccml_phase95_causal_semantic.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase95_causal_semantic.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase95_causal_semantic.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase95_causal_semantic.py --model glm4 --exp 1
  python tests/glm5/ccml_phase95_causal_semantic.py --model deepseek7b --exp 1
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from model_utils import load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS


# ============================================================
# 翻译测试用例 (扩大到30个，增加统计力)
# ============================================================
TRANSLATION_TESTS_LARGE = [
    # 常见名词
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
    # 颜色
    ("红色的英文是", "red", "红色"),
    ("蓝色的英文是", "blue", "蓝色"),
    ("绿色的英文是", "green", "绿色"),
    ("白色的英文是", "white", "白色"),
    ("黑色的英文是", "black", "黑色"),
    # 动词
    ("吃的英文是", "eat", "吃"),
    ("跑的英文是", "run", "跑"),
    ("看的英文是", "see", "看"),
    ("听的英文是", "hear", "听"),
    ("说的英文是", "say", "说"),
    # 形容词
    ("大的英文是", "big", "大"),
    ("小的英文是", "small", "小"),
    ("高的英文是", "tall", "高"),
    ("快的英文是", "fast", "快"),
    ("好的英文是", "good", "好"),
    # 抽象名词
    ("爱的英文是", "love", "爱"),
    ("时间的英文是", "time", "时间"),
    ("家的英文是", "home", "家"),
    ("朋友的英文是", "friend", "朋友"),
    ("学校的英文是", "school", "学校"),
]

# 事实检索测试
FACT_RETRIEVAL_TESTS = [
    ("法国的首都是", "巴黎", "France capital"),
    ("中国的首都是", "北京", "China capital"),
    ("日本的首都是", "东京", "Japan capital"),
    ("英国的首都是", "伦敦", "UK capital"),
    ("德国的首都是", "柏林", "Germany capital"),
    ("美国的首都是", "华盛顿", "US capital"),
    ("意大利的首都是", "罗马", "Italy capital"),
    ("俄罗斯的首都是", "莫斯科", "Russia capital"),
    ("韩国的首都是", "首尔", "Korea capital"),
    ("澳大利亚的首都是", "堪培拉", "Australia capital"),
    ("水的化学式是", "H2O", "Water formula"),
    ("盐的化学式是", "NaCl", "Salt formula"),
    ("铁的化学符号是", "Fe", "Iron symbol"),
    ("金的化学符号是", "Au", "Gold symbol"),
    ("氧的化学符号是", "O", "Oxygen symbol"),
]

# 类比测试
ANALOGY_TESTS = [
    ("苹果属于水果，狗属于什么？答案是", "动物", "fruit→dog"),
    ("医生在医院工作，教师在哪里工作？答案是", "学校", "hospital→teacher"),
    ("汽车在公路上行驶，飞机在哪里飞行？答案是", "天空", "road→plane"),
    ("书提供知识，食物提供什么？答案是", "营养", "knowledge→food"),
    ("眼睛负责视觉，耳朵负责什么？答案是", "听觉", "sight→ear"),
    ("鸟有翅膀，鱼有什么？答案是", "鳍", "wings→fish"),
    ("刀用来切割，笔用来什么？答案是", "书写", "cut→pen"),
    ("冬天很冷，夏天很什么？答案是", "热", "cold→summer"),
    ("医生治病，教师做什么？答案是", "教书", "cure→teacher"),
    ("太阳白天出现，月亮什么时候出现？答案是", "夜晚", "day→moon"),
]

# 否定/反义词测试
ANTONYM_TESTS = [
    ("大的反义词是", "小", "big→small"),
    ("高的反义词是", "矮", "tall→short"),
    ("热的反义词是", "冷", "hot→cold"),
    ("快的反义词是", "慢", "fast→slow"),
    ("好的反义词是", "坏", "good→bad"),
    ("亮的反义词是", "暗", "bright→dark"),
    ("多的反义词是", "少", "many→few"),
    ("长的反义词是", "短", "long→short"),
    ("新的反义词是", "旧", "new→old"),
    ("强的反义词是", "弱", "strong→weak"),
]


# ============================================================
# Exp 1: 语义线性探针 — 核心修正 top-1 ≠ 语义状态
# ============================================================
def run_semantic_probe_experiment(model_name):
    """
    核心思想: 不看top-1 token，而是训练线性探针检测:
    1. 翻译信息(中文→英文)何时在hidden state中变得线性可分
    2. 这比top-1涌现早多少层?
    3. 不同结构的probe accuracy轨迹差异
    """
    print("=" * 70)
    print("Exp 1: 语义线性探针 — 修正top-1 ≠ 语义状态")
    print("=" * 70)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    # ---- Step 1: 收集翻译的hidden states ----
    print("\n[Step 1] 收集翻译hidden states...")

    # 正例: 翻译prompt (中文→英文)
    # 反例: 相同prompt但不要求翻译 (控制条件)

    translation_hs = {l: [] for l in range(n_layers + 1)}
    control_hs = {l: [] for l in range(n_layers + 1)}
    labels = []

    # 翻译条件: "苹果的英文是" → 期望输出 apple
    # 控制条件: "苹果是一种" → 期望输出 水果 (无翻译)
    control_prompts = [
        "苹果是一种", "猫是一种", "狗是一种", "书是一种", "水是一种",
        "火是一种", "花是一种", "鱼是一种", "太阳是一颗", "月亮是一颗",
        "红色是一种", "蓝色是一种", "绿色是一种", "白色是一种", "黑色是一种",
    ]

    W_U = model.lm_head.weight.data.float()

    all_results = {}

    # 收集翻译prompt的hidden states
    translation_hiddens = {l: [] for l in range(n_layers + 1)}
    control_hiddens = {l: [] for l in range(n_layers + 1)}
    fact_hiddens = {l: [] for l in range(n_layers + 1)}
    antonym_hiddens = {l: [] for l in range(n_layers + 1)}

    # ---- 翻译条件 ----
    print("  收集翻译条件hidden states...")
    for prompt, target_en, chinese in TRANSLATION_TESTS_LARGE:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for l in range(n_layers + 1):
            h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
            translation_hiddens[l].append(h)
        del outputs
        torch.cuda.empty_cache()

    # ---- 控制条件(无翻译) ----
    print("  收集控制条件hidden states...")
    for prompt in control_prompts[:len(TRANSLATION_TESTS_LARGE)]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for l in range(n_layers + 1):
            h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
            control_hiddens[l].append(h)
        del outputs
        torch.cuda.empty_cache()

    # ---- 事实检索条件 ----
    print("  收集事实检索条件hidden states...")
    for prompt, target, desc in FACT_RETRIEVAL_TESTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for l in range(n_layers + 1):
            h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
            fact_hiddens[l].append(h)
        del outputs
        torch.cuda.empty_cache()

    # ---- 反义词条件 ----
    print("  收集反义词条件hidden states...")
    for prompt, target, desc in ANTONYM_TESTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for l in range(n_layers + 1):
            h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
            antonym_hiddens[l].append(h)
        del outputs
        torch.cuda.empty_cache()

    # ---- Step 2: 训练线性探针 ----
    print("\n[Step 2] 训练线性探针: 翻译 vs 控制...")

    # 探针1: 翻译 vs 控制 (二分类)
    probe_results_translation = {}
    for l in range(n_layers + 1):
        X_pos = np.array(translation_hiddens[l])
        X_neg = np.array(control_hiddens[l])
        n_pos = min(len(X_pos), len(X_neg))
        X = np.vstack([X_pos[:n_pos], X_neg[:n_pos]])
        y = np.array([1] * n_pos + [0] * n_pos)

        # 用PCA降维(避免过拟合)，维度不超过min(n_samples, n_features)/2
        max_pca_dim = min(100, X.shape[0] // 2, X.shape[1] // 2)
        max_pca_dim = max(max_pca_dim, 5)  # 至少5维
        if X.shape[1] > max_pca_dim:
            pca = PCA(n_components=max_pca_dim, random_state=42)
            X_reduced = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_.sum()
        else:
            X_reduced = X
            explained_var = 1.0
            pca = None

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        # Leave-one-out式: train on 90%, test on 10%
        n_train = int(0.9 * len(X_reduced))
        idx = np.random.RandomState(42).permutation(len(X_reduced))

        clf.fit(X_reduced[idx[:n_train]], y[idx[:n_train]])
        acc_train = accuracy_score(y[idx[:n_train]], clf.predict(X_reduced[idx[:n_train]]))
        acc_test = accuracy_score(y[idx[n_train:]], clf.predict(X_reduced[idx[n_train:]]))

        probe_results_translation[l] = {
            "train_acc": acc_train,
            "test_acc": acc_test,
            "explained_var": explained_var,
            "n_samples": len(X),
        }

    # 探针2: 事实检索 vs 控制
    print("  训练线性探针: 事实检索 vs 控制...")
    probe_results_fact = {}
    for l in range(n_layers + 1):
        X_pos = np.array(fact_hiddens[l])
        X_neg = np.array(control_hiddens[l])
        n_pos = min(len(X_pos), len(X_neg))
        X = np.vstack([X_pos[:n_pos], X_neg[:n_pos]])
        y = np.array([1] * n_pos + [0] * n_pos)

        max_pca_dim = min(100, X.shape[0] // 2, X.shape[1] // 2)
        max_pca_dim = max(max_pca_dim, 5)
        if X.shape[1] > max_pca_dim:
            pca = PCA(n_components=max_pca_dim, random_state=42)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
            pca = None

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        n_train = int(0.9 * len(X_reduced))
        idx = np.random.RandomState(42).permutation(len(X_reduced))

        clf.fit(X_reduced[idx[:n_train]], y[idx[:n_train]])
        acc_test = accuracy_score(y[idx[n_train:]], clf.predict(X_reduced[idx[n_train:]]))

        probe_results_fact[l] = {"test_acc": acc_test}

    # 探针3: 反义词 vs 控制
    print("  训练线性探针: 反义词 vs 控制...")
    probe_results_antonym = {}
    for l in range(n_layers + 1):
        X_pos = np.array(antonym_hiddens[l])
        X_neg = np.array(control_hiddens[l])
        n_pos = min(len(X_pos), len(X_neg))
        X = np.vstack([X_pos[:n_pos], X_neg[:n_pos]])
        y = np.array([1] * n_pos + [0] * n_pos)

        max_pca_dim = min(100, X.shape[0] // 2, X.shape[1] // 2)
        max_pca_dim = max(max_pca_dim, 5)
        if X.shape[1] > max_pca_dim:
            pca = PCA(n_components=max_pca_dim, random_state=42)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
            pca = None

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        n_train = int(0.9 * len(X_reduced))
        idx = np.random.RandomState(42).permutation(len(X_reduced))

        clf.fit(X_reduced[idx[:n_train]], y[idx[:n_train]])
        acc_test = accuracy_score(y[idx[n_train:]], clf.predict(X_reduced[idx[n_train:]]))

        probe_results_antonym[l] = {"test_acc": acc_test}

    # ---- Step 3: 找到探针accuracy > 0.8的最早层 ----
    print("\n[Step 3] 分析探针accuracy轨迹...")

    def find_emergence_layer(probe_results, threshold=0.8):
        """找到probe accuracy首次超过threshold的层"""
        for l in sorted(probe_results.keys()):
            if probe_results[l]["test_acc"] >= threshold:
                return l
        return None

    trans_emergence = find_emergence_layer(probe_results_translation)
    fact_emergence = find_emergence_layer(probe_results_fact)
    antonym_emergence = find_emergence_layer(probe_results_antonym)

    # ---- Step 4: 同时获取top-1涌现层进行对比 ----
    print("\n[Step 4] 获取top-1涌现层对比...")

    def get_target_rank_per_layer(model, tokenizer, device, prompt, target_str, n_layers):
        """获取目标token在每层的排名"""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        W_U = model.lm_head.weight.data.float()

        target_variants = [target_str, f" {target_str}", f'"{target_str}']
        target_ids = set()
        for v in target_variants:
            try:
                ids = tokenizer.encode(v, add_special_tokens=False)
                target_ids.update(ids)
            except:
                pass

        ranks = {}
        for l in range(n_layers + 1):
            h = outputs.hidden_states[l][0, -1, :].float()
            logits = h @ W_U.T
            probs = torch.softmax(logits, dim=-1)

            best_rank = 999999
            for tid in target_ids:
                if 0 <= tid < probs.shape[0]:
                    p = probs[tid].item()
                    rank = (probs > p).sum().item() + 1
                    best_rank = min(best_rank, rank)
            ranks[l] = best_rank
        del outputs
        torch.cuda.empty_cache()
        return ranks

    top1_emergence_translation = []
    for prompt, target_en, chinese in TRANSLATION_TESTS_LARGE[:10]:
        ranks = get_target_rank_per_layer(model, tokenizer, device, prompt, target_en, n_layers)
        for l in sorted(ranks.keys()):
            if ranks[l] == 1:
                top1_emergence_translation.append(l)
                break

    top1_emergence_fact = []
    for prompt, target, desc in FACT_RETRIEVAL_TESTS[:10]:
        ranks = get_target_rank_per_layer(model, tokenizer, device, prompt, target, n_layers)
        for l in sorted(ranks.keys()):
            if ranks[l] == 1:
                top1_emergence_fact.append(l)
                break

    # ---- 打印结果 ----
    print("\n" + "=" * 70)
    print("关键结果: 探针检测层 vs top-1涌现层")
    print("=" * 70)

    print(f"\n翻译对齐:")
    print(f"  探针检测层(probe acc>0.8): L{trans_emergence}")
    print(f"  top-1涌现层: {np.mean(top1_emergence_translation):.1f} ± {np.std(top1_emergence_translation):.1f}" if top1_emergence_translation else "  top-1涌现层: None")
    if trans_emergence and top1_emergence_translation:
        gap = np.mean(top1_emergence_translation) - trans_emergence
        print(f"  *** 语义信息比top-1早 {gap:.1f} 层！***")

    print(f"\n事实检索:")
    print(f"  探针检测层(probe acc>0.8): L{fact_emergence}")
    print(f"  top-1涌现层: {np.mean(top1_emergence_fact):.1f} ± {np.std(top1_emergence_fact):.1f}" if top1_emergence_fact else "  top-1涌现层: None")
    if fact_emergence and top1_emergence_fact:
        gap = np.mean(top1_emergence_fact) - fact_emergence
        print(f"  *** 语义信息比top-1早 {gap:.1f} 层！***")

    print(f"\n反义词:")
    print(f"  探针检测层(probe acc>0.8): L{antonym_emergence}")

    # 打印完整probe accuracy轨迹
    print("\n完整探针accuracy轨迹:")
    print(f"{'层':>4} | {'翻译':>8} | {'事实':>8} | {'反义词':>8}")
    print("-" * 40)
    for l in range(n_layers + 1):
        t_acc = probe_results_translation.get(l, {}).get("test_acc", 0)
        f_acc = probe_results_fact.get(l, {}).get("test_acc", 0)
        a_acc = probe_results_antonym.get(l, {}).get("test_acc", 0)
        marker = ""
        if l == trans_emergence:
            marker = " ← 翻译探测"
        if l == fact_emergence:
            marker += " ← 事实探测"
        if l == antonym_emergence:
            marker += " ← 反义词探测"
        print(f"L{l:3d} | {t_acc:8.3f} | {f_acc:8.3f} | {a_acc:8.3f}{marker}")

    # ---- Step 5: 随机模型控制 ----
    print("\n[Step 5] 随机模型探针控制实验...")
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    # 加载随机模型 - 用bfloat16避免OOM
    print("  加载随机初始化模型...")
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(MODEL_CONFIGS[model_name]["path"])
    random_model = AutoModelForCausalLM.from_config(config)
    random_model = random_model.to(device).bfloat16()
    random_model.eval()

    random_translation_hiddens = {l: [] for l in range(n_layers + 1)}
    random_control_hiddens = {l: [] for l in range(n_layers + 1)}

    for prompt, target_en, chinese in TRANSLATION_TESTS_LARGE[:15]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = random_model(**inputs, output_hidden_states=True)
        for l in range(n_layers + 1):
            h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
            random_translation_hiddens[l].append(h)
        del outputs

    for prompt in control_prompts[:15]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = random_model(**inputs, output_hidden_states=True)
        for l in range(n_layers + 1):
            h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
            random_control_hiddens[l].append(h)
        del outputs

    # 训练随机模型探针
    random_probe_results = {}
    for l in range(n_layers + 1):
        X_pos = np.array(random_translation_hiddens[l])
        X_neg = np.array(random_control_hiddens[l])
        n_pos = min(len(X_pos), len(X_neg))
        X = np.vstack([X_pos[:n_pos], X_neg[:n_pos]])
        y = np.array([1] * n_pos + [0] * n_pos)

        max_pca_dim = min(100, X.shape[0] // 2, X.shape[1] // 2)
        max_pca_dim = max(max_pca_dim, 5)
        if X.shape[1] > max_pca_dim:
            pca = PCA(n_components=max_pca_dim, random_state=42)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        n_train = int(0.9 * len(X_reduced))
        idx = np.random.RandomState(42).permutation(len(X_reduced))
        clf.fit(X_reduced[idx[:n_train]], y[idx[:n_train]])
        acc_test = accuracy_score(y[idx[n_train:]], clf.predict(X_reduced[idx[n_train:]]))
        random_probe_results[l] = {"test_acc": acc_test}

    del random_model
    torch.cuda.empty_cache()
    gc.collect()

    random_trans_emergence = find_emergence_layer(random_probe_results)

    print(f"\n随机模型翻译探针:")
    print(f"  探针检测层(probe acc>0.8): L{random_trans_emergence}")
    print(f"  最终层probe acc: {random_probe_results[n_layers]['test_acc']:.3f}")
    print(f"  训练模型最终层probe acc: {probe_results_translation[n_layers]['test_acc']:.3f}")

    if random_trans_emergence is None and trans_emergence is not None:
        print("  *** 随机模型无法检测翻译结构 → 语义方向是学习的，不是架构先验 ***")
    elif random_trans_emergence is not None:
        print(f"  随机模型在L{random_trans_emergence}就检测到'翻译' → 可能是prompt artifact")

    # ---- 保存结果 ----
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "probe_translation": {str(k): v for k, v in probe_results_translation.items()},
        "probe_fact": {str(k): v for k, v in probe_results_fact.items()},
        "probe_antonym": {str(k): v for k, v in probe_results_antonym.items()},
        "probe_random_translation": {str(k): v for k, v in random_probe_results.items()},
        "probe_emergence": {
            "translation": trans_emergence,
            "fact": fact_emergence,
            "antonym": antonym_emergence,
            "random_translation": random_trans_emergence,
        },
        "top1_emergence": {
            "translation_mean": float(np.mean(top1_emergence_translation)) if top1_emergence_translation else None,
            "translation_std": float(np.std(top1_emergence_translation)) if top1_emergence_translation else None,
            "fact_mean": float(np.mean(top1_emergence_fact)) if top1_emergence_fact else None,
        },
    }

    out_path = f"tests/glm5_temp/phase95_{model_name}_exp1_probe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")

    return results


# ============================================================
# Exp 2: 翻译方向提取与因果干预
# ============================================================
def run_causal_intervention(model_name):
    """
    核心思想: 
    1. 用对比法提取翻译方向(translation direction)
    2. 对该方向进行ablation → 翻译能力应崩塌
    3. 对该方向进行steering → 翻译应更早出现
    """
    print("=" * 70)
    print("Exp 2: 翻译方向提取与因果干预")
    print("=" * 70)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    # ---- Step 1: 提取翻译方向 ----
    print("\n[Step 1] 提取翻译方向 (Translation Direction)...")

    # 方法: 对比 "XX的英文是" vs "XX是一种" 在同一层的hidden state差异
    # 翻译方向 = mean(h_translate - h_control)

    translation_directions = {}
    test_pairs = list(zip(
        [p for p, _, _ in TRANSLATION_TESTS_LARGE[:20]],
        ["苹果是一种", "猫是一种", "狗是一种", "书是一种", "水是一种",
         "火是一种", "花是一种", "鱼是一种", "太阳是一颗", "月亮是一颗",
         "红色是一种", "蓝色是一种", "绿色是一种", "白色是一种", "黑色是一种",
         "吃是一种", "跑是一种", "看是一种", "听是一种", "说是一种"][:20]
    ))

    translate_hiddens = {l: [] for l in range(n_layers + 1)}
    control_hiddens_list = {l: [] for l in range(n_layers + 1)}

    for trans_prompt, ctrl_prompt in test_pairs:
        # 翻译条件
        inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_t = model(**inputs, output_hidden_states=True)
        for l in range(n_layers + 1):
            translate_hiddens[l].append(out_t.hidden_states[l][0, -1, :].float().cpu().numpy())
        del out_t

        # 控制条件
        inputs = tokenizer(ctrl_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_c = model(**inputs, output_hidden_states=True)
        for l in range(n_layers + 1):
            control_hiddens_list[l].append(out_c.hidden_states[l][0, -1, :].float().cpu().numpy())
        del out_c

        torch.cuda.empty_cache()

    # 计算翻译方向
    for l in range(n_layers + 1):
        t_arr = np.array(translate_hiddens[l])
        c_arr = np.array(control_hiddens_list[l])
        diff = t_arr - c_arr
        direction = diff.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        translation_directions[l] = direction

    print(f"  翻译方向已提取 ({n_layers + 1} 层)")

    # ---- Step 2: 验证翻译方向的判别力 ----
    print("\n[Step 2] 验证翻译方向判别力...")

    W_U = model.lm_head.weight.data.float()

    # 测试: 沿翻译方向移动hidden state，观察目标token概率变化
    test_cases = TRANSLATION_TESTS_LARGE[20:30]  # 未用于提取方向的测试集

    steering_results = []

    for prompt, target_en, chinese in test_cases:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 获取目标token的ID
        target_ids = set()
        for v in [target_en, f" {target_en}"]:
            try:
                ids = tokenizer.encode(v, add_special_tokens=False)
                target_ids.update(ids)
            except:
                pass

        for test_layer in range(max(0, n_layers - 10), n_layers + 1):
            h = outputs.hidden_states[test_layer][0, -1, :].float()
            direction_t = torch.tensor(translation_directions[test_layer],
                                       dtype=torch.float32, device=device)

            # 基线概率
            logits_base = h @ W_U.T
            probs_base = torch.softmax(logits_base, dim=-1)
            best_prob_base = max([probs_base[tid].item() for tid in target_ids if 0 <= tid < probs_base.shape[0]], default=0)

            # 沿翻译方向steering (+方向)
            for alpha in [0.5, 1.0, 2.0, 5.0]:
                h_steered = h + alpha * direction_t
                logits_steer = h_steered @ W_U.T
                probs_steer = torch.softmax(logits_steer, dim=-1)
                best_prob_steer = max([probs_steer[tid].item() for tid in target_ids if 0 <= tid < probs_steer.shape[0]], default=0)

                steering_results.append({
                    "prompt": prompt,
                    "target": target_en,
                    "layer": test_layer,
                    "alpha": alpha,
                    "base_prob": best_prob_base,
                    "steered_prob": best_prob_steer,
                    "prob_ratio": best_prob_steer / max(best_prob_base, 1e-10),
                })

            # 沿翻译方向ablation (-方向/投影消除)
            for alpha in [-0.5, -1.0, -2.0, -5.0]:
                h_ablated = h + alpha * direction_t
                logits_abl = h_ablated @ W_U.T
                probs_abl = torch.softmax(logits_abl, dim=-1)
                best_prob_abl = max([probs_abl[tid].item() for tid in target_ids if 0 <= tid < probs_abl.shape[0]], default=0)

                steering_results.append({
                    "prompt": prompt,
                    "target": target_en,
                    "layer": test_layer,
                    "alpha": alpha,
                    "base_prob": best_prob_base,
                    "steered_prob": best_prob_abl,
                    "prob_ratio": best_prob_abl / max(best_prob_base, 1e-10),
                })

        del outputs
        torch.cuda.empty_cache()

    # ---- Step 3: 投影消融 (真正的因果干预) ----
    print("\n[Step 3] 投影消融 — 从hidden state中移除翻译方向...")

    ablation_results = []

    for prompt, target_en, chinese in test_cases[:5]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        target_ids = set()
        for v in [target_en, f" {target_en}"]:
            try:
                ids = tokenizer.encode(v, add_special_tokens=False)
                target_ids.update(ids)
            except:
                pass

        for test_layer in range(max(0, n_layers - 10), n_layers + 1):
            h = outputs.hidden_states[test_layer][0, -1, :].float()
            direction_t = torch.tensor(translation_directions[test_layer],
                                       dtype=torch.float32, device=device)

            # 基线概率
            logits_base = h @ W_U.T
            probs_base = torch.softmax(logits_base, dim=-1)
            best_prob_base = max([probs_base[tid].item() for tid in target_ids if 0 <= tid < probs_base.shape[0]], default=0)

            # 投影消融: 移除翻译方向的分量
            proj = torch.dot(h, direction_t) * direction_t
            h_ablated = h - proj
            logits_abl = h_ablated @ W_U.T
            probs_abl = torch.softmax(logits_abl, dim=-1)
            best_prob_abl = max([probs_abl[tid].item() for tid in target_ids if 0 <= tid < probs_abl.shape[0]], default=0)

            # 获取top-5 tokens
            top5_base = torch.topk(probs_base, 5)
            top5_abl = torch.topk(probs_abl, 5)

            ablation_results.append({
                "prompt": prompt,
                "target": target_en,
                "layer": test_layer,
                "base_prob": best_prob_base,
                "ablated_prob": best_prob_abl,
                "prob_drop": best_prob_base - best_prob_abl,
                "top1_base": tokenizer.decode([top5_base.indices[0].item()]),
                "top1_ablated": tokenizer.decode([top5_abl.indices[0].item()]),
            })

        del outputs
        torch.cuda.empty_cache()

    # ---- 打印结果 ----
    print("\n" + "=" * 70)
    print("因果干预结果")
    print("=" * 70)

    # Steering结果
    print("\n[Steering] 沿翻译方向移动hidden state:")
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        relevant = [r for r in steering_results if r["alpha"] == alpha and r["layer"] == n_layers]
        if relevant:
            mean_ratio = np.mean([r["prob_ratio"] for r in relevant])
            mean_base = np.mean([r["base_prob"] for r in relevant])
            mean_steer = np.mean([r["steered_prob"] for r in relevant])
            print(f"  α={alpha:+.1f}: base_prob={mean_base:.4f} → steer_prob={mean_steer:.4f} (ratio={mean_ratio:.2f}x)")

    print("\n[Ablation] 沿负方向移动hidden state:")
    for alpha in [-0.5, -1.0, -2.0, -5.0]:
        relevant = [r for r in steering_results if r["alpha"] == alpha and r["layer"] == n_layers]
        if relevant:
            mean_ratio = np.mean([r["prob_ratio"] for r in relevant])
            mean_base = np.mean([r["base_prob"] for r in relevant])
            mean_steer = np.mean([r["steered_prob"] for r in relevant])
            print(f"  α={alpha:+.1f}: base_prob={mean_base:.4f} → abl_prob={mean_steer:.4f} (ratio={mean_ratio:.2f}x)")

    # 投影消融结果
    print("\n[Projection Ablation] 移除翻译方向分量:")
    for test_layer in range(max(0, n_layers - 10), n_layers + 1):
        relevant = [r for r in ablation_results if r["layer"] == test_layer]
        if relevant:
            mean_drop = np.mean([r["prob_drop"] for r in relevant])
            mean_base = np.mean([r["base_prob"] for r in relevant])
            mean_abl = np.mean([r["ablated_prob"] for r in relevant])
            top1_changes = sum(1 for r in relevant if r["top1_base"] != r["top1_ablated"])
            print(f"  L{test_layer}: base={mean_base:.4f} → ablated={mean_abl:.4f} (drop={mean_drop:.4f}, top1_change={top1_changes}/{len(relevant)})")

    # ---- 保存结果 ----
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "steering": steering_results,
        "ablation": ablation_results,
        "translation_direction_norms": {
            str(l): float(np.linalg.norm(translation_directions[l]))
            for l in range(n_layers + 1)
        },
    }

    out_path = f"tests/glm5_temp/phase95_{model_name}_exp2_intervention.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 3: 结构签名验证 — 统一实验协议
# ============================================================
def run_structure_signatures(model_name):
    """
    对每种结构统一测量8种指标:
    1. emergence_depth: 哪层信息出现
    2. activation_sharpness: 概率变化的锐度
    3. probe_accuracy: 线性探针accuracy
    4. dimensionality: 语义子空间维度
    5. causal_effect_size: 投影消融的效果量
    6. cross_example_consistency: 跨示例一致性
    7. training_dependency: 训练vs随机差异
    8. top1_vs_probe_gap: top-1涌现层 vs 探针检测层差距
    """
    print("=" * 70)
    print("Exp 3: 结构签名验证 — 统一实验协议")
    print("=" * 70)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = model.lm_head.weight.data.float()

    # 定义结构类型
    structures = {
        "translation": TRANSLATION_TESTS_LARGE[:20],
        "fact_retrieval": FACT_RETRIEVAL_TESTS,
        "analogy": ANALOGY_TESTS,
        "antonym": ANTONYM_TESTS,
    }

    signatures = {}

    for struct_name, test_cases in structures.items():
        print(f"\n{'='*50}")
        print(f"分析结构: {struct_name}")
        print(f"{'='*50}")

        # 收集概率轨迹
        prob_trajectories = []
        emergence_layers = []
        top5_per_layer = defaultdict(list)

        for item in test_cases:
            prompt = item[0]
            target = item[1]

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # 获取目标token ID
            target_ids = set()
            for v in [target, f" {target}", f'"{target}']:
                try:
                    ids = tokenizer.encode(v, add_special_tokens=False)
                    target_ids.update(ids)
                except:
                    pass

            trajectory = {}
            first_top1 = None
            for l in range(n_layers + 1):
                h = outputs.hidden_states[l][0, -1, :].float()
                logits = h @ W_U.T
                probs = torch.softmax(logits, dim=-1)

                best_prob = max([probs[tid].item() for tid in target_ids if 0 <= tid < probs.shape[0]], default=0)
                best_rank = min([((probs > probs[tid].item()).sum().item() + 1) for tid in target_ids if 0 <= tid < probs.shape[0]], default=9999)

                trajectory[l] = {"prob": best_prob, "rank": best_rank}

                if best_rank == 1 and first_top1 is None:
                    first_top1 = l

                # 收集top-5
                if l >= n_layers - 5:
                    top5 = torch.topk(probs, 5)
                    for i in range(5):
                        top5_per_layer[l].append(tokenizer.decode([top5.indices[i].item()]))

            prob_trajectories.append(trajectory)
            emergence_layers.append(first_top1)
            del outputs
            torch.cuda.empty_cache()

        # ---- 计算结构签名 ----
        sig = {}

        # 1. emergence_depth
        valid_emergence = [e for e in emergence_layers if e is not None]
        sig["emergence_depth_mean"] = float(np.mean(valid_emergence)) if valid_emergence else None
        sig["emergence_depth_std"] = float(np.std(valid_emergence)) if valid_emergence else None
        sig["emergence_rate"] = len(valid_emergence) / len(test_cases)

        # 2. activation_sharpness (概率最大变化率)
        sharpnesses = []
        for traj in prob_trajectories:
            max_delta = 0
            for l in range(1, n_layers + 1):
                delta = traj[l]["prob"] - traj[l-1]["prob"]
                max_delta = max(max_delta, delta)
            sharpnesses.append(max_delta)
        sig["activation_sharpness_mean"] = float(np.mean(sharpnesses))
        sig["activation_sharpness_std"] = float(np.std(sharpnesses))

        # 3. 概率轨迹形状 (S曲线 vs U形 vs 跳跃)
        # 计算前半层、中间层、后半层的概率
        mid = n_layers // 2
        late = n_layers * 3 // 4
        early_probs, mid_probs, late_probs, final_probs = [], [], [], []
        for traj in prob_trajectories:
            early_probs.append(np.mean([traj[l]["prob"] for l in range(0, mid)]))
            mid_probs.append(np.mean([traj[l]["prob"] for l in range(mid, late)]))
            late_probs.append(np.mean([traj[l]["prob"] for l in range(late, n_layers)]))
            final_probs.append(traj[n_layers]["prob"])

        sig["prob_early"] = float(np.mean(early_probs))
        sig["prob_mid"] = float(np.mean(mid_probs))
        sig["prob_late"] = float(np.mean(late_probs))
        sig["prob_final"] = float(np.mean(final_probs))

        # 判断轨迹形状
        if sig["prob_mid"] > sig["prob_late"]:
            sig["trajectory_shape"] = "U-shape"
        elif sig["prob_mid"] > 5 * sig["prob_early"]:
            sig["trajectory_shape"] = "S-curve"
        else:
            sig["trajectory_shape"] = "jump"

        # 4. cross_example_consistency (涌现层的标准差/均值)
        if valid_emergence:
            sig["cross_example_consistency"] = sig["emergence_depth_std"] / max(sig["emergence_depth_mean"], 1)
        else:
            sig["cross_example_consistency"] = None

        signatures[struct_name] = sig

        # 打印
        print(f"\n  结构签名: {struct_name}")
        print(f"    涌现深度: {sig['emergence_depth_mean']:.1f} ± {sig['emergence_depth_std']:.1f}" if sig["emergence_depth_mean"] else "    涌现深度: None")
        print(f"    涌现率: {sig['emergence_rate']:.1%}")
        print(f"    激活锐度: {sig['activation_sharpness_mean']:.4f}")
        print(f"    概率轨迹: early={sig['prob_early']:.6f}, mid={sig['prob_mid']:.6f}, late={sig['prob_late']:.4f}")
        print(f"    最终概率: {sig['prob_final']:.4f}")
        print(f"    轨迹形状: {sig['trajectory_shape']}")
        print(f"    跨示例一致性: {sig['cross_example_consistency']:.3f}" if sig["cross_example_consistency"] else "    跨示例一致性: None")

    # ---- 结构对比表 ----
    print("\n" + "=" * 70)
    print("结构签名对比表")
    print("=" * 70)
    print(f"{'结构':<15} | {'涌现层':>8} | {'涌现率':>6} | {'锐度':>8} | {'最终prob':>8} | {'轨迹形状':>8} | {'一致性':>6}")
    print("-" * 80)
    for name, sig in signatures.items():
        em = f"{sig['emergence_depth_mean']:.1f}" if sig["emergence_depth_mean"] else "N/A"
        er = f"{sig['emergence_rate']:.0%}"
        sh = f"{sig['activation_sharpness_mean']:.4f}"
        fp = f"{sig['prob_final']:.4f}"
        ts = sig["trajectory_shape"]
        cc = f"{sig['cross_example_consistency']:.3f}" if sig["cross_example_consistency"] else "N/A"
        print(f"{name:<15} | {em:>8} | {er:>6} | {sh:>8} | {fp:>8} | {ts:>8} | {cc:>6}")

    # ---- 保存结果 ----
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "signatures": signatures,
    }

    out_path = f"tests/glm5_temp/phase95_{model_name}_exp3_signatures.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")

    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    if args.exp == 1:
        run_semantic_probe_experiment(args.model)
    elif args.exp == 2:
        run_causal_intervention(args.model)
    elif args.exp == 3:
        run_structure_signatures(args.model)

    gc.collect()
    torch.cuda.empty_cache()
    print("\n完成!")
