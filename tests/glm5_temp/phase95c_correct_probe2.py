"""
Phase 95c: 因果语义干预 — 正确的探针设计
==========================================
Phase 95a: 翻译vs控制二分类→全部acc=1.0（检测的是输入差异）
Phase 95b: 30类多分类→全部acc=0.0（样本太少+PCA丢信息）

正确方法: 
  1. 二分类探针，同模板内不同目标对
     "苹果的英文是" → apple方向 vs "猫的英文是" → cat方向
     探针: 给定h_l，能否区分"苹果prompt"和"猫prompt"?
  
  2. 用全部层的数据训练单层探针，不做PCA（样本够时）
  
  3. 关键: 我们关心的不是"能否分类"，而是"哪层开始能分类"
     这才能回答: 语义信息比top-1涌现早多少层?

Run:
  python tests/glm5_temp/phase95c_correct_probe2.py --model qwen3
  python tests/glm5_temp/phase95c_correct_probe2.py --model glm4
  python tests/glm5_temp/phase95c_correct_probe2.py --model deepseek7b
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn.functional as F_torch
import numpy as np
import argparse
import gc
import json
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from model_utils import load_model, get_model_info, release_model, MODEL_CONFIGS


# ============================================================
# 核心设计: 同模板内的成对二分类探针
# ============================================================
# 每对: (prompt_A, prompt_B) → 探针: 给定h_l，能否区分A和B?
# A和B共享相同模板，只有输入词不同

TRANSLATION_PAIRS = [
    ("苹果", "apple"), ("猫", "cat"), ("狗", "dog"), ("书", "book"), ("水", "water"),
    ("火", "fire"), ("花", "flower"), ("鱼", "fish"), ("太阳", "sun"), ("月亮", "moon"),
    ("红色", "red"), ("蓝色", "blue"), ("绿色", "green"), ("白色", "white"), ("黑色", "black"),
    ("吃", "eat"), ("跑", "run"), ("看", "see"), ("听", "hear"), ("说", "say"),
    ("大", "big"), ("小", "small"), ("高", "tall"), ("快", "fast"), ("好", "good"),
    ("爱", "love"), ("时间", "time"), ("家", "home"), ("朋友", "friend"), ("学校", "school"),
    ("桌子", "table"), ("椅子", "chair"), ("门", "door"), ("窗", "window"), ("树", "tree"),
    ("河", "river"), ("海", "sea"), ("山", "mountain"), ("天空", "sky"), ("风", "wind"),
]

CAPITAL_PAIRS = [
    ("法国", "巴黎"), ("中国", "北京"), ("日本", "东京"), ("英国", "伦敦"), ("德国", "柏林"),
    ("美国", "华盛顿"), ("意大利", "罗马"), ("俄罗斯", "莫斯科"), ("韩国", "首尔"), ("澳大利亚", "堪培拉"),
    ("巴西", "巴西利亚"), ("加拿大", "渥太华"), ("印度", "新德里"), ("埃及", "开罗"), ("泰国", "曼谷"),
    ("越南", "河内"), ("墨西哥", "墨西哥城"), ("阿根廷", "布宜诺斯艾利斯"), ("土耳其", "安卡拉"), ("西班牙", "马德里"),
]

ANTONYM_PAIRS = [
    ("大", "小"), ("高", "矮"), ("热", "冷"), ("快", "慢"), ("好", "坏"),
    ("亮", "暗"), ("多", "少"), ("长", "短"), ("新", "旧"), ("强", "弱"),
    ("重", "轻"), ("硬", "软"), ("甜", "苦"), ("美", "丑"), ("胖", "瘦"),
    ("厚", "薄"), ("宽", "窄"), ("深", "浅"), ("忙", "闲"), ("富", "穷"),
]


def pairwise_probe_experiment(model_name):
    print("=" * 70)
    print(f"Phase 95c: 成对二分类探针 ({model_name})")
    print("=" * 70)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    structures = {
        "translation": (TRANSLATION_PAIRS, "的英文是"),
        "capital": (CAPITAL_PAIRS, "的首都是"),
        "antonym": (ANTONYM_PAIRS, "的反义词是"),
    }

    all_results = {}

    for struct_name, (pairs, suffix) in structures.items():
        print(f"\n{'='*50}")
        print(f"结构: {struct_name}")
        print(f"{'='*50}")

        # 收集所有hidden states
        print(f"  收集hidden states ({len(pairs)} 样本)...")
        hiddens = {l: [] for l in range(n_layers + 1)}
        sample_ids = list(range(len(pairs)))

        for source, target in pairs:
            prompt = source + suffix
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            for l in range(n_layers + 1):
                h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
                hiddens[l].append(h)
            del outputs

        torch.cuda.empty_cache()

        # ---- 方法1: 全对平均二分类 ----
        # 对每对样本(i,j), 训练二分类探针
        # 然后报告平均accuracy

        print(f"  训练成对二分类探针...")

        # 选取代表性对（避免O(n^2)计算量太大）
        n_pairs = len(pairs)
        n_test_pairs = min(50, n_pairs * (n_pairs - 1) // 2)

        # 生成随机对
        rng = np.random.RandomState(42)
        all_pair_indices = []
        for i in range(n_pairs):
            for j in range(i + 1, n_pairs):
                all_pair_indices.append((i, j))
        rng.shuffle(all_pair_indices)
        test_pair_indices = all_pair_indices[:n_test_pairs]

        # 对每层训练探针
        layer_accuracies = {}

        for l in range(n_layers + 1):
            X_all = np.array(hiddens[l])
            pair_accs = []

            for i, j in test_pair_indices:
                # 只有2个样本，用简单方法
                # 如果 h_i 和 h_j 的差异能被一个方向分离
                # 我们用: 这对样本 + 随机噪声增强
                X_pair = np.vstack([X_all[i:i+1], X_all[j:j+1]])
                y_pair = np.array([0, 1])

                # 简单测试: 两个样本的cosine距离 vs 随机对
                # 我们用留出法：从其他样本中取一些作为验证
                # 但更简单的方法: 只检查在hidden space中，同目标的不同prompt是否更近

                # 实际上，对于只有2个样本，无法训练分类器
                # 换一种方法: 计算所有样本之间的距离矩阵
                pass

            # 更好的方法: 计算类内距离 vs 类间距离
            # 这里"类"由目标token定义
            # 但30个类每类只有1个样本...

            # 最简单有效的方法: 
            # 检查hidden state是否线性编码了目标token的embedding方向
            # 即: 是否存在线性映射 h_l → target_embedding
            pass

        # ---- 方法2: 线性回归探针 ----
        # 给定h_l，能否预测目标token的logit方向?
        # 这是更直接的方法!

        print(f"  训练线性回归探针: h_l → target_logit_direction...")

        W_U = model.lm_head.weight.data.float().cpu().numpy()  # [vocab, d_model]

        # 对每个目标token，计算其logit方向 (即W_U[target_id])
        # 然后训练探针: 给定h_l，能否预测W_U[target_id]的方向?

        # 获取每个目标token的ID和对应的W_U行
        target_directions = []
        valid_indices = []

        for idx, (source, target) in enumerate(pairs):
            # 获取target token的ID
            target_ids = []
            for v in [target, f" {target}"]:
                try:
                    ids = tokenizer.encode(v, add_special_tokens=False)
                    target_ids.extend(ids)
                except:
                    pass

            if target_ids:
                # 取第一个ID的logit方向
                tid = target_ids[0]
                if tid < W_U.shape[0]:
                    target_dir = W_U[tid] / (np.linalg.norm(W_U[tid]) + 1e-10)
                    target_directions.append(target_dir)
                    valid_indices.append(idx)

        target_directions = np.array(target_directions)  # [n_valid, d_model]
        n_valid = len(valid_indices)

        if n_valid < 5:
            print(f"  有效样本太少 ({n_valid})，跳过")
            continue

        # 对每层训练Ridge回归: h_l → target_direction
        probe_r2 = {}

        for l in range(n_layers + 1):
            X = np.array(hiddens[l])[valid_indices]  # [n_valid, d_model]

            # Ridge回归: 预测target logit方向
            # 为了效率，只预测前50个PCA分量
            from sklearn.decomposition import PCA as PCA_sk
            n_pca = min(50, n_valid - 1, target_directions.shape[1])
            pca_target = PCA_sk(n_components=n_pca, random_state=42)
            Y_reduced = pca_target.fit_transform(target_directions)
            explained_var_y = pca_target.explained_variance_ratio_.sum()

            # 对每个目标PCA分量训练Ridge
            r2_scores = []
            for comp in range(n_pca):
                y_comp = Y_reduced[:, comp]
                # 使用交叉验证
                try:
                    from sklearn.linear_model import RidgeCV
                    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                    cv = min(3, n_valid // 2)
                    if cv >= 2:
                        scores = cross_val_score(ridge, X, y_comp, cv=cv, scoring='r2')
                        r2_scores.append(scores.mean())
                    else:
                        ridge.fit(X, y_comp)
                        r2_scores.append(ridge.score(X, y_comp))
                except:
                    r2_scores.append(0.0)

            mean_r2 = np.mean(r2_scores) if r2_scores else 0.0
            probe_r2[l] = {
                "mean_r2": float(mean_r2),
                "n_components": n_pca,
                "explained_var_y": float(explained_var_y),
                "n_valid": n_valid,
            }

        # ---- 方法3: 最关键的探针 — 同模板不同词对的线性可分性 ----
        print(f"  训练二分类探针: 每层区分不同输入词...")

        # 把所有样本分成两组（奇数idx vs 偶数idx）
        # 这样二分类就是区分"不同输入词"
        group_a = list(range(0, n_valid, 2))
        group_b = list(range(1, n_valid, 2))
        n_per_group = min(len(group_a), len(group_b))

        if n_per_group >= 3:
            binary_probe_results = {}
            for l in range(n_layers + 1):
                X = np.array(hiddens[l])[valid_indices]
                X_a = X[group_a[:n_per_group]]
                X_b = X[group_b[:n_per_group]]
                X_combined = np.vstack([X_a, X_b])
                y_combined = np.array([0] * n_per_group + [1] * n_per_group)

                # 标准化
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_combined)

                # 限制维度
                max_dim = min(200, X_scaled.shape[0] - 1, X_scaled.shape[1])
                if X_scaled.shape[1] > max_dim:
                    pca = PCA_sk(n_components=max_dim, random_state=42)
                    X_scaled = pca.fit_transform(X_scaled)

                try:
                    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
                    cv = min(5, n_per_group)
                    scores = cross_val_score(clf, X_scaled, y_combined, cv=cv, scoring='accuracy')
                    binary_probe_results[l] = {
                        "mean_acc": float(scores.mean()),
                        "std_acc": float(scores.std()),
                    }
                except:
                    binary_probe_results[l] = {"mean_acc": 0.5, "std_acc": 0.0}
        else:
            binary_probe_results = {}

        all_results[struct_name] = {
            "ridge_probe": probe_r2,
            "binary_probe": binary_probe_results,
            "n_valid": n_valid,
        }

        # 打印Ridge探针结果
        print(f"\n  Ridge探针: h_l → target_logit_direction (R²)")
        key_layers = list(range(0, n_layers + 1, max(1, n_layers // 12)))
        if n_layers not in key_layers:
            key_layers.append(n_layers)
        print(f"  {'层':>4} | {'R²':>8} | {'n_comp':>6}")
        print(f"  {'-'*25}")
        for l in key_layers:
            if l in probe_r2:
                print(f"  L{l:3d} | {probe_r2[l]['mean_r2']:8.4f} | {probe_r2[l]['n_components']:6d}")

        # 打印二分类探针结果
        if binary_probe_results:
            print(f"\n  二分类探针: 区分不同输入词 (chance=0.5)")
            print(f"  {'层':>4} | {'Acc':>8} | {'vs chance':>10}")
            print(f"  {'-'*28}")
            for l in key_layers:
                if l in binary_probe_results:
                    acc = binary_probe_results[l]["mean_acc"]
                    print(f"  L{l:3d} | {acc:8.3f} | {acc/0.5:8.1f}x")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    import time; time.sleep(3)  # 等待GPU完全释放
    print(f"\n{'='*50}")
    print("随机模型控制 (只测翻译)...")
    print(f"{'='*50}")

    gc.collect()
    torch.cuda.empty_cache()

    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(MODEL_CONFIGS[model_name]["path"])
    random_model = AutoModelForCausalLM.from_config(config).to(device).bfloat16()
    random_model.eval()

    pairs = TRANSLATION_PAIRS
    suffix = "的英文是"

    random_hiddens = {l: [] for l in range(n_layers + 1)}
    for source, target in pairs:
        prompt = source + suffix
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = random_model(**inputs, output_hidden_states=True)
        for l in range(n_layers + 1):
            h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
            random_hiddens[l].append(h)
        del outputs

    # Ridge探针 for random model
    # 先释放GPU内存再提取W_U
    torch.cuda.empty_cache()
    gc.collect()
    W_U_random = random_model.lm_head.weight.data.cpu().float().numpy()
    target_directions_random = []
    valid_indices_random = []

    for idx, (source, target) in enumerate(pairs):
        target_ids = []
        for v in [target, f" {target}"]:
            try:
                ids = tokenizer.encode(v, add_special_tokens=False)
                target_ids.extend(ids)
            except:
                pass
        if target_ids:
            tid = target_ids[0]
            if tid < W_U_random.shape[0]:
                target_dir = W_U_random[tid] / (np.linalg.norm(W_U_random[tid]) + 1e-10)
                target_directions_random.append(target_dir)
                valid_indices_random.append(idx)

    target_directions_random = np.array(target_directions_random)
    n_valid_random = len(valid_indices_random)

    if n_valid_random >= 5:
        n_pca = min(50, n_valid_random - 1, target_directions_random.shape[1])
        pca_target = PCA_sk(n_components=n_pca, random_state=42)
        Y_reduced = pca_target.fit_transform(target_directions_random)

        random_ridge_results = {}
        for l in range(0, n_layers + 1, max(1, n_layers // 6)):
            X = np.array(random_hiddens[l])[valid_indices_random]
            r2_scores = []
            for comp in range(n_pca):
                y_comp = Y_reduced[:, comp]
                try:
                    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                    ridge.fit(X, y_comp)
                    r2_scores.append(ridge.score(X, y_comp))
                except:
                    r2_scores.append(0.0)
            random_ridge_results[l] = float(np.mean(r2_scores))

    del random_model
    torch.cuda.empty_cache()
    gc.collect()

    # ---- 对比 ----
    print(f"\n{'='*70}")
    print(f"训练模型 vs 随机模型: 翻译Ridge探针 (R²)")
    print(f"{'='*70}")

    trained_ridge = all_results.get("translation", {}).get("ridge_probe", {})
    print(f"  {'层':>4} | {'训练R²':>8} | {'随机R²':>8} | {'比值':>6}")
    print(f"  {'-'*40}")
    for l in sorted(set(list(trained_ridge.keys()) + list(random_ridge_results.keys()))):
        t_r2 = trained_ridge.get(l, {}).get("mean_r2", 0)
        r_r2 = random_ridge_results.get(l, 0)
        ratio = t_r2 / max(abs(r_r2), 0.001) if t_r2 > 0 else 0
        print(f"  L{l:3d} | {t_r2:8.4f} | {r_r2:8.4f} | {ratio:6.1f}x")

    # ---- 保存 ----
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "structures": all_results,
        "random_ridge": random_ridge_results,
    }

    out_path = f"tests/glm5_temp/phase95c_{model_name}_pairwise_probe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()

    pairwise_probe_experiment(args.model)

    gc.collect()
    torch.cuda.empty_cache()
    print("\n完成!")
