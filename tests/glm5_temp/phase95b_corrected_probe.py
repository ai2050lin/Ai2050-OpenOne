"""
Phase 95b: 因果语义干预 — 修正版探针实验
==========================================
Phase 95a的关键发现: 简单的翻译vs控制二分类探针全部acc=1.0
因为不同prompt的token序列本身就不同，探针只是在检测输入差异！

修正方案: 
  使用"同prompt不同目标"的探针设计
  例如: "___的英文是" 后面接不同中文词，探针检测的是hidden state中
        哪些方向编码了"目标英文词"的信息

  具体方法: 
  1. 同一prompt模板，不同目标 → 收集每层的hidden state
  2. 训练多分类探针: 给定h_l, 能否预测目标token?
  3. 与随机模型对比: 随机模型应该无法预测

Run:
  python tests/glm5_temp/phase95b_corrected_probe.py --model qwen3
  python tests/glm5_temp/phase95b_corrected_probe.py --model glm4
  python tests/glm5_temp/phase95b_corrected_probe.py --model deepseek7b
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import cross_val_score

from model_utils import load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS


# ============================================================
# 核心设计: 同模板不同目标的探针
# ============================================================
# 翻译测试: 同一模板"XX的英文是"，不同XX对应不同英文目标
TRANSLATION_PAIRS = [
    ("苹果", "apple"), ("猫", "cat"), ("狗", "dog"), ("书", "book"), ("水", "water"),
    ("火", "fire"), ("花", "flower"), ("鱼", "fish"), ("太阳", "sun"), ("月亮", "moon"),
    ("红色", "red"), ("蓝色", "blue"), ("绿色", "green"), ("白色", "white"), ("黑色", "black"),
    ("吃", "eat"), ("跑", "run"), ("看", "see"), ("听", "hear"), ("说", "say"),
    ("大", "big"), ("小", "small"), ("高", "tall"), ("快", "fast"), ("好", "good"),
    ("爱", "love"), ("时间", "time"), ("家", "home"), ("朋友", "friend"), ("学校", "school"),
]

# 首都测试: 同一模板"XX的首都是"，不同XX对应不同首都
CAPITAL_PAIRS = [
    ("法国", "巴黎"), ("中国", "北京"), ("日本", "东京"), ("英国", "伦敦"), ("德国", "柏林"),
    ("美国", "华盛顿"), ("意大利", "罗马"), ("俄罗斯", "莫斯科"), ("韩国", "首尔"), ("澳大利亚", "堪培拉"),
    ("巴西", "巴西利亚"), ("加拿大", "渥太华"), ("印度", "新德里"), ("埃及", "开罗"), ("泰国", "曼谷"),
]

# 反义词测试: 同一模板"XX的反义词是"，不同XX对应不同反义词
ANTONYM_PAIRS = [
    ("大", "小"), ("高", "矮"), ("热", "冷"), ("快", "慢"), ("好", "坏"),
    ("亮", "暗"), ("多", "少"), ("长", "短"), ("新", "旧"), ("强", "弱"),
    ("重", "轻"), ("硬", "软"), ("甜", "苦"), ("美", "丑"), ("胖", "瘦"),
]


def run_corrected_probe(model_name):
    print("=" * 70)
    print(f"Phase 95b: 修正版语义探针 — 同模板不同目标 ({model_name})")
    print("=" * 70)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    # ---- 收集hidden states ----
    # 对每种结构，收集同模板下不同目标的hidden states
    # 然后训练多分类探针: 给定h_l，能否预测目标token的类别?

    structures = {
        "translation": (TRANSLATION_PAIRS, "的英文是"),
        "capital": (CAPITAL_PAIRS, "的首都是"),
        "antonym": (ANTONYM_PAIRS, "的反义词是"),
    }

    all_probe_results = {}

    for struct_name, (pairs, template_suffix) in structures.items():
        print(f"\n{'='*50}")
        print(f"结构: {struct_name} (模板: XX{template_suffix})")
        print(f"{'='*50}")

        # 为每个目标token分配类别ID
        target_to_class = {}
        class_to_target = {}
        for idx, (source, target) in enumerate(pairs):
            if target not in target_to_class:
                target_to_class[target] = len(target_to_class)
                class_to_target[len(class_to_target)] = target

        n_classes = len(target_to_class)

        # 收集hidden states
        print(f"  收集hidden states ({len(pairs)} 样本, {n_classes} 类)...")
        hiddens = {l: [] for l in range(n_layers + 1)}
        labels = []

        for source, target in pairs:
            prompt = source + template_suffix
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            for l in range(n_layers + 1):
                h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
                hiddens[l].append(h)

            labels.append(target_to_class[target])
            del outputs

        labels = np.array(labels)

        # 训练多分类探针
        print(f"  训练多分类探针 ({n_classes} 类)...")
        probe_results = {}

        for l in range(n_layers + 1):
            X = np.array(hiddens[l])

            # PCA降维
            max_pca_dim = min(80, X.shape[0] - 1, X.shape[1])
            max_pca_dim = max(max_pca_dim, 5)
            if X.shape[1] > max_pca_dim:
                pca = PCA(n_components=max_pca_dim, random_state=42)
                X_reduced = pca.fit_transform(X)
                explained_var = pca.explained_variance_ratio_.sum()
            else:
                X_reduced = X
                explained_var = 1.0

            # 交叉验证
            if n_classes >= 3 and len(X_reduced) >= n_classes * 2:
                try:
                    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42,
                                            multi_class='multinomial')
                    # 使用留一法或5折CV
                    cv = min(5, len(X_reduced) // n_classes)
                    if cv >= 2:
                        cv_scores = cross_val_score(clf, X_reduced, labels, cv=cv, scoring='accuracy')
                        mean_acc = cv_scores.mean()
                        std_acc = cv_scores.std()
                    else:
                        clf.fit(X_reduced, labels)
                        mean_acc = accuracy_score(labels, clf.predict(X_reduced))
                        std_acc = 0.0
                except Exception as e:
                    mean_acc = 0.0
                    std_acc = 0.0
            else:
                mean_acc = 0.0
                std_acc = 0.0

            probe_results[l] = {
                "cv_mean_acc": float(mean_acc),
                "cv_std_acc": float(std_acc),
                "n_classes": n_classes,
                "n_samples": len(X),
                "explained_var": float(explained_var),
                "chance_level": float(1.0 / n_classes),
            }

        # 找到探针首次超过chance * 2的层
        chance = 1.0 / n_classes
        probe_emergence = None
        for l in range(n_layers + 1):
            if probe_results[l]["cv_mean_acc"] > chance * 2:
                probe_emergence = l
                break

        all_probe_results[struct_name] = {
            "probe_results": probe_results,
            "probe_emergence": probe_emergence,
            "n_classes": n_classes,
            "chance_level": chance,
        }

        # 打印关键层的结果
        print(f"\n  探针结果 (chance={chance:.3f}, {n_classes}类):")
        key_layers = list(range(0, n_layers + 1, max(1, n_layers // 12)))
        if n_layers not in key_layers:
            key_layers.append(n_layers)
        if probe_emergence is not None and probe_emergence not in key_layers:
            key_layers.append(probe_emergence)
        key_layers = sorted(set(key_layers))

        print(f"  {'层':>4} | {'CV acc':>8} | {'vs chance':>10}")
        print(f"  {'-'*30}")
        for l in key_layers:
            acc = probe_results[l]["cv_mean_acc"]
            std = probe_results[l]["cv_std_acc"]
            ratio = acc / chance if chance > 0 else 0
            marker = " ← 探针涌现" if l == probe_emergence else ""
            print(f"  L{l:3d} | {acc:8.3f}±{std:.3f} | {ratio:8.1f}x{marker}")

    # ---- 随机模型控制 ----
    print(f"\n{'='*50}")
    print("随机模型控制实验...")
    print(f"{'='*50}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(MODEL_CONFIGS[model_name]["path"])
    random_model = AutoModelForCausalLM.from_config(config).to(device).bfloat16()
    random_model.eval()

    # 只测试翻译结构
    pairs = TRANSLATION_PAIRS
    template_suffix = "的英文是"

    target_to_class = {}
    for source, target in pairs:
        if target not in target_to_class:
            target_to_class[target] = len(target_to_class)

    n_classes = len(target_to_class)
    chance = 1.0 / n_classes

    random_hiddens = {l: [] for l in range(n_layers + 1)}
    random_labels = []

    for source, target in pairs:
        prompt = source + template_suffix
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = random_model(**inputs, output_hidden_states=True)
        for l in range(n_layers + 1):
            h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
            random_hiddens[l].append(h)
        random_labels.append(target_to_class[target])
        del outputs

    random_labels = np.array(random_labels)

    random_probe_results = {}
    for l in range(n_layers + 1):
        X = np.array(random_hiddens[l])
        max_pca_dim = min(80, X.shape[0] - 1, X.shape[1])
        max_pca_dim = max(max_pca_dim, 5)
        if X.shape[1] > max_pca_dim:
            pca = PCA(n_components=max_pca_dim, random_state=42)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X

        try:
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42,
                                    multi_class='multinomial')
            cv = min(5, len(X_reduced) // n_classes)
            if cv >= 2:
                cv_scores = cross_val_score(clf, X_reduced, random_labels, cv=cv, scoring='accuracy')
                mean_acc = cv_scores.mean()
            else:
                clf.fit(X_reduced, random_labels)
                mean_acc = accuracy_score(random_labels, clf.predict(X_reduced))
        except:
            mean_acc = 0.0

        random_probe_results[l] = {"cv_mean_acc": float(mean_acc)}

    del random_model
    torch.cuda.empty_cache()
    gc.collect()

    # ---- 对比训练模型 vs 随机模型 ----
    print(f"\n{'='*70}")
    print("训练模型 vs 随机模型: 翻译探针对比")
    print(f"{'='*70}")

    trained_trans = all_probe_results["translation"]["probe_results"]
    print(f"  {'层':>4} | {'训练acc':>8} | {'随机acc':>8} | {'比值':>6}")
    print(f"  {'-'*40}")
    for l in range(0, n_layers + 1, max(1, n_layers // 12)):
        t_acc = trained_trans.get(l, {}).get("cv_mean_acc", 0)
        r_acc = random_probe_results.get(l, {}).get("cv_mean_acc", 0)
        ratio = t_acc / max(r_acc, 0.001)
        print(f"  L{l:3d} | {t_acc:8.3f} | {r_acc:8.3f} | {ratio:6.1f}x")

    # ---- 关键分析: 探针acc vs top-1涌现 ----
    # 还需要加载训练模型做top-1分析
    print(f"\n{'='*70}")
    print("探针信息检测层 vs top-1涌现层")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = model.lm_head.weight.data.float()

    for struct_name, data in all_probe_results.items():
        probe_emergence = data["probe_emergence"]
        chance = data["chance_level"]

        # 获取top-1涌现层
        if struct_name == "translation":
            test_pairs = TRANSLATION_PAIRS[:15]
        elif struct_name == "capital":
            test_pairs = CAPITAL_PAIRS
        elif struct_name == "antonym":
            test_pairs = ANTONYM_PAIRS

        template_suffix = structures[struct_name][1]
        top1_layers = []

        for source, target in test_pairs:
            prompt = source + template_suffix
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            target_ids = set()
            for v in [target, f" {target}", f'"{target}']:
                try:
                    ids = tokenizer.encode(v, add_special_tokens=False)
                    target_ids.update(ids)
                except:
                    pass

            for l in range(n_layers + 1):
                h = outputs.hidden_states[l][0, -1, :].float()
                logits = h @ W_U.T
                probs = torch.softmax(logits, dim=-1)
                best_rank = min([((probs > probs[tid].item()).sum().item() + 1)
                                for tid in target_ids if 0 <= tid < probs.shape[0]], default=9999)
                if best_rank == 1:
                    top1_layers.append(l)
                    break

            del outputs
            torch.cuda.empty_cache()

        top1_mean = np.mean(top1_layers) if top1_layers else None
        top1_std = np.std(top1_layers) if top1_layers else None

        # 找到探针首次超过chance*3的层
        probe_emergence_3x = None
        for l in range(n_layers + 1):
            if data["probe_results"][l]["cv_mean_acc"] > chance * 3:
                probe_emergence_3x = l
                break

        # 找到探针首次超过50%的层
        probe_emergence_50 = None
        for l in range(n_layers + 1):
            if data["probe_results"][l]["cv_mean_acc"] > 0.5:
                probe_emergence_50 = l
                break

        print(f"\n  {struct_name}:")
        print(f"    chance level: {chance:.3f}")
        print(f"    探针>2x chance: L{probe_emergence}")
        print(f"    探针>3x chance: L{probe_emergence_3x}")
        print(f"    探针>50%:      L{probe_emergence_50}")
        print(f"    top-1涌现层:    {top1_mean:.1f} ± {top1_std:.1f}" if top1_mean else "    top-1涌现层:    None")

        if probe_emergence is not None and top1_mean is not None:
            gap = top1_mean - probe_emergence
            print(f"    *** 探针检测比top-1早 {gap:.1f} 层 ***")

    release_model(model)

    # ---- 保存完整结果 ----
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "structures": all_probe_results,
        "random_model_translation": random_probe_results,
    }

    out_path = f"tests/glm5_temp/phase95b_{model_name}_corrected_probe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()

    run_corrected_probe(args.model)

    gc.collect()
    torch.cuda.empty_cache()
    print("\n完成!")
