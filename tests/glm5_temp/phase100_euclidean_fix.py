"""
Phase 100 关键补充: 解决余弦相似度饱和问题
==========================================
Exp 2/3发现: L9以后所有余弦相似度→1.0，无法区分语义距离

原因: 高维空间中，归一化后的向量几乎都指向同一方向
      余弦相似度在L9+完全饱和，不能衡量语义差异

解决方案: 
  1. 用欧氏距离替代余弦相似度
  2. 用PCA子空间投影分析
  3. 用"语义判别距离": 同类距离 vs 跨类距离的比值

Run:
  python tests/glm5_temp/phase100_euclidean_fix.py --model qwen3
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests', 'glm5'))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from collections import defaultdict

from model_utils import load_model, get_layers, get_model_info, release_model

SEMANTIC_CATEGORIES = {
    "动物": ["猫", "狗", "鱼", "鸟", "马", "牛", "羊", "猪", "鸡", "虎"],
    "食物": ["米", "茶", "肉", "面", "饼", "菜", "果", "糖", "酒", "奶"],
    "自然": ["水", "火", "风", "雪", "星", "月", "日", "山", "河", "海"],
    "身体": ["眼", "手", "头", "足", "心", "耳", "口", "鼻", "指", "骨"],
    "颜色": ["红", "白", "黑", "绿", "蓝", "黄", "紫", "灰", "橙", "粉"],
}

EN_WORDS = {
    "动物": ["cat", "dog", "fish", "bird", "horse", "cow", "sheep", "pig", "chicken", "tiger"],
    "食物": ["rice", "tea", "meat", "noodle", "cake", "vegetable", "fruit", "sugar", "wine", "milk"],
    "自然": ["water", "fire", "wind", "snow", "star", "moon", "sun", "mountain", "river", "sea"],
    "身体": ["eye", "hand", "head", "foot", "heart", "ear", "mouth", "nose", "finger", "bone"],
    "颜色": ["red", "white", "black", "green", "blue", "yellow", "purple", "gray", "orange", "pink"],
}

TRANSLATION_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("书", "book"),
    ("水", "water"), ("火", "fire"), ("花", "flower"),
    ("鱼", "fish"), ("树", "tree"), ("鸟", "bird"),
    ("马", "horse"), ("铁", "iron"), ("金", "gold"),
    ("茶", "tea"), ("米", "rice"), ("血", "blood"),
]


def get_last_token_hidden(model, input_ids, device):
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hiddens = []
    for l in range(len(outputs.hidden_states)):
        h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
        hiddens.append(h)
    return hiddens


def run_euclidean_analysis(model_name):
    print(f"\n{'='*60}")
    print(f"欧氏距离语义分析 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    # 收集各类别的hidden state
    zh_cat_hiddens = {}
    en_cat_hiddens = {}

    for cat_name, zh_words in SEMANTIC_CATEGORIES.items():
        zh_cat_hiddens[cat_name] = defaultdict(list)
        for word in zh_words:
            inputs = tokenizer(word, return_tensors="pt").to(device)
            hiddens = get_last_token_hidden(model, inputs["input_ids"], device)
            for l in range(n_layers + 1):
                zh_cat_hiddens[cat_name][l].append(hiddens[l])

    for cat_name, en_words in EN_WORDS.items():
        en_cat_hiddens[cat_name] = defaultdict(list)
        for word in en_words:
            inputs = tokenizer(word, return_tensors="pt").to(device)
            hiddens = get_last_token_hidden(model, inputs["input_ids"], device)
            for l in range(n_layers + 1):
                en_cat_hiddens[cat_name][l].append(hiddens[l])

    # 翻译prompt的hidden state
    trans_hiddens = defaultdict(list)
    for zh, en in TRANSLATION_PAIRS:
        prompt = f"{zh}的英文是"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        hiddens = get_last_token_hidden(model, inputs["input_ids"], device)
        for l in range(n_layers + 1):
            trans_hiddens[l].append(hiddens[l])

    results = {}

    # ===== 1. 欧氏距离: 类内 vs 类间 =====
    print("\n  === 欧氏距离分析 ===")
    for l in [0, 3, 6, 9, 12, 18, 24, 30, 35]:
        if l > n_layers:
            continue

        # 类内欧氏距离
        intra_dists = {}
        for cat_name in SEMANTIC_CATEGORIES:
            hiddens_list = zh_cat_hiddens[cat_name][l]
            if len(hiddens_list) < 2:
                continue
            dists = []
            for i in range(len(hiddens_list)):
                for j in range(i+1, len(hiddens_list)):
                    dists.append(np.linalg.norm(hiddens_list[i] - hiddens_list[j]))
            intra_dists[cat_name] = float(np.mean(dists))

        # 类间欧氏距离 (类别中心间)
        centers = {}
        for cat_name in SEMANTIC_CATEGORIES:
            hiddens_list = zh_cat_hiddens[cat_name][l]
            if hiddens_list:
                centers[cat_name] = np.mean(hiddens_list, axis=0)

        inter_dists = {}
        cat_names = list(centers.keys())
        for i in range(len(cat_names)):
            for j in range(i+1, len(cat_names)):
                dist = np.linalg.norm(centers[cat_names[i]] - centers[cat_names[j]])
                inter_dists[f"{cat_names[i]}_vs_{cat_names[j]}"] = float(dist)

        avg_intra = np.mean(list(intra_dists.values()))
        avg_inter = np.mean(list(inter_dists.values()))
        discriminability = avg_inter / (avg_intra + 1e-8)  # >1=可区分

        results[f"L{l}"] = {
            "intra_euclidean": intra_dists,
            "inter_euclidean": inter_dists,
            "avg_intra": float(avg_intra),
            "avg_inter": float(avg_inter),
            "discriminability": float(discriminability),
        }
        print(f"    L{l}: intra={avg_intra:.2f}, inter={avg_inter:.2f}, "
              f"discriminability={discriminability:.2f}")

    # ===== 2. 翻译轨迹: 欧氏距离到zh/en中心 =====
    print("\n  === 翻译轨迹欧氏距离 ===")
    trajectory_results = {}
    for l in [0, 3, 6, 9, 12, 18, 24, 30, 35]:
        if l > n_layers:
            continue

        # zh/en中心
        zh_center = np.mean([zh_cat_hiddens[cat][l][0] for cat in SEMANTIC_CATEGORIES
                            for _ in range(1)], axis=0)  # 用所有类中心的中点
        # 更好的: 用所有zh词的平均
        all_zh = [h for cat in SEMANTIC_CATEGORIES for h in zh_cat_hiddens[cat][l]]
        zh_center = np.mean(all_zh, axis=0)
        all_en = [h for cat in SEMANTIC_CATEGORIES for h in en_cat_hiddens[cat][l]]
        en_center = np.mean(all_en, axis=0)

        # 翻译prompt到zh/en中心的欧氏距离
        dists_to_zh = [np.linalg.norm(trans_hiddens[l][i] - zh_center)
                      for i in range(len(trans_hiddens[l]))]
        dists_to_en = [np.linalg.norm(trans_hiddens[l][i] - en_center)
                      for i in range(len(trans_hiddens[l]))]

        # 同样测zh词和en词到各中心的距离
        zh_to_zh = [np.linalg.norm(h - zh_center) for h in all_zh]
        zh_to_en = [np.linalg.norm(h - en_center) for h in all_zh]
        en_to_zh = [np.linalg.norm(h - zh_center) for h in all_en]
        en_to_en = [np.linalg.norm(h - en_center) for h in all_en]

        trajectory_results[f"L{l}"] = {
            "trans_to_zh": float(np.mean(dists_to_zh)),
            "trans_to_en": float(np.mean(dists_to_en)),
            "zh_to_zh": float(np.mean(zh_to_zh)),
            "zh_to_en": float(np.mean(zh_to_en)),
            "en_to_zh": float(np.mean(en_to_zh)),
            "en_to_en": float(np.mean(en_to_en)),
            "trans_closer_to": "en" if np.mean(dists_to_en) < np.mean(dists_to_zh) else "zh",
        }
        print(f"    L{l}: trans→zh={np.mean(dists_to_zh):.2f}, trans→en={np.mean(dists_to_en):.2f}, "
              f"closer_to={trajectory_results[f'L{l}']['trans_closer_to']}, "
              f"zh→zh={np.mean(zh_to_zh):.2f}, zh→en={np.mean(zh_to_en):.2f}, "
              f"en→zh={np.mean(en_to_zh):.2f}, en→en={np.mean(en_to_en):.2f}")

    # ===== 3. PCA子空间分析 =====
    print("\n  === PCA子空间判别分析 ===")
    pca_results = {}
    for l in [0, 3, 6, 9, 12, 18, 24, 30, 35]:
        if l > n_layers:
            continue

        all_hiddens = []
        labels = []
        for cat_name in SEMANTIC_CATEGORIES:
            all_hiddens.extend(zh_cat_hiddens[cat_name][l])
            labels.extend([cat_name] * len(zh_cat_hiddens[cat_name][l]))

        all_hiddens = np.array(all_hiddens)
        mean_h = np.mean(all_hiddens, axis=0)
        centered = all_hiddens - mean_h

        # PCA
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # 前10个主成分解释的方差
        total_var = np.sum(S**2)
        top10_var = np.sum(S[:10]**2) / total_var

        # 在前10个主成分上的类别可分性 (用ANOVA-like方法)
        from collections import defaultdict as dd
        projections = centered @ Vt[:10].T  # [50, 10]

        # 每个主成分上的类间/类内方差比
        pc_discriminability = []
        for pc in range(10):
            cat_means = dd(list)
            for i, cat in enumerate(labels):
                cat_means[cat].append(projections[i, pc])

            grand_mean = np.mean(projections[:, pc])
            between_var = sum(len(cat_means[c]) * (np.mean(cat_means[c]) - grand_mean)**2
                            for c in cat_means)
            within_var = sum(sum((x - np.mean(cat_means[c]))**2 for x in cat_means[c])
                           for c in cat_means)

            f_ratio = between_var / (within_var + 1e-8)
            pc_discriminability.append(float(f_ratio))

        pca_results[f"L{l}"] = {
            "top10_var_explained": float(top10_var),
            "pc_discriminability": pc_discriminability,
            "best_pc": int(np.argmax(pc_discriminability)),
            "best_pc_f_ratio": float(np.max(pc_discriminability)),
        }
        print(f"    L{l}: top10_var={top10_var:.4f}, "
              f"best_pc={np.argmax(pc_discriminability)} (F={np.max(pc_discriminability):.2f}), "
              f"PC F-ratios={[f'{f:.2f}' for f in pc_discriminability[:5]]}")

    # 保存
    all_results = {
        "euclidean_analysis": results,
        "trajectory_euclidean": trajectory_results,
        "pca_discriminability": pca_results,
    }
    save_path = f"tests/glm5_temp/phase100_euclidean_{model_name}_semantic.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_euclidean_analysis(args.model)
