"""
Phase 61: 因果干预 + 方法修正
=============================

三个方案，按优先级：
  方案1 (Part 1): 基于区分度重定义overlap — 用logistic regression权重选维度
  方案2 (Part 2): 语义轴因果干预 — 在"温和→极端"方向施加perturbation
  方案3 (Part 3): 子空间消融 — 消融极端/方向维度观察生成变化

用法:
  python tests/glm5/phase61_causal_intervention.py --model qwen3 --part 1
  python tests/glm5/phase61_causal_intervention.py --model qwen3 --part 2
  python tests/glm5/phase61_causal_intervention.py --model qwen3 --part 3
  python tests/glm5/phase61_causal_intervention.py --model qwen3 --part all
"""

import sys, os, json, argparse, gc, time, copy
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT = Path("d:/Ai2050/TransformerLens-Project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests" / "glm5"))

RESULT_DIR = PROJECT / "results" / "subspace_topology"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

model_name_global = ""


def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
    print(f"[{ts}] {safe_msg}", flush=True)


# =====================================================================
# 模型加载 (BF16 + device_map="auto" + Flash Attention)
# =====================================================================

def load_model_bf16(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + device_map=auto + flash)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        log_time(f"{model_name} loaded with flash_attention_2")
    except Exception as e:
        log_time(f"Flash attention failed ({e}), falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        )

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"{model_name}: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


import torch


# =====================================================================
# 通用: 收集hidden states
# =====================================================================

def collect_hidden_states(model, tokenizer, device, sentences, target_layers, batch_size=4):
    from model_utils import get_model_info
    info = get_model_info(model, model_name_global)
    all_hidden = {li: [] for li in target_layers}

    for batch_start in range(0, len(sentences), batch_size):
        batch_sents = sentences[batch_start:batch_start + batch_size]
        inputs = tokenizer(batch_sents, return_tensors="pt", padding=True,
                           truncation=True, max_length=64)
        input_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)

        for li in target_layers:
            hs = outputs.hidden_states[li].float().cpu().numpy()
            for i in range(len(batch_sents)):
                mask = inputs["attention_mask"][i].numpy()
                last_pos = np.where(mask > 0)[0][-1]
                all_hidden[li].append(hs[i, last_pos])

        if batch_start % (batch_size * 10) == 0:
            log_time(f"  Collected {batch_start + len(batch_sents)}/{len(sentences)}")

    for li in target_layers:
        all_hidden[li] = np.array(all_hidden[li])
    return all_hidden


def compute_overlap(S1, S2):
    M = S1.T @ S2
    svals = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(svals ** 2))


def extract_subspace(activations, n_dims=10):
    mean = activations.mean(axis=0, keepdims=True)
    centered = activations - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:n_dims].T  # [d_model, n_dims] orthonormal basis


# =====================================================================
# 扩展词集 — 更多样本用于U-shape和区分度分析
# =====================================================================

# 温度轴: 10个词，从极冷到极热
TEMPERATURE_WORDS = ["freezing", "frigid", "cold", "cool", "lukewarm", "warm", "hot", "scorching", "blazing", "searing"]

# 大小轴: 10个词
SIZE_WORDS = ["microscopic", "tiny", "small", "moderate", "medium", "large", "big", "huge", "enormous", "gigantic"]

# 速度轴: 10个词
SPEED_WORDS = ["glacial", "sluggish", "slow", "steady", "moderate", "fast", "quick", "rapid", "swift", "lightning"]

# 情感轴 (valence × intensity): 12个词
EMOTION_WORDS = ["love", "joy", "like", "content", "calm", "neutral",
                 "annoyed", "dislike", "sad", "anger", "hate", "despair"]

# 概念区分实验: 5个类别 × 8个词 = 40词
CONCEPT_GROUPS = {
    "temperature": ["freezing", "cold", "cool", "warm", "hot", "scorching", "lukewarm", "blazing"],
    "size": ["tiny", "small", "medium", "large", "big", "huge", "enormous", "gigantic"],
    "emotion": ["love", "hate", "like", "dislike", "joy", "sad", "anger", "calm"],
    "animal": ["cat", "dog", "bird", "horse", "fish", "bear", "lion", "eagle"],
    "fruit": ["apple", "banana", "orange", "grape", "mango", "peach", "cherry", "pear"],
}

# 30个模板/词
TEMPLATES_PER_WORD = 30

def generate_templates(word_list, category="generic"):
    """为每个词生成30个模板"""
    templates = {}
    for word in word_list:
        sents = [
            f"The {word} was very noticeable today",
            f"She described it as {word}",
            f"He found the {word} quite interesting",
            f"This {word} thing caught my attention",
            f"The weather felt {word} this morning",
            f"Everyone noticed the {word} change",
            f"The {word} conditions were remarkable",
            f"I have never seen anything so {word}",
            f"The {word} experience was memorable",
            f"She thought it was rather {word}",
            f"That {word} feeling was overwhelming",
            f"The situation became {word} quickly",
            f"He considered the {word} aspect important",
            f"The {word} quality stood out immediately",
            f"They described the event as {word}",
            f"A {word} atmosphere filled the room",
            f"The {word} sensation was unmistakable",
            f"Nothing could be more {word} than this",
            f"She appreciated the {word} nature of it",
            f"The {word} phenomenon was well documented",
            f"He encountered a {word} situation",
            f"The {word} characteristic was defining",
            f"People often describe it as {word}",
            f"The {word} element was crucial",
            f"We observed the {word} pattern clearly",
            f"The {word} property was significant",
            f"She recognized the {word} feature",
            f"The {word} condition required attention",
            f"He labeled the experience as {word}",
            f"The {word} state was evident to all",
        ]
        templates[word] = sents[:TEMPLATES_PER_WORD]
    return templates


# =====================================================================
# Part 1: 基于区分度重定义overlap
# =====================================================================

def run_part1(model, tokenizer, device):
    """用logistic regression权重选择区分度维度，与top-k方差对比"""
    from model_utils import get_model_info
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from scipy.stats import spearmanr

    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers
    d_model = info.d_model

    target_layers = [n_layers // 3, n_layers * 2 // 3, n_layers - 1]
    log_time(f"Part 1: Discriminative overlap (layers={target_layers}, d_model={d_model})")

    # 收集所有概念组的hidden states
    all_words = []
    word_labels = {}  # word -> group_idx
    group_names = list(CONCEPT_GROUPS.keys())

    for gi, (gname, words) in enumerate(CONCEPT_GROUPS.items()):
        for w in words:
            all_words.append(w)
            word_labels[w] = gi

    # 生成模板并收集hidden states
    all_templates = generate_templates(all_words)
    all_sentences = []
    sent_word_map = []  # 每个句子属于哪个词
    for w in all_words:
        for s in all_templates[w]:
            all_sentences.append(s)
            sent_word_map.append(w)

    log_time(f"Total sentences: {len(all_sentences)}")
    hidden = collect_hidden_states(model, tokenizer, device, all_sentences, target_layers)

    results = {}

    for li in target_layers:
        log_time(f"  Processing layer {li}...")
        hs = hidden[li]  # [n_sents, d_model]

        # 构建每个词的平均激活
        word_means = {}
        for w in all_words:
            idxs = [i for i, sw in enumerate(sent_word_map) if sw == w]
            word_means[w] = hs[idxs].mean(axis=0)

        # === A. Top-k方差overlap (原方法) ===
        topk_overlaps = {}
        for gi, gname in enumerate(group_names):
            words = CONCEPT_GROUPS[gname]
            word_acts = {w: hs[[i for i, sw in enumerate(sent_word_map) if sw == w]] for w in words}
            for w1 in words:
                for w2 in words:
                    if w1 >= w2:
                        continue
                    s1 = extract_subspace(word_acts[w1], n_dims=10)
                    s2 = extract_subspace(word_acts[w2], n_dims=10)
                    topk_overlaps[f"{w1}-{w2}"] = compute_overlap(s1, s2)

        # === B. 区分度overlap (logistic regression权重选择维度) ===
        discrim_overlaps = {}
        discrim_dims = {}  # 存储每对的区分维度

        for gi, gname in enumerate(group_names):
            words = CONCEPT_GROUPS[gname]
            for w1 in words:
                for w2 in words:
                    if w1 >= w2:
                        continue
                    # 收集两个词的所有样本
                    idxs1 = [i for i, sw in enumerate(sent_word_map) if sw == w1]
                    idxs2 = [i for i, sw in enumerate(sent_word_map) if sw == w2]
                    X = np.vstack([hs[idxs1], hs[idxs2]])
                    y = np.array([0] * len(idxs1) + [1] * len(idxs2))

                    # 训练logistic regression获取权重
                    try:
                        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
                        clf.fit(X, y)
                        weights = np.abs(clf.coef_[0])  # [d_model]
                    except Exception:
                        weights = np.zeros(d_model)

                    # 选择权重最高的10个维度
                    top_discrim_dims = np.argsort(weights)[-10:]
                    discrim_dims[f"{w1}-{w2}"] = top_discrim_dims.tolist()

                    # 用区分度维度的子空间overlap
                    # 对w1和w2分别用区分度维度上的激活值构建子空间
                    act1_discrim = hs[idxs1][:, top_discrim_dims]  # [n, 10]
                    act2_discrim = hs[idxs2][:, top_discrim_dims]  # [n, 10]

                    # 用区分度维度上的协方差方向
                    mean1 = act1_discrim.mean(axis=0)
                    mean2 = act2_discrim.mean(axis=0)
                    centered1 = act1_discrim - mean1
                    centered2 = act2_discrim - mean2

                    # 在全d_model空间中构建区分度子空间
                    full_sub1 = np.zeros((d_model, 10))
                    full_sub2 = np.zeros((d_model, 10))
                    if centered1.shape[0] >= 10:
                        U1, S1, Vt1 = np.linalg.svd(centered1, full_matrices=False)
                        for j, d in enumerate(top_discrim_dims):
                            full_sub1[d, j] = 1.0  # 区分度维度的标准基
                    else:
                        for j, d in enumerate(top_discrim_dims):
                            full_sub1[d, j] = 1.0

                    if centered2.shape[0] >= 10:
                        for j, d in enumerate(top_discrim_dims):
                            full_sub2[d, j] = 1.0
                    else:
                        for j, d in enumerate(top_discrim_dims):
                            full_sub2[d, j] = 1.0

                    # 简单overlap: 共享区分度维度的比例
                    dims1 = set(top_discrim_dims.tolist())
                    dims2_set = set()  # 反向训练得到w2的区分维度
                    try:
                        clf2 = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
                        clf2.fit(X, 1 - y)  # 反向标签
                        weights2 = np.abs(clf2.coef_[0])
                        top_discrim_dims2 = np.argsort(weights2)[-10:]
                        dims2_set = set(top_discrim_dims2.tolist())
                    except Exception:
                        pass

                    jaccard = len(dims1 & dims2_set) / max(len(dims1 | dims2_set), 1)
                    discrim_overlaps[f"{w1}-{w2}"] = jaccard

        # === C. 分类准确率对比: Top-k vs 区分度 vs 随机 ===
        classification_results = {}
        for gi, gname in enumerate(group_names):
            words = CONCEPT_GROUPS[gname]
            word_idxs = {w: [i for i, sw in enumerate(sent_word_map) if sw == w] for w in words}

            X_all = []
            y_all = []
            for wi, w in enumerate(words):
                for idx in word_idxs[w]:
                    X_all.append(hs[idx])
                    y_all.append(wi)
            X_all = np.array(X_all)
            y_all = np.array(y_all)

            # All dims
            try:
                clf = LogisticRegression(max_iter=1000, C=1.0)
                acc_all = np.mean(cross_val_score(clf, X_all, y_all, cv=3))
            except Exception:
                acc_all = -1

            # Top-k variance dims
            all_acts = hs[np.array([i for i, sw in enumerate(sent_word_map) if sw in words])]
            centered = all_acts - all_acts.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            topk_dims = np.argsort(np.abs(Vt[:10]).sum(axis=0))[-10:]
            try:
                clf = LogisticRegression(max_iter=1000, C=1.0)
                acc_topk = np.mean(cross_val_score(clf, X_all[:, topk_dims], y_all, cv=3))
            except Exception:
                acc_topk = -1

            # Discriminative dims (one-vs-rest per word)
            discrim_dim_set = set()
            for w in words:
                idxs_w = [i for i, sw in enumerate(sent_word_map) if sw == w]
                idxs_other = [i for i, sw in enumerate(sent_word_map) if sw in words and sw != w]
                X_bin = np.vstack([hs[idxs_w], hs[idxs_other]])
                y_bin = np.array([1] * len(idxs_w) + [0] * len(idxs_other))
                try:
                    clf = LogisticRegression(max_iter=1000, C=1.0)
                    clf.fit(X_bin, y_bin)
                    top_d = np.argsort(np.abs(clf.coef_[0]))[-5:]
                    discrim_dim_set.update(top_d.tolist())
                except Exception:
                    pass

            discrim_dims_list = sorted(discrim_dim_set)
            if len(discrim_dims_list) > 0:
                try:
                    clf = LogisticRegression(max_iter=1000, C=1.0)
                    acc_discrim = np.mean(cross_val_score(clf, X_all[:, discrim_dims_list], y_all, cv=3))
                except Exception:
                    acc_discrim = -1
            else:
                acc_discrim = -1

            # Random dims baseline (10 trials)
            random_accs = []
            for _ in range(10):
                rand_dims = np.random.choice(d_model, size=min(len(discrim_dims_list), 10), replace=False)
                try:
                    clf = LogisticRegression(max_iter=1000, C=1.0)
                    random_accs.append(np.mean(cross_val_score(clf, X_all[:, rand_dims], y_all, cv=3)))
                except Exception:
                    random_accs.append(-1)
            acc_random = np.mean(random_accs)

            classification_results[gname] = {
                "n_words": len(words),
                "n_dims_discrim": len(discrim_dims_list),
                "acc_all": round(acc_all, 4),
                "acc_topk": round(acc_topk, 4),
                "acc_discrim": round(acc_discrim, 4),
                "acc_random": round(acc_random, 4),
            }
            log_time(f"    {gname}: all={acc_all:.3f}, topk={acc_topk:.3f}, "
                      f"discrim={acc_discrim:.3f}, random={acc_random:.3f}")

        # === D. Spearman对比: top-k overlap vs discrim overlap ===
        common_pairs = set(topk_overlaps.keys()) & set(discrim_overlaps.keys())
        if len(common_pairs) > 5:
            topk_vals = [topk_overlaps[p] for p in sorted(common_pairs)]
            discrim_vals = [discrim_overlaps[p] for p in sorted(common_pairs)]
            rho, pval = spearmanr(topk_vals, discrim_vals)
        else:
            rho, pval = 0, 1

        results[f"layer_{li}"] = {
            "topk_overlaps": topk_overlaps,
            "discrim_overlaps": discrim_overlaps,
            "spearman_topk_vs_discrim": {"rho": round(rho, 4), "p": round(pval, 4)},
            "classification": classification_results,
        }

        log_time(f"  Layer {li}: Spearman(topk, discrim)={rho:.3f}, p={pval:.4f}")

    results["meta"] = {
        "model": model_name_global,
        "target_layers": target_layers,
        "d_model": d_model,
        "n_concept_groups": len(CONCEPT_GROUPS),
        "n_words_total": len(all_words),
        "templates_per_word": TEMPLATES_PER_WORD,
    }

    out_path = RESULT_DIR / f"phase61_part1_{model_name_global}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_time(f"Part 1 results saved to {out_path}")

    return results


# =====================================================================
# Part 2: 语义轴因果干预
# =====================================================================

def run_part2(model, tokenizer, device):
    """在"温和→极端"方向上施加perturbation，检验输出变化"""
    from model_utils import get_model_info, get_W_U, get_layers
    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers
    d_model = info.d_model

    target_layers = [n_layers // 3, n_layers * 2 // 3, n_layers - 1]
    log_time(f"Part 2: Causal intervention on semantic axes (layers={target_layers})")

    # 步骤1: 收集温度轴的hidden states，提取"温和vs极端"方向
    temp_templates = generate_templates(TEMPERATURE_WORDS)
    all_temp_sents = []
    sent_temp_word = []
    for w in TEMPERATURE_WORDS:
        for s in temp_templates[w]:
            all_temp_sents.append(s)
            sent_temp_word.append(w)

    log_time(f"Collecting temperature hidden states ({len(all_temp_sents)} sents)...")
    temp_hidden = collect_hidden_states(model, tokenizer, device, all_temp_sents, target_layers)

    # 大小轴
    size_templates = generate_templates(SIZE_WORDS)
    all_size_sents = []
    sent_size_word = []
    for w in SIZE_WORDS:
        for s in size_templates[w]:
            all_size_sents.append(s)
            sent_size_word.append(w)

    log_time(f"Collecting size hidden states ({len(all_size_sents)} sents)...")
    size_hidden = collect_hidden_states(model, tokenizer, device, all_size_sents, target_layers)

    results = {}

    for li in target_layers:
        log_time(f"  Processing layer {li}...")

        # 温度轴: 提取PC1 (温和vs极端)
        temp_hs = temp_hidden[li]
        word_means_temp = {}
        for w in TEMPERATURE_WORDS:
            idxs = [i for i, sw in enumerate(sent_temp_word) if sw == w]
            word_means_temp[w] = temp_hs[idxs].mean(axis=0)

        means_matrix = np.array([word_means_temp[w] for w in TEMPERATURE_WORDS])
        centered = means_matrix - means_matrix.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        pc1 = Vt[0]  # 温度轴PC1方向
        pc2 = Vt[1]  # 温度轴PC2方向

        # 验证PC1是否编码"温和vs极端"
        mild_words = ["cool", "lukewarm", "warm"]
        extreme_words = ["freezing", "scorching", "blazing", "searing"]
        mild_proj = np.mean([abs(np.dot(word_means_temp[w], pc1)) for w in mild_words])
        extreme_proj = np.mean([abs(np.dot(word_means_temp[w], pc1)) for w in extreme_words])

        log_time(f"    Temp PC1: mild_proj={mild_proj:.3f}, extreme_proj={extreme_proj:.3f}, "
                  f"ratio={extreme_proj/max(mild_proj, 0.001):.2f}")

        # 大小轴: 同样提取
        size_hs = size_hidden[li]
        word_means_size = {}
        for w in SIZE_WORDS:
            idxs = [i for i, sw in enumerate(sent_size_word) if sw == w]
            word_means_size[w] = size_hs[idxs].mean(axis=0)

        size_means = np.array([word_means_size[w] for w in SIZE_WORDS])
        size_centered = size_means - size_means.mean(axis=0)
        U_s, S_s, Vt_s = np.linalg.svd(size_centered, full_matrices=False)
        size_pc1 = Vt_s[0]
        size_pc2 = Vt_s[1]

        # 步骤2: 因果干预 — 在hidden state上施加偏移
        # 测试句: 从温和词开始，偏移向极端方向
        test_prompts = [
            "The weather today is",
            "She described the temperature as",
            "The size of the object is",
            "He thought it was rather",
            "The water felt",
            "The building was",
        ]

        intervention_results = []

        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)

            # 基线生成
            with torch.no_grad():
                base_out = model.generate(
                    input_ids, attention_mask=attention_mask,
                    max_new_tokens=15, do_sample=False, repetition_penalty=1.2,
                )
            base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)

            # 在第li层hook注入偏移
            layers_list = get_layers(model)
            betas = [0, 2, 5, 10, 15, 20]

            for axis_name, direction in [("temp_pc1", pc1), ("temp_pc2", pc2),
                                          ("size_pc1", size_pc1), ("size_pc2", size_pc2)]:
                for beta in betas:
                    direction_t = torch.tensor(direction, dtype=torch.bfloat16, device=input_device)

                    def make_intervention_hook(dir_vec, beta_val, layer_idx):
                        captured_out = {}
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                h = output[0].clone()
                                # 在最后一个token位置注入方向
                                h[:, -1, :] += beta_val * dir_vec.to(h.dtype)
                                captured_out['output'] = (h,) + output[1:]
                                return captured_out['output']
                            else:
                                h = output.clone()
                                h[:, -1, :] += beta_val * dir_vec.to(h.dtype)
                                return h
                        return hook, captured_out

                    hook_fn, captured = make_intervention_hook(direction_t, beta, li)
                    handle = layers_list[li].register_forward_hook(hook_fn)

                    try:
                        with torch.no_grad():
                            int_out = model.generate(
                                input_ids, attention_mask=attention_mask,
                                max_new_tokens=15, do_sample=False, repetition_penalty=1.2,
                            )
                        int_text = tokenizer.decode(int_out[0], skip_special_tokens=True)
                    except Exception as e:
                        int_text = f"ERROR: {e}"

                    handle.remove()

                    intervention_results.append({
                        "prompt": prompt,
                        "axis": axis_name,
                        "beta": beta,
                        "layer": li,
                        "baseline": base_text,
                        "intervened": int_text,
                        "changed": base_text != int_text,
                    })

                    if beta in [0, 10, 20] and beta != 0:
                        log_time(f"      {prompt[:30]} | {axis_name} b={beta}: "
                                  f"{'CHANGED' if base_text != int_text else 'same'}")

        results[f"layer_{li}"] = {
            "temp_pc1_mild_proj": round(float(mild_proj), 4),
            "temp_pc1_extreme_proj": round(float(extreme_proj), 4),
            "temp_pc1_extreme_ratio": round(float(extreme_proj / max(mild_proj, 0.001)), 2),
            "interventions": intervention_results,
        }

        # 统计因果效应
        for axis_name in ["temp_pc1", "temp_pc2", "size_pc1", "size_pc2"]:
            axis_ints = [r for r in intervention_results if r["axis"] == axis_name]
            for beta in [5, 10, 15, 20]:
                beta_ints = [r for r in axis_ints if r["beta"] == beta]
                n_changed = sum(1 for r in beta_ints if r["changed"])
                log_time(f"    {axis_name} beta={beta}: {n_changed}/{len(beta_ints)} changed")

    results["meta"] = {
        "model": model_name_global,
        "target_layers": target_layers,
        "n_temperature_words": len(TEMPERATURE_WORDS),
        "n_size_words": len(SIZE_WORDS),
    }

    out_path = RESULT_DIR / f"phase61_part2_{model_name_global}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_time(f"Part 2 results saved to {out_path}")

    return results


# =====================================================================
# Part 3: 子空间消融
# =====================================================================

def run_part3(model, tokenizer, device):
    """消融极端维度和方向维度，观察对生成的影响"""
    from model_utils import get_model_info, get_layers, get_W_U
    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers
    d_model = info.d_model

    target_layer = n_layers * 2 // 3  # 中间偏高层
    log_time(f"Part 3: Subspace ablation (target_layer={target_layer})")

    # 收集温度轴hidden states
    temp_templates = generate_templates(TEMPERATURE_WORDS)
    all_temp_sents = []
    sent_temp_word = []
    for w in TEMPERATURE_WORDS:
        for s in temp_templates[w]:
            all_temp_sents.append(s)
            sent_temp_word.append(w)

    log_time(f"Collecting hidden states ({len(all_temp_sents)} sents)...")
    hidden = collect_hidden_states(model, tokenizer, device, all_temp_sents, [target_layer])
    hs = hidden[target_layer]

    # 提取各词均值
    word_means = {}
    for w in TEMPERATURE_WORDS:
        idxs = [i for i, sw in enumerate(sent_temp_word) if sw == w]
        word_means[w] = hs[idxs].mean(axis=0)

    means_matrix = np.array([word_means[w] for w in TEMPERATURE_WORDS])
    centered = means_matrix - means_matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # 主要方向
    pc1 = Vt[0]  # 温和vs极端 (假设)
    pc2 = Vt[1]  # 方向 (假设)

    # 消融方式: 在特定方向上投影置零
    def project_out(vec, direction):
        """从vec中移除direction方向的分量"""
        proj = np.dot(vec, direction) * direction
        return vec - proj

    # 获取W_U用于解码
    W_U = get_W_U(model, model_name_global)  # [vocab, d_model]

    # 测试句和消融条件
    test_cases = [
        ("The water felt", "temperature"),
        ("She said it was", "temperature"),
        ("The object was", "size"),
        ("He felt very", "emotion"),
        ("Today the weather is", "temperature"),
        ("The room was", "temperature"),
    ]

    ablation_conditions = [
        ("none", None),  # 基线
        ("remove_pc1", pc1),  # 移除温和/极端维度
        ("remove_pc2", pc2),  # 移除方向维度
        ("remove_pc1_pc2", None),  # 同时移除PC1和PC2
        ("amplify_pc1_pos", None),  # 放大PC1正向 (偏向极端?)
        ("amplify_pc1_neg", None),  # 放大PC1负向
    ]

    results = {}
    layers_list = get_layers(model)

    for prompt, category in test_cases:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        prompt_results = {}

        for cond_name, direction in ablation_conditions:
            # 设置hook
            def make_ablation_hook(condition, pc1_dir, pc2_dir, target_li):
                def hook(module, input, output):
                    if not isinstance(output, tuple):
                        return output
                    h = output[0].clone().float().cpu().numpy()  # [batch, seq, d]

                    for b in range(h.shape[0]):
                        for s in range(h.shape[1]):
                            vec = h[b, s]
                            if condition == "remove_pc1":
                                h[b, s] = project_out(vec, pc1_dir)
                            elif condition == "remove_pc2":
                                h[b, s] = project_out(vec, pc2_dir)
                            elif condition == "remove_pc1_pc2":
                                vec = project_out(vec, pc1_dir)
                                h[b, s] = project_out(vec, pc2_dir)
                            elif condition == "amplify_pc1_pos":
                                proj = np.dot(vec, pc1_dir)
                                h[b, s] = vec + 5.0 * proj * pc1_dir
                            elif condition == "amplify_pc1_neg":
                                proj = np.dot(vec, pc1_dir)
                                h[b, s] = vec - 5.0 * proj * pc1_dir

                    # 转回tensor
                    new_h = torch.tensor(h, dtype=output[0].dtype, device=output[0].device)
                    return (new_h,) + output[1:]
                return hook

            hook_fn = make_ablation_hook(cond_name, pc1, pc2, target_layer)
            handle = layers_list[target_layer].register_forward_hook(hook_fn)

            try:
                with torch.no_grad():
                    gen_ids = model.generate(
                        input_ids, attention_mask=attention_mask,
                        max_new_tokens=15, do_sample=False, repetition_penalty=1.2,
                    )
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            except Exception as e:
                gen_text = f"ERROR: {e}"

            handle.remove()
            prompt_results[cond_name] = gen_text

            if cond_name in ["none", "remove_pc1", "remove_pc2", "amplify_pc1_pos"]:
                log_time(f"  {prompt[:25]} | {cond_name}: {gen_text[:60]}")

        results[prompt] = prompt_results

    # 额外: 分析消融对logit分布的影响
    log_time("Analyzing logit distribution changes under ablation...")

    temp_word_tokens = {}
    for w in TEMPERATURE_WORDS:
        tok_ids = tokenizer.encode(w, add_special_tokens=False)
        if tok_ids:
            temp_word_tokens[w] = tok_ids[0]

    size_word_tokens = {}
    for w in SIZE_WORDS:
        tok_ids = tokenizer.encode(w, add_special_tokens=False)
        if tok_ids:
            size_word_tokens[w] = tok_ids[0]

    prompt_for_logit = "The temperature was"
    inputs = tokenizer(prompt_for_logit, return_tensors="pt", truncation=True, max_length=64)
    input_device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    logit_analysis = {}

    for cond_name, direction in [("none", None), ("remove_pc1", pc1), ("remove_pc2", pc2)]:
        hook_fn = make_ablation_hook(cond_name, pc1, pc2, target_layer)
        handle = layers_list[target_layer].register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
            logits = outputs.logits[0, -1].float().cpu().numpy()
        except Exception as e:
            logits = np.zeros(W_U.shape[0])

        handle.remove()

        # 温度词的logit
        temp_logits = {}
        for w, tid in temp_word_tokens.items():
            temp_logits[w] = round(float(logits[tid]), 4)

        # 极端 vs 温和 的平均logit
        extreme_words_list = ["freezing", "scorching", "blazing", "searing", "frigid"]
        mild_words_list = ["cool", "lukewarm", "warm", "moderate"]

        extreme_logits = [temp_logits.get(w, 0) for w in extreme_words_list if w in temp_logits]
        mild_logits = [temp_logits.get(w, 0) for w in mild_words_list if w in temp_logits]

        logit_analysis[cond_name] = {
            "temp_word_logits": temp_logits,
            "extreme_avg": round(np.mean(extreme_logits), 4) if extreme_logits else 0,
            "mild_avg": round(np.mean(mild_logits), 4) if mild_logits else 0,
            "extreme_minus_mild": round(
                np.mean(extreme_logits) - np.mean(mild_logits), 4
            ) if extreme_logits and mild_logits else 0,
        }
        log_time(f"  Logit analysis {cond_name}: extreme_avg={logit_analysis[cond_name]['extreme_avg']:.2f}, "
                  f"mild_avg={logit_analysis[cond_name]['mild_avg']:.2f}, "
                  f"diff={logit_analysis[cond_name]['extreme_minus_mild']:.2f}")

    final_results = {
        "generations": results,
        "logit_analysis": logit_analysis,
        "meta": {
            "model": model_name_global,
            "target_layer": target_layer,
            "temperature_words": TEMPERATURE_WORDS,
            "size_words": SIZE_WORDS,
        }
    }

    out_path = RESULT_DIR / f"phase61_part3_{model_name_global}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    log_time(f"Part 3 results saved to {out_path}")

    return final_results


# =====================================================================
# 主函数
# =====================================================================

def main():
    global model_name_global
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--part", type=str, required=True,
                        choices=["1", "2", "3", "all"])
    args = parser.parse_args()

    model_name_global = args.model
    log_time(f"=== Phase 61: {args.model} Part {args.part} ===")

    model, tokenizer, device = load_model_bf16(args.model)

    try:
        if args.part == "1":
            run_part1(model, tokenizer, device)
        elif args.part == "2":
            run_part2(model, tokenizer, device)
        elif args.part == "3":
            run_part3(model, tokenizer, device)
        elif args.part == "all":
            run_part1(model, tokenizer, device)
            gc.collect(); torch.cuda.empty_cache(); time.sleep(5)
            run_part2(model, tokenizer, device)
            gc.collect(); torch.cuda.empty_cache(); time.sleep(5)
            run_part3(model, tokenizer, device)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        log_time(f"Model {args.model} released")

    log_time(f"=== Phase 61 Part {args.part} complete ===")


if __name__ == "__main__":
    main()
