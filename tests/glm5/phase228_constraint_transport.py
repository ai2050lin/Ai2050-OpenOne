"""
Phase 228: 约束输运与语言守恒量 — Constraint Transport & Linguistic Conserved Quantities
=========================================================================================

用户核心洞察:
  1. Transformer不是统一动力系统, 而是三阶段系统:
     - Phase A (浅层L0-L3): 嵌入重构, 离散token→连续约束态
     - Phase B (中层L6-L24): 约束传播, 真正的语言计算
     - Phase C (深层L24+): 惯性保持, 约束固化

  2. Koopman失败的根本原因: Transformer是条件动力系统 h_{l+1}=F(h_l, c_l),
     c_l来自注意力上下文, 不是自治系统

  3. 最大风险: 真正的状态可能不是h_l, 而是Attention Graph / KV Memory

  4. 核心问题转变: 从"什么方向传播"→"什么约束被保持(守恒)"
     → 语言守恒量 = 跨层近似不变的约束关系

本Phase四个实验:

  Exp1: 语言守恒量检测 (Linguistic Conserved Quantities) ★★★★★
    - 在每层训练线性探针预测约束标签(singular/plural, present/past等)
    - 追踪探针准确率vs层: 如果高且稳定→守恒量; 如果变化→正在被变换
    - 对比: "守恒" vs "被编码" vs "被丢弃"

  Exp2: 注意力约束输运 (Attention Constraint Transport) ★★★★★
    - 提取各层attention pattern, 看约束相关token对之间的注意力权重
    - 关键: 约束是否通过attention输运而非hidden state传播?
    - 指标: attention权重的约束敏感性(约束对之间的注意力差异)

  Exp3: 残差流分解 (Residual Stream Decomposition) ★★★★
    - 分解 h_{l+1} = h_l + Δh_l (残差 + 变化量)
    - Δh_l携带约束信息吗? Δh_l可预测吗?
    - 关键: 如果Δh_l不可预测但h_{l+1}可预测→只是惯性, 非动力学

  Exp4: 跨token约束耦合图 (Cross-Token Constraint Coupling) ★★★
    - 对每个token对, 计算它们的hidden state变化是否耦合
    - 约束相关token(主语-动词)的变化是否比无关token对更耦合?
    - 这构建"约束关系图"

跨模型: Qwen3 → GLM4 → DS7B
BF16 + device_map="auto" + eager(需attention输出) + 定期GC
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import warnings
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from model_utils import (get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS,
                          get_sample_layers)

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 模型加载 (BF16 + device_map="auto" + eager for attention)
# ============================================================

def load_model_eager(model_name: str):
    """BF16 + device_map="auto" + eager attention (need output_attentions)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bf16 + auto + eager)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    print(f"[load] device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# ============================================================
# 句子生成 — 带token位置标注
# ============================================================

def generate_constraint_pairs_with_positions():
    """
    生成约束对, 返回结构包含:
    - sentence A/B
    - constraint type & label (0/1)
    - 关键token位置 (subject_idx, verb_idx, last_idx)
    """
    pairs = []

    # === SVA ===
    sva_data = [
        ("cat", "chases"), ("dog", "runs"), ("bird", "sings"),
        ("girl", "reads"), ("boy", "walks"), ("tree", "grows"),
        ("car", "moves"), ("child", "plays"), ("man", "works"),
        ("woman", "dances"), ("fish", "swims"), ("horse", "jumps"),
        ("student", "studies"), ("teacher", "speaks"), ("king", "rules"),
        ("queen", "smiles"), ("doctor", "heals"), ("soldier", "marches"),
        ("farmer", "plants"), ("artist", "paints"),
        ("river", "flows"), ("mountain", "stands"), ("forest", "whispers"),
        ("ocean", "crashes"), ("star", "shines"), ("moon", "glows"),
        ("sun", "rises"), ("cloud", "drifts"), ("wind", "blows"),
        ("rain", "falls"), ("fire", "burns"), ("snow", "melts"),
        ("bell", "rings"), ("book", "opens"), ("door", "closes"),
        ("flower", "blooms"), ("wolf", "howls"), ("bear", "sleeps"),
        ("deer", "runs"), ("lion", "roars"),
    ]
    for subj, verb_s in sva_data:
        verb_p = verb_s.rstrip('s') if verb_s.endswith('s') else verb_s + 'es'
        # "The cat chases" → [The, cat, chases] → subj=1, verb=2
        pairs.append({
            "A": f"The {subj} {verb_s}",
            "B": f"The {subj}s {verb_p}",
            "type": "number_sva",
            "label_A": 0,  # singular
            "label_B": 1,  # plural
            "subj_offset": 1,  # subject is 2nd token
            "verb_offset": 2,  # verb is 3rd token
        })

    # === Tense ===
    tense_data = [
        ("cat", "sleeps", "slept"), ("dog", "runs", "ran"),
        ("bird", "sings", "sang"), ("girl", "reads", "read"),
        ("boy", "walks", "walked"), ("tree", "grows", "grew"),
        ("car", "moves", "moved"), ("child", "plays", "played"),
        ("man", "works", "worked"), ("woman", "dances", "danced"),
        ("fish", "swims", "swam"), ("student", "studies", "studied"),
        ("teacher", "speaks", "spoke"), ("river", "flows", "flowed"),
        ("wind", "blows", "blew"), ("sun", "shines", "shone"),
        ("rain", "falls", "fell"), ("fire", "burns", "burned"),
        ("snow", "melts", "melted"), ("bell", "rings", "rang"),
        ("king", "rules", "ruled"), ("queen", "smiles", "smiled"),
        ("doctor", "heals", "healed"), ("soldier", "marches", "marched"),
        ("farmer", "plants", "planted"), ("artist", "paints", "painted"),
        ("river", "drifts", "drifted"), ("mountain", "towers", "towered"),
        ("forest", "whispers", "whispered"), ("ocean", "crashes", "crashed"),
        ("star", "twinkles", "twinkled"), ("moon", "glows", "glowed"),
        ("cloud", "drifts", "drifted"), ("thunder", "roars", "roared"),
        ("snake", "crawls", "crawled"), ("rabbit", "hops", "hopped"),
        ("eagle", "soars", "soared"), ("whale", "dives", "dived"),
        ("tiger", "hunts", "hunted"), ("lion", "roars", "roared"),
    ]
    for subj, pres, past in tense_data:
        pairs.append({
            "A": f"The {subj} {pres}",
            "B": f"The {subj} {past}",
            "type": "tense",
            "label_A": 0,  # present
            "label_B": 1,  # past
            "subj_offset": 1,
            "verb_offset": 2,
        })

    # === Negation ===
    neg_data = [
        ("cat", "can sleep", "cannot sleep"),
        ("dog", "will run", "will not run"),
        ("bird", "does sing", "does not sing"),
        ("girl", "is reading", "is not reading"),
        ("boy", "has eaten", "has not eaten"),
        ("car", "was moving", "was not moving"),
        ("child", "should play", "should not play"),
        ("man", "could work", "could not work"),
        ("woman", "would dance", "would not dance"),
        ("fish", "can swim", "cannot swim"),
        ("student", "must study", "must not study"),
        ("teacher", "will speak", "will not speak"),
        ("river", "is flowing", "is not flowing"),
        ("wind", "might blow", "might not blow"),
        ("sun", "is shining", "is not shining"),
        ("rain", "was falling", "was not falling"),
        ("fire", "has burned", "has not burned"),
        ("snow", "will melt", "will not melt"),
        ("king", "will rule", "will not rule"),
        ("queen", "is smiling", "is not smiling"),
        ("doctor", "should help", "should not help"),
        ("soldier", "can fight", "cannot fight"),
        ("farmer", "will plant", "will not plant"),
        ("artist", "could paint", "could not paint"),
        ("river", "was flowing", "was not flowing"),
        ("star", "was twinkling", "was not twinkling"),
        ("moon", "is glowing", "is not glowing"),
        ("cloud", "was drifting", "was not drifting"),
        ("snake", "can crawl", "cannot crawl"),
        ("rabbit", "will hop", "will not hop"),
        ("eagle", "is soaring", "is not soaring"),
        ("whale", "was diving", "was not diving"),
        ("tiger", "can hunt", "cannot hunt"),
        ("lion", "is roaring", "is not roaring"),
        ("wolf", "will howl", "will not howl"),
        ("bear", "was sleeping", "was not sleeping"),
        ("deer", "is running", "is not running"),
        ("fox", "can hide", "cannot hide"),
        ("hawk", "will soar", "will not soar"),
        ("owl", "is hunting", "is not hunting"),
    ]
    for subj, aff, neg in neg_data:
        pairs.append({
            "A": f"The {subj} {aff}",
            "B": f"The {subj} {neg}",
            "type": "negation",
            "label_A": 0,  # affirmative
            "label_B": 1,  # negative
            "subj_offset": 1,
            # verb position varies with negation structure
            "verb_offset": -1,  # will be determined dynamically
        })

    return pairs


# ============================================================
# 收集带attention的完整前向传播结果
# ============================================================

def collect_with_attention(model, tokenizer, device, sentences, n_layers,
                           desc="collecting"):
    """
    收集隐藏状态 + attention权重
    
    Returns:
        all_h: {layer_idx: np.array [n_sentences, d_model]}
        all_attn: {layer_idx: np.array [n_sentences, n_heads, seq_len, seq_len]}
        token_positions: list of dict with key token offsets
    """
    layers = get_layers(model)
    all_h = {l: [] for l in range(n_layers)}
    all_attn = {l: [] for l in range(n_layers)}

    for si, text in enumerate(sentences):
        try:
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True, output_attentions=True)

            hs = out.hidden_states
            attns = out.attentions

            # hidden states: last token position
            for li in range(n_layers):
                if li < len(hs):
                    h = hs[li][0, -1, :].float().cpu().numpy()
                    all_h[li].append(h)

            # attention weights: [batch, n_heads, seq_len, seq_len]
            for li in range(min(n_layers, len(attns) if attns else 0)):
                if attns and li < len(attns):
                    attn = attns[li][0].float().cpu().numpy()  # [n_heads, seq, seq]
                    all_attn[li].append(attn)

        except Exception as e:
            print(f"    [!] Sentence {si} failed: {e}")
            for li in range(n_layers):
                all_h[li].append(None)

        if (si + 1) % 20 == 0:
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] {desc}: {si+1}/{len(sentences)} done")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Convert to numpy
    for li in range(n_layers):
        valid_h = [h for h in all_h[li] if h is not None]
        if valid_h:
            all_h[li] = np.array(valid_h)
        else:
            all_h[li] = None

    return all_h, all_attn


# ============================================================
# Exp1: 语言守恒量检测 ★★★★★
# ============================================================

def linguistic_conserved_quantities(train_h, test_h, train_labels, test_labels,
                                     n_layers, d_model, sample_layers):
    """
    在每层训练线性探针预测约束标签
    
    关键指标:
    - 准确率 vs 层: 如果高且稳定 → 守恒量
    - 准确率变化模式: 编码(上升) / 保持(平稳) / 丢弃(下降)
    
    对比3种hidden state:
    1. 最后token位置 h[-1] (约束最相关)
    2. 约束差分 Δh = h(constraint_B) - h(constraint_A)
    3. 全句均值
    """
    print(f"\n{'='*60}")
    print("Exp1: Linguistic Conserved Quantities (Probe Accuracy)")
    print(f"{'='*60}")

    results = {}

    for ctype in set(train_labels.keys()) & set(test_labels.keys()):
        y_train = np.array(train_labels[ctype])
        y_test = np.array(test_labels[ctype])

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        acc_per_layer = {}

        for li in sample_layers:
            h_train = train_h.get(li)
            h_test = test_h.get(li)
            if h_train is None or h_test is None:
                continue

            n_train = min(len(h_train), len(y_train))
            n_test = min(len(h_test), len(y_test))

            X_train = h_train[:n_train]
            X_test = h_test[:n_test]
            y_tr = y_train[:n_train]
            y_te = y_test[:n_test]

            # Ridge logistic regression (simplified: linear + threshold)
            # 用LDA代替 (更稳定)
            try:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                clf = LinearDiscriminantAnalysis()
                clf.fit(X_train, y_tr)
                acc = clf.score(X_test, y_te)
            except Exception:
                # fallback: 最近中心分类
                centers = []
                for c in np.unique(y_tr):
                    centers.append(X_train[y_tr == c].mean(axis=0))
                preds = []
                for x in X_test:
                    dists = [np.linalg.norm(x - c) for c in centers]
                    preds.append(np.argmin(dists))
                acc = np.mean(np.array(preds) == y_te)

            acc_per_layer[f"L{li}"] = round(acc, 4)

        results[ctype] = acc_per_layer

        # 打印摘要
        accs = [v for v in acc_per_layer.values()]
        if accs:
            print(f"  {ctype}: min={min(accs):.3f}, max={max(accs):.3f}, "
                  f"mean={np.mean(accs):.3f}")
            # 打印前3层和后3层
            keys = sorted(acc_per_layer.keys())
            for k in keys[:3]:
                print(f"    {k}: {acc_per_layer[k]:.3f}")
            if len(keys) > 6:
                print(f"    ...")
            for k in keys[-3:]:
                print(f"    {k}: {acc_per_layer[k]:.3f}")

    return results


# ============================================================
# Exp2: 注意力约束输运 ★★★★★
# ============================================================

def attention_constraint_transport(model, tokenizer, device, pairs_by_type,
                                    n_layers, sample_layers):
    """
    测试约束是否通过attention输运
    
    方法:
    1. 对每个约束对(A,B), 提取各层attention pattern
    2. 计算约束相关token对(主语→动词)之间的attention权重
    3. 比较 A vs B 的attention权重差异
    4. 如果差异显著 → attention携带约束信息
    """
    print(f"\n{'='*60}")
    print("Exp2: Attention Constraint Transport")
    print(f"{'='*60}")

    results = {}

    for ctype, pairs in pairs_by_type.items():
        # 只取前20对
        test_pairs = pairs[:20]

        attn_diff_per_layer = {li: [] for li in sample_layers}
        last_attn_diff_per_layer = {li: [] for li in sample_layers}

        for pair in test_pairs:
            sent_A = pair["A"]
            sent_B = pair["B"]

            for sent, label in [(sent_A, 0), (sent_B, 1)]:
                try:
                    inputs = tokenizer(sent, return_tensors="pt",
                                       truncation=True, max_length=64)
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)

                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attention_mask,
                                    output_attentions=True)

                    attns = out.attentions
                    n_tokens = input_ids.shape[1]

                    for li_idx, li in enumerate(sample_layers):
                        if li >= len(attns) if attns else True:
                            continue
                        if attns is None or li >= len(attns):
                            continue

                        attn = attns[li][0].float().cpu().numpy()  # [n_heads, seq, seq]

                        # 指标1: 最后一个token对所有其他token的平均attention
                        # 这是"约束聚合"指标: 最后token从其他token获取多少信息
                        last_to_all = attn[:, -1, :n_tokens-1].mean()  # scalar

                        # 指标2: 主语→动词的attention (如果位置已知)
                        subj_off = pair.get("subj_offset", 1)
                        verb_off = pair.get("verb_offset", 2)
                        if verb_off >= 0 and subj_off < n_tokens and verb_off < n_tokens:
                            subj_to_verb = attn[:, verb_off, subj_off].mean()
                        else:
                            subj_to_verb = 0.0

                        # 指标3: attention entropy (注意力分散度)
                        # 高熵 = 均匀关注; 低熵 = 集中关注
                        attn_last = attn[:, -1, :n_tokens]  # [n_heads, n_tokens]
                        attn_last_norm = attn_last / (attn_last.sum(axis=-1, keepdims=True) + 1e-10)
                        entropy = -np.sum(attn_last_norm * np.log(attn_last_norm + 1e-10), axis=-1).mean()

                        key = (li, label)
                        if li not in attn_diff_per_layer:
                            continue
                        attn_diff_per_layer[li].append({
                            "last_to_all": float(last_to_all),
                            "subj_to_verb": float(subj_to_verb),
                            "entropy": float(entropy),
                            "label": label,
                        })

                except Exception as e:
                    continue

        # 汇总: 比较A(label=0) vs B(label=1)的attention差异
        for li in sample_layers:
            data = attn_diff_per_layer[li]
            if len(data) < 4:
                continue

            data_A = [d for d in data if d["label"] == 0]
            data_B = [d for d in data if d["label"] == 1]

            if not data_A or not data_B:
                continue

            # A vs B的平均差异
            metrics = ["last_to_all", "subj_to_verb", "entropy"]
            layer_result = {}
            for m in metrics:
                vals_A = [d[m] for d in data_A]
                vals_B = [d[m] for d in data_B]
                mean_A = np.mean(vals_A)
                mean_B = np.mean(vals_B)
                # Cohen's d (效应量)
                pooled_std = np.sqrt((np.var(vals_A) + np.var(vals_B)) / 2 + 1e-10)
                cohens_d = abs(mean_A - mean_B) / pooled_std
                layer_result[m] = {
                    "mean_A": round(mean_A, 6),
                    "mean_B": round(mean_B, 6),
                    "cohens_d": round(cohens_d, 4),
                }

            if ctype not in results:
                results[ctype] = {}
            results[ctype][f"L{li}"] = layer_result

        # 打印摘要
        if ctype in results:
            print(f"  {ctype}:")
            for li in sample_layers:
                key = f"L{li}"
                if key in results[ctype]:
                    r = results[ctype][key]
                    d_last = r["last_to_all"]["cohens_d"]
                    d_sv = r["subj_to_verb"]["cohens_d"]
                    d_ent = r["entropy"]["cohens_d"]
                    print(f"    L{li}: Cohen's d (last_to_all={d_last:.3f}, "
                          f"subj→verb={d_sv:.3f}, entropy={d_ent:.3f})")

    return results


# ============================================================
# Exp3: 残差流分解 ★★★★
# ============================================================

def residual_decomposition(train_h, test_h, n_layers, d_model, sample_layers):
    """
    分解 h_{l+1} = h_l + Δh_l
    
    核心问题:
    - Δh_l携带约束信息吗?
    - Δh_l可预测吗?
    - 如果Δh_l不可预测但h_{l+1}可预测 → 只是惯性, 非动力学
    - 如果Δh_l可预测 → 存在真实动力学
    """
    print(f"\n{'='*60}")
    print("Exp3: Residual Stream Decomposition (Δh predictability)")
    print(f"{'='*60}")

    results = {}

    # PCA基 (用于降维)
    all_h_train = []
    for li in sample_layers:
        h = train_h.get(li)
        if h is not None:
            all_h_train.append(h)
    if len(all_h_train) < 2:
        return {}
    all_h_train = np.vstack(all_h_train)
    mean_h = all_h_train.mean(axis=0)
    h_centered = all_h_train - mean_h
    _, S_h, Vt_h = np.linalg.svd(h_centered, full_matrices=False)

    k = 20  # 用20维PCA
    P = Vt_h[:k, :]

    for li in sample_layers:
        if li + 1 >= n_layers:
            continue
        if train_h.get(li) is None or train_h.get(li+1) is None:
            continue
        if test_h.get(li) is None or test_h.get(li+1) is None:
            continue

        h_l_train = train_h[li]
        h_l1_train = train_h[li+1]
        h_l_test = test_h[li]
        h_l1_test = test_h[li+1]

        n_train = min(h_l_train.shape[0], h_l1_train.shape[0])
        n_test = min(h_l_test.shape[0], h_l1_test.shape[0])

        # Δh = h_{l+1} - h_l
        delta_train = h_l1_train[:n_train] - h_l_train[:n_train]
        delta_test = h_l1_test[:n_test] - h_l_test[:n_test]

        # 投影到PCA空间
        z_l_train = (h_l_train[:n_train] - mean_h) @ P.T
        z_l1_train = (h_l1_train[:n_train] - mean_h) @ P.T
        z_delta_train = (delta_train - mean_h) @ P.T

        z_l_test = (h_l_test[:n_test] - mean_h) @ P.T
        z_l1_test = (h_l1_test[:n_test] - mean_h) @ P.T
        z_delta_test = (delta_test - mean_h) @ P.T

        # R²( h_{l+1} | h_l ) — 全状态可预测性
        A_full, b_full = _fit_linear(z_l_train, z_l1_train)
        z_pred_full = z_l_test @ A_full.T + b_full
        r2_full = _compute_r2(z_l1_test, z_pred_full)

        # R²( Δh | h_l ) — 残差可预测性
        A_delta, b_delta = _fit_linear(z_l_train, z_delta_train)
        z_pred_delta = z_l_test @ A_delta.T + b_delta
        r2_delta = _compute_r2(z_delta_test, z_pred_delta)

        # Δh的相对大小: ||Δh|| / ||h||
        rel_size_train = np.mean(np.linalg.norm(delta_train, axis=1)) / (np.mean(np.linalg.norm(h_l_train[:n_train], axis=1)) + 1e-10)
        rel_size_test = np.mean(np.linalg.norm(delta_test, axis=1)) / (np.mean(np.linalg.norm(h_l_test[:n_test], axis=1)) + 1e-10)

        # Δh中约束信息的度量: 不同约束类型的Δh是否可区分
        # 简化: 用Δh的方差比 (signal/noise)
        delta_var = np.mean(np.var(delta_test, axis=0))
        delta_mean_var = np.var(delta_test.mean(axis=0))

        results[f"L{li}"] = {
            "r2_full": round(r2_full, 4),         # h_{l+1} = f(h_l) 的R²
            "r2_delta": round(r2_delta, 4),        # Δh = f(h_l) 的R²
            "rel_delta_size": round(float(rel_size_test), 6),  # ||Δh||/||h||
            "delta_variance": round(float(delta_var), 6),
        }

    # 打印
    print(f"  {'Layer':<8} {'R²(h_{l+1}|h_l)':<20} {'R²(Δh|h_l)':<15} {'||Δh||/||h||':<15}")
    for li in sample_layers:
        key = f"L{li}"
        if key in results:
            r = results[key]
            print(f"  L{li:<6} {r['r2_full']:<20} {r['r2_delta']:<15} {r['rel_delta_size']:<15}")

    return results


# ============================================================
# Exp4: 跨token约束耦合图 ★★★
# ============================================================

def cross_token_coupling(model, tokenizer, device, pairs_by_type,
                          n_layers, sample_layers):
    """
    对约束对, 计算主语token和动词token的hidden state变化是否耦合
    
    方法:
    1. 对SVA约束对, 分别提取主语位置和动词位置的hidden state
    2. 计算: singular→plural时, 主语位置的变化 vs 动词位置的变化
    3. 如果两者的变化方向相关 → 耦合
    """
    print(f"\n{'='*60}")
    print("Exp4: Cross-Token Constraint Coupling")
    print(f"{'='*60}")

    results = {}

    # 只测SVA (主语-动词位置明确)
    sva_pairs = pairs_by_type.get("number_sva", [])[:30]
    if not sva_pairs:
        print("  No SVA pairs available")
        return {}

    for li in sample_layers:
        subj_changes = []
        verb_changes = []

        for pair in sva_pairs:
            sent_A = pair["A"]
            sent_B = pair["B"]

            try:
                # 获取A和B的hidden states
                inputs_A = tokenizer(sent_A, return_tensors="pt",
                                      truncation=True, max_length=64)
                inputs_B = tokenizer(sent_B, return_tensors="pt",
                                      truncation=True, max_length=64)

                input_ids_A = inputs_A["input_ids"].to(device)
                input_ids_B = inputs_B["input_ids"].to(device)

                captured = {}
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0].detach().float().cpu()
                        else:
                            captured[key] = output.detach().float().cpu()
                    return hook

                layers = get_layers(model)
                hook = layers[li].register_forward_hook(make_hook(f"L{li}"))

                # Forward A
                with torch.no_grad():
                    _ = model(input_ids=input_ids_A, attention_mask=inputs_A["attention_mask"].to(device))
                h_A = captured[f"L{li}"][0].numpy() if f"L{li}" in captured else None  # [seq, d]

                # Forward B
                captured.clear()
                with torch.no_grad():
                    _ = model(input_ids=input_ids_B, attention_mask=inputs_B["attention_mask"].to(device))
                h_B = captured[f"L{li}"][0].numpy() if f"L{li}" in captured else None

                hook.remove()

                if h_A is None or h_B is None:
                    continue

                # 确保可以取主语和动词位置
                n_A = h_A.shape[0]
                n_B = h_B.shape[0]
                subj_off = pair.get("subj_offset", 1)
                verb_off = pair.get("verb_offset", 2)

                # 主语在A和B中的位置可能不同(singular vs plural添加了's')
                # A: "The cat chases" → [The, cat, chases]
                # B: "The cats chase" → [The, cats, chase]
                # 主语位置: A中1, B中1
                # 动词位置: A中2, B中2

                if subj_off < n_A and subj_off < n_B and verb_off < n_A and verb_off < n_B:
                    # 主语位置的变化
                    delta_subj = h_B[subj_off] - h_A[subj_off]
                    # 动词位置的变化
                    delta_verb = h_B[verb_off] - h_A[verb_off]

                    # 归一化
                    norm_s = np.linalg.norm(delta_subj) + 1e-10
                    norm_v = np.linalg.norm(delta_verb) + 1e-10
                    subj_changes.append(delta_subj / norm_s)
                    verb_changes.append(delta_verb / norm_v)

            except Exception as e:
                continue

        if len(subj_changes) < 5:
            continue

        subj_arr = np.array(subj_changes)  # [n, d]
        verb_arr = np.array(verb_changes)   # [n, d]

        # 计算耦合: 每对的主语变化和动词变化的余弦相似度
        cos_sims = []
        for i in range(len(subj_arr)):
            cs = np.dot(subj_arr[i], verb_arr[i]) / (
                np.linalg.norm(subj_arr[i]) * np.linalg.norm(verb_arr[i]) + 1e-10)
            cos_sims.append(cs)

        mean_cos = np.mean(cos_sims)

        # 随机基准: 打乱动词变化
        rng = np.random.RandomState(42)
        rand_cos = []
        for _ in range(100):
            perm = rng.permutation(len(verb_arr))
            for i in range(min(len(subj_arr), 20)):
                cs = np.dot(subj_arr[i], verb_arr[perm[i]]) / (
                    np.linalg.norm(subj_arr[i]) * np.linalg.norm(verb_arr[perm[i]]) + 1e-10)
                rand_cos.append(cs)
        rand_mean = np.mean(rand_cos)

        results[f"L{li}"] = {
            "coupling_cos": round(float(mean_cos), 4),
            "random_cos": round(float(rand_mean), 4),
            "excess": round(float(mean_cos - rand_mean), 4),
            "n_pairs": len(cos_sims),
        }

    # 打印
    print(f"  {'Layer':<8} {'Coupling':<12} {'Random':<12} {'Excess':<12}")
    for li in sample_layers:
        key = f"L{li}"
        if key in results:
            r = results[key]
            print(f"  L{li:<6} {r['coupling_cos']:<12} {r['random_cos']:<12} {r['excess']:<12}")

    return results


# ============================================================
# 辅助函数
# ============================================================

def _fit_linear(X, Y, lam=0.1):
    n = X.shape[0]
    ones = np.ones((n, 1))
    X_aug = np.hstack([X, ones])
    k_in = X.shape[1]
    k_out = Y.shape[1] if Y.ndim > 1 else 1
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    reg = lam * np.eye(k_in + 1)
    reg[-1, -1] = 0
    W = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ Y)
    A = W[:k_in, :].T
    b = W[k_in:, :].reshape(1, -1)
    return A, b


def _compute_r2(Y_true, Y_pred):
    ss_res = np.sum((Y_true - Y_pred)**2)
    ss_tot = np.sum((Y_true - Y_true.mean(axis=0))**2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1 - ss_res / ss_tot)


# ============================================================
# 主流程
# ============================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    model_key = {"qwen3": "qwen3", "glm4": "glm4", "deepseek7b": "deepseek7b",
                 "ds7b": "deepseek7b"}.get(model_name.lower(), model_name.lower())

    print(f"\n{'#'*70}")
    print(f"Phase 228: Constraint Transport & Linguistic Conserved Quantities — {model_key}")
    print(f"{'#'*70}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # === 1. 加载模型 ===
    model, tokenizer, device = load_model_eager(model_key)
    info = get_model_info(model, model_key)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  n_layers={n_layers}, d_model={d_model}")

    sample_layers = get_sample_layers(n_layers)

    # === 2. 生成约束对 ===
    all_pairs = generate_constraint_pairs_with_positions()

    # 按类型分组
    pairs_by_type = {}
    for p in all_pairs:
        ctype = p["type"]
        if ctype not in pairs_by_type:
            pairs_by_type[ctype] = []
        pairs_by_type[ctype].append(p)

    # 划分训练/测试 (每种类型: 前30训练, 后10测试)
    train_pairs_by_type = {}
    test_pairs_by_type = {}
    for ctype, pairs in pairs_by_type.items():
        train_pairs_by_type[ctype] = pairs[:30]
        test_pairs_by_type[ctype] = pairs[30:40]

    # 收集句子和标签
    train_sentences = []
    test_sentences = []
    train_labels = {ctype: [] for ctype in pairs_by_type}
    test_labels = {ctype: [] for ctype in pairs_by_type}

    for ctype in pairs_by_type:
        for pair in train_pairs_by_type[ctype]:
            train_sentences.extend([pair["A"], pair["B"]])
            train_labels[ctype].extend([pair["label_A"], pair["label_B"]])
        for pair in test_pairs_by_type[ctype]:
            test_sentences.extend([pair["A"], pair["B"]])
            test_labels[ctype].extend([pair["label_A"], pair["label_B"]])

    print(f"  Train: {len(train_sentences)} sentences, Test: {len(test_sentences)} sentences")
    print(f"  Constraint types: {list(pairs_by_type.keys())}")

    # === 3. 收集隐藏状态 (用于Exp1和Exp3) ===
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Collecting train hidden states...")
    train_h = _collect_hidden_states_simple(model, tokenizer, device,
                                             train_sentences, n_layers, "train")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Collecting test hidden states...")
    test_h = _collect_hidden_states_simple(model, tokenizer, device,
                                            test_sentences, n_layers, "test")

    # === 4. 运行实验 ===
    all_results = {"model": model_key, "d_model": d_model, "n_layers": n_layers}

    # Exp1: 语言守恒量
    try:
        all_results["exp1_conserved"] = linguistic_conserved_quantities(
            train_h, test_h, train_labels, test_labels,
            n_layers, d_model, sample_layers)
    except Exception as e:
        print(f"  Exp1 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_conserved"] = {"error": str(e)}

    # Exp2: 注意力约束输运
    try:
        all_results["exp2_attn_transport"] = attention_constraint_transport(
            model, tokenizer, device, pairs_by_type,
            n_layers, sample_layers)
    except Exception as e:
        print(f"  Exp2 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_attn_transport"] = {"error": str(e)}

    # Exp3: 残差流分解
    try:
        all_results["exp3_residual"] = residual_decomposition(
            train_h, test_h, n_layers, d_model, sample_layers)
    except Exception as e:
        print(f"  Exp3 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_residual"] = {"error": str(e)}

    # Exp4: 跨token约束耦合
    try:
        all_results["exp4_coupling"] = cross_token_coupling(
            model, tokenizer, device, pairs_by_type,
            n_layers, sample_layers)
    except Exception as e:
        print(f"  Exp4 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_coupling"] = {"error": str(e)}

    # === 5. 保存结果 ===
    out_path = OUTPUT_DIR / f"phase228_{model_key}_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")

    # === 6. 释放模型 ===
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nPhase 228 ({model_key}) complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _collect_hidden_states_simple(model, tokenizer, device, sentences, n_layers, desc):
    """简单收集hidden states (不需要attention)"""
    layers = get_layers(model)
    all_h = {l: [] for l in range(n_layers)}

    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook

    hooks = []
    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))

    for si, text in enumerate(sentences):
        captured.clear()
        try:
            input_ids = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=64).input_ids.to(device)
            with torch.no_grad():
                _ = model(input_ids)
            for li in range(n_layers):
                key = f"L{li}"
                if key in captured:
                    h = captured[key][0, -1, :].numpy()
                    all_h[li].append(h)
                else:
                    all_h[li].append(np.zeros(1))
        except Exception as e:
            for li in range(n_layers):
                all_h[li].append(np.zeros(1))

        if (si + 1) % 20 == 0:
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] {desc}: {si+1}/{len(sentences)} done")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    for li in range(n_layers):
        valid = [h for h in all_h[li] if h.shape != (1,)]
        if valid:
            all_h[li] = np.array(valid)
        else:
            all_h[li] = None

    return all_h


if __name__ == "__main__":
    main()
