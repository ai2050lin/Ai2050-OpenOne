"""
Phase 229: Relational Constraint Mechanics (关系约束力学)
========================================================

核心假说: h_l不是真正状态变量(R²(Δh|h_l)≈0),
         真正传播的是"关系闭合性" — R²(R(h_{l+1})|R(h_l)) >> R²(Δh|h_l)

4个实验:
  Exp1: 关系评分传播 — probe score的层间可预测性 vs Δh的可预测性
  Exp2: 未来状态空间压缩 — token对next-token熵的压缩能力
  Exp3: 约束闭合误差 — 正确vs错误句子的约束闭合性层序演化
  Exp4: 约束操作非交换性 — AB vs BA的hidden state距离

用法: python tests/glm5/phase229_relational_constraint_mechanics.py [qwen3|glm4|deepseek7b]
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import json
import numpy as np
import torch
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from model_utils import (get_layers, get_model_info, release_model, MODEL_CONFIGS)

# ===== 数据生成 =====

def generate_constraint_sentences(n=250):
    """生成大量约束句子对, 每种约束类型n对"""
    sentences = []
    
    # 1. Subject-Verb Agreement (SVA)
    sva_templates = [
        # Singular subject + singular verb (correct)
        ("The cat {} quietly on the mat", "sits", "sit", "singular", "correct"),
        ("A bird {} in the tall tree", "sings", "sing", "singular", "correct"),
        ("The dog {} after the ball", "runs", "run", "singular", "correct"),
        ("The child {} to the music", "dances", "dance", "singular", "correct"),
        ("The man {} his car every day", "washes", "wash", "singular", "correct"),
        ("The woman {} the book carefully", "reads", "read", "singular", "correct"),
        ("The student {} the answer quickly", "writes", "write", "singular", "correct"),
        ("The fish {} in the clear water", "swims", "swim", "singular", "correct"),
        ("The horse {} across the field", "gallops", "gallop", "singular", "correct"),
        ("The doctor {} the patient gently", "treats", "treat", "singular", "correct"),
        # Plural subject + plural verb (correct)
        ("The cats {} quietly on the mat", "sit", "sits", "plural", "correct"),
        ("The birds {} in the tall tree", "sing", "sings", "plural", "correct"),
        ("The dogs {} after the ball", "run", "runs", "plural", "correct"),
        ("The children {} to the music", "dance", "dances", "plural", "correct"),
        ("The men {} their cars every day", "wash", "washes", "plural", "correct"),
        ("The women {} the book carefully", "read", "reads", "plural", "correct"),
        ("The students {} the answer quickly", "write", "writes", "plural", "correct"),
        ("The fish {} in the clear water", "swim", "swims", "plural", "correct"),
        ("The horses {} across the field", "gallop", "gallops", "plural", "correct"),
        ("The doctors {} the patient gently", "treat", "treats", "plural", "correct"),
    ]
    
    # 2. Tense consistency
    tense_templates = [
        ("Yesterday she {} to the store", "walked", "walks", "past", "correct"),
        ("Last week they {} the project", "finished", "finish", "past", "correct"),
        ("He {} his homework last night", "completed", "completes", "past", "correct"),
        ("The team {} the game yesterday", "won", "wins", "past", "correct"),
        ("She {} a letter to her friend", "wrote", "writes", "past", "correct"),
        ("Today she {} to the store", "walks", "walked", "present", "correct"),
        ("Now they {} the project", "finish", "finished", "present", "correct"),
        ("He {} his homework every night", "completes", "completed", "present", "correct"),
        ("The team always {} the game", "wins", "won", "present", "correct"),
        ("She {} letters to her friends", "writes", "wrote", "present", "correct"),
    ]
    
    # 3. Negation scope
    neg_templates = [
        ("The cat did not {} on the mat", "sit", "sitting", "negated", "correct"),
        ("The dog will not {} the ball", "chase", "chasing", "negated", "correct"),
        ("She does not {} the answer", "know", "knowing", "negated", "correct"),
        ("They cannot {} the problem", "solve", "solving", "negated", "correct"),
        ("He should not {} so fast", "drive", "driving", "negated", "correct"),
        # Non-negated controls
        ("The cat will {} on the mat", "sit", "sitting", "affirmative", "correct"),
        ("The dog will {} the ball", "chase", "chasing", "affirmative", "correct"),
        ("She does {} the answer", "know", "knowing", "affirmative", "correct"),
        ("They can {} the problem", "solve", "solving", "affirmative", "correct"),
        ("He should {} carefully", "drive", "driving", "affirmative", "correct"),
    ]
    
    # 4. Non-commutativity pairs (AB vs BA)
    noncomm_templates = [
        ("not always", "always not"),
        ("not really", "really not"),
        ("not quite", "quite not"),
        ("not very", "very not"),
        ("not just", "just not"),
        ("not only", "only not"),
        ("never really", "really never"),
        ("not exactly", "exactly not"),
        ("not completely", "completely not"),
        ("not entirely", "entirely not"),
    ]
    
    for template, correct, wrong, label, status in sva_templates:
        sentences.append({
            "type": "sva",
            "correct": template.format(correct),
            "wrong": template.format(wrong),
            "label": label,
            "constraint": "number_sva",
            "verb_pos": len(template.split("{}")[0].split())  # verb position
        })
    
    for template, correct, wrong, label, status in tense_templates:
        sentences.append({
            "type": "tense",
            "correct": template.format(correct),
            "wrong": template.format(wrong),
            "label": label,
            "constraint": "tense",
            "verb_pos": len(template.split("{}")[0].split())
        })
    
    for template, correct, wrong, label, status in neg_templates:
        sentences.append({
            "type": "negation",
            "correct": template.format(correct),
            "wrong": template.format(wrong),
            "label": label,
            "constraint": "negation",
            "verb_pos": len(template.split("{}")[0].split())
        })
    
    noncomm_data = []
    for ab, ba in noncomm_templates:
        noncomm_data.append({
            "type": "noncommutative",
            "sentence_ab": f"The result is {ab} what we expected",
            "sentence_ba": f"The result is {ba} what we expected",
        })
    
    return sentences, noncomm_data


# ===== 模型加载 =====

def load_model_bf16(model_name):
    """BF16 + device_map=auto加载, 不用8bit"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bfloat16 + device_map=auto)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",  # flash_attn not installed, use eager
    )
    model.eval()
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[load] {model_name}: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# ===== Exp1: 关系评分传播 =====

def exp1_relational_score_propagation(model, tokenizer, device, sentences, n_layers, d_model):
    """
    核心实验: 关系评分的层间可预测性 vs Δh的可预测性
    
    方法:
    1. 在每层训练逻辑回归探针, 预测约束标签(singular/plural, past/present, negated/affirmative)
    2. 获取每层的探针概率分数(连续值)
    3. 计算 R²(probe_score_{l+1} | probe_score_l) vs R²(Δh | h_l)
    4. 如果 R²(probe_score) >> R²(Δh), 则关系评分比hidden state更有动力学闭包
    """
    print("\n" + "="*60)
    print("Exp1: Relational Score Propagation (关系评分传播)")
    print("="*60)
    
    # 收集每层的hidden states
    sample_layers = list(range(n_layers))
    
    all_data = defaultdict(lambda: {"h": [], "labels": []})
    
    processed = 0
    for sent in sentences:
        if sent["type"] == "noncommutative":
            continue
        
        text = sent["correct"]
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = toks["input_ids"].to(device)
        attn_mask = toks["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=True)
        
        # 提取verb位置的hidden states
        verb_pos = min(sent["verb_pos"], input_ids.shape[1] - 1)
        
        for li in sample_layers:
            h = out.hidden_states[li + 1][0, verb_pos].float().cpu().numpy()  # [d_model]
            all_data[sent["constraint"]]["h"].append(h)
            all_data[sent["constraint"]]["labels"].append(sent["label"])
        
        processed += 1
        if processed % 20 == 0:
            print(f"  [Exp1] Processed {processed}/{len(sentences)} sentences...")
            torch.cuda.empty_cache()
    
    print(f"  [Exp1] Total sentences processed: {processed}")
    
    # 对每种约束类型, 在每层训练探针, 记录score
    results = {}
    
    for constraint_type in ["number_sva", "tense", "negation"]:
        data = all_data[constraint_type]
        if len(data["h"]) < 10:
            print(f"  [Exp1] Skipping {constraint_type}: too few samples ({len(data['h'])})")
            continue
        
        H = np.array(data["h"])  # [n_samples * n_layers, d_model]
        labels = np.array(data["labels"])
        n_total = len(labels)
        
        # 检查标签是否有足够的两类
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print(f"  [Exp1] Skipping {constraint_type}: only 1 label class")
            continue
        
        print(f"  [Exp1] {constraint_type}: {n_total} samples, labels={dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # 逐层探针: 每层的样本数 = n_sentences (因为每个句子在每个层都有一个h)
        # 重塑: [n_sentences, n_layers, d_model]
        n_sents = n_total // n_layers
        if n_sents * n_layers != n_total:
            # 不整除, 截断
            n_sents = n_total // n_layers
            H = H[:n_sents * n_layers]
            labels = labels[:n_sents * n_layers]
        
        H_3d = H.reshape(n_sents, n_layers, d_model)
        labels_2d = labels.reshape(n_sents, n_layers)
        
        # 每层的label应该相同(同一句子的约束标签在所有层相同)
        layer_labels = labels_2d[:, 0]  # [n_sents]
        
        # 逐层训练探针
        probe_scores = np.zeros((n_sents, n_layers))  # 每层每个样本的探针概率分数
        
        for li in range(n_layers):
            h_l = H_3d[:, li, :]  # [n_sents, d_model]
            
            # 5折交叉验证
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores = np.zeros(n_sents)
            
            for train_idx, test_idx in kf.split(h_l, layer_labels):
                clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
                clf.fit(h_l[train_idx], layer_labels[train_idx])
                # 获取正类的概率分数
                proba = clf.predict_proba(h_l[test_idx])
                # 找到正类列
                pos_class = clf.classes_[1] if len(clf.classes_) > 1 else clf.classes_[0]
                pos_idx = list(clf.classes_).index(pos_class)
                fold_scores[test_idx] = proba[:, pos_idx]
            
            probe_scores[:, li] = fold_scores
        
        # 计算关系评分的层间可预测性
        score_r2_list = []
        for li in range(n_layers - 1):
            score_l = probe_scores[:, li]
            score_l1 = probe_scores[:, li + 1]
            # 线性回归 R²
            r2 = r2_score(score_l1, np.polyval(np.polyfit(score_l, score_l1, 1), score_l))
            score_r2_list.append(r2)
        
        # 计算Δh的可预测性 (对照)
        delta_r2_list = []
        for li in range(n_layers - 1):
            h_l = H_3d[:, li, :]
            h_l1 = H_3d[:, li + 1, :]
            delta_h = h_l1 - h_l
            # R²(Δh | h_l): 用h_l的PCA投影预测Δh
            from sklearn.decomposition import PCA
            k = min(20, d_model, n_sents - 1)
            pca = PCA(n_components=k)
            z_l = pca.fit_transform(h_l)
            # 线性回归: delta_h = z_l @ W + b
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=1.0)
            reg.fit(z_l, delta_h)
            delta_pred = reg.predict(z_l)
            # 总R²
            ss_res = np.sum((delta_h - delta_pred) ** 2)
            ss_tot = np.sum((delta_h - delta_h.mean(axis=0)) ** 2)
            r2_delta = 1 - ss_res / max(ss_tot, 1e-10)
            delta_r2_list.append(r2_delta)
        
        # 计算R²(h_{l+1} | h_l) (另一个对照)
        full_r2_list = []
        for li in range(n_layers - 1):
            h_l = H_3d[:, li, :]
            h_l1 = H_3d[:, li + 1, :]
            from sklearn.decomposition import PCA
            k = min(20, d_model, n_sents - 1)
            pca = PCA(n_components=k)
            z_l = pca.fit_transform(h_l)
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=1.0)
            reg.fit(z_l, h_l1)
            h_pred = reg.predict(z_l)
            ss_res = np.sum((h_l1 - h_pred) ** 2)
            ss_tot = np.sum((h_l1 - h_l1.mean(axis=0)) ** 2)
            r2_full = 1 - ss_res / max(ss_tot, 1e-10)
            full_r2_list.append(r2_full)
        
        results[constraint_type] = {
            "probe_score_r2": score_r2_list,
            "delta_h_r2": delta_r2_list,
            "full_h_r2": full_r2_list,
            "n_samples": n_sents,
        }
        
        # 打印摘要
        print(f"\n  [{constraint_type}] Probe score R² summary:")
        for phase_name, lo, hi in [("Shallow(L0-5)", 0, min(6, n_layers-1)),
                                    ("Middle(L6-18)", 6, min(19, n_layers-1)),
                                    ("Deep(L18+)", 18, n_layers-1)]:
            score_seg = score_r2_list[lo:hi]
            delta_seg = delta_r2_list[lo:hi]
            full_seg = full_r2_list[lo:hi]
            if score_seg:
                print(f"    {phase_name}: R²(score)={np.mean(score_seg):.4f}, "
                      f"R²(Δh)={np.mean(delta_seg):.4f}, R²(h)={np.mean(full_seg):.4f}")
        
        # 关键比较: probe score R² vs Δh R²
        mean_score = np.mean(score_r2_list)
        mean_delta = np.mean(delta_r2_list)
        mean_full = np.mean(full_r2_list)
        print(f"  → Overall: R²(score)={mean_score:.4f}, R²(Δh)={mean_delta:.4f}, R²(h)={mean_full:.4f}")
        if mean_score > mean_delta:
            print(f"  ★ R²(score) > R²(Δh): 关系评分比Δh更可预测! 差={mean_score-mean_delta:.4f}")
        else:
            print(f"  ✗ R²(score) <= R²(Δh): 关系评分不比Δh更可预测")
    
    return results


# ===== Exp2: 未来状态空间压缩 =====

def exp2_state_space_compression(model, tokenizer, device, sentences, n_layers):
    """
    测量token对next-token熵的压缩能力
    
    核心思想: Meaning(x) = ΔH(future|x) — 强约束token极大压缩未来空间
    """
    print("\n" + "="*60)
    print("Exp2: State Space Compression (未来状态空间压缩)")
    print("="*60)
    
    # 定义关键token组
    constraint_tokens = {
        "strong_constraint": ["not", "never", "if", "all", "no", "must", "only", "every", "cannot", "neither"],
        "weak_constraint": ["the", "a", "an", "this", "that", "is", "was", "are", "were", "has"],
        "medium_constraint": ["because", "although", "unless", "since", "while", "therefore", "however", "but", "and", "or"],
    }
    
    # 构造测试句子: [prefix] [target_token] [continuation]
    test_pairs = []
    for group_name, tokens in constraint_tokens.items():
        for token in tokens:
            # 用通用前缀
            prefix = f"The result was"
            sentence = f"{prefix} {token}"
            test_pairs.append({
                "group": group_name,
                "token": token,
                "prefix": prefix,
                "sentence": sentence,
                "prefix_only": prefix,
            })
    
    print(f"  [Exp2] Testing {len(test_pairs)} token pairs across {len(constraint_tokens)} groups")
    
    # 对每个pair计算熵
    results = defaultdict(list)
    
    for i, pair in enumerate(test_pairs):
        # 计算prefix的next-token熵 (baseline)
        prefix_ids = tokenizer(pair["prefix_only"], return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out_prefix = model(input_ids=prefix_ids["input_ids"].to(device),
                              attention_mask=prefix_ids["attention_mask"].to(device))
        logits_prefix = out_prefix.logits[0, -1].float().cpu().numpy()
        prob_prefix = np.exp(logits_prefix - np.max(logits_prefix))
        prob_prefix = prob_prefix / prob_prefix.sum()
        entropy_prefix = -np.sum(prob_prefix * np.log(prob_prefix + 1e-10))
        
        # 计算prefix+token的next-token熵
        sent_ids = tokenizer(pair["sentence"], return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out_sent = model(input_ids=sent_ids["input_ids"].to(device),
                            attention_mask=sent_ids["attention_mask"].to(device))
        logits_sent = out_sent.logits[0, -1].float().cpu().numpy()
        prob_sent = np.exp(logits_sent - np.max(logits_sent))
        prob_sent = prob_sent / prob_sent.sum()
        entropy_sent = -np.sum(prob_sent * np.log(prob_sent + 1e-10))
        
        delta_entropy = entropy_sent - entropy_prefix  # 负值 = 压缩
        
        results[pair["group"]].append({
            "token": pair["token"],
            "entropy_before": float(entropy_prefix),
            "entropy_after": float(entropy_sent),
            "delta_entropy": float(delta_entropy),
            "compression_ratio": float(entropy_sent / max(entropy_prefix, 1e-10)),
        })
        
        if (i + 1) % 10 == 0:
            print(f"  [Exp2] Processed {i+1}/{len(test_pairs)} tokens...")
    
    # 汇总
    summary = {}
    for group_name, entries in results.items():
        deltas = [e["delta_entropy"] for e in entries]
        ratios = [e["compression_ratio"] for e in entries]
        summary[group_name] = {
            "mean_delta_entropy": float(np.mean(deltas)),
            "std_delta_entropy": float(np.std(deltas)),
            "mean_compression_ratio": float(np.mean(ratios)),
            "tokens": entries,
        }
        print(f"\n  [{group_name}]:")
        print(f"    ΔH = {np.mean(deltas):.4f} ± {np.std(deltas):.4f}")
        print(f"    Compression ratio = {np.mean(ratios):.4f}")
        # 列出最强/最弱的压缩token
        sorted_entries = sorted(entries, key=lambda x: x["delta_entropy"])
        print(f"    Strongest compressor: {sorted_entries[0]['token']} (ΔH={sorted_entries[0]['delta_entropy']:.4f})")
        print(f"    Weakest compressor: {sorted_entries[-1]['token']} (ΔH={sorted_entries[-1]['delta_entropy']:.4f})")
    
    return summary


# ===== Exp3: 约束闭合误差 =====

def exp3_constraint_closure_error(model, tokenizer, device, sentences, n_layers, d_model):
    """
    测量正确vs错误句子的约束闭合性层序演化
    
    核心思想: 
    - 正确句子: 约束闭合误差 E_R(l) 在深层趋近0
    - 错误句子: 约束闭合误差 E_R(l) 不趋0
    
    E_R(l) = |probe_score_subject(l) - probe_score_verb(l)|
    """
    print("\n" + "="*60)
    print("Exp3: Constraint Closure Error (约束闭合误差)")
    print("="*60)
    
    # 只用SVA类型的句子 (有明确的subject和verb)
    sva_sents = [s for s in sentences if s["constraint"] == "number_sva"]
    
    if len(sva_sents) < 5:
        print("  [Exp3] Not enough SVA sentences, skipping")
        return {}
    
    print(f"  [Exp3] Testing {len(sva_sents)} SVA sentence pairs")
    
    # 对正确和错误句子分别提取hidden states
    correct_hs = defaultdict(list)  # {layer_idx: [h_subject, h_verb]}
    wrong_hs = defaultdict(list)
    
    for sent in sva_sents:
        for is_correct, text_key in [(True, "correct"), (False, "wrong")]:
            text = sent[text_key]
            toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = toks["input_ids"].to(device)
            attn_mask = toks["attention_mask"].to(device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)
            
            # 找subject位置 (通常在位置1或2) 和verb位置
            tokens_str = [tokenizer.decode([t]) for t in input_ids[0].cpu().numpy()]
            
            # 简化: subject=position 1 (The之后的词), verb=verb_pos
            subj_pos = min(1, input_ids.shape[1] - 1)
            verb_pos = min(sent["verb_pos"], input_ids.shape[1] - 1)
            
            for li in range(n_layers):
                h_subj = out.hidden_states[li + 1][0, subj_pos].float().cpu().numpy()
                h_verb = out.hidden_states[li + 1][0, verb_pos].float().cpu().numpy()
                
                if is_correct:
                    correct_hs[li].append((h_subj, h_verb))
                else:
                    wrong_hs[li].append((h_subj, h_verb))
            
            torch.cuda.empty_cache()
    
    # 在每层训练number probe, 计算闭合误差
    results = {"correct": {}, "wrong": {}}
    
    # 需要足够的样本才能训练探针
    n_correct = len(correct_hs[0]) if 0 in correct_hs else 0
    n_wrong = len(wrong_hs[0]) if 0 in wrong_hs else 0
    
    print(f"  [Exp3] Samples: correct={n_correct}, wrong={n_wrong}")
    
    if n_correct < 10 or n_wrong < 10:
        print("  [Exp3] Not enough samples for probe training, using cosine distance instead")
        
        # 退而求其次: 用subject和verb的余弦距离作为闭合性的代理
        for li in range(n_layers):
            if li in correct_hs and len(correct_hs[li]) > 0:
                cos_correct = [np.dot(s, v) / (np.linalg.norm(s) * np.linalg.norm(v) + 1e-10)
                              for s, v in correct_hs[li]]
                results["correct"][li] = {"cosine_similarity": float(np.mean(cos_correct))}
            
            if li in wrong_hs and len(wrong_hs[li]) > 0:
                cos_wrong = [np.dot(s, v) / (np.linalg.norm(s) * np.linalg.norm(v) + 1e-10)
                            for s, v in wrong_hs[li]]
                results["wrong"][li] = {"cosine_similarity": float(np.mean(cos_wrong))}
        
        # 打印摘要
        print(f"\n  [Exp3] Subject-Verb Cosine Similarity (proxy for closure):")
        for phase_name, lo, hi in [("Shallow(L0-5)", 0, min(6, n_layers)),
                                    ("Middle(L6-18)", 6, min(19, n_layers)),
                                    ("Deep(L18+)", 18, n_layers)]:
            cos_c = [results["correct"][li]["cosine_similarity"] 
                     for li in range(lo, hi) if li in results["correct"]]
            cos_w = [results["wrong"][li]["cosine_similarity"]
                     for li in range(lo, hi) if li in results["wrong"]]
            if cos_c and cos_w:
                print(f"    {phase_name}: correct={np.mean(cos_c):.4f}, wrong={np.mean(cos_w):.4f}, "
                      f"Δ={np.mean(cos_c)-np.mean(cos_w):.4f}")
        
        return results
    
    # 有足够样本: 训练probe
    for li in range(n_layers):
        # 收集所有subject和verb的hidden states
        all_subj_h = []
        all_verb_h = []
        all_labels = []  # 0=correct, 1=wrong
        
        for s, v in correct_hs[li]:
            all_subj_h.append(s)
            all_verb_h.append(v)
            all_labels.append(0)
        
        for s, v in wrong_hs[li]:
            all_subj_h.append(s)
            all_verb_h.append(v)
            all_labels.append(1)
        
        all_subj_h = np.array(all_subj_h)
        all_verb_h = np.array(all_verb_h)
        all_labels = np.array(all_labels)
        
        # 训练number probe on subject positions
        # 标签: singular vs plural (从sentence labels获取)
        sva_labels = []
        for sent in sva_sents:
            sva_labels.append(1 if sent["label"] == "plural" else 0)
            sva_labels.append(1 if sent["label"] == "plural" else 0)  # wrong has same label
        
        # 简化: 计算subject-verb hidden state距离
        h_diff_correct = [np.abs(s - v) for s, v in correct_hs[li]]
        h_diff_wrong = [np.abs(s - v) for s, v in wrong_hs[li]]
        
        # 闭合误差 = ||h_subj - h_verb|| (对于同number应该小)
        closure_error_correct = [np.linalg.norm(d) for d in h_diff_correct]
        closure_error_wrong = [np.linalg.norm(d) for d in h_diff_wrong]
        
        results["correct"][li] = {
            "closure_error": float(np.mean(closure_error_correct)),
            "closure_std": float(np.std(closure_error_correct)),
        }
        results["wrong"][li] = {
            "closure_error": float(np.mean(closure_error_wrong)),
            "closure_std": float(np.std(closure_error_wrong)),
        }
    
    # 打印摘要
    print(f"\n  [Exp3] Closure Error ||h_subj - h_verb||:")
    for phase_name, lo, hi in [("Shallow(L0-5)", 0, min(6, n_layers)),
                                ("Middle(L6-18)", 6, min(19, n_layers)),
                                ("Deep(L18+)", 18, n_layers)]:
        err_c = [results["correct"][li]["closure_error"]
                 for li in range(lo, hi) if li in results["correct"]]
        err_w = [results["wrong"][li]["closure_error"]
                 for li in range(lo, hi) if li in results["wrong"]]
        if err_c and err_w:
            print(f"    {phase_name}: correct={np.mean(err_c):.4f}, wrong={np.mean(err_w):.4f}, "
                  f"Δ={np.mean(err_w)-np.mean(err_c):.4f}")
    
    return results


# ===== Exp4: 约束操作非交换性 =====

def exp4_noncommutativity(model, tokenizer, device, noncomm_data, n_layers):
    """
    测量约束操作的非交换性
    
    核心思想: AB ≠ BA → 语言操作可能是非交换群上的作用
    """
    print("\n" + "="*60)
    print("Exp4: Constraint Operation Non-commutativity (约束操作非交换性)")
    print("="*60)
    
    print(f"  [Exp4] Testing {len(noncomm_data)} non-commutative pairs")
    
    results = []
    
    for i, pair in enumerate(noncomm_data):
        # AB
        toks_ab = tokenizer(pair["sentence_ab"], return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out_ab = model(input_ids=toks_ab["input_ids"].to(device),
                          attention_mask=toks_ab["attention_mask"].to(device),
                          output_hidden_states=True)
        
        # BA
        toks_ba = tokenizer(pair["sentence_ba"], return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out_ba = model(input_ids=toks_ba["input_ids"].to(device),
                          attention_mask=toks_ba["attention_mask"].to(device),
                          output_hidden_states=True)
        
        # 计算每层的cosine距离 (last token)
        layer_distances = {}
        for li in range(n_layers):
            h_ab = out_ab.hidden_states[li + 1][0, -1].float().cpu().numpy()
            h_ba = out_ba.hidden_states[li + 1][0, -1].float().cpu().numpy()
            cos_dist = 1 - np.dot(h_ab, h_ba) / (np.linalg.norm(h_ab) * np.linalg.norm(h_ba) + 1e-10)
            layer_distances[li] = float(cos_dist)
        
        results.append({
            "ab": pair["sentence_ab"],
            "ba": pair["sentence_ba"],
            "layer_distances": layer_distances,
            "mean_distance": float(np.mean(list(layer_distances.values()))),
        })
        
        if (i + 1) % 5 == 0:
            print(f"  [Exp4] Processed {i+1}/{len(noncomm_data)} pairs...")
            torch.cuda.empty_cache()
    
    # 汇总
    print(f"\n  [Exp4] Results:")
    for r in results:
        dists = list(r["layer_distances"].values())
        # 找最大距离的层
        max_layer = max(r["layer_distances"], key=r["layer_distances"].get)
        print(f"    '{r['ab']}' vs '{r['ba']}': "
              f"mean_dist={np.mean(dists):.6f}, max_dist={np.max(dists):.6f} @ L{max_layer}")
    
    # 按层汇总
    layer_means = defaultdict(list)
    for r in results:
        for li, d in r["layer_distances"].items():
            layer_means[li].append(d)
    
    print(f"\n  [Exp4] Layer-wise mean cosine distance:")
    for phase_name, lo, hi in [("Shallow(L0-5)", 0, min(6, n_layers)),
                                ("Middle(L6-18)", 6, min(19, n_layers)),
                                ("Deep(L18+)", 18, n_layers)]:
        dists = [d for li in range(lo, hi) for d in layer_means.get(li, [])]
        if dists:
            print(f"    {phase_name}: mean={np.mean(dists):.6f}")
    
    # 判断: 距离是否显著大于0?
    all_dists = [r["mean_distance"] for r in results]
    mean_dist = np.mean(all_dists)
    print(f"\n  → Overall mean distance: {mean_dist:.6f}")
    if mean_dist > 0.01:
        print(f"  ★ AB ≠ BA confirmed! Mean distance > 0.01")
    else:
        print(f"  ✗ Distance too small to confirm non-commutativity")
    
    return results


# ===== 主函数 =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'#'*60}")
    print(f"Phase 229: Relational Constraint Mechanics — {model_name}")
    print(f"{'#'*60}")
    
    # 生成数据
    sentences, noncomm_data = generate_constraint_sentences()
    print(f"Data: {len(sentences)} constraint sentences, {len(noncomm_data)} non-commutative pairs")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"Model: {info.model_class}, {n_layers} layers, d_model={d_model}")
    
    all_results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}
    
    # Exp1: 关系评分传播
    try:
        r1 = exp1_relational_score_propagation(model, tokenizer, device, sentences, n_layers, d_model)
        all_results["exp1"] = r1
    except Exception as e:
        print(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    torch.cuda.empty_cache()
    
    # Exp2: 未来状态空间压缩
    try:
        r2 = exp2_state_space_compression(model, tokenizer, device, sentences, n_layers)
        all_results["exp2"] = r2
    except Exception as e:
        print(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    torch.cuda.empty_cache()
    
    # Exp3: 约束闭合误差
    try:
        r3 = exp3_constraint_closure_error(model, tokenizer, device, sentences, n_layers, d_model)
        all_results["exp3"] = r3
    except Exception as e:
        print(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    torch.cuda.empty_cache()
    
    # Exp4: 约束操作非交换性
    try:
        r4 = exp4_noncommutativity(model, tokenizer, device, noncomm_data, n_layers)
        all_results["exp4"] = r4
    except Exception as e:
        print(f"Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    # 保存结果
    import os
    os.makedirs("tests/glm5_temp", exist_ok=True)
    
    # 转换numpy类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj
    
    out_path = f"tests/glm5_temp/phase229_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    
    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\nPhase 229 ({model_name}) complete!")


if __name__ == "__main__":
    main()
