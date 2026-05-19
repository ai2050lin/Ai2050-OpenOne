"""
Phase 231: Operator Mechanics (算子力学)
=========================================

Phase 230核心发现: 形容词方向不存在跨名词稳定性 (分离度仅1.0-1.2x)
→ 线性可组合特征假说被否定

Phase 231核心问题: 如果feature不是方向, 那它是什么?
→ 假说: feature是算子(operator/变换), 不是方向向量

关键数学区别:
  方向假说: h(red apple) ≈ h(apple) + v_red        (加法)
  算子假说: h(red apple) ≈ A_red · h(apple)         (线性变换)
  
如果算子假说成立: A_red应该在上下文间更稳定, 因为它捕捉的是"如何变换"而非"加什么"

4个实验:
  ExpA: 线性算子拟合 ★★★★★ — 决定性实验
        对每个形容词, 用N个名词拟合W_A, 使 W_A·h(noun) ≈ Δ(adj, noun)
        在held-out名词上测试泛化能力
        如果泛化 >> 方向模型 → feature是算子

  ExpB: 操作因果注入 ★★★★ — 修正Phase230的Exp4
        用大beta(50-500)在embedding层注入操作方向
        测量是否改变模型行为(续写→翻译等)

  ExpC: 预测回路发现 ★★★ — 行为级feature
        不研究"red"这种人类概念, 研究:
        - 否定翻转: "not"是否翻转概率分布?
        - 条件约束: "if"是否改变后续概率结构?
        - 疑问模式: "?"是否切换输出模式?
        这些是"预测修正器", 可能是真正的feature primitive

  ExpD: 算子非交换性验证 ★★★
        如果A_red和A_big是算子, 那么 A_red·A_big ≠ A_big·A_red?
        这是Phase 229非交换性发现在算子层面的验证

用法: python tests/glm5/phase231_operator_mechanics.py [qwen3|glm4|deepseek7b]
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from model_utils import (get_layers, get_model_info, release_model, 
                          get_sample_layers, MODEL_CONFIGS)


# ===== 数据定义 =====

# 形容词 (精选: Phase 230中最稳定和最不稳定的各一半)
ADJECTIVES = [
    # 评价类 (Phase 230中最稳定)
    "ugly", "beautiful", "dangerous", "safe", "clean",
    # 颜色类 (Phase 230中最不稳定)
    "red", "blue", "green", "white", "black",
    # 大小/物理类
    "big", "small", "hot", "cold", "heavy",
    # 情感/状态类
    "happy", "sad", "old", "new", "fast",
]

# 名词 — 扩大到40个, 其中30个训练, 10个测试
NOUNS_TRAIN = [
    "cat", "dog", "bird", "fish", "horse", "bear", "lion", "snake",
    "apple", "banana", "orange", "cake", "bread", "rice", "soup", "cheese",
    "mountain", "river", "tree", "flower", "ocean", "forest", "cloud", "stone",
    "house", "car", "bridge", "road", "building", "ship",
]

NOUNS_TEST = [
    "doctor", "teacher", "child", "woman", "knife", 
    "table", "music", "book", "fire", "wind",
]

# 操作 (Phase 230中分离度最高的)
OPERATIONS = [
    ("translate", "Translate to French:"),
    ("explain", "Explain why:"),
    ("summarize", "Summarize this:"),
    ("rewrite", "Rewrite this:"),
    ("negate", "State the opposite of:"),
]

OP_SENTENCES_TRAIN = [
    "The cat sat on the mat and looked out the window.",
    "Scientists discovered a new element in the laboratory.",
    "The river flows through the valley to the sea.",
    "She finished reading the book before dinner.",
    "The children played happily in the garden.",
    "A strong wind blew across the open field.",
    "The teacher explained the lesson to the students.",
    "He walked slowly along the dark corridor.",
    "The company launched a new product this year.",
    "Rain fell steadily throughout the long night.",
    "The artist painted a beautiful landscape scene.",
    "They built a small cabin near the lake.",
    "The musician played a soft melody on piano.",
    "We watched the sunset from the hilltop.",
    "The old man told stories about his youth.",
]

OP_SENTENCES_TEST = [
    "Birds sang in the trees every morning.",
    "The chef prepared a delicious meal.",
    "She wrote a letter to her old friend.",
    "The train arrived late at the station.",
    "Snow covered the mountains during winter.",
]

# 预测回路测试用句
PREDICTION_CIRCUIT_TEMPLATES = {
    "negation": {
        "affirmative": "The sky is blue and the weather is",
        "negated": "The sky is not blue and the weather is",
        "neutral": "The sky and the weather are both",
    },
    "conditional": {
        "factual": "The door is open so people can",
        "conditional": "If the door is open then people can",
        "neutral": "The door and people can",
    },
    "question": {
        "declarative": "The answer to this problem is",
        "interrogative": "What is the answer to this problem?",
        "neutral": "This problem has an answer that is",
    },
    "negation_strong": {
        "affirmative": "The bird can fly because it has",
        "negated": "The bird cannot fly because it has",
        "neutral": "The bird has wings and can",
    },
    "temporal": {
        "past": "Yesterday the scientists discovered that",
        "present": "Today the scientists discover that",
        "future": "Tomorrow the scientists will discover that",
    },
}


# ===== 模型加载 =====

def load_model_bf16(model_name):
    """BF16 + device_map="auto" 加载模型"""
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
        attn_implementation="eager",
    )
    model.eval()
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[load] {model_name}: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# ===== 通用: 提取各层hidden states =====

def extract_hidden_states_batch(model, tokenizer, device, texts, n_layers,
                                 position="last", max_length=64, log_interval=50):
    """批量提取每层hidden states"""
    all_hidden = {l: [] for l in range(n_layers)}
    
    for i, text in enumerate(texts):
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = toks["input_ids"].to(device)
        attn_mask = toks["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=True)
        
        hs = out.hidden_states
        
        if position == "last":
            mask_np = attn_mask[0].cpu().numpy()
            last_pos = np.where(mask_np == 1)[0][-1]
            for l in range(n_layers):
                h = hs[l][0, last_pos].float().cpu().numpy()
                all_hidden[l].append(h)
        elif position == "mean":
            for l in range(n_layers):
                mask = attn_mask[0].unsqueeze(-1).float()
                h = (hs[l][0] * mask.to(hs[l].device)).sum(dim=0) / mask.sum()
                all_hidden[l].append(h.float().cpu().numpy())
        
        del out, hs
        if (i + 1) % log_interval == 0:
            print(f"    extracted {i+1}/{len(texts)} texts", flush=True)
            if torch.cuda.is_available():
                print(f"    GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    
    for l in range(n_layers):
        all_hidden[l] = np.array(all_hidden[l])
    
    return all_hidden


def extract_logits_and_hidden(model, tokenizer, device, text, n_layers, max_length=64):
    """提取logits和所有层hidden states"""
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = toks["input_ids"].to(device)
    attn_mask = toks["attention_mask"].to(device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                   output_hidden_states=True)
    
    logits = out.logits[0, -1].float().cpu().numpy()
    hs = {}
    for l in range(n_layers):
        hs[l] = out.hidden_states[l][0, -1].float().cpu().numpy()
    
    del out
    return logits, hs


# ===== ExpA: 线性算子拟合 ★★★★★ =====

def expA_linear_operator_fitting(model, tokenizer, device, n_layers, d_model):
    """
    ★★★★★ 决定性实验: feature是方向还是算子?
    
    方向模型: Δ(adj, noun) ≈ v_adj (对所有noun相同)
    算子模型: Δ(adj, noun) ≈ W_adj · h(noun) (依赖于noun)
    
    方法:
    1. 对每个形容词A, 用训练名词拟合线性算子W_A
    2. 在测试名词上评估泛化能力
    3. 比较算子模型 vs 方向模型的预测R²
    
    如果算子R² >> 方向R² → feature是算子
    """
    print("\n" + "="*60)
    print("ExpA: Linear Operator Fitting (线性算子拟合) ★★★★★")
    print("="*60)
    print(f"  Adjectives: {len(ADJECTIVES)}, Train nouns: {len(NOUNS_TRAIN)}, Test nouns: {len(NOUNS_TEST)}")
    
    t0 = time.time()
    
    # 构造所有句子
    all_texts = []
    text_info = []  # (type, adj_idx, noun_idx)
    
    # 训练集
    for ai, adj in enumerate(ADJECTIVES):
        for ni, noun in enumerate(NOUNS_TRAIN):
            all_texts.append(f"The {adj} {noun}.")
            text_info.append(("train_adj_noun", ai, ni))
            all_texts.append(f"The {noun}.")
            text_info.append(("train_noun", ai, ni))
    
    # 测试集
    for ai, adj in enumerate(ADJECTIVES):
        for ni, noun in enumerate(NOUNS_TEST):
            all_texts.append(f"The {adj} {noun}.")
            text_info.append(("test_adj_noun", ai, ni))
            all_texts.append(f"The {noun}.")
            text_info.append(("test_noun", ai, ni))
    
    n_total = len(all_texts)
    print(f"  Total texts: {n_total}")
    print(f"  Extracting hidden states...", flush=True)
    
    # 提取hidden states (采样关键层)
    sample_layers = get_sample_layers(n_layers, n_samples=10)
    print(f"  Sample layers: {sample_layers}")
    
    hs_all = extract_hidden_states_batch(model, tokenizer, device, all_texts, n_layers,
                                          log_interval=100)
    
    print(f"  Hidden states extracted in {time.time()-t0:.1f}s", flush=True)
    
    # 分析每层
    results = {}
    
    for l in sample_layers:
        h = hs_all[l]  # [n_total, d_model]
        
        # 重组数据
        train_h_noun = {}   # adj_idx -> [n_train, d_model]
        train_h_an = {}     # adj_idx -> [n_train, d_model]
        test_h_noun = {}
        test_h_an = {}
        
        idx = 0
        for ai in range(len(ADJECTIVES)):
            train_h_noun[ai] = []
            train_h_an[ai] = []
            for ni in range(len(NOUNS_TRAIN)):
                train_h_an[ai].append(h[idx]); idx += 1
                train_h_noun[ai].append(h[idx]); idx += 1
            
            test_h_noun[ai] = []
            test_h_an[ai] = []
            for ni in range(len(NOUNS_TEST)):
                test_h_an[ai].append(h[idx]); idx += 1
                test_h_noun[ai].append(h[idx]); idx += 1
        
        # 对每个形容词拟合算子
        operator_results = {}
        
        for ai, adj in enumerate(ADJECTIVES):
            X_train = np.array(train_h_noun[ai])  # [n_train, d_model]
            Y_train = np.array(train_h_an[ai]) - np.array(train_h_noun[ai])  # Δ = h(adj+noun) - h(noun)
            X_test = np.array(test_h_noun[ai])
            Y_test = np.array(test_h_an[ai]) - np.array(test_h_noun[ai])
            
            n_train = X_train.shape[0]
            n_test = X_test.shape[0]
            
            # --- 方向模型: Δ ≈ v_adj (均值方向) ---
            v_dir = Y_train.mean(axis=0)  # [d_model]
            
            # 方向模型预测: 对每个noun, 预测Δ = v_dir
            dir_pred_train = np.tile(v_dir, (n_train, 1))
            dir_pred_test = np.tile(v_dir, (n_test, 1))
            
            dir_r2_train = r2_score(Y_train, dir_pred_train)
            dir_r2_test = r2_score(Y_test, dir_pred_test)
            
            # 方向模型的cosine稳定性 (Phase 230的结果)
            dir_cos_train = []
            dir_cos_test = []
            v_norm = np.linalg.norm(v_dir)
            if v_norm > 1e-10:
                v_hat = v_dir / v_norm
                for i in range(n_train):
                    yi_norm = np.linalg.norm(Y_train[i])
                    if yi_norm > 1e-10:
                        dir_cos_train.append(float(np.dot(Y_train[i], v_hat) / yi_norm))
                for i in range(n_test):
                    yi_norm = np.linalg.norm(Y_test[i])
                    if yi_norm > 1e-10:
                        dir_cos_test.append(float(np.dot(Y_test[i], v_hat) / yi_norm))
            
            # --- 算子模型: Δ ≈ W_adj · h(noun) ---
            # Ridge回归: Y = X @ W.T, 即 W = Ridge(Y, X)
            # 但d_model可能很大(~2560), 30个训练样本不够
            # 解决: 用低秩近似, 先PCA降维到k维, 拟合后投影回原空间
            # 或者: 直接拟合 W 使得 W@h_noun ≈ Δ
            # W是 [d_model, d_model], 太大! n_train=30 << d_model
            # 
            # 更实际的方案: 对每个输出维度d, 拟合 w_d 使得 w_d·h_noun ≈ Δ[d]
            # 即 d_model个独立的Ridge回归, 每个是 [d_model] -> [1]
            # 这样每个回归的参数量=d_model, 样本量=30, 仍不够
            #
            # 最实际的方案: 降维
            # 用训练数据的PCA, 将h_noun投影到k=20维, 在低维空间拟合
            
            k_dim = min(20, n_train - 1, d_model)  # PCA维度
            
            # PCA on training noun representations
            h_mean = X_train.mean(axis=0)
            X_centered = X_train - h_mean
            X_test_centered = X_test - h_mean
            
            # SVD for PCA
            U_pca, s_pca, Vt_pca = np.linalg.svd(X_centered, full_matrices=False)
            # Vt_pca: [min(n,d), d_model], 取前k行作为主成分
            pc = Vt_pca[:k_dim]  # [k, d_model]
            
            X_train_pca = X_centered @ pc.T  # [n_train, k]
            X_test_pca = X_test_centered @ pc.T  # [n_test, k]
            
            # Ridge回归: 对每个输出维度
            # Y_train: [n_train, d_model]
            # 但d_model=2560太大了, 我们只对Δ的主成分拟合
            # 或者: 也对Y做PCA, 然后在双PCA空间中拟合
            
            # 对Y也做PCA
            Y_mean = Y_train.mean(axis=0)
            Y_centered = Y_train - Y_mean
            Y_test_centered = Y_test - Y_mean
            
            U_y, s_y, Vt_y = np.linalg.svd(Y_centered, full_matrices=False)
            pc_y = Vt_y[:k_dim]  # [k, d_model]
            
            Y_train_pca = Y_centered @ pc_y.T  # [n_train, k]
            Y_test_pca = Y_test_centered @ pc_y.T  # [n_test, k]
            
            # Ridge回归: Y_pca ≈ X_pca @ W_pca.T
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_pca, Y_train_pca)
            
            op_pred_train_pca = ridge.predict(X_train_pca)  # [n_train, k]
            op_pred_test_pca = ridge.predict(X_test_pca)    # [n_test, k]
            
            # 投影回原空间
            op_pred_train = op_pred_train_pca @ pc_y + Y_mean  # [n_train, d_model]
            op_pred_test = op_pred_test_pca @ pc_y + Y_mean    # [n_test, d_model]
            
            op_r2_train = r2_score(Y_train, op_pred_train)
            op_r2_test = r2_score(Y_test, op_pred_test)
            
            # 也在PCA空间中计算R²
            op_r2_pca_train = r2_score(Y_train_pca, op_pred_train_pca)
            op_r2_pca_test = r2_score(Y_test_pca, op_pred_test_pca)
            
            # 方向模型在PCA空间中的R²
            dir_pred_train_pca = (dir_pred_train - Y_mean) @ pc_y.T
            dir_pred_test_pca = (dir_pred_test - Y_mean) @ pc_y.T
            dir_r2_pca_train = r2_score(Y_train_pca, dir_pred_train_pca)
            dir_r2_pca_test = r2_score(Y_test_pca, dir_pred_test_pca)
            
            # 逐样本cosine similarity for operator prediction
            op_cos_test = []
            for i in range(n_test):
                pred_norm = np.linalg.norm(op_pred_test[i])
                true_norm = np.linalg.norm(Y_test[i])
                if pred_norm > 1e-10 and true_norm > 1e-10:
                    op_cos_test.append(float(np.dot(op_pred_test[i], Y_test[i]) / (pred_norm * true_norm)))
            
            operator_results[adj] = {
                "dir_r2_train": float(dir_r2_train),
                "dir_r2_test": float(dir_r2_test),
                "op_r2_train": float(op_r2_train),
                "op_r2_test": float(op_r2_test),
                "dir_r2_pca_train": float(dir_r2_pca_train),
                "dir_r2_pca_test": float(dir_r2_pca_test),
                "op_r2_pca_train": float(op_r2_pca_train),
                "op_r2_pca_test": float(op_r2_pca_test),
                "dir_cos_train": float(np.mean(dir_cos_train)) if dir_cos_train else 0.0,
                "dir_cos_test": float(np.mean(dir_cos_test)) if dir_cos_test else 0.0,
                "op_cos_test": float(np.mean(op_cos_test)) if op_cos_test else 0.0,
                "pca_dim": k_dim,
                "delta_norm": float(np.mean(np.linalg.norm(Y_train, axis=1))),
                "var_explained_y": float(np.sum(s_y[:k_dim]**2) / (np.sum(s_y**2) + 1e-20)) if len(s_y) > 0 else 0,
                "var_explained_x": float(np.sum(s_pca[:k_dim]**2) / (np.sum(s_pca**2) + 1e-20)) if len(s_pca) > 0 else 0,
            }
        
        # 汇总
        dir_r2_tests = [operator_results[a]["dir_r2_test"] for a in ADJECTIVES]
        op_r2_tests = [operator_results[a]["op_r2_test"] for a in ADJECTIVES]
        dir_r2_pca_tests = [operator_results[a]["dir_r2_pca_test"] for a in ADJECTIVES]
        op_r2_pca_tests = [operator_results[a]["op_r2_pca_test"] for a in ADJECTIVES]
        dir_cos_tests = [operator_results[a]["dir_cos_test"] for a in ADJECTIVES]
        op_cos_tests = [operator_results[a]["op_cos_test"] for a in ADJECTIVES]
        
        # 逐类别汇总
        cat_results = {}
        categories = {
            "evaluative": ADJECTIVES[:5],
            "color": ADJECTIVES[5:10],
            "physical": ADJECTIVES[10:15],
            "state": ADJECTIVES[15:20],
        }
        for cat_name, cat_adjs in categories.items():
            cat_dir_r2 = [operator_results[a]["dir_r2_test"] for a in cat_adjs]
            cat_op_r2 = [operator_results[a]["op_r2_test"] for a in cat_adjs]
            cat_dir_cos = [operator_results[a]["dir_cos_test"] for a in cat_adjs]
            cat_op_cos = [operator_results[a]["op_cos_test"] for a in cat_adjs]
            cat_results[cat_name] = {
                "dir_r2": float(np.mean(cat_dir_r2)),
                "op_r2": float(np.mean(cat_op_r2)),
                "op_advantage": float(np.mean(cat_op_r2) - np.mean(cat_dir_r2)),
                "dir_cos": float(np.mean(cat_dir_cos)),
                "op_cos": float(np.mean(cat_op_cos)),
            }
        
        results[f"L{l}"] = {
            "mean_dir_r2_test": float(np.mean(dir_r2_tests)),
            "mean_op_r2_test": float(np.mean(op_r2_tests)),
            "mean_dir_r2_pca_test": float(np.mean(dir_r2_pca_tests)),
            "mean_op_r2_pca_test": float(np.mean(op_r2_pca_tests)),
            "op_advantage": float(np.mean(op_r2_tests) - np.mean(dir_r2_tests)),
            "op_advantage_pca": float(np.mean(op_r2_pca_tests) - np.mean(dir_r2_pca_tests)),
            "mean_dir_cos_test": float(np.mean(dir_cos_tests)),
            "mean_op_cos_test": float(np.mean(op_cos_tests)),
            "per_adj": operator_results,
            "per_category": cat_results,
        }
        
        print(f"  L{l:2d}: dir_R²={np.mean(dir_r2_tests):.4f}, op_R²={np.mean(op_r2_tests):.4f}, "
              f"advantage={np.mean(op_r2_tests)-np.mean(dir_r2_tests):.4f}, "
              f"dir_cos={np.mean(dir_cos_tests):.4f}, op_cos={np.mean(op_cos_tests):.4f}", flush=True)
    
    print(f"  ExpA completed in {time.time()-t0:.1f}s")
    return results


# ===== ExpB: 操作因果注入 =====

def expB_operation_causal_injection(model, tokenizer, device, n_layers, d_model):
    """
    ★★★★ 因果验证: 注入操作方向是否改变模型行为?
    
    Phase 230的Exp4用β=5无效。这次用β=50-500, 在embedding层注入。
    
    测试: 注入Δ_translate方向, 模型输出是否从"续写"变为"翻译"?
    """
    print("\n" + "="*60)
    print("ExpB: Operation Causal Injection (操作因果注入) ★★★★")
    print("="*60)
    
    t0 = time.time()
    
    # 先提取操作方向 (用训练句子)
    print("  Extracting operation directions...", flush=True)
    
    base_texts = OP_SENTENCES_TRAIN[:5]  # 用5个句子计算方向
    op_directions = {}  # op_name -> {layer: direction}
    
    # 获取各层的baseline和操作hidden states
    sample_layers = get_sample_layers(n_layers, n_samples=6)
    
    for op_name, op_prefix in OPERATIONS:
        hs_base = extract_hidden_states_batch(
            model, tokenizer, device, base_texts, n_layers, log_interval=10)
        hs_op = extract_hidden_states_batch(
            model, tokenizer, device, [f"{op_prefix} {t}" for t in base_texts], n_layers, log_interval=10)
        
        op_directions[op_name] = {}
        for l in sample_layers:
            delta_mean = (hs_op[l] - hs_base[l]).mean(axis=0)
            norm = np.linalg.norm(delta_mean)
            if norm > 1e-10:
                op_directions[op_name][l] = delta_mean / norm
            else:
                op_directions[op_name][l] = delta_mean
        
        del hs_base, hs_op
    
    # 因果注入测试
    test_prompts = [
        "The scientist walked into the",
        "Yesterday I went to the store and",
        "The little cat sat on the",
    ]
    
    betas = [10, 50, 100, 200, 500]
    target_layer_idx = sample_layers[len(sample_layers)//2]  # 中间层
    target_layer_idx_deep = sample_layers[-1]  # 深层
    
    results = {}
    
    for op_name in ["translate", "explain", "negate"]:
        results[op_name] = {}
        
        for layer_idx in [target_layer_idx, target_layer_idx_deep]:
            direction = op_directions[op_name].get(layer_idx)
            if direction is None:
                continue
            
            results[op_name][f"L{layer_idx}"] = {}
            
            for prompt in test_prompts:
                # Baseline: 正常推理
                toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = toks["input_ids"].to(device)
                attn_mask = toks["attention_mask"].to(device)
                
                with torch.no_grad():
                    base_out = model(input_ids=input_ids, attention_mask=attn_mask)
                base_logits = base_out.logits[0, -1].float().cpu().numpy()
                base_top10 = np.argsort(base_logits)[-10:][::-1]
                base_top10_tokens = [tokenizer.decode([t]).strip() for t in base_top10]
                base_probs_full = torch.softmax(base_out.logits[0, -1].float(), dim=-1).cpu().numpy()
                base_top10_probs = base_probs_full[base_top10.copy()]
                
                del base_out
                
                # 注入: 在embedding层last token位置注入
                embed_layer = model.get_input_embeddings()
                inputs_embeds_base = embed_layer(input_ids).detach().clone()
                
                inject_results = {}
                for beta in betas:
                    direction_t = torch.tensor(
                        beta * direction, 
                        dtype=inputs_embeds_base.dtype, 
                        device=device
                    )
                    inputs_embeds_inj = inputs_embeds_base.clone()
                    inputs_embeds_inj[0, -1, :] += direction_t.to(model.dtype)
                    
                    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                    
                    with torch.no_grad():
                        inj_out = model(inputs_embeds=inputs_embeds_inj, position_ids=position_ids)
                    inj_logits = inj_out.logits[0, -1].float().cpu().numpy()
                    inj_top10 = np.argsort(inj_logits)[-10:][::-1]
                    inj_top10_tokens = [tokenizer.decode([t]).strip() for t in inj_top10]
                    inj_probs_full = torch.softmax(inj_out.logits[0, -1].float(), dim=-1).cpu().numpy()
                    inj_top10_probs = inj_probs_full[inj_top10.copy()]
                    
                    # 计算KL散度和top-1变化
                    kl_div = float(np.sum(base_probs_full * np.log(base_probs_full / (inj_probs_full + 1e-10) + 1e-10)))
                    
                    # top-1是否改变?
                    top1_changed = base_top10_tokens[0] != inj_top10_tokens[0]
                    
                    # 与操作相关的token出现?
                    op_related = False
                    if op_name == "translate":
                        op_related = any(t.lower() in " ".join(inj_top10_tokens).lower() 
                                        for t in ["french", "english", "translation", "traduire"])
                    elif op_name == "explain":
                        op_related = any(t.lower() in " ".join(inj_top10_tokens).lower() 
                                        for t in ["because", "reason", "since", "explain"])
                    elif op_name == "negate":
                        op_related = any(t.lower() in " ".join(inj_top10_tokens).lower() 
                                        for t in ["not", "no", "never", "neither"])
                    
                    inject_results[f"beta_{beta}"] = {
                        "kl_divergence": float(kl_div),
                        "top1_changed": bool(top1_changed),
                        "op_related_token": bool(op_related),
                        "base_top3": base_top10_tokens[:3],
                        "inj_top3": inj_top10_tokens[:3],
                        "base_top3_probs": [float(p) for p in base_top10_probs[:3]],
                        "inj_top3_probs": [float(p) for p in inj_top10_probs[:3]],
                        "logit_diff_norm": float(np.linalg.norm(inj_logits - base_logits)),
                    }
                    
                    del inj_out
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                results[op_name][f"L{layer_idx}"][prompt[:30]] = inject_results
                
                del inputs_embeds_base
    
    print(f"  ExpB completed in {time.time()-t0:.1f}s")
    return results


# ===== ExpC: 预测回路发现 =====

def expC_prediction_circuits(model, tokenizer, device, n_layers, d_model):
    """
    ★★★ 行为级feature: 不是"red"这种人类概念, 而是"预测修正器"
    
    测试:
    - 否定翻转: "not"是否翻转top-k概率?
    - 条件约束: "if"是否改变概率结构?
    - 疑问模式: "?"是否切换输出分布?
    - 时态约束: 时态标记是否约束动词形式?
    """
    print("\n" + "="*60)
    print("ExpC: Prediction Circuits (预测回路发现) ★★★")
    print("="*60)
    
    t0 = time.time()
    sample_layers = get_sample_layers(n_layers, n_samples=6)
    
    results = {}
    
    for circuit_name, templates in PREDICTION_CIRCUIT_TEMPLATES.items():
        print(f"  Testing circuit: {circuit_name}", flush=True)
        results[circuit_name] = {}
        
        # 提取各变体的logits和hidden states
        variants = {}
        for var_name, text in templates.items():
            logits, hs = extract_logits_and_hidden(model, tokenizer, device, text, n_layers)
            variants[var_name] = {"logits": logits, "hs": hs}
        
        # 分析: 关键词概率变化
        for var_name, var_data in variants.items():
            top10 = np.argsort(var_data["logits"])[-10:][::-1]
            top10_tokens = [tokenizer.decode([t]).strip() for t in top10]
            top10_probs_all = torch.softmax(torch.tensor(var_data["logits"]), dim=-1).numpy()
            top10_probs = top10_probs_all[top10.copy()]
            results[circuit_name][var_name] = {
                "top10_tokens": top10_tokens,
                "top10_probs": [float(p) for p in top10_probs],
            }
        
        # 核心: 计算概率分布的变化
        # 否定翻转: affirmative的top-1概率在negated中是否降低?
        if "affirmative" in variants and "negated" in variants:
            aff_logits = variants["affirmative"]["logits"]
            neg_logits = variants["negated"]["logits"]
            
            aff_probs = torch.softmax(torch.tensor(aff_logits), dim=-1).numpy()
            neg_probs = torch.softmax(torch.tensor(neg_logits), dim=-1).numpy()
            
            # 是否翻转? 看affirmative的top-10在negated中是否降低
            aff_top10 = np.argsort(aff_logits)[-10:][::-1]
            flip_ratios = []
            for tok_id in aff_top10:
                if aff_probs[tok_id] > 1e-6:
                    flip_ratios.append(float(neg_probs[tok_id] / aff_probs[tok_id]))
            
            # KL散度
            kl_div = float(np.sum(aff_probs * np.log(aff_probs / (neg_probs + 1e-10) + 1e-10)))
            
            results[circuit_name]["negation_analysis"] = {
                "flip_ratios_mean": float(np.mean(flip_ratios)),
                "flip_ratios_median": float(np.median(flip_ratios)),
                "kl_divergence": kl_div,
                "prob_suppression": float(np.mean([r for r in flip_ratios if r < 1.0])) if any(r < 1.0 for r in flip_ratios) else 0,
            }
        
        # 时态约束: 过去/现在/未来的动词形式概率
        if circuit_name == "temporal":
            past_logits = variants["past"]["logits"]
            pres_logits = variants["present"]["logits"]
            fut_logits = variants["future"]["logits"]
            
            past_probs = torch.softmax(torch.tensor(past_logits), dim=-1).numpy()
            pres_probs = torch.softmax(torch.tensor(pres_logits), dim=-1).numpy()
            fut_probs = torch.softmax(torch.tensor(fut_logits), dim=-1).numpy()
            
            results[circuit_name]["temporal_analysis"] = {
                "past_pres_kl": float(np.sum(past_probs * np.log(past_probs / (pres_probs + 1e-10) + 1e-10))),
                "past_fut_kl": float(np.sum(past_probs * np.log(past_probs / (fut_probs + 1e-10) + 1e-10))),
                "pres_fut_kl": float(np.sum(pres_probs * np.log(pres_probs / (fut_probs + 1e-10) + 1e-10))),
            }
        
        # Hidden state差异分析
        for var_name, var_data in variants.items():
            for other_name, other_data in variants.items():
                if var_name >= other_name:
                    continue
                key = f"hs_dist_{var_name}_vs_{other_name}"
                for l in sample_layers:
                    h1 = var_data["hs"][l]
                    h2 = other_data["hs"][l]
                    cos = float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-10))
                    eucl = float(np.linalg.norm(h1 - h2))
                    results[circuit_name].setdefault(key, {})[f"L{l}"] = {
                        "cosine": cos,
                        "euclidean": eucl,
                    }
    
    print(f"  ExpC completed in {time.time()-t0:.1f}s")
    return results


# ===== ExpD: 算子非交换性验证 =====

def expD_operator_noncommutativity(model, tokenizer, device, n_layers, d_model):
    """
    ★★★ 如果A_red和A_big是算子, 那么 A_red·A_big ≠ A_big·A_red?
    
    方法:
    1. 提取 h("The red big cat."), h("The big red cat."), h("The cat.")
    2. 计算顺序差异: h(red big cat) - h(cat) vs h(big red cat) - h(cat)
    3. 测试: 是否 red(big(cat)) ≠ big(red(cat))?
    
    用线性算子近似: 
      Δ_RB = h(red big noun) - h(big noun)  (先big再red)
      Δ_BR = h(big red noun) - h(red noun)  (先red再big)
    """
    print("\n" + "="*60)
    print("ExpD: Operator Non-commutativity (算子非交换性) ★★★")
    print("="*60)
    
    t0 = time.time()
    
    # 选几对非交换形容词
    adj_pairs = [
        ("red", "big"), ("red", "hot"), ("beautiful", "old"),
        ("dangerous", "fast"), ("ugly", "new"), ("clean", "small"),
        ("happy", "young"), ("cold", "heavy"), ("blue", "tall"),
        ("safe", "slow"),
    ]
    
    test_nouns = NOUNS_TEST[:5]  # 5个名词
    
    # 构造所有句子
    all_texts = []
    text_info = []
    
    for adj1, adj2 in adj_pairs:
        for noun in test_nouns:
            # adj1 adj2 noun
            all_texts.append(f"The {adj1} {adj2} {noun}.")
            text_info.append(("A1A2N", adj1, adj2, noun))
            # adj2 adj1 noun
            all_texts.append(f"The {adj2} {adj1} {noun}.")
            text_info.append(("A2A1N", adj1, adj2, noun))
            # adj1 noun (for Δ calculation)
            all_texts.append(f"The {adj1} {noun}.")
            text_info.append(("A1N", adj1, adj2, noun))
            # adj2 noun
            all_texts.append(f"The {adj2} {noun}.")
            text_info.append(("A2N", adj1, adj2, noun))
            # noun only
            all_texts.append(f"The {noun}.")
            text_info.append(("N", adj1, adj2, noun))
    
    n_total = len(all_texts)
    print(f"  Total texts: {n_total}")
    
    sample_layers = get_sample_layers(n_layers, n_samples=8)
    print(f"  Sample layers: {sample_layers}")
    
    hs_all = extract_hidden_states_batch(model, tokenizer, device, all_texts, n_layers,
                                          log_interval=50)
    
    results = {}
    
    for l in sample_layers:
        h = hs_all[l]
        
        # 重组
        idx = 0
        pair_results = {}
        
        for pi, (adj1, adj2) in enumerate(adj_pairs):
            for ni, noun in enumerate(test_nouns):
                h_a1a2n = h[idx]; idx += 1
                h_a2a1n = h[idx]; idx += 1
                h_a1n = h[idx]; idx += 1
                h_a2n = h[idx]; idx += 1
                h_n = h[idx]; idx += 1
                
                # Δ for "先adj2再adj1" = h(adj1, adj2, noun) - h(adj2, noun)
                delta_a1_after_a2 = h_a1a2n - h_a2n
                # Δ for "先adj1再adj2" = h(adj2, adj1, noun) - h(adj1, noun)
                delta_a2_after_a1 = h_a2a1n - h_a1n
                
                # Δ for 单独的 adj1 = h(adj1, noun) - h(noun)
                delta_a1_alone = h_a1n - h_n
                # Δ for 单独的 adj2 = h(adj2, noun) - h(noun)
                delta_a2_alone = h_a2n - h_n
                
                # 非交换性度量:
                # 1. 顺序差异: ||(A1A2N - A2N) - (A1N - N)|| vs ||(A2A1N - A1N) - (A2N - N)||
                # 即: adj1在adj2之后的Δ vs adj1单独的Δ
                context_dep_a1 = delta_a1_after_a2 - delta_a1_alone
                context_dep_a2 = delta_a2_after_a1 - delta_a2_alone
                
                # 2. 总非交换距离: ||h(A1A2N) - h(A2A1N)||
                noncomm_dist = float(np.linalg.norm(h_a1a2n - h_a2a1n))
                
                # 3. 顺序与baseline的距离
                a1_alone_norm = float(np.linalg.norm(delta_a1_alone))
                a1_after_a2_norm = float(np.linalg.norm(delta_a1_after_a2))
                
                # cosine between "adj1 alone" and "adj1 after adj2"
                norm1 = np.linalg.norm(delta_a1_alone)
                norm2 = np.linalg.norm(delta_a1_after_a2)
                cos_a1_alone_vs_after = float(np.dot(delta_a1_alone, delta_a1_after_a2) / (norm1 * norm2)) if norm1 > 1e-10 and norm2 > 1e-10 else 0.0
                
                pair_key = f"{adj1}_{adj2}_{noun}"
                pair_results[pair_key] = {
                    "noncomm_dist": noncomm_dist,
                    "context_dep_a1_norm": float(np.linalg.norm(context_dep_a1)),
                    "context_dep_a2_norm": float(np.linalg.norm(context_dep_a2)),
                    "cos_a1_alone_vs_after_a2": cos_a1_alone_vs_after,
                    "delta_a1_alone_norm": a1_alone_norm,
                    "delta_a1_after_a2_norm": a1_after_a2_norm,
                }
        
        # 汇总
        noncomm_dists = [v["noncomm_dist"] for v in pair_results.values()]
        context_deps = [v["context_dep_a1_norm"] for v in pair_results.values()]
        cos_context_deps = [v["cos_a1_alone_vs_after_a2"] for v in pair_results.values()]
        alone_norms = [v["delta_a1_alone_norm"] for v in pair_results.values()]
        after_norms = [v["delta_a1_after_a2_norm"] for v in pair_results.values()]
        
        # 按pair平均
        pair_summary = {}
        for pi, (adj1, adj2) in enumerate(adj_pairs):
            pair_dists = []
            pair_cos = []
            for ni, noun in enumerate(test_nouns):
                key = f"{adj1}_{adj2}_{noun}"
                if key in pair_results:
                    pair_dists.append(pair_results[key]["noncomm_dist"])
                    pair_cos.append(pair_results[key]["cos_a1_alone_vs_after_a2"])
            pair_summary[f"{adj1}_{adj2}"] = {
                "mean_noncomm_dist": float(np.mean(pair_dists)),
                "mean_cos_context_dep": float(np.mean(pair_cos)),
            }
        
        results[f"L{l}"] = {
            "mean_noncomm_dist": float(np.mean(noncomm_dists)),
            "mean_context_dep_norm": float(np.mean(context_deps)),
            "mean_cos_context_dep": float(np.mean(cos_context_deps)),
            "mean_alone_norm": float(np.mean(alone_norms)),
            "mean_after_norm": float(np.mean(after_norms)),
            "context_dep_ratio": float(np.mean(context_deps) / (np.mean(alone_norms) + 1e-10)),
            "per_pair": pair_summary,
        }
        
        print(f"  L{l:2d}: noncomm_dist={np.mean(noncomm_dists):.4f}, "
              f"cos_context_dep={np.mean(cos_context_deps):.4f}, "
              f"context_dep_ratio={np.mean(context_deps)/(np.mean(alone_norms)+1e-10):.4f}", flush=True)
    
    print(f"  ExpD completed in {time.time()-t0:.1f}s")
    return results


# ===== 主函数 =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print(f"\n{'='*60}")
    print(f"Phase 231: Operator Mechanics (算子力学)")
    print(f"Model: {model_name}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\nModel info: class={info.model_class}, n_layers={n_layers}, "
          f"d_model={d_model}, vocab={info.vocab_size}")
    
    all_results = {
        "model": model_name,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "n_layers": n_layers,
        "d_model": d_model,
    }
    
    # ExpA: 线性算子拟合 ★★★★★
    try:
        all_results["expA"] = expA_linear_operator_fitting(model, tokenizer, device, n_layers, d_model)
    except Exception as e:
        print(f"  ExpA FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["expA"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  After ExpA: GPU={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    
    # ExpB: 操作因果注入 ★★★★
    try:
        all_results["expB"] = expB_operation_causal_injection(model, tokenizer, device, n_layers, d_model)
    except Exception as e:
        print(f"  ExpB FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["expB"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  After ExpB: GPU={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    
    # ExpC: 预测回路发现 ★★★
    try:
        all_results["expC"] = expC_prediction_circuits(model, tokenizer, device, n_layers, d_model)
    except Exception as e:
        print(f"  ExpC FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["expC"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  After ExpC: GPU={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    
    # ExpD: 算子非交换性 ★★★
    try:
        all_results["expD"] = expD_operator_noncommutativity(model, tokenizer, device, n_layers, d_model)
    except Exception as e:
        print(f"  ExpD FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["expD"] = {"error": str(e)}
    
    # 释放模型
    release_model(model)
    model = None
    
    # 保存结果
    out_path = f"tests/glm5_temp/phase231_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("Phase 231 Summary")
    print(f"{'='*60}")
    
    if "expA" in all_results and "error" not in all_results["expA"]:
        expA = all_results["expA"]
        best_layer = max(expA.keys(), key=lambda k: expA[k].get("op_advantage", 0))
        best = expA[best_layer]
        print(f"\n  ExpA (Operator Fitting):")
        print(f"    Best layer: {best_layer}")
        print(f"    Direction R²: {best['mean_dir_r2_test']:.4f}")
        print(f"    Operator R²: {best['mean_op_r2_test']:.4f}")
        print(f"    Operator advantage: {best['op_advantage']:.4f}")
        print(f"    Direction cos: {best['mean_dir_cos_test']:.4f}")
        print(f"    Operator cos: {best['mean_op_cos_test']:.4f}")
        if "per_category" in best:
            for cat, cd in best["per_category"].items():
                print(f"    {cat}: dir_R²={cd['dir_r2']:.4f}, op_R²={cd['op_r2']:.4f}, adv={cd['op_advantage']:.4f}")
    
    if "expC" in all_results and "error" not in all_results["expC"]:
        expC = all_results["expC"]
        print(f"\n  ExpC (Prediction Circuits):")
        for circuit_name, circuit_data in expC.items():
            if "negation_analysis" in circuit_data:
                na = circuit_data["negation_analysis"]
                print(f"    {circuit_name}: flip_ratio={na['flip_ratios_mean']:.4f}, "
                      f"KL={na['kl_divergence']:.4f}")
    
    print(f"\nPhase 231 completed!")


if __name__ == "__main__":
    main()
