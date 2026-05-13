"""
Phase 154: 约束动力学 — 从向量几何到约束传播
=============================================

核心理论转向:
- 不再追踪向量方向(cos, PCA), 追踪约束结构
- 修正Phase 153的CCA问题(置换检验)
- 研究: logit排序稳定性、约束传播、边界流场、路由图动力学

五大实验:
  Exp 1: CCA置换检验 — CCA≈1是真信号还是高维伪相关?
  Exp 2: Logit排序稳定性 — top-k ordering跨层保持(核心约束!)
  Exp 3: 约束传播图 — 语法/逻辑约束如何跨层传播
  Exp 4: 边界流场 — logit边界如何逐层演化
  Exp 5: 注意力路由图 — 动态图结构+社区检测

用法:
  python tests/glm5/phase154_constraint_dynamics.py qwen3
  python tests/glm5/phase154_constraint_dynamics.py deepseek7b
  python tests/glm5/phase154_constraint_dynamics.py glm4
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS)
from collections import defaultdict

OUTPUT_DIR = Path("tests/glm5_temp")

# ============================================================
# 大规模测试语料 — 200句, 覆盖多种语法/语义约束
# ============================================================
GRAMMAR_PROMPTS = [
    # Subject-verb agreement (主谓一致)
    "The cat sits on the",
    "The cats sit on the",
    "The dog runs toward the",
    "The dogs run toward the",
    "The bird flies over the",
    "The birds fly over the",
    "The child plays in the",
    "The children play in the",
    "The woman walks to the",
    "The women walk to the",
    # Negation scope (否定辖域)
    "The student did not finish the",
    "No one believed that the",
    "She never mentioned the",
    "They hardly noticed the",
    "He barely touched the",
    # Tense consistency (时态一致)
    "Yesterday she went to the",
    "Tomorrow they will visit the",
    "Right now he is reading the",
    "Last week the team completed the",
    "Next year the company will launch the",
    # Conditional (条件句)
    "If it rains tomorrow then the",
    "If she studies hard then the",
    "Unless the weather improves the",
    "Provided that the results are positive the",
]

ATTRIBUTE_PROMPTS = [
    # Attribute binding (属性绑定)
    "The red apple was placed on the",
    "The blue car drove past the",
    "The tall building stood near the",
    "The small bird sat on the",
    "The old man walked to the",
    "The young woman entered the",
    "The hot coffee spilled on the",
    "The cold wind blew through the",
    "The bright light shone on the",
    "The dark room contained a",
]

LOGIC_PROMPTS = [
    # Causal/logical reasoning (因果/逻辑)
    "Because it was raining the",
    "Since the evidence was clear the",
    "Although the task was difficult the",
    "While the first option was safer the",
    "Therefore the committee decided to",
    "Consequently the researchers concluded that",
    "However the alternative approach would",
    "Moreover the additional data showed that",
    "Nevertheless the team continued to",
    "Thus the final result indicated that",
]

COREF_PROMPTS = [
    # Coreference (指代)
    "Mary gave Jane the book because she",
    "The manager told the employee that he",
    "After Anna met Lisa she decided to",
    "When the teacher asked the student he",
    "Although John helped Mary he",
]

GENERAL_PROMPTS = [
    "The scientist discovered that the",
    "In the morning she decided to",
    "The book on the table was about",
    "After the rain stopped the children",
    "The most important thing about science is",
    "When the sun sets over the ocean",
    "She walked into the room and saw",
    "The professor explained that the theory",
    "Despite the challenges the team managed",
    "The ancient city was known for its",
    "He realized that the answer was",
    "The relationship between language and thought",
    "Every morning she would read the",
    "The experiment showed that the results",
    "Music has the power to change how",
    "The government announced that the new policy",
    "In the future artificial intelligence will",
    "The philosopher argued that consciousness is",
    "After years of research they found that",
    "The key difference between the two approaches is",
    "The cat sat on the windowsill and watched",
    "Through the telescope they observed a new",
    "The river flowed gently through the valley",
    "She opened the letter and read the",
    "The painting on the wall depicted a",
    "During the concert the audience was",
    "The invention changed the way people",
    "He wrote a letter to his friend about",
    "The students in the classroom were learning",
    "The old building at the corner had",
    "The doctor told him that he needed",
    "A sudden noise from outside made her",
    "The forest was filled with ancient trees",
    "She picked up the phone and called",
    "The road to the village was long and",
    "They stood at the edge of the cliff and",
    "The novel she was reading described a",
    "At the conference the speaker presented",
    "The children played in the garden while",
    "The old man smiled and said that",
    "The company decided to invest in new",
    "The train arrived at the station just as",
    "Through the window she could see the",
    "The puzzle was more difficult than they",
    "The artist carefully mixed the colors to",
    "The report concluded that the main cause",
    "She remembered the day when they first",
    "The mountain was covered with snow and",
    "The debate focused on whether the government",
    "He turned the key and opened the door to",
]

ALL_PROMPTS = GRAMMAR_PROMPTS + ATTRIBUTE_PROMPTS + LOGIC_PROMPTS + COREF_PROMPTS + GENERAL_PROMPTS

# 缩短版用于8bit模型
SHORT_PROMPTS = ALL_PROMPTS[:80]


def load_model_custom(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cfg = MODEL_CONFIGS[model_name]
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    print(f"  Loading {model_name} (8bit={use_8bit})...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gc.collect()
    torch.cuda.empty_cache()
    if use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        attn_impl = "sdpa" if model_name == "deepseek7b" else "eager"
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True, local_files_only=True,
            attn_implementation=attn_impl, low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16,
            device_map="cpu", trust_remote_code=True, local_files_only=True,
            low_cpu_mem_usage=True, attn_implementation="eager")
        if torch.cuda.is_available():
            model = model.to("cuda")
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


def get_sample_layers(n_layers, n_max=12):
    """均匀采样层, 包括首尾"""
    if n_layers <= n_max:
        return list(range(n_layers + 1))
    result = set()
    result.add(0)
    result.add(1)
    result.add(n_layers - 1)
    result.add(n_layers)
    step = (n_layers - 1) / (n_max - 3)
    for i in range(1, n_max - 2):
        result.add(int(round(i * step)))
    return sorted(result)


# ============================================================
# Exp 1: CCA置换检验 — CCA≈1是真信号还是高维伪相关?
# ============================================================
def exp1_cca_permutation(model, tokenizer, model_name, n_sents=200, n_perm=50):
    """
    关键修正: Phase 153的CCA≈1可能是d>>n导致的高维伪相关
    
    方法:
    1. 计算真实CCA: CCA(hs[1], hs[ℓ])
    2. 随机打乱样本顺序后重算CCA: CCA_shuffled
    3. 如果 CCA_real >> CCA_shuffled → 真信号
    4. 如果 CCA_real ≈ CCA_shuffled → 伪相关
    
    同时: 去均值/去范数后重做CCA, 看是否只是范数效应
    """
    print("\n" + "="*60)
    print("Exp 1: CCA Permutation Test (d>>n修正)")
    print("="*60)
    from sklearn.cross_decomposition import CCA as SklearnCCA
    
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers
    d_model = info.d_model
    
    prompts = ALL_PROMPTS[:n_sents] if model_name == "qwen3" else SHORT_PROMPTS[:min(80, n_sents)]
    actual_n = len(prompts)
    print(f"  n_sents={actual_n}, d_model={d_model}, ratio d/n={d_model/actual_n:.1f}")
    
    sample_layers = get_sample_layers(n_layers, 10)
    print(f"  Sample layers: {sample_layers}")
    
    # 收集hidden states
    all_hs = {li: [] for li in sample_layers}
    all_norms = {li: [] for li in sample_layers}  # 范数
    all_means = {li: [] for li in sample_layers}   # 均值
    
    for si, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        for li in sample_layers:
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            all_hs[li].append(h)
            all_norms[li].append(float(np.linalg.norm(h)))
            all_means[li].append(float(np.mean(h)))
    
    ref_layer = 1
    X_ref = np.array(all_hs[ref_layer])  # [n, d]
    norms_ref = np.array(all_norms[ref_layer])
    means_ref = np.array(all_means[ref_layer])
    
    # PCA降维 (保留前60个分量)
    n_pca_keep = min(actual_n - 2, 60, d_model)
    X_ref_c = X_ref - X_ref.mean(axis=0)
    U_ref, s_ref, _ = np.linalg.svd(X_ref_c, full_matrices=False)
    X_ref_pca = U_ref[:, :n_pca_keep] * s_ref[:n_pca_keep]
    
    # 去范数版本: 用normalized hidden states
    X_ref_normed = np.array([h / max(np.linalg.norm(h), 1e-10) for h in all_hs[ref_layer]])
    X_ref_normed_c = X_ref_normed - X_ref_normed.mean(axis=0)
    U_ref_n, s_ref_n, _ = np.linalg.svd(X_ref_normed_c, full_matrices=False)
    X_ref_normed_pca = U_ref_n[:, :n_pca_keep] * s_ref_n[:n_pca_keep]
    
    n_cca = min(5, n_pca_keep - 1)
    
    results = {}
    test_layers = [li for li in sample_layers if li != ref_layer]
    
    for li in test_layers:
        X_li = np.array(all_hs[li])
        X_li_c = X_li - X_li.mean(axis=0)
        U_li, s_li, _ = np.linalg.svd(X_li_c, full_matrices=False)
        X_li_pca = U_li[:, :n_pca_keep] * s_li[:n_pca_keep]
        
        # 去范数版本
        X_li_normed = np.array([h / max(np.linalg.norm(h), 1e-10) for h in all_hs[li]])
        X_li_normed_c = X_li_normed - X_li_normed.mean(axis=0)
        U_li_n, s_li_n, _ = np.linalg.svd(X_li_normed_c, full_matrices=False)
        X_li_normed_pca = U_li_n[:, :n_pca_keep] * s_li_n[:n_pca_keep]
        
        # 1a. 真实CCA
        try:
            cca = SklearnCCA(n_components=n_cca)
            cca.fit(X_ref_pca, X_li_pca)
            X_r, X_l = cca.transform(X_ref_pca, X_li_pca)
            cca_real = [abs(np.corrcoef(X_r[:, i], X_l[:, i])[0, 1]) for i in range(n_cca)]
        except:
            cca_real = [0.0] * n_cca
        
        # 1b. 置换CCA: 打乱样本顺序
        cca_perm_vals = []
        for perm_i in range(n_perm):
            perm_idx = np.random.permutation(actual_n)
            X_li_perm = X_li_pca[perm_idx]
            try:
                cca_p = SklearnCCA(n_components=n_cca)
                cca_p.fit(X_ref_pca, X_li_perm)
                X_rp, X_lp = cca_p.transform(X_ref_pca, X_li_perm)
                cca_perm_vals.append([abs(np.corrcoef(X_rp[:, i], X_lp[:, i])[0, 1]) for i in range(n_cca)])
            except:
                pass
        
        if cca_perm_vals:
            cca_perm_mean = [float(np.mean([v[i] for v in cca_perm_vals])) for i in range(n_cca)]
            cca_perm_max = [float(np.max([v[i] for v in cca_perm_vals])) for i in range(n_cca)]
        else:
            cca_perm_mean = [0.0] * n_cca
            cca_perm_max = [0.0] * n_cca
        
        # 1c. 去范数CCA
        try:
            cca_n = SklearnCCA(n_components=n_cca)
            cca_n.fit(X_ref_normed_pca, X_li_normed_pca)
            X_rn, X_ln = cca_n.transform(X_ref_normed_pca, X_li_normed_pca)
            cca_normed = [abs(np.corrcoef(X_rn[:, i], X_ln[:, i])[0, 1]) for i in range(n_cca)]
        except:
            cca_normed = [0.0] * n_cca
        
        # 1d. 去范数+置换
        cca_normed_perm = []
        for perm_i in range(min(n_perm, 20)):
            perm_idx = np.random.permutation(actual_n)
            X_li_norm_perm = X_li_normed_pca[perm_idx]
            try:
                cca_np = SklearnCCA(n_components=n_cca)
                cca_np.fit(X_ref_normed_pca, X_li_norm_perm)
                X_rnp, X_lnp = cca_np.transform(X_ref_normed_pca, X_li_norm_perm)
                cca_normed_perm.append([abs(np.corrcoef(X_rnp[:, i], X_lnp[:, i])[0, 1]) for i in range(n_cca)])
            except:
                pass
        
        cca_normed_perm_mean = [float(np.mean([v[i] for v in cca_normed_perm])) for i in range(n_cca)] if cca_normed_perm else [0.0] * n_cca
        
        # 信噪比: CCA_real / CCA_perm_mean
        snr = [cca_real[i] / max(cca_perm_mean[i], 1e-10) for i in range(n_cca)]
        
        results[li] = {
            'cca1_real': float(cca_real[0]),
            'cca_mean_real': float(np.mean(cca_real)),
            'cca1_perm_mean': float(cca_perm_mean[0]),
            'cca1_perm_max': float(cca_perm_max[0]),
            'cca_mean_perm': float(np.mean(cca_perm_mean)),
            'cca1_normed': float(cca_normed[0]),
            'cca_mean_normed': float(np.mean(cca_normed)),
            'cca1_normed_perm': float(cca_normed_perm_mean[0]),
            'snr_cca1': float(snr[0]),
            'snr_cca_mean': float(np.mean(snr)),
            'all_cca_real': [float(c) for c in cca_real],
            'all_cca_perm_mean': [float(c) for c in cca_perm_mean],
        }
        
        print(f"  hs[{li:>3d}]: CCA1_real={cca_real[0]:.4f}, CCA1_perm={cca_perm_mean[0]:.4f} (max={cca_perm_max[0]:.4f}), "
              f"SNR={snr[0]:.1f}x, CCA1_normed={cca_normed[0]:.4f}, normed_perm={cca_normed_perm_mean[0]:.4f}")
    
    return results


# ============================================================
# Exp 2: Logit排序稳定性 — 核心约束!
# ============================================================
def exp2_logit_ranking(model, tokenizer, model_name, n_sents=200):
    """
    核心假说: Transformer保持的不是向量方向, 而是logit排序
    
    测量:
    1. top-k ranking在不同层的保持度 (rank correlation)
    2. 排序稳定性 vs cos稳定性 的对比
    3. margin (top1-top2) 如何逐层演化
    
    这是"约束传播"的核心实验:
    如果排序稳定但方向消失 → 约束传播假说成立
    如果排序也消失 → 需要重新思考
    """
    print("\n" + "="*60)
    print("Exp 2: Logit Ranking Stability (约束传播核心)")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers
    d_model = info.d_model
    W_U = get_W_U(model, model_name)  # [vocab, d]
    
    prompts = ALL_PROMPTS[:n_sents] if model_name == "qwen3" else SHORT_PROMPTS[:min(80, n_sents)]
    actual_n = len(prompts)
    sample_layers = get_sample_layers(n_layers, 12)
    print(f"  n_sents={actual_n}, sample_layers={sample_layers}")
    
    results_by_layer = {li: {'tau_list': [], 'spearman_list': [], 'top1_match': [], 
                              'top5_match': [], 'margin': [], 'cos_delta': []} 
                        for li in sample_layers}
    
    for si, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        
        # 最终层logits (ground truth)
        final_hs = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
        final_logits = final_hs @ W_U.T  # [vocab]
        final_top1 = int(np.argmax(final_logits))
        final_top5 = set(np.argsort(-final_logits)[:5])
        final_ranking = np.argsort(-final_logits)  # descending
        
        for li in sample_layers:
            h_li = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            logits_li = h_li @ W_U.T  # [vocab]
            
            # Top-1 match
            li_top1 = int(np.argmax(logits_li))
            top1_match = 1 if li_top1 == final_top1 else 0
            
            # Top-5 match (Jaccard)
            li_top5 = set(np.argsort(-logits_li)[:5])
            top5_match = len(final_top5 & li_top5) / 5.0
            
            # Ranking correlation (top-100 tokens)
            top100_final = np.argsort(-final_logits)[:100]
            ranks_final = np.argsort(np.argsort(-final_logits[top100_final]))
            ranks_li = np.argsort(np.argsort(-logits_li[top100_final]))
            
            # Kendall tau
            from scipy.stats import kendalltau, spearmanr
            tau, _ = kendalltau(ranks_final, ranks_li)
            spear, _ = spearmanr(ranks_final, ranks_li)
            
            # Margin at this layer
            sorted_logits = np.sort(-logits_li)
            margin = float(-sorted_logits[0] + sorted_logits[1])
            
            # Cos similarity of hidden state delta
            h0 = out.hidden_states[0][0, -1, :].float().cpu().numpy()
            delta_0 = h0 - np.mean(h0)
            delta_li = h_li - np.mean(h_li)
            cos_dl = np.dot(delta_0, delta_li) / (max(np.linalg.norm(delta_0), 1e-10) * max(np.linalg.norm(delta_li), 1e-10))
            
            results_by_layer[li]['tau_list'].append(float(tau) if not np.isnan(tau) else 0)
            results_by_layer[li]['spearman_list'].append(float(spear) if not np.isnan(spear) else 0)
            results_by_layer[li]['top1_match'].append(top1_match)
            results_by_layer[li]['top5_match'].append(top5_match)
            results_by_layer[li]['margin'].append(margin)
            results_by_layer[li]['cos_delta'].append(float(cos_dl))
    
    # 汇总
    summary = {}
    for li in sample_layers:
        d = results_by_layer[li]
        if not d['tau_list']:
            continue
        summary[li] = {
            'mean_tau': float(np.mean(d['tau_list'])),
            'mean_spearman': float(np.mean(d['spearman_list'])),
            'mean_top1_match': float(np.mean(d['top1_match'])),
            'mean_top5_match': float(np.mean(d['top5_match'])),
            'mean_margin': float(np.mean(d['margin'])),
            'mean_cos': float(np.mean(d['cos_delta'])),
        }
        print(f"  L{li:>3d}: tau={summary[li]['mean_tau']:.3f}, spearman={summary[li]['mean_spearman']:.3f}, "
              f"top1={summary[li]['mean_top1_match']:.3f}, top5={summary[li]['mean_top5_match']:.3f}, "
              f"margin={summary[li]['mean_margin']:.2f}, cos={summary[li]['mean_cos']:.3f}")
    
    # 核心对比: cos消失 vs ranking保留
    cos_values = [summary[li]['mean_cos'] for li in sorted(summary.keys())]
    tau_values = [summary[li]['mean_tau'] for li in sorted(summary.keys())]
    top1_values = [summary[li]['mean_top1_match'] for li in sorted(summary.keys())]
    
    print(f"\n  *** CORE COMPARISON ***")
    print(f"  cos:     early={cos_values[0]:.3f} → late={cos_values[-1]:.3f} (decay={cos_values[-1]/max(cos_values[0],1e-10):.3f}x)")
    print(f"  tau:     early={tau_values[0]:.3f} → late={tau_values[-1]:.3f} (decay={tau_values[-1]/max(tau_values[0],1e-10):.3f}x)")
    print(f"  top1:    early={top1_values[0]:.3f} → late={top1_values[-1]:.3f}")
    print(f"  tau/cos ratio at late layer: {tau_values[-1]/max(cos_values[-1],1e-10):.1f}x")
    
    return {'per_layer': summary,
            'cos_late': float(cos_values[-1]), 'tau_late': float(tau_values[-1]),
            'top1_late': float(top1_values[-1]),
            'cos_early': float(cos_values[0]), 'tau_early': float(tau_values[0])}


# ============================================================
# Exp 3: 约束传播图 — 特定语言约束如何跨层传播
# ============================================================
def exp3_constraint_propagation(model, tokenizer, model_name, n_sents=40):
    """
    追踪特定语言约束在层间的传播状态
    
    测量:
    1. 主谓一致约束: "The cat sit*" vs "The cat sits" — 哪层开始区分?
    2. 属性绑定约束: "red apple" vs "blue apple" — 哪层开始编码颜色?
    3. 否定约束: "did not finish" vs "did finish" — 哪层编码否定?
    
    方法: 用logit lens看每层的logit分布
    """
    print("\n" + "="*60)
    print("Exp 3: Constraint Propagation (约束传播)")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    
    sample_layers = get_sample_layers(n_layers, 10)
    
    # 约束测试对
    constraint_pairs = [
        # (prompt_a, prompt_b, probe_tokens_a, probe_tokens_b, constraint_name)
        ("The cat", "The cats", ["sits", "sit"], ["sit", "sits"], "singular_plural"),
        ("The dog", "The dogs", ["runs", "run"], ["run", "runs"], "singular_plural"),
        ("The bird", "The birds", ["flies", "fly"], ["fly", "flies"], "singular_plural"),
        ("The red apple was", "The blue car was", ["sweet", "fast"], ["fast", "sweet"], "attribute_binding"),
        ("The tall building", "The small bird", ["stood", "sat"], ["sat", "stood"], "attribute_binding"),
        ("She did not finish", "She did finish", ["the", "the"], ["the", "the"], "negation"),
        ("Because it rained", "Although it rained", ["the", "the"], ["the", "the"], "causal_concessive"),
        ("If she studies", "Since she studies", ["then", "then"], ["then", "then"], "conditional_causal"),
    ]
    
    results = {}
    
    for pair in constraint_pairs:
        prompt_a, prompt_b, probes_a, probes_b, cname = pair
        tok_ids_a = [tokenizer.encode(p, add_special_tokens=False)[0] for p in probes_a if tokenizer.encode(p, add_special_tokens=False)]
        tok_ids_b = [tokenizer.encode(p, add_special_tokens=False)[0] for p in probes_b if tokenizer.encode(p, add_special_tokens=False)]
        
        if not tok_ids_a or not tok_ids_b:
            continue
        
        layer_data = []
        for prompt, tok_ids in [(prompt_a, tok_ids_a), (prompt_b, tok_ids_b)]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attn_mask = inputs["attention_mask"].to(device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            
            prompt_layers = []
            for li in sample_layers:
                h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
                logits = h @ W_U.T
                # 获取目标token的logit
                target_logits = {tid: float(logits[tid]) for tid in tok_ids_a + tok_ids_b}
                top1 = int(np.argmax(logits))
                prompt_layers.append({'layer': li, 'target_logits': target_logits, 'top1': top1})
            layer_data.append(prompt_layers)
        
        # 对比: 约束在哪层开始分化
        constraint_onset = None
        for i, li in enumerate(sample_layers):
            logits_a = layer_data[0][i]['target_logits']
            logits_b = layer_data[1][i]['target_logits']
            # 检查是否logit排序已经分化
            if tok_ids_a and tok_ids_b:
                diff_a = logits_a.get(tok_ids_a[0], 0) - logits_a.get(tok_ids_b[0], 0)
                diff_b = logits_b.get(tok_ids_a[0], 0) - logits_b.get(tok_ids_b[0], 0)
                # 如果两个prompt的目标token排序不同
                if diff_a * diff_b < 0:  # 排序翻转
                    constraint_onset = li
                    break
        
        results[cname] = {
            'constraint_onset_layer': constraint_onset,
            'onset_normalized': float(constraint_onset / n_layers) if constraint_onset else None,
            'pair_a': prompt_a,
            'pair_b': prompt_b,
        }
        print(f"  {cname}: onset_layer={constraint_onset}, "
              f"onset_norm={results[cname]['onset_normalized']:.2f}" if constraint_onset else 
              f"  {cname}: onset_layer=NOT_FOUND")
    
    # 汇总
    onsets = [r['onset_normalized'] for r in results.values() if r['onset_normalized'] is not None]
    if onsets:
        print(f"\n  Mean constraint onset: {np.mean(onsets):.2f} (normalized depth)")
        print(f"  Range: [{np.min(onsets):.2f}, {np.max(onsets):.2f}]")
    
    return results


# ============================================================
# Exp 4: 边界流场 — logit边界如何逐层演化
# ============================================================
def exp4_boundary_flow(model, tokenizer, model_name, n_sents=60):
    """
    不是只看最终层的margin, 而是追踪margin如何逐层形成
    
    测量:
    1. 每层的margin (top1-top2 logit差)
    2. 边界锐度: margin / logit_std
    3. 边界稳定性: 扰动后margin变化 vs 层深度
    4. 最终top1在各层的排名变化
    """
    print("\n" + "="*60)
    print("Exp 4: Boundary Flow Field (边界流场)")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers
    d_model = info.d_model
    W_U = get_W_U(model, model_name)
    
    prompts = ALL_PROMPTS[:n_sents] if model_name == "qwen3" else SHORT_PROMPTS[:min(40, n_sents)]
    sample_layers = get_sample_layers(n_layers, 12)
    
    layer_stats = {li: {'margins': [], 'logit_stds': [], 'sharpness': [], 
                         'final_top1_ranks': [], 'top1_entropy': []} 
                   for li in sample_layers}
    
    for si, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        
        # 最终层top1
        final_h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
        final_logits = final_h @ W_U.T
        final_top1 = int(np.argmax(final_logits))
        
        for li in sample_layers:
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            logits = h @ W_U.T
            
            sorted_logits = np.sort(logits)[::-1]
            margin = float(sorted_logits[0] - sorted_logits[1])
            logit_std = float(np.std(logits))
            sharpness = margin / max(logit_std, 1e-10)
            
            # 最终top1在此层的排名
            rank_of_final_top1 = int(np.sum(logits > logits[final_top1]))
            
            # Top-1 token在此层的entropy (衡量决策确定性)
            top_logits = sorted_logits[:10]
            top_probs = np.exp(top_logits - top_logits.max())
            top_probs /= top_probs.sum()
            entropy = -np.sum(top_probs * np.log(top_probs + 1e-10))
            
            layer_stats[li]['margins'].append(margin)
            layer_stats[li]['logit_stds'].append(logit_std)
            layer_stats[li]['sharpness'].append(sharpness)
            layer_stats[li]['final_top1_ranks'].append(rank_of_final_top1)
            layer_stats[li]['top1_entropy'].append(float(entropy))
    
    # 汇总
    summary = {}
    for li in sample_layers:
        d = layer_stats[li]
        if not d['margins']:
            continue
        summary[li] = {
            'mean_margin': float(np.mean(d['margins'])),
            'mean_sharpness': float(np.mean(d['sharpness'])),
            'mean_top1_rank': float(np.mean(d['final_top1_ranks'])),
            'top1_rank_0_rate': float(np.mean([1 for r in d['final_top1_ranks'] if r == 0])),
            'mean_entropy': float(np.mean(d['top1_entropy'])),
        }
        print(f"  L{li:>3d}: margin={summary[li]['mean_margin']:.3f}, "
              f"sharpness={summary[li]['mean_sharpness']:.3f}, "
              f"top1_rank={summary[li]['mean_top1_rank']:.1f}, "
              f"top1_rank0_rate={summary[li]['top1_rank_0_rate']:.3f}, "
              f"entropy={summary[li]['mean_entropy']:.3f}")
    
    # 关键: top1从哪层开始稳定(rank=0)?
    print(f"\n  *** BOUNDARY FORMATION ***")
    for li in sorted(summary.keys()):
        r0 = summary[li]['top1_rank_0_rate']
        if r0 > 0.8:
            print(f"  ★ Boundary stabilizes at L{li} (top1_rank0_rate={r0:.3f})")
            break
    
    return summary


# ============================================================
# Exp 5: 注意力路由图 — 动态图结构+社区检测
# ============================================================
def exp5_routing_graph(model, tokenizer, model_name, n_sents=40):
    """
    把attention看成动态图, 分析:
    1. Token间的注意力强度矩阵 → 图
    2. 图的社区结构 (哪些token形成稳定子图)
    3. 跨prompt的图稳定性
    4. 稀疏性 vs 密集性的层间变化
    """
    print("\n" + "="*60)
    print("Exp 5: Attention Routing Graph (路由图动力学)")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers
    
    # 检查是否支持output_attentions
    supports = True
    try:
        inputs = tokenizer(ALL_PROMPTS[0], return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            t = model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device),
                      output_attentions=True)
        if t.attentions is None:
            supports = False
    except:
        supports = False
    
    print(f"  output_attentions support: {supports}")
    
    prompts = ALL_PROMPTS[:n_sents] if model_name == "qwen3" else SHORT_PROMPTS[:min(20, n_sents)]
    attn_layers = sorted(set([0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]))[:7]
    
    # 收集跨prompt的图统计
    graph_stats = {li: {'sparsity': [], 'max_attn_to_last': [], 'n_communities': [],
                         'modularity': [], 'entropy_last': []} 
                   for li in attn_layers}
    
    for si, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        seq_len = input_ids.shape[1]
        last_pos = seq_len - 1
        
        if supports:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
            
            for li in attn_layers:
                if li >= len(out.attentions) or out.attentions[li] is None:
                    continue
                aw = out.attentions[li][0].float().cpu().numpy()  # [n_heads, seq, seq]
                
                # 平均所有head的attention
                aw_mean = aw.mean(axis=0)  # [seq, seq]
                
                # Sparsity: top-10% attention mass
                flat = aw_mean[last_pos, :last_pos+1].flatten()
                if len(flat) < 2:
                    continue
                sorted_vals = np.sort(flat)[::-1]
                top10_pct = max(1, len(sorted_vals) // 10)
                sparsity = float(np.sum(sorted_vals[:top10_pct]) / max(np.sum(sorted_vals), 1e-10))
                
                # Max attention to last token
                max_attn_last = float(np.max(aw_mean[last_pos, :last_pos+1]))
                
                # Attention entropy at last position
                ad = aw_mean[last_pos, :last_pos+1]
                ad = ad / max(ad.sum(), 1e-10)
                ent = -np.sum(ad * np.log(ad + 1e-10))
                max_ent = np.log(len(ad)) if len(ad) > 1 else 1
                
                # Simple community detection: 按attention强度聚类
                # 看哪些token对last_pos的attention > threshold
                threshold = 1.0 / (last_pos + 1) * 2  # 2x uniform
                strong_connections = np.sum(aw_mean[last_pos, :last_pos+1] > threshold)
                
                graph_stats[li]['sparsity'].append(sparsity)
                graph_stats[li]['max_attn_to_last'].append(max_attn_last)
                graph_stats[li]['n_communities'].append(int(strong_connections))
                graph_stats[li]['entropy_last'].append(float(ent / max_ent))
        else:
            # Fallback: 使用QK近似
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            
            for li in attn_layers:
                layers_list = get_layers(model)
                sa = layers_list[li].self_attn
                h_in = out.hidden_states[li][0]
                W_q = sa.q_proj.weight.detach().float()
                W_k = sa.k_proj.weight.detach().float()
                Q = (h_in @ W_q.T).cpu().numpy()
                K = (h_in @ W_k.T).cpu().numpy()
                n_h = min(8, Q.shape[1] // 32)
                d_h = Q.shape[1] // n_h if n_h > 0 else Q.shape[1]
                
                # 平均head的attention (last position)
                all_ad = []
                for h in range(n_h):
                    Qh = Q[last_pos, h*d_h:(h+1)*d_h]
                    Kh = K[:last_pos+1, h*d_h:(h+1)*d_h]
                    scores = Kh @ Qh / max(np.sqrt(d_h), 1.0)
                    es = np.exp(scores - np.max(scores))
                    ad = es / es.sum()
                    all_ad.append(ad)
                
                avg_ad = np.mean(all_ad, axis=0)
                sorted_vals = np.sort(avg_ad)[::-1]
                top10_pct = max(1, len(sorted_vals) // 10)
                sparsity = float(np.sum(sorted_vals[:top10_pct]) / max(np.sum(sorted_vals), 1e-10))
                max_attn = float(np.max(avg_ad))
                threshold = 1.0 / len(avg_ad) * 2
                strong_conns = int(np.sum(avg_ad > threshold))
                ent = -np.sum(avg_ad * np.log(avg_ad + 1e-10))
                max_ent = np.log(len(avg_ad)) if len(avg_ad) > 1 else 1
                
                graph_stats[li]['sparsity'].append(sparsity)
                graph_stats[li]['max_attn_to_last'].append(max_attn)
                graph_stats[li]['n_communities'].append(strong_conns)
                graph_stats[li]['entropy_last'].append(float(ent / max_ent))
    
    # 汇总
    summary = {}
    for li in sorted(graph_stats.keys()):
        d = graph_stats[li]
        if not d['sparsity']:
            continue
        summary[li] = {
            'mean_sparsity': float(np.mean(d['sparsity'])),
            'mean_max_attn': float(np.mean(d['max_attn_to_last'])),
            'mean_n_strong': float(np.mean(d['n_communities'])),
            'mean_entropy': float(np.mean(d['entropy_last'])),
            'cv_sparsity': float(np.std(d['sparsity']) / max(np.mean(d['sparsity']), 0.01)),
            'cv_entropy': float(np.std(d['entropy_last']) / max(np.mean(d['entropy_last']), 0.01)),
        }
        print(f"  L{li:>3d}: sparsity={summary[li]['mean_sparsity']:.3f}, "
              f"max_attn={summary[li]['mean_max_attn']:.3f}, "
              f"strong_conns={summary[li]['mean_n_strong']:.1f}, "
              f"entropy={summary[li]['mean_entropy']:.3f}, "
              f"CV(sparsity)={summary[li]['cv_sparsity']:.3f}")
    
    return {'per_layer': summary, 'supports_output_attentions': supports}


# ============================================================
# Main
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Phase 154: Constraint Dynamics — {model_name}, {ts}")

    t0 = time.time()
    model, tokenizer, device = load_model_custom(model_name)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}, load={time.time()-t0:.1f}s")

    # Exp 1: CCA置换检验 (最重要! 修正Phase 153的硬伤)
    e1 = exp1_cca_permutation(model, tokenizer, model_name)
    
    # Exp 2: Logit排序稳定性 (约束传播核心)
    e2 = exp2_logit_ranking(model, tokenizer, model_name)
    
    # Exp 3: 约束传播
    e3 = exp3_constraint_propagation(model, tokenizer, model_name)
    
    # Exp 4: 边界流场
    e4 = exp4_boundary_flow(model, tokenizer, model_name)
    
    # Exp 5: 路由图动力学
    e5 = exp5_routing_graph(model, tokenizer, model_name)

    all_r = {
        "phase": "154_constraint_dynamics",
        "model": model_name,
        "timestamp": ts,
        "model_info": {"class": info.model_class, "n_layers": info.n_layers, "d_model": info.d_model},
        "exp1_cca_permutation": e1,
        "exp2_logit_ranking": e2,
        "exp3_constraint_propagation": e3,
        "exp4_boundary_flow": e4,
        "exp5_routing_graph": e5,
    }

    rf = OUTPUT_DIR / f"phase154_{model_name}_{ts}.json"
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, (np.int32, np.int64)): return int(o)
        raise TypeError(f"Cannot serialize {type(o)}")
    with open(rf, 'w', encoding='utf-8') as f:
        json.dump(all_r, f, indent=2, default=conv, ensure_ascii=False)
    print(f"\nSaved: {rf}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("Phase 154 done.")


if __name__ == "__main__":
    main()
