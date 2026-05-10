"""
Phase 115: 置换破坏 + 自举稳定性 + 因果特征子空间干预
==========================================================
核心目标 (用户定义的三个生死检验):

1. Permutation Destruction (置换破坏):
   打乱翻译配对 (猫→water, 火→moon), 如果spike仍存在 → 数据集结构伪影
   如果spike消失 → 翻译配对是真信号

2. Bootstrap Stability (自举稳定性):
   随机抽词对子集, 看leading eigenvector是否稳定
   如果不稳定 → 低秩集中可能只是采样偶然

3. Causal Eigenspace Intervention (因果特征子空间干预):
   只干预spike eigenspace (L12的4个信号维度), 观察翻译输出变化
   - Remove: 在信号子空间上做零化
   - Amplify: 在信号子空间上做放大
   - Rotate: 在信号子空间内做旋转
   如果翻译行为可控改变 → spike是计算基底
   如果不可控 → spike是统计副现象

4. Tokenizer Control (分词器控制):
   控制子词长度、频率、合并模式, 看spike是否由token统计产生

理论纪律:
- 使用"低秩集中"(low-rank concentration)而非"凝聚"(condensation)
- 不假设L12的4维就是"翻译核心维度" — 可能是tokenizer/frequency/script伪影
- Fisher≈1是重要警告: spike可能是"状态协调编码"而非"输出决策编码"
"""

import os, sys, json, gc, time, argparse
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import load_model, get_layers, get_model_info, release_model

# ============================================================
# 设置
# ============================================================
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# 翻译词对 - 与Phase 114相同
WORD_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("鸟", "bird"), ("鱼", "fish"), ("马", "horse"),
    ("牛", "cow"), ("羊", "sheep"), ("猪", "pig"), ("鸡", "chicken"), ("鼠", "mouse"),
    ("水", "water"), ("火", "fire"), ("土", "earth"), ("风", "wind"), ("金", "gold"),
    ("木", "wood"), ("铁", "iron"), ("石", "stone"), ("沙", "sand"), ("冰", "ice"),
    ("月", "moon"), ("星", "star"), ("云", "cloud"), ("雨", "rain"), ("雪", "snow"),
    ("日", "sun"), ("天", "sky"), ("海", "sea"), ("河", "river"), ("山", "mountain"),
    ("红", "red"), ("蓝", "blue"), ("绿", "green"), ("白", "white"), ("黑", "black"),
    ("黄", "yellow"), ("紫", "purple"), ("灰", "gray"), ("棕", "brown"), ("粉", "pink"),
    ("手", "hand"), ("足", "foot"), ("目", "eye"), ("耳", "ear"), ("口", "mouth"),
    ("心", "heart"), ("头", "head"), ("骨", "bone"), ("血", "blood"), ("发", "hair"),
    ("父", "father"), ("母", "mother"), ("子", "son"), ("女", "daughter"), ("友", "friend"),
    ("王", "king"), ("师", "teacher"), ("医", "doctor"), ("兵", "soldier"), ("农", "farmer"),
    ("爱", "love"), ("恨", "hate"), ("善", "good"), ("恶", "evil"), ("真", "truth"),
    ("美", "beauty"), ("智", "wisdom"), ("力", "power"), ("光", "light"), ("影", "shadow"),
    ("走", "walk"), ("跑", "run"), ("飞", "fly"), ("吃", "eat"), ("喝", "drink"),
    ("看", "see"), ("听", "hear"), ("说", "say"), ("写", "write"), ("读", "read"),
    ("年", "year"), ("月", "month"), ("日", "day"), ("时", "hour"), ("分", "minute"),
    ("米", "rice"), ("茶", "tea"), ("酒", "wine"), ("肉", "meat"), ("盐", "salt"),
    ("车", "car"), ("船", "ship"), ("路", "road"), ("桥", "bridge"), ("门", "door"),
    ("花", "flower"), ("草", "grass"), ("树", "tree"), ("叶", "leaf"), ("根", "root"),
    ("喜", "joy"), ("怒", "anger"), ("哀", "sorrow"), ("惧", "fear"), ("思", "thought"),
    ("法", "law"), ("国", "country"), ("城", "city"), ("家", "home"), ("书", "book"),
    ("电", "electricity"), ("网", "network"), ("数", "number"), ("算", "compute"), ("器", "device"),
    ("湖", "lake"), ("岛", "island"), ("春", "spring"), ("夏", "summer"), ("秋", "autumn"),
    ("冬", "winter"), ("晨", "morning"), ("暮", "dusk"), ("雷", "thunder"), ("雾", "fog"),
    ("龙", "dragon"), ("蛇", "snake"), ("虎", "tiger"), ("鹿", "deer"), ("兔", "rabbit"),
    ("道", "way"), ("德", "virtue"), ("礼", "ritual"), ("义", "justice"), ("信", "trust"),
    ("剑", "sword"), ("笔", "pen"), ("琴", "lute"), ("画", "painting"), ("棋", "chess"),
    ("砖", "brick"), ("瓦", "tile"), ("丝", "silk"), ("布", "cloth"), ("纸", "paper"),
]

# 去重
seen_zh = set()
UNIQUE_PAIRS = []
for zh, en in WORD_PAIRS:
    if zh not in seen_zh:
        seen_zh.add(zh)
        UNIQUE_PAIRS.append((zh, en))
WORD_PAIRS = UNIQUE_PAIRS[:150]

# 采样层
SAMPLE_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]


# ============================================================
# MP分析核心函数 (复用Phase 114)
# ============================================================
def marchenko_pastur_bounds(N, P, sigma2=1.0):
    ratio = P / N
    lambda_minus = sigma2 * (1 - np.sqrt(ratio)) ** 2
    lambda_plus = sigma2 * (1 + np.sqrt(ratio)) ** 2
    return lambda_minus, lambda_plus


def analyze_eigenvalue_spectrum(eigenvalues, N_samples, P_dim, sigma2_est=None):
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    if sigma2_est is None:
        n_noise = max(len(eigenvalues) - 10, len(eigenvalues) // 2)
        sigma2_est = np.median(eigenvalues[-n_noise:]) if n_noise > 0 else np.median(eigenvalues)
    
    lam_minus, lam_plus = marchenko_pastur_bounds(N_samples, P_dim, sigma2_est)
    
    signal_eigs = eigenvalues[eigenvalues > lam_plus]
    noise_eigs = eigenvalues[eigenvalues <= lam_plus]
    true_dim = len(signal_eigs)
    
    total_var = np.sum(eigenvalues)
    signal_var = np.sum(signal_eigs)
    signal_ratio = signal_var / total_var if total_var > 0 else 0
    
    pr_signal = (np.sum(signal_eigs))**2 / np.sum(signal_eigs**2) if len(signal_eigs) > 0 else 0
    pr_full = (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2) if total_var > 0 else 0
    
    return {
        "true_dimensionality": true_dim,
        "signal_variance_ratio": float(signal_ratio),
        "pr_signal": float(pr_signal),
        "pr_full": float(pr_full),
        "mp_lambda_plus": float(lam_plus),
        "top10_eigenvalues": [float(x) for x in eigenvalues[:10]],
        "gap_to_noise": float(eigenvalues[true_dim-1] / lam_plus) if true_dim > 0 and lam_plus > 0 else 0,
    }


def compute_mp_analysis(diffs, N, P):
    """对差分矩阵做完整的MP分析, 返回特征值谱和信号信息"""
    diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
    small_cov = (diffs_centered @ diffs_centered.T) / N
    eigenvalues = np.linalg.eigvalsh(small_cov)[::-1]
    
    element_var = np.var(diffs_centered)
    mp_result = analyze_eigenvalue_spectrum(eigenvalues, N, P, sigma2_est=element_var)
    
    return {
        "eigenvalues": eigenvalues,
        "mp_result": mp_result,
        "element_var": float(element_var),
    }


# ============================================================
# 数据收集: 隐藏状态 + MLP输出 (带tokenizer信息)
# ============================================================
def collect_hidden_states_with_tokenizer_info(model, tokenizer, device, n_layers, word_pairs, batch_size=5):
    """收集隐藏状态, 同时记录tokenizer信息"""
    layers = get_layers(model)
    
    all_states = {'zh': defaultdict(list), 'trans': defaultdict(list)}
    token_info = []  # 每个词对的tokenizer信息
    
    for batch_start in range(0, len(word_pairs), batch_size):
        batch = word_pairs[batch_start:batch_start+batch_size]
        
        for zh, en in batch:
            # Tokenizer信息
            zh_tokens = tokenizer(zh, add_special_tokens=False)
            en_tokens = tokenizer(en, add_special_tokens=False)
            zh_prompt_tokens = tokenizer(f"翻译以下中文词：{zh}", add_special_tokens=False)
            trans_prompt_tokens = tokenizer(f"Translate the following Chinese word: {zh}", add_special_tokens=False)
            
            info = {
                "zh_word": zh, "en_word": en,
                "zh_token_count": len(zh_tokens['input_ids']),
                "en_token_count": len(en_tokens['input_ids']),
                "zh_prompt_token_count": len(zh_prompt_tokens['input_ids']),
                "trans_prompt_token_count": len(trans_prompt_tokens['input_ids']),
                "token_count_diff": len(trans_prompt_tokens['input_ids']) - len(zh_prompt_tokens['input_ids']),
            }
            token_info.append(info)
            
            zh_prompt = f"翻译以下中文词：{zh}"
            trans_prompt = f"Translate the following Chinese word: {zh}"
            
            for task, prompt in [('zh', zh_prompt), ('trans', trans_prompt)]:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                layer_acts = {}
                hooks = []
                
                def make_hook(l):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            layer_acts[l] = output[0][0, -1, :].detach().float().cpu().numpy()
                        else:
                            layer_acts[l] = output[0, -1, :].detach().float().cpu().numpy()
                    return hook_fn
                
                for l, layer in enumerate(layers):
                    h = layer.register_forward_hook(make_hook(l))
                    hooks.append(h)
                
                with torch.no_grad():
                    _ = model(inputs["input_ids"])
                
                for h in hooks:
                    h.remove()
                del inputs
                gc.collect()
                torch.cuda.empty_cache()
                
                for l in layer_acts:
                    all_states[task][l].append(layer_acts[l])
        
        print(f"  [collect] {min(batch_start+batch_size, len(word_pairs))}/{len(word_pairs)} 词对")
    
    result = {}
    for task in ['zh', 'trans']:
        result[task] = {}
        for l in all_states[task]:
            result[task][l] = np.array(all_states[task][l])
    
    d_model = result['zh'][0].shape[1] if 0 in result['zh'] else 2560
    return result, d_model, token_info


def collect_mlp_outputs(model, tokenizer, device, n_layers, word_pairs, batch_size=5):
    """收集MLP输出"""
    layers = get_layers(model)
    
    all_mlp = {'zh': defaultdict(list), 'trans': defaultdict(list)}
    
    for batch_start in range(0, len(word_pairs), batch_size):
        batch = word_pairs[batch_start:batch_start+batch_size]
        
        for zh, en in batch:
            zh_prompt = f"翻译以下中文词：{zh}"
            trans_prompt = f"Translate the following Chinese word: {zh}"
            
            for task, prompt in [('zh', zh_prompt), ('trans', trans_prompt)]:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                mlp_outs = {}
                hooks = []
                
                def make_mlp_hook(l):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            mlp_outs[l] = output[0][0, -1, :].detach().float().cpu().numpy()
                        else:
                            mlp_outs[l] = output[0, -1, :].detach().float().cpu().numpy()
                    return hook_fn
                
                for l, layer in enumerate(layers):
                    if hasattr(layer, 'mlp'):
                        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))
                
                with torch.no_grad():
                    _ = model(inputs["input_ids"])
                
                for h in hooks:
                    h.remove()
                del inputs
                gc.collect()
                torch.cuda.empty_cache()
                
                for l in mlp_outs:
                    all_mlp[task][l].append(mlp_outs[l])
        
        print(f"  [collect_mlp] {min(batch_start+batch_size, len(word_pairs))}/{len(word_pairs)}")
    
    result = {}
    for task in ['zh', 'trans']:
        result[task] = {}
        for l in all_mlp[task]:
            result[task][l] = np.array(all_mlp[task][l])
    
    return result


# ============================================================
# Exp 1: Permutation Destruction (置换破坏) — 最关键!
# ============================================================
def exp1_permutation_destruction(mlp_outs, d_model, n_permutations=20):
    """
    置换破坏检验: 打乱翻译配对, 看低秩集中是否仍存在
    
    逻辑:
    - 正确配对: 猫→cat, 火→fire → 差分 = trans(猫) - zh(猫)
    - 置换配对: 猫→water, 火→moon → 差分 = trans(猫) - zh(火)
    
    如果spike在置换后消失 → spike依赖于正确的翻译配对 → 真信号
    如果spike在置换后仍存在 → spike来自数据集结构(tokenizer/frequency等) → 伪影
    """
    print("\n" + "="*70)
    print("Exp 1: Permutation Destruction — 置换破坏检验")
    print("="*70)
    print("核心逻辑: 如果打乱翻译配对后spike消失 → 真信号; 仍存在 → 伪影")
    
    results = {}
    
    for l in SAMPLE_LAYERS:
        if l not in mlp_outs['zh'] or l not in mlp_outs['trans']:
            continue
        
        zh_data = mlp_outs['zh'][l]  # [N, d_model]
        trans_data = mlp_outs['trans'][l]  # [N, d_model]
        N = zh_data.shape[0]
        P = zh_data.shape[1]
        
        if N < 5:
            continue
        
        # 1. 正确配对的差分
        correct_diffs = trans_data - zh_data
        correct_result = compute_mp_analysis(correct_diffs, N, P)
        
        # 2. 置换配对的差分 (打乱中文样本的索引)
        perm_true_dims = []
        perm_pr_signals = []
        perm_signal_ratios = []
        perm_top_eigs = []
        
        for p_idx in range(n_permutations):
            # 随机打乱中文样本索引
            perm_idx = np.random.permutation(N)
            perm_diffs = trans_data - zh_data[perm_idx]  # 打乱配对
            perm_result = compute_mp_analysis(perm_diffs, N, P)
            
            perm_true_dims.append(perm_result['mp_result']['true_dimensionality'])
            perm_pr_signals.append(perm_result['mp_result']['pr_signal'])
            perm_signal_ratios.append(perm_result['mp_result']['signal_variance_ratio'])
            perm_top_eigs.append(perm_result['eigenvalues'][:5])
        
        # 3. 完全独立随机差分 (最严格的null)
        random_true_dims = []
        random_pr_signals = []
        
        for _ in range(n_permutations):
            rand_diffs = np.random.randn(N, P) * np.sqrt(correct_result['element_var'])
            rand_result = compute_mp_analysis(rand_diffs, N, P)
            random_true_dims.append(rand_result['mp_result']['true_dimensionality'])
            random_pr_signals.append(rand_result['mp_result']['pr_signal'])
        
        # 4. 另一种置换: 只打乱trans索引 (保持zh顺序)
        perm_trans_dims = []
        perm_trans_concentrations = []
        for p_idx in range(n_permutations):
            perm_idx = np.random.permutation(N)
            perm_diffs = trans_data[perm_idx] - zh_data  # 打乱英文样本配对
            perm_result = compute_mp_analysis(perm_diffs, N, P)
            perm_trans_dims.append(perm_result['mp_result']['true_dimensionality'])
            # 浓度比: top1特征值 / 总方差
            total_var = np.sum(perm_result['eigenvalues'])
            perm_trans_concentrations.append(perm_result['eigenvalues'][0] / total_var if total_var > 0 else 0)
        
        # ============================================================
        # 关键重新解读: 正确配对应该比置换配对更集中!
        # 
        # 正确逻辑:
        # - 正确配对: 翻译差分 = trans(猫) - zh(猫), 语义相关 → 维度低/浓度高
        # - 置换配对: 错误差分 = trans(猫) - zh(火), 语义无关 → 维度高/浓度低
        # - 如果正确配对维度更低/浓度更高 → 低秩集中是翻译特定结构 → 真信号!
        # - 如果正确配对维度相同/浓度相同 → 与配对无关 → 数据集伪影
        # ============================================================
        
        # 核心指标: 浓度比 (concentration ratio)
        # = top1特征值 / 总方差  (越高 = 越集中)
        correct_total_var = np.sum(correct_result['eigenvalues'])
        correct_concentration = correct_result['eigenvalues'][0] / correct_total_var if correct_total_var > 0 else 0
        
        perm_concentrations = []
        for perm_eigs in perm_top_eigs:
            # 近似: 用top5占总方差的比例 (更稳定)
            perm_concentrations.append(perm_eigs[0] / correct_total_var if correct_total_var > 0 else 0)
        
        # 更精确: 用完整置换数据的浓度比
        # 重新计算 (置换差分的浓度比)
        perm_concentration_ratios = []
        for p_idx in range(n_permutations):
            perm_idx = np.random.permutation(N)
            perm_diffs = trans_data - zh_data[perm_idx]
            perm_diffs_centered = perm_diffs - perm_diffs.mean(axis=0, keepdims=True)
            small_cov = (perm_diffs_centered @ perm_diffs_centered.T) / N
            perm_eigs = np.linalg.eigvalsh(small_cov)[::-1]
            total = np.sum(perm_eigs)
            perm_concentration_ratios.append(perm_eigs[0] / total if total > 0 else 0)
        
        # PR浓度比 (PR越低 = 越集中)
        correct_pr = correct_result['mp_result']['pr_signal']
        perm_pr_mean = np.mean(perm_pr_signals)
        perm_pr_std = np.std(perm_pr_signals)
        
        # 维度压缩比 (正确配对维度 / 置换配对维度, <1 = 有压缩)
        correct_dim = correct_result['mp_result']['true_dimensionality']
        perm_dim_mean = np.mean(perm_true_dims)
        perm_dim_std = np.std(perm_true_dims)
        random_dim_mean = np.mean(random_true_dims)
        dim_compression_ratio = correct_dim / perm_dim_mean if perm_dim_mean > 0 else 1.0
        
        # 判定逻辑 (修正版!)
        # 1. 正确配对维度是否显著低于置换? (dim_compression_ratio < 1)
        # 2. 正确配对PR是否显著低于置换? (更集中)
        # 3. 正确配对浓度比是否显著高于置换? (更集中)
        
        if perm_pr_std > 0:
            z_pr = (correct_pr - perm_pr_mean) / perm_pr_std  # 负值 = 正确配对更集中
        else:
            z_pr = 0
        
        perm_conc_mean = np.mean(perm_concentration_ratios)
        perm_conc_std = np.std(perm_concentration_ratios)
        if perm_conc_std > 0:
            z_concentration = (correct_concentration - perm_conc_mean) / perm_conc_std  # 正值 = 正确配对更集中
        else:
            z_concentration = 0
        
        # 综合判定
        concentration_effect = correct_concentration - perm_conc_mean
        dimension_effect = perm_dim_mean - correct_dim  # 正值 = 正确配对维度更低
        
        if dim_compression_ratio < 0.5 and z_concentration > 3.0:
            verdict = "REAL_SIGNAL_STRONG"  # 正确配对显著更集中
        elif dim_compression_ratio < 0.7 and (z_concentration > 2.0 or z_pr < -2.0):
            verdict = "REAL_SIGNAL_MODERATE"
        elif dim_compression_ratio < 0.85:
            verdict = "POSSIBLE_SIGNAL"
        elif dim_compression_ratio > 1.0 and abs(z_concentration) < 1.0:
            verdict = "DATASET_ARTIFACT"  # 配对不影响结构
        else:
            verdict = "AMBIGUOUS"
        
        results[f"L{l}"] = {
            "correct_pairing": {
                "true_dim": correct_dim,
                "pr_signal": correct_result['mp_result']['pr_signal'],
                "signal_ratio": correct_result['mp_result']['signal_variance_ratio'],
                "concentration_ratio": float(correct_concentration),
                "top5_eigenvalues": [float(x) for x in correct_result['eigenvalues'][:5]],
            },
            "permutation_zh_shuffled": {
                "true_dim_mean": float(perm_dim_mean),
                "true_dim_std": float(perm_dim_std),
                "pr_signal_mean": float(perm_pr_mean),
                "pr_signal_std": float(perm_pr_std),
                "concentration_ratio_mean": float(perm_conc_mean),
                "concentration_ratio_std": float(perm_conc_std),
                "top5_eig_mean": [float(x) for x in np.mean(perm_top_eigs, axis=0)],
            },
            "permutation_trans_shuffled": {
                "true_dim_mean": float(np.mean(perm_trans_dims)),
                "true_dim_std": float(np.std(perm_trans_dims)),
                "concentration_ratio_mean": float(np.mean(perm_trans_concentrations)),
            },
            "random_baseline": {
                "true_dim_mean": float(random_dim_mean),
                "true_dim_std": float(np.std(random_true_dims)),
            },
            "key_metrics": {
                "dim_compression_ratio": float(dim_compression_ratio),  # <1 = 正确配对更压缩
                "concentration_effect": float(concentration_effect),  # >0 = 正确配对更集中
                "dimension_effect": float(dimension_effect),  # >0 = 正确配对维度更低
                "z_concentration": float(z_concentration),  # >0 = 正确配对浓度显著更高
                "z_pr": float(z_pr),  # <0 = 正确配对PR显著更低(更集中)
            },
            "verdict": verdict,
        }
        
        print(f"  L{l}: correct_dim={correct_dim}, perm_dim={perm_dim_mean:.1f}±{perm_dim_std:.1f}, "
              f"compression_ratio={dim_compression_ratio:.3f}, "
              f"concentration: correct={correct_concentration:.4f} vs perm={perm_conc_mean:.4f}, "
              f"Δ_conc={concentration_effect:.4f}, z_conc={z_concentration:.2f}, "
              f"verdict={verdict}")
        print(f"    正确配对top5 eig: {[f'{x:.4f}' for x in correct_result['eigenvalues'][:5]]}")
        print(f"    置换配对top5 eig: {[f'{x:.4f}' for x in np.mean(perm_top_eigs, axis=0)]}")
    
    return results


# ============================================================
# Exp 2: Bootstrap Stability (自举稳定性)
# ============================================================
def exp2_bootstrap_stability(mlp_outs, d_model, n_bootstrap=50, subsample_ratio=0.7):
    """
    自举稳定性检验: 随机抽词对子集, leading eigenvector是否稳定?
    
    如果主特征向量在不同子样本间高度一致 → 低秩集中是稳定结构
    如果主特征向量高度变化 → 可能只是采样偶然
    """
    print("\n" + "="*70)
    print("Exp 2: Bootstrap Stability — 主特征向量是否跨子样本稳定?")
    print("="*70)
    
    results = {}
    
    for l in SAMPLE_LAYERS:
        if l not in mlp_outs['zh'] or l not in mlp_outs['trans']:
            continue
        
        zh_data = mlp_outs['zh'][l]
        trans_data = mlp_outs['trans'][l]
        N = zh_data.shape[0]
        P = zh_data.shape[1]
        
        if N < 20:
            continue
        
        n_subsample = int(N * subsample_ratio)
        
        # 全样本的主特征向量
        full_diffs = trans_data - zh_data
        full_diffs_centered = full_diffs - full_diffs.mean(axis=0, keepdims=True)
        U_full, S_full, Vt_full = np.linalg.svd(full_diffs_centered, full_matrices=False)
        # Vt_full: [min(N,P), P] — 行是主方向 (在P维空间中)
        # 但N << P时, 只有N个非零奇异值
        # 主特征向量 = Vt_full[0, :] (第一个右奇异向量)
        
        # Bootstrap
        boot_top_vectors = []  # 每个bootstrap的top-3特征向量
        boot_eigenvalues = []
        
        for b in range(n_bootstrap):
            idx = np.random.choice(N, size=n_subsample, replace=True)
            boot_zh = zh_data[idx]
            boot_trans = trans_data[idx]
            boot_diffs = boot_trans - boot_zh
            boot_diffs_centered = boot_diffs - boot_diffs.mean(axis=0, keepdims=True)
            
            # SVD
            U_b, S_b, Vt_b = np.linalg.svd(boot_diffs_centered, full_matrices=False)
            
            # top-3特征向量 (在P维空间中)
            k = min(3, Vt_b.shape[0])
            boot_top_vectors.append(Vt_b[:k, :])  # [k, P]
            boot_eigenvalues.append(S_b[:5])
        
        # 计算特征向量稳定性
        # 1. Top-1向量与全样本top-1的cosine相似度
        top1_cosines = []
        for b_vecs in boot_top_vectors:
            if b_vecs.shape[0] > 0:
                cos = np.abs(np.dot(b_vecs[0], Vt_full[0])) / (
                    np.linalg.norm(b_vecs[0]) * np.linalg.norm(Vt_full[0]) + 1e-10)
                top1_cosines.append(cos)
        
        # 2. Top-1向量在bootstrap间的互相关
        if len(boot_top_vectors) >= 2:
            pairwise_cos_top1 = []
            for i in range(0, min(20, len(boot_top_vectors))):
                for j in range(i+1, min(20, len(boot_top_vectors))):
                    if boot_top_vectors[i].shape[0] > 0 and boot_top_vectors[j].shape[0] > 0:
                        cos = np.abs(np.dot(boot_top_vectors[i][0], boot_top_vectors[j][0])) / (
                            np.linalg.norm(boot_top_vectors[i][0]) * np.linalg.norm(boot_top_vectors[j][0]) + 1e-10)
                        pairwise_cos_top1.append(cos)
        else:
            pairwise_cos_top1 = [0]
        
        # 3. 特征值稳定性 (变异系数)
        boot_eigs_array = np.array(boot_eigenvalues)  # [n_bootstrap, 5]
        eig1_cv = np.std(boot_eigs_array[:, 0]) / np.mean(boot_eigs_array[:, 0]) if np.mean(boot_eigs_array[:, 0]) > 0 else 0
        eig1_eig2_ratio = np.mean(boot_eigs_array[:, 0]) / np.mean(boot_eigs_array[:, 1]) if np.mean(boot_eigs_array[:, 1]) > 0 else 0
        
        # 4. Top-3子空间稳定性
        top3_subspace_cosines = []
        for b_vecs in boot_top_vectors:
            if b_vecs.shape[0] >= min(3, Vt_full.shape[0]):
                k = min(3, b_vecs.shape[0], Vt_full.shape[0])
                # 子空间投影: boot top-k到full top-k的投影比
                V_full_k = Vt_full[:k, :].T  # [P, k]
                V_boot_k = b_vecs[:k, :].T  # [P, k]
                # 投影矩阵 P_full = V_full_k @ V_full_k^T
                proj = V_full_k @ (V_full_k.T @ V_boot_k)  # [P, k]
                subspace_cos = np.linalg.norm(proj) / (np.linalg.norm(V_boot_k) + 1e-10)
                top3_subspace_cosines.append(subspace_cos)
        
        # 判定
        mean_top1_cos = np.mean(top1_cosines)
        mean_pairwise_cos = np.mean(pairwise_cos_top1)
        
        if mean_top1_cos > 0.9 and mean_pairwise_cos > 0.8:
            stability_verdict = "HIGHLY_STABLE"
        elif mean_top1_cos > 0.7 and mean_pairwise_cos > 0.5:
            stability_verdict = "MODERATELY_STABLE"
        elif mean_top1_cos > 0.5:
            stability_verdict = "WEAKLY_STABLE"
        else:
            stability_verdict = "UNSTABLE"
        
        results[f"L{l}"] = {
            "top1_cosine_to_full": {
                "mean": float(np.mean(top1_cosines)),
                "std": float(np.std(top1_cosines)),
                "median": float(np.median(top1_cosines)),
            },
            "pairwise_top1_cosine": {
                "mean": float(np.mean(pairwise_cos_top1)),
                "std": float(np.std(pairwise_cos_top1)),
            },
            "top3_subspace_cosine": {
                "mean": float(np.mean(top3_subspace_cosines)) if top3_subspace_cosines else 0,
                "std": float(np.std(top3_subspace_cosines)) if top3_subspace_cosines else 0,
            },
            "eigenvalue_stability": {
                "eig1_cv": float(eig1_cv),
                "eig1_eig2_ratio": float(eig1_eig2_ratio),
            },
            "stability_verdict": stability_verdict,
        }
        
        print(f"  L{l}: top1_cos_to_full={np.mean(top1_cosines):.3f}±{np.std(top1_cosines):.3f}, "
              f"pairwise_top1_cos={np.mean(pairwise_cos_top1):.3f}, "
              f"top3_subspace={np.mean(top3_subspace_cosines) if top3_subspace_cosines else 0:.3f}, "
              f"eig1/eig2={eig1_eig2_ratio:.2f}, verdict={stability_verdict}")
    
    return results


# ============================================================
# Exp 3: Causal Eigenspace Intervention (因果特征子空间干预)
# ============================================================
def exp3_causal_intervention(model, tokenizer, device, n_layers, mlp_outs, d_model,
                              test_pairs=None, n_test=30):
    """
    因果特征子空间干预: 在spike eigenspace上做干预, 观察翻译输出变化
    
    三种干预:
    1. Remove: 在信号子空间上做零化 (减去信号子空间投影)
    2. Amplify: 在信号子空间上做放大 (x3)
    3. Rotate: 在信号子空间内做90°旋转
    
    如果翻译行为可控改变 → spike是计算基底
    如果不可控 → spike是统计副现象
    
    注意: 只干预L12 (4个信号维度) 和 L18 (8个信号维度) 的spike子空间
    """
    print("\n" + "="*70)
    print("Exp 3: Causal Eigenspace Intervention — 因果特征子空间干预")
    print("="*70)
    print("核心逻辑: 干预spike子空间, 翻译输出是否可控改变?")
    
    if test_pairs is None:
        test_pairs = WORD_PAIRS[:n_test]
    
    layers = get_layers(model)
    
    # 目标层: L12 (最深的低秩集中点) 和 L18/L27 (其他集中点)
    # 先从Phase 114结果获取信号维度数, 这里用固定值
    target_layers_dims = {12: 4, 18: 8, 27: 18, 35: 14}  # L12:4维, L18:8维
    
    # 首先收集各层的spike子空间
    print("\n--- 收集spike子空间 ---")
    spike_subspaces = {}  # {层: 投影矩阵 [P, k]}
    
    for l, k in target_layers_dims.items():
        if l not in mlp_outs['zh'] or l not in mlp_outs['trans']:
            print(f"  L{l}: 无数据, 跳过")
            continue
        
        zh_data = mlp_outs['zh'][l]
        trans_data = mlp_outs['trans'][l]
        N = zh_data.shape[0]
        P = zh_data.shape[1]
        
        diffs = trans_data - zh_data
        diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
        
        # SVD获取主方向
        U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
        
        # Spike子空间: top-k右奇异向量
        k_actual = min(k, Vt.shape[0])
        spike_subspaces[l] = {
            'V': Vt[:k_actual, :].T,  # [P, k_actual] — 投影矩阵
            'S': S[:k_actual],
            'k': k_actual,
            'mean_diff': diffs.mean(axis=0),  # 平均差分 (用于baseline)
        }
        
        print(f"  L{l}: k={k_actual}, top-5 S={[f'{x:.2f}' for x in S[:5]]}, "
              f"总方差解释={np.sum(S[:k_actual]**2)/np.sum(S**2):.3f}")
    
    if not spike_subspaces:
        print("  无可用spike子空间!")
        return {}
    
    # 定义翻译评估函数
    def evaluate_translation(zh_word, en_word, model, tokenizer, device, layers_list):
        """评估翻译输出是否包含目标英文词"""
        trans_prompt = f"Translate the following Chinese word: {zh_word}"
        inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"])
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
        
        # 目标词的token id
        target_ids = tokenizer.encode(en_word, add_special_tokens=False)
        
        # 检查top-10预测
        top10_ids = torch.argsort(logits, descending=True)[:10].cpu().numpy()
        top10_tokens = [tokenizer.decode([t]) for t in top10_ids]
        top10_probs = [float(probs[t]) for t in top10_ids]
        
        # 目标词是否在top-10
        target_in_top10 = any(tid in top10_ids for tid in target_ids)
        
        # 目标词的log概率
        target_log_probs = []
        for tid in target_ids:
            if tid < len(probs):
                target_log_probs.append(float(torch.log(probs[tid] + 1e-10)))
        
        result = {
            "target_in_top10": target_in_top10,
            "target_log_prob": float(np.mean(target_log_probs)) if target_log_probs else -100,
            "top3_tokens": top10_tokens[:3],
            "top3_probs": top10_probs[:3],
        }
        
        del inputs, outputs, logits, probs
        gc.collect()
        torch.cuda.empty_cache()
        
        return result
    
    # 进行干预实验
    results = {}
    
    for l, subspace_info in spike_subspaces.items():
        if l >= len(layers):
            continue
        
        V = subspace_info['V']  # [P, k]
        k = subspace_info['k']
        mean_diff = subspace_info['mean_diff']
        
        print(f"\n--- L{l} 干预 (k={k}) ---")
        
        layer_results = {
            "layer": l, "spike_dim": k,
            "baseline": [], "remove": [], "amplify": [], "rotate": [],
            "random_remove": [],  # 控制: 在随机子空间上remove
        }
        
        for zh, en in test_pairs:
            # Baseline (无干预)
            base_result = evaluate_translation(zh, en, model, tokenizer, device, layers)
            layer_results["baseline"].append({
                "zh": zh, "en": en, **base_result
            })
            
            # 干预: Remove (减去spike子空间投影)
            # 实现: 在L层的输出上, 减去spike子空间方向的投影
            # hook: output = output - P_spike @ P_spike^T @ output
            # 注意: 需要匹配模型输出的dtype (bfloat16)
            
            # Remove spike subspace
            def make_remove_hook(proj_matrix_np):
                proj_matrix = torch.tensor(proj_matrix_np, dtype=torch.float32, device=device)
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].detach().clone().float()  # 转float32
                        # h: [batch, seq, P]
                        last_h = h[0, -1, :].unsqueeze(0)  # [1, P]
                        # 投影到spike子空间
                        proj = proj_matrix @ (proj_matrix.T @ last_h.T)  # [P, 1]
                        # 减去投影
                        h[0, -1, :] -= proj.squeeze() * 0.5  # 只减去50%, 避免过大扰动
                        return (h.to(output[0].dtype),) + output[1:]
                    else:
                        h = output.detach().clone().float()
                        last_h = h[0, -1, :].unsqueeze(0)
                        proj = proj_matrix @ (proj_matrix.T @ last_h.T)
                        h[0, -1, :] -= proj.squeeze() * 0.5
                        return h.to(output.dtype)
                return hook_fn
            
            hooks = [layers[l].register_forward_hook(make_remove_hook(V))]
            remove_result = evaluate_translation(zh, en, model, tokenizer, device, layers)
            for h in hooks:
                h.remove()
            layer_results["remove"].append({
                "zh": zh, "en": en, **remove_result
            })
            
            # Amplify spike subspace (x3)
            def make_amplify_hook(proj_matrix_np, factor=2.0):
                proj_matrix = torch.tensor(proj_matrix_np, dtype=torch.float32, device=device)
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].detach().clone().float()
                        last_h = h[0, -1, :].unsqueeze(0)
                        proj = proj_matrix @ (proj_matrix.T @ last_h.T)
                        h[0, -1, :] += proj.squeeze() * factor  # 放大spike子空间分量
                        return (h.to(output[0].dtype),) + output[1:]
                    else:
                        h = output.detach().clone().float()
                        last_h = h[0, -1, :].unsqueeze(0)
                        proj = proj_matrix @ (proj_matrix.T @ last_h.T)
                        h[0, -1, :] += proj.squeeze() * factor
                        return h.to(output.dtype)
                return hook_fn
            
            hooks = [layers[l].register_forward_hook(make_amplify_hook(V, factor=2.0))]
            amplify_result = evaluate_translation(zh, en, model, tokenizer, device, layers)
            for h in hooks:
                h.remove()
            layer_results["amplify"].append({
                "zh": zh, "en": en, **amplify_result
            })
            
            # Rotate spike subspace (90° within subspace)
            # 在k维子空间内旋转90°: 对V做QR分解, 用正交旋转矩阵
            if k >= 2:
                # 构造旋转矩阵: 在前两维之间旋转90°
                R = np.eye(k)
                R[0, 0] = 0; R[0, 1] = -1
                R[1, 0] = 1; R[1, 1] = 0
                V_rotated = V @ R  # 旋转后的子空间
                
                def make_rotate_hook(proj_old_np, proj_new_np, scale=1.0):
                    proj_old = torch.tensor(proj_old_np, dtype=torch.float32, device=device)
                    proj_new = torch.tensor(proj_new_np, dtype=torch.float32, device=device)
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0].detach().clone().float()
                            last_h = h[0, -1, :].unsqueeze(0)
                            # 减去旧子空间投影, 加上新子空间投影
                            proj_old_coeff = proj_old.T @ last_h.T  # [k, 1]
                            proj_old_vec = proj_old @ proj_old_coeff  # [P, 1]
                            proj_new_vec = proj_new @ proj_old_coeff  # [k, 1] → [P, 1]
                            h[0, -1, :] = h[0, -1, :] - proj_old_vec.squeeze() + proj_new_vec.squeeze()
                            return (h.to(output[0].dtype),) + output[1:]
                        else:
                            h = output.detach().clone().float()
                            last_h = h[0, -1, :].unsqueeze(0)
                            proj_old_coeff = proj_old.T @ last_h.T
                            proj_old_vec = proj_old @ proj_old_coeff
                            proj_new_vec = proj_new @ proj_old_coeff
                            h[0, -1, :] = h[0, -1, :] - proj_old_vec.squeeze() + proj_new_vec.squeeze()
                            return h.to(output.dtype)
                    return hook_fn
                
                hooks = [layers[l].register_forward_hook(make_rotate_hook(V, V_rotated))]
                rotate_result = evaluate_translation(zh, en, model, tokenizer, device, layers)
                for h in hooks:
                    h.remove()
                layer_results["rotate"].append({
                    "zh": zh, "en": en, **rotate_result
                })
            
            # 控制: 在随机子空间上remove (同维度)
            V_random = np.random.randn(P, k)
            V_random, _ = np.linalg.qr(V_random)  # 正交化
            
            hooks = [layers[l].register_forward_hook(make_remove_hook(V_random))]
            random_remove_result = evaluate_translation(zh, en, model, tokenizer, device, layers)
            for h in hooks:
                h.remove()
            layer_results["random_remove"].append({
                "zh": zh, "en": en, **random_remove_result
            })
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # 汇总结果
        def summarize_intervention(key):
            items = layer_results[key]
            target_in_top10 = sum(1 for x in items if x.get("target_in_top10", False))
            log_probs = [x.get("target_log_prob", -100) for x in items]
            return {
                "target_in_top10_count": target_in_top10,
                "target_in_top10_rate": float(target_in_top10 / len(items)) if items else 0,
                "mean_log_prob": float(np.mean(log_probs)),
                "std_log_prob": float(np.std(log_probs)),
            }
        
        base_summary = summarize_intervention("baseline")
        remove_summary = summarize_intervention("remove")
        amplify_summary = summarize_intervention("amplify")
        rotate_summary = summarize_intervention("rotate")
        random_remove_summary = summarize_intervention("random_remove")
        
        # 因果判定
        # 如果spike remove比random remove更影响翻译 → spike有因果力量
        log_prob_change_remove = base_summary["mean_log_prob"] - remove_summary["mean_log_prob"]
        log_prob_change_random = base_summary["mean_log_prob"] - random_remove_summary["mean_log_prob"]
        
        causal_effect = log_prob_change_remove - log_prob_change_random
        
        if causal_effect > 0.5 and log_prob_change_remove > 0.3:
            causal_verdict = "CAUSAL_SUBSTRATE"  # spike是因果计算基底
        elif causal_effect > 0.2:
            causal_verdict = "PARTIALLY_CAUSAL"
        elif abs(causal_effect) < 0.2:
            causal_verdict = "NO_CAUSAL_EFFECT"  # spike是统计副现象
        else:
            causal_verdict = "NEGATIVE_CAUSAL"  # spike干预反而提升翻译
        
        results[f"L{l}"] = {
            "spike_dim": k,
            "baseline": base_summary,
            "remove_spike": remove_summary,
            "amplify_spike": amplify_summary,
            "rotate_spike": rotate_summary,
            "remove_random": random_remove_summary,
            "log_prob_change_remove_spike": float(log_prob_change_remove),
            "log_prob_change_remove_random": float(log_prob_change_random),
            "causal_effect_size": float(causal_effect),
            "causal_verdict": causal_verdict,
        }
        
        print(f"\n  L{l} 汇总:")
        print(f"    Baseline:     top10_rate={base_summary['target_in_top10_rate']:.2f}, "
              f"log_prob={base_summary['mean_log_prob']:.2f}")
        print(f"    Remove spike: top10_rate={remove_summary['target_in_top10_rate']:.2f}, "
              f"log_prob={remove_summary['mean_log_prob']:.2f} (Δ={log_prob_change_remove:.3f})")
        print(f"    Amplify spike: top10_rate={amplify_summary['target_in_top10_rate']:.2f}, "
              f"log_prob={amplify_summary['mean_log_prob']:.2f}")
        print(f"    Remove random: top10_rate={random_remove_summary['target_in_top10_rate']:.2f}, "
              f"log_prob={random_remove_summary['mean_log_prob']:.2f} (Δ={log_prob_change_random:.3f})")
        print(f"    Causal effect={causal_effect:.3f}, verdict={causal_verdict}")
    
    return results


# ============================================================
# Exp 4: Tokenizer Control (分词器控制)
# ============================================================
def exp4_tokenizer_control(mlp_outs, d_model, token_info):
    """
    分词器控制检验: 低秩集中是否由token统计产生?
    
    控制:
    1. 词元数量差异 (中文vs英文prompt的token数差)
    2. 子词长度
    3. 合并模式
    
    方法: 将差分按token统计分组, 看spike是否在所有组中都存在
    """
    print("\n" + "="*70)
    print("Exp 4: Tokenizer Control — 低秩集中是否由token统计产生?")
    print("="*70)
    
    results = {}
    
    for l in SAMPLE_LAYERS:
        if l not in mlp_outs['zh'] or l not in mlp_outs['trans']:
            continue
        
        zh_data = mlp_outs['zh'][l]
        trans_data = mlp_outs['trans'][l]
        N = zh_data.shape[0]
        P = zh_data.shape[1]
        
        if N < 20 or len(token_info) < N:
            continue
        
        # 全样本差分
        all_diffs = trans_data - zh_data
        full_result = compute_mp_analysis(all_diffs, N, P)
        
        # 按token数量差异分组
        token_diffs = [info['token_count_diff'] for info in token_info[:N]]
        median_token_diff = np.median(token_diffs)
        
        # 组1: token差异小 (<= 中位数)
        group1_idx = [i for i, d in enumerate(token_diffs) if d <= median_token_diff]
        # 组2: token差异大 (> 中位数)
        group2_idx = [i for i, d in enumerate(token_diffs) if d > median_token_diff]
        
        group_results = {}
        for gname, gidx in [("small_token_diff", group1_idx), ("large_token_diff", group2_idx)]:
            if len(gidx) < 10:
                group_results[gname] = {"n": len(gidx), "skipped": True}
                continue
            
            g_zh = zh_data[gidx]
            g_trans = trans_data[gidx]
            g_diffs = g_trans - g_zh
            g_result = compute_mp_analysis(g_diffs, len(gidx), P)
            
            group_results[gname] = {
                "n": len(gidx),
                "true_dim": g_result['mp_result']['true_dimensionality'],
                "pr_signal": g_result['mp_result']['pr_signal'],
                "signal_ratio": g_result['mp_result']['signal_variance_ratio'],
                "top5_eigenvalues": [float(x) for x in g_result['eigenvalues'][:5]],
            }
        
        # 按中文字符数分组 (单字 vs 多字)
        zh_char_counts = [len(info['zh_word']) for info in token_info[:N]]
        
        single_char_idx = [i for i, c in enumerate(zh_char_counts) if c == 1]
        multi_char_idx = [i for i, c in enumerate(zh_char_counts) if c > 1]
        
        char_group_results = {}
        for gname, gidx in [("single_char_zh", single_char_idx), ("multi_char_zh", multi_char_idx)]:
            if len(gidx) < 10:
                char_group_results[gname] = {"n": len(gidx), "skipped": True}
                continue
            
            g_zh = zh_data[gidx]
            g_trans = trans_data[gidx]
            g_diffs = g_trans - g_zh
            g_result = compute_mp_analysis(g_diffs, len(gidx), P)
            
            char_group_results[gname] = {
                "n": len(gidx),
                "true_dim": g_result['mp_result']['true_dimensionality'],
                "pr_signal": g_result['mp_result']['pr_signal'],
                "signal_ratio": g_result['mp_result']['signal_variance_ratio'],
            }
        
        # 按英文词长度分组
        en_word_lengths = [len(info['en_word']) for info in token_info[:N]]
        median_en_len = np.median(en_word_lengths)
        
        short_en_idx = [i for i, l in enumerate(en_word_lengths) if l <= median_en_len]
        long_en_idx = [i for i, l in enumerate(en_word_lengths) if l > median_en_len]
        
        en_len_results = {}
        for gname, gidx in [("short_en", short_en_idx), ("long_en", long_en_idx)]:
            if len(gidx) < 10:
                en_len_results[gname] = {"n": len(gidx), "skipped": True}
                continue
            
            g_zh = zh_data[gidx]
            g_trans = trans_data[gidx]
            g_diffs = g_trans - g_zh
            g_result = compute_mp_analysis(g_diffs, len(gidx), P)
            
            en_len_results[gname] = {
                "n": len(gidx),
                "true_dim": g_result['mp_result']['true_dimensionality'],
                "pr_signal": g_result['mp_result']['pr_signal'],
            }
        
        # 判定: spike是否在所有分组中一致存在
        token_diff_dims = [v.get("true_dim", 0) for v in group_results.values() if not v.get("skipped")]
        char_diff_dims = [v.get("true_dim", 0) for v in char_group_results.values() if not v.get("skipped")]
        
        full_dim = full_result['mp_result']['true_dimensionality']
        
        # 如果分组间的true_dim差异很大 → spike依赖token统计
        if token_diff_dims and max(token_diff_dims) - min(token_diff_dims) > full_dim * 0.5:
            tokenizer_verdict = "TOKENIZER_DEPENDENT"
        elif token_diff_dims and all(d > 0 for d in token_diff_dims):
            tokenizer_verdict = "ROBUST_TO_TOKENIZER"
        else:
            tokenizer_verdict = "INCONCLUSIVE"
        
        results[f"L{l}"] = {
            "full_sample": {
                "true_dim": full_dim,
                "pr_signal": full_result['mp_result']['pr_signal'],
            },
            "by_token_diff": group_results,
            "by_zh_char_count": char_group_results,
            "by_en_word_length": en_len_results,
            "tokenizer_verdict": tokenizer_verdict,
        }
        
        print(f"  L{l}: full_dim={full_dim}, "
              f"token_diff_dims={token_diff_dims}, "
              f"char_diff_dims={char_diff_dims}, "
              f"verdict={tokenizer_verdict}")
    
    return results


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["all", "1", "2", "3", "4"])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'='*70}")
    print(f"Phase 115: 置换破坏 + 自举稳定性 + 因果特征子空间干预")
    print(f"模型: {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    layers = get_layers(model)
    n_layers = len(layers)
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    print(f"模型: {model_info.model_class}, {n_layers}层, d_model={d_model}, device={device}")
    
    # 更新采样层
    if n_layers <= 36:
        global SAMPLE_LAYERS
        SAMPLE_LAYERS = [l for l in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35] if l < n_layers]
    
    all_results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}
    
    # 收集MLP输出 (所有实验共用)
    print("\n--- 收集MLP输出 ---")
    mlp_outs = collect_mlp_outputs(model, tokenizer, device, n_layers, WORD_PAIRS)
    
    # Exp 1: Permutation Destruction
    if args.exp in ["all", "1"]:
        print("\n--- Exp 1: Permutation Destruction ---")
        exp1_result = exp1_permutation_destruction(mlp_outs, d_model, n_permutations=20)
        all_results["exp1_permutation"] = exp1_result
        
        out_path = os.path.join(OUT_DIR, f"phase115_exp1_{model_name}_permutation.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(exp1_result, f, indent=2, ensure_ascii=False)
        print(f"  保存到 {out_path}")
    
    # Exp 2: Bootstrap Stability
    if args.exp in ["all", "2"]:
        print("\n--- Exp 2: Bootstrap Stability ---")
        exp2_result = exp2_bootstrap_stability(mlp_outs, d_model, n_bootstrap=50)
        all_results["exp2_bootstrap"] = exp2_result
        
        out_path = os.path.join(OUT_DIR, f"phase115_exp2_{model_name}_bootstrap.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(exp2_result, f, indent=2, ensure_ascii=False)
        print(f"  保存到 {out_path}")
    
    # Exp 3: Causal Intervention (需要模型, 计算密集)
    if args.exp in ["all", "3"]:
        print("\n--- Exp 3: Causal Eigenspace Intervention ---")
        exp3_result = exp3_causal_intervention(model, tokenizer, device, n_layers, mlp_outs, d_model)
        all_results["exp3_causal"] = exp3_result
        
        out_path = os.path.join(OUT_DIR, f"phase115_exp3_{model_name}_causal.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(exp3_result, f, indent=2, ensure_ascii=False)
        print(f"  保存到 {out_path}")
    
    # Exp 4: Tokenizer Control
    if args.exp in ["all", "4"]:
        print("\n--- Exp 4: Tokenizer Control ---")
        # 需要收集带tokenizer信息的数据
        _, _, token_info = collect_hidden_states_with_tokenizer_info(
            model, tokenizer, device, n_layers, WORD_PAIRS[:30])  # 只用30个以节省时间
        exp4_result = exp4_tokenizer_control(mlp_outs, d_model, token_info)
        all_results["exp4_tokenizer"] = exp4_result
        
        out_path = os.path.join(OUT_DIR, f"phase115_exp4_{model_name}_tokenizer.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(exp4_result, f, indent=2, ensure_ascii=False)
        print(f"  保存到 {out_path}")
    
    # 保存全部结果
    all_out_path = os.path.join(OUT_DIR, f"phase115_{model_name}_all_results.json")
    with open(all_out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n全部结果保存到 {all_out_path}")
    
    # 释放模型
    release_model(model)
    print("\nPhase 115 完成!")


if __name__ == "__main__":
    main()
