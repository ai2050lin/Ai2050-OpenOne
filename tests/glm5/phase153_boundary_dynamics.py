"""
Phase 153: 边界驱动统计动力学 — 跨模型验证
============================================

理论修正(基于Phase 152反思):
1. R² ≠ MI — R²只测线性可预测性, 不等于互信息
2. PCA overlap不是二阶结构正确量度 — 改用CCA
3. LayerNorm是尺度投影, 不是信息销毁器
4. 核心方向: "边界驱动统计动力学"

四大实验(优先级排序):
- Exp A: Logit Boundary Propagation — margin如何在层间传播
- Exp B: CCA between layers — 真正的二阶结构保留量度
- Exp C: Attention Routing Topology — 注意力图稳定性
- Exp D: Trajectory Stability — 微扰后的轨迹分叉

数据量: 50个句子(重要结果加大到100)
扰动次数: 200(重要实验加大到500)

用法:
  python tests/glm5/phase153_boundary_dynamics.py qwen3
  python tests/glm5/phase153_boundary_dynamics.py deepseek7b
  python tests/glm5/phase153_boundary_dynamics.py glm4
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

OUTPUT_DIR = Path("tests/glm5_temp")

# ===== 扩大测试数据集 =====
TEST_PROMPTS = [
    "The scientist discovered that the",
    "In the morning, she decided to",
    "The book on the table was about",
    "After the rain stopped, the children",
    "The most important thing about science is",
    "When the sun sets over the ocean,",
    "She walked into the room and saw",
    "The professor explained that the theory",
    "Despite the challenges, the team managed",
    "The ancient city was known for its",
    "He realized that the answer was",
    "The relationship between language and thought",
    "Every morning she would read the",
    "The experiment showed that the results",
    "Music has the power to change how",
    "The government announced that the new policy",
    "In the future, artificial intelligence will",
    "The philosopher argued that consciousness is",
    "After years of research, they found that",
    "The key difference between the two approaches is",
    "The cat sat on the windowsill and watched",
    "Through the telescope, they observed a new",
    "The river flowed gently through the valley",
    "She opened the letter and read the",
    "The painting on the wall depicted a",
    "During the concert, the audience was",
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
    "At the conference, the speaker presented",
    "The children played in the garden while",
    "The old man smiled and said that",
    "The company decided to invest in new",
    "The train arrived at the station just as",
    "Through the window, she could see the",
    "The puzzle was more difficult than they",
    "The artist carefully mixed the colors to",
    "The report concluded that the main cause",
    "She remembered the day when they first",
    "The mountain was covered with snow and",
    "The debate focused on whether the government",
    "He turned the key and opened the door to",
]

N_SENTENCES = 50
EPSILON = 1.0


def load_model_custom(model_name: str):
    """加载模型 — 处理DS7B的sliding window attention问题"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = MODEL_CONFIGS[model_name]
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16

    print(f"  Loading {model_name} (8bit={use_8bit})...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gc.collect()
    torch.cuda.empty_cache()

    if use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        attn_impl = "sdpa" if model_name == "deepseek7b" else "eager"
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


def get_device_for_input(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def get_sample_layers(n_layers):
    """动态采样 — 覆盖早期+中期+晚期"""
    if n_layers <= 12:
        return list(range(0, n_layers + 1))
    # 早期密采样, 中期均匀, 晚期密采样
    early = [0, 1, 2, 3, 4]
    mid_step = max(1, (n_layers - 10) // 5)
    mid = list(range(5, n_layers - 4, mid_step))
    late = [n_layers - 4, n_layers - 3, n_layers - 2, n_layers - 1, n_layers]
    layers = sorted(set(early + mid + late))
    return layers


# ============================================================
# Exp A: Logit Boundary Propagation
# ============================================================
def expA_logit_boundary_propagation(model, tokenizer, model_name):
    """
    核心问题: logit margin m = z_top1 - z_top2 如何随层间传播?
    
    方法:
    1. 在不同层注入扰动
    2. 测量最终logit空间中margin的变化
    3. 计算: margin变化量 vs 注入层深度 的关系
    
    这直接测试: "边界动力学"是否真的是核心传播机制
    """
    print("\n" + "="*60)
    print("Exp A: Logit Boundary Propagation")
    print("="*60)

    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    d_model = info.d_model

    sample_layers = get_sample_layers(n_layers)
    # 只测试中间层附近的层(完整传播路径)
    inject_layers = [0, 1, 2] + list(range(4, n_layers, max(1, n_layers // 8))) + [n_layers - 2, n_layers - 1]
    inject_layers = sorted(set(inject_layers))
    # 限制数量避免太慢
    inject_layers = inject_layers[:15]

    n_sents = min(30, N_SENTENCES)
    n_perturb_per_layer = 50

    W_U = get_W_U(model, model_name)  # [vocab, d_model]

    # 对每个注入层, 测量: margin变化分布
    results_per_layer = {}

    for inject_li in inject_layers:
        margin_changes = []
        baseline_margins = []
        top1_stay_rates = []

        for sent_idx in range(n_sents):
            prompt = TEST_PROMPTS[sent_idx]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            last_pos = input_ids.shape[1] - 1

            # 基线logits
            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
            clean_logits = out_clean.logits[0, -1, :].float().cpu().numpy()
            sorted_ids = np.argsort(-clean_logits)
            top1_id = sorted_ids[0]
            top2_id = sorted_ids[1]
            baseline_margin = clean_logits[top1_id] - clean_logits[top2_id]
            baseline_margins.append(baseline_margin)

            # 扰动: 随机方向
            top1_changed_count = 0
            total_perturb = 0

            for p_idx in range(n_perturb_per_layer):
                np.random.seed(inject_li * 1000 + sent_idx * 100 + p_idx)
                delta = np.random.randn(d_model)
                delta = delta / np.linalg.norm(delta) * EPSILON

                layers_list = get_layers(model)
                delta_tensor = torch.tensor(delta, dtype=torch.float32)

                def make_hook(pos, delta_t):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            out = output[0].clone()
                            out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                            return (out,) + output[1:]
                        else:
                            out = output.clone()
                            out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                            return out
                    return hook

                hooks = [layers_list[inject_li].register_forward_hook(make_hook(last_pos, delta_tensor))]

                try:
                    with torch.no_grad():
                        out_p = model(input_ids=input_ids, attention_mask=attention_mask)
                    perturbed_logits = out_p.logits[0, -1, :].float().cpu().numpy()
                    perturbed_top1 = int(np.argmax(perturbed_logits))

                    perturbed_margin_val = perturbed_logits[top1_id] - perturbed_logits[top2_id]
                    margin_change = abs(baseline_margin - perturbed_margin_val)
                    margin_changes.append(margin_change)

                    if perturbed_top1 != top1_id:
                        top1_changed_count += 1
                    total_perturb += 1
                except:
                    pass

                for h in hooks:
                    h.remove()

            if total_perturb > 0:
                top1_stay_rates.append(1.0 - top1_changed_count / total_perturb)

        if margin_changes:
            results_per_layer[inject_li] = {
                'mean_margin_change': float(np.mean(margin_changes)),
                'std_margin_change': float(np.std(margin_changes)),
                'mean_baseline_margin': float(np.mean(baseline_margins)),
                'top1_stay_rate': float(np.mean(top1_stay_rates)) if top1_stay_rates else 0,
                'n_samples': len(margin_changes),
            }
            r = results_per_layer[inject_li]
            print(f"  Inject L{inject_li:>2d}: margin_change={r['mean_margin_change']:.4f}±{r['std_margin_change']:.4f}, "
                  f"baseline_m={r['mean_baseline_margin']:.3f}, stay_rate={r['top1_stay_rate']:.3f}")

    # 分析: 注入层深度 vs margin变化
    print("\n  === Margin Change vs Inject Depth ===")
    depths = sorted(results_per_layer.keys())
    mc_values = [results_per_layer[d]['mean_margin_change'] for d in depths]
    stay_values = [results_per_layer[d]['top1_stay_rate'] for d in depths]

    # 增长率
    if len(mc_values) > 2:
        # 后半层 vs 前半层
        mid = len(mc_values) // 2
        early_mc = np.mean(mc_values[:mid])
        late_mc = np.mean(mc_values[mid:])
        print(f"  Early layers margin change: {early_mc:.4f}")
        print(f"  Late layers margin change: {late_mc:.4f}")
        print(f"  Ratio (late/early): {late_mc/early_mc:.2f}x")

    return results_per_layer


# ============================================================
# Exp B: CCA (Canonical Correlation Analysis) between layers
# ============================================================
def expB_cca_between_layers(model, tokenizer, model_name):
    """
    核心问题: 层间真正的统计结构保留量是多少?
    
    修正: 用CCA代替PCA overlap
    - PCA overlap要求u=v → 子空间旋转时会误判为"结构消失"
    - CCA允许u≠v → 正确测量"旋转不变"的统计相关性
    
    CCA = max_{u,v} corr(u^T X, v^T Y)
    """
    print("\n" + "="*60)
    print("Exp B: CCA Between Layers (NOT PCA overlap!)")
    print("="*60)

    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    d_model = info.d_model

    sample_layers = get_sample_layers(n_layers)

    # 收集50个句子在每层的hidden states
    n_sents = min(50, N_SENTENCES)
    all_hs = {}
    for li in sample_layers:
        all_hs[li] = []

    for sent_idx in range(n_sents):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)

        for li in sample_layers:
            vec = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            all_hs[li].append(vec)

        if sent_idx % 10 == 0:
            print(f"  Collecting hidden states: {sent_idx}/{n_sents}")

    # 参考层: hs[1]
    ref_layer = 1
    X_ref = np.array(all_hs[ref_layer])  # [N, d_model]
    X_ref_centered = X_ref - X_ref.mean(axis=0)

    # 对每层计算CCA
    n_cca_components = 10
    cca_results = {}

    def compute_cca(X, Y, n_components=10):
        """
        计算X和Y之间的CCA
        X: [N, p], Y: [N, q]
        返回: top-k canonical correlations
        """
        N = X.shape[0]
        if N < 5:
            return np.zeros(n_components)

        # SVD方法(更稳定)
        # 1. 中心化
        X_c = X - X.mean(axis=0)
        Y_c = Y - Y.mean(axis=0)

        # 2. 白化(降维到min(N-1, d))
        p_reduce = min(N - 1, X_c.shape[1], 100)
        q_reduce = min(N - 1, Y_c.shape[1], 100)

        try:
            # SVD for whitening
            Ux, sx, Vtx = np.linalg.svd(X_c, full_matrices=False)
            Uy, sy, Vty = np.linalg.svd(Y_c, full_matrices=False)

            # 保留主要成分
            X_white = Ux[:, :p_reduce]  # [N, p_reduce]
            Y_white = Uy[:, :q_reduce]  # [N, q_reduce]

            # 3. CCA via SVD of cross-covariance
            M = X_white.T @ Y_white / (N - 1)  # [p_reduce, q_reduce]

            # SVD of M gives canonical correlations
            U_m, s_m, Vt_m = np.linalg.svd(M, full_matrices=False)

            # s_m就是canonical correlations
            return s_m[:n_components]
        except:
            return np.zeros(n_components)

    print("\n  --- CCA: hs[1] vs hs[ℓ] ---")
    print(f"  {'Layer':>6} | {'CCA_1':>8} | {'CCA_2':>8} | {'CCA_3':>8} | {'CCA_mean(1-5)':>14} | {'vs R²':>8}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*14}-+-{'-'*8}")

    # 也计算R²作为对比
    for li in sample_layers:
        X_li = np.array(all_hs[li])
        X_li_centered = X_li - X_li.mean(axis=0)

        cca_corrs = compute_cca(X_ref_centered, X_li_centered, n_cca_components)
        cca_results[li] = {
            'cca_correlations': cca_corrs.tolist(),
            'cca_mean_top5': float(np.mean(cca_corrs[:5])),
            'cca_mean_top10': float(np.mean(cca_corrs[:10])),
        }

        # 计算R²做对比
        try:
            n_pca = min(50, d_model, X_ref_centered.shape[0] - 1)
            cov_ref = (X_ref_centered.T @ X_ref_centered) / (X_ref_centered.shape[0] - 1)
            eigvals, eigvecs = np.linalg.eigh(cov_ref)
            idx = np.argsort(-eigvals)[:n_pca]
            X_ref_pca = X_ref_centered @ eigvecs[:, idx]

            cov_li = (X_li_centered.T @ X_li_centered) / (X_li_centered.shape[0] - 1)
            eigvals_li, eigvecs_li = np.linalg.eigh(cov_li)
            idx_li = np.argsort(-eigvals_li)[:n_pca]
            X_li_pca = X_li_centered @ eigvecs_li[:, idx_li]

            N_s = X_ref_pca.shape[0]
            n_train = max(5, N_s * 2 // 3)
            r2_list = []
            for comp_j in range(min(10, X_li_pca.shape[1])):
                y_train = X_li_pca[:n_train, comp_j]
                y_test = X_li_pca[n_train:, comp_j]
                try:
                    W, _, _, _ = np.linalg.lstsq(
                        np.column_stack([X_ref_pca[:n_train], np.ones(n_train)]),
                        y_train, rcond=None)
                    y_pred = np.column_stack([X_ref_pca[n_train:], np.ones(N_s - n_train)]) @ W
                    ss_res = np.sum((y_test - y_pred) ** 2)
                    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                    r2_list.append(max(0, 1 - ss_res / max(ss_tot, 1e-10)))
                except:
                    r2_list.append(0)
            r2_val = float(np.mean(r2_list))
        except:
            r2_val = 0.0

        cca_r = cca_results[li]
        c1 = cca_corrs[0] if len(cca_corrs) > 0 else 0
        c2 = cca_corrs[1] if len(cca_corrs) > 1 else 0
        c3 = cca_corrs[2] if len(cca_corrs) > 2 else 0
        c5m = cca_r['cca_mean_top5']
        print(f"  hs[{li:>3d}] | {c1:>8.4f} | {c2:>8.4f} | {c3:>8.4f} | {c5m:>14.4f} | {r2_val:>8.4f}")

    # 关键对比: CCA vs PCA overlap vs R²
    print("\n  === CRITICAL: CCA vs R² (hs[1] → hs[ℓ]) ===")
    print(f"  CCA measures: 'rotation-invariant statistical correlation'")
    print(f"  R² measures:  'linear predictability'")
    print(f"  If CCA >> R²: information preserved but in rotated coordinates")
    print(f"  If CCA ≈ R²:  information truly lost, not just rotated")

    return cca_results


# ============================================================
# Exp C: Attention Routing Topology
# ============================================================
def expC_attention_routing(model, tokenizer, model_name):
    """
    核心问题: 注意力路由拓扑是否跨prompt稳定?
    
    方法:
    1. 对多个prompt提取attention weights
    2. 计算每个head的entropy profile
    3. 计算跨prompt的head specialization稳定性
    4. 构建attention routing graph
    
    注意: DS7B使用sdpa, 不支持output_attentions
    → 对DS7B使用替代方案: 通过hook提取Q,K,V计算attention
    """
    print("\n" + "="*60)
    print("Exp C: Attention Routing Topology")
    print("="*60)

    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers

    # 检查是否支持output_attentions
    prompt = TEST_PROMPTS[0]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    supports_attentions = True
    try:
        with torch.no_grad():
            test_out = model(input_ids=input_ids, attention_mask=attention_mask,
                             output_attentions=True)
        if test_out.attentions is None:
            supports_attentions = False
    except Exception as e:
        supports_attentions = False
        print(f"  output_attentions not supported: {e}")

    if not supports_attentions:
        print("  Using QKV-hook method for attention extraction...")

    n_sents = min(30, N_SENTENCES)
    # 采样层
    attn_sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))[:8]

    # 每个layer, 每个head: 跨prompt的entropy profile
    # entropy of attention distribution → 衡量"聚焦"vs"扩散"
    head_entropy_profiles = {}  # {layer_idx: {head_idx: [entropy_per_prompt]}}

    for li in attn_sample_layers:
        head_entropy_profiles[li] = {}

    for sent_idx in range(n_sents):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = input_ids.shape[1]
        last_pos = seq_len - 1

        if supports_attentions:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_attentions=True)

            for li_idx, li in enumerate(attn_sample_layers):
                if li < len(out.attentions):
                    attn_weights = out.attentions[li]  # [1, n_heads, seq_len, seq_len]
                    # 最后一个token对各token的attention
                    last_token_attn = attn_weights[0, :, last_pos, :last_pos+1].float().cpu().numpy()
                    n_heads = last_token_attn.shape[0]

                    for h in range(n_heads):
                        if h not in head_entropy_profiles[li]:
                            head_entropy_profiles[li][h] = []
                        # 计算entropy
                        attn_dist = last_token_attn[h]
                        attn_dist = attn_dist / max(attn_dist.sum(), 1e-10)
                        entropy = -np.sum(attn_dist * np.log(attn_dist + 1e-10))
                        max_entropy = np.log(len(attn_dist))
                        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                        head_entropy_profiles[li][h].append(normalized_entropy)
        else:
            # QKV hook方法: 通过W_q, W_k, W_v计算attention
            layers_list = get_layers(model)
            layer = layers_list[li]
            sa = layer.self_attn

            with torch.no_grad():
                # 获取该层的hidden state
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
                h_input = out.hidden_states[li][0]  # [seq_len, d_model]

                # 计算Q, K, V
                W_q = sa.q_proj.weight.detach().float()
                W_k = sa.k_proj.weight.detach().float()
                W_v = sa.v_proj.weight.detach().float()

                Q = (h_input @ W_q.T).cpu().numpy()  # [seq_len, d_model]
                K = (h_input @ W_k.T).cpu().numpy()

                # 简化: 取前几个head
                n_heads = 8  # 只分析前8个head
                d_head = Q.shape[1] // n_heads

                for h in range(n_heads):
                    if h not in head_entropy_profiles[li]:
                        head_entropy_profiles[li][h] = []

                    Q_h = Q[last_pos, h*d_head:(h+1)*d_head]  # [d_head]
                    K_h = K[:last_pos+1, h*d_head:(h+1)*d_head]  # [seq_len, d_head]

                    # Attention scores
                    scores = K_h @ Q_h / np.sqrt(d_head)
                    # Softmax
                    exp_scores = np.exp(scores - np.max(scores))
                    attn_dist = exp_scores / exp_scores.sum()

                    entropy = -np.sum(attn_dist * np.log(attn_dist + 1e-10))
                    max_entropy = np.log(len(attn_dist))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    head_entropy_profiles[li][h].append(normalized_entropy)

    # 分析: head specialization stability
    print("\n  --- Head Entropy Stability (跨prompt) ---")

    # 对每个layer, 找出: 哪些head始终低entropy(聚焦), 哪些始终高entropy(扩散)
    head_specialization = {}

    for li in sorted(head_entropy_profiles.keys()):
        head_data = head_entropy_profiles[li]
        if not head_data:
            continue

        focused_heads = []  # entropy < 0.3 (聚焦到少数token)
        diffuse_heads = []  # entropy > 0.7 (扩散到多数token)
        mid_heads = []

        for h, entropies in head_data.items():
            if len(entropies) < 5:
                continue
            mean_ent = np.mean(entropies)
            std_ent = np.std(entropies)

            if mean_ent < 0.3:
                focused_heads.append((h, mean_ent, std_ent))
            elif mean_ent > 0.7:
                diffuse_heads.append((h, mean_ent, std_ent))
            else:
                mid_heads.append((h, mean_ent, std_ent))

        # 稳定性: std/mean越小越稳定
        all_stabilities = []
        for h, entropies in head_data.items():
            if len(entropies) >= 5 and np.mean(entropies) > 0.01:
                cv = np.std(entropies) / np.mean(entropies)  # coefficient of variation
                all_stabilities.append(cv)

        mean_cv = np.mean(all_stabilities) if all_stabilities else 0
        head_specialization[li] = {
            'n_focused': len(focused_heads),
            'n_diffuse': len(diffuse_heads),
            'n_mid': len(mid_heads),
            'mean_cv': float(mean_cv),
        }

        print(f"  L{li:>2d}: focused={len(focused_heads)}, diffuse={len(diffuse_heads)}, "
              f"mid={len(mid_heads)}, stability(CV)={mean_cv:.3f}")

    # 跨层routing graph: 哪些层有稳定的聚焦head
    print("\n  --- Routing Graph Summary ---")
    total_focused = sum(h['n_focused'] for h in head_specialization.values())
    total_diffuse = sum(h['n_diffuse'] for h in head_specialization.values())
    total_mid = sum(h['n_mid'] for h in head_specialization.values())
    print(f"  Total: focused={total_focused}, diffuse={total_diffuse}, mid={total_mid}")

    mean_stability = np.mean([h['mean_cv'] for h in head_specialization.values()]) if head_specialization else 0
    print(f"  Mean stability (CV, lower=more stable): {mean_stability:.3f}")

    return {
        'head_specialization': head_specialization,
        'supports_attentions': supports_attentions,
        'mean_stability_cv': float(mean_stability),
    }


# ============================================================
# Exp D: Trajectory Stability — 微扰后的轨迹分叉
# ============================================================
def expD_trajectory_stability(model, tokenizer, model_name):
    """
    核心问题: 微扰hidden state后, 生成的轨迹如何分叉?
    
    方法:
    1. 对一个prompt, 在不同层注入微扰
    2. 继续生成后续token
    3. 测量: 生成的轨迹分叉点(boundary crossing time)
    4. 分叉点越早 → 该层对因果影响越大
    
    关键指标:
    - boundary crossing time: 第几个token开始分叉?
    - entropy bifurcation: 分叉后的entropy变化
    - margin at crossing: 分叉点的logit margin
    """
    print("\n" + "="*60)
    print("Exp D: Trajectory Stability (Rollout Divergence)")
    print("="*60)

    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    d_model = info.d_model

    # 采样注入层
    inject_layers = [0, 1, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]
    inject_layers = sorted(set([l for l in inject_layers if l < n_layers]))

    n_sents = min(15, N_SENTENCES)  # 生成较慢, 减少句子数
    n_generate_tokens = 20
    n_perturb = 10  # 每层扰动次数

    results = []

    for sent_idx in range(n_sents):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # 基线生成
        with torch.no_grad():
            gen_kwargs = dict(max_new_tokens=n_generate_tokens, do_sample=False,
                              repetition_penalty=1.2)
            baseline_gen = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        baseline_text = tokenizer.decode(baseline_gen[0], skip_special_tokens=True)
        baseline_ids = baseline_gen[0, input_ids.shape[1]:].cpu().numpy()

        for inject_li in inject_layers:
            for p_idx in range(n_perturb):
                np.random.seed(sent_idx * 100 + inject_li * 10 + p_idx)
                delta = np.random.randn(d_model)
                delta = delta / np.linalg.norm(delta) * EPSILON

                # 在inject_li层注入扰动, 然后继续生成
                layers_list = get_layers(model)
                delta_tensor = torch.tensor(delta, dtype=torch.float32)
                last_pos = input_ids.shape[1] - 1

                def make_hook(pos, delta_t):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            out = output[0].clone()
                            out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                            return (out,) + output[1:]
                        else:
                            out = output.clone()
                            out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                            return out
                    return hook

                hooks = [layers_list[inject_li].register_forward_hook(make_hook(last_pos, delta_tensor))]

                try:
                    with torch.no_grad():
                        perturbed_gen = model.generate(input_ids, attention_mask=attention_mask,
                                                       **gen_kwargs)
                    perturbed_ids = perturbed_gen[0, input_ids.shape[1]:].cpu().numpy()

                    # 计算分叉点: 第几个token开始不同
                    crossing_time = n_generate_tokens  # 默认: 没分叉
                    for t in range(min(len(baseline_ids), len(perturbed_ids))):
                        if baseline_ids[t] != perturbed_ids[t]:
                            crossing_time = t
                            break

                    # 计算token匹配率
                    min_len = min(len(baseline_ids), len(perturbed_ids))
                    match_rate = np.mean(baseline_ids[:min_len] == perturbed_ids[:min_len])

                    results.append({
                        'sent_idx': sent_idx,
                        'inject_layer': inject_li,
                        'perturb_idx': p_idx,
                        'crossing_time': int(crossing_time),
                        'match_rate': float(match_rate),
                    })

                except:
                    pass

                for h in hooks:
                    h.remove()

        if sent_idx % 5 == 0:
            print(f"  Progress: {sent_idx}/{n_sents}")

    # 汇总分析
    print("\n  === Trajectory Stability by Inject Layer ===")
    print(f"  {'Layer':>6} | {'Mean Crossing':>14} | {'Mean Match Rate':>15} | {'N'}")
    print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*15}-+-{'-'*5}")

    layer_summary = {}
    for li in inject_layers:
        subset = [r for r in results if r['inject_layer'] == li]
        if subset:
            mean_ct = np.mean([r['crossing_time'] for r in subset])
            mean_mr = np.mean([r['match_rate'] for r in subset])
            layer_summary[li] = {
                'mean_crossing_time': float(mean_ct),
                'mean_match_rate': float(mean_mr),
                'n_samples': len(subset),
            }
            print(f"  L{li:>4d} | {mean_ct:>14.2f} | {mean_mr:>15.3f} | {len(subset)}")

    # 关键分析: 早期层 vs 晚期层的轨迹影响
    if len(layer_summary) > 2:
        early_layers = [li for li in layer_summary if li < n_layers // 3]
        late_layers = [li for li in layer_summary if li >= 2 * n_layers // 3]

        if early_layers and late_layers:
            early_ct = np.mean([layer_summary[li]['mean_crossing_time'] for li in early_layers])
            late_ct = np.mean([layer_summary[li]['mean_crossing_time'] for li in late_layers])
            early_mr = np.mean([layer_summary[li]['mean_match_rate'] for li in early_layers])
            late_mr = np.mean([layer_summary[li]['mean_match_rate'] for li in late_layers])

            print(f"\n  Early layers (<{n_layers//3}): crossing_time={early_ct:.2f}, match_rate={early_mr:.3f}")
            print(f"  Late layers (>={2*n_layers//3}): crossing_time={late_ct:.2f}, match_rate={late_mr:.3f}")
            print(f"  Early crossing is {'earlier' if early_ct < late_ct else 'later'} than late → "
                  f"{'early layers have MORE causal impact' if early_ct < late_ct else 'late layers have MORE causal impact'}")

    return layer_summary


# ============================================================
# 主函数
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"Phase 153: Boundary-Driven Statistical Dynamics")
    print(f"Model: {model_name}")
    print(f"Time: {timestamp}")
    print(f"N_SENTENCES: {N_SENTENCES}")

    # 加载模型
    t0 = time.time()
    model, tokenizer, device = load_model_custom(model_name)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}")
    print(f"Load time: {time.time()-t0:.1f}s")

    # 运行实验
    print("\n" + "#"*60)
    print("# Running 4 Experiments in Priority Order")
    print("#"*60)

    expA_results = expA_logit_boundary_propagation(model, tokenizer, model_name)
    expB_results = expB_cca_between_layers(model, tokenizer, model_name)
    expC_results = expC_attention_routing(model, tokenizer, model_name)
    expD_results = expD_trajectory_stability(model, tokenizer, model_name)

    # 保存结果
    all_results = {
        "phase": "153_boundary_dynamics",
        "model": model_name,
        "timestamp": timestamp,
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
        "expA_logit_boundary_propagation": expA_results,
        "expB_cca": expB_results,
        "expC_attention_routing": expC_results,
        "expD_trajectory_stability": expD_results,
    }

    result_file = OUTPUT_DIR / f"phase153_{model_name}_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)

    print(f"\nResults saved to: {result_file}")

    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("Model released.")


if __name__ == "__main__":
    main()
