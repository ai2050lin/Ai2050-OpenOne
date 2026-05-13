"""
Phase 153 修正版: 边界驱动统计动力学
=====================================

修正:
1. CCA实现修正 — 使用正确的sklearn CCA, 而非SVD白化法
2. Exp D修正 — 不使用generate(太慢且hook不兼容), 改为直接测量logit层面的边界穿越
3. 增大数据量: 100个句子(重要实验)

核心实验:
- Exp 1: Logit Boundary Propagation — margin变化 vs 注入层深度
- Exp 2: CCA (sklearn) — 真正的二阶结构保留
- Exp 3: Attention Routing Topology — 注意力图稳定性
- Exp 4: Boundary Crossing — 边界穿越的细粒度分析(替代trajectory)

用法:
  python tests/glm5_temp/phase153_fixed.py qwen3
  python tests/glm5_temp/phase153_fixed.py deepseek7b
  python tests/glm5_temp/phase153_fixed.py glm4
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
    "The restaurant was famous for its delicious",
    "The engineer designed a bridge that could",
    "The bird flew across the sky and landed",
    "The teacher asked the students to write",
    "The museum exhibit showed how people lived",
    "She found the key hidden under the",
    "The dog ran across the field chasing",
    "The library had a collection of rare",
    "The weather forecast predicted that tomorrow",
    "The baby laughed when her mother made",
    "The computer program was designed to help",
    "The garden was full of colorful flowers",
    "He finished reading the newspaper and then",
    "The movie told the story of a young",
    "The football team won the championship by",
    "The cake in the kitchen smelled absolutely",
    "The boat sailed across the lake toward",
    "She studied hard for the exam because",
    "The clock on the wall showed that it was",
    "The dictionary definition of the word was",
    "The factory produced thousands of items every",
    "The island was surrounded by clear blue",
    "The musician played a beautiful melody on",
    "The newspaper reported that the election",
    "The path through the woods led to a",
    "The scientist published a paper about the",
    "The tower could be seen from miles away",
    "The university offered courses in many different",
    "The village was located near the river and",
    "The wind blew through the trees making a",
    "The writer spent months working on her new",
    "Their house was the largest on the street",
    "The airplane flew above the clouds and",
    "The ice cream shop had more than fifty",
    "The poem described the beauty of the autumn",
    "The security guard noticed that the door was",
    "The ship left the harbor early in the",
    "The smartphone had a feature that allowed",
    "The sunset painted the sky in shades of",
    "The teenager wanted to learn how to play",
    "The umbrella was left behind at the restaurant",
    "The volcano erupted after years of being",
    "The whale jumped out of the water and",
    "The zoo had animals from all around the",
    "The astronaut floated in zero gravity while",
]

N_SENTENCES = 95
EPSILON = 1.0


def load_model_custom(model_name: str):
    """加载模型"""
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
    if n_layers <= 12:
        return list(range(0, n_layers + 1))
    early = [0, 1, 2, 3, 4]
    mid_step = max(1, (n_layers - 10) // 5)
    mid = list(range(5, n_layers - 4, mid_step))
    late = [n_layers - 4, n_layers - 3, n_layers - 2, n_layers - 1, n_layers]
    return sorted(set(early + mid + late))


# ============================================================
# Exp 1: Logit Boundary Propagation (加大数据量)
# ============================================================
def exp1_boundary_propagation(model, tokenizer, model_name):
    """
    核心问题: 扰动注入不同层后, logit margin如何变化?
    
    关键变量:
    - inject_layer: 在哪层注入扰动
    - margin_change: |baseline_margin - perturbed_margin|
    - switching_rate: top-1 token被替换的比例
    
    如果margin_change随注入层单调递增 → 晚层对决策边界影响更大
    如果margin_change在所有层相同 → 扰动等价传播
    """
    print("\n" + "="*60)
    print("Exp 1: Logit Boundary Propagation (100 sentences)")
    print("="*60)

    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    d_model = info.d_model

    # 采样注入层: 早期密, 晚期密
    inject_layers = sorted(set(
        [0, 1, 2, 3, 4] +
        list(range(6, n_layers - 4, max(1, n_layers // 8))) +
        [n_layers - 4, n_layers - 3, n_layers - 2, n_layers - 1]
    ))[:16]

    n_sents = min(50, N_SENTENCES)
    n_perturb = 50

    results_per_layer = {}

    for inject_li in inject_layers:
        margin_changes = []
        baseline_margins = []
        top1_stay_count = 0
        total_perturb = 0

        for sent_idx in range(n_sents):
            prompt = TEST_PROMPTS[sent_idx]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            last_pos = input_ids.shape[1] - 1

            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
            clean_logits = out_clean.logits[0, -1, :].float().cpu().numpy()
            sorted_ids = np.argsort(-clean_logits)
            top1_id = sorted_ids[0]
            top2_id = sorted_ids[1]
            baseline_margin = clean_logits[top1_id] - clean_logits[top2_id]
            baseline_margins.append(baseline_margin)

            for p_idx in range(n_perturb):
                np.random.seed(inject_li * 10000 + sent_idx * 100 + p_idx)
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
                    margin_changes.append(abs(baseline_margin - perturbed_margin_val))

                    if perturbed_top1 == top1_id:
                        top1_stay_count += 1
                    total_perturb += 1
                except:
                    pass

                for h in hooks:
                    h.remove()

        if margin_changes:
            stay_rate = top1_stay_count / total_perturb if total_perturb > 0 else 0
            results_per_layer[inject_li] = {
                'mean_margin_change': float(np.mean(margin_changes)),
                'std_margin_change': float(np.std(margin_changes)),
                'mean_baseline_margin': float(np.mean(baseline_margins)),
                'top1_stay_rate': float(stay_rate),
                'n_samples': len(margin_changes),
            }
            r = results_per_layer[inject_li]
            print(f"  L{inject_li:>2d}: margin_chg={r['mean_margin_change']:.4f}±{r['std_margin_change']:.4f}, "
                  f"stay={r['top1_stay_rate']:.3f}, base_m={r['mean_baseline_margin']:.3f}")

    # 关键分析: margin_change随层深度变化
    if len(results_per_layer) > 2:
        early_layers = [l for l in results_per_layer if l < n_layers // 3]
        late_layers = [l for l in results_per_layer if l >= 2 * n_layers // 3]

        if early_layers and late_layers:
            early_mc = np.mean([results_per_layer[l]['mean_margin_change'] for l in early_layers])
            late_mc = np.mean([results_per_layer[l]['mean_margin_change'] for l in late_layers])
            early_sr = np.mean([results_per_layer[l]['top1_stay_rate'] for l in early_layers])
            late_sr = np.mean([results_per_layer[l]['top1_stay_rate'] for l in late_layers])
            print(f"\n  Early(<{n_layers//3}): margin_change={early_mc:.4f}, stay_rate={early_sr:.3f}")
            print(f"  Late(>={2*n_layers//3}): margin_change={late_mc:.4f}, stay_rate={late_sr:.3f}")
            print(f"  Late/Early margin_change ratio: {late_mc/early_mc:.2f}x" if early_mc > 0 else "  N/A")

    return results_per_layer


# ============================================================
# Exp 2: CCA using sklearn (修正版)
# ============================================================
def exp2_cca_sklearn(model, tokenizer, model_name):
    """
    使用sklearn的CCA实现 — 正确的高维CCA
    
    关键修正: 之前SVD白化法在N<d时完全失效
    sklearn的CCA使用SVD分解, 但内部处理了N<d的情况
    
    核心对比: CCA_1 vs R²
    - CCA_1: 旋转不变的最大相关性
    - R²: 线性可预测性
    - 如果CCA_1 >> R² → 信息保留但在旋转坐标系中
    - 如果CCA_1 ≈ R² → 信息真正线性可预测
    """
    print("\n" + "="*60)
    print("Exp 2: CCA (sklearn) — Rotation-Invariant Correlation")
    print("="*60)

    from sklearn.cross_decomposition import CCA as SklearnCCA

    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    d_model = info.d_model

    sample_layers = get_sample_layers(n_layers)

    # 收集100个句子的hidden states — CCA需要大样本
    n_sents = min(100, N_SENTENCES)
    all_hs = {}
    for li in sample_layers:
        all_hs[li] = []

    print(f"  Collecting {n_sents} hidden states...")
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

        if sent_idx % 20 == 0:
            print(f"  Progress: {sent_idx}/{n_sents}")

    # 参考层: hs[1]
    ref_layer = 1
    X_ref = np.array(all_hs[ref_layer])  # [N, d_model]

    # 对每层计算CCA vs R²
    n_cca_comp = min(5, n_sents - 1)  # CCA分量数不能超过N-1

    cca_results = {}

    # 先对参考层做PCA降维(因为d_model >> N, sklearn CCA在d>>N时需要降维)
    n_pca = min(n_sents - 1, d_model)  # 最多N-1个主成分

    # PCA for X_ref
    X_ref_c = X_ref - X_ref.mean(axis=0)
    U_ref, s_ref, Vt_ref = np.linalg.svd(X_ref_c, full_matrices=False)
    # 保留95%方差
    cumvar = np.cumsum(s_ref**2) / np.sum(s_ref**2)
    n_keep_ref = max(n_cca_comp + 1, np.searchsorted(cumvar, 0.95) + 1)
    n_keep_ref = min(n_keep_ref, n_sents - 1)
    X_ref_pca = U_ref[:, :n_keep_ref] * s_ref[:n_keep_ref]  # [N, n_keep_ref]

    print(f"\n  PCA dimensions: ref={n_keep_ref} (from {d_model}, {cumvar[min(n_keep_ref-1, len(cumvar)-1)]:.3f} variance)")

    print(f"\n  {'Layer':>6} | {'CCA_1':>8} | {'CCA_2':>8} | {'CCA_3':>8} | {'CCA_mean':>8} | {'R²':>8} | {'CCA/R²':>7}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")

    for li in sample_layers:
        X_li = np.array(all_hs[li])  # [N, d_model]
        X_li_c = X_li - X_li.mean(axis=0)

        # PCA for X_li
        U_li, s_li, Vt_li = np.linalg.svd(X_li_c, full_matrices=False)
        n_keep_li = max(n_cca_comp + 1, np.searchsorted(np.cumsum(s_li**2) / np.sum(s_li**2), 0.95) + 1)
        n_keep_li = min(n_keep_li, n_sents - 1)
        X_li_pca = U_li[:, :n_keep_li] * s_li[:n_keep_li]  # [N, n_keep_li]

        # CCA
        try:
            cca = SklearnCCA(n_components=n_cca_comp)
            cca.fit(X_ref_pca, X_li_pca)
            X_ref_c, X_li_c = cca.transform(X_ref_pca, X_li_pca)
            # CCA correlations = corr between transformed components
            cca_corrs = []
            for i in range(n_cca_comp):
                c = np.corrcoef(X_ref_c[:, i], X_li_c[:, i])[0, 1]
                cca_corrs.append(abs(c))
        except:
            cca_corrs = [0.0] * n_cca_comp

        # R²
        try:
            r2_list = []
            n_train = n_sents * 2 // 3
            for comp_j in range(min(10, X_li_pca.shape[1])):
                y_train = X_li_pca[:n_train, comp_j]
                y_test = X_li_pca[n_train:, comp_j]
                X_train = X_ref_pca[:n_train]
                X_test = X_ref_pca[n_train:]
                try:
                    W, _, _, _ = np.linalg.lstsq(
                        np.column_stack([X_train, np.ones(n_train)]), y_train, rcond=None)
                    y_pred = np.column_stack([X_test, np.ones(X_test.shape[0])]) @ W
                    ss_res = np.sum((y_test - y_pred) ** 2)
                    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                    r2_list.append(max(0, 1 - ss_res / max(ss_tot, 1e-10)))
                except:
                    r2_list.append(0)
            r2_val = float(np.mean(r2_list))
        except:
            r2_val = 0.0

        cca1 = cca_corrs[0] if len(cca_corrs) > 0 else 0
        cca2 = cca_corrs[1] if len(cca_corrs) > 1 else 0
        cca3 = cca_corrs[2] if len(cca_corrs) > 2 else 0
        cca_mean = float(np.mean(cca_corrs))
        ratio = cca1 / r2_val if r2_val > 0.001 else float('inf')

        cca_results[li] = {
            'cca_correlations': cca_corrs,
            'cca_mean': cca_mean,
            'r2': r2_val,
            'cca_r2_ratio': float(ratio) if ratio != float('inf') else -1,
        }

        ratio_str = f"{ratio:.2f}x" if ratio != float('inf') else "INF"
        print(f"  hs[{li:>3d}] | {cca1:>8.4f} | {cca2:>8.4f} | {cca3:>8.4f} | {cca_mean:>8.4f} | {r2_val:>8.4f} | {ratio_str:>7}")

    # 核心分析
    print(f"\n  === CRITICAL ANALYSIS ===")
    high_ratio_layers = [li for li, r in cca_results.items()
                         if r['cca_r2_ratio'] > 2 and r['cca_r2_ratio'] != -1]
    if high_ratio_layers:
        print(f"  CCA >> R² at layers: {high_ratio_layers}")
        print(f"  → Information preserved in ROTATED coordinates!")
    else:
        print(f"  CCA ≈ R² across all layers")
        print(f"  → Information is directly linear-predictable, no rotation")

    return cca_results


# ============================================================
# Exp 3: Attention Routing Topology
# ============================================================
def exp3_attention_routing(model, tokenizer, model_name):
    """
    注意力路由拓扑 — head specialization稳定性
    """
    print("\n" + "="*60)
    print("Exp 3: Attention Routing Topology")
    print("="*60)

    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers

    # 检查是否支持output_attentions
    supports_attentions = True
    try:
        prompt = TEST_PROMPTS[0]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            test_out = model(input_ids=input_ids, attention_mask=attention_mask,
                             output_attentions=True)
        if test_out.attentions is None:
            supports_attentions = False
    except:
        supports_attentions = False

    if not supports_attentions:
        print("  output_attentions not supported — using QKV hook method")

    n_sents = min(40, N_SENTENCES)
    attn_sample_layers = sorted(set(
        [0, 1, 2, 3] + list(range(4, n_layers, max(1, n_layers // 6))) + [n_layers - 2, n_layers - 1]
    ))[:10]

    head_entropy_profiles = {li: {} for li in attn_sample_layers}

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

            for li in attn_sample_layers:
                if li < len(out.attentions) and out.attentions[li] is not None:
                    attn_weights = out.attentions[li]  # [1, n_heads, seq, seq]
                    last_token_attn = attn_weights[0, :, last_pos, :last_pos+1].float().cpu().numpy()
                    n_heads = last_token_attn.shape[0]

                    for h in range(n_heads):
                        if h not in head_entropy_profiles[li]:
                            head_entropy_profiles[li][h] = []
                        attn_dist = last_token_attn[h]
                        attn_dist = attn_dist / max(attn_dist.sum(), 1e-10)
                        entropy = -np.sum(attn_dist * np.log(attn_dist + 1e-10))
                        max_entropy = np.log(len(attn_dist))
                        head_entropy_profiles[li][h].append(entropy / max_entropy if max_entropy > 0 else 0)
        else:
            # QKV hook方法
            layers_list = get_layers(model)
            for li in attn_sample_layers:
                layer = layers_list[li]
                sa = layer.self_attn

                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
                    h_input = out.hidden_states[li][0]  # [seq, d_model]

                    W_q = sa.q_proj.weight.detach().float()
                    W_k = sa.k_proj.weight.detach().float()

                    Q = (h_input @ W_q.T).cpu().numpy()
                    K = (h_input @ W_k.T).cpu().numpy()

                    n_heads = min(8, Q.shape[1] // max(1, Q.shape[1] // 32))
                    d_head = Q.shape[1] // n_heads if n_heads > 0 else Q.shape[1]

                    for h in range(n_heads):
                        if h not in head_entropy_profiles[li]:
                            head_entropy_profiles[li][h] = []

                        Q_h = Q[last_pos, h*d_head:(h+1)*d_head]
                        K_h = K[:last_pos+1, h*d_head:(h+1)*d_head]

                        scores = K_h @ Q_h / max(np.sqrt(d_head), 1.0)
                        exp_scores = np.exp(scores - np.max(scores))
                        attn_dist = exp_scores / exp_scores.sum()

                        entropy = -np.sum(attn_dist * np.log(attn_dist + 1e-10))
                        max_entropy = np.log(len(attn_dist)) if len(attn_dist) > 1 else 1
                        head_entropy_profiles[li][h].append(entropy / max_entropy if max_entropy > 0 else 0)

    # 分析head specialization
    print("\n  --- Head Specialization ---")
    head_specialization = {}

    for li in sorted(head_entropy_profiles.keys()):
        head_data = head_entropy_profiles[li]
        if not head_data:
            continue

        focused = sum(1 for h, e in head_data.items() if len(e) >= 5 and np.mean(e) < 0.3)
        diffuse = sum(1 for h, e in head_data.items() if len(e) >= 5 and np.mean(e) > 0.7)
        mid = sum(1 for h, e in head_data.items() if len(e) >= 5 and 0.3 <= np.mean(e) <= 0.7)

        stabilities = [np.std(e) / max(np.mean(e), 0.01) for e in head_data.values() if len(e) >= 5 and np.mean(e) > 0.01]
        mean_cv = float(np.mean(stabilities)) if stabilities else 0

        head_specialization[li] = {
            'n_focused': focused, 'n_diffuse': diffuse, 'n_mid': mid,
            'mean_cv': mean_cv, 'n_heads': len(head_data),
        }
        print(f"  L{li:>2d}: focused={focused}, diffuse={diffuse}, mid={mid}, "
              f"CV={mean_cv:.3f}, n_heads={len(head_data)}")

    return {
        'head_specialization': head_specialization,
        'supports_attentions': supports_attentions,
    }


# ============================================================
# Exp 4: Boundary Crossing — 细粒度边界穿越分析
# ============================================================
def exp4_boundary_crossing(model, tokenizer, model_name):
    """
    核心问题: logit margin和边界穿越的精确关系
    
    方法:
    1. 对每个prompt, 精确扫描eps找到switching point
    2. 分析: switching_eps vs baseline_margin 的精确关系
    3. 在不同层注入扰动, 看"决策层"在哪
    
    这比Phase 152的Exp 2更精细:
    - 更大样本量(100句子)
    - 更细的eps扫描
    - 不同注入层的对比
    """
    print("\n" + "="*60)
    print("Exp 4: Boundary Crossing — Fine-Grained Analysis")
    print("="*60)

    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    d_model = info.d_model

    n_sents = min(80, N_SENTENCES)

    # 精细eps扫描
    eps_scan = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    # 在三个关键层注入: L0, L_{n/2}, L_{n-2}
    inject_layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2]))[:5]

    results = {li: [] for li in inject_layers}

    for sent_idx in range(n_sents):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        last_pos = input_ids.shape[1] - 1

        # 基线
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
        clean_logits = out_clean.logits[0, -1, :].float().cpu().numpy()
        sorted_ids = np.argsort(-clean_logits)
        top1_id = sorted_ids[0]
        top2_id = sorted_ids[1]
        baseline_margin = clean_logits[top1_id] - clean_logits[top2_id]

        # 每个句子用一个固定随机方向(跨层一致)
        np.random.seed(sent_idx * 7 + 13)
        delta_dir = np.random.randn(d_model)
        delta_dir = delta_dir / np.linalg.norm(delta_dir)

        for inject_li in inject_layers:
            switching_eps = None

            for eps in eps_scan:
                delta_scaled = delta_dir * eps

                layers_list = get_layers(model)
                delta_tensor = torch.tensor(delta_scaled, dtype=torch.float32)

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
                    perturbed_top1 = int(np.argmax(out_p.logits[0, -1, :].float().cpu().numpy()))

                    if perturbed_top1 != top1_id and switching_eps is None:
                        switching_eps = eps
                except:
                    pass

                for h in hooks:
                    h.remove()

            results[inject_li].append({
                'sent_idx': sent_idx,
                'logit_margin': float(baseline_margin),
                'switching_eps': switching_eps,
            })

        if sent_idx % 20 == 0:
            print(f"  Progress: {sent_idx}/{n_sents}")

    # 分析
    print("\n  === Boundary Crossing by Inject Layer ===")
    layer_summary = {}

    for li in inject_layers:
        data = results[li]
        switching = [r for r in data if r['switching_eps'] is not None]
        no_switch = [r for r in data if r['switching_eps'] is None]

        switch_rate = len(switching) / len(data) if data else 0
        switch_eps_mean = float(np.mean([r['switching_eps'] for r in switching])) if switching else None

        # 按margin分组
        narrow = [r for r in data if r['logit_margin'] < 1]
        medium = [r for r in data if 1 <= r['logit_margin'] < 3]
        wide = [r for r in data if r['logit_margin'] >= 3]

        narrow_sr = sum(1 for r in narrow if r['switching_eps'] is not None) / len(narrow) if narrow else 0
        medium_sr = sum(1 for r in medium if r['switching_eps'] is not None) / len(medium) if medium else 0
        wide_sr = sum(1 for r in wide if r['switching_eps'] is not None) / len(wide) if wide else 0

        # margin vs switching_eps correlation
        if switching:
            margins_sw = [r['logit_margin'] for r in switching]
            eps_sw = [r['switching_eps'] for r in switching]
            if len(margins_sw) > 3:
                corr = np.corrcoef(margins_sw, eps_sw)[0, 1]
            else:
                corr = 0
        else:
            corr = 0

        layer_summary[li] = {
            'switch_rate': float(switch_rate),
            'switch_eps_mean': switch_eps_mean,
            'narrow_sr': float(narrow_sr),
            'medium_sr': float(medium_sr),
            'wide_sr': float(wide_sr),
            'margin_eps_corr': float(corr),
            'n_switching': len(switching),
            'n_total': len(data),
        }

        eps_str = f"{switch_eps_mean:.3f}" if switch_eps_mean else "N/A"
        print(f"  L{li:>2d}: switch_rate={switch_rate:.1%}, eps={eps_str}, "
              f"narrow_sr={narrow_sr:.1%}, medium_sr={medium_sr:.1%}, wide_sr={wide_sr:.1%}, "
              f"margin-eps_corr={corr:.3f}")

    # 核心分析: margin-eps correlation
    print(f"\n  === CRITICAL: Margin-Eps Correlation ===")
    for li, s in layer_summary.items():
        if s['margin_eps_corr'] > 0.5:
            print(f"  L{li}: STRONG positive correlation ({s['margin_eps_corr']:.3f}) → larger margin needs larger eps to switch")
        elif s['margin_eps_corr'] < -0.5:
            print(f"  L{li}: STRONG negative correlation ({s['margin_eps_corr']:.3f}) → UNEXPECTED!")
        else:
            print(f"  L{li}: weak/no correlation ({s['margin_eps_corr']:.3f})")

    return {'layer_summary': layer_summary, 'raw_data': results}


# ============================================================
# 主函数
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"Phase 153 (Fixed): Boundary-Driven Statistical Dynamics")
    print(f"Model: {model_name}, Time: {timestamp}")

    t0 = time.time()
    model, tokenizer, device = load_model_custom(model_name)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}")
    print(f"Load time: {time.time()-t0:.1f}s")

    exp1 = exp1_boundary_propagation(model, tokenizer, model_name)
    exp2 = exp2_cca_sklearn(model, tokenizer, model_name)
    exp3 = exp3_attention_routing(model, tokenizer, model_name)
    exp4 = exp4_boundary_crossing(model, tokenizer, model_name)

    all_results = {
        "phase": "153_fixed",
        "model": model_name,
        "timestamp": timestamp,
        "model_info": {"class": info.model_class, "n_layers": info.n_layers, "d_model": info.d_model},
        "exp1_boundary_propagation": exp1,
        "exp2_cca": exp2,
        "exp3_attention_routing": exp3,
        "exp4_boundary_crossing": exp4,
    }

    result_file = OUTPUT_DIR / f"phase153_{model_name}_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)

    print(f"\nResults saved to: {result_file}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("Model released.")


if __name__ == "__main__":
    main()
