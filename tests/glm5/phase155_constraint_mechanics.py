"""
Phase 155: 约束力学 — 决定性实验: W_U对齐 vs 约束传播
=====================================================

核心问题: Phase 154发现的tau构建律, 到底是:
  A. 真正的约束传播(独立于W_U的统计结构)
  B. W_U对齐伪影(hidden state逐渐对齐W_U行空间)

这是决定理论生死的实验!

四大实验:
  Exp 1: W_U对齐决定性测试 (4个控制条件)
    - 1a: 真实W_U (baseline, 复现Phase 154)
    - 1b: 随机W_rand (5个随机矩阵平均) — 如果tau_rand也构建→泛化性质
    - 1c: 正交旋转W_U (Q@W_U) — 如果tau仍构建→约束独立于W_U坐标系
    - 1d: 跨句控制 (sentence_i的h_ℓ vs sentence_j的h_final) — 破坏句内结构

  Exp 2: 中间层Logit动力学 — 约束搜索过程
    - Top-5 token跨层追踪
    - 翻转率(flipping rate), 吸引子稳定性
    - DS7B负tau的因果分析

  Exp 3: 约束违反层级 — 哪个约束先崩?
    - 语法 vs 语义 vs 逻辑
    - 错误预测时: 哪些约束被违反?

  Exp 4: 能量景观 — 约束场的统计物理
    - 正确token的"能量"(-log prob)跨层变化
    - 决策边界的锐化过程

用法:
  python tests/glm5/phase155_constraint_mechanics.py qwen3
  python tests/glm5/phase155_constraint_mechanics.py deepseek7b
  python tests/glm5/phase155_constraint_mechanics.py glm4
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
from scipy.stats import kendalltau, spearmanr
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS)
from collections import defaultdict

OUTPUT_DIR = Path("tests/glm5_temp")

# ============================================================
# 大规模测试语料 — 300句(关键测试加大数据量!)
# ============================================================
GRAMMAR_PROMPTS = [
    "The cat sits on the", "The cats sit on the",
    "The dog runs toward the", "The dogs run toward the",
    "The bird flies over the", "The birds fly over the",
    "The child plays in the", "The children play in the",
    "The woman walks to the", "The women walk to the",
    "The man stands by the", "The men stand by the",
    "The horse gallops across the", "The horses gallop across the",
    "The fish swims in the", "The fish swim in the",
    "The student reads the", "The students read the",
    "The teacher explains the", "The teachers explain the",
    "The scientist discovers the", "The scientists discover the",
    "The writer publishes the", "The writers publish the",
    "The artist paints the", "The artists paint the",
    "The doctor examines the", "The doctors examine the",
    "The engineer builds the", "The engineers build the",
    "The musician plays the", "The musicians play the",
    "The driver stops at the", "The drivers stop at the",
    "The student did not finish the", "No one believed that the",
    "She never mentioned the", "They hardly noticed the",
    "He barely touched the", "She seldom visited the",
    "Yesterday she went to the", "Tomorrow they will visit the",
    "Right now he is reading the", "Last week the team completed the",
    "Next year the company will launch the", "Previously the group had discussed the",
    "If it rains tomorrow then the", "If she studies hard then the",
    "Unless the weather improves the", "Provided that the results are positive the",
    "When the sun rises the", "While the storm raged the",
]

ATTRIBUTE_PROMPTS = [
    "The red apple was placed on the", "The blue car drove past the",
    "The tall building stood near the", "The small bird sat on the",
    "The old man walked to the", "The young woman entered the",
    "The hot coffee spilled on the", "The cold wind blew through the",
    "The bright light shone on the", "The dark room contained a",
    "The heavy box was moved to the", "The soft pillow lay on the",
    "The sharp knife cut through the", "The smooth surface reflected the",
    "The sweet smell filled the", "The bitter taste lingered in the",
    "The loud noise startled the", "The quiet room had a",
    "The fast runner crossed the", "The slow turtle crawled across the",
    "The green forest surrounded the", "The white snow covered the",
    "The black cat jumped over the", "The golden ring was found in the",
    "The wooden door opened into the", "The stone wall surrounded the",
]

LOGIC_PROMPTS = [
    "Because it was raining the", "Since the evidence was clear the",
    "Although the task was difficult the", "While the first option was safer the",
    "Therefore the committee decided to", "Consequently the researchers concluded that",
    "However the alternative approach would", "Moreover the additional data showed that",
    "Nevertheless the team continued to", "Thus the final result indicated that",
    "Given that the budget was limited the", "Assuming the hypothesis is correct the",
    "In spite of the obstacles the", "On the contrary the evidence suggests",
    "As a result of the investigation the", "Prior to the announcement the",
    "In addition to the main finding the", "Despite the lack of support the",
    "The reason for this is that the", "One implication of this is that",
]

COREF_PROMPTS = [
    "Mary gave Jane the book because she", "The manager told the employee that he",
    "After Anna met Lisa she decided to", "When the teacher asked the student he",
    "Although John helped Mary he", "Before Sarah left she told",
    "Since David arrived early he", "While Rachel was cooking she",
    "The king told the queen that he", "The mother asked her daughter if she",
]

GENERAL_PROMPTS_1 = [
    "The scientist discovered that the", "In the morning she decided to",
    "The book on the table was about", "After the rain stopped the children",
    "The most important thing about science is", "When the sun sets over the ocean",
    "She walked into the room and saw", "The professor explained that the theory",
    "Despite the challenges the team managed", "The ancient city was known for its",
    "He realized that the answer was", "The relationship between language and thought",
    "Every morning she would read the", "The experiment showed that the results",
    "Music has the power to change how", "The government announced that the new policy",
    "In the future artificial intelligence will", "The philosopher argued that consciousness is",
    "After years of research they found that", "The key difference between the two approaches is",
]

GENERAL_PROMPTS_2 = [
    "The cat sat on the windowsill and watched", "Through the telescope they observed a new",
    "The river flowed gently through the valley", "She opened the letter and read the",
    "The painting on the wall depicted a", "During the concert the audience was",
    "The invention changed the way people", "He wrote a letter to his friend about",
    "The students in the classroom were learning", "The old building at the corner had",
    "The doctor told him that he needed", "A sudden noise from outside made her",
    "The forest was filled with ancient trees", "She picked up the phone and called",
    "The road to the village was long and", "They stood at the edge of the cliff and",
    "The novel she was reading described a", "At the conference the speaker presented",
    "The children played in the garden while", "The old man smiled and said that",
    "The company decided to invest in new", "The train arrived at the station just as",
    "Through the window she could see the", "The puzzle was more difficult than they",
    "The artist carefully mixed the colors to", "The report concluded that the main cause",
    "She remembered the day when they first", "The mountain was covered with snow and",
    "The debate focused on whether the government", "He turned the key and opened the door to",
    "The library contained thousands of books about", "The garden was beautiful in the spring when",
    "The river had frozen during the cold winter", "She found the document hidden in the",
    "The earthquake caused significant damage to the", "The new technology allowed scientists to",
    "The conversation turned to the topic of", "The laboratory was equipped with the latest",
    "The museum exhibited artifacts from the ancient", "The research paper proposed a novel method for",
    "The journey took longer than expected because", "The neighborhood was quiet during the early",
    "The competition attracted participants from across the", "The discovery changed our understanding of the",
    "The algorithm was designed to optimize the", "The story began in a small village where",
    "The pattern repeated itself throughout the", "The evidence pointed to a conclusion that",
    "The solution required both creativity and", "The tradition dated back to the time when",
    "The analysis revealed an unexpected correlation between", "The proposal was rejected because the committee",
    "The forest path led to a clearing where", "The spacecraft transmitted data back to the",
    "The experiment was repeated three times to", "The manuscript was found in the archives of",
    "The community organized a festival to celebrate the", "The disease spread rapidly through the",
    "The painting was restored by a team of", "The interview revealed that the candidate had",
    "The bridge connected the two sides of the", "The festival attracted visitors from neighboring",
    "The performance received a standing ovation from the", "The expedition set out to explore the",
    "The committee voted unanimously to approve the", "The storm caused widespread flooding in the",
    "The teacher encouraged the students to think about", "The documentary explored the impact of the",
    "The orchestra played a symphony composed by", "The treaty was signed by representatives of the",
    "The flower bloomed in the garden beside the", "The recipe called for ingredients that were",
    "The foundation was established to support the", "The rescue team arrived just in time to",
    "The recipe had been passed down through", "The sculpture was carved from a single piece of",
    "The discovery was made by accident when the", "The population of the city had grown since the",
    "The novel explored themes of loss and", "The conference brought together experts from the",
    "The tradition was kept alive by the", "The project was completed ahead of schedule and the",
    "The analysis showed a significant improvement in the", "The museum collection included items from the",
    "The game was interrupted by a sudden", "The harvest was abundant this year because the",
    "The announcement came as a surprise to the", "The building was designed by a famous",
    "The theory was later confirmed by additional", "The letter was delivered to the wrong",
    "The crisis was resolved through diplomatic", "The construction was delayed due to the",
    "The machine was invented to solve the problem of", "The river provided water for the entire",
    "The ceremony was attended by dignitaries from", "The debate ended without a clear",
    "The forest fire was caused by a lightning", "The movie was based on a true story about the",
    "The system was designed to handle large volumes of", "The decision was made after careful consideration of the",
    "The market crashed following the announcement of the", "The dog barked loudly when the stranger approached the",
    "The recipe required precise measurements of the", "The school offered courses in a variety of",
]

ALL_PROMPTS = (GRAMMAR_PROMPTS + ATTRIBUTE_PROMPTS + LOGIC_PROMPTS + COREF_PROMPTS +
               GENERAL_PROMPTS_1 + GENERAL_PROMPTS_2)

# 缩短版用于8bit模型
SHORT_PROMPTS = ALL_PROMPTS[:150]


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


def generate_random_orthogonal(n, rng=None):
    """生成n×n随机正交矩阵"""
    if rng is None:
        rng = np.random.default_rng()
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    # 确保行列式为正
    d = np.diag(np.sign(np.diag(R)))
    Q = Q @ d
    return Q


# ============================================================
# Exp 1: W_U对齐决定性测试
# ============================================================
def exp1_wu_alignment_test(model, tokenizer, model_name, n_sents=300, n_random=5, n_rotations=3):
    """
    决定性实验: 区分"约束传播" vs "W_U对齐伪影"
    
    4个控制条件:
    1a. 真实W_U → tau应该构建 (baseline)
    1b. 随机W_rand → 如果tau也构建→泛化性质; 如果tau~0→W_U特定
    1c. 正交旋转Q@W_U → 如果tau仍构建→约束独立于W_U坐标系
    1d. 跨句控制 → 破坏句内结构
    
    关键判据:
    - tau_real >> tau_random 且 tau_real >> tau_cross → 约束传播存在
    - tau_real ≈ tau_random → tau只是泛化性质(W_U对齐伪影)
    - tau_rotated ≈ tau_real → 约束独立于W_U坐标系
    - tau_rotated << tau_real → 约束依赖于W_U坐标系
    """
    print("\n" + "="*60)
    print("Exp 1: W_U Alignment Decisive Test (决定理论生死!)")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers
    d_model = info.d_model
    W_U = get_W_U(model, model_name)  # [vocab, d]
    vocab_size = W_U.shape[0]
    
    prompts = ALL_PROMPTS[:n_sents] if model_name == "qwen3" else SHORT_PROMPTS[:min(150, n_sents)]
    actual_n = len(prompts)
    sample_layers = get_sample_layers(n_layers, 12)
    print(f"  n_sents={actual_n}, d_model={d_model}, vocab={vocab_size}, layers={sample_layers}")
    
    # ---- 收集所有hidden states (一次性收集, 复用于所有控制条件) ----
    print(f"  Collecting hidden states for {actual_n} sentences...")
    all_hs = {li: [] for li in sample_layers}
    
    for si, prompt in enumerate(prompts):
        if si % 50 == 0:
            print(f"    Sentence {si}/{actual_n}...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        for li in sample_layers:
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            all_hs[li].append(h)
    
    # 标准化: 每层收集为 [n, d] 矩阵
    hs_arrays = {li: np.array(all_hs[li]) for li in sample_layers}
    
    final_layer = n_layers  # 最后一层 (after final LN)
    
    # ---- 1a: 真实W_U baseline ----
    print("\n  --- 1a: Real W_U (baseline) ---")
    tau_real = {}
    cos_real = {}
    top1_real = {}
    
    final_logits_real = hs_arrays[final_layer] @ W_U.T  # [n, vocab]
    final_top1_real = np.argmax(final_logits_real, axis=1)  # [n]
    
    for li in sample_layers:
        logits_li = hs_arrays[li] @ W_U.T  # [n, vocab]
        
        # Kendall tau (top-100 tokens, 逐句平均)
        tau_vals = []
        for si in range(actual_n):
            top100_idx = np.argsort(-final_logits_real[si])[:100]
            ranks_final = np.argsort(np.argsort(-final_logits_real[si, top100_idx]))
            ranks_li = np.argsort(np.argsort(-logits_li[si, top100_idx]))
            t, _ = kendalltau(ranks_final, ranks_li)
            tau_vals.append(float(t) if not np.isnan(t) else 0)
        
        # Top-1 match
        li_top1 = np.argmax(logits_li, axis=1)
        top1_match = float(np.mean(li_top1 == final_top1_real))
        
        # Cos similarity (delta)
        cos_vals = []
        for si in range(actual_n):
            d0 = hs_arrays[0][si] - np.mean(hs_arrays[0][si])
            dli = hs_arrays[li][si] - np.mean(hs_arrays[li][si])
            c = np.dot(d0, dli) / (max(np.linalg.norm(d0), 1e-10) * max(np.linalg.norm(dli), 1e-10))
            cos_vals.append(float(c))
        
        tau_real[li] = float(np.mean(tau_vals))
        cos_real[li] = float(np.mean(cos_vals))
        top1_real[li] = top1_match
        print(f"    L{li:>3d}: tau={tau_real[li]:.4f}, top1={top1_real[li]:.4f}, cos={cos_real[li]:.4f}")
    
    # ---- 1b: 随机W_rand控制 ----
    print(f"\n  --- 1b: Random W_rand ({n_random} random matrices) ---")
    rng = np.random.default_rng(42)
    tau_random_all = {li: [] for li in sample_layers}
    
    for ri in range(n_random):
        W_rand = rng.standard_normal((vocab_size, d_model)) / np.sqrt(d_model)
        final_logits_rand = hs_arrays[final_layer] @ W_rand.T
        final_top1_rand = np.argmax(final_logits_rand, axis=1)
        
        for li in sample_layers:
            logits_li = hs_arrays[li] @ W_rand.T
            tau_vals = []
            for si in range(actual_n):
                top100_idx = np.argsort(-final_logits_rand[si])[:100]
                ranks_final = np.argsort(np.argsort(-final_logits_rand[si, top100_idx]))
                ranks_li = np.argsort(np.argsort(-logits_li[si, top100_idx]))
                t, _ = kendalltau(ranks_final, ranks_li)
                tau_vals.append(float(t) if not np.isnan(t) else 0)
            tau_random_all[li].append(float(np.mean(tau_vals)))
        
        print(f"    Random matrix {ri+1}/{n_random} done")
    
    tau_random = {li: float(np.mean(tau_random_all[li])) for li in sample_layers}
    tau_random_std = {li: float(np.std(tau_random_all[li])) for li in sample_layers}
    
    for li in sample_layers:
        print(f"    L{li:>3d}: tau_rand={tau_random[li]:.4f} ± {tau_random_std[li]:.4f}, "
              f"tau_real={tau_real[li]:.4f}, ratio={tau_real[li]/max(tau_random[li],1e-6):.1f}x")
    
    # ---- 1c: 正交旋转Q@W_U ----
    print(f"\n  --- 1c: Orthogonal Rotation Q@W_U ({n_rotations} rotations) ---")
    tau_rotated_all = {li: [] for li in sample_layers}
    
    for ri in range(n_rotations):
        # 生成d_model×d_model正交矩阵Q, 作用于W_U的列(W_U的d_model维)
        Q = generate_random_orthogonal(d_model, rng=rng)
        W_rot = (Q @ W_U.T).T  # [vocab, d_model] — W_U的列空间被旋转
        
        final_logits_rot = hs_arrays[final_layer] @ W_rot.T
        final_top1_rot = np.argmax(final_logits_rot, axis=1)
        
        for li in sample_layers:
            logits_li = hs_arrays[li] @ W_rot.T
            tau_vals = []
            for si in range(actual_n):
                top100_idx = np.argsort(-final_logits_rot[si])[:100]
                ranks_final = np.argsort(np.argsort(-final_logits_rot[si, top100_idx]))
                ranks_li = np.argsort(np.argsort(-logits_li[si, top100_idx]))
                t, _ = kendalltau(ranks_final, ranks_li)
                tau_vals.append(float(t) if not np.isnan(t) else 0)
            tau_rotated_all[li].append(float(np.mean(tau_vals)))
        
        print(f"    Rotation {ri+1}/{n_rotations} done")
    
    tau_rotated = {li: float(np.mean(tau_rotated_all[li])) for li in sample_layers}
    tau_rotated_std = {li: float(np.std(tau_rotated_all[li])) for li in sample_layers}
    
    for li in sample_layers:
        print(f"    L{li:>3d}: tau_rot={tau_rotated[li]:.4f} ± {tau_rotated_std[li]:.4f}, "
              f"tau_real={tau_real[li]:.4f}, ratio={tau_real[li]/max(tau_rotated[li],1e-6):.1f}x")
    
    # ---- 1d: 跨句控制 ----
    print(f"\n  --- 1d: Cross-Sentence Control ---")
    # 用sentence_i的h_ℓ与sentence_j的h_final计算tau (j≠i)
    tau_cross = {}
    n_cross = min(50, actual_n // 2)
    
    for li in sample_layers:
        tau_vals = []
        for si in range(n_cross):
            sj = (si + 1) % actual_n  # 用下一个句子
            logits_li = hs_arrays[li][si] @ W_U.T
            logits_final_other = hs_arrays[final_layer][sj] @ W_U.T
            top100_idx = np.argsort(-logits_final_other)[:100]
            ranks_final = np.argsort(np.argsort(-logits_final_other[top100_idx]))
            ranks_li = np.argsort(np.argsort(-logits_li[top100_idx]))
            t, _ = kendalltau(ranks_final, ranks_li)
            tau_vals.append(float(t) if not np.isnan(t) else 0)
        tau_cross[li] = float(np.mean(tau_vals))
        print(f"    L{li:>3d}: tau_cross={tau_cross[li]:.4f}, tau_real={tau_real[li]:.4f}, "
              f"ratio={tau_real[li]/max(abs(tau_cross[li]),1e-6):.1f}x")
    
    # ---- 核心判读 ----
    print("\n  *** DECISIVE JUDGMENT ***")
    # 末层前tau的构建比
    pre_final_layers = [li for li in sample_layers if li < final_layer]
    if pre_final_layers:
        late_layer = pre_final_layers[-1]
        tau_real_late = tau_real[late_layer]
        tau_rand_late = tau_random[late_layer]
        tau_rot_late = tau_rotated[late_layer]
        tau_cross_late = tau_cross[late_layer]
        
        print(f"  Late layer (L{late_layer}):")
        print(f"    tau_real    = {tau_real_late:.4f}")
        print(f"    tau_random  = {tau_rand_late:.4f}  (ratio: {tau_real_late/max(tau_rand_late,1e-6):.1f}x)")
        print(f"    tau_rotated = {tau_rot_late:.4f}  (ratio: {tau_real_late/max(tau_rot_late,1e-6):.1f}x)")
        print(f"    tau_cross   = {tau_cross_late:.4f}  (ratio: {tau_real_late/max(abs(tau_cross_late),1e-6):.1f}x)")
        
        if tau_real_late > 3 * tau_rand_late and tau_real_late > 3 * abs(tau_cross_late):
            print(f"  ★★★ tau_real >> tau_random AND tau_real >> tau_cross → 约束传播信号存在!")
        else:
            print(f"  ✗✗✗ tau_real ≈ tau_random → tau可能是W_U对齐伪影!")
        
        if abs(tau_rot_late - tau_real_late) < 0.05:
            print(f"  ★★★ tau_rotated ≈ tau_real → 约束独立于W_U坐标系!")
        elif tau_rot_late < 0.5 * tau_real_late:
            print(f"  ✗✗✗ tau_rotated << tau_real → 约束依赖于W_U坐标系!")
        else:
            print(f"  ? tau_rotated介于中间 → 部分独立, 部分依赖W_U")
    
    return {
        'tau_real': tau_real,
        'cos_real': cos_real,
        'top1_real': top1_real,
        'tau_random': tau_random,
        'tau_random_std': tau_random_std,
        'tau_random_all': {li: [float(v) for v in vals] for li, vals in tau_random_all.items()},
        'tau_rotated': tau_rotated,
        'tau_rotated_std': tau_rotated_std,
        'tau_rotated_all': {li: [float(v) for v in vals] for li, vals in tau_rotated_all.items()},
        'tau_cross': tau_cross,
    }


# ============================================================
# Exp 2: 中间层Logit动力学 — 约束搜索过程
# ============================================================
def exp2_intermediate_logit_dynamics(model, tokenizer, model_name, n_sents=100):
    """
    追踪top-5 token在层间的变化, 揭示约束搜索过程
    
    测量:
    1. 翻转率: top-1 token在相邻层之间改变的频率
    2. 吸引子稳定性: top-1一旦出现, 保持多少层
    3. 正确token的排名轨迹: 正确token在各层的排名变化
    4. 约束竞争: top-5中是否出现"语法竞争"vs"语义竞争"
    """
    print("\n" + "="*60)
    print("Exp 2: Intermediate Logit Dynamics (约束搜索)")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    
    prompts = ALL_PROMPTS[:n_sents] if model_name == "qwen3" else SHORT_PROMPTS[:min(60, n_sents)]
    actual_n = len(prompts)
    # 用更密集的层采样来追踪动力学
    sample_layers = get_sample_layers(n_layers, min(20, n_layers))
    print(f"  n_sents={actual_n}, sample_layers={len(sample_layers)} layers")
    
    # 每句的logit轨迹
    sentence_trajectories = []
    
    for si, prompt in enumerate(prompts):
        if si % 30 == 0:
            print(f"    Sentence {si}/{actual_n}...")
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        
        # 最终层正确token
        final_h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
        final_logits = final_h @ W_U.T
        correct_top1 = int(np.argmax(final_logits))
        
        # 每层的logit信息
        layer_data = []
        for li in sample_layers:
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            logits = h @ W_U.T
            
            top5_ids = np.argsort(-logits)[:5]
            top5_logits = logits[top5_ids]
            top5_tokens = [tokenizer.decode([int(tid)]).strip() for tid in top5_ids]
            
            # 正确token的排名
            correct_rank = int(np.sum(logits > logits[correct_top1]))
            
            # Margin
            sorted_logits = np.sort(logits)[::-1]
            margin = float(sorted_logits[0] - sorted_logits[1])
            
            # Entropy (top-10)
            top10 = sorted_logits[:10]
            probs = np.exp(top10 - top10.max())
            probs /= probs.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            layer_data.append({
                'layer': li,
                'top1_id': int(top5_ids[0]),
                'top1_token': top5_tokens[0],
                'top5_ids': [int(t) for t in top5_ids],
                'top5_tokens': top5_tokens,
                'top5_logits': [float(l) for l in top5_logits],
                'correct_rank': correct_rank,
                'margin': margin,
                'entropy': float(entropy),
                'is_correct': int(top5_ids[0] == correct_top1),
            })
        
        sentence_trajectories.append({
            'prompt': prompt,
            'correct_top1': correct_top1,
            'correct_token': tokenizer.decode([correct_top1]).strip(),
            'layers': layer_data,
        })
    
    # ---- 汇总统计 ----
    print("\n  --- Flipping Rate & Attractor Stability ---")
    flip_rates = []
    attractor_lengths = []
    correct_rank_trajectories = {li: [] for li in sample_layers}
    correct_in_top5_rate = {li: [] for li in sample_layers}
    
    for traj in sentence_trajectories:
        layers = traj['layers']
        
        # 翻转率: top-1在相邻采样层之间改变的比例
        flips = sum(1 for i in range(1, len(layers)) if layers[i]['top1_id'] != layers[i-1]['top1_id'])
        flip_rate = flips / max(len(layers) - 1, 1)
        flip_rates.append(flip_rate)
        
        # 吸引子长度: 正确token一旦成为top-1, 保持多少层
        correct_runs = []
        current_run = 0
        for l in layers:
            if l['is_correct']:
                current_run += 1
            else:
                if current_run > 0:
                    correct_runs.append(current_run)
                current_run = 0
        if current_run > 0:
            correct_runs.append(current_run)
        attractor_lengths.extend(correct_runs) if correct_runs else attractor_lengths.append(0)
        
        # 正确token排名轨迹
        for l in layers:
            correct_rank_trajectories[l['layer']].append(l['correct_rank'])
            correct_in_top5_rate[l['layer']].append(1 if l['correct_rank'] < 5 else 0)
    
    print(f"  Mean flip rate: {np.mean(flip_rates):.3f} ± {np.std(flip_rates):.3f}")
    print(f"  Mean attractor length (correct as top-1): {np.mean(attractor_lengths):.2f}")
    
    # ---- 逐层正确token排名 ----
    print("\n  --- Correct Token Rank Trajectory ---")
    rank_summary = {}
    for li in sample_layers:
        ranks = correct_rank_trajectories[li]
        top5_rate = np.mean(correct_in_top5_rate[li])
        median_rank = np.median(ranks) if ranks else 999
        rank_summary[li] = {
            'mean_rank': float(np.mean(ranks)),
            'median_rank': float(median_rank),
            'top5_rate': float(top5_rate),
        }
        print(f"    L{li:>3d}: mean_rank={rank_summary[li]['mean_rank']:.1f}, "
              f"median={rank_summary[li]['median_rank']:.0f}, "
              f"top5_rate={rank_summary[li]['top5_rate']:.3f}")
    
    # ---- DS7B特殊分析: 中间层负tau的因果 ----
    print("\n  --- Intermediate Wrong Prediction Analysis ---")
    # 找出: 中间层top-1≠最终top-1 的句子, 分析中间层在"想"什么
    wrong_at_mid = 0
    wrong_flips_to_correct = 0
    mid_layer_idx = len(sample_layers) // 2
    
    for traj in sentence_trajectories:
        mid_l = traj['layers'][mid_layer_idx]
        final_l = traj['layers'][-1]
        if not mid_l['is_correct'] and final_l['is_correct']:
            wrong_at_mid += 1
            wrong_flips_to_correct += 1
        elif not mid_l['is_correct']:
            wrong_at_mid += 1
    
    mid_wrong_rate = wrong_at_mid / max(actual_n, 1)
    flip_to_correct_rate = wrong_flips_to_correct / max(wrong_at_mid, 1)
    print(f"  Mid-layer wrong rate: {mid_wrong_rate:.3f}")
    print(f"  Of wrong at mid, flip to correct at end: {flip_to_correct_rate:.3f}")
    
    return {
        'flip_rate_mean': float(np.mean(flip_rates)),
        'flip_rate_std': float(np.std(flip_rates)),
        'attractor_length_mean': float(np.mean(attractor_lengths)),
        'rank_summary': rank_summary,
        'mid_wrong_rate': float(mid_wrong_rate),
        'flip_to_correct_rate': float(flip_to_correct_rate),
        'sample_trajectories': sentence_trajectories[:10],  # 保存10个样例
    }


# ============================================================
# Exp 3: 约束违反层级 — 哪个约束先崩?
# ============================================================
def exp3_constraint_violation(model, tokenizer, model_name, n_sents=60):
    """
    当模型预测错误时, 分析哪些约束被违反
    
    约束层级:
    1. 语法约束: POS序列, 主谓一致, 时态
    2. 语义约束: 词汇搭配, 常识
    3. 逻辑约束: 因果, 条件, 否定
    4. 指代约束: 代词解析
    
    方法: 在每个中间层, 检查top-1 token是否违反各种约束
    """
    print("\n" + "="*60)
    print("Exp 3: Constraint Violation Hierarchy (约束违反层级)")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    
    sample_layers = get_sample_layers(n_layers, 8)
    
    # 精心设计的约束测试对
    # 每对: (prompt, correct_continuation_type, wrong_continuation_type)
    constraint_tests = [
        # 语法: 主谓一致
        ("The cat", "sits", "sit", "grammar_sva"),
        ("The cats", "sit", "sits", "grammar_sva"),
        ("The dog runs", "toward", "towards", "grammar_form"),
        # 语义: 词汇搭配
        ("The red apple", "was", "were", "grammar_sva"),
        ("Hot coffee", "is", "are", "grammar_sva"),
        # 逻辑: 条件句
        ("If it rains", "then", "because", "logic_cond"),
        ("Because she studied", "she", "it", "logic_causal"),
        # 否定
        ("She did not", "finish", "finished", "logic_neg"),
        ("He never", "goes", "went", "logic_neg"),
    ]
    
    results = {}
    
    for prompt, correct, wrong, ctype in constraint_tests:
        correct_ids = tokenizer.encode(correct, add_special_tokens=False)
        wrong_ids = tokenizer.encode(wrong, add_special_tokens=False)
        if not correct_ids or not wrong_ids:
            continue
        correct_id = correct_ids[0]
        wrong_id = wrong_ids[0]
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        
        layer_violation = {}
        for li in sample_layers:
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            logits = h @ W_U.T
            
            correct_logit = float(logits[correct_id])
            wrong_logit = float(logits[wrong_id])
            diff = correct_logit - wrong_logit
            
            # 两者都在top-20中的排名
            top20 = np.argsort(-logits)[:20]
            correct_in_top20 = correct_id in top20
            wrong_in_top20 = wrong_id in top20
            
            layer_violation[li] = {
                'correct_logit': correct_logit,
                'wrong_logit': wrong_logit,
                'diff': diff,
                'correct_wins': diff > 0,
                'correct_in_top20': bool(correct_in_top20),
                'wrong_in_top20': bool(wrong_in_top20),
            }
        
        # 约束onset: 哪层开始correct > wrong
        onset_layer = None
        for li in sample_layers:
            if layer_violation[li]['correct_wins']:
                onset_layer = li
                break
        
        results[ctype] = {
            'prompt': prompt,
            'correct': correct,
            'wrong': wrong,
            'onset_layer': onset_layer,
            'onset_normalized': float(onset_layer / n_layers) if onset_layer is not None else None,
            'layer_data': {str(li): layer_violation[li] for li in sample_layers},
        }
        print(f"  {ctype}: onset_layer={onset_layer}, "
              f"onset_norm={results[ctype]['onset_normalized']:.2f}" if onset_layer is not None 
              else f"  {ctype}: onset_layer=NOT_FOUND")
    
    # 汇总: 按约束类型统计onset
    onset_by_type = defaultdict(list)
    for ctype, r in results.items():
        if r['onset_normalized'] is not None:
            category = ctype.split('_')[0]
            onset_by_type[category].append(r['onset_normalized'])
    
    print("\n  --- Constraint Onset by Category ---")
    for cat, onsets in onset_by_type.items():
        print(f"  {cat}: mean_onset={np.mean(onsets):.2f}, range=[{np.min(onsets):.2f}, {np.max(onsets):.2f}]")
    
    return results


# ============================================================
# Exp 4: 能量景观 — 约束场的统计物理
# ============================================================
def exp4_energy_landscape(model, tokenizer, model_name, n_sents=80):
    """
    从统计物理视角: 把logit空间看作能量景观
    
    定义:
    - E_correct(x) = -log P(correct_token | h_ℓ) — 正确token的能量
    - E_top1(x) = -log P(top1_token | h_ℓ) — 当前top-1的能量
    - ΔE = E_top1 - E_correct — 能量差(正确token比top-1低多少)
    - Barrier = E_second - E_top1 — 决策边界势垒
    
    核心假说: 
    如果约束传播存在, E_correct应该逐层下降(正确token越来越"低能")
    """
    print("\n" + "="*60)
    print("Exp 4: Energy Landscape (能量景观)")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    
    prompts = ALL_PROMPTS[:n_sents] if model_name == "qwen3" else SHORT_PROMPTS[:min(40, n_sents)]
    actual_n = len(prompts)
    sample_layers = get_sample_layers(n_layers, 12)
    
    energy_stats = {li: {'E_correct': [], 'E_top1': [], 'delta_E': [],
                          'barrier': [], 'top1_is_correct': []}
                    for li in sample_layers}
    
    for si, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        
        # 最终层正确token
        final_h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
        final_logits = final_h @ W_U.T
        correct_id = int(np.argmax(final_logits))
        
        for li in sample_layers:
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            logits = h @ W_U.T
            
            # Log-softmax for energy (负log概率 = 能量)
            max_logit = np.max(logits)
            log_sum_exp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
            
            E_correct = -(logits[correct_id] - log_sum_exp)  # -log P(correct)
            
            top1_id = int(np.argmax(logits))
            E_top1 = -(logits[top1_id] - log_sum_exp)  # -log P(top1)
            
            delta_E = E_correct - E_top1  # >0 means correct has higher energy (worse)
            
            # Barrier: top1 vs top2
            sorted_logits = np.sort(logits)[::-1]
            barrier = float(sorted_logits[0] - sorted_logits[1])
            
            energy_stats[li]['E_correct'].append(float(E_correct))
            energy_stats[li]['E_top1'].append(float(E_top1))
            energy_stats[li]['delta_E'].append(float(delta_E))
            energy_stats[li]['barrier'].append(float(barrier))
            energy_stats[li]['top1_is_correct'].append(1 if top1_id == correct_id else 0)
    
    # 汇总
    summary = {}
    print("\n  --- Energy Trajectory ---")
    for li in sample_layers:
        d = energy_stats[li]
        summary[li] = {
            'mean_E_correct': float(np.mean(d['E_correct'])),
            'mean_E_top1': float(np.mean(d['E_top1'])),
            'mean_delta_E': float(np.mean(d['delta_E'])),
            'mean_barrier': float(np.mean(d['barrier'])),
            'correct_rate': float(np.mean(d['top1_is_correct'])),
        }
        print(f"    L{li:>3d}: E_correct={summary[li]['mean_E_correct']:.3f}, "
              f"E_top1={summary[li]['mean_E_top1']:.3f}, "
              f"ΔE={summary[li]['mean_delta_E']:.3f}, "
              f"barrier={summary[li]['mean_barrier']:.3f}, "
              f"correct_rate={summary[li]['correct_rate']:.3f}")
    
    # 核心判读: E_correct是否逐层下降?
    E_values = [summary[li]['mean_E_correct'] for li in sorted(summary.keys())]
    if len(E_values) > 2:
        # 线性趋势
        x = np.arange(len(E_values))
        slope, _ = np.polyfit(x, E_values, 1)
        print(f"\n  E_correct slope: {slope:.4f} per layer sample")
        if slope < -0.01:
            print(f"  ★ E_correct逐层下降 → 正确token'能量'降低 → 约束在收敛!")
        elif slope > 0.01:
            print(f"  ✗ E_correct逐层上升 → 正确token变得更'高能' → 不支持约束收敛")
        else:
            print(f"  ? E_correct基本不变")
    
    return summary


# ============================================================
# Main
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Phase 155: Constraint Mechanics — {model_name}, {ts}")

    t0 = time.time()
    model, tokenizer, device = load_model_custom(model_name)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}, load={time.time()-t0:.1f}s")

    # Exp 1: W_U对齐决定性测试 (最重要!)
    e1 = exp1_wu_alignment_test(model, tokenizer, model_name,
                                 n_sents=300 if model_name == "qwen3" else 150,
                                 n_random=5, n_rotations=3)
    
    # Exp 2: 中间层Logit动力学
    e2 = exp2_intermediate_logit_dynamics(model, tokenizer, model_name,
                                           n_sents=100 if model_name == "qwen3" else 60)
    
    # Exp 3: 约束违反层级
    e3 = exp3_constraint_violation(model, tokenizer, model_name, n_sents=60)
    
    # Exp 4: 能量景观
    e4 = exp4_energy_landscape(model, tokenizer, model_name,
                                n_sents=80 if model_name == "qwen3" else 40)

    all_r = {
        "phase": "155_constraint_mechanics",
        "model": model_name,
        "timestamp": ts,
        "model_info": {"class": info.model_class, "n_layers": info.n_layers, "d_model": info.d_model},
        "exp1_wu_alignment": e1,
        "exp2_logit_dynamics": e2,
        "exp3_constraint_violation": e3,
        "exp4_energy_landscape": e4,
    }

    rf = OUTPUT_DIR / f"phase155_{model_name}_{ts}.json"
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, (np.int32, np.int64)): return int(o)
        if isinstance(o, bool): return bool(o)
        raise TypeError(f"Cannot serialize {type(o)}")
    with open(rf, 'w', encoding='utf-8') as f:
        json.dump(all_r, f, indent=2, default=conv, ensure_ascii=False)
    print(f"\nSaved: {rf}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("Phase 155 done.")


if __name__ == "__main__":
    main()
