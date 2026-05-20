"""
Phase 236: Program Geometry — Δ Structure Analysis
====================================================

核心目标: 从"现象描述"转向"数学结构"的验证

关键问题:
  1. Δ_not有多少个自由度? (SVD分析 — 最关键!)
  2. 不同控制算子的Δ是否共享子空间?
  3. 双重否定为什么走得更远? (长度控制实验)

三模型: qwen3 / glm4 / deepseek7b

使用方式:
  python tests/glm5/phase236_program_geometry.py qwen3 --quick
  python tests/glm5/phase236_program_geometry.py qwen3
  python tests/glm5/phase236_program_geometry.py qwen3 --large
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import json
import argparse
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from model_utils import (get_layers, get_model_info, get_W_U,
                          MODEL_CONFIGS, release_model)


# ===== 进度日志 =====
class ProgressLogger:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.last_time = time.time()
    
    def update(self, msg):
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        print(f"  {self.prefix}{msg} ({elapsed:.1f}s)", flush=True)


# ===== 模型加载 (BF16, device_map="auto" for GLM4/DS7B) =====
def load_model_bf16(model_name: str):
    """BF16加载 — 所有模型用bfloat16, GLM4/DS7B用device_map='auto'"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bfloat16)...", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 所有模型用BF16 + device_map="auto"
    # 优先flash_attention_2节省内存, 失败则回退eager
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        print(f"[load] Using flash_attention_2", flush=True)
    except Exception as e:
        print(f"[load] flash_attention_2 failed ({e}), falling back to eager", flush=True)
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
    
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        print(f"[load] {model_name}: GPU={gpu_count} comps, CPU={cpu_count} comps, "
              f"class={type(model).__name__}, GPU mem={gpu_mem:.2f}GB", flush=True)
    else:
        print(f"[load] {model_name}: device={device}, class={type(model).__name__}, "
              f"GPU mem={gpu_mem:.2f}GB", flush=True)
    
    return model, tokenizer, device


def get_input_device(model):
    """获取输入tensor应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== 句子集合 =====

# 大规模句子对用于SVD分析 (ExpA & ExpB)
# 覆盖: 不同主语、不同形容词、不同句型
BASE_SENTENCES = [
    # 简单主语+形容词 (200+对)
    "The cat is happy.", "The dog is friendly.", "The bird is small.",
    "The car is fast.", "The house is big.", "The book is useful.",
    "The food is delicious.", "The movie is interesting.", "The idea is original.",
    "The plan is feasible.", "The result is surprising.", "The answer is correct.",
    "The method is efficient.", "The problem is simple.", "The solution is elegant.",
    "The weather is warm.", "The water is cold.", "The fire is hot.",
    "The sky is clear.", "The road is long.", "The river is deep.",
    "The mountain is tall.", "The forest is dense.", "The desert is dry.",
    "The city is noisy.", "The village is quiet.", "The garden is beautiful.",
    "The flower is red.", "The tree is old.", "The grass is green.",
    "The snow is white.", "The night is dark.", "The sun is bright.",
    "The wind is strong.", "The rain is heavy.", "The cloud is grey.",
    "The ocean is vast.", "The island is small.", "The lake is calm.",
    "The valley is wide.", "The hill is steep.", "The cave is deep.",
    "The bridge is narrow.", "The tower is high.", "The wall is thick.",
    "The door is open.", "The window is closed.", "The floor is clean.",
    "The table is round.", "The chair is comfortable.", "The bed is soft.",
    "The lamp is bright.", "The mirror is clear.", "The clock is accurate.",
    "The phone is smart.", "The computer is powerful.", "The screen is large.",
    "The keyboard is responsive.", "The mouse is wireless.", "The printer is fast.",
    "The camera is digital.", "The speaker is loud.", "The microphone is sensitive.",
    "The battery is rechargeable.", "The cable is long.", "The charger is portable.",
    "The student is diligent.", "The teacher is patient.", "The doctor is experienced.",
    "The engineer is creative.", "The artist is talented.", "The musician is skilled.",
    "The writer is prolific.", "The scientist is curious.", "The lawyer is persuasive.",
    "The manager is organized.", "The leader is inspiring.", "The worker is reliable.",
    "The athlete is strong.", "The player is fast.", "The team is united.",
    "The game is exciting.", "The match is close.", "The score is tied.",
    "The prize is valuable.", "The award is prestigious.", "The reward is generous.",
    "The cake is sweet.", "The bread is fresh.", "The soup is warm.",
    "The tea is hot.", "The coffee is strong.", "The juice is cold.",
    "The fruit is ripe.", "The vegetable is fresh.", "The meat is tender.",
    "The fish is delicious.", "The rice is cooked.", "The salad is healthy.",
    "The project is successful.", "The experiment is valid.", "The theory is sound.",
    "The hypothesis is testable.", "The data is reliable.", "The evidence is convincing.",
    "The argument is logical.", "The conclusion is reasonable.", "The prediction is accurate.",
    "The model is complex.", "The system is stable.", "The process is efficient.",
    "The procedure is standard.", "The protocol is strict.", "The rule is clear.",
    "The law is just.", "The policy is fair.", "The regulation is necessary.",
    "The standard is high.", "The quality is excellent.", "The performance is outstanding.",
    "The service is prompt.", "The product is durable.", "The design is innovative.",
    "The feature is unique.", "The function is useful.", "The interface is intuitive.",
    "The experience is memorable.", "The journey is long.", "The adventure is thrilling.",
    "The story is compelling.", "The character is complex.", "The plot is engaging.",
    "The ending is surprising.", "The beginning is promising.", "The middle is exciting.",
    "The message is clear.", "The meaning is deep.", "The purpose is noble.",
    "The goal is achievable.", "The task is challenging.", "The mission is critical.",
    "The vision is grand.", "The dream is ambitious.", "The hope is alive.",
    "The fear is real.", "The danger is imminent.", "The risk is significant.",
    "The opportunity is rare.", "The chance is slim.", "The possibility is endless.",
    "The future is bright.", "The past is gone.", "The present is precious.",
    "The moment is fleeting.", "The time is limited.", "The space is vast.",
    "The universe is infinite.", "The world is changing.", "The life is beautiful.",
    "The love is deep.", "The hate is strong.", "The joy is pure.",
    "The sorrow is heavy.", "The anger is fierce.", "The peace is lasting.",
    "The war is brutal.", "The conflict is intense.", "The harmony is perfect.",
    "The balance is delicate.", "The order is stable.", "The chaos is overwhelming.",
    "The pattern is clear.", "The structure is solid.", "The foundation is strong.",
    "The building is tall.", "The room is spacious.", "The hall is grand.",
    "The park is peaceful.", "The street is busy.", "The market is crowded.",
    "The shop is open.", "The restaurant is popular.", "The hotel is luxurious.",
    "The airport is modern.", "The station is busy.", "The port is active.",
    "The factory is large.", "The farm is productive.", "The mine is deep.",
    "The well is dry.", "The spring is fresh.", "The stream is clear.",
    "The path is narrow.", "The trail is winding.", "The route is direct.",
    "The map is accurate.", "The guide is helpful.", "The sign is visible.",
    "The signal is strong.", "The message is urgent.", "The news is surprising.",
    "The report is detailed.", "The analysis is thorough.", "The review is positive.",
    "The feedback is constructive.", "The criticism is fair.", "The praise is deserved.",
    "The award is honorary.", "The title is prestigious.", "The rank is high.",
    "The position is important.", "The role is crucial.", "The responsibility is great.",
    "The duty is clear.", "The obligation is binding.", "The commitment is firm.",
    "The promise is sacred.", "The oath is solemn.", "The vow is unbreakable.",
    "The contract is valid.", "The agreement is mutual.", "The deal is fair.",
    "The price is reasonable.", "The cost is minimal.", "The value is high.",
    "The benefit is obvious.", "The advantage is clear.", "The improvement is significant.",
    "The progress is steady.", "The development is rapid.", "The growth is sustainable.",
    "The change is inevitable.", "The transformation is complete.", "The evolution is ongoing.",
    "The revolution is coming.", "The innovation is groundbreaking.", "The discovery is remarkable.",
    "The invention is ingenious.", "The creation is original.", "The work is impressive.",
    "The effort is commendable.", "The achievement is extraordinary.", "The success is well-deserved.",
    "The failure is temporary.", "The setback is minor.", "The obstacle is surmountable.",
    "The challenge is exciting.", "The problem is solvable.", "The question is answerable.",
    "The mystery is intriguing.", "The puzzle is complex.", "The riddle is clever.",
    "The secret is safe.", "The truth is absolute.", "The fact is undeniable.",
    "The reality is harsh.", "The illusion is convincing.", "The dream is vivid.",
    "The memory is fading.", "The thought is profound.", "The idea is brilliant.",
    "The concept is abstract.", "The principle is fundamental.", "The rule is simple.",
    "The law is universal.", "The theory is elegant.", "The model is accurate.",
    "The formula is beautiful.", "The equation is balanced.", "The proof is rigorous.",
    "The logic is flawless.", "The reason is sound.", "The argument is compelling.",
    "The evidence is overwhelming.", "The data is consistent.", "The pattern is unmistakable.",
    "The trend is upward.", "The direction is clear.", "The path is forward.",
    "The way is open.", "The door is unlocked.", "The gate is wide.",
    "The entrance is welcoming.", "The exit is narrow.", "The passage is hidden.",
    "The corridor is dark.", "The staircase is steep.", "The elevator is fast.",
    "The escalator is moving.", "The platform is crowded.", "The stage is set.",
]

# ExpC: 长度控制的双重否定四元组
# A: 肯定句 | B: 否定句 | C: 双重否定 | D: 等长肯定句(长度控制)
LENGTH_CONTROLLED_QUADS = [
    ("The cat is black.",
     "The cat is not black.",
     "It is not true that the cat is not black.",
     "I would say that the cat is black indeed."),
    ("The sky is blue.",
     "The sky is not blue.",
     "It is not true that the sky is not blue.",
     "I would say that the sky is blue indeed."),
    ("The water is cold.",
     "The water is not cold.",
     "It is not true that the water is not cold.",
     "I would say that the water is cold indeed."),
    ("The food is hot.",
     "The food is not hot.",
     "It is not true that the food is not hot.",
     "I would say that the food is hot indeed."),
    ("The dog is friendly.",
     "The dog is not friendly.",
     "It is not true that the dog is not friendly.",
     "I would say that the dog is friendly indeed."),
    ("The car is fast.",
     "The car is not fast.",
     "It is not true that the car is not fast.",
     "I would say that the car is fast indeed."),
    ("The house is big.",
     "The house is not big.",
     "It is not true that the house is not big.",
     "I would say that the house is big indeed."),
    ("The book is useful.",
     "The book is not useful.",
     "It is not true that the book is not useful.",
     "I would say that the book is useful indeed."),
    ("The plan is good.",
     "The plan is not good.",
     "It is not true that the plan is not good.",
     "I would say that the plan is good indeed."),
    ("The result is clear.",
     "The result is not clear.",
     "It is not true that the result is not clear.",
     "I would say that the result is clear indeed."),
    ("The method is simple.",
     "The method is not simple.",
     "It is not true that the method is not simple.",
     "I would say that the method is simple indeed."),
    ("The answer is correct.",
     "The answer is not correct.",
     "It is not true that the answer is not correct.",
     "I would say that the answer is correct indeed."),
    ("The road is long.",
     "The road is not long.",
     "It is not true that the road is not long.",
     "I would say that the road is long indeed."),
    ("The fire is hot.",
     "The fire is not hot.",
     "It is not true that the fire is not hot.",
     "I would say that the fire is hot indeed."),
    ("The night is dark.",
     "The night is not dark.",
     "It is not true that the night is not dark.",
     "I would say that the night is dark indeed."),
    ("The city is noisy.",
     "The city is not noisy.",
     "It is not true that the city is not noisy.",
     "I would say that the city is noisy indeed."),
    ("The garden is beautiful.",
     "The garden is not beautiful.",
     "It is not true that the garden is not beautiful.",
     "I would say that the garden is beautiful indeed."),
    ("The mountain is tall.",
     "The mountain is not tall.",
     "It is not true that the mountain is not tall.",
     "I would say that the mountain is tall indeed."),
    ("The river is deep.",
     "The river is not deep.",
     "It is not true that the river is not deep.",
     "I would say that the river is deep indeed."),
    ("The wind is strong.",
     "The wind is not strong.",
     "It is not true that the wind is not strong.",
     "I would say that the wind is strong indeed."),
    ("The student is smart.",
     "The student is not smart.",
     "It is not true that the student is not smart.",
     "I would say that the student is smart indeed."),
    ("The doctor is kind.",
     "The doctor is not kind.",
     "It is not true that the doctor is not kind.",
     "I would say that the doctor is kind indeed."),
    ("The game is fun.",
     "The game is not fun.",
     "It is not true that the game is not fun.",
     "I would say that the game is fun indeed."),
    ("The cake is sweet.",
     "The cake is not sweet.",
     "It is not true that the cake is not sweet.",
     "I would say that the cake is sweet indeed."),
    ("The coffee is strong.",
     "The coffee is not strong.",
     "It is not true that the coffee is not strong.",
     "I would say that the coffee is strong indeed."),
    ("The movie is scary.",
     "The movie is not scary.",
     "It is not true that the movie is not scary.",
     "I would say that the movie is scary indeed."),
    ("The song is loud.",
     "The song is not loud.",
     "It is not true that the song is not loud.",
     "I would say that the song is loud indeed."),
    ("The story is sad.",
     "The story is not sad.",
     "It is not true that the story is not sad.",
     "I would say that the story is sad indeed."),
    ("The weather is nice.",
     "The weather is not nice.",
     "It is not true that the weather is not nice.",
     "I would say that the weather is nice indeed."),
    ("The price is fair.",
     "The price is not fair.",
     "It is not true that the price is not fair.",
     "I would say that the price is fair indeed."),
]

# ExpB: 跨控制算子 — 对同一批基础句子施加不同算子
OPERATORS = {
    "not":     lambda s: s.replace(" is ", " is not ", 1),
    "never":   lambda s: s.replace(" is ", " is never ", 1),
    "always":  lambda s: s.replace(" is ", " is always ", 1),
    "rarely":  lambda s: s.replace(" is ", " is rarely ", 1),
    "often":   lambda s: s.replace(" is ", " is often ", 1),
}


# ===== 安全计算 =====

def safe_cosine(a, b):
    """安全计算余弦相似度"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_kl(p_logits, q_logits):
    """安全计算KL散度 (基于softmax概率)"""
    p = np.exp(p_logits - np.max(p_logits))
    p = p / p.sum()
    q = np.exp(q_logits - np.max(q_logits))
    q = q / q.sum()
    q = np.maximum(q, 1e-10)
    p = np.maximum(p, 1e-10)
    return float(np.sum(p * np.log(p / q)))


# ===== Hidden State提取 =====

def get_last_token_hidden(model, input_ids, n_layers, W_U=None):
    """
    获取最后一层最后一个token的hidden state和logits
    
    Returns:
        h_last: [d_model] numpy
        logits: [vocab_size] numpy
    """
    input_device = get_input_device(model)
    ids = input_ids.to(input_device)
    
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    
    h_last = out.hidden_states[-1][0, -1].detach().float().cpu().numpy()  # [d_model]
    logits = out.logits[0, -1].detach().float().cpu().numpy()  # [vocab_size]
    
    return h_last, logits


def get_all_layer_hiddens(model, input_ids, n_layers):
    """
    获取所有层最后一个token的hidden state
    
    Returns:
        list of [d_model] numpy arrays
    """
    input_device = get_input_device(model)
    ids = input_ids.to(input_device)
    
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    
    result = []
    for hs in out.hidden_states:
        result.append(hs[0, -1].detach().float().cpu().numpy())
    return result


# ===== ExpA: Δ_not SVD自由度分析 =====

def expA_svd_freedom(model, tokenizer, device, info, n_sentences=200, model_name=None):
    """
    核心实验: Δ_not的SVD分解
    
    目标: 回答"否定的数学结构有多简洁?"
    - 前k个奇异值解释90%方差 → k就是否定的有效自由度
    - k<10: 简洁低维结构, "背后有数学结构"假设得到支持
    - k>50: 高度情境化, 可能没有统一数学结构
    
    同时在hidden state空间和logit空间做SVD
    """
    print(f"\n{'='*60}", flush=True)
    print(f"ExpA: Δ_not SVD Freedom Analysis", flush=True)
    print(f"{'='*60}", flush=True)
    
    logger = ProgressLogger("ExpA: ")
    n_layers = info.n_layers
    d_model = info.d_model
    W_U = get_W_U(model, model_name)  # [vocab_size, d_model]
    
    sentences = BASE_SENTENCES[:n_sentences]
    
    # 收集Δ向量
    delta_h_matrix = []   # hidden state Δ: [n_sentences, d_model]
    delta_z_matrix = []   # logit Δ: [n_sentences, vocab_size]
    valid_count = 0
    
    for si, sent in enumerate(sentences):
        if si % 20 == 0:
            logger.update(f"sentence {si+1}/{len(sentences)}")
        
        # 构造否定句
        if " is " not in sent:
            continue
        
        neg_sent = sent.replace(" is ", " is not ", 1)
        
        # 编码
        aff_ids = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        neg_ids = tokenizer(neg_sent, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        
        # 获取hidden states
        aff_h, aff_logits = get_last_token_hidden(model, aff_ids, n_layers, W_U)
        neg_h, neg_logits = get_last_token_hidden(model, neg_ids, n_layers, W_U)
        
        # 计算Δ
        delta_h = neg_h - aff_h  # [d_model]
        delta_z = neg_logits - aff_logits  # [vocab_size]
        
        # 过滤零向量
        if np.linalg.norm(delta_h) > 1e-6 and np.linalg.norm(delta_z) > 1e-6:
            delta_h_matrix.append(delta_h)
            delta_z_matrix.append(delta_z)
            valid_count += 1
        
        # 定期释放内存
        if si % 50 == 49:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logger.update(f"Collected {valid_count} valid Δ vectors")
    
    if valid_count < 10:
        print("  ERROR: Too few valid vectors for SVD!", flush=True)
        return {"error": "insufficient data"}
    
    # 构建矩阵
    delta_h_matrix = np.array(delta_h_matrix)  # [N, d_model]
    delta_z_matrix = np.array(delta_z_matrix)  # [N, vocab_size]
    
    # ===== Hidden State Space SVD =====
    print(f"\n  --- Hidden State Space SVD ---", flush=True)
    # 中心化
    delta_h_centered = delta_h_matrix - delta_h_matrix.mean(axis=0, keepdims=True)
    # SVD (用economy SVD, N < d_model时更高效)
    U_h, S_h, Vt_h = np.linalg.svd(delta_h_centered, full_matrices=False)
    
    # 累积方差解释比
    total_var_h = np.sum(S_h ** 2)
    cum_var_h = np.cumsum(S_h ** 2) / total_var_h
    
    # 找到90%和95%方差的k
    k_90_h = int(np.searchsorted(cum_var_h, 0.90)) + 1
    k_95_h = int(np.searchsorted(cum_var_h, 0.95)) + 1
    k_99_h = int(np.searchsorted(cum_var_h, 0.99)) + 1
    
    print(f"  N={valid_count}, d_model={d_model}", flush=True)
    print(f"  Top-10 singular values: {S_h[:10].round(4)}", flush=True)
    print(f"  k(90%)={k_90_h}, k(95%)={k_95_h}, k(99%)={k_99_h}", flush=True)
    print(f"  Top-1 explains: {cum_var_h[0]*100:.1f}%", flush=True)
    print(f"  Top-3 explains: {cum_var_h[2]*100:.1f}%", flush=True)
    print(f"  Top-5 explains: {cum_var_h[min(4,len(cum_var_h)-1)]*100:.1f}%", flush=True)
    print(f"  Top-10 explains: {cum_var_h[min(9,len(cum_var_h)-1)]*100:.1f}%", flush=True)
    
    # ===== Logit Space SVD =====
    print(f"\n  --- Logit Space SVD ---", flush=True)
    delta_z_centered = delta_z_matrix - delta_z_matrix.mean(axis=0, keepdims=True)
    U_z, S_z, Vt_z = np.linalg.svd(delta_z_centered, full_matrices=False)
    
    total_var_z = np.sum(S_z ** 2)
    cum_var_z = np.cumsum(S_z ** 2) / total_var_z
    
    k_90_z = int(np.searchsorted(cum_var_z, 0.90)) + 1
    k_95_z = int(np.searchsorted(cum_var_z, 0.95)) + 1
    k_99_z = int(np.searchsorted(cum_var_z, 0.99)) + 1
    
    print(f"  Top-10 singular values: {S_z[:10].round(4)}", flush=True)
    print(f"  k(90%)={k_90_z}, k(95%)={k_95_z}, k(99%)={k_99_z}", flush=True)
    print(f"  Top-1 explains: {cum_var_z[0]*100:.1f}%", flush=True)
    print(f"  Top-3 explains: {cum_var_z[2]*100:.1f}%", flush=True)
    print(f"  Top-5 explains: {cum_var_z[min(4,len(cum_var_z)-1)]*100:.1f}%", flush=True)
    print(f"  Top-10 explains: {cum_var_z[min(9,len(cum_var_z)-1)]*100:.1f}%", flush=True)
    
    # ===== 逐层SVD (采样层) =====
    print(f"\n  --- Per-Layer SVD (sampled layers) ---", flush=True)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    sample_layers = sorted(set(sample_layers + [n_layers - 1]))
    
    # 一次性收集所有层的Δ (避免重复前向传播)
    layer_delta_cache = {li: [] for li in sample_layers}  # li -> list of [d_model]
    
    for si, sent in enumerate(sentences):
        if si % 30 == 0:
            logger.update(f"collecting per-layer data {si+1}/{len(sentences)}")
        if " is " not in sent:
            continue
        
        neg_sent = sent.replace(" is ", " is not ", 1)
        aff_ids = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        neg_ids = tokenizer(neg_sent, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        
        aff_all = get_all_layer_hiddens(model, aff_ids, n_layers)
        neg_all = get_all_layer_hiddens(model, neg_ids, n_layers)
        
        for li in sample_layers:
            if li < len(aff_all) and li < len(neg_all):
                delta = neg_all[li] - aff_all[li]
                if np.linalg.norm(delta) > 1e-6:
                    layer_delta_cache[li].append(delta)
        
        del aff_all, neg_all
        if si % 30 == 29:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 对每个采样层做SVD
    per_layer_k90 = {}
    for li in sample_layers:
        layer_deltas = layer_delta_cache[li]
        if len(layer_deltas) >= 10:
            layer_deltas = np.array(layer_deltas)
            layer_deltas_c = layer_deltas - layer_deltas.mean(axis=0, keepdims=True)
            _, S_layer, _ = np.linalg.svd(layer_deltas_c, full_matrices=False)
            cum_var_layer = np.cumsum(S_layer ** 2) / np.sum(S_layer ** 2)
            k90 = int(np.searchsorted(cum_var_layer, 0.90)) + 1
            per_layer_k90[str(li)] = {
                "k90": k90,
                "top1_var": float(cum_var_layer[0]),
                "top3_var": float(cum_var_layer[min(2, len(cum_var_layer)-1)]),
                "n": len(layer_deltas),
            }
            print(f"  L{li}: k90={k90}, top1={cum_var_layer[0]*100:.1f}%, top3={cum_var_layer[min(2,len(cum_var_layer)-1)]*100:.1f}%", flush=True)
        else:
            print(f"  L{li}: insufficient data ({len(layer_deltas)} pairs)", flush=True)
    
    del layer_delta_cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ===== 判定 =====
    if k_90_h < 10:
        verdict_h = f"LOW_DIM (k90={k_90_h}): Negation has concise structure!"
    elif k_90_h < 50:
        verdict_h = f"MEDIUM_DIM (k90={k_90_h}): Partial structure, context-dependent"
    else:
        verdict_h = f"HIGH_DIM (k90={k_90_h}): No unified structure, highly contextual"
    
    if k_90_z < 10:
        verdict_z = f"LOW_DIM (k90={k_90_z}): Logit-space negation is concise!"
    elif k_90_z < 50:
        verdict_z = f"MEDIUM_DIM (k90={k_90_z}): Partial structure in logit space"
    else:
        verdict_z = f"HIGH_DIM (k90={k_90_z}): Logit-space negation is high-dimensional"
    
    print(f"\n  VERDICT (hidden state): {verdict_h}", flush=True)
    print(f"  VERDICT (logit space): {verdict_z}", flush=True)
    
    return {
        "n_valid": valid_count,
        "hidden_state_svd": {
            "k90": k_90_h, "k95": k_95_h, "k99": k_99_h,
            "top1_var": float(cum_var_h[0]),
            "top3_var": float(cum_var_h[min(2, len(cum_var_h)-1)]),
            "top5_var": float(cum_var_h[min(4, len(cum_var_h)-1)]),
            "top10_var": float(cum_var_h[min(9, len(cum_var_h)-1)]),
            "top10_sv": [float(x) for x in S_h[:10]],
            "verdict": verdict_h,
        },
        "logit_svd": {
            "k90": k_90_z, "k95": k_95_z, "k99": k_99_z,
            "top1_var": float(cum_var_z[0]),
            "top3_var": float(cum_var_z[min(2, len(cum_var_z)-1)]),
            "top5_var": float(cum_var_z[min(4, len(cum_var_z)-1)]),
            "top10_var": float(cum_var_z[min(9, len(cum_var_z)-1)]),
            "top10_sv": [float(x) for x in S_z[:10]],
            "verdict": verdict_z,
        },
        "per_layer_k90": per_layer_k90,
    }


# ===== ExpB: 跨控制算子Δ结构对比 =====

def expB_cross_operator(model, tokenizer, device, info, n_sentences=100, model_name=None):
    """
    对比不同控制算子(not/never/always/rarely/often)的Δ结构
    
    核心检验:
    - 不同算子的Δ是否共享子空间?
    - not vs never: 是否共享"否定子空间"?
    - not vs always: 是否共享"量词子空间"?
    """
    print(f"\n{'='*60}", flush=True)
    print(f"ExpB: Cross-Operator Δ Structure Comparison", flush=True)
    print(f"{'='*60}", flush=True)
    
    logger = ProgressLogger("ExpB: ")
    n_layers = info.n_layers
    
    sentences = BASE_SENTENCES[:n_sentences]
    # 过滤只保留含" is "的句子
    sentences = [s for s in sentences if " is " in s]
    
    operator_deltas = {}  # op_name -> [N, d_model]
    operator_logits_deltas = {}  # op_name -> [N, vocab_size]
    
    for op_name, op_func in OPERATORS.items():
        logger.update(f"Operator: {op_name}")
        deltas_h = []
        deltas_z = []
        
        for si, sent in enumerate(sentences):
            mod_sent = op_func(sent)
            if mod_sent == sent:
                continue  # 替换失败
            
            aff_ids = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            mod_ids = tokenizer(mod_sent, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            
            aff_h, aff_logits = get_last_token_hidden(model, aff_ids, n_layers)
            mod_h, mod_logits = get_last_token_hidden(model, mod_ids, n_layers)
            
            delta_h = mod_h - aff_h
            delta_z = mod_logits - aff_logits
            
            if np.linalg.norm(delta_h) > 1e-6:
                deltas_h.append(delta_h)
                deltas_z.append(delta_z)
        
        operator_deltas[op_name] = np.array(deltas_h) if deltas_h else None
        operator_logits_deltas[op_name] = np.array(deltas_z) if deltas_z else None
        logger.update(f"  {op_name}: {len(deltas_h)} valid pairs")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===== 对每个算子做SVD =====
    print(f"\n  --- Per-Operator SVD ---", flush=True)
    operator_svd = {}
    
    for op_name in OPERATORS:
        deltas = operator_deltas[op_name]
        if deltas is None or len(deltas) < 10:
            print(f"  {op_name}: insufficient data", flush=True)
            continue
        
        deltas_c = deltas - deltas.mean(axis=0, keepdims=True)
        _, S, Vt = np.linalg.svd(deltas_c, full_matrices=False)
        cum_var = np.cumsum(S ** 2) / np.sum(S ** 2)
        k90 = int(np.searchsorted(cum_var, 0.90)) + 1
        
        operator_svd[op_name] = {
            "k90": k90,
            "top1_var": float(cum_var[0]),
            "top3_var": float(cum_var[min(2, len(cum_var)-1)]),
            "top5_var": float(cum_var[min(4, len(cum_var)-1)]),
            "n": len(deltas),
            "top5_sv": [float(x) for x in S[:5]],
        }
        print(f"  {op_name}: k90={k90}, top1={cum_var[0]*100:.1f}%, top3={cum_var[min(2,len(cum_var)-1)]*100:.1f}%", flush=True)
    
    # ===== 子空间重叠分析 =====
    print(f"\n  --- Subspace Overlap Analysis ---", flush=True)
    # 对每对算子, 计算主成分方向的子空间重叠
    # 用投影矩阵的Frobenius范数比: ||P_A * P_B||_F / sqrt(min(k_A, k_B))
    
    subspace_overlaps = {}
    op_names = list(operator_svd.keys())
    
    for i, op_a in enumerate(op_names):
        for j, op_b in enumerate(op_names):
            if i >= j:
                continue
            
            deltas_a = operator_deltas[op_a]
            deltas_b = operator_deltas[op_b]
            
            if deltas_a is None or deltas_b is None:
                continue
            
            # 获取各算子的前k个主方向
            k = min(10, operator_svd[op_a]["k90"], operator_svd[op_b]["k90"])
            
            # SVD of A
            U_a, _, _ = np.linalg.svd(
                deltas_a - deltas_a.mean(axis=0, keepdims=True), full_matrices=False)
            # SVD of B
            U_b, _, _ = np.linalg.svd(
                deltas_b - deltas_b.mean(axis=0, keepdims=True), full_matrices=True)
            
            # 子空间投影重叠: ||U_a[:,:k]^T @ U_b[:,:k]||_F / k
            # 1.0 = 完全重叠, 0.0 = 完全正交
            proj = U_a[:, :k].T @ U_b[:, :k]  # [k, k]
            overlap = np.linalg.norm(proj, 'fro') / np.sqrt(k)
            
            # 也计算逐方向余弦
            cosines = []
            for di in range(min(5, k)):
                cos = abs(float(np.dot(U_a[:, di], U_b[:, di])))
                cosines.append(cos)
            
            key = f"{op_a}_vs_{op_b}"
            subspace_overlaps[key] = {
                "overlap_frobenius": float(overlap),
                "k_used": k,
                "top5_dir_cosines": cosines,
            }
            print(f"  {op_a} vs {op_b}: overlap={overlap:.4f}, top5_cos={[f'{c:.3f}' for c in cosines]}", flush=True)
    
    # ===== 判定 =====
    # 关键问题: not vs never 共享否定子空间吗? not vs always 共享量词子空间吗?
    not_vs_never = subspace_overlaps.get("not_vs_never", {})
    not_vs_always = subspace_overlaps.get("not_vs_always", {})
    not_vs_rarely = subspace_overlaps.get("not_vs_rarely", {})
    never_vs_always = subspace_overlaps.get("never_vs_always", {})
    
    verdict_parts = []
    if not_vs_never.get("overlap_frobenius", 0) > 0.5:
        verdict_parts.append("not/never SHARE negation subspace")
    else:
        verdict_parts.append("not/never DO NOT share negation subspace")
    
    if not_vs_always.get("overlap_frobenius", 0) > 0.5:
        verdict_parts.append("not/always SHARE quantifier subspace")
    else:
        verdict_parts.append("not/always DO NOT share quantifier subspace")
    
    verdict = "; ".join(verdict_parts)
    print(f"\n  VERDICT: {verdict}", flush=True)
    
    return {
        "operator_svd": operator_svd,
        "subspace_overlaps": subspace_overlaps,
        "verdict": verdict,
    }


# ===== ExpC: 长度控制的双重否定 =====

def expC_length_controlled_double_neg(model, tokenizer, device, info, n_quads=20, model_name=None):
    """
    区分双重否定偏离的三种解释:
    1. 语义漂移 (semantic drift): 上下文污染
    2. 方向性位移 (directional displacement): 连续施加走更远
    3. 长度效应 (length effect): 更长句子自然不同
    
    四元组: A(肯定), B(否定), C(双重否定), D(等长肯定)
    - KL(C||A) vs KL(D||A): 长度效应控制
    - 如果 KL(C||A) >> KL(D||A): 双重否定有真实语义偏离, 非长度效应
    """
    print(f"\n{'='*60}", flush=True)
    print(f"ExpC: Length-Controlled Double Negation", flush=True)
    print(f"{'='*60}", flush=True)
    
    logger = ProgressLogger("ExpC: ")
    n_layers = info.n_layers
    
    quads = LENGTH_CONTROLLED_QUADS[:n_quads]
    
    kl_ab_list = []  # KL(B||A) = 单次否定
    kl_ac_list = []  # KL(C||A) = 双重否定
    kl_ad_list = []  # KL(D||A) = 等长肯定 (长度控制)
    
    cos_ab_list = []
    cos_ac_list = []
    cos_ad_list = []
    
    for qi, (a, b, c, d) in enumerate(quads):
        logger.update(f"quad {qi+1}/{len(quads)}")
        
        ids_a = tokenizer(a, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        ids_b = tokenizer(b, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        ids_c = tokenizer(c, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        ids_d = tokenizer(d, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        
        _, logits_a = get_last_token_hidden(model, ids_a, n_layers)
        _, logits_b = get_last_token_hidden(model, ids_b, n_layers)
        _, logits_c = get_last_token_hidden(model, ids_c, n_layers)
        _, logits_d = get_last_token_hidden(model, ids_d, n_layers)
        
        kl_ab = safe_kl(logits_b, logits_a)
        kl_ac = safe_kl(logits_c, logits_a)
        kl_ad = safe_kl(logits_d, logits_a)
        
        kl_ab_list.append(kl_ab)
        kl_ac_list.append(kl_ac)
        kl_ad_list.append(kl_ad)
        
        # hidden state cosine
        h_a, _ = get_last_token_hidden(model, ids_a, n_layers)
        h_b, _ = get_last_token_hidden(model, ids_b, n_layers)
        h_c, _ = get_last_token_hidden(model, ids_c, n_layers)
        h_d, _ = get_last_token_hidden(model, ids_d, n_layers)
        
        cos_ab_list.append(safe_cosine(h_a, h_b))
        cos_ac_list.append(safe_cosine(h_a, h_c))
        cos_ad_list.append(safe_cosine(h_a, h_d))
    
    # 汇总
    mean_kl_ab = float(np.mean(kl_ab_list))
    mean_kl_ac = float(np.mean(kl_ac_list))
    mean_kl_ad = float(np.mean(kl_ad_list))
    
    mean_cos_ab = float(np.mean(cos_ab_list))
    mean_cos_ac = float(np.mean(cos_ac_list))
    mean_cos_ad = float(np.mean(cos_ad_list))
    
    # 关键比值
    ratio_length = mean_kl_ac / max(mean_kl_ad, 1e-10)  # 双重否定 vs 等长肯定
    ratio_neg = mean_kl_ac / max(mean_kl_ab, 1e-10)  # 双重否定 vs 单次否定
    
    print(f"\n  ExpC Results:", flush=True)
    print(f"  KL(A||B) single negation: {mean_kl_ab:.4f}", flush=True)
    print(f"  KL(A||C) double negation: {mean_kl_ac:.4f}", flush=True)
    print(f"  KL(A||D) length control:   {mean_kl_ad:.4f}", flush=True)
    print(f"  Ratio KL(C)/KL(D): {ratio_length:.2f}", flush=True)
    print(f"  Ratio KL(C)/KL(B): {ratio_neg:.2f}", flush=True)
    print(f"  cos(A,B): {mean_cos_ab:.4f}", flush=True)
    print(f"  cos(A,C): {mean_cos_ac:.4f}", flush=True)
    print(f"  cos(A,D): {mean_cos_ad:.4f}", flush=True)
    
    # 判定
    if ratio_length > 3.0:
        verdict = f"SEMANTIC_DRIFT: Double negation KL >> length control (ratio={ratio_length:.1f})"
    elif ratio_length > 1.5:
        verdict = f"PARTIAL_LENGTH+SEMANTIC: ratio={ratio_length:.1f}"
    else:
        verdict = f"LENGTH_DOMINATED: ratio={ratio_length:.1f}, double negation ≈ length effect"
    
    print(f"\n  VERDICT: {verdict}", flush=True)
    
    return {
        "mean_kl_single_neg": mean_kl_ab,
        "mean_kl_double_neg": mean_kl_ac,
        "mean_kl_length_ctrl": mean_kl_ad,
        "ratio_double_vs_length": ratio_length,
        "ratio_double_vs_single": ratio_neg,
        "mean_cos_single_neg": mean_cos_ab,
        "mean_cos_double_neg": mean_cos_ac,
        "mean_cos_length_ctrl": mean_cos_ad,
        "n_quads": len(quads),
        "verdict": verdict,
    }


# ===== 主函数 =====

def main():
    parser = argparse.ArgumentParser(description="Phase 236: Program Geometry")
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"],
                        help="Model to test")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer sentences)")
    parser.add_argument("--large", action="store_true", help="Large test (more sentences)")
    parser.add_argument("--exp", choices=["A", "B", "C", "all"], default="all",
                        help="Which experiment to run")
    args = parser.parse_args()
    
    print(f"\n{'#'*60}", flush=True)
    print(f"Phase 236: Program Geometry — Δ Structure Analysis", flush=True)
    print(f"Model: {args.model}, Quick={args.quick}, Large={args.large}", flush=True)
    print(f"{'#'*60}\n", flush=True)
    
    # 加载模型
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(args.model)
    info = get_model_info(model, args.model)
    print(f"[init] {args.model}: {info.model_class}, {info.n_layers} layers, "
          f"d_model={info.d_model}, vocab={info.vocab_size}", flush=True)
    print(f"[init] Load time: {time.time()-t0:.1f}s", flush=True)
    
    # 数据量
    if args.quick:
        n_svd = 30
        n_cross = 20
        n_quads = 10
    elif args.large:
        n_svd = 200
        n_cross = 100
        n_quads = 30
    else:
        n_svd = 100
        n_cross = 60
        n_quads = 20
    
    results = {}
    
    try:
        # ExpA: Δ_not SVD
        if args.exp in ("A", "all"):
            results["expA"] = expA_svd_freedom(
                model, tokenizer, device, info, n_sentences=n_svd, model_name=args.model)
        
        # ExpB: 跨控制算子
        if args.exp in ("B", "all"):
            results["expB"] = expB_cross_operator(
                model, tokenizer, device, info, n_sentences=n_cross, model_name=args.model)
        
        # ExpC: 长度控制双重否定
        if args.exp in ("C", "all"):
            results["expC"] = expC_length_controlled_double_neg(
                model, tokenizer, device, info, n_quads=n_quads, model_name=args.model)
    
    finally:
        # 释放模型
        release_model(model)
    
    # 保存结果
    out_path = f"tests/glm5_temp/phase236_{args.model}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[save] Results saved to {out_path}", flush=True)
    
    # 汇总
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 236 Summary: {args.model}", flush=True)
    print(f"{'='*60}", flush=True)
    
    if "expA" in results and "error" not in results["expA"]:
        r = results["expA"]
        print(f"  ExpA (SVD): hidden k90={r['hidden_state_svd']['k90']}, "
              f"logit k90={r['logit_svd']['k90']}", flush=True)
        print(f"    HS verdict: {r['hidden_state_svd']['verdict']}", flush=True)
        print(f"    LS verdict: {r['logit_svd']['verdict']}", flush=True)
    
    if "expB" in results:
        r = results["expB"]
        for op, svd in r.get("operator_svd", {}).items():
            print(f"  ExpB ({op}): k90={svd['k90']}, top1={svd['top1_var']*100:.1f}%", flush=True)
        print(f"  ExpB verdict: {r.get('verdict', 'N/A')}", flush=True)
    
    if "expC" in results:
        r = results["expC"]
        print(f"  ExpC: KL(single)={r['mean_kl_single_neg']:.4f}, "
              f"KL(double)={r['mean_kl_double_neg']:.4f}, "
              f"KL(length_ctrl)={r['mean_kl_length_ctrl']:.4f}", flush=True)
        print(f"    ratio(double/length)={r['ratio_double_vs_length']:.2f}", flush=True)
        print(f"    verdict: {r['verdict']}", flush=True)
    
    print(f"\nPhase 236 done!", flush=True)


if __name__ == "__main__":
    main()
