"""
Phase 237: Negation Primitive Decoding — d_not 的语义身份
==========================================================

核心目标: 从"DS7B有一个1维否定方向"到"这个方向在语义上是什么"

关键实验:
  ExpA: 提取d_not + logit空间解码 — d_not在token层面做什么?
  ExpB: 否定行为测试 — 低维是能力还是缺陷?
  ExpC: 多句型鲁棒性 — 1维结构是否跨句型成立?
  ExpD: 跨模型方向对齐 — 三模型是否共享核心方向?

使用方式:
  python tests/glm5/phase237_negation_primitive.py qwen3 --quick
  python tests/glm5/phase237_negation_primitive.py qwen3
  python tests/glm5/phase237_negation_primitive.py qwen3 --large
  python tests/glm5/phase237_negation_primitive.py qwen3 --exp A
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
        self.start_time = time.time()
    
    def update(self, msg):
        now = time.time()
        elapsed = now - self.last_time
        total = now - self.start_time
        self.last_time = now
        print(f"  {self.prefix}{msg} ({elapsed:.1f}s, total {total:.0f}s)", flush=True)


# ===== 模型加载 =====
def load_model_bf16(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bfloat16)...", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        print(f"[load] Using flash_attention_2", flush=True)
    except Exception as e:
        print(f"[load] flash_attention_2 failed ({e}), falling back to eager", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager",
        )
    model.eval()
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[load] {model_name}: device={device}, class={type(model).__name__}, "
          f"GPU mem={gpu_mem:.2f}GB", flush=True)
    
    return model, tokenizer, device


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== 句子集合 =====

# 系动词句型 ("X is Y" / "X is not Y")
COPULA_SENTENCES = [
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
    "The ocean is vast.", "The lake is calm.", "The valley is wide.",
    "The hill is steep.", "The bridge is narrow.", "The tower is high.",
    "The door is open.", "The window is closed.", "The floor is clean.",
    "The student is diligent.", "The teacher is patient.", "The doctor is experienced.",
    "The engineer is creative.", "The artist is talented.", "The musician is skilled.",
    "The writer is prolific.", "The scientist is curious.", "The lawyer is persuasive.",
    "The cake is sweet.", "The bread is fresh.", "The soup is warm.",
    "The tea is hot.", "The coffee is strong.", "The juice is cold.",
    "The fruit is ripe.", "The vegetable is fresh.", "The meat is tender.",
    "The project is successful.", "The experiment is valid.", "The theory is sound.",
    "The data is reliable.", "The evidence is convincing.", "The argument is logical.",
    "The conclusion is reasonable.", "The prediction is accurate.", "The model is complex.",
    "The system is stable.", "The process is efficient.", "The rule is clear.",
    "The law is just.", "The policy is fair.", "The standard is high.",
    "The quality is excellent.", "The performance is outstanding.", "The design is innovative.",
    "The feature is unique.", "The function is useful.", "The experience is memorable.",
    "The story is compelling.", "The character is complex.", "The plot is engaging.",
    "The message is clear.", "The purpose is noble.", "The goal is achievable.",
    "The task is challenging.", "The mission is critical.", "The vision is grand.",
    "The dream is ambitious.", "The hope is alive.", "The fear is real.",
    "The opportunity is rare.", "The possibility is endless.", "The future is bright.",
    "The past is gone.", "The present is precious.", "The moment is fleeting.",
    "The time is limited.", "The space is vast.", "The universe is infinite.",
    "The world is changing.", "The life is beautiful.", "The love is deep.",
    "The joy is pure.", "The sorrow is heavy.", "The anger is fierce.",
    "The peace is lasting.", "The balance is delicate.", "The order is stable.",
    "The chaos is overwhelming.", "The pattern is clear.", "The structure is solid.",
    "The foundation is strong.", "The building is tall.", "The room is spacious.",
    "The park is peaceful.", "The street is busy.", "The market is crowded.",
    "The shop is open.", "The restaurant is popular.", "The hotel is luxurious.",
    "The airport is modern.", "The station is busy.", "The factory is large.",
    "The farm is productive.", "The school is excellent.", "The hospital is modern.",
    "The library is quiet.", "The museum is impressive.", "The theater is elegant.",
    "The church is old.", "The temple is sacred.", "The palace is magnificent.",
]

# 行为动词句型 ("X runs fast" / "X does not run fast")
ACTION_SENTENCES = [
    "The cat runs fast.", "The dog jumps high.", "The bird flies low.",
    "The car moves slowly.", "The train travels fast.", "The boat sails smoothly.",
    "The child plays quietly.", "The man works hard.", "The woman sings beautifully.",
    "The student studies diligently.", "The teacher explains clearly.", "The doctor treats carefully.",
    "The engineer designs creatively.", "The artist paints vividly.", "The musician performs brilliantly.",
    "The writer thinks deeply.", "The scientist researches thoroughly.", "The lawyer argues persuasively.",
    "The manager organizes efficiently.", "The leader inspires consistently.", "The worker produces steadily.",
    "The athlete trains intensely.", "The player competes fiercely.", "The team cooperates smoothly.",
    "The rain falls gently.", "The wind blows strongly.", "The sun shines brightly.",
    "The river flows calmly.", "The fire burns fiercely.", "The snow melts slowly.",
    "The tree grows tall.", "The flower blooms beautifully.", "The grass spreads widely.",
    "The machine operates smoothly.", "The engine runs quietly.", "The system functions properly.",
    "The process continues steadily.", "The experiment proceeds carefully.", "The project advances quickly.",
    "The economy grows rapidly.", "The population increases slowly.", "The technology improves constantly.",
    "The society develops progressively.", "The culture evolves gradually.", "The language changes slowly.",
    "The market fluctuates wildly.", "The price rises steadily.", "The demand increases significantly.",
    "The supply decreases gradually.", "The competition intensifies constantly.", "The innovation accelerates rapidly.",
]

# 量词句型 ("All X are Y" / "Not all X are Y")
QUANTIFIER_SENTENCES = [
    "All cats are independent.", "All dogs are loyal.", "All birds are free.",
    "All fish swim.", "All trees grow.", "All flowers bloom.",
    "All children learn.", "All students study.", "All teachers teach.",
    "All doctors help.", "All engineers build.", "All artists create.",
    "All musicians play.", "All writers write.", "All scientists discover.",
    "All leaders guide.", "All workers contribute.", "All athletes compete.",
    "All rivers flow.", "All mountains stand.", "All oceans move.",
    "All stars shine.", "All planets orbit.", "All moons reflect.",
    "All seasons change.", "All days pass.", "All nights end.",
    "All stories begin.", "All journeys start.", "All adventures await.",
    "All problems have solutions.", "All questions have answers.", "All challenges bring growth.",
    "All failures teach lessons.", "All successes require effort.", "All dreams need courage.",
    "All ideas need support.", "All plans need execution.", "All goals need commitment.",
    "All friendships need trust.", "All relationships need communication.", "All partnerships need respect.",
    "All communities need cooperation.", "All societies need justice.", "All nations need peace.",
    "All economies need innovation.", "All businesses need customers.", "All markets need regulation.",
    "All systems need maintenance.", "All machines need energy.", "All processes need oversight.",
]

# 信念嵌套句型 ("I think X" / "I don't think X")
BELIEF_SENTENCES = [
    "I think the plan will work.", "I think the answer is correct.", "I think the method is sound.",
    "I think the result is valid.", "I think the theory is true.", "I think the data is accurate.",
    "I think the experiment will succeed.", "I think the project will finish on time.",
    "I think the team will win.", "I think the candidate will win the election.",
    "I think the economy will improve.", "I think the weather will be good tomorrow.",
    "I think the stock will rise.", "I think the price will fall.", "I think the demand will increase.",
    "I think the technology will advance.", "I think the situation will improve.",
    "I think the problem will be solved.", "I think the conflict will end.",
    "I think the agreement will hold.", "I think the policy will work.",
    "I think the reform will succeed.", "I think the change will be positive.",
    "I think the future will be better.", "I think the opportunity will arise.",
    "I believe the evidence is strong.", "I believe the argument is valid.",
    "I believe the conclusion is correct.", "I believe the hypothesis is supported.",
    "I believe the claim is justified.", "I believe the assertion is reasonable.",
    "I believe the proposal is feasible.", "I believe the strategy is effective.",
    "I believe the approach is sound.", "I believe the solution is practical.",
    "I believe the recommendation is wise.", "I believe the decision is right.",
    "I believe the choice is good.", "I believe the path is clear.",
    "I believe the direction is correct.", "I believe the goal is achievable.",
    "I believe the mission is important.", "I believe the purpose is worthy.",
    "I believe the cause is just.", "I believe the effort is worthwhile.",
    "I believe the investment is sound.", "I believe the risk is acceptable.",
    "I believe the reward is sufficient.", "I believe the outcome will be positive.",
]


# ===== NLI-style 否定理解行为测试 =====
NEGATION_BEHAVIOR_TESTS = [
    # 格式: (前提, 假设, 正确标签) — entailment/contradiction/neutral
    # 简单否定
    ("The cat is black.", "The cat is not black.", "contradiction"),
    ("The cat is not black.", "The cat is white.", "neutral"),
    ("The cat is not black.", "The cat is black.", "contradiction"),
    ("The sky is clear.", "The sky is not clear.", "contradiction"),
    ("The water is cold.", "The water is not cold.", "contradiction"),
    ("The door is open.", "The door is not open.", "contradiction"),
    ("The bird is small.", "The bird is not small.", "contradiction"),
    ("The food is delicious.", "The food is not delicious.", "contradiction"),
    # 双重否定
    ("The cat is black.", "It is not true that the cat is not black.", "entailment"),
    ("The cat is not black.", "It is not true that the cat is black.", "entailment"),
    ("The sky is clear.", "It is not true that the sky is not clear.", "entailment"),
    ("The water is cold.", "It is not true that the water is not cold.", "entailment"),
    # 否定+量词
    ("All birds can fly.", "Not all birds can fly.", "neutral"),
    ("All cats are black.", "No cats are black.", "contradiction"),
    ("Some birds can swim.", "No birds can swim.", "contradiction"),
    # 否定+条件
    ("If it rains, the ground gets wet.", "It does not rain.", "neutral"),
    ("If the cat is hungry, it meows.", "The cat does not meow.", "neutral"),
    # 否定作用域
    ("The cat is not black and white.", "The cat is black and not white.", "neutral"),
    ("Not the cat but the dog is fast.", "The cat is fast.", "contradiction"),
    # 更复杂的
    ("The student is diligent.", "The student is not lazy.", "entailment"),
    ("The room is not small.", "The room is large.", "neutral"),
    ("The answer is not incorrect.", "The answer is correct.", "entailment"),
    ("The method is not ineffective.", "The method is effective.", "entailment"),
    ("The result is not unimportant.", "The result is important.", "entailment"),
    # 行为动词否定
    ("The cat runs fast.", "The cat does not run fast.", "contradiction"),
    ("The dog jumps high.", "The dog does not jump high.", "contradiction"),
    ("The bird flies low.", "The bird does not fly low.", "contradiction"),
    ("The child plays quietly.", "The child does not play quietly.", "contradiction"),
    ("The man works hard.", "The man does not work hard.", "contradiction"),
    # 信念否定
    ("I think the plan will work.", "I don't think the plan will work.", "contradiction"),
    ("I believe the evidence is strong.", "I don't believe the evidence is strong.", "contradiction"),
    ("I think the answer is correct.", "I don't think the answer is correct.", "contradiction"),
    # 更多
    ("The movie is interesting.", "The movie is not boring.", "entailment"),
    ("The problem is not simple.", "The problem is complex.", "neutral"),
    ("The weather is not warm.", "The weather is cold.", "neutral"),
    ("The building is not short.", "The building is tall.", "neutral"),
    ("The river is not shallow.", "The river is deep.", "neutral"),
    ("The price is not cheap.", "The price is expensive.", "neutral"),
    ("The task is not easy.", "The task is difficult.", "neutral"),
    ("The road is not short.", "The road is long.", "neutral"),
    ("The sound is not loud.", "The sound is quiet.", "neutral"),
    ("The light is not bright.", "The light is dim.", "neutral"),
]


# ===== 核心函数 =====

def get_last_token_hidden(model, input_ids, n_layers):
    """获取最后一token在所有层的hidden state"""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple of [1, seq_len, d_model]
    result = []
    for l in range(min(n_layers + 1, len(hs))):
        h = hs[l][0, -1, :].detach().float().cpu().numpy()  # last token
        result.append(h)
    return result


def get_last_token_logit(model, input_ids):
    """获取最后一token的logit向量"""
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[0, -1, :].detach().float().cpu().numpy()
    return logits


def make_negated_copula(sent: str) -> Optional[str]:
    """系动词否定: 'X is Y' -> 'X is not Y'"""
    if " is " in sent:
        return sent.replace(" is ", " is not ", 1)
    return None


def make_negated_action(sent: str) -> Optional[str]:
    """行为动词否定: 'X runs fast' -> 'X does not run fast'"""
    # 匹配 "The X Vs Y" 格式, V是第三人称单数动词
    parts = sent.rstrip(".").split(" ")
    if len(parts) >= 4 and parts[0] in ("The", "A", "This", "That", "Every"):
        # parts[2] 是动词, 需要变成 "does not V原形"
        verb = parts[2]
        # 简单的第三人称还原
        if verb.endswith("ies"):
            base = verb[:-3] + "y"
        elif verb.endswith("es") and verb not in ("goes", "does", "has"):
            base = verb[:-2]
        elif verb.endswith("s") and not verb.endswith("ss"):
            base = verb[:-1]
        else:
            base = verb
        parts_neg = parts[:2] + ["does", "not", base] + parts[3:]
        return " ".join(parts_neg) + "."
    return None


def make_negated_quantifier(sent: str) -> Optional[str]:
    """量词否定: 'All X are Y' -> 'Not all X are Y'"""
    if sent.startswith("All "):
        return "Not " + sent[0].lower() + sent[1:]
    return None


def make_negated_belief(sent: str) -> Optional[str]:
    """信念否定: 'I think X' -> 'I don't think X'"""
    if sent.startswith("I think "):
        return "I don't think " + sent[8:]
    if sent.startswith("I believe "):
        return "I don't believe " + sent[10:]
    return None


# ===== ExpA: 提取d_not + logit空间解码 =====

def run_expA(model, tokenizer, device, model_name, n_sentences=100, logger=None):
    """
    提取否定方向d_not, 并在logit空间解码其语义
    
    步骤:
    1. 收集N对(P, ¬P)的Δ向量
    2. SVD分解, 提取top-k方向
    3. 保存d_not (hidden state空间)
    4. 投影到logit空间: logit_dir = W_U @ d_not
    5. 分析top boosted/suppressed tokens
    6. 逐层提取d_not (检测1维瓶颈)
    """
    if logger is None:
        logger = ProgressLogger("[ExpA] ")
    
    print(f"\n{'='*60}", flush=True)
    print(f"ExpA: d_not Extraction & Logit Decoding", flush=True)
    print(f"{'='*60}", flush=True)
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    # 使用系动词句型(与Phase 236一致, 保证可比性)
    sentences = COPULA_SENTENCES[:n_sentences]
    
    # 1. 收集最后一层的Δ向量 (hidden state + logit)
    logger.update(f"Collecting Δ vectors from {len(sentences)} sentence pairs...")
    
    hidden_deltas = []  # last layer hidden state Δ
    logit_deltas = []   # logit Δ
    
    # 逐层收集 (采样层)
    sample_layers = sorted(set(
        list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    ))
    layer_delta_cache = {li: [] for li in sample_layers}
    
    input_dev = get_input_device(model)
    
    for si, sent in enumerate(sentences):
        if si % 20 == 0:
            logger.update(f"  processing {si+1}/{len(sentences)}")
        
        neg_sent = make_negated_copula(sent)
        if neg_sent is None:
            continue
        
        aff_ids = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(input_dev)
        neg_ids = tokenizer(neg_sent, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(input_dev)
        
        # 获取所有层hidden states
        with torch.no_grad():
            aff_out = model(input_ids=aff_ids, output_hidden_states=True)
            neg_out = model(input_ids=neg_ids, output_hidden_states=True)
        
        aff_hs = aff_out.hidden_states
        neg_hs = neg_out.hidden_states
        
        # 最后一层Δ (hidden state)
        if len(aff_hs) > 0 and len(neg_hs) > 0:
            aff_last = aff_hs[-1][0, -1, :].detach().float().cpu().numpy()
            neg_last = neg_hs[-1][0, -1, :].detach().float().cpu().numpy()
            delta_h = neg_last - aff_last
            if np.linalg.norm(delta_h) > 1e-6:
                hidden_deltas.append(delta_h)
        
        # Logit Δ
        aff_logits = aff_out.logits[0, -1, :].detach().float().cpu().numpy()
        neg_logits = neg_out.logits[0, -1, :].detach().float().cpu().numpy()
        delta_l = neg_logits - aff_logits
        if np.linalg.norm(delta_l) > 1e-6:
            logit_deltas.append(delta_l)
        
        # 逐层Δ
        for li in sample_layers:
            if li < len(aff_hs) and li < len(neg_hs):
                aff_l = aff_hs[li][0, -1, :].detach().float().cpu().numpy()
                neg_l = neg_hs[li][0, -1, :].detach().float().cpu().numpy()
                dl = neg_l - aff_l
                if np.linalg.norm(dl) > 1e-6:
                    layer_delta_cache[li].append(dl)
        
        del aff_out, neg_out
        if si % 20 == 19:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    n_valid = len(hidden_deltas)
    logger.update(f"Collected {n_valid} valid Δ pairs")
    
    if n_valid < 10:
        return {"error": f"Insufficient data: {n_valid} pairs"}
    
    # 2. Hidden state SVD
    hidden_deltas = np.array(hidden_deltas)
    hidden_deltas_c = hidden_deltas - hidden_deltas.mean(axis=0, keepdims=True)
    U_h, S_h, Vt_h = np.linalg.svd(hidden_deltas_c, full_matrices=False)
    cumvar_h = np.cumsum(S_h**2) / np.sum(S_h**2)
    k90_h = int(np.searchsorted(cumvar_h, 0.90)) + 1
    
    # 提取top-5方向
    d_not_h = Vt_h[0]  # 1st principal direction (hidden state space)
    d2_h = Vt_h[1] if len(Vt_h) > 1 else None
    d3_h = Vt_h[2] if len(Vt_h) > 2 else None
    
    logger.update(f"Hidden SVD: k90={k90_h}, top1={cumvar_h[0]*100:.1f}%, top3={cumvar_h[min(2,len(cumvar_h)-1)]*100:.1f}%")
    
    # 3. Logit Δ SVD
    logit_deltas = np.array(logit_deltas)
    logit_deltas_c = logit_deltas - logit_deltas.mean(axis=0, keepdims=True)
    U_l, S_l, Vt_l = np.linalg.svd(logit_deltas_c, full_matrices=False)
    cumvar_l = np.cumsum(S_l**2) / np.sum(S_l**2)
    k90_l = int(np.searchsorted(cumvar_l, 0.90)) + 1
    
    d_not_l = Vt_l[0]  # 1st principal direction (logit space)
    d2_l = Vt_l[1] if len(Vt_l) > 1 else None
    d3_l = Vt_l[2] if len(Vt_l) > 2 else None
    
    logger.update(f"Logit SVD: k90={k90_l}, top1={cumvar_l[0]*100:.1f}%, top3={cumvar_l[min(2,len(cumvar_l)-1)]*100:.1f}%")
    
    # 4. 获取W_U并解码d_not的logit语义
    logger.update("Decoding d_not in logit space...")
    W_U = get_W_U(model, model_name)  # [vocab_size, d_model]
    
    # 方法1: 将hidden state的d_not投影到logit空间
    logit_direction_from_h = W_U @ d_not_h  # [vocab_size]
    
    # 方法2: 直接使用logit空间的d_not_l (已在logit空间)
    # d_not_l是logit差值的SVD方向, 已经是[vocab_size]维
    
    # Top boosted tokens (d_not使概率增大的token)
    top_boosted_idx = np.argsort(logit_direction_from_h)[-30:][::-1]
    top_suppressed_idx = np.argsort(logit_direction_from_h)[:30]
    
    top_boosted = [(tokenizer.decode([i]).strip(), float(logit_direction_from_h[i])) for i in top_boosted_idx]
    top_suppressed = [(tokenizer.decode([i]).strip(), float(logit_direction_from_h[i])) for i in top_suppressed_idx]
    
    # 对logit空间的d_not_l也做解码
    top_boosted_l = [(tokenizer.decode([i]).strip(), float(d_not_l[i])) for i in np.argsort(d_not_l)[-30:][::-1]]
    top_suppressed_l = [(tokenizer.decode([i]).strip(), float(d_not_l[i])) for i in np.argsort(d_not_l)[:30]]
    
    logger.update(f"Top-5 boosted (from hidden d_not via W_U): {top_boosted[:5]}")
    logger.update(f"Top-5 suppressed (from hidden d_not via W_U): {top_suppressed[:5]}")
    logger.update(f"Top-5 boosted (logit d_not directly): {top_boosted_l[:5]}")
    logger.update(f"Top-5 suppressed (logit d_not directly): {top_suppressed_l[:5]}")
    
    # 5. 逐层SVD + d_not提取
    logger.update("Per-layer SVD & d_not extraction...")
    per_layer_info = {}
    per_layer_d_not = {}  # 保存每层的d_not方向
    
    for li in sample_layers:
        deltas = layer_delta_cache[li]
        if len(deltas) >= 10:
            deltas = np.array(deltas)
            deltas_c = deltas - deltas.mean(axis=0, keepdims=True)
            _, S, Vt = np.linalg.svd(deltas_c, full_matrices=False)
            cv = np.cumsum(S**2) / np.sum(S**2)
            k90 = int(np.searchsorted(cv, 0.90)) + 1
            per_layer_info[str(li)] = {
                "k90": k90, "top1_var": float(cv[0]),
                "top3_var": float(cv[min(2, len(cv)-1)]),
                "n": len(deltas),
            }
            per_layer_d_not[li] = Vt[0]  # 1st principal direction
            print(f"  L{li}: k90={k90}, top1={cv[0]*100:.1f}%", flush=True)
    
    # 6. d_not的α分布 (每个句子的Δ在d_not上的投影系数)
    alphas = hidden_deltas_c @ d_not_h  # [N]
    
    # 7. 保存d_not方向向量 (用于ExpD跨模型对齐)
    # 保存: hidden d_not, logit d_not, per-layer d_not
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "n_valid": n_valid,
        "hidden_svd": {
            "k90": k90_h, "top1_var": float(cumvar_h[0]),
            "top3_var": float(cumvar_h[min(2, len(cumvar_h)-1)]),
        },
        "logit_svd": {
            "k90": k90_l, "top1_var": float(cumvar_l[0]),
            "top3_var": float(cumvar_l[min(2, len(cumvar_l)-1)]),
        },
        "top_boosted_hidden": top_boosted[:20],
        "top_suppressed_hidden": top_suppressed[:20],
        "top_boosted_logit": top_boosted_l[:20],
        "top_suppressed_logit": top_suppressed_l[:20],
        "alpha_stats": {
            "mean": float(np.mean(alphas)),
            "std": float(np.std(alphas)),
            "min": float(np.min(alphas)),
            "max": float(np.max(alphas)),
        },
        "per_layer_k90": per_layer_info,
        # 保存方向向量用于跨模型对齐
        "d_not_hidden": d_not_h.tolist(),
        "d_not_logit": d_not_l.tolist(),
        "d2_hidden": d2_h.tolist() if d2_h is not None else None,
        "d3_hidden": d3_h.tolist() if d3_h is not None else None,
        "per_layer_d_not_dim": d_model,
        "sample_layers": sample_layers,
    }


# ===== ExpB: 否定行为测试 =====

def run_expB(model, tokenizer, device, model_name, logger=None):
    """
    否定理解行为测试: DS7B低维是能力还是缺陷?
    
    使用两种互补的测试方法:
    1. 直接否定判断: 给定句子和否定句, 判断是否一致 (是/否)
    2. 蕴含判断: 给定前提和假设, 判断是否蕴含 (是/否/可能)
    """
    if logger is None:
        logger = ProgressLogger("[ExpB] ")
    
    print(f"\n{'='*60}", flush=True)
    print(f"ExpB: Negation Behavior Test", flush=True)
    print(f"{'='*60}", flush=True)
    
    input_dev = get_input_device(model)
    
    # ===== 测试1: 简单否定判断 =====
    # "The cat is black. Is the cat not black?" → No
    # "The cat is not black. Is the cat not black?" → Yes
    simple_tests = [
        # (句子, 问题, 正确答案)
        ("The cat is black.", "Is the cat not black?", "No"),
        ("The cat is not black.", "Is the cat not black?", "Yes"),
        ("The sky is clear.", "Is the sky not clear?", "No"),
        ("The water is cold.", "Is the water not cold?", "No"),
        ("The door is open.", "Is the door not open?", "No"),
        ("The bird is small.", "Is the bird not small?", "No"),
        ("The food is delicious.", "Is the food not delicious?", "No"),
        ("The student is diligent.", "Is the student not diligent?", "No"),
        ("The result is surprising.", "Is the result not surprising?", "No"),
        ("The method is efficient.", "Is the method not efficient?", "No"),
        ("The cat is not black.", "Is the cat black?", "No"),
        ("The sky is not clear.", "Is the sky clear?", "No"),
        ("The water is not cold.", "Is the water cold?", "No"),
        ("The door is not open.", "Is the door open?", "No"),
        ("The bird is not small.", "Is the bird small?", "No"),
        # 行为动词
        ("The cat runs fast.", "Does the cat not run fast?", "Yes"),
        ("The dog jumps high.", "Does the dog not jump high?", "Yes"),
        ("The man works hard.", "Does the man not work hard?", "Yes"),
        # 量词
        ("All birds can fly.", "Not all birds can fly?", "Maybe"),
        # 信念
        ("I think the plan will work.", "Don't I think the plan will work?", "Yes"),
        # 双重否定
        ("The cat is black.", "Is it not true that the cat is not black?", "Yes"),
        ("The cat is not black.", "Is it not true that the cat is black?", "Yes"),
    ]
    
    # Yes/No token IDs
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    maybe_ids = tokenizer.encode("Maybe", add_special_tokens=False)
    
    yes_id = yes_ids[0] if yes_ids else None
    no_id = no_ids[0] if no_ids else None
    maybe_id = maybe_ids[0] if maybe_ids else None
    
    logger.update(f"Yes token: {yes_id} ({tokenizer.decode([yes_id]) if yes_id else 'N/A'})")
    logger.update(f"No token: {no_id} ({tokenizer.decode([no_id]) if no_id else 'N/A'})")
    
    # ===== 测试2: 蕴含关系 (用自然格式) =====
    # "If X is true, is Y also true?" → Yes/No/Maybe
    entail_tests = [
        # 简单否定 → contradiction
        ("The cat is black.", "The cat is not black.", "No"),
        ("The sky is clear.", "The sky is not clear.", "No"),
        # 肯定 → 肯定 → entailment
        ("The cat is black.", "The cat is not white.", "Maybe"),
        ("The student is diligent.", "The student is not lazy.", "Yes"),
        ("The answer is not incorrect.", "The answer is correct.", "Yes"),
        ("The method is not ineffective.", "The method is effective.", "Yes"),
        ("The result is not unimportant.", "The result is important.", "Yes"),
        ("The movie is interesting.", "The movie is not boring.", "Yes"),
        # 否定 → 否定 → 中性
        ("The room is not small.", "The room is large.", "Maybe"),
        ("The problem is not simple.", "The problem is complex.", "Maybe"),
        ("The weather is not warm.", "The weather is cold.", "Maybe"),
    ]
    
    # ===== 综合测试 =====
    all_correct = 0
    all_total = 0
    
    # 简单否定判断
    simple_correct = 0
    simple_total = 0
    for si, (statement, question, true_answer) in enumerate(simple_tests):
        if si % 8 == 0:
            logger.update(f"  simple test {si+1}/{len(simple_tests)}")
        
        prompt = f"{statement} {question}"
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(input_dev)
        
        with torch.no_grad():
            out = model(input_ids=input_ids)
        logits = out.logits[0, -1, :].detach().float().cpu().numpy()
        
        # 比较Yes vs No的logit
        if yes_id is not None and no_id is not None:
            yes_logit = float(logits[yes_id])
            no_logit = float(logits[no_id])
            
            if true_answer == "Yes":
                pred_correct = (yes_logit > no_logit)
            elif true_answer == "No":
                pred_correct = (no_logit > yes_logit)
            else:  # Maybe
                pred_correct = True  # 不确定时都算对
            
            simple_correct += int(pred_correct)
            simple_total += 1
            all_correct += int(pred_correct)
            all_total += 1
    
    # 蕴含判断
    entail_correct = 0
    entail_total = 0
    for ei, (premise, hypothesis, true_answer) in enumerate(entail_tests):
        if ei % 5 == 0:
            logger.update(f"  entail test {ei+1}/{len(entail_tests)}")
        
        prompt = f"If \"{premise}\" is true, is \"{hypothesis}\" also true?"
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(input_dev)
        
        with torch.no_grad():
            out = model(input_ids=input_ids)
        logits = out.logits[0, -1, :].detach().float().cpu().numpy()
        
        if yes_id is not None and no_id is not None:
            yes_logit = float(logits[yes_id])
            no_logit = float(logits[no_id])
            
            if true_answer == "Yes":
                pred_correct = (yes_logit > no_logit)
            elif true_answer == "No":
                pred_correct = (no_logit > yes_logit)
            else:  # Maybe
                pred_correct = True
            
            entail_correct += int(pred_correct)
            entail_total += 1
            all_correct += int(pred_correct)
            all_total += 1
    
    simple_acc = simple_correct / max(simple_total, 1)
    entail_acc = entail_correct / max(entail_total, 1)
    overall_acc = all_correct / max(all_total, 1)
    
    logger.update(f"Simple negation: {simple_acc:.3f} ({simple_correct}/{simple_total})")
    logger.update(f"Entailment: {entail_acc:.3f} ({entail_correct}/{entail_total})")
    logger.update(f"Overall: {overall_acc:.3f} ({all_correct}/{all_total})")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "simple_accuracy": simple_acc,
        "simple_correct": simple_correct,
        "simple_total": simple_total,
        "entail_accuracy": entail_acc,
        "entail_correct": entail_correct,
        "entail_total": entail_total,
        "overall_accuracy": overall_acc,
        "overall_correct": all_correct,
        "overall_total": all_total,
        "verdict": "low_dim_is_ability" if overall_acc >= 0.6 else "low_dim_is_defect",
    }


# ===== ExpC: 多句型鲁棒性 =====

def run_expC(model, tokenizer, device, model_name, n_per_type=50, logger=None):
    """
    多句型SVD鲁棒性测试
    
    四种句型: 系动词/行为动词/量词/信念
    每种句型做SVD, 检查k90是否保持低维
    """
    if logger is None:
        logger = ProgressLogger("[ExpC] ")
    
    print(f"\n{'='*60}", flush=True)
    print(f"ExpC: Multi-Sentence-Type Robustness", flush=True)
    print(f"{'='*60}", flush=True)
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    input_dev = get_input_device(model)
    
    sentence_types = {
        "copula": (COPULA_SENTENCES[:n_per_type], make_negated_copula),
        "action": (ACTION_SENTENCES[:n_per_type], make_negated_action),
        "quantifier": (QUANTIFIER_SENTENCES[:n_per_type], make_negated_quantifier),
        "belief": (BELIEF_SENTENCES[:n_per_type], make_negated_belief),
    }
    
    type_results = {}
    
    for stype, (sents, negate_fn) in sentence_types.items():
        logger.update(f"Processing {stype} ({len(sents)} sentences)...")
        
        hidden_deltas = []
        logit_deltas = []
        # 逐层 (只看中间层和最后层)
        mid_layer = n_layers // 2
        target_layers = [0, mid_layer, n_layers - 1]
        layer_deltas = {li: [] for li in target_layers}
        
        for si, sent in enumerate(sents):
            neg_sent = negate_fn(sent)
            if neg_sent is None:
                continue
            
            aff_ids = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(input_dev)
            neg_ids = tokenizer(neg_sent, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(input_dev)
            
            with torch.no_grad():
                aff_out = model(input_ids=aff_ids, output_hidden_states=True)
                neg_out = model(input_ids=neg_ids, output_hidden_states=True)
            
            # Last layer hidden Δ
            aff_last = aff_out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
            neg_last = neg_out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
            dh = neg_last - aff_last
            if np.linalg.norm(dh) > 1e-6:
                hidden_deltas.append(dh)
            
            # Logit Δ
            aff_logits = aff_out.logits[0, -1, :].detach().float().cpu().numpy()
            neg_logits = neg_out.logits[0, -1, :].detach().float().cpu().numpy()
            dl = neg_logits - aff_logits
            if np.linalg.norm(dl) > 1e-6:
                logit_deltas.append(dl)
            
            # Per-layer Δ
            for li in target_layers:
                if li < len(aff_out.hidden_states) and li < len(neg_out.hidden_states):
                    a_l = aff_out.hidden_states[li][0, -1, :].detach().float().cpu().numpy()
                    n_l = neg_out.hidden_states[li][0, -1, :].detach().float().cpu().numpy()
                    d_l = n_l - a_l
                    if np.linalg.norm(d_l) > 1e-6:
                        layer_deltas[li].append(d_l)
            
            del aff_out, neg_out
            if si % 20 == 19:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # SVD分析
        result = {"n_valid": len(hidden_deltas)}
        
        if len(hidden_deltas) >= 10:
            hd = np.array(hidden_deltas)
            hd_c = hd - hd.mean(axis=0, keepdims=True)
            _, S, _ = np.linalg.svd(hd_c, full_matrices=False)
            cv = np.cumsum(S**2) / np.sum(S**2)
            k90 = int(np.searchsorted(cv, 0.90)) + 1
            result["hidden_k90"] = k90
            result["hidden_top1"] = float(cv[0])
            result["hidden_top3"] = float(cv[min(2, len(cv)-1)])
        
        if len(logit_deltas) >= 10:
            ld = np.array(logit_deltas)
            ld_c = ld - ld.mean(axis=0, keepdims=True)
            _, S, _ = np.linalg.svd(ld_c, full_matrices=False)
            cv = np.cumsum(S**2) / np.sum(S**2)
            k90 = int(np.searchsorted(cv, 0.90)) + 1
            result["logit_k90"] = k90
            result["logit_top1"] = float(cv[0])
            result["logit_top3"] = float(cv[min(2, len(cv)-1)])
        
        # Per-layer
        for li in target_layers:
            deltas = layer_deltas[li]
            if len(deltas) >= 10:
                dd = np.array(deltas)
                dd_c = dd - dd.mean(axis=0, keepdims=True)
                _, S, _ = np.linalg.svd(dd_c, full_matrices=False)
                cv = np.cumsum(S**2) / np.sum(S**2)
                k90 = int(np.searchsorted(cv, 0.90)) + 1
                result[f"L{li}_k90"] = k90
                result[f"L{li}_top1"] = float(cv[0])
        
        type_results[stype] = result
        logger.update(f"  {stype}: HS k90={result.get('hidden_k90','?')}, "
                      f"LS k90={result.get('logit_k90','?')}, "
                      f"n={result['n_valid']}")
    
    # 判定
    k90s = [type_results[st].get("hidden_k90", 999) for st in sentence_types]
    all_low = all(k < 10 for k in k90s)
    only_copula_low = (type_results.get("copula", {}).get("hidden_k90", 999) < 10 and
                       any(type_results[st].get("hidden_k90", 999) >= 10 
                           for st in sentence_types if st != "copula"))
    
    if all_low:
        verdict = "robust_low_dim"
    elif only_copula_low:
        verdict = "copula_only_shortcut"
    else:
        verdict = "type_dependent"
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "type_results": type_results,
        "k90_summary": {st: type_results[st].get("hidden_k90", "?") for st in sentence_types},
        "verdict": verdict,
    }


# ===== ExpD: 跨模型方向对齐 =====

def run_expD(all_results: Dict, logger=None):
    """
    跨模型方向对齐检验
    
    核心问题: DS7B的1维d_not是否与Qwen3/GLM4的top-1方向对齐?
    如果对齐 → 所有模型共享核心否定方向
    如果不对齐 → DS7B用了不同的策略
    
    注意: 这个实验不需要模型, 只需要ExpA保存的d_not向量
    但由于不同模型的d_model不同, 需要在logit空间对齐
    
    方法: 在logit空间比较方向 (使用共享token的子集)
    """
    if logger is None:
        logger = ProgressLogger("[ExpD] ")
    
    print(f"\n{'='*60}", flush=True)
    print(f"ExpD: Cross-Model Direction Alignment", flush=True)
    print(f"{'='*60}", flush=True)
    
    models = ["qwen3", "glm4", "deepseek7b"]
    
    # 收集各模型的logit d_not
    logit_d_nots = {}
    for m in models:
        if m in all_results and "expA" in all_results[m]:
            d_not_l = all_results[m]["expA"].get("d_not_logit")
            if d_not_l is not None:
                logit_d_nots[m] = np.array(d_not_l)
    
    if len(logit_d_nots) < 2:
        logger.update("Insufficient models for alignment test")
        return {"error": "Need at least 2 models with d_not_logit"}
    
    # 不同模型的vocab_size可能不同!
    # 我们只能比较相同vocab_size的模型 (Qwen3和DS7B都基于Qwen架构, 可能共享tokenizer)
    # 对于不同vocab的模型, 我们用另一种方法: 
    # 在hidden state空间通过W_U间接对齐
    
    # 先尝试直接比较 (如果vocab_size相同)
    alignment_results = {}
    model_list = list(logit_d_nots.keys())
    
    for i in range(len(model_list)):
        for j in range(i+1, len(model_list)):
            m1, m2 = model_list[i], model_list[j]
            d1 = logit_d_nots[m1]
            d2 = logit_d_nots[m2]
            
            if len(d1) == len(d2):
                # 相同vocab_size, 直接cosine
                cos_val = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10))
                alignment_results[f"{m1}_vs_{m2}"] = {
                    "cosine": cos_val,
                    "method": "direct_logit_space",
                    "vocab_match": True,
                }
                logger.update(f"  {m1} vs {m2} (same vocab): cosine={cos_val:.4f}")
            else:
                # 不同vocab_size — 用top-k token分析语义重叠
                # 比较各模型d_not中top boosted/suppressed token的语义类别
                alignment_results[f"{m1}_vs_{m2}"] = {
                    "method": "semantic_overlap",
                    "vocab_match": False,
                    "vocab1": len(d1),
                    "vocab2": len(d2),
                }
                logger.update(f"  {m1} vs {m2} (different vocab: {len(d1)} vs {len(d2)}): "
                              "will use semantic overlap analysis")
    
    # Hidden state d_not对齐 — 通过W_U投影
    # 如果两个模型的W_U投影后的logit方向相似, 则hidden d_not语义对齐
    # 这个在ExpA中已经做了(W_U @ d_not_h = logit_direction)
    # 所以我们比较各模型的logit_direction_from_hidden
    
    hidden_logit_dirs = {}
    for m in models:
        if m in all_results and "expA" in all_results[m]:
            expA = all_results[m]["expA"]
            # 我们需要重新计算 W_U @ d_not_h
            # 但W_U在模型释放后不可用
            # 所以直接用logit d_not (更直接)
            d_not_l = expA.get("d_not_logit")
            if d_not_l is not None:
                hidden_logit_dirs[m] = np.array(d_not_l)
    
    # 综合判定
    cosines = [v["cosine"] for v in alignment_results.values() if "cosine" in v]
    if cosines:
        max_cos = max(cosines)
        mean_cos = np.mean(cosines)
        if max_cos > 0.7:
            verdict = "shared_core_direction"
        elif mean_cos > 0.4:
            verdict = "partially_shared"
        else:
            verdict = "model_specific_directions"
    else:
        verdict = "vocab_mismatch_need_semantic_analysis"
    
    logger.update(f"Verdict: {verdict}")
    
    return {
        "alignment_results": alignment_results,
        "verdict": verdict,
    }


# ===== 主函数 =====

def main():
    parser = argparse.ArgumentParser(description="Phase 237: Negation Primitive Decoding")
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--quick", action="store_true", help="Quick test (30 sentences)")
    parser.add_argument("--large", action="store_true", help="Large test (200 sentences)")
    parser.add_argument("--exp", choices=["A", "B", "C", "D"], help="Run only one experiment")
    args = parser.parse_args()
    
    model_name = args.model
    n_sentences = 30 if args.quick else (200 if args.large else 100)
    n_per_type = 15 if args.quick else (80 if args.large else 50)
    
    print(f"\n{'#'*60}", flush=True)
    print(f"Phase 237: Negation Primitive Decoding", flush=True)
    print(f"Model: {model_name}, n_sentences={n_sentences}", flush=True)
    print(f"{'#'*60}", flush=True)
    
    # 加载模型
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    print(f"[main] Model loaded in {t_load:.1f}s", flush=True)
    
    info = get_model_info(model, model_name)
    print(f"[main] {info.model_class}, {info.n_layers} layers, d_model={info.d_model}, vocab={info.vocab_size}", flush=True)
    
    results = {}
    logger = ProgressLogger()
    
    # ExpA: d_not提取与logit解码
    if args.exp is None or args.exp == "A":
        results["expA"] = run_expA(model, tokenizer, device, model_name, n_sentences, logger)
    
    # ExpB: 否定行为测试
    if args.exp is None or args.exp == "B":
        results["expB"] = run_expB(model, tokenizer, device, model_name, logger)
    
    # ExpC: 多句型鲁棒性
    if args.exp is None or args.exp == "C":
        results["expC"] = run_expC(model, tokenizer, device, model_name, n_per_type, logger)
    
    # 保存结果
    out_path = f"tests/glm5_temp/phase237_{model_name}_results.json"
    if args.quick:
        out_path = out_path.replace("_results.json", "_quick_results.json")
    if args.large:
        out_path = out_path.replace("_results.json", "_large_results.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[main] Results saved to {out_path}", flush=True)
    
    # ExpD需要跨模型数据, 在所有模型测试完后单独运行
    if args.exp == "D":
        # 加载所有已有结果
        all_results = {}
        for m in ["qwen3", "glm4", "deepseek7b"]:
            try:
                r = json.load(open(f"tests/glm5_temp/phase237_{m}_results.json", encoding="utf-8"))
                all_results[m] = r
            except:
                print(f"[ExpD] Warning: No results for {m}", flush=True)
        
        results["expD"] = run_expD(all_results, logger)
        
        # 重新保存
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{'#'*60}", flush=True)
    print(f"Phase 237 complete for {model_name}!", flush=True)
    print(f"{'#'*60}", flush=True)


if __name__ == "__main__":
    main()
