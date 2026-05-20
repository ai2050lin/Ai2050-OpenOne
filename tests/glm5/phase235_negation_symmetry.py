"""
Phase 235: Negation Symmetries & Mechanisms
============================================

核心目标: 从"现象描述"转向"数学不变量"寻找

关键批判回应:
1. Value@L0 cos=1.0是数学必然 — 需测L1/L2的稳定性
2. Logit Lens的ρ(early)≈0.03可能是噪声 — 需对比否定vs形容词
3. Sparsity被softmax混淆 — 改用top-k贡献比
4. 需找不变量和对称性，而非更精细的现象描述

4个实验:
  ExpA: 双重否定对称性 — T_not ∘ T_not ≈ I ? (最关键!)
  ExpB: 门控vs加法 — 否定是乘法还是加法机制?
  ExpC: 激活修补(Activation Patching) — 因果层定位
  ExpD: 否定vs形容词对比 — Logit Lens低ρ是否否定特有?

使用方式:
  python tests/glm5/phase235_negation_symmetry.py qwen3 --quick
  python tests/glm5/phase235_negation_symmetry.py qwen3
  python tests/glm5/phase235_negation_symmetry.py qwen3 --large
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

# ===== 全局W_U缓存 =====
_W_U_CACHE = {}


# ===== 句子集合 =====

# ExpA: 双重否定三元组 (P, ¬P, ¬¬P)
DOUBLE_NEG_TRIPLETS_TYPE_A = [
    # 结构性双重否定: "It is not true that X is not Y"
    ("The cat is happy.", "The cat is not happy.", "It is not true that the cat is not happy."),
    ("The sky is blue.", "The sky is not blue.", "It is not true that the sky is not blue."),
    ("The food is delicious.", "The food is not delicious.", "It is not true that the food is not delicious."),
    ("The movie is interesting.", "The movie is not interesting.", "It is not true that the movie is not interesting."),
    ("The answer is correct.", "The answer is not correct.", "It is not true that the answer is not correct."),
    ("The plan is feasible.", "The plan is not feasible.", "It is not true that the plan is not feasible."),
    ("The result is surprising.", "The result is not surprising.", "It is not true that the result is not surprising."),
    ("The method is efficient.", "The method is not efficient.", "It is not true that the method is not efficient."),
    ("The evidence is reliable.", "The evidence is not reliable.", "It is not true that the evidence is not reliable."),
    ("The situation is dangerous.", "The situation is not dangerous.", "It is not true that the situation is not dangerous."),
    ("The door is open.", "The door is not open.", "It is not true that the door is not open."),
    ("The light is on.", "The light is not on.", "It is not true that the light is not on."),
    ("The machine is working.", "The machine is not working.", "It is not true that the machine is not working."),
    ("The system is stable.", "The system is not stable.", "It is not true that the system is not stable."),
    ("The process is complete.", "The process is not complete.", "It is not true that the process is not complete."),
]

DOUBLE_NEG_TRIPLETS_TYPE_B = [
    # 形态学双重否定: "not un-Y"
    ("The result was expected.", "The result was unexpected.", "The result was not unexpected."),
    ("The behavior was usual.", "The behavior was unusual.", "The behavior was not unusual."),
    ("The action was fair.", "The action was unfair.", "The action was not unfair."),
    ("The outcome was certain.", "The outcome was uncertain.", "The outcome was not uncertain."),
    ("The decision was reasonable.", "The decision was unreasonable.", "The decision was not unreasonable."),
    ("The request was reasonable.", "The request was unreasonable.", "The request was not unreasonable."),
    ("The approach was conventional.", "The approach was unconventional.", "The approach was not unconventional."),
    ("The belief was justified.", "The belief was unjustified.", "The belief was not unjustified."),
    ("The response was appropriate.", "The response was inappropriate.", "The response was not inappropriate."),
    ("The method was effective.", "The method was ineffective.", "The method was not ineffective."),
    ("The habit was healthy.", "The habit was unhealthy.", "The habit was not unhealthy."),
    ("The attitude was friendly.", "The attitude was unfriendly.", "The attitude was not unfriendly."),
    ("The argument was logical.", "The argument was illogical.", "The argument was not illogical."),
    ("The choice was wise.", "The choice was unwise.", "The choice was not unwise."),
    ("The feeling was comfortable.", "The feeling was uncomfortable.", "The feeling was not uncomfortable."),
]

# ExpB: 否定句对 (用于门控vs加法测试)
NEGATION_PAIRS = [
    ("The cat is happy.", "The cat is not happy."),
    ("The dog is friendly.", "The dog is not friendly."),
    ("The weather is warm.", "The weather is not warm."),
    ("The food is delicious.", "The food is not delicious."),
    ("The movie is interesting.", "The movie is not interesting."),
    ("The book is useful.", "The book is not useful."),
    ("The car is fast.", "The car is not fast."),
    ("The house is big.", "The house is not big."),
    ("The problem is simple.", "The problem is not simple."),
    ("The idea is original.", "The idea is not original."),
    ("She is beautiful.", "She is not beautiful."),
    ("He is intelligent.", "He is not intelligent."),
    ("The plan is feasible.", "The plan is not feasible."),
    ("The result is surprising.", "The result is not surprising."),
    ("The solution is obvious.", "The solution is not obvious."),
    ("The answer is correct.", "The answer is not correct."),
    ("The method is efficient.", "The method is not efficient."),
    ("The design is elegant.", "The design is not elegant."),
    ("The argument is convincing.", "The argument is not convincing."),
    ("The evidence is reliable.", "The evidence is not reliable."),
    ("The door is open.", "The door is not open."),
    ("The light is on.", "The light is not on."),
    ("The machine is working.", "The machine is not working."),
    ("The system is stable.", "The system is not stable."),
    ("The process is complete.", "The process is not complete."),
    ("The project is finished.", "The project is not finished."),
    ("The task is easy.", "The task is not easy."),
    ("The situation is dangerous.", "The situation is not dangerous."),
    ("The animal is alive.", "The animal is not alive."),
    ("The device is connected.", "The device is not connected."),
]

# ExpC: 等长句对 (用于position-aligned activation patching)
# 关键: "very"/"really" vs "not" — 相同token数, 位置完美对齐
SAME_LENGTH_PAIRS = [
    ("The cat is very happy.", "The cat is not happy."),
    ("The dog is very friendly.", "The dog is not friendly."),
    ("The weather is very warm.", "The weather is not warm."),
    ("The food is really delicious.", "The food is not delicious."),
    ("The movie is really interesting.", "The movie is not interesting."),
    ("The book is very useful.", "The book is not useful."),
    ("The car is really fast.", "The car is not fast."),
    ("The house is very big.", "The house is not big."),
    ("The problem is very simple.", "The problem is not simple."),
    ("The idea is really original.", "The idea is not original."),
    ("She is very beautiful.", "She is not beautiful."),
    ("He is really intelligent.", "He is not intelligent."),
    ("The plan is very feasible.", "The plan is not feasible."),
    ("The result is really surprising.", "The result is not surprising."),
    ("The answer is very correct.", "The answer is not correct."),
    ("The method is really efficient.", "The method is not efficient."),
    ("The door is very open.", "The door is not open."),
    ("The light is really on.", "The light is not on."),
    ("The machine is very working.", "The machine is not working."),
    ("The system is really stable.", "The system is not stable."),
]

# ExpD: 形容词句对 (用于对比否定vs内容修改)
ADJECTIVE_PAIRS = [
    ("The cat is happy.", "The cat is sad."),
    ("The dog is friendly.", "The dog is aggressive."),
    ("The weather is warm.", "The weather is cold."),
    ("The food is delicious.", "The food is terrible."),
    ("The movie is interesting.", "The movie is boring."),
    ("The car is fast.", "The car is slow."),
    ("The house is big.", "The house is small."),
    ("The problem is simple.", "The problem is complex."),
    ("She is beautiful.", "She is plain."),
    ("He is intelligent.", "He is foolish."),
    ("The plan is feasible.", "The plan is impossible."),
    ("The answer is correct.", "The answer is wrong."),
    ("The door is open.", "The door is closed."),
    ("The light is on.", "The light is off."),
    ("The machine is working.", "The machine is broken."),
]


# ===== 模型加载 (bf16 + device_map="auto", 参考model_demo_bf16.py) =====

def load_model_bf16(model_name: str):
    """BF16 + device_map="auto" 加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[bf16] Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",  # hook兼容
    )
    model.eval()
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        print(f"[bf16] {model_name}: GPU={gpu_count} components, CPU={cpu_count} components, "
              f"class={type(model).__name__}, GPU mem={gpu_mem:.2f}GB")
    else:
        print(f"[bf16] {model_name}: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    
    return model, tokenizer, device


# ===== 工具函数 =====

def get_input_device(model):
    """获取输入tensor应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logits_from_hidden_safe(model, hidden_state, model_name=None):
    """安全地从hidden state计算logits — 兼容device_map="auto"和meta tensor"""
    with torch.no_grad():
        h = hidden_state
        if not isinstance(h, torch.Tensor):
            h = torch.tensor(h, dtype=torch.float32)
        
        if h.dim() == 1:
            h = h.unsqueeze(0).unsqueeze(0)
        elif h.dim() == 2:
            h = h.unsqueeze(0)
        
        # 统一转到CPU float32计算
        try:
            h_cpu = h.detach().float().cpu()
        except (NotImplementedError, RuntimeError):
            if hasattr(h, 'numpy'):
                h_cpu = torch.tensor(h.numpy(), dtype=torch.float32)
                if h_cpu.dim() == 1:
                    h_cpu = h_cpu.unsqueeze(0).unsqueeze(0)
                elif h_cpu.dim() == 2:
                    h_cpu = h_cpu.unsqueeze(0)
            else:
                h_cpu = torch.tensor(h.tolist(), dtype=torch.float32)
                if h_cpu.dim() == 1:
                    h_cpu = h_cpu.unsqueeze(0).unsqueeze(0)
                elif h_cpu.dim() == 2:
                    h_cpu = h_cpu.unsqueeze(0)
        
        # 手动LayerNorm + W_U投影
        mean = h_cpu.mean(dim=-1, keepdim=True)
        var = h_cpu.var(dim=-1, keepdim=True, unbiased=False)
        
        try:
            norm_w = model.model.norm.weight.detach().cpu().float()
            eps = model.model.norm.eps if hasattr(model.model.norm, 'eps') else 1e-5
            norm_b = model.model.norm.bias.detach().cpu().float() if hasattr(model.model.norm, 'bias') and model.model.norm.bias is not None else torch.zeros_like(norm_w)
            normed = (h_cpu - mean) / torch.sqrt(var + eps) * norm_w + norm_b
        except Exception:
            normed = (h_cpu - mean) / torch.sqrt(var + 1e-5)
        
        # W_U投影 (使用缓存)
        model_id = id(model)
        if model_id not in _W_U_CACHE:
            _W_U_CACHE[model_id] = get_W_U(model, model_name)
        W_U = _W_U_CACHE[model_id]
        logits = normed[0, 0].numpy() @ W_U.T
        return logits


def get_final_logits(model, input_ids, attention_mask=None):
    """获取模型最终层logits"""
    input_device = get_input_device(model)
    ids = input_ids.to(input_device)
    mask = attention_mask.to(input_device) if attention_mask is not None else None
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    return out.logits[0, -1].float().cpu().numpy()


def get_hidden_states(model, input_ids, attention_mask=None):
    """获取所有层的hidden states"""
    input_device = get_input_device(model)
    ids = input_ids.to(input_device)
    mask = attention_mask.to(input_device) if attention_mask is not None else None
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    # 返回每层最后一个token的hidden state (float32 CPU numpy)
    result = []
    for hs in out.hidden_states:
        result.append(hs[0, -1].float().cpu().numpy())
    return result, out.logits[0, -1].float().cpu().numpy()


def safe_cosine(a, b):
    """安全计算余弦相似度"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_kl(p_logits, q_logits):
    """从logits计算KL散度 (先转概率)"""
    p = np.exp(p_logits - p_logits.max())
    p = p / p.sum()
    q = np.exp(q_logits - q_logits.max())
    q = q / q.sum()
    q = np.clip(q, 1e-10, 1.0)
    p = np.clip(p, 1e-10, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def safe_corr(a, b):
    """安全计算相关系数"""
    if len(a) < 3:
        return 0.0
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    na = np.std(a)
    nb = np.std(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class ProgressLogger:
    """进度日志"""
    def __init__(self, prefix="", interval=30):
        self.prefix = prefix
        self.interval = interval
        self.last_time = time.time()
        self.count = 0
    
    def update(self, msg):
        self.count += 1
        now = time.time()
        if now - self.last_time > self.interval or self.count <= 2:
            elapsed = now - self.last_time
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  {self.prefix}{msg} (GPU={gpu_mem:.1f}GB)")
            self.last_time = now
            sys.stdout.flush()


# ===== ExpA: 双重否定对称性 =====

def expA_double_negation(model, tokenizer, device, info, n_triplets=30, model_name=None):
    """
    测试: T_not ∘ T_not ≈ I ?
    
    对每个三元组 (P, ¬P, ¬¬P):
    1. 比较logit层面: KL(P||¬¬P) vs KL(P||¬P)
    2. 比较hidden state层面: cos(h_P, h_¬¬P) vs cos(h_P, h_¬P)
    3. 逐层Logit Lens: 在哪一层¬¬P开始"恢复"P?
    
    如果双重否定恢复肯定句:
      - KL(P||¬¬P) << KL(P||¬P)
      - cos(h_P, h_¬¬P) >> cos(h_P, h_¬P)
      - 逐层recovery曲线应显示¬¬P逐渐靠近P
    """
    print(f"\n{'='*60}")
    print(f"ExpA: Double Negation Symmetry (T_not ∘ T_not ≈ I ?)")
    print(f"{'='*60}")
    
    logger = ProgressLogger("ExpA: ")
    n_layers = info.n_layers
    
    # 合并Type A和Type B三元组
    all_triplets = DOUBLE_NEG_TRIPLETS_TYPE_A + DOUBLE_NEG_TRIPLETS_TYPE_B
    triplets = all_triplets[:n_triplets]
    
    # 采样层 (不全量，选关键层)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    results_by_type = {"A": [], "B": []}
    
    for ti, (affirm, negated, double_neg) in enumerate(triplets):
        logger.update(f"triplet {ti+1}/{len(triplets)}")
        
        triplet_type = "A" if ti < len(DOUBLE_NEG_TRIPLETS_TYPE_A) else "B"
        
        # 编码三个句子
        aff_ids = tokenizer(affirm, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        neg_ids = tokenizer(negated, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        dneg_ids = tokenizer(double_neg, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        
        # 获取hidden states和logits
        aff_hs, aff_logits = get_hidden_states(model, aff_ids)
        neg_hs, neg_logits = get_hidden_states(model, neg_ids)
        dneg_hs, dneg_logits = get_hidden_states(model, dneg_ids)
        
        # --- 1. Logit层面比较 ---
        kl_aff_dneg = safe_kl(aff_logits, dneg_logits)
        kl_aff_neg = safe_kl(aff_logits, neg_logits)
        corr_aff_dneg = safe_corr(aff_logits, dneg_logits)
        corr_aff_neg = safe_corr(aff_logits, neg_logits)
        
        # --- 2. Hidden state层面比较 (最后一层) ---
        cos_aff_dneg_final = safe_cosine(aff_hs[-1], dneg_hs[-1])
        cos_aff_neg_final = safe_cosine(aff_hs[-1], neg_hs[-1])
        cos_neg_dneg_final = safe_cosine(neg_hs[-1], dneg_hs[-1])
        
        # --- 3. 逐层Logit Lens比较 ---
        layer_recovery = []
        for li in sample_layers:
            if li >= len(aff_hs) or li >= len(neg_hs) or li >= len(dneg_hs):
                continue
            # Logit Lens: W_U @ h_l
            aff_logit_l = get_logits_from_hidden_safe(model, aff_hs[li], model_name)
            neg_logit_l = get_logits_from_hidden_safe(model, neg_hs[li], model_name)
            dneg_logit_l = get_logits_from_hidden_safe(model, dneg_hs[li], model_name)
            
            # Δlogit: 否定效果
            delta_neg = neg_logit_l - aff_logit_l
            delta_dneg = dneg_logit_l - aff_logit_l
            
            # 恢复度: ¬¬P的Δlogit与¬P的Δlogit的相关性
            # 如果¬¬P恢复P, 则delta_dneg应该接近0(与affirmative相同)
            # 所以corr(delta_dneg, delta_neg)应该为负(方向相反)
            recovery_corr = safe_corr(delta_dneg, delta_neg)
            
            # KL恢复度
            kl_l_aff_dneg = safe_kl(aff_logit_l, dneg_logit_l)
            kl_l_aff_neg = safe_kl(aff_logit_l, neg_logit_l)
            
            # hidden state恢复度
            cos_l_aff_dneg = safe_cosine(aff_hs[li], dneg_hs[li])
            cos_l_aff_neg = safe_cosine(aff_hs[li], neg_hs[li])
            
            layer_recovery.append({
                "layer": li,
                "recovery_corr": recovery_corr,
                "kl_ratio": kl_l_aff_dneg / max(kl_l_aff_neg, 1e-10),
                "cos_aff_dneg": cos_l_aff_dneg,
                "cos_aff_neg": cos_l_aff_neg,
            })
        
        result = {
            "affirm": affirm,
            "negated": negated,
            "double_neg": double_neg,
            "kl_aff_dneg": kl_aff_dneg,
            "kl_aff_neg": kl_aff_neg,
            "kl_ratio": kl_aff_dneg / max(kl_aff_neg, 1e-10),
            "corr_aff_dneg": corr_aff_dneg,
            "corr_aff_neg": corr_aff_neg,
            "cos_aff_dneg_final": cos_aff_dneg_final,
            "cos_aff_neg_final": cos_aff_neg_final,
            "cos_neg_dneg_final": cos_neg_dneg_final,
            "layer_recovery": layer_recovery,
        }
        results_by_type[triplet_type].append(result)
    
    # --- 汇总统计 ---
    all_results = results_by_type["A"] + results_by_type["B"]
    
    mean_kl_ratio = np.mean([r["kl_ratio"] for r in all_results])
    mean_corr_dneg = np.mean([r["corr_aff_dneg"] for r in all_results])
    mean_corr_neg = np.mean([r["corr_aff_neg"] for r in all_results])
    mean_cos_dneg = np.mean([r["cos_aff_dneg_final"] for r in all_results])
    mean_cos_neg = np.mean([r["cos_aff_neg_final"] for r in all_results])
    
    # 逐层平均恢复度
    layer_avg = defaultdict(lambda: {"recovery_corr": [], "kl_ratio": [], "cos_aff_dneg": [], "cos_aff_neg": []})
    for r in all_results:
        for lr in r["layer_recovery"]:
            li = lr["layer"]
            layer_avg[li]["recovery_corr"].append(lr["recovery_corr"])
            layer_avg[li]["kl_ratio"].append(lr["kl_ratio"])
            layer_avg[li]["cos_aff_dneg"].append(lr["cos_aff_dneg"])
            layer_avg[li]["cos_aff_neg"].append(lr["cos_aff_neg"])
    
    layer_summary = {}
    for li in sorted(layer_avg.keys()):
        layer_summary[str(li)] = {
            "recovery_corr": float(np.mean(layer_avg[li]["recovery_corr"])),
            "kl_ratio": float(np.mean(layer_avg[li]["kl_ratio"])),
            "cos_aff_dneg": float(np.mean(layer_avg[li]["cos_aff_dneg"])),
            "cos_aff_neg": float(np.mean(layer_avg[li]["cos_aff_neg"])),
        }
    
    # 判定
    if mean_kl_ratio < 0.5 and mean_cos_dneg > mean_cos_neg + 0.1:
        verdict = "STRONG_RECOVERY: 双重否定强恢复肯定句 (T_not∘T_not ≈ I)"
    elif mean_kl_ratio < 0.8 and mean_cos_dneg > mean_cos_neg:
        verdict = "PARTIAL_RECOVERY: 双重否定部分恢复肯定句"
    elif mean_kl_ratio < 1.0:
        verdict = "WEAK_RECOVERY: 双重否定弱恢复"
    else:
        verdict = "NO_RECOVERY: 双重否定不恢复肯定句 (T_not∘T_not ≠ I)"
    
    print(f"\n  ExpA Results:")
    print(f"  KL ratio (¬¬P/P vs ¬P/P): {mean_kl_ratio:.4f}")
    print(f"  Corr: aff-¬¬P={mean_corr_dneg:.4f}, aff-¬P={mean_corr_neg:.4f}")
    print(f"  Cos: aff-¬¬P={mean_cos_dneg:.4f}, aff-¬P={mean_cos_neg:.4f}")
    print(f"  Verdict: {verdict}")
    
    return {
        "n_triplets": len(all_results),
        "mean_kl_ratio": float(mean_kl_ratio),
        "mean_corr_dneg": float(mean_corr_dneg),
        "mean_corr_neg": float(mean_corr_neg),
        "mean_cos_dneg": float(mean_cos_dneg),
        "mean_cos_neg": float(mean_cos_neg),
        "type_a_kl_ratio": float(np.mean([r["kl_ratio"] for r in results_by_type["A"]])),
        "type_b_kl_ratio": float(np.mean([r["kl_ratio"] for r in results_by_type["B"]])),
        "layer_summary": layer_summary,
        "verdict": verdict,
        "individual_results": [
            {"affirm": r["affirm"], "kl_ratio": r["kl_ratio"], 
             "cos_dneg": r["cos_aff_dneg_final"], "cos_neg": r["cos_aff_neg_final"]}
            for r in all_results
        ],
    }


# ===== ExpB: 门控 vs 加法 =====

def expB_gating_vs_additive(model, tokenizer, device, info, n_pairs=30, model_name=None):
    """
    测试否定是门控(乘法)还是加法机制
    
    如果是门控: h_¬P = h_P ⊙ g(context)
      → 元素级比值 ratio = h_¬P / h_P 跨句子更稳定
    
    如果是加法: h_¬P = h_P + Δ(context)
      → 元素级差值 diff = h_¬P - h_P 跨句子更稳定
    
    关键: 在最后一层和关键中间层测量, 过滤近零元素
    """
    print(f"\n{'='*60}")
    print(f"ExpB: Gating vs Additive Mechanism")
    print(f"{'='*60}")
    
    logger = ProgressLogger("ExpB: ")
    n_layers = info.n_layers
    
    pairs = NEGATION_PAIRS[:n_pairs]
    
    # 采样层
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    sample_layers = sorted(set([l for l in sample_layers if l < n_layers]))
    
    # 收集每层的ratio和diff
    layer_ratios = defaultdict(list)  # layer -> list of ratio vectors
    layer_diffs = defaultdict(list)   # layer -> list of diff vectors
    
    for pi, (affirm, negated) in enumerate(pairs):
        logger.update(f"pair {pi+1}/{len(pairs)}")
        
        aff_ids = tokenizer(affirm, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        neg_ids = tokenizer(negated, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        
        aff_hs, _ = get_hidden_states(model, aff_ids)
        neg_hs, _ = get_hidden_states(model, neg_ids)
        
        for li in sample_layers:
            if li >= len(aff_hs) or li >= len(neg_hs):
                continue
            h_aff = aff_hs[li]
            h_neg = neg_hs[li]
            
            # 过滤近零元素 (|h_aff| > threshold)
            threshold = np.std(h_aff) * 0.01  # 1% of std
            mask = np.abs(h_aff) > threshold
            
            if mask.sum() < 10:
                continue
            
            ratio = np.zeros_like(h_aff)
            ratio[mask] = h_neg[mask] / h_aff[mask]
            diff = h_neg - h_aff
            
            layer_ratios[li].append(ratio)
            layer_diffs[li].append(diff)
    
    # 计算跨句子稳定性
    layer_stability = {}
    for li in sorted(layer_ratios.keys()):
        ratios = np.array(layer_ratios[li])  # [n_pairs, d_model]
        diffs = np.array(layer_diffs[li])     # [n_pairs, d_model]
        
        if len(ratios) < 3:
            continue
        
        # 归一化方差: var / mean^2 (coefficient of variation squared)
        ratio_mean = ratios.mean(axis=0)
        ratio_var = ratios.var(axis=0)
        diff_mean = diffs.mean(axis=0)
        diff_var = diffs.var(axis=0)
        
        # 过滤无效值
        valid_mask = (np.abs(ratio_mean) > 1e-6) & (np.abs(diff_mean) > 1e-6)
        
        if valid_mask.sum() < 10:
            continue
        
        # 归一化方差的平均值 (越低越稳定)
        ratio_cv2 = np.mean(ratio_var[valid_mask] / (ratio_mean[valid_mask]**2 + 1e-10))
        diff_cv2 = np.mean(diff_var[valid_mask] / (diff_mean[valid_mask]**2 + 1e-10))
        
        # 另一个指标: 均值向量的范数 / 方差向量的范数 (信噪比)
        ratio_snr = np.linalg.norm(ratio_mean) / (np.linalg.norm(ratio_var**0.5) + 1e-10)
        diff_snr = np.linalg.norm(diff_mean) / (np.linalg.norm(diff_var**0.5) + 1e-10)
        
        # 直接比较: ratio的跨句子cosine vs diff的跨句子cosine
        ratio_cosines = []
        diff_cosines = []
        for i in range(len(ratios)):
            for j in range(i+1, min(i+5, len(ratios))):
                ratio_cosines.append(safe_cosine(ratios[i], ratios[j]))
                diff_cosines.append(safe_cosine(diffs[i], diffs[j]))
        
        mean_ratio_cos = float(np.mean(ratio_cosines)) if ratio_cosines else 0.0
        mean_diff_cos = float(np.mean(diff_cosines)) if diff_cosines else 0.0
        
        layer_stability[str(li)] = {
            "ratio_cv2": float(ratio_cv2),
            "diff_cv2": float(diff_cv2),
            "ratio_snr": float(ratio_snr),
            "diff_snr": float(diff_snr),
            "ratio_cross_cos": mean_ratio_cos,
            "diff_cross_cos": mean_diff_cos,
            "n_pairs": len(ratios),
        }
    
    # 汇总
    all_ratio_cos = [v["ratio_cross_cos"] for v in layer_stability.values()]
    all_diff_cos = [v["diff_cross_cos"] for v in layer_stability.values()]
    
    mean_ratio_cos = float(np.mean(all_ratio_cos)) if all_ratio_cos else 0.0
    mean_diff_cos = float(np.mean(all_diff_cos)) if all_diff_cos else 0.0
    
    # 判定
    if mean_ratio_cos > mean_diff_cos + 0.05:
        verdict = "GATING: ratio更稳定 → 否定更接近门控(乘法)机制"
    elif mean_diff_cos > mean_ratio_cos + 0.05:
        verdict = "ADDITIVE: diff更稳定 → 否定更接近加法机制"
    else:
        verdict = "MIXED: ratio和diff稳定性接近 → 否定可能是混合机制"
    
    print(f"\n  ExpB Results:")
    print(f"  Cross-sentence cosine: ratio={mean_ratio_cos:.4f}, diff={mean_diff_cos:.4f}")
    for li_str, stab in sorted(layer_stability.items()):
        print(f"  L{li_str}: ratio_cos={stab['ratio_cross_cos']:.4f}, diff_cos={stab['diff_cross_cos']:.4f}")
    print(f"  Verdict: {verdict}")
    
    return {
        "mean_ratio_cos": mean_ratio_cos,
        "mean_diff_cos": mean_diff_cos,
        "layer_stability": layer_stability,
        "verdict": verdict,
    }


def get_all_position_hidden_states(model, input_ids, attention_mask=None):
    """获取所有层所有位置的hidden states (返回tensor list)"""
    input_device = get_input_device(model)
    ids = input_ids.to(input_device)
    mask = attention_mask.to(input_device) if attention_mask is not None else None
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    result = []
    for hs in out.hidden_states:
        result.append(hs[0].detach().float().cpu())  # [seq_len, d_model] tensor
    return result, out.logits[0, -1].float().cpu().numpy()


# ===== ExpC: 激活修补 (Position-Aligned Activation Patching) =====

def expC_activation_patching(model, tokenizer, device, info, n_pairs=10, model_name=None):
    """
    因果层定位: 用等长句对做position-aligned patching
    
    关键设计:
    - 使用 "X is very Y" vs "X is not Y" 等长句对
    - 只patch "very/not" 位置 (单点干预)
    - 测量哪个层在该位置的表示对否定效果最因果关键
    
    这是真正的因果测试 — 不是相关性
    """
    print(f"\n{'='*60}")
    print(f"ExpC: Position-Aligned Activation Patching")
    print(f"{'='*60}")
    
    logger = ProgressLogger("ExpC: ")
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    
    pairs = SAME_LENGTH_PAIRS[:n_pairs]
    
    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    sample_layers = sorted(set(sample_layers + [n_layers - 1]))
    
    layer_effects_attn = defaultdict(list)   # li -> list of KL for attn patch
    layer_effects_last = defaultdict(list)   # li -> list of KL for last-pos patch
    
    for pi, (control, negated) in enumerate(pairs):
        logger.update(f"pair {pi+1}/{len(pairs)}")
        
        # 编码
        ctrl_ids = tokenizer(control, return_tensors="pt", truncation=True, max_length=128)
        neg_ids = tokenizer(negated, return_tensors="pt", truncation=True, max_length=128)
        
        ctrl_input_ids = ctrl_ids["input_ids"]
        neg_input_ids = neg_ids["input_ids"]
        
        # 验证等长
        if ctrl_input_ids.shape[1] != neg_input_ids.shape[1]:
            print(f"    SKIP: length mismatch ({ctrl_input_ids.shape[1]} vs {neg_input_ids.shape[1]})")
            continue
        
        seq_len = ctrl_input_ids.shape[1]
        
        # 找到 "very/not" 位置 (两个句子不同的位置)
        ctrl_tokens = ctrl_input_ids[0].tolist()
        neg_tokens = neg_input_ids[0].tolist()
        diff_positions = [i for i in range(seq_len) if ctrl_tokens[i] != neg_tokens[i]]
        
        if len(diff_positions) == 0:
            print(f"    SKIP: no differing positions")
            continue
        
        # 取第一个不同位置作为 "operator position"
        op_pos = diff_positions[0]
        print(f"    Op position: {op_pos} ('{tokenizer.decode([ctrl_tokens[op_pos]])}' -> '{tokenizer.decode([neg_tokens[op_pos]])}')")
        
        # 获取negated的所有层hidden states
        neg_all_hs, neg_final_logits = get_all_position_hidden_states(model, neg_input_ids)
        ctrl_all_hs, ctrl_final_logits = get_all_position_hidden_states(model, ctrl_input_ids)
        
        # --- Patching策略1: 在op_pos位置patch ---
        for li in sample_layers:
            if li >= len(neg_all_hs) or li >= len(ctrl_all_hs):
                continue
            
            # 获取negated在op_pos的hidden state
            neg_h_at_pos = neg_all_hs[li][op_pos].clone()  # [d_model]
            
            # 注册hook: 在layer li的输出中,替换op_pos位置
            def make_pos_patch_hook(patch_hs, pos):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        out[0, pos, :] = patch_hs.to(out.device, out.dtype)
                        return (out,) + output[1:]
                    else:
                        out = output.clone()
                        out[0, pos, :] = patch_hs.to(out.device, out.dtype)
                        return out
                return hook
            
            hook = layers[li].register_forward_hook(make_pos_patch_hook(neg_h_at_pos, op_pos))
            
            with torch.no_grad():
                try:
                    out = model(input_ids=ctrl_input_ids.to(input_device))
                    patched_logits = out.logits[0, -1].float().cpu().numpy()
                except Exception as e:
                    print(f"    Patch L{li} op_pos failed: {e}")
                    patched_logits = None
            
            hook.remove()
            
            if patched_logits is not None:
                # 测量: patched vs control 的KL (越大说明该层该位置越因果重要)
                kl_effect = safe_kl(ctrl_final_logits, patched_logits)
                layer_effects_attn[li].append(kl_effect)
            
            # 释放
            del neg_h_at_pos
        
        # --- Patching策略2: 在last position patch ---
        last_pos = seq_len - 1
        for li in sample_layers:
            if li >= len(neg_all_hs) or li >= len(ctrl_all_hs):
                continue
            
            neg_h_at_last = neg_all_hs[li][last_pos].clone()
            
            def make_last_patch_hook(patch_hs, pos):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        out[0, pos, :] = patch_hs.to(out.device, out.dtype)
                        return (out,) + output[1:]
                    else:
                        out = output.clone()
                        out[0, pos, :] = patch_hs.to(out.device, out.dtype)
                        return out
                return hook
            
            hook = layers[li].register_forward_hook(make_last_patch_hook(neg_h_at_last, last_pos))
            
            with torch.no_grad():
                try:
                    out = model(input_ids=ctrl_input_ids.to(input_device))
                    patched_logits = out.logits[0, -1].float().cpu().numpy()
                except Exception as e:
                    patched_logits = None
            
            hook.remove()
            
            if patched_logits is not None:
                kl_effect = safe_kl(ctrl_final_logits, patched_logits)
                layer_effects_last[li].append(kl_effect)
            
            del neg_h_at_last
        
        # 释放内存
        del neg_all_hs, ctrl_all_hs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    op_pos_summary = {}
    for li in sorted(layer_effects_attn.keys()):
        effects = layer_effects_attn[li]
        op_pos_summary[str(li)] = {
            "mean_kl": float(np.mean(effects)),
            "std_kl": float(np.std(effects)),
            "n": len(effects),
        }
    
    last_pos_summary = {}
    for li in sorted(layer_effects_last.keys()):
        effects = layer_effects_last[li]
        last_pos_summary[str(li)] = {
            "mean_kl": float(np.mean(effects)),
            "std_kl": float(np.std(effects)),
            "n": len(effects),
        }
    
    # 找关键层
    if op_pos_summary:
        best_op = max(op_pos_summary.items(), key=lambda x: x[1]["mean_kl"])
    else:
        best_op = ("?", {"mean_kl": 0})
    
    if last_pos_summary:
        best_last = max(last_pos_summary.items(), key=lambda x: x[1]["mean_kl"])
    else:
        best_last = ("?", {"mean_kl": 0})
    
    print(f"\n  ExpC Results:")
    print(f"  === Operator Position Patching ===")
    for li_str, summ in sorted(op_pos_summary.items(), key=lambda x: int(x[0])):
        print(f"  L{li_str}: mean_KL={summ['mean_kl']:.4f} ± {summ['std_kl']:.4f}")
    print(f"  Most causal layer (op_pos): L{best_op[0]} (KL={best_op[1]['mean_kl']:.4f})")
    
    print(f"  === Last Position Patching ===")
    for li_str, summ in sorted(last_pos_summary.items(), key=lambda x: int(x[0])):
        print(f"  L{li_str}: mean_KL={summ['mean_kl']:.4f} ± {summ['std_kl']:.4f}")
    print(f"  Most causal layer (last_pos): L{best_last[0]} (KL={best_last[1]['mean_kl']:.4f})")
    
    return {
        "op_pos_summary": op_pos_summary,
        "last_pos_summary": last_pos_summary,
        "most_causal_op_layer": best_op[0],
        "most_causal_op_kl": best_op[1]["mean_kl"],
        "most_causal_last_layer": best_last[0],
        "most_causal_last_kl": best_last[1]["mean_kl"],
    }


# ===== ExpD: 否定 vs 形容词对比 (Logit Lens有效性检验) =====

def expD_negation_vs_adjective(model, tokenizer, device, info, n_pairs=15, model_name=None):
    """
    关键检验: Logit Lens的低ρ(early)是否否定特有?
    
    如果否定的ρ(early)和形容词的ρ(early)一样低,
    说明低ρ是Logit Lens的投影噪声,不是否定的特性。
    
    如果否定的ρ(early)显著低于形容词,说明否定确实有特殊的计算模式。
    """
    print(f"\n{'='*60}")
    print(f"ExpD: Negation vs Adjective (Logit Lens Validity Check)")
    print(f"{'='*60}")
    
    logger = ProgressLogger("ExpD: ")
    n_layers = info.n_layers
    
    neg_pairs = NEGATION_PAIRS[:n_pairs]
    adj_pairs = ADJECTIVE_PAIRS[:n_pairs]
    
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    sample_layers = sorted(set(sample_layers + [n_layers - 1]))
    
    def compute_layer_corr(pairs, label):
        """计算一组句对的逐层Δlogit相关性"""
        layer_deltas = defaultdict(list)  # li -> list of Δlogit vectors
        final_deltas = []
        
        for pi, (base, modified) in enumerate(pairs):
            logger.update(f"{label} pair {pi+1}/{len(pairs)}")
            
            base_ids = tokenizer(base, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            mod_ids = tokenizer(modified, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            
            base_hs, base_logits = get_hidden_states(model, base_ids)
            mod_hs, mod_logits = get_hidden_states(model, mod_ids)
            
            final_deltas.append(mod_logits - base_logits)
            
            for li in sample_layers:
                if li >= len(base_hs) or li >= len(mod_hs):
                    continue
                base_logit_l = get_logits_from_hidden_safe(model, base_hs[li], model_name)
                mod_logit_l = get_logits_from_hidden_safe(model, mod_hs[li], model_name)
                layer_deltas[li].append(mod_logit_l - base_logit_l)
        
        # 计算每层Δlogit与最终Δlogit的相关性
        final_delta_arr = np.array(final_deltas)  # [n_pairs, vocab]
        layer_corrs = {}
        for li in sorted(layer_deltas.keys()):
            delta_arr = np.array(layer_deltas[li])  # [n_pairs, vocab]
            # 对每个pair计算corr,然后平均
            corrs = []
            for i in range(len(delta_arr)):
                c = safe_corr(delta_arr[i], final_delta_arr[i])
                if not np.isnan(c):
                    corrs.append(c)
            layer_corrs[li] = float(np.mean(corrs)) if corrs else 0.0
        
        return layer_corrs
    
    neg_corrs = compute_layer_corr(neg_pairs, "Negation")
    adj_corrs = compute_layer_corr(adj_pairs, "Adjective")
    
    # 汇总
    early_layers = [l for l in sample_layers if l <= n_layers // 4]
    late_layers = [l for l in sample_layers if l > 3 * n_layers // 4]
    
    neg_early = np.mean([neg_corrs.get(l, 0) for l in early_layers])
    neg_late = np.mean([neg_corrs.get(l, 0) for l in late_layers])
    adj_early = np.mean([adj_corrs.get(l, 0) for l in early_layers])
    adj_late = np.mean([adj_corrs.get(l, 0) for l in late_layers])
    
    # 判定
    early_diff = adj_early - neg_early
    if early_diff > 0.1:
        verdict = f"NEGATION_SPECIAL: 否定的ρ(early)显著低于形容词 (diff={early_diff:.4f}), 低ρ是否定特有"
    elif early_diff > 0:
        verdict = f"NEGATION_WEAKLY_SPECIAL: 否定的ρ(early)略低于形容词 (diff={early_diff:.4f})"
    else:
        verdict = f"PROJECTION_NOISE: 否定和形容词的ρ(early)接近 (diff={early_diff:.4f}), 低ρ是Logit Lens投影噪声"
    
    print(f"\n  ExpD Results:")
    print(f"  Negation: ρ(early)={neg_early:.4f}, ρ(late)={neg_late:.4f}")
    print(f"  Adjective: ρ(early)={adj_early:.4f}, ρ(late)={adj_late:.4f}")
    print(f"  Early diff (adj-neg): {early_diff:.4f}")
    print(f"  Verdict: {verdict}")
    
    # 逐层对比
    layer_comparison = {}
    for li in sorted(set(list(neg_corrs.keys()) + list(adj_corrs.keys()))):
        layer_comparison[str(li)] = {
            "neg_corr": neg_corrs.get(li, 0.0),
            "adj_corr": adj_corrs.get(li, 0.0),
            "diff": adj_corrs.get(li, 0.0) - neg_corrs.get(li, 0.0),
        }
    
    return {
        "neg_early": float(neg_early),
        "neg_late": float(neg_late),
        "adj_early": float(adj_early),
        "adj_late": float(adj_late),
        "early_diff": float(early_diff),
        "layer_comparison": layer_comparison,
        "verdict": verdict,
    }


# ===== 主函数 =====

def main():
    parser = argparse.ArgumentParser(description="Phase 235: Negation Symmetries")
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--quick", action="store_true", help="Quick validation (5 sentences)")
    parser.add_argument("--large", action="store_true", help="Large test (80+ sentences)")
    args = parser.parse_args()
    
    model_name = args.model
    
    # 数据量控制
    if args.quick:
        n_triplets = 5
        n_pairs = 5
        n_patching = 3
    elif args.large:
        n_triplets = 30
        n_pairs = 30
        n_patching = 15
    else:
        n_triplets = 20
        n_pairs = 20
        n_patching = 10
    
    print(f"\n{'='*70}")
    print(f"Phase 235: Negation Symmetries & Mechanisms")
    print(f"Model: {model_name}, Mode: {'quick' if args.quick else 'large' if args.large else 'standard'}")
    print(f"Triplets: {n_triplets}, Pairs: {n_pairs}, Patching: {n_patching}")
    print(f"{'='*70}")
    
    # 加载模型
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    t_load = time.time() - t0
    print(f"Model loaded in {t_load:.1f}s: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    # 预加载W_U
    print("Pre-loading W_U matrix...")
    _W_U_CACHE[id(model)] = get_W_U(model, model_name)
    print(f"W_U loaded: shape={_W_U_CACHE[id(model)].shape}")
    
    # 运行实验
    results = {}
    
    try:
        # ExpA: 双重否定对称性 (最关键)
        results["expA"] = expA_double_negation(
            model, tokenizer, device, info, n_triplets=n_triplets, model_name=model_name)
    except Exception as e:
        print(f"ExpA FAILED: {e}")
        import traceback; traceback.print_exc()
        results["expA"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # ExpB: 门控vs加法
        results["expB"] = expB_gating_vs_additive(
            model, tokenizer, device, info, n_pairs=n_pairs, model_name=model_name)
    except Exception as e:
        print(f"ExpB FAILED: {e}")
        import traceback; traceback.print_exc()
        results["expB"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # ExpC: 激活修补
        results["expC"] = expC_activation_patching(
            model, tokenizer, device, info, n_pairs=n_patching, model_name=model_name)
    except Exception as e:
        print(f"ExpC FAILED: {e}")
        import traceback; traceback.print_exc()
        results["expC"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # ExpD: 否定vs形容词对比
        results["expD"] = expD_negation_vs_adjective(
            model, tokenizer, device, info, n_pairs=n_pairs, model_name=model_name)
    except Exception as e:
        print(f"ExpD FAILED: {e}")
        import traceback; traceback.print_exc()
        results["expD"] = {"error": str(e)}
    
    # 保存结果
    output_path = f"tests/glm5_temp/phase235_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {output_path}")
    
    # 打印摘要
    print(f"\n{'='*70}")
    print(f"Phase 235 Summary ({model_name})")
    print(f"{'='*70}")
    for exp_name, exp_result in results.items():
        if "error" in exp_result:
            print(f"  {exp_name}: ERROR - {exp_result['error'][:100]}")
        else:
            verdict = exp_result.get("verdict", "N/A")
            print(f"  {exp_name}: {verdict}")
    
    # 释放模型
    _W_U_CACHE.clear()
    release_model(model)
    
    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s")
    print("Phase 235 complete!")


if __name__ == "__main__":
    main()
