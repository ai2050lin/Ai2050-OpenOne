"""
Phase 233: 否定计算程序提取 (Negation Program Extraction)
==========================================================

核心目标: 从Phase 232的"测量效果"进入"提取程序"。
综合Phase 232两份数学分析的核心洞见:

1. ExpC的因果解读存在根本性歧义:
   - KL单调递增可能只是残差连接的信息保存,不是分布式计算
   - 修复: 增量残差分解 — 只patch δ_l = h_l - h_{l-1}

2. "条件变换T(h,context)"需要精确分解:
   - T = attention路由 × value向量语义
   - 需要测量"not"的value向量在不同上下文中的稳定性
   - 如果value稳定但attention权重变化 → 条件化来自路由

3. ExpD的zero ablation有OOD问题:
   - 修复: mean ablation (均值消融)

4. 数据量需要扩大到100+对

5个实验:
  ExpA: 增量残差分解 (Incremental Residual Decomposition)
  ExpB: Value向量稳定性 (Value Vector Stability)
  ExpC: Mean Ablation组件消融 (替换zero ablation)
  ExpD: 逐Head贡献分析 (Head-level contribution)
  ExpE: 否定-肯定探针 (Negation-Affirmation Probe)

使用方式:
  python tests/glm5/phase233_negation_program.py qwen3
  python tests/glm5/phase233_negation_program.py glm4
  python tests/glm5/phase233_negation_program.py deepseek7b
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
from typing import Dict, List, Tuple, Optional

from model_utils import (get_layers, get_model_info, get_W_U,
                          MODEL_CONFIGS, release_model)


# ===== 否定句对 (扩大到100对) =====
NEGATION_PAIRS = [
    # 1-10: 基础形容词
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
    # 11-20: 情感/评价
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
    # 21-30: 状态/存在
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
    # 31-40: 动词短语
    ("The student passed the exam.", "The student did not pass the exam."),
    ("The team won the game.", "The team did not win the game."),
    ("The company made a profit.", "The company did not make a profit."),
    ("The scientist discovered the truth.", "The scientist did not discover the truth."),
    ("The artist created a masterpiece.", "The artist did not create a masterpiece."),
    ("The pilot landed safely.", "The pilot did not land safely."),
    ("The patient recovered quickly.", "The patient did not recover quickly."),
    ("The engine started smoothly.", "The engine did not start smoothly."),
    ("The experiment succeeded.", "The experiment did not succeed."),
    ("The building collapsed.", "The building did not collapse."),
    # 41-50: 抽象概念
    ("Freedom is important.", "Freedom is not important."),
    ("Justice is fair.", "Justice is not fair."),
    ("Knowledge is power.", "Knowledge is not power."),
    ("Love is eternal.", "Love is not eternal."),
    ("Truth is objective.", "Truth is not objective."),
    ("Science is perfect.", "Science is not perfect."),
    ("Democracy is efficient.", "Democracy is not efficient."),
    ("Technology is neutral.", "Technology is not neutral."),
    ("Art is subjective.", "Art is not subjective."),
    ("History is predictable.", "History is not predictable."),
    # 51-60: 日常描述
    ("The sky is clear today.", "The sky is not clear today."),
    ("The water is cold.", "The water is not cold."),
    ("The road is safe.", "The road is not safe."),
    ("The price is reasonable.", "The price is not reasonable."),
    ("The quality is excellent.", "The quality is not excellent."),
    ("The service is good.", "The service is not good."),
    ("The location is convenient.", "The location is not convenient."),
    ("The atmosphere is pleasant.", "The atmosphere is not pleasant."),
    ("The temperature is comfortable.", "The temperature is not comfortable."),
    ("The noise is noticeable.", "The noise is not noticeable."),
    # 61-70: 专业领域
    ("The code is efficient.", "The code is not efficient."),
    ("The algorithm is optimal.", "The algorithm is not optimal."),
    ("The model is accurate.", "The model is not accurate."),
    ("The data is sufficient.", "The data is not sufficient."),
    ("The hypothesis is valid.", "The hypothesis is not valid."),
    ("The conclusion is sound.", "The conclusion is not sound."),
    ("The method is rigorous.", "The method is not rigorous."),
    ("The analysis is thorough.", "The analysis is not thorough."),
    ("The prediction is reliable.", "The prediction is not reliable."),
    ("The measurement is precise.", "The measurement is not precise."),
    # 71-80: 社会关系
    ("She is trustworthy.", "She is not trustworthy."),
    ("He is responsible.", "He is not responsible."),
    ("They are cooperative.", "They are not cooperative."),
    ("The leader is competent.", "The leader is not competent."),
    ("The teacher is patient.", "The teacher is not patient."),
    ("The doctor is experienced.", "The doctor is not experienced."),
    ("The lawyer is honest.", "The lawyer is not honest."),
    ("The manager is fair.", "The manager is not fair."),
    ("The employee is productive.", "The employee is not productive."),
    ("The student is diligent.", "The student is not diligent."),
    # 81-90: 自然描述
    ("The flower is blooming.", "The flower is not blooming."),
    ("The river is deep.", "The river is not deep."),
    ("The mountain is tall.", "The mountain is not tall."),
    ("The forest is dense.", "The forest is not dense."),
    ("The ocean is calm.", "The ocean is not calm."),
    ("The wind is strong.", "The wind is not strong."),
    ("The rain is heavy.", "The rain is not heavy."),
    ("The snow is thick.", "The snow is not thick."),
    ("The ice is solid.", "The ice is not solid."),
    ("The soil is fertile.", "The soil is not fertile."),
    # 91-100: 时间/程度
    ("The change is permanent.", "The change is not permanent."),
    ("The effect is significant.", "The effect is not significant."),
    ("The difference is obvious.", "The difference is not obvious."),
    ("The improvement is substantial.", "The improvement is not substantial."),
    ("The damage is severe.", "The damage is not severe."),
    ("The risk is high.", "The risk is not high."),
    ("The cost is acceptable.", "The cost is not acceptable."),
    ("The benefit is clear.", "The benefit is not clear."),
    ("The progress is steady.", "The progress is not steady."),
    ("The impact is positive.", "The impact is not positive."),
]

# 均值消融用的正常句子
NORMAL_SENTENCES = [
    "The cat sat on the mat.",
    "A bird flew over the house.",
    "She walked to the store.",
    "The sun set behind the hills.",
    "He read a book yesterday.",
    "The children played outside.",
    "Water flows downhill naturally.",
    "The train arrived on time.",
    "Music played softly in the background.",
    "The tree grew tall over the years.",
    "Rain fell gently on the roof.",
    "The student studied for the exam.",
    "A dog chased the ball.",
    "The wind blew through the window.",
    "They ate dinner at seven.",
    "The flower opened in the morning.",
    "Snow covered the mountain top.",
    "The car stopped at the signal.",
    "She wrote a letter to her friend.",
    "The clock struck midnight.",
]


class ProgressLogger:
    def __init__(self, interval=30):
        self.start = time.time()
        self.interval = interval
        self.last = self.start
    
    def update(self, msg=""):
        now = time.time()
        gpu_alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_res = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        elapsed = now - self.start
        print(f"[progress] {msg} | GPU: {gpu_alloc:.2f}GB alloc, {gpu_res:.2f}GB reserved | elapsed: {elapsed:.0f}s")
        self.last = now
    
    def stop(self):
        elapsed = time.time() - self.start
        print(f"[progress] Done in {elapsed:.1f}s")


def load_model_bf16_auto(model_name: str):
    """BF16 + device_map='auto' 统一加载 — 参考 model_demo_bf16.py"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bfloat16 + device_map=auto + flash_attn)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 尝试flash_attention_2, 失败回退eager
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        print("[load] flash_attention_2 enabled")
    except Exception as e:
        print(f"[load] flash_attention_2 failed ({e}), falling back to eager")
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
    
    info = get_model_info(model, model_name)
    print(f"[load] {model_name}: device={device}, GPU={gpu_mem:.2f}GB")
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    return model, tokenizer, device, info


def get_logits_from_hidden(model, hidden_state, device):
    """用LayerNorm + lm_head从hidden state得到logits"""
    with torch.no_grad():
        h = hidden_state.to(device)
        # 确保维度正确: [1, 1, d_model]
        if h.dim() == 1:
            h = h.unsqueeze(0).unsqueeze(0)
        elif h.dim() == 2:
            h = h.unsqueeze(0)
        # Apply final LayerNorm
        normed = model.model.norm(h)
        # Cast to match lm_head dtype
        lm_dtype = next(model.lm_head.parameters()).dtype
        logits = model.lm_head(normed.to(lm_dtype))
        return logits[0, 0].float().cpu().numpy()


def safe_get_weight(param):
    """安全获取权重, 处理meta device问题"""
    if param.is_meta:
        # 需要materialize - 使用dequantize或直接从原始参数获取
        # 对于device_map="auto"的模型, 权重应该可以被访问
        # 尝试直接 .data
        try:
            return param.data.cpu().float().numpy()
        except NotImplementedError:
            # 权重在meta device上, 需要加载
            return None
    return param.detach().cpu().float().numpy()


def get_weight_from_layer(layer, proj_name):
    """从层中安全获取投影权重"""
    sa = layer.self_attn
    proj = getattr(sa, proj_name, None)
    if proj is None:
        return None
    
    weight = proj.weight
    if weight.is_meta:
        # 对于meta device上的权重, 在forward pass中会被正确加载
        # 我们需要在forward时通过hook获取
        return None
    return weight.detach().cpu().float()


# ===== ExpA: 增量残差分解 =====

def expA_incremental_residual(model, tokenizer, device, info, n_pairs=60):
    """
    核心改进: 只patch每层的残差增量 δ_l = h_l - h_{l-1}
    
    这直接区分:
    - 情形A(信息累积): 某层δ_l的patch产生大KL → 该层在计算否定
    - 情形B(信息保存): 每层δ_l的patch都很小 → 否定信息只是被携带
    
    同时做cumulative patching (Phase 232方式)作为对比。
    """
    print("\n" + "="*60)
    print("ExpA: Incremental Residual Decomposition (增量残差分解)")
    print("="*60)
    
    layers = get_layers(model)
    n_layers = info.n_layers
    pairs = NEGATION_PAIRS[:n_pairs]
    
    logger = ProgressLogger()
    
    # 存储结果
    incremental_kl = defaultdict(list)   # layer -> [KL values]
    cumulative_kl = defaultdict(list)    # layer -> [KL values]
    
    for pi, (affirm, negated) in enumerate(pairs):
        if pi % 10 == 0:
            logger.update(f"ExpA pair {pi}/{len(pairs)}")
        
        aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
        neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
        
        # 收集每层hidden state
        aff_hidden = {}
        neg_hidden = {}
        
        def make_hook(store, key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    store[key] = output[0].detach()
                else:
                    store[key] = output.detach()
            return hook
        
        # 采样所有层
        sample_layers = list(range(n_layers))
        
        aff_hooks = [layers[li].register_forward_hook(make_hook(aff_hidden, f"L{li}")) for li in sample_layers]
        with torch.no_grad():
            _ = model(input_ids=aff_ids)
        for h in aff_hooks:
            h.remove()
        
        neg_hooks = [layers[li].register_forward_hook(make_hook(neg_hidden, f"L{li}")) for li in sample_layers]
        with torch.no_grad():
            _ = model(input_ids=neg_ids)
        for h in neg_hooks:
            h.remove()
        
        # 基线: 完整肯定/否定句的logits
        with torch.no_grad():
            aff_out = model(input_ids=aff_ids)
            neg_out = model(input_ids=neg_ids)
        
        aff_logits_full = aff_out.logits[0, -1].float().cpu().numpy()
        neg_logits_full = neg_out.logits[0, -1].float().cpu().numpy()
        
        aff_probs_full = np.exp(aff_logits_full - aff_logits_full.max())
        aff_probs_full /= aff_probs_full.sum() + 1e-20
        neg_probs_full = np.exp(neg_logits_full - neg_logits_full.max())
        neg_probs_full /= neg_probs_full.sum() + 1e-20
        
        baseline_kl = float(np.sum(neg_probs_full * np.log(neg_probs_full / (aff_probs_full + 1e-20) + 1e-20)))
        
        del aff_out, neg_out
        
        # 逐层做incremental和cumulative patching
        for li in sample_layers:
            key = f"L{li}"
            if key not in aff_hidden or key not in neg_hidden:
                continue
            
            aff_h = aff_hidden[key][0, -1].float()  # [d_model]
            neg_h = neg_hidden[key][0, -1].float()
            
            # === Incremental patch: 只替换δ_l ===
            if li == 0:
                # L0没有前层, δ_0 = h_0本身
                delta_neg = neg_h
                # 用肯定句跑,但L0换成否定句的h_0
                patched_h = delta_neg  # 就是否定句的h_0
            else:
                prev_key = f"L{li-1}"
                if prev_key not in aff_hidden or prev_key not in neg_hidden:
                    continue
                aff_h_prev = aff_hidden[prev_key][0, -1].float()
                neg_h_prev = neg_hidden[prev_key][0, -1].float()
                
                delta_aff = aff_h - aff_h_prev  # 肯定句在L_l的增量
                delta_neg = neg_h - neg_h_prev  # 否定句在L_l的增量
                
                # Patched: 用肯定句的累积 + 否定句的增量
                patched_h = aff_h_prev + delta_neg
            
            # 投影到logit空间 (用LayerNorm + lm_head)
            patched_logits = get_logits_from_hidden(model, patched_h, device)
            patched_probs = np.exp(patched_logits - patched_logits.max())
            patched_probs /= patched_probs.sum() + 1e-20
            
            # KL(肯定 vs patched) — patching否定增量后,输出是否变向否定?
            inc_kl = float(np.sum(patched_probs * np.log(patched_probs / (aff_probs_full + 1e-20) + 1e-20)))
            incremental_kl[li].append(inc_kl)
            
            # === Cumulative patch: 替换整个h_l (Phase 232方式) ===
            cum_logits = get_logits_from_hidden(model, neg_h, device)
            cum_probs = np.exp(cum_logits - cum_logits.max())
            cum_probs /= cum_probs.sum() + 1e-20
            
            cum_kl = float(np.sum(cum_probs * np.log(cum_probs / (aff_probs_full + 1e-20) + 1e-20)))
            cumulative_kl[li].append(cum_kl)
        
        del aff_hidden, neg_hidden
        if pi % 10 == 9:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logger.stop()
    
    # 汇总
    print(f"\n  {'Layer':<6} {'Incr_KL':<10} {'Cum_KL':<10} {'Incr/Cum':<10} {'Verdict'}")
    print("  " + "-"*55)
    
    results = {}
    for li in sorted(incremental_kl.keys()):
        inc_mean = np.mean(incremental_kl[li])
        cum_mean = np.mean(cumulative_kl[li])
        ratio = inc_mean / (cum_mean + 1e-10)
        
        if ratio > 0.5:
            verdict = "COMPUTATION (该层在计算否定)"
        elif ratio > 0.1:
            verdict = "PARTIAL (部分计算+部分携带)"
        else:
            verdict = "CARRY (主要携带早期信息)"
        
        print(f"  L{li:<4} {inc_mean:<10.4f} {cum_mean:<10.4f} {ratio:<10.4f} {verdict}")
        
        results[li] = {
            "incremental_kl": float(inc_mean),
            "cumulative_kl": float(cum_mean),
            "ratio": float(ratio),
            "n_pairs": len(incremental_kl[li]),
        }
    
    # 找计算层
    computation_layers = [li for li in sorted(results.keys()) if results[li]["ratio"] > 0.1]
    print(f"\n  >>> 否定计算层 (ratio>0.1): {computation_layers}")
    print(f"  >>> 基线KL (完整否定vs肯定): {baseline_kl:.4f}")
    
    return {
        "layer_results": {str(k): v for k, v in results.items()},
        "computation_layers": computation_layers,
        "baseline_kl": float(baseline_kl),
        "n_pairs": len(pairs),
    }


# ===== ExpB: Value向量稳定性 =====

def expB_value_vector_stability(model, tokenizer, device, info, n_pairs=50):
    """
    核心问题: "not"的条件化来自哪里?
    - Value向量 v_not = W_V · x_not 的方向?
    - Attention权重 α_not 的分布?
    
    如果value向量跨上下文稳定(cos>0.8), 但attention权重变化大 →
    条件化来自attention路由, "not"的语义是固定的。
    
    如果value向量跨上下文不稳定 → "not"的语义本身依赖上下文。
    """
    print("\n" + "="*60)
    print("ExpB: Value Vector Stability (Value向量稳定性)")
    print("="*60)
    
    layers = get_layers(model)
    n_layers = info.n_layers
    
    # 获取n_heads
    if hasattr(model.config, 'num_attention_heads'):
        n_heads = model.config.num_attention_heads
    elif hasattr(model.config, 'num_heads'):
        n_heads = model.config.num_heads
    else:
        n_heads = info.d_model // 128
    head_dim = info.d_model // n_heads
    
    print(f"  n_heads={n_heads}, head_dim={head_dim}")
    
    pairs = NEGATION_PAIRS[:n_pairs]
    logger = ProgressLogger()
    
    # 对每层每个head, 收集"not"token的value向量
    # value_vectors[layer][head] = [v_not_1, v_not_2, ...] 跨上下文
    value_vectors = defaultdict(lambda: defaultdict(list))
    attn_weights = defaultdict(lambda: defaultdict(list))  # α_not对last token
    
    for pi, (affirm, negated) in enumerate(pairs):
        if pi % 10 == 0:
            logger.update(f"ExpB pair {pi}/{len(pairs)}")
        
        neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
        
        # 找"not"token的位置
        neg_tokens = tokenizer.decode(neg_ids[0].tolist())
        
        # 提取attention输出和value projections
        # 使用output_attentions=True获取attention weights
        with torch.no_grad():
            out = model(input_ids=neg_ids, output_attentions=True, output_hidden_states=True)
        
        # attention_weights: [n_layers, batch, n_heads, seq, seq]
        if out.attentions is not None:
            for li in range(min(n_layers, len(out.attentions))):
                attn = out.attentions[li]  # [1, n_heads, seq, seq]
                # 找"not"token的位置 — 通常在倒数第2或第3个位置
                # 简化: 取"not"对last token的attention weight
                # last token attends to "not" token
                seq_len = attn.shape[-1]
                # 假设"not"在位置 seq_len-2 (倒数第二个content token)
                for not_pos in range(max(0, seq_len-4), seq_len-1):
                    for hi in range(n_heads):
                        # last token对not_pos的attention weight
                        w = attn[0, hi, -1, not_pos].float().cpu().item()
                        if w > 0.01:  # 只记录有意义的权重
                            attn_weights[li][hi].append(w)
        
        # 提取value向量 — 需要从W_V和x_not计算
        # 简化: 直接从intermediate activations提取
        # 在每层, "not"token经过W_V投影后的向量
        hidden_states = out.hidden_states  # [n_layers+1, 1, seq, d]
        
        for li in range(n_layers):
            layer = layers[li]
            sa = layer.self_attn
            
            # "not" token的hidden state (取倒数第2个位置作为近似)
            h_not = hidden_states[li][0, -2].float()  # [d_model] 输入到该层
            
            # W_V projection - 处理meta device
            W_V_param = sa.v_proj.weight
            if W_V_param.is_meta:
                # 权重在meta device, 通过hook获取v_proj输出
                continue
            
            W_V = W_V_param.detach().cpu().float()  # [d_model, d_model]
            h_not_cpu = hidden_states[li][0, -2].float().cpu()  # [d_model]
            v_proj = W_V @ h_not_cpu  # [d_model]
            
            # 拆分为per-head
            for hi in range(n_heads):
                start = hi * head_dim
                end = (hi + 1) * head_dim
                v_head = v_proj[start:end].numpy()
                value_vectors[li][hi].append(v_head)
        
        del out
        if pi % 10 == 9:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logger.stop()
    
    # 计算跨上下文cosine稳定性
    print(f"\n  Value向量跨上下文稳定性:")
    print(f"  {'Layer':<6} {'Head':<6} {'Mean_Cos':<10} {'Std_Cos':<10} {'Verdict'}")
    print("  " + "-"*50)
    
    value_stability = {}
    for li in range(n_layers):
        layer_stable_heads = 0
        for hi in range(n_heads):
            vecs = value_vectors[li][hi]
            if len(vecs) < 5:
                continue
            
            vecs_np = np.array(vecs)  # [n_contexts, head_dim]
            
            # 逐对cosine
            cosines = []
            n_sample = min(20, len(vecs_np))
            indices = np.random.choice(len(vecs_np), n_sample, replace=False) if len(vecs_np) > n_sample else range(len(vecs_np))
            sample_vecs = vecs_np[indices]
            
            for i in range(len(sample_vecs)):
                for j in range(i+1, min(i+5, len(sample_vecs))):
                    v1 = sample_vecs[i]
                    v2 = sample_vecs[j]
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos = float(np.dot(v1, v2) / (n1 * n2))
                        cosines.append(cos)
            
            if cosines:
                mean_cos = np.mean(cosines)
                std_cos = np.std(cosines)
                
                if mean_cos > 0.8:
                    verdict = "STABLE"
                    layer_stable_heads += 1
                elif mean_cos > 0.5:
                    verdict = "MODERATE"
                else:
                    verdict = "UNSTABLE"
                
                # 只打印有意义的head
                if mean_cos > 0.5 or li % 4 == 0:
                    print(f"  L{li:<4} H{hi:<4} {mean_cos:<10.4f} {std_cos:<10.4f} {verdict}")
                
                value_stability[f"L{li}_H{hi}"] = {
                    "mean_cosine": float(mean_cos),
                    "std_cosine": float(std_cos),
                    "verdict": verdict,
                }
        
        if li % 4 == 0:
            print(f"  --- L{li}: {layer_stable_heads}/{n_heads} heads with stable value vectors ---")
    
    # 找最稳定的heads
    stable_heads = {k: v for k, v in value_stability.items() if v["mean_cosine"] > 0.8}
    print(f"\n  >>> Stable value vector heads (cos>0.8): {len(stable_heads)} / {len(value_stability)} total")
    if stable_heads:
        top_stable = sorted(stable_heads.items(), key=lambda x: -x[1]["mean_cosine"])[:10]
        for k, v in top_stable:
            print(f"    {k}: cos={v['mean_cosine']:.4f}")
    
    return {
        "value_stability": value_stability,
        "n_stable_heads": len(stable_heads),
        "n_total_heads": len(value_stability),
        "n_pairs": len(pairs),
    }


# ===== ExpC: Mean Ablation组件消融 =====

def expC_mean_ablation(model, tokenizer, device, info, n_pairs=30):
    """
    修复Phase 232的zero ablation问题:
    用mean ablation (均值消融) 替代 zero out。
    
    1. 在正常句子上收集每层self_attn/MLP的平均输出
    2. 在否定实验中,把目标层输出替换为均值(而不是0)
    """
    print("\n" + "="*60)
    print("ExpC: Mean Ablation (均值消融, 修复OOD问题)")
    print("="*60)
    
    layers = get_layers(model)
    n_layers = info.n_layers
    pairs = NEGATION_PAIRS[:n_pairs]
    logger = ProgressLogger()
    
    # Step 1: 收集均值激活
    print("  Step 1: Computing mean activations on normal sentences...")
    mean_attn_outputs = {}
    mean_mlp_outputs = {}
    
    for li in range(n_layers):
        attn_outs = []
        mlp_outs = []
        
        def make_attn_hook(store):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    store.append(output[0][0, -1].detach().float().cpu())
                else:
                    store.append(output[0, -1].detach().float().cpu())
            return hook
        
        def make_mlp_hook(store):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    store.append(output[0][0, -1].detach().float().cpu())
                else:
                    store.append(output[0, -1].detach().float().cpu())
            return hook
        
        hook_a = layers[li].self_attn.register_forward_hook(make_attn_hook(attn_outs))
        hook_m = layers[li].mlp.register_forward_hook(make_mlp_hook(mlp_outs))
        
        for sent in NORMAL_SENTENCES:
            ids = tokenizer(sent, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                _ = model(input_ids=ids)
        
        hook_a.remove()
        hook_m.remove()
        
        if attn_outs:
            mean_attn_outputs[li] = torch.stack(attn_outs).mean(dim=0)  # [d_model]
        if mlp_outs:
            mean_mlp_outputs[li] = torch.stack(mlp_outs).mean(dim=0)  # [d_model]
        
        if li % 5 == 0:
            print(f"    L{li}: mean_attn_norm={mean_attn_outputs.get(li, torch.zeros(1)).norm():.2f}, "
                  f"mean_mlp_norm={mean_mlp_outputs.get(li, torch.zeros(1)).norm():.2f}")
    
    print(f"  Mean activations computed for {len(mean_attn_outputs)} layers")
    
    # Step 2: 在否定句对上测量mean ablation效果
    print("  Step 2: Measuring mean ablation effects on negation pairs...")
    
    # 选关键层 (每4层一个)
    critical_layers = list(range(0, n_layers, 4)) + [n_layers - 1]
    critical_layers = sorted(set(critical_layers))
    
    # 先测基线KL
    baseline_kls = []
    for affirm, negated in pairs[:10]:
        aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
        neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            aff_out = model(input_ids=aff_ids)
            neg_out = model(input_ids=neg_ids)
        aff_p = torch.softmax(aff_out.logits[0, -1].float(), dim=-1).cpu().numpy()
        neg_p = torch.softmax(neg_out.logits[0, -1].float(), dim=-1).cpu().numpy()
        kl = float(np.sum(neg_p * np.log(neg_p / (aff_p + 1e-20) + 1e-20)))
        baseline_kls.append(kl)
        del aff_out, neg_out
    
    mean_baseline_kl = np.mean(baseline_kls)
    print(f"  Baseline KL (affirm vs negated): {mean_baseline_kl:.4f}")
    
    # Mean ablation
    component_importance = defaultdict(list)
    
    for li in critical_layers:
        logger.update(f"ExpC layer {li}/{n_layers}")
        print(f"\n  Ablating layer {li}...")
        
        for comp_name, mean_output in [("self_attn", mean_attn_outputs.get(li)),
                                        ("mlp", mean_mlp_outputs.get(li))]:
            if mean_output is None:
                continue
            
            # Mean ablation hook
            mean_out = mean_output.to(device)
            
            def make_mean_hook(mean_val):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        # 替换为均值 (保持batch和seq维度)
                        batch_size = output[0].shape[0]
                        seq_len = output[0].shape[1]
                        mean_expanded = mean_val.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                        return (mean_expanded.to(output[0].dtype),) + output[1:]
                    else:
                        batch_size = output.shape[0]
                        seq_len = output.shape[1]
                        mean_expanded = mean_val.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                        return mean_expanded.to(output.dtype)
                return hook
            
            if comp_name == "self_attn":
                hook = layers[li].self_attn.register_forward_hook(make_mean_hook(mean_out))
            else:
                hook = layers[li].mlp.register_forward_hook(make_mean_hook(mean_out))
            
            ablated_kls = []
            for affirm, negated in pairs[:10]:
                aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
                neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    aff_out = model(input_ids=aff_ids)
                    neg_out = model(input_ids=neg_ids)
                aff_p = torch.softmax(aff_out.logits[0, -1].float(), dim=-1).cpu().numpy()
                neg_p = torch.softmax(neg_out.logits[0, -1].float(), dim=-1).cpu().numpy()
                kl = float(np.sum(neg_p * np.log(neg_p / (aff_p + 1e-20) + 1e-20)))
                ablated_kls.append(kl)
                del aff_out, neg_out
            
            hook.remove()
            
            kl_reduction = mean_baseline_kl - np.mean(ablated_kls)
            component_importance[(li, comp_name)].append(kl_reduction)
            print(f"    {comp_name} mean ablation: KL reduction = {kl_reduction:.4f}")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logger.stop()
    
    # 汇总
    print(f"\n  Component Ablation Results (Mean Ablation):")
    sorted_comps = sorted(component_importance.items(), key=lambda x: -np.mean(x[1]))
    for (li, comp), reductions in sorted_comps:
        mean_red = np.mean(reductions)
        print(f"    L{li}_{comp}: reduction={mean_red:.4f}")
    
    return {
        "component_importance": {f"L{l}_{c}": {"mean_reduction": float(np.mean(v)), 
                                                "reductions": [float(x) for x in v]}
                                for (l, c), v in component_importance.items()},
        "baseline_kl": float(mean_baseline_kl),
        "ablation_type": "mean_ablation",
        "critical_layers": critical_layers,
        "n_pairs": len(pairs),
    }


# ===== ExpD: 逐Head贡献 =====

def expD_head_contribution(model, tokenizer, device, info, n_pairs=20):
    """
    在ExpB找到的稳定value vector heads上做逐head mean ablation。
    用mean ablation替代zero out。
    """
    print("\n" + "="*60)
    print("ExpD: Head-Level Contribution Analysis")
    print("="*60)
    
    layers = get_layers(model)
    n_layers = info.n_layers
    pairs = NEGATION_PAIRS[:n_pairs]
    
    if hasattr(model.config, 'num_attention_heads'):
        n_heads = model.config.num_attention_heads
    elif hasattr(model.config, 'num_heads'):
        n_heads = model.config.num_heads
    else:
        n_heads = info.d_model // 128
    head_dim = info.d_model // n_heads
    
    logger = ProgressLogger()
    
    # 先收集每层每个head的均值输出
    print("  Computing per-head mean outputs...")
    
    # 选取关键层 (每4层)
    target_layers = list(range(0, n_layers, 4)) + [n_layers - 1]
    target_layers = sorted(set(target_layers))
    
    # 对每个目标层,收集每个head的输出均值
    head_means = {}  # (layer, head) -> mean_output [head_dim]
    
    for li in target_layers:
        # 收集o_proj的输入 (即concatenated attn_output)
        attn_outputs = []
        
        def make_hook(store):
            def hook(module, input, output):
                # input[0]是o_proj的输入: [batch, seq, n_heads*head_dim]
                if isinstance(input, tuple):
                    store.append(input[0][0, -1].detach().float().cpu())
            return hook
        
        hook = layers[li].self_attn.o_proj.register_forward_hook(make_hook(attn_outputs))
        
        for sent in NORMAL_SENTENCES[:10]:
            ids = tokenizer(sent, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                _ = model(input_ids=ids)
        
        hook.remove()
        
        if attn_outputs:
            all_outs = torch.stack(attn_outputs)  # [n_sents, n_heads*head_dim]
            for hi in range(n_heads):
                start = hi * head_dim
                end = (hi + 1) * head_dim
                head_means[(li, hi)] = all_outs[:, start:end].mean(dim=0)  # [head_dim]
    
    print(f"  Collected mean outputs for {len(head_means)} heads across {len(target_layers)} layers")
    
    # 基线KL
    baseline_kls = []
    for affirm, negated in pairs[:10]:
        aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
        neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            aff_out = model(input_ids=aff_ids)
            neg_out = model(input_ids=neg_ids)
        aff_p = torch.softmax(aff_out.logits[0, -1].float(), dim=-1).cpu().numpy()
        neg_p = torch.softmax(neg_out.logits[0, -1].float(), dim=-1).cpu().numpy()
        kl = float(np.sum(neg_p * np.log(neg_p / (aff_p + 1e-20) + 1e-20)))
        baseline_kls.append(kl)
        del aff_out, neg_out
    mean_baseline_kl = np.mean(baseline_kls)
    print(f"  Baseline KL: {mean_baseline_kl:.4f}")
    
    # Per-head mean ablation
    head_importance = {}
    
    for li in target_layers:
        logger.update(f"ExpD layer {li}, testing heads")
        
        for hi in range(n_heads):
            mean_h = head_means.get((li, hi))
            if mean_h is None:
                continue
            
            # Mean ablation: 在o_proj输入中替换head hi的部分为均值
            mean_h_dev = mean_h.to(device)
            
            def make_head_ablation_hook(head_idx, h_dim, mean_vec):
                def hook(module, input, output):
                    # input[0]: [batch, seq, n_heads*head_dim]
                    if isinstance(input, tuple):
                        attn_in = input[0]
                    else:
                        attn_in = input
                    
                    patched = attn_in.clone()
                    start = head_idx * h_dim
                    end = (head_idx + 1) * h_dim
                    # 替换last token的head输出为均值
                    patched[:, -1, start:end] = mean_vec.unsqueeze(0).expand(patched.shape[0], -1).to(patched.dtype)
                    
                    # 需要让o_proj使用patched input
                    # 但hook不能修改input... 只能修改output
                    # 简化: 在output上减去原始head贡献, 加上均值贡献
                    return output
                return hook
            
            # 更简单的方法: 直接在o_proj输出上做head-level mean ablation
            # o_proj(h) = sum_i W_o[:, i*hd:(i+1)*hd] @ h[i*hd:(i+1)*hd]
            # 我们要把head_idx的贡献替换为均值
            
            # 获取W_o - 处理meta device
            W_o_param = layers[li].self_attn.o_proj.weight
            if W_o_param.is_meta:
                # 权重在meta device, 无法做head-level ablation
                # 改用layer-level ablation
                continue
            
            W_o = W_o_param.detach().cpu().float()  # [d_model, d_model]
            W_o_head = W_o[:, hi*head_dim:(hi+1)*head_dim]  # [d_model, head_dim]
            
            # 在前向传播后, 减去原始head贡献, 加上均值贡献
            head_contributions = []
            
            def make_output_hook(w_o_h, mean_vec, store):
                def hook(module, input, output):
                    if isinstance(input, tuple):
                        attn_in = input[0][0, -1].detach().cpu().float()
                    else:
                        attn_in = input[0, -1].detach().cpu().float()
                    
                    head_out = attn_in[hi*head_dim:(hi+1)*head_dim]
                    original_contrib = w_o_h @ head_out  # [d_model] on CPU
                    mean_contrib = w_o_h @ mean_vec.cpu()  # [d_model] on CPU
                    
                    # delta = mean - original, move to output device
                    delta = (mean_contrib - original_contrib)
                    delta_dev = torch.tensor(delta, dtype=output[0].dtype if isinstance(output, tuple) else output.dtype, device=device)
                    
                    if isinstance(output, tuple):
                        patched = output[0].clone()
                        patched[:, -1, :] = patched[:, -1, :] + delta_dev.unsqueeze(0)
                        return (patched,) + output[1:]
                    else:
                        patched = output.clone()
                        patched[:, -1, :] = patched[:, -1, :] + delta_dev.unsqueeze(0)
                        return patched
                return hook
            
            hook = layers[li].self_attn.o_proj.register_forward_hook(
                make_output_hook(W_o_head, mean_h, head_contributions)
            )
            
            ablated_kls = []
            for affirm, negated in pairs[:8]:
                aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
                neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    aff_out = model(input_ids=aff_ids)
                    neg_out = model(input_ids=neg_ids)
                aff_p = torch.softmax(aff_out.logits[0, -1].float(), dim=-1).cpu().numpy()
                neg_p = torch.softmax(neg_out.logits[0, -1].float(), dim=-1).cpu().numpy()
                kl = float(np.sum(neg_p * np.log(neg_p / (aff_p + 1e-20) + 1e-20)))
                ablated_kls.append(kl)
                del aff_out, neg_out
            
            hook.remove()
            
            kl_reduction = mean_baseline_kl - np.mean(ablated_kls)
            head_importance[f"L{li}_H{hi}"] = float(kl_reduction)
            
            if hi % 8 == 7:
                print(f"    L{li} heads {hi+1}/{n_heads} done")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    logger.stop()
    
    # 汇总
    sorted_heads = sorted(head_importance.items(), key=lambda x: -x[1])
    print(f"\n  Top-10 Most Important Heads (mean ablation):")
    for k, v in sorted_heads[:10]:
        print(f"    {k}: KL reduction = {v:.4f}")
    
    # 找最不重要的 (KL reduction最负 = 消融后增强否定)
    print(f"\n  Bottom-5 Heads (ablation enhances negation):")
    for k, v in sorted_heads[-5:]:
        print(f"    {k}: KL reduction = {v:.4f}")
    
    return {
        "head_importance": head_importance,
        "baseline_kl": float(mean_baseline_kl),
        "n_heads": n_heads,
        "target_layers": target_layers,
        "n_pairs": len(pairs),
        "ablation_type": "per_head_mean_ablation",
    }


# ===== ExpE: 否定-肯定探针 =====

def expE_negation_probe(model, tokenizer, device, info, n_pairs=80):
    """
    训练线性探针区分否定/肯定hidden states。
    如果某层探针准确率高 → 该层是否定信息的显式编码层。
    如果探针方向可用于causal steering → 进一步验证因果关系。
    """
    print("\n" + "="*60)
    print("ExpE: Negation-Affirmation Linear Probe")
    print("="*60)
    
    layers = get_layers(model)
    n_layers = info.n_layers
    pairs = NEGATION_PAIRS[:n_pairs]
    logger = ProgressLogger()
    
    # 收集每层hidden states
    layer_affirm_h = defaultdict(list)  # layer -> [h_vectors]
    layer_negate_h = defaultdict(list)
    
    sample_layers = list(range(0, n_layers, 2)) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    for pi, (affirm, negated) in enumerate(pairs):
        if pi % 10 == 0:
            logger.update(f"ExpE pair {pi}/{len(pairs)}")
        
        aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
        neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            aff_out = model(input_ids=aff_ids, output_hidden_states=True)
            neg_out = model(input_ids=neg_ids, output_hidden_states=True)
        
        for li in sample_layers:
            if li < len(aff_out.hidden_states):
                aff_h = aff_out.hidden_states[li][0, -1].float().cpu().numpy()
                neg_h = neg_out.hidden_states[li][0, -1].float().cpu().numpy()
                layer_affirm_h[li].append(aff_h)
                layer_negate_h[li].append(neg_h)
        
        del aff_out, neg_out
        if pi % 20 == 19:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logger.stop()
    
    # 训练线性探针 (简单logistic regression)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    probe_results = {}
    print(f"\n  {'Layer':<6} {'Acc':<8} {'Cos_Sep':<10} {'Verdict'}")
    print("  " + "-"*40)
    
    negation_directions = {}
    
    for li in sorted(sample_layers):
        X_aff = np.array(layer_affirm_h[li])
        X_neg = np.array(layer_negate_h[li])
        
        if len(X_aff) < 10:
            continue
        
        X = np.concatenate([X_aff, X_neg], axis=0)
        y = np.concatenate([np.zeros(len(X_aff)), np.ones(len(X_neg))])
        
        # Linear probe
        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        acc = scores.mean()
        
        # 分离度: 否定/肯定中心点的cosine距离
        aff_center = X_aff.mean(axis=0)
        neg_center = X_neg.mean(axis=0)
        n1 = np.linalg.norm(aff_center)
        n2 = np.linalg.norm(neg_center)
        cos_sep = float(np.dot(aff_center, neg_center) / (n1 * n2 + 1e-20))
        
        # 否定方向
        neg_dir = neg_center - aff_center
        neg_dir_norm = neg_dir / (np.linalg.norm(neg_dir) + 1e-20)
        negation_directions[li] = neg_dir_norm
        
        if acc > 0.95:
            verdict = "STRONG ENCODING"
        elif acc > 0.8:
            verdict = "MODERATE"
        else:
            verdict = "WEAK"
        
        print(f"  L{li:<4} {acc:<8.3f} {cos_sep:<10.4f} {verdict}")
        
        probe_results[li] = {
            "accuracy": float(acc),
            "cosine_separation": float(cos_sep),
            "verdict": verdict,
            "n_samples": len(X),
        }
    
    # 找最佳编码层
    best_layers = sorted(probe_results.items(), key=lambda x: -x[1]["accuracy"])[:5]
    print(f"\n  >>> Best negation encoding layers:")
    for li, v in best_layers:
        print(f"    L{li}: acc={v['accuracy']:.3f}, cos_sep={v['cosine_separation']:.4f}")
    
    # Causal steering test: 在肯定句上加否定方向
    print(f"\n  Causal Steering Test:")
    steering_results = {}
    
    for li, neg_dir in list(negation_directions.items())[::4]:  # 每4层测一次
        beta_values = [5, 10, 20, 50]
        for beta in beta_values:
            steer_kls = []
            for affirm, negated in pairs[:10]:
                aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
                neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
                
                with torch.no_grad():
                    aff_out = model(input_ids=aff_ids, output_hidden_states=True)
                    neg_out = model(input_ids=neg_ids)
                
                # Get hidden state at layer li
                h_li = aff_out.hidden_states[li][0, -1].float().cpu().numpy()
                
                # Steer
                h_steered = h_li + beta * neg_dir
                
                # Project to logits
                steered_logits = get_logits_from_hidden(model, 
                    torch.tensor(h_steered), device)
                
                aff_logits = aff_out.logits[0, -1].float().cpu().numpy()
                neg_logits = neg_out.logits[0, -1].float().cpu().numpy()
                
                aff_probs = np.exp(aff_logits - aff_logits.max())
                aff_probs /= aff_probs.sum() + 1e-20
                steer_probs = np.exp(steered_logits - steered_logits.max())
                steer_probs /= steer_probs.sum() + 1e-20
                neg_probs = np.exp(neg_logits - neg_logits.max())
                neg_probs /= neg_probs.sum() + 1e-20
                
                # KL(steered vs affirm) — 越大越好
                kl_steer = float(np.sum(steer_probs * np.log(steer_probs / (aff_probs + 1e-20) + 1e-20)))
                
                # cosine with negated
                n1 = np.linalg.norm(steer_probs)
                n2 = np.linalg.norm(neg_probs)
                cos_neg = float(np.dot(steer_probs, neg_probs) / (n1 * n2 + 1e-20))
                
                steer_kls.append({"kl": kl_steer, "cos_with_negated": cos_neg})
                
                del aff_out, neg_out
            
            mean_kl = np.mean([x["kl"] for x in steer_kls])
            mean_cos = np.mean([x["cos_with_negated"] for x in steer_kls])
            
            steering_results[f"L{li}_b{beta}"] = {
                "mean_kl": float(mean_kl),
                "mean_cos_with_negated": float(mean_cos),
            }
            
            if beta == 20:  # 只打印一个beta
                print(f"    L{li} beta=20: KL={mean_kl:.4f}, cos_with_neg={mean_cos:.4f}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return {
        "probe_results": {str(k): v for k, v in probe_results.items()},
        "steering_results": steering_results,
        "n_pairs": len(pairs),
    }


# ===== Main =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print("="*60)
    print(f"Phase 233: Negation Program Extraction")
    print(f"Model: {model_name}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    model, tokenizer, device, info = load_model_bf16_auto(model_name)
    
    all_results = {
        "model": model_name,
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }
    
    experiments = [
        ("expA", expA_incremental_residual, 60),
        ("expB", expB_value_vector_stability, 50),
        ("expC", expC_mean_ablation, 30),
        ("expD", expD_head_contribution, 20),
        ("expE", expE_negation_probe, 80),
    ]
    
    for exp_name, exp_func, n_pairs in experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp_name}...")
        print(f"{'='*60}")
        
        t0 = time.time()
        try:
            result = exp_func(model, tokenizer, device, info, n_pairs=n_pairs)
            all_results[exp_name] = result
            print(f"  {exp_name} done in {time.time()-t0:.1f}s")
        except Exception as e:
            import traceback
            print(f"  {exp_name} ERROR: {e}")
            traceback.print_exc()
            all_results[exp_name] = {"error": str(e)}
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 保存结果
    out_path = f"tests/glm5_temp/phase233_{model_name}_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    model = None
    
    # 最终总结
    print(f"\n{'='*60}")
    print(f"Phase 233 Final Summary")
    print(f"{'='*60}")
    
    if "expA" in all_results and "error" not in all_results["expA"]:
        comp_layers = all_results["expA"].get("computation_layers", [])
        print(f"  ExpA (增量分解): 否定计算层 = {comp_layers}")
    
    if "expB" in all_results and "error" not in all_results["expB"]:
        n_stable = all_results["expB"].get("n_stable_heads", 0)
        n_total = all_results["expB"].get("n_total_heads", 0)
        print(f"  ExpB (Value稳定): {n_stable}/{n_total} heads have stable value vectors")
    
    if "expC" in all_results and "error" not in all_results["expC"]:
        ci = all_results["expC"].get("component_importance", {})
        top3 = sorted(ci.items(), key=lambda x: -x[1]["mean_reduction"])[:3]
        top3_strs = [f"{k}={v['mean_reduction']:.4f}" for k, v in top3]
        print(f"  ExpC (均值消融): Top = {top3_strs}")
    
    if "expD" in all_results and "error" not in all_results["expD"]:
        hi = all_results["expD"].get("head_importance", {})
        top3 = sorted(hi.items(), key=lambda x: -x[1])[:3]
        print(f"  ExpD (Head消融): Top heads = {[(k, f'{v:.4f}') for k, v in top3]}")
    
    if "expE" in all_results and "error" not in all_results["expE"]:
        pr = all_results["expE"].get("probe_results", {})
        if pr:
            best = max(pr.items(), key=lambda x: x[1]["accuracy"])
            print(f"  ExpE (探针): Best layer L{best[0]}, acc={best[1]['accuracy']:.3f}")
    
    print(f"\nDone! {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
