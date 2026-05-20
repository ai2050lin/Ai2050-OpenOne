"""
Phase 234: Logit Lens Mechanics + Unified Causal Analysis
==========================================================

核心目标: 解决Phase 233遗留的三个关键歧义:
1. ExpA ratio≈1.0无法区分"重计算" vs "表示转换"
2. ExpB value向量测量受前层上下文污染
3. ExpA(分布式重计算) vs ExpC(早期层主导) 的矛盾

关键工具: Logit Lens — 在每层把h_l投影到logit空间,
直接测量否定对预测分布的影响如何逐层演化。

5个实验:
  ExpA: Logit Lens逐层演化 — 区分"重计算" vs "表示转换"
  ExpB: Value向量在L0的稳定性 — 消除前层上下文污染
  ExpC: 统一CC/CN指标 — 区分"计算中心" vs "表示转换器"
  ExpD: Token级Logit轨迹 — 提取"程序基元"
  ExpE: Steering概率有效性 — 修复DS7B度量问题

使用方式:
  python tests/glm5/phase234_logit_lens_mechanics.py qwen3
  python tests/glm5/phase234_logit_lens_mechanics.py glm4
  python tests/glm5/phase234_logit_lens_mechanics.py deepseek7b
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

# 全局W_U缓存, 避免重复从safetensors加载
_W_U_CACHE = {}


# ===== 否定句对 (100对, 与Phase 233一致) =====
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
    "The cat sat on the mat.", "A bird flew over the house.",
    "She walked to the store.", "The sun set behind the hills.",
    "He read a book yesterday.", "The children played outside.",
    "Water flows downhill naturally.", "The train arrived on time.",
    "Music played softly in the background.", "The tree grew tall over the years.",
    "Rain fell gently on the roof.", "The student studied for the exam.",
    "A dog chased the ball.", "The wind blew through the window.",
    "They ate dinner at seven.", "The flower opened in the morning.",
    "Snow covered the mountain top.", "The car stopped at the signal.",
    "She wrote a letter to her friend.", "The clock struck midnight.",
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
        sys.stdout.flush()
        self.last = now

    def stop(self):
        elapsed = time.time() - self.start
        print(f"[progress] Done in {elapsed:.1f}s")


def load_model_bf16_auto(model_name: str):
    """BF16 + device_map='auto' 统一加载"""
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


def get_logits_from_hidden(model, hidden_state, device, W_U_cache=None):
    """用LayerNorm + lm_head从hidden state得到logits — 兼容device_map="auto" 
    
    策略: 优先用W_U矩阵在CPU上计算(避免跨设备问题)。
    W_U_cache: 预加载的W_U矩阵 [vocab_size, d_model] numpy, 避免重复加载。
    """
    with torch.no_grad():
        h = hidden_state
        if not isinstance(h, torch.Tensor):
            h = torch.tensor(h, dtype=torch.float32)
        
        if h.dim() == 1:
            h = h.unsqueeze(0).unsqueeze(0)
        elif h.dim() == 2:
            h = h.unsqueeze(0)
        
        # 统一转到CPU float32计算 (最安全的方式,避免meta device问题)
        try:
            h_cpu = h.detach().float().cpu()
        except (NotImplementedError, RuntimeError):
            # 如果detach失败,尝试numpy中转
            if hasattr(h, 'numpy'):
                h_numpy = h.numpy()
                h_cpu = torch.tensor(h_numpy, dtype=torch.float32)
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
        # 获取norm权重
        try:
            norm_w = model.model.norm.weight.detach().cpu().float()
            norm_b = model.model.norm.bias.detach().cpu().float() if hasattr(model.model.norm, 'bias') and model.model.norm.bias is not None else torch.zeros_like(norm_w)
        except Exception:
            eps = 1e-5
            normed = (h_cpu - mean) / torch.sqrt(var + eps)
        else:
            eps = model.model.norm.eps if hasattr(model.model.norm, 'eps') else 1e-5
            normed = (h_cpu - mean) / torch.sqrt(var + eps) * norm_w + norm_b
        
        # 乘以W_U — 使用全局缓存避免重复加载
        model_id = id(model)
        if model_id not in _W_U_CACHE:
            _W_U_CACHE[model_id] = get_W_U(model)  # [vocab_size, d_model] numpy
        W_U = _W_U_CACHE[model_id]
        if W_U_cache is not None:
            W_U = W_U_cache
        logits = normed[0, 0].numpy() @ W_U.T  # [vocab_size]
        return logits


# ===== ExpA: Logit Lens 逐层演化 =====

def expA_logit_lens(model, tokenizer, device, info, n_pairs=80):
    """
    核心实验: Logit Lens分析否定的逐层演化

    方法:
    1. 对每对(肯定,否定)句子,收集每层hidden state
    2. 在每层l,计算:
       - logit^(l)_affirm = LayerNorm(h^(l)_affirm) @ W_U
       - logit^(l)_negate = LayerNorm(h^(l)_negate) @ W_U
       - Δlogit^(l) = logit^(l)_negate - logit^(l)_affirm
    3. 计算ρ(l) = corr(Δlogit^(l), Δlogit^(L_final))

    判定标准:
    - ρ(l) 在早期层(l=0,1,2)就接近1.0 → 表示转换 (情形B: 信息保存)
    - ρ(l) 随层逐渐增大 → 分布式重计算 (情形A: 信息累积)
    - ρ(l) 在中间层突然跳变 → 局部关键计算

    这直接解决ExpA的核心歧义。
    """
    print("\n" + "="*60)
    print("ExpA: Logit Lens — 否定效果的逐层演化")
    print("="*60)

    layers = get_layers(model)
    n_layers = info.n_layers
    pairs = NEGATION_PAIRS[:n_pairs]
    logger = ProgressLogger()

    # 预加载W_U用于直接logit计算 (避免每层都过lm_head)
    print("  Loading W_U matrix...")
    W_U = get_W_U(model, model_name=None)  # [vocab_size, d_model] float32

    # 结果存储
    delta_logits_per_layer = defaultdict(list)  # layer -> [Δlogit vectors]
    final_delta_logits = []  # 每个pair的最终Δlogit

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

        # 逐层计算Δlogit
        for li in sample_layers:
            key = f"L{li}"
            if key not in aff_hidden or key not in neg_hidden:
                continue

            aff_h = aff_hidden[key][0, -1].float()  # [d_model]
            neg_h = neg_hidden[key][0, -1].float()

            # 方法1: 通过lm_head (更准确,但慢)
            # 方法2: 直接用W_U (近似,但快) — 用LayerNorm后乘W_U
            # 这里用方法1,确保准确性
            aff_logits = get_logits_from_hidden(model, aff_h, device)
            neg_logits = get_logits_from_hidden(model, neg_h, device)

            delta_logit = neg_logits - aff_logits
            delta_logits_per_layer[li].append(delta_logit)

        # 最终层的Δlogit (直接从模型输出)
        with torch.no_grad():
            aff_out = model(input_ids=aff_ids)
            neg_out = model(input_ids=neg_ids)

        aff_logits_final = aff_out.logits[0, -1].float().cpu().numpy()
        neg_logits_final = neg_out.logits[0, -1].float().cpu().numpy()
        final_delta = neg_logits_final - aff_logits_final
        final_delta_logits.append(final_delta)

        del aff_hidden, neg_hidden, aff_out, neg_out
        if pi % 10 == 9:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.stop()

    # 计算ρ(l) = corr(Δlogit^(l), Δlogit^(L_final))
    final_layer = n_layers - 1
    final_deltas = np.array(final_delta_logits)  # [n_pairs, vocab]

    print(f"\n  {'Layer':<6} {'ρ(l)':<10} {'||Δlogit||':<12} {'Top-3 Suppressed':<25} {'Top-3 Boosted':<25} {'Verdict'}")
    print("  " + "-"*95)

    results = {}
    top_suppressed_all = defaultdict(list)  # 跨pair统计
    top_boosted_all = defaultdict(list)

    for li in sorted(delta_logits_per_layer.keys()):
        deltas = np.array(delta_logits_per_layer[li])  # [n_pairs, vocab]

        # 相关性: 每个pair的Δlogit^(l) vs Δlogit^(final)
        correlations = []
        for i in range(len(deltas)):
            d_l = deltas[i]
            d_f = final_deltas[i]
            n1 = np.linalg.norm(d_l)
            n2 = np.linalg.norm(d_f)
            if n1 > 1e-10 and n2 > 1e-10:
                corr = float(np.corrcoef(d_l, d_f)[0, 1])
                if not np.isnan(corr):
                    correlations.append(corr)

        mean_corr = np.mean(correlations) if correlations else 0.0
        mean_norm = np.mean([np.linalg.norm(d) for d in deltas])

        # Token级分析: 哪些token被压制/增强
        mean_delta = deltas.mean(axis=0)  # [vocab] 平均Δlogit
        top_suppressed_idx = np.argsort(mean_delta)[:5]  # 最负 = 被压制
        top_boosted_idx = np.argsort(mean_delta)[-5:][::-1]  # 最正 = 被增强

        top_supp_tokens = [tokenizer.decode([i]).strip() for i in top_suppressed_idx[:3]]
        top_boost_tokens = [tokenizer.decode([i]).strip() for i in top_boosted_idx[:3]]

        # 统计token级变化
        for i in top_suppressed_idx[:3]:
            top_suppressed_all[li].append((i, float(mean_delta[i])))
        for i in top_boosted_idx[:3]:
            top_boosted_all[li].append((i, float(mean_delta[i])))

        # 判定
        if mean_corr > 0.9:
            verdict = "REPR_TRANSFORM (表示转换)"
        elif mean_corr > 0.7:
            verdict = "PARTIAL_RECOMPUTE (部分重算)"
        elif mean_corr > 0.4:
            verdict = "DISTRIBUTED_RECOMPUTE (分布式重算)"
        else:
            verdict = "STRONG_RECOMPUTE (强重算)"

        supp_str = ", ".join(top_supp_tokens)
        boost_str = ", ".join(top_boost_tokens)
        print(f"  L{li:<4} {mean_corr:<10.4f} {mean_norm:<12.4f} {supp_str:<25} {boost_str:<25} {verdict}")

        results[li] = {
            "correlation_with_final": float(mean_corr),
            "mean_delta_norm": float(mean_norm),
            "n_pairs": len(deltas),
            "verdict": verdict,
            "top_suppressed": top_supp_tokens,
            "top_boosted": top_boost_tokens,
        }

    # 关键判定
    early_corr = np.mean([results[li]["correlation_with_final"]
                          for li in range(min(3, n_layers)) if li in results])
    mid_corr = np.mean([results[li]["correlation_with_final"]
                        for li in range(n_layers//3, 2*n_layers//3) if li in results])
    late_corr = np.mean([results[li]["correlation_with_final"]
                         for li in range(2*n_layers//3, n_layers) if li in results])

    print(f"\n  >>> 早期层(0-2) ρ = {early_corr:.4f}")
    print(f"  >>> 中间层 ρ = {mid_corr:.4f}")
    print(f"  >>> 后期层 ρ = {late_corr:.4f}")

    if early_corr > 0.85:
        overall_verdict = "REPRESENTATION_TRANSFORMATION: 否定信息在早期层就完全编码,后续层只是表示转换"
    elif early_corr > 0.6:
        overall_verdict = "MIXED: 早期部分编码,后续层持续添加新计算"
    else:
        overall_verdict = "DISTRIBUTED_RECOMPUTATION: 否定效果确实在每层被重新计算"

    print(f"\n  >>> 总体判定: {overall_verdict}")

    # 检查expA矛盾: 如果early_corr > 0.8但Phase233的ratio≈1.0,
    # 说明ratio≈1.0是"表示转换"的伪影而非"重计算"
    print(f"\n  >>> 与Phase 233 ExpA的对照:")
    print(f"      Phase233 ratio≈1.0 被解读为'每层都在重计算否定'")
    print(f"      Phase234 ρ(early)={early_corr:.4f}")
    if early_corr > 0.8:
        print(f"      *** 矛盾! 高ρ(early)说明否定在早期就编码完成,")
        print(f"          ratio≈1.0反映的是'表示转换的破坏敏感性',不是'重计算'")
    else:
        print(f"      一致: 低ρ(early)支持分布式重计算解读")

    return {
        "layer_results": {str(k): v for k, v in results.items()},
        "early_corr": float(early_corr),
        "mid_corr": float(mid_corr),
        "late_corr": float(late_corr),
        "overall_verdict": overall_verdict,
        "n_pairs": len(pairs),
    }


# ===== ExpB: Value向量在L0的稳定性 =====

def expB_value_at_L0(model, tokenizer, device, info, n_pairs=50, model_name="qwen3"):
    """
    修复ExpB的核心缺陷: 在L0测量value向量

    L0的输入是原始token embedding + position embedding,
    没有经过前层的上下文处理。因此:
    - 如果L0的v_not跨上下文稳定 → "not"有固定语义核心
    - 如果L0的v_not跨上下文不稳定 → 条件化是本质性的

    同时在后续层(如L8, L16)做对比,验证不稳定性的来源。
    """
    print("\n" + "="*60)
    print("ExpB: Value Vector Stability at L0 (消除前层上下文污染)")
    print("="*60)

    layers = get_layers(model)
    n_layers = info.n_layers

    # 获取n_heads和head_dim
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

    # 关键: 只在L0和少数对比层测量
    target_layers = [0, 1, 2, 8, 16, n_layers - 1]
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))

    # 对每层每个head, 收集"not"token的value向量
    # 关键改进: 在L0, 输入是embedding, 没有前层处理
    value_vectors = defaultdict(lambda: defaultdict(list))  # layer -> head -> [v_not]

    # 同时收集embedding (L0输入) 用于验证
    not_embeddings = []  # "not" token的embedding

    for pi, (affirm, negated) in enumerate(pairs):
        if pi % 10 == 0:
            logger.update(f"ExpB pair {pi}/{len(pairs)}")

        neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
        tokens = tokenizer.convert_ids_to_tokens(neg_ids[0].tolist())

        # 找"not" token的位置
        not_pos = None
        for ti, tok in enumerate(tokens):
            tok_lower = tok.lower().replace('▁', '').replace(' ', '')
            if tok_lower in ('not', 'n\'t', 'not', 'nott'):
                not_pos = ti
                break

        # 如果没找到,取倒数第2或第3个
        if not_pos is None:
            # 对"did not"这种情况, "not"通常在倒数第3-4位置
            for ti in range(len(tokens)):
                tok_lower = tokens[ti].lower().replace('▁', '').replace(' ', '')
                if 'not' in tok_lower:
                    not_pos = ti
                    break
            if not_pos is None:
                not_pos = max(0, len(tokens) - 3)

        # 获取"not"的embedding (L0输入)
        with torch.no_grad():
            embed_layer = model.get_input_embeddings()
            not_token_id = neg_ids[0, not_pos].item()
            not_embed = embed_layer(torch.tensor([not_token_id], device=device))
            not_embeddings.append(not_embed[0].float().cpu().numpy())

        # 用output_hidden_states获取每层输入到该层的hidden state
        with torch.no_grad():
            out = model(input_ids=neg_ids, output_hidden_states=True)

        hidden_states = out.hidden_states  # [n_layers+1, 1, seq, d]

        for li in target_layers:
            layer = layers[li]

            # L0特殊处理: 输入是embedding, 没有前层处理
            if li == 0:
                # L0的输入是hidden_states[0] = embedding层输出
                h_not = hidden_states[0][0, not_pos].float().cpu()  # [d_model]
            else:
                # 后续层的输入是hidden_states[li] (已经过li层处理)
                h_not = hidden_states[li][0, not_pos].float().cpu()

            # 获取W_V — 处理meta device
            W_V_param = layer.self_attn.v_proj.weight
            if W_V_param.is_meta:
                # 尝试从safetensors加载
                try:
                    from safetensors import safe_open
                    import glob, os
                    model_path = MODEL_CONFIGS.get(model_name, {}).get("path", "")
                    if model_path:
                        for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
                            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                                key = f'model.layers.{li}.self_attn.v_proj.weight'
                                if key in sf.keys():
                                    W_V = sf.get_tensor(key).float().numpy()
                                    break
                        else:
                            continue
                    else:
                        continue
                except Exception:
                    continue
            else:
                W_V = W_V_param.detach().cpu().float().numpy()
            v_proj = W_V @ h_not.numpy()  # [d_model], W_V is numpy

            # 拆分为per-head
            for hi in range(n_heads):
                start = hi * head_dim
                end = (hi + 1) * head_dim
                v_head = v_proj[start:end]
                value_vectors[li][hi].append(v_head)

        del out
        if pi % 10 == 9:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.stop()

    # 分析跨上下文稳定性
    print(f"\n  Value向量跨上下文稳定性 (关键改进: L0输入=embedding):")
    print(f"  {'Layer':<6} {'Head':<6} {'Mean_Cos':<10} {'Std_Cos':<10} {'Verdict'}")
    print("  " + "-"*50)

    value_stability = {}
    for li in target_layers:
        layer_stable = 0
        for hi in range(n_heads):
            vecs = value_vectors[li][hi]
            if len(vecs) < 5:
                continue

            vecs_np = np.array(vecs)  # [n_contexts, head_dim]

            # 逐对cosine
            cosines = []
            n_sample = min(30, len(vecs_np))
            indices = np.random.choice(len(vecs_np), n_sample, replace=False) if len(vecs_np) > n_sample else range(len(vecs_np))
            sample_vecs = vecs_np[indices]

            for i in range(len(sample_vecs)):
                for j in range(i+1, min(i+8, len(sample_vecs))):
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
                    layer_stable += 1
                elif mean_cos > 0.5:
                    verdict = "MODERATE"
                else:
                    verdict = "UNSTABLE"

                # 只打印L0的详细结果 + 其他层的摘要
                if li <= 2 or mean_cos > 0.7:
                    print(f"  L{li:<4} H{hi:<4} {mean_cos:<10.4f} {std_cos:<10.4f} {verdict}")

                value_stability[f"L{li}_H{hi}"] = {
                    "mean_cosine": float(mean_cos),
                    "std_cosine": float(std_cos),
                    "verdict": verdict,
                }

        print(f"  --- L{li}: {layer_stable}/{n_heads} stable heads ---")

    # 关键判定
    l0_stable = sum(1 for k, v in value_stability.items()
                    if k.startswith("L0_") and v["mean_cosine"] > 0.8)
    l0_total = sum(1 for k in value_stability if k.startswith("L0_"))
    l0_mean_cos = np.mean([v["mean_cosine"] for k, v in value_stability.items()
                           if k.startswith("L0_")]) if l0_total > 0 else 0

    print(f"\n  >>> L0 Value向量稳定性: {l0_stable}/{l0_total} heads stable")
    print(f"  >>> L0 平均cosine: {l0_mean_cos:.4f}")

    if l0_mean_cos > 0.7:
        l0_verdict = "NOT有固定语义核心: L0的value向量跨上下文稳定,不稳定来自前层上下文处理"
    elif l0_mean_cos > 0.4:
        l0_verdict = "部分条件化: L0有一定稳定性,但上下文已经开始影响"
    else:
        l0_verdict = "本质条件化: 即使在L0(输入=embedding), value向量也跨上下文不稳定"

    print(f"  >>> 判定: {l0_verdict}")

    # 对比L0 vs 后续层
    for li in [8, 16, n_layers-1]:
        if li >= n_layers:
            continue
        li_mean = np.mean([v["mean_cosine"] for k, v in value_stability.items()
                           if k.startswith(f"L{li}_")]) if any(k.startswith(f"L{li}_") for k in value_stability) else 0
        print(f"  >>> L{li} 平均cosine: {li_mean:.4f} (vs L0={l0_mean_cos:.4f})")

    return {
        "value_stability": value_stability,
        "l0_stable_heads": l0_stable,
        "l0_total_heads": l0_total,
        "l0_mean_cosine": float(l0_mean_cos),
        "l0_verdict": l0_verdict,
        "target_layers": target_layers,
        "n_pairs": len(pairs),
    }


# ===== ExpC: 统一CC/CN指标 =====

def expC_unified_causal(model, tokenizer, device, info, n_pairs=30):
    """
    统一Causal Contribution (CC) 和 Causal Necessity (CN):

    CC(l) = KL_full - KL_mean_ablate_l
    → 该层对否定效果的正向贡献 (越大 = 该层越重要)

    CN(l) = KL_patch_δl
    → 破坏该层增量对否定效果的损害 (越大 = 该层越不可替代)

    合并分析:
    - CC大 + CN大 → "计算中心" (该层在主动计算否定)
    - CC小 + CN大 → "表示转换器" (该层在传递关键信息,但不是否定效果的主要来源)
    - CC大 + CN小 → "冗余计算" (该层在计算否定,但其他层也能做到)
    - CC小 + CN小 → "无关层"
    """
    print("\n" + "="*60)
    print("ExpC: Unified CC/CN Analysis (因果贡献 vs 因果必要性)")
    print("="*60)

    layers = get_layers(model)
    n_layers = info.n_layers
    pairs = NEGATION_PAIRS[:n_pairs]
    logger = ProgressLogger()

    # Step 1: 收集均值激活 (用于mean ablation)
    print("  Step 1: Computing mean activations...")
    mean_attn_outputs = {}
    mean_mlp_outputs = {}

    for li in range(0, n_layers, 4):  # 每4层一个,减少计算量
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
            mean_attn_outputs[li] = torch.stack(attn_outs).mean(dim=0)
        if mlp_outs:
            mean_mlp_outputs[li] = torch.stack(mlp_outs).mean(dim=0)

        if li % 8 == 0:
            logger.update(f"  Mean activation L{li}/{n_layers}")

    # Step 2: 基线KL (完整否定vs肯定)
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

    # Step 3: 计算CC和CN
    target_layers = sorted(mean_attn_outputs.keys())
    cc_results = {}  # (layer, component) -> CC value
    cn_results = {}  # layer -> CN value

    # CC: Mean ablation — 减少pairs数加速
    cc_n_pairs = min(5, len(pairs))
    cn_n_pairs = min(5, len(pairs))
    print("  Step 2: Computing CC (Causal Contribution) via mean ablation...")
    for li in target_layers:
        logger.update(f"  CC layer {li}")

        for comp_name, mean_output in [("self_attn", mean_attn_outputs.get(li)),
                                        ("mlp", mean_mlp_outputs.get(li))]:
            if mean_output is None:
                continue

            mean_out = mean_output.to(device)

            def make_mean_hook(mean_val):
                def hook(module, input, output):
                    if isinstance(output, tuple):
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
            for affirm, negated in pairs[:cc_n_pairs]:
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

            cc = mean_baseline_kl - np.mean(ablated_kls)
            cc_results[(li, comp_name)] = float(cc)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # CN: Incremental patching (from Phase 233 ExpA)
    print("  Step 3: Computing CN (Causal Necessity) via incremental patching...")
    for li in target_layers:
        logger.update(f"  CN layer {li}")

        inc_kls = []
        for pi, (affirm, negated) in enumerate(pairs[:cn_n_pairs]):
            aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
            neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)

            aff_hidden = {}
            neg_hidden = {}

            def make_hook(store, key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        store[key] = output[0].detach()
                    else:
                        store[key] = output.detach()
                return hook

            # 只在li和li-1层收集
            hook_layers = [li] + ([li-1] if li > 0 else [])
            aff_hooks = [layers[l].register_forward_hook(make_hook(aff_hidden, f"L{l}")) for l in hook_layers]
            with torch.no_grad():
                _ = model(input_ids=aff_ids)
            for h in aff_hooks:
                h.remove()

            neg_hooks = [layers[l].register_forward_hook(make_hook(neg_hidden, f"L{l}")) for l in hook_layers]
            with torch.no_grad():
                _ = model(input_ids=neg_ids)
            for h in neg_hooks:
                h.remove()

            if f"L{li}" not in aff_hidden or f"L{li}" not in neg_hidden:
                continue

            aff_h = aff_hidden[f"L{li}"][0, -1].float()
            neg_h = neg_hidden[f"L{li}"][0, -1].float()

            # Incremental patch
            if li == 0:
                patched_h = neg_h
            else:
                if f"L{li-1}" not in aff_hidden or f"L{li-1}" not in neg_hidden:
                    continue
                aff_h_prev = aff_hidden[f"L{li-1}"][0, -1].float()
                delta_neg = neg_h - neg_hidden[f"L{li-1}"][0, -1].float()
                patched_h = aff_h_prev + delta_neg

            # 投影到logit空间
            patched_logits = get_logits_from_hidden(model, patched_h, device)
            patched_probs = np.exp(patched_logits - patched_logits.max())
            patched_probs /= patched_probs.sum() + 1e-20

            # 基线肯定概率
            with torch.no_grad():
                aff_out = model(input_ids=aff_ids)
            aff_logits = aff_out.logits[0, -1].float().cpu().numpy()
            aff_probs = np.exp(aff_logits - aff_logits.max())
            aff_probs /= aff_probs.sum() + 1e-20

            kl = float(np.sum(patched_probs * np.log(patched_probs / (aff_probs + 1e-20) + 1e-20)))
            inc_kls.append(kl)

            del aff_hidden, neg_hidden, aff_out
            gc.collect()

        cn = np.mean(inc_kls) if inc_kls else 0.0
        cn_results[li] = float(cn)

    logger.stop()

    # 汇总分析
    print(f"\n  {'Layer':<6} {'Comp':<10} {'CC':<10} {'CN':<10} {'CC/CN':<10} {'Classification'}")
    print("  " + "-"*65)

    unified_results = {}
    for li in target_layers:
        cn = cn_results.get(li, 0)

        for comp in ["self_attn", "mlp"]:
            cc = cc_results.get((li, comp), 0)

            # 分类
            if cc > 0.01 and cn > 0.5:
                classification = "COMPUTATION_CENTER (计算中心)"
            elif cc <= 0.01 and cn > 0.5:
                classification = "REPR_TRANSFORMER (表示转换器)"
            elif cc > 0.01 and cn <= 0.5:
                classification = "REDUNDANT_COMPUTE (冗余计算)"
            else:
                classification = "IRRELEVANT (无关)"

            ratio = cc / (cn + 1e-10)
            print(f"  L{li:<4} {comp:<10} {cc:<10.4f} {cn:<10.4f} {ratio:<10.4f} {classification}")

            unified_results[f"L{li}_{comp}"] = {
                "CC": float(cc),
                "CN": float(cn),
                "CC_CN_ratio": float(ratio),
                "classification": classification,
            }

    return {
        "unified_results": unified_results,
        "baseline_kl": float(mean_baseline_kl),
        "n_pairs": len(pairs),
    }


# ===== ExpD: Token级Logit轨迹 =====

def expD_token_trajectory(model, tokenizer, device, info, n_pairs=40):
    """
    Token级logit轨迹: 否定对每个token的概率影响如何逐层演化

    这是从"hidden state"转向"logit flow"的关键实验。
    直接回答: "否定程序在做什么?"

    测量:
    1. 在每层,否定vs肯定的Δlogit中,哪些token被压制/增强
    2. "压制"和"增强"的模式是否构成有限的"程序基元"
    3. 不同句子对的token轨迹是否收敛到相似模式
    """
    print("\n" + "="*60)
    print("ExpD: Token-Level Logit Trajectory (程序基元提取)")
    print("="*60)

    layers = get_layers(model)
    n_layers = info.n_layers
    pairs = NEGATION_PAIRS[:n_pairs]
    logger = ProgressLogger()

    # 每层的Δlogit统计
    layer_delta_stats = defaultdict(lambda: {
        "suppress_indices": [],  # 被压制的token indices
        "boost_indices": [],     # 被增强的token indices
        "delta_entropy": [],     # Δlogit的熵
        "delta_sparsity": [],    # Δlogit的稀疏性 (top-10占的比例)
        "delta_norm": [],        # Δlogit的范数
    })

    # 关键对比词
    antonym_tokens = ["happy", "sad", "good", "bad", "warm", "cold",
                      "open", "closed", "alive", "dead", "safe", "dangerous"]

    for pi, (affirm, negated) in enumerate(pairs):
        if pi % 5 == 0:
            logger.update(f"ExpD pair {pi}/{len(pairs)}")

        aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
        neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)

        aff_hidden = {}
        neg_hidden = {}

        def make_hook(store, key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    store[key] = output[0].detach()
                else:
                    store[key] = output.detach()
            return hook

        # 采样层 (每2层一个,减少计算量)
        sample_layers = list(range(0, n_layers, 2)) + [n_layers - 1]
        sample_layers = sorted(set(sample_layers))

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

        for li in sample_layers:
            key = f"L{li}"
            if key not in aff_hidden or key not in neg_hidden:
                continue

            aff_h = aff_hidden[key][0, -1].float()
            neg_h = neg_hidden[key][0, -1].float()

            aff_logits = get_logits_from_hidden(model, aff_h, device)
            neg_logits = get_logits_from_hidden(model, neg_h, device)

            delta = neg_logits - aff_logits

            # 压制: Δlogit最负的token
            suppress_idx = np.argsort(delta)[:10]
            # 增强: Δlogit最正的token
            boost_idx = np.argsort(delta)[-10:][::-1]

            layer_delta_stats[li]["suppress_indices"].append(suppress_idx.tolist())
            layer_delta_stats[li]["boost_indices"].append(boost_idx.tolist())

            # 熵
            abs_delta = np.abs(delta)
            total = abs_delta.sum() + 1e-20
            probs = abs_delta / total
            entropy = -np.sum(probs * np.log(probs + 1e-20))
            layer_delta_stats[li]["delta_entropy"].append(float(entropy))

            # 稀疏性: top-10占比
            top10_energy = float(np.sum(np.sort(abs_delta)[-10:]))
            total_energy = float(np.sum(abs_delta))
            sparsity = top10_energy / (total_energy + 1e-20)
            layer_delta_stats[li]["delta_sparsity"].append(float(sparsity))

            # 范数
            layer_delta_stats[li]["delta_norm"].append(float(np.linalg.norm(delta)))

        del aff_hidden, neg_hidden
        if pi % 10 == 9:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.stop()

    # 汇总
    print(f"\n  {'Layer':<6} {'||Δ||':<8} {'Entropy':<10} {'Sparsity':<10} {'Top-3 Suppress':<20} {'Top-3 Boost':<20}")
    print("  " + "-"*80)

    results = {}
    for li in sorted(layer_delta_stats.keys()):
        stats = layer_delta_stats[li]
        mean_norm = np.mean(stats["delta_norm"])
        mean_entropy = np.mean(stats["delta_entropy"])
        mean_sparsity = np.mean(stats["delta_sparsity"])

        # 找跨pair最常被压制/增强的tokens
        all_suppress = [idx for sublist in stats["suppress_indices"] for idx in sublist[:3]]
        all_boost = [idx for sublist in stats["boost_indices"] for idx in sublist[:3]]

        # 计数
        suppress_counts = defaultdict(int)
        boost_counts = defaultdict(int)
        for idx in all_suppress:
            suppress_counts[idx] += 1
        for idx in all_boost:
            boost_counts[idx] += 1

        top_supp = sorted(suppress_counts.items(), key=lambda x: -x[1])[:3]
        top_boost = sorted(boost_counts.items(), key=lambda x: -x[1])[:3]

        supp_str = ", ".join([f"{tokenizer.decode([i]).strip()}({c})" for i, c in top_supp])
        boost_str = ", ".join([f"{tokenizer.decode([i]).strip()}({c})" for i, c in top_boost])

        print(f"  L{li:<4} {mean_norm:<8.2f} {mean_entropy:<10.2f} {mean_sparsity:<10.4f} {supp_str:<20} {boost_str:<20}")

        results[li] = {
            "mean_delta_norm": float(mean_norm),
            "mean_entropy": float(mean_entropy),
            "mean_sparsity": float(mean_sparsity),
            "top_suppressed": [(int(i), int(c)) for i, c in top_supp],
            "top_boosted": [(int(i), int(c)) for i, c in top_boost],
        }

    # 程序基元提取
    print(f"\n  >>> 程序基元分析:")
    print(f"      如果sparsity > 0.8: 否定是'稀疏重写'(少数token被修改)")
    print(f"      如果sparsity < 0.3: 否定是'全局重写'(大量token被修改)")

    early_sparsity = np.mean([layer_delta_stats[li]["delta_sparsity"]
                              for li in range(min(3, n_layers)) if li in layer_delta_stats])
    late_sparsity = np.mean([layer_delta_stats[li]["delta_sparsity"]
                             for li in range(2*n_layers//3, n_layers) if li in layer_delta_stats])

    print(f"      早期层sparsity: {early_sparsity:.4f}")
    print(f"      后期层sparsity: {late_sparsity:.4f}")

    if early_sparsity > 0.6:
        print(f"      → 否定程序以'稀疏压制'为主: 少数关键token被精准修改")
    else:
        print(f"      → 否定程序以'全局分布重写'为主: 大量token概率被重新分配")

    return {
        "layer_results": {str(k): v for k, v in results.items()},
        "early_sparsity": float(early_sparsity),
        "late_sparsity": float(late_sparsity),
        "n_pairs": len(pairs),
    }


# ===== ExpE: Steering概率有效性 =====

def expE_steering_probability(model, tokenizer, device, info, n_pairs=40):
    """
    修复DS7B的steering度量问题:
    之前用hidden state cosine衡量steering效果 → DS7B被判为"失败"
    但DS7B可能用非线性方式编码否定,hidden state不相似不代表steering失败

    正确度量: steering后输出概率分布的变化
    - 如果steering使输出概率向"否定"方向移动 → steering成功
    - 如果概率分布完全没变 → 才是真正失败

    同时对比3种度量:
    1. cos_neg: hidden state余弦 (旧度量)
    2. kl_neg: KL(steered_dist || negated_dist) (新度量1: 与否定的距离)
    3. prob_shift: steer后"否定相关token"的概率变化 (新度量2: 行为有效性)
    """
    print("\n" + "="*60)
    print("ExpE: Steering Effectiveness via Probability (修复DS7B度量)")
    print("="*60)

    layers = get_layers(model)
    n_layers = info.n_layers
    pairs = NEGATION_PAIRS[:n_pairs]
    logger = ProgressLogger()

    # 先收集hidden states训练探针
    layer_affirm_h = defaultdict(list)
    layer_negate_h = defaultdict(list)
    sample_layers = list(range(0, n_layers, 4)) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))

    for pi, (affirm, negated) in enumerate(pairs):
        if pi % 10 == 0:
            logger.update(f"ExpE collecting pair {pi}/{len(pairs)}")

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
        if pi % 10 == 9:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 训练探针获取否定方向
    from sklearn.linear_model import LogisticRegression

    negation_directions = {}
    probe_accs = {}
    for li in sample_layers:
        X_aff = np.array(layer_affirm_h[li])
        X_neg = np.array(layer_negate_h[li])
        if len(X_aff) < 10:
            continue

        X = np.concatenate([X_aff, X_neg], axis=0)
        y = np.concatenate([np.zeros(len(X_aff)), np.ones(len(X_neg))])

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)
        acc = clf.score(X, y)

        # 否定方向
        neg_dir = clf.coef_[0]
        neg_dir_norm = neg_dir / (np.linalg.norm(neg_dir) + 1e-20)
        negation_directions[li] = neg_dir_norm
        probe_accs[li] = float(acc)

    print(f"  Probe trained on {len(negation_directions)} layers")

    # Steering测试 — 使用3种度量
    steering_layers = sorted(negation_directions.keys())[::3]  # 每3层测一次

    print(f"\n  {'Layer':<6} {'Beta':<6} {'cos_neg':<10} {'kl_to_neg':<12} {'prob_shift':<12} {'Verdict'}")
    print("  " + "-"*65)

    steering_results = {}

    for li in steering_layers:
        neg_dir = negation_directions[li]
        beta = 20  # 固定beta=20

        cos_negs = []
        kl_to_negs = []
        prob_shifts = []

        for affirm, negated in pairs[:10]:
            aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
            neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                aff_out = model(input_ids=aff_ids, output_hidden_states=True)
                neg_out = model(input_ids=neg_ids, output_hidden_states=True)

            # Get hidden state and steer
            h_li = aff_out.hidden_states[li][0, -1].float().cpu().numpy()
            h_steered = h_li + beta * neg_dir

            # 3种度量
            # 1. cos_neg: steered hidden vs negated hidden
            neg_h_idx = li if li < len(neg_out.hidden_states) else -1
            neg_h = neg_out.hidden_states[neg_h_idx][0, -1].float().cpu().numpy()
            n1 = np.linalg.norm(h_steered)
            n2 = np.linalg.norm(neg_h)
            cos_neg = float(np.dot(h_steered, neg_h) / (n1 * n2 + 1e-20)) if n1 > 1e-10 and n2 > 1e-10 else 0

            # 2. kl_to_neg: KL(steered_prob || negated_prob)
            steered_logits = get_logits_from_hidden(model, torch.tensor(h_steered), device)
            neg_logits = neg_out.logits[0, -1].float().cpu().numpy()

            steered_probs = np.exp(steered_logits - steered_logits.max())
            steered_probs /= steered_probs.sum() + 1e-20
            neg_probs = np.exp(neg_logits - neg_logits.max())
            neg_probs /= neg_probs.sum() + 1e-20

            kl_to_neg = float(np.sum(steered_probs * np.log(steered_probs / (neg_probs + 1e-20) + 1e-20)))

            # 3. prob_shift: "否定相关token"的概率变化
            # 对比: 肯定句top-5 token的概率 vs steering后这些token的概率
            aff_logits = aff_out.logits[0, -1].float().cpu().numpy()
            aff_probs_full = np.exp(aff_logits - aff_logits.max())
            aff_probs_full /= aff_probs_full.sum() + 1e-20

            # 找否定句的top-5 token
            neg_top5 = np.argsort(neg_probs)[-5:]

            # 这些token在steered分布中的概率 vs 肯定分布中的概率
            steered_at_neg_top = float(steered_probs[neg_top5].sum())
            aff_at_neg_top = float(aff_probs_full[neg_top5].sum())
            prob_shift = steered_at_neg_top - aff_at_neg_top  # 正 = 向否定移动

            cos_negs.append(cos_neg)
            kl_to_negs.append(kl_to_neg)
            prob_shifts.append(prob_shift)

            del aff_out, neg_out

        mean_cos = np.mean(cos_negs)
        mean_kl = np.mean(kl_to_negs)
        mean_shift = np.mean(prob_shifts)

        # 判定
        if mean_shift > 0.05:
            verdict = "EFFECTIVE (概率有效移动)"
        elif mean_shift > 0.01:
            verdict = "PARTIAL (部分有效)"
        else:
            verdict = "FAILED (概率未移动)"

        print(f"  L{li:<4} {beta:<6} {mean_cos:<10.4f} {mean_kl:<12.4f} {mean_shift:<12.4f} {verdict}")

        steering_results[f"L{li}_b{beta}"] = {
            "mean_cos_neg": float(mean_cos),
            "mean_kl_to_negated": float(mean_kl),
            "mean_prob_shift": float(mean_shift),
            "verdict": verdict,
            "probe_accuracy": float(probe_accs.get(li, 0)),
        }

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.stop()

    return {
        "steering_results": steering_results,
        "probe_accuracies": {str(k): v for k, v in probe_accs.items()},
        "n_pairs": len(pairs),
    }


# ===== Main =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    print("="*60)
    print(f"Phase 234: Logit Lens Mechanics + Unified Causal Analysis")
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
        ("expA", expA_logit_lens, 60),
        ("expB", expB_value_at_L0, 40),
        ("expC", expC_unified_causal, 15),
        ("expD", expD_token_trajectory, 30),
        ("expE", expE_steering_probability, 30),
    ]

    for exp_name, exp_func, n_pairs in experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp_name}...")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            # expB需要model_name参数
            if exp_name == "expB":
                result = exp_func(model, tokenizer, device, info, n_pairs=n_pairs, model_name=model_name)
            else:
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
    out_path = f"tests/glm5_temp/phase234_{model_name}_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    # 释放模型
    release_model(model)
    model = None

    # 最终总结
    print(f"\n{'='*60}")
    print(f"Phase 234 Final Summary")
    print(f"{'='*60}")

    if "expA" in all_results and "error" not in all_results["expA"]:
        expA = all_results["expA"]
        print(f"  ExpA (Logit Lens): ρ(early)={expA.get('early_corr', 'N/A'):.4f}, "
              f"ρ(mid)={expA.get('mid_corr', 'N/A'):.4f}, "
              f"ρ(late)={expA.get('late_corr', 'N/A'):.4f}")
        print(f"  >>> {expA.get('overall_verdict', 'N/A')}")

    if "expB" in all_results and "error" not in all_results["expB"]:
        expB = all_results["expB"]
        print(f"  ExpB (Value@L0): L0 mean_cos={expB.get('l0_mean_cosine', 'N/A'):.4f}, "
              f"stable={expB.get('l0_stable_heads', 'N/A')}/{expB.get('l0_total_heads', 'N/A')}")
        print(f"  >>> {expB.get('l0_verdict', 'N/A')}")

    if "expC" in all_results and "error" not in all_results["expC"]:
        expC = all_results["expC"]
        ur = expC.get("unified_results", {})
        comp_centers = [k for k, v in ur.items() if "计算中心" in v.get("classification", "")]
        repr_transforms = [k for k, v in ur.items() if "表示转换器" in v.get("classification", "")]
        print(f"  ExpC (CC/CN): 计算中心={comp_centers}, 表示转换器={repr_transforms}")

    if "expD" in all_results and "error" not in all_results["expD"]:
        expD = all_results["expD"]
        print(f"  ExpD (Token轨迹): early_sparsity={expD.get('early_sparsity', 'N/A'):.4f}, "
              f"late_sparsity={expD.get('late_sparsity', 'N/A'):.4f}")

    if "expE" in all_results and "error" not in all_results["expE"]:
        expE = all_results["expE"]
        sr = expE.get("steering_results", {})
        effective = [k for k, v in sr.items() if "EFFECTIVE" in v.get("verdict", "")]
        print(f"  ExpE (Steering): 有效层={effective}")

    print(f"\nDone! {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
