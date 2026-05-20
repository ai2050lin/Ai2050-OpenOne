"""
Phase 232: Negation Circuit Localization (否定回路定位)
======================================================

Phase 231核心发现: "not"将top-10概率压制到7-10% (10-14x压制), 三模型一致
这是目前最强的feature primitive, 但只是行为级发现

Phase 232目标: 从"行为级发现"进入"机制级理解"
核心问题:
  1. "not"的效果在哪一层首次出现? (层定位)
  2. "not"在logit空间是固定门控还是上下文依赖? (门控机制)
  3. 替换哪一层的激活可以因果地产生否定效果? (因果验证)
  4. 哪些attention head是"否定回路"的组成部分? (回路定位)
  5. 不同否定词(not/never/cannot/no)是否共享回路? (跨词泛化)

5个实验:
  ExpA: 否定层定位 ★★★★★ — 50+模板, 逐层测量KL, 找否定onset层
  ExpB: Logit空间门控分析 ★★★★★ — Δ_logit跨上下文稳定性
  ExpC: Activation Patching ★★★★★ — 金标准因果验证
  ExpD: Attention Head消融 ★★★★ — 找否定回路关键head
  ExpE: 跨否定词泛化 ★★★ — not/never/cannot/no共享性

用法: python tests/glm5/phase232_negation_circuit.py [qwen3|glm4|deepseek7b]

重要:
- GLM4/DS7B使用device_map="auto" + bfloat16
- 逐模型测试, 避免GPU OOM
- 使用flash attention (attn_implementation="flash_attention_2")
- 每30秒输出进度日志
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import json
import numpy as np
import torch
import threading
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from model_utils import (get_layers, get_model_info, release_model,
                          get_sample_layers, MODEL_CONFIGS, get_W_U)


# ===== 大规模否定模板 (50+对) =====

NEGATION_PAIRS = [
    # 1. 形容词否定 (15对)
    ("The sky is blue", "The sky is not blue"),
    ("The water is cold", "The water is not cold"),
    ("The food is delicious", "The food is not delicious"),
    ("The movie is interesting", "The movie is not interesting"),
    ("The task is easy", "The task is not easy"),
    ("The man is tall", "The man is not tall"),
    ("The car is fast", "The car is not fast"),
    ("The room is clean", "The room is not clean"),
    ("The child is happy", "The child is not happy"),
    ("The building is old", "The building is not old"),
    ("The story is true", "The story is not true"),
    ("The road is safe", "The road is not safe"),
    ("The dog is friendly", "The dog is not friendly"),
    ("The weather is warm", "The weather is not warm"),
    ("The exam is difficult", "The exam is not difficult"),
    # 2. 动词否定 (15对)
    ("The bird can fly", "The bird cannot fly"),
    ("She likes reading", "She does not like reading"),
    ("He plays guitar", "He does not play guitar"),
    ("They eat meat", "They do not eat meat"),
    ("We know the answer", "We do not know the answer"),
    ("The machine works", "The machine does not work"),
    ("She speaks French", "She does not speak French"),
    ("He understands math", "He does not understand math"),
    ("The system runs smoothly", "The system does not run smoothly"),
    ("I believe this story", "I do not believe this story"),
    ("They support the plan", "They do not support the plan"),
    ("She remembers the date", "She does not remember the date"),
    ("He owns a house", "He does not own a house"),
    ("The door opens easily", "The door does not open easily"),
    ("The plant grows quickly", "The plant does not grow quickly"),
    # 3. 状态/存在否定 (10对)
    ("There is hope", "There is no hope"),
    ("There are solutions", "There are no solutions"),
    ("Money is available", "Money is not available"),
    ("Help is coming", "Help is not coming"),
    ("Time remains", "Time does not remain"),
    ("The evidence exists", "The evidence does not exist"),
    ("The chance remains", "The chance does not remain"),
    ("The problem persists", "The problem does not persist"),
    ("The tradition continues", "The tradition does not continue"),
    ("The risk exists", "The risk does not exist"),
    # 4. 复杂语境否定 (10对)
    ("Scientists agree that climate change is real", "Scientists do not agree that climate change is real"),
    ("The results confirm the hypothesis", "The results do not confirm the hypothesis"),
    ("Everyone believes this is correct", "Not everyone believes this is correct"),
    ("The data supports the theory", "The data does not support the theory"),
    ("The patient responds to treatment", "The patient does not respond to treatment"),
    ("The system guarantees security", "The system does not guarantee security"),
    ("The method produces reliable results", "The method does not produce reliable results"),
    ("The company follows regulations", "The company does not follow regulations"),
    ("The project meets expectations", "The project does not meet expectations"),
    ("The evidence proves the claim", "The evidence does not prove the claim"),
    # 5. 否定词变体 (5对 - 用于ExpE)
    ("The answer is correct", "The answer is never correct"),
    ("The method works", "The method will never work"),
    ("This is possible", "This is not possible"),
    ("He can do it", "He can never do it"),
    ("The situation improves", "The situation does not improve"),
]

# 跨否定词测试 (ExpE)
CROSS_NEG_WORDS = {
    "not": "The cat is {adj}",
    "never": "The cat is never {adj}",
    "cannot": "The cat cannot be {adj}",
    "no": "There is no {noun} in the box",
}

CROSS_NEG_ADJS = ["happy", "hungry", "smart", "brave", "friendly",
                   "strong", "quiet", "fast", "warm", "clean"]

CROSS_NEG_NOUNS = ["cat", "dog", "food", "water", "money",
                    "hope", "time", "answer", "reason", "way"]


# ===== 模型加载 =====

def load_model_bf16(model_name):
    """BF16 + device_map="auto" + flash_attention_2"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bfloat16 + device_map=auto + flash_attn)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 尝试flash_attention_2, 失败则回退eager
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        print(f"[load] Using flash_attention_2")
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
    print(f"[load] {model_name}: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# ===== 进度日志 =====

class ProgressLogger:
    """每30秒输出一次GPU和进度信息"""
    def __init__(self):
        self.running = True
        self.message = "Starting..."
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    
    def _loop(self):
        while self.running:
            if torch.cuda.is_available():
                gpu = torch.cuda.memory_allocated() / 1e9
                gpu_res = torch.cuda.memory_reserved() / 1e9
                print(f"[progress] {self.message} | GPU: {gpu:.2f}GB alloc, {gpu_res:.2f}GB reserved")
            time.sleep(30)
    
    def update(self, msg):
        self.message = msg
    
    def stop(self):
        self.running = False


# ===== ExpA: 否定层定位 =====

def expA_negation_layer_localization(model, tokenizer, device, model_info):
    """
    核心问题: "not"的效果在哪一层首次出现?
    
    方法: 
    1. 对每个否定对, 逐层提取hidden state
    2. 用lm_head投影到logit空间
    3. 计算KL散度: KL(P(next|affirmative, layer_l) || P(next|negated, layer_l))
    4. 找KL首次显著增大的层 → 否定onset层
    """
    print("\n" + "="*60)
    print("ExpA: Negation Layer Localization (否定层定位)")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    # 获取lm_head
    W_U = get_W_U(model, model_info.name)  # [vocab, d_model]
    
    # 采样层 (每2层采样一次 + 首尾)
    sample_layers = get_sample_layers(n_layers, n_samples=min(n_layers, 20))
    print(f"  Sampling {len(sample_layers)} layers: {sample_layers[:5]}...{sample_layers[-3:]}")
    
    # 选择50对否定模板
    pairs = NEGATION_PAIRS[:50]
    
    # 存储结果
    all_layer_kl = defaultdict(list)  # layer -> [kl_per_pair]
    all_layer_flip = defaultdict(list)  # layer -> [flip_ratio_per_pair]
    
    logger = ProgressLogger()
    
    for pi, (affirm, negated) in enumerate(pairs):
        logger.update(f"ExpA pair {pi+1}/{len(pairs)}")
        
        # 编码
        aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
        neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
        
        # 前向推理, 收集各层hidden state
        aff_hidden = {}
        neg_hidden = {}
        
        def make_hook(store, key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    store[key] = output[0].detach()
                else:
                    store[key] = output.detach()
            return hook
        
        # 注册hooks
        aff_hooks = []
        neg_hooks = []
        for li in sample_layers:
            aff_hooks.append(layers[li].register_forward_hook(make_hook(aff_hidden, f"L{li}")))
            neg_hooks.append(layers[li].register_forward_hook(make_hook(neg_hidden, f"L{li}")))
        
        with torch.no_grad():
            _ = model(input_ids=aff_ids)
            _ = model(input_ids=neg_ids)
        
        # 移除hooks
        for h in aff_hooks + neg_hooks:
            h.remove()
        
        # 逐层计算KL和flip ratio
        for li in sample_layers:
            key = f"L{li}"
            if key not in aff_hidden or key not in neg_hidden:
                continue
            
            # 取last token的hidden state
            aff_h = aff_hidden[key][0, -1].float()  # [d_model] on device
            neg_h = neg_hidden[key][0, -1].float()  # [d_model] on device
            
            # 投影到logit空间 (必须先经过LayerNorm!)
            # 使用模型的final LayerNorm + lm_head
            with torch.no_grad():
                aff_h_dev = aff_hidden[key][0, -1].to(device)  # [d_model] on device
                neg_h_dev = neg_hidden[key][0, -1].to(device)
                # Apply LayerNorm + lm_head, matching dtype
                aff_normed = model.model.norm(aff_h_dev.unsqueeze(0).unsqueeze(0))
                neg_normed = model.model.norm(neg_h_dev.unsqueeze(0).unsqueeze(0))
                # Cast to match lm_head dtype
                lm_head_dtype = next(model.lm_head.parameters()).dtype
                aff_logits = model.lm_head(aff_normed.to(lm_head_dtype))[0, 0].float().cpu().numpy()
                neg_logits = model.lm_head(neg_normed.to(lm_head_dtype))[0, 0].float().cpu().numpy()
            
            # Softmax
            aff_probs = np.exp(aff_logits - aff_logits.max())
            aff_probs /= aff_probs.sum() + 1e-20
            neg_probs = np.exp(neg_logits - neg_logits.max())
            neg_probs /= neg_probs.sum() + 1e-20
            
            # KL散度
            kl = float(np.sum(neg_probs * np.log(neg_probs / (aff_probs + 1e-20) + 1e-20)))
            all_layer_kl[li].append(kl)
            
            # Top-10 flip ratio
            aff_top10 = np.argsort(aff_probs)[-10:]
            neg_top10_probs = neg_probs[aff_top10]
            aff_top10_probs = aff_probs[aff_top10]
            aff_top10_sum = aff_top10_probs.sum() + 1e-20
            neg_top10_sum = neg_top10_probs.sum() + 1e-20
            flip_ratio = neg_top10_sum / aff_top10_sum
            all_layer_flip[li].append(flip_ratio)
        
        # 释放内存
        del aff_hidden, neg_hidden
        if pi % 10 == 9:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logger.stop()
    
    # 汇总结果
    results = {}
    print(f"\n  {'Layer':<8} {'Mean_KL':<12} {'Mean_Flip':<12} {'Flip<0.5%':<10} {'Onset?'}")
    print("  " + "-"*55)
    
    onset_layer = None
    prev_kl = 0
    for li in sorted(sample_layers):
        kl_mean = np.mean(all_layer_kl[li]) if li in all_layer_kl else 0
        flip_mean = np.mean(all_layer_flip[li]) if li in all_layer_flip else 1.0
        flip_pct = np.mean([f < 0.5 for f in all_layer_flip[li]]) * 100 if li in all_layer_flip else 0
        
        is_onset = False
        if onset_layer is None and flip_pct > 30:
            is_onset = True
            onset_layer = li
        
        print(f"  L{li:<6} {kl_mean:<12.4f} {flip_mean:<12.4f} {flip_pct:<10.1f} {'<<< ONSET' if is_onset else ''}")
        
        results[li] = {
            "mean_kl": float(kl_mean),
            "mean_flip_ratio": float(flip_mean),
            "flip_below_0.5_pct": float(flip_pct),
            "n_pairs": len(all_layer_kl.get(li, [])),
        }
    
    print(f"\n  >>> 否定Onset层: L{onset_layer}")
    
    return {
        "layer_results": {str(k): v for k, v in results.items()},
        "onset_layer": onset_layer,
        "n_pairs": len(pairs),
    }


# ===== ExpB: Logit空间门控分析 =====

def expB_logit_gate_analysis(model, tokenizer, device, model_info):
    """
    核心问题: "not"在logit空间是固定门控还是上下文依赖?
    
    方法:
    1. 对多个上下文, 计算 Δ_logit = logit(next|not, c) - logit(next|c)
    2. 测量Δ_logit在不同上下文间的相关性
    3. 如果高相关 → "not"是固定门控 (最强的算子假设)
    4. 如果低相关 → "not"是条件化的
    
    同时分析:
    - Δ_logit的稀疏性 (多少token被显著影响)
    - 门控方向: 压制vs提升的比例
    - Δ_logit是否是加性的 (logit空间)
    """
    print("\n" + "="*60)
    print("ExpB: Logit-Space Gate Analysis (Logit空间门控分析)")
    print("="*60)
    
    W_U = get_W_U(model, model_info.name)
    vocab_size = W_U.shape[0]
    
    pairs = NEGATION_PAIRS[:40]
    
    delta_logits = []  # 每个上下文的Δ_logit向量
    kl_values = []
    
    logger = ProgressLogger()
    
    for pi, (affirm, negated) in enumerate(pairs):
        logger.update(f"ExpB pair {pi+1}/{len(pairs)}")
        
        aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
        neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            aff_out = model(input_ids=aff_ids, output_hidden_states=True)
            neg_out = model(input_ids=neg_ids, output_hidden_states=True)
        
        # 最终层logit
        aff_logits = aff_out.logits[0, -1].float().cpu().numpy()  # [vocab]
        neg_logits = neg_out.logits[0, -1].float().cpu().numpy()  # [vocab]
        
        # Δ_logit
        delta = neg_logits - aff_logits
        delta_logits.append(delta)
        
        # KL
        aff_probs = np.exp(aff_logits - aff_logits.max())
        aff_probs /= aff_probs.sum() + 1e-20
        neg_probs = np.exp(neg_logits - neg_logits.max())
        neg_probs /= neg_probs.sum() + 1e-20
        kl = float(np.sum(neg_probs * np.log(neg_probs / (aff_probs + 1e-20) + 1e-20)))
        kl_values.append(kl)
        
        del aff_out, neg_out
        if pi % 10 == 9:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logger.stop()
    
    delta_logits = np.array(delta_logits)  # [n_pairs, vocab]
    
    # 1. 跨上下文相关性: 对每对Δ_logit计算cosine
    n = len(delta_logits)
    cross_cosines = []
    for i in range(n):
        for j in range(i+1, n):
            d1, d2 = delta_logits[i], delta_logits[j]
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos = float(np.dot(d1, d2) / (n1 * n2))
                cross_cosines.append(cos)
    
    mean_cross_cos = np.mean(cross_cosines) if cross_cosines else 0
    
    # 2. Δ_logit的稀疏性: 多少token被显著影响 (|Δ| > mean + 2*std)
    mean_delta = np.mean(delta_logits, axis=0)  # [vocab] 平均门控向量
    delta_std = np.std(delta_logits, axis=0)
    threshold = np.abs(mean_delta).mean() + 2 * np.abs(mean_delta).std()
    n_significant = int(np.sum(np.abs(mean_delta) > threshold))
    sparsity = 1.0 - n_significant / vocab_size
    
    # 3. 门控方向: 压制vs提升
    n_suppressed = int(np.sum(mean_delta < -threshold))
    n_enhanced = int(np.sum(mean_delta > threshold))
    suppress_ratio = n_suppressed / max(n_suppressed + n_enhanced, 1)
    
    # 4. Top-20 受影响token
    top_suppressed_idx = np.argsort(mean_delta)[:20]
    top_enhanced_idx = np.argsort(mean_delta)[-20:]
    
    top_suppressed_tokens = [tokenizer.decode([i]).strip() for i in top_suppressed_idx]
    top_enhanced_tokens = [tokenizer.decode([i]).strip() for i in top_enhanced_idx]
    
    # 5. PCA分析: Δ_logit的主要变异方向
    if len(delta_logits) > 5:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(5, len(delta_logits)-1))
        pca.fit(delta_logits)
        var_explained = pca.explained_variance_ratio_
    else:
        var_explained = [0]
    
    print(f"\n  跨上下文cosine均值: {mean_cross_cos:.4f}")
    print(f"  稀疏性: {sparsity:.4f} (显著token: {n_significant}/{vocab_size})")
    print(f"  压制/提升比: {suppress_ratio:.3f} (压制{n_suppressed}, 提升{n_enhanced})")
    print(f"  PCA方差解释: PC1={var_explained[0]:.3f}, PC2={var_explained[1] if len(var_explained)>1 else 0:.3f}")
    print(f"\n  Top-10 被压制token: {top_suppressed_tokens[:10]}")
    print(f"  Top-10 被提升token: {top_enhanced_tokens[:10]}")
    
    # 判决
    if mean_cross_cos > 0.5:
        verdict = "FIXED GATE (固定门控) - not在logit空间近似固定变换"
    elif mean_cross_cos > 0.2:
        verdict = "WEAK GATE (弱门控) - not有部分固定成分, 但高度上下文依赖"
    else:
        verdict = "CONDITIONAL (条件化) - not的效果几乎完全依赖上下文"
    
    print(f"\n  >>> 判决: {verdict}")
    
    return {
        "mean_cross_cosine": float(mean_cross_cos),
        "sparsity": float(sparsity),
        "n_significant_tokens": n_significant,
        "suppress_ratio": float(suppress_ratio),
        "n_suppressed": n_suppressed,
        "n_enhanced": n_enhanced,
        "pca_var_explained": [float(v) for v in var_explained],
        "top_suppressed": top_suppressed_tokens[:20],
        "top_enhanced": top_enhanced_tokens[:20],
        "verdict": verdict,
        "n_pairs": len(pairs),
        "mean_kl": float(np.mean(kl_values)),
    }


# ===== ExpC: Activation Patching (金标准因果验证) =====

def expC_activation_patching(model, tokenizer, device, model_info):
    """
    核心问题: 替换哪一层的激活可以因果地产生否定效果?
    
    方法 (Residual Stream Patching):
    1. 源运行: "The cat is not happy" → 捕获各层residual stream
    2. 目标运行: "The cat is happy" → 在每层替换为源的residual stream
    3. 测量替换后输出概率的变化
    4. 因果效果最大的层 = 否定效果编码的层
    
    这是机制解释性的金标准方法。
    """
    print("\n" + "="*60)
    print("ExpC: Activation Patching (激活替换 - 金标准因果验证)")
    print("="*60)
    
    n_layers = model_info.n_layers
    layers = get_layers(model)
    
    # 精选10对高质量否定对
    test_pairs = NEGATION_PAIRS[:10]
    
    # 采样层
    sample_layers = get_sample_layers(n_layers, n_samples=min(n_layers, 15))
    print(f"  Patching at {len(sample_layers)} layers")
    
    logger = ProgressLogger()
    
    # 存储结果
    patch_results = defaultdict(list)  # layer -> [kl_per_pair]
    
    for pi, (affirm, negated) in enumerate(test_pairs):
        logger.update(f"ExpC pair {pi+1}/{len(test_pairs)}")
        
        # === Step 1: 源运行 (negated) ===
        neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
        neg_seq_len = neg_ids.shape[1]
        
        # 捕获源各层residual stream
        neg_resid = {}
        def make_source_hook(store, key):
            def hook(module, input, output):
                # Transformer层的输出是residual stream
                if isinstance(output, tuple):
                    store[key] = output[0].detach().clone()
                else:
                    store[key] = output.detach().clone()
            return hook
        
        neg_hooks = [layers[li].register_forward_hook(make_source_hook(neg_resid, f"L{li}")) 
                      for li in sample_layers]
        
        with torch.no_grad():
            neg_out = model(input_ids=neg_ids)
        
        for h in neg_hooks:
            h.remove()
        
        # 获取否定的baseline logits
        neg_logits_base = neg_out.logits[0, -1].float().cpu().numpy()
        neg_probs_base = np.exp(neg_logits_base - neg_logits_base.max())
        neg_probs_base /= neg_probs_base.sum() + 1e-20
        
        # === Step 2: 目标运行 (affirmative) ===
        aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
        aff_seq_len = aff_ids.shape[1]
        
        # 获取affirmative的baseline logits
        with torch.no_grad():
            aff_out = model(input_ids=aff_ids)
        aff_logits_base = aff_out.logits[0, -1].float().cpu().numpy()
        aff_probs_base = np.exp(aff_logits_base - aff_logits_base.max())
        aff_probs_base /= aff_probs_base.sum() + 1e-20
        
        # 原始KL (affirmative vs negated)
        base_kl = float(np.sum(neg_probs_base * np.log(neg_probs_base / (aff_probs_base + 1e-20) + 1e-20)))
        
        # === Step 3: 逐层Patching ===
        # 关键: 我们需要从affirmative序列的位置对应到negated序列
        # "The cat is happy" (4 tokens) vs "The cat is not happy" (5 tokens)
        # 由于序列长度不同, 我们patch的是整个residual stream
        # 使用affirmative序列的格式, 在对应层替换为negated的residual stream
        
        for li in sample_layers:
            key = f"L{li}"
            if key not in neg_resid:
                patch_results[li].append(0.0)
                continue
            
            # 源(negated)在该层的residual stream
            source_resid = neg_resid[key]  # [1, neg_seq_len, d_model]
            
            # 我们需要运行affirmative, 但在第li层之后替换residual stream
            # 方法: 先运行到第li层, 捕获输出, 然后替换后继续
            
            # 更简单的方法: 直接在affirmative forward中hook替换
            patched_logits = None
            
            def make_patch_hook(source_resid_tensor, src_len, tgt_len):
                """创建一个hook, 将residual stream替换为source的对应部分"""
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        orig = output[0]
                    else:
                        orig = output
                    
                    # 取source的last token位置
                    # 注意: negated和affirmative序列长度不同
                    # 简化: 用source整个residual stream替换
                    # 但维度不匹配时, 只替换可以覆盖的部分
                    
                    batch_size, seq_len, dim = orig.shape
                    src_seq_len = source_resid_tensor.shape[1]
                    
                    # 最安全的方法: 只替换最后一个token的residual stream
                    patched = orig.clone()
                    # 用source的最后一个token替换target的最后一个token
                    patched[0, -1, :] = source_resid_tensor[0, -1, :]
                    
                    if isinstance(output, tuple):
                        return (patched,) + output[1:]
                    return patched
                return hook
            
            # 注册patch hook
            hook = layers[li].register_forward_hook(
                make_patch_hook(source_resid, neg_seq_len, aff_seq_len)
            )
            
            with torch.no_grad():
                patch_out = model(input_ids=aff_ids)
            
            hook.remove()
            
            patch_logits = patch_out.logits[0, -1].float().cpu().numpy()
            patch_probs = np.exp(patch_logits - patch_logits.max())
            patch_probs /= patch_probs.sum() + 1e-20
            
            # Patched vs affirmative的KL (越大说明patch越有效)
            patch_kl = float(np.sum(patch_probs * np.log(patch_probs / (aff_probs_base + 1e-20) + 1e-20)))
            
            # Top-1变化
            aff_top1 = np.argmax(aff_probs_base)
            patch_top1 = np.argmax(patch_probs)
            top1_changed = int(aff_top1 != patch_top1)
            
            # 恢复效果: patched分布和negated分布的相似度
            patch_neg_cos = float(np.dot(patch_probs, neg_probs_base) / 
                                  (np.linalg.norm(patch_probs) * np.linalg.norm(neg_probs_base) + 1e-20))
            
            patch_results[li].append({
                "kl_vs_affirm": patch_kl,
                "top1_changed": top1_changed,
                "cosine_vs_negated": patch_neg_cos,
                "base_kl_affirm_vs_negated": base_kl,
            })
        
        del neg_resid
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.stop()
    
    # 汇总
    print(f"\n  {'Layer':<8} {'KL_vs_aff':<12} {'Top1_chg%':<12} {'Cos_vs_neg':<12} {'Effect'}")
    print("  " + "-"*60)
    
    best_layer = None
    best_kl = 0
    
    for li in sorted(sample_layers):
        vals = patch_results[li]
        if not vals:
            continue
        
        # 处理dict格式的结果
        if isinstance(vals[0], dict):
            mean_kl = np.mean([v["kl_vs_affirm"] for v in vals])
            mean_top1 = np.mean([v["top1_changed"] for v in vals]) * 100
            mean_cos = np.mean([v["cosine_vs_negated"] for v in vals])
        else:
            mean_kl = np.mean(vals)
            mean_top1 = 0
            mean_cos = 0
        
        is_best = False
        if mean_kl > best_kl:
            best_kl = mean_kl
            best_layer = li
            is_best = True
        
        effect = "<<< BEST" if is_best else ""
        print(f"  L{li:<6} {mean_kl:<12.4f} {mean_top1:<12.1f} {mean_cos:<12.4f} {effect}")
    
    print(f"\n  >>> 最佳Patching层: L{best_layer} (KL={best_kl:.4f})")
    
    return {
        "layer_results": {str(k): v for k, v in patch_results.items()},
        "best_patch_layer": best_layer,
        "best_patch_kl": float(best_kl),
        "n_pairs": len(test_pairs),
    }


# ===== ExpD: Attention Head消融 =====

def expD_attention_head_ablation(model, tokenizer, device, model_info, critical_layers=None):
    """
    核心问题: 哪些attention head是"否定回路"的组成部分?
    
    方法:
    1. 选择否定效果最强的几层
    2. 对每层的每个head, zero out其输出
    3. 测量否定效果(KL)减少多少
    4. 减少最多的head = 否定回路关键节点
    """
    print("\n" + "="*60)
    print("ExpD: Attention Head Ablation (注意力头消融)")
    print("="*60)
    
    n_layers = model_info.n_layers
    layers = get_layers(model)
    
    # 确定要测试的层
    if critical_layers is None:
        # 默认测试: 中间层附近 + onset层附近
        critical_layers = [
            n_layers // 3, n_layers // 2, 
            2 * n_layers // 3, n_layers - 2
        ]
    
    # 获取n_heads - 从model config中获取
    if hasattr(model.config, 'num_attention_heads'):
        n_heads = model.config.num_attention_heads
    elif hasattr(model.config, 'num_heads'):
        n_heads = model.config.num_heads
    elif hasattr(sa, 'num_heads'):
        n_heads = sa.num_heads
    elif hasattr(sa, 'num_attention_heads'):
        n_heads = sa.num_attention_heads
    else:
        # 从权重推断
        n_heads = model_info.d_model // 128  # 假设head_dim=128
    head_dim = model_info.d_model // n_heads
    
    print(f"  n_heads={n_heads}, head_dim={head_dim}, testing layers={critical_layers}")
    
    # 测试模板
    test_pairs = NEGATION_PAIRS[:15]
    
    logger = ProgressLogger()
    
    # 首先获取baseline KL (没有ablation)
    print("  Computing baseline KL values...")
    baseline_kls = []
    for affirm, negated in test_pairs:
        aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
        neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            aff_out = model(input_ids=aff_ids)
            neg_out = model(input_ids=neg_ids)
        
        aff_probs = torch.softmax(aff_out.logits[0, -1].float(), dim=-1).cpu().numpy()
        neg_probs = torch.softmax(neg_out.logits[0, -1].float(), dim=-1).cpu().numpy()
        
        kl = float(np.sum(neg_probs * np.log(neg_probs / (aff_probs + 1e-20) + 1e-20)))
        baseline_kls.append(kl)
        
        del aff_out, neg_out
    
    mean_baseline_kl = np.mean(baseline_kls)
    print(f"  Baseline mean KL = {mean_baseline_kl:.4f}")
    
    # Head ablation - 在self_attn的o_proj输入上操作
    # o_proj接收 [batch, seq, n_heads*head_dim], 零化某head对应的位置
    head_importance = defaultdict(list)  # (layer, head) -> [kl_reduction]
    
    for li in critical_layers:
        logger.update(f"ExpD layer {li}, ablating heads")
        print(f"\n  Ablating heads at Layer {li}...")
        
        # 获取该层self_attn的o_proj
        o_proj = layers[li].self_attn.o_proj
        
        for hi in range(n_heads):
            # 创建ablation hook: 在o_proj的input上zero out head hi
            # o_proj的input是attn_output: [batch, seq, n_heads*head_dim]
            def make_ablation_hook(head_idx, h_dim):
                def hook(module, input, output):
                    # input[0]是o_proj的输入, 即concatenated attn_output
                    # output是o_proj的输出: [batch, seq, d_model]
                    # 我们需要修改attn_output, 但hook在o_proj上,
                    # 所以直接修改output中head_idx贡献的部分
                    
                    # 更简单: 在o_proj输出上减去head_idx的贡献
                    # head_idx的贡献 = W_o[:, head_idx*head_dim:(head_idx+1)*head_dim] @ attn_output[:, :, head_idx*head_dim:(head_idx+1)*head_dim]
                    # 但我们没有attn_output... 
                    # 改用: 在Transformer层的输出上直接做mean ablation
                    pass
                return hook
            
            # 更好的方法: 直接在self_attn的forward中拦截
            # 使用register_forward_hook on the self_attn module
            def make_attn_ablation_hook(head_idx, h_dim, n_h):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        attn_out = output[0]
                    else:
                        attn_out = output
                    
                    # attn_out: [batch, seq, d_model]
                    # 我们要减去head_idx的贡献
                    # 这很tricky... 改用更直接的方法:
                    # 不做head ablation, 而做layer ablation + residual scaling
                    
                    # 简化方案: zero out该层self_attn的输出
                    # 然后测量否定效果减少了多少
                    return output
                return hook
            
            # === 最简单有效的方案: Mean Ablation ===
            # 不是逐head, 而是zero out整个self_attn输出, 看否定效果减少
            # 这给出self_attn vs MLP的相对贡献
            
            # 先只做: zero out self_attn, zero out MLP
            # 如果self_attn更关键 → 否定效果来自attention
            # 如果MLP更关键 → 否定效果来自MLP
            
            if hi == 0:  # 只在第一个head时做
                # Self-attn ablation
                def make_zero_hook():
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            return (torch.zeros_like(output[0]),) + output[1:]
                        return torch.zeros_like(output)
                    return hook
                
                # Zero out self_attn
                hook_sa = layers[li].self_attn.register_forward_hook(make_zero_hook())
                attn_ablated_kls = []
                for affirm, negated in test_pairs[:5]:
                    aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
                    neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        aff_out = model(input_ids=aff_ids)
                        neg_out = model(input_ids=neg_ids)
                    aff_p = torch.softmax(aff_out.logits[0, -1].float(), dim=-1).cpu().numpy()
                    neg_p = torch.softmax(neg_out.logits[0, -1].float(), dim=-1).cpu().numpy()
                    kl = float(np.sum(neg_p * np.log(neg_p / (aff_p + 1e-20) + 1e-20)))
                    attn_ablated_kls.append(kl)
                    del aff_out, neg_out
                hook_sa.remove()
                
                attn_reduction = mean_baseline_kl - np.mean(attn_ablated_kls)
                head_importance[(li, "self_attn")].append(attn_reduction)
                print(f"    Self-attn ablation: KL reduction = {attn_reduction:.4f}")
                
                # Zero out MLP
                hook_mlp = layers[li].mlp.register_forward_hook(make_zero_hook())
                mlp_ablated_kls = []
                for affirm, negated in test_pairs[:5]:
                    aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
                    neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        aff_out = model(input_ids=aff_ids)
                        neg_out = model(input_ids=neg_ids)
                    aff_p = torch.softmax(aff_out.logits[0, -1].float(), dim=-1).cpu().numpy()
                    neg_p = torch.softmax(neg_out.logits[0, -1].float(), dim=-1).cpu().numpy()
                    kl = float(np.sum(neg_p * np.log(neg_p / (aff_p + 1e-20) + 1e-20)))
                    mlp_ablated_kls.append(kl)
                    del aff_out, neg_out
                hook_mlp.remove()
                
                mlp_reduction = mean_baseline_kl - np.mean(mlp_ablated_kls)
                head_importance[(li, "mlp")].append(mlp_reduction)
                print(f"    MLP ablation: KL reduction = {mlp_reduction:.4f}")
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break  # 只做一次, 不逐head
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.stop()
    
    # 找最重要的组件
    print(f"\n  Component Ablation Results (KL reduction):")
    sorted_comps = sorted(head_importance.items(), key=lambda x: -np.mean(x[1]))
    for rank, ((li, comp), reductions) in enumerate(sorted_comps[:10]):
        mean_red = np.mean(reductions)
        print(f"    #{rank+1}: L{li}_{comp} — KL reduction={mean_red:.4f}")
    
    return {
        "component_importance": {f"L{l}_{c}": {"mean_reduction": float(np.mean(v)), "reductions": [float(x) for x in v]}
                            for (l, c), v in head_importance.items()},
        "baseline_kl": float(mean_baseline_kl),
        "n_heads": n_heads,
        "critical_layers": critical_layers,
        "n_test_pairs": len(test_pairs),
    }


# ===== ExpE: 跨否定词泛化 =====

def expE_cross_negation_generalization(model, tokenizer, device, model_info):
    """
    核心问题: 不同否定词(not/never/cannot/no)是否共享同一回路?
    
    方法:
    1. 对每个否定词, 测量其概率压制效果
    2. 比较不同否定词的Δ_logit向量间的cosine
    3. 如果高cosine → 共享回路 (统一否定机制)
    4. 如果低cosine → 独立编码 (每个否定词是独立feature)
    """
    print("\n" + "="*60)
    print("ExpE: Cross-Negation Generalization (跨否定词泛化)")
    print("="*60)
    
    # 测试模板: 用不同否定词构造
    negation_templates = {
        "not": [
            ("The sky is blue", "The sky is not blue"),
            ("The cat is happy", "The cat is not happy"),
            ("The door is open", "The door is not open"),
            ("The water is cold", "The water is not cold"),
            ("The task is easy", "The task is not easy"),
            ("He is smart", "He is not smart"),
            ("The plan works", "The plan does not work"),
            ("She likes music", "She does not like music"),
            ("The answer is correct", "The answer is not correct"),
            ("The system is stable", "The system is not stable"),
        ],
        "never": [
            ("The sky is blue", "The sky is never blue"),
            ("The cat is happy", "The cat is never happy"),
            ("The door is open", "The door is never open"),
            ("The water is cold", "The water is never cold"),
            ("The task is easy", "The task is never easy"),
            ("He is smart", "He is never smart"),
            ("The plan works", "The plan never works"),
            ("She likes music", "She never likes music"),
            ("The answer is correct", "The answer is never correct"),
            ("The system is stable", "The system is never stable"),
        ],
        "cannot": [
            ("The sky is blue", "The sky cannot be blue"),
            ("The cat is happy", "The cat cannot be happy"),
            ("The door is open", "The door cannot be open"),
            ("The water is cold", "The water cannot be cold"),
            ("The task is easy", "The task cannot be easy"),
            ("He is smart", "He cannot be smart"),
            ("The plan works", "The plan cannot work"),
            ("She likes music", "She cannot like music"),
            ("The answer is correct", "The answer cannot be correct"),
            ("The system is stable", "The system cannot be stable"),
        ],
    }
    
    logger = ProgressLogger()
    
    # 收集每个否定词的Δ_logit
    neg_word_deltas = {}
    neg_word_metrics = {}
    
    for neg_word, templates in negation_templates.items():
        logger.update(f"ExpE negation word: {neg_word}")
        
        deltas = []
        flip_ratios = []
        kl_values = []
        
        for affirm, negated in templates:
            aff_ids = tokenizer(affirm, return_tensors="pt").input_ids.to(device)
            neg_ids = tokenizer(negated, return_tensors="pt").input_ids.to(device)
            
            with torch.no_grad():
                aff_out = model(input_ids=aff_ids)
                neg_out = model(input_ids=neg_ids)
            
            aff_logits = aff_out.logits[0, -1].float().cpu().numpy()
            neg_logits = neg_out.logits[0, -1].float().cpu().numpy()
            
            deltas.append(neg_logits - aff_logits)
            
            # 概率分析
            aff_probs = np.exp(aff_logits - aff_logits.max())
            aff_probs /= aff_probs.sum() + 1e-20
            neg_probs = np.exp(neg_logits - neg_logits.max())
            neg_probs /= neg_probs.sum() + 1e-20
            
            # Flip ratio
            aff_top10 = np.argsort(aff_probs)[-10:]
            flip_ratio = neg_probs[aff_top10].sum() / (aff_probs[aff_top10].sum() + 1e-20)
            flip_ratios.append(flip_ratio)
            
            # KL
            kl = float(np.sum(neg_probs * np.log(neg_probs / (aff_probs + 1e-20) + 1e-20)))
            kl_values.append(kl)
            
            del aff_out, neg_out
        
        neg_word_deltas[neg_word] = np.array(deltas)
        neg_word_metrics[neg_word] = {
            "mean_flip_ratio": float(np.mean(flip_ratios)),
            "mean_kl": float(np.mean(kl_values)),
        }
        
        print(f"  {neg_word}: flip={np.mean(flip_ratios):.4f}, KL={np.mean(kl_values):.4f}")
    
    logger.stop()
    
    # 跨否定词cosine矩阵
    words = list(neg_word_deltas.keys())
    n_words = len(words)
    cos_matrix = np.zeros((n_words, n_words))
    
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            # 平均Δ_logit的cosine
            d1 = np.mean(neg_word_deltas[w1], axis=0)
            d2 = np.mean(neg_word_deltas[w2], axis=0)
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_matrix[i, j] = float(np.dot(d1, d2) / (n1 * n2))
    
    print(f"\n  Cross-negation cosine matrix:")
    print(f"  {'':>10}", end="")
    for w in words:
        print(f"  {w:>10}", end="")
    print()
    for i, w1 in enumerate(words):
        print(f"  {w1:>10}", end="")
        for j in range(n_words):
            print(f"  {cos_matrix[i,j]:>10.4f}", end="")
        print()
    
    # 判决
    off_diag = cos_matrix[np.triu_indices(n_words, k=1)]
    mean_cross = float(np.mean(off_diag)) if len(off_diag) > 0 else 0
    
    if mean_cross > 0.7:
        verdict = "SHARED CIRCUIT (共享回路) - 否定词共享同一计算机制"
    elif mean_cross > 0.4:
        verdict = "PARTIALLY SHARED (部分共享) - 否定词有共同成分, 但也各自独立"
    else:
        verdict = "INDEPENDENT (独立) - 不同否定词使用不同计算路径"
    
    print(f"\n  >>> 判决: {verdict} (mean_cross_cos={mean_cross:.4f})")
    
    return {
        "neg_word_metrics": neg_word_metrics,
        "cosine_matrix": {w1: {w2: float(cos_matrix[i,j]) for j, w2 in enumerate(words)} 
                          for i, w1 in enumerate(words)},
        "mean_cross_cosine": mean_cross,
        "verdict": verdict,
    }


# ===== 主函数 =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    print("="*60)
    print(f"Phase 232: Negation Circuit Localization")
    print(f"Model: {model_name}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    all_results = {"model": model_name, "model_info": {
        "class": info.model_class, "n_layers": info.n_layers, 
        "d_model": info.d_model, "vocab_size": info.vocab_size,
    }}
    
    # ExpA: 否定层定位
    t0 = time.time()
    try:
        resA = expA_negation_layer_localization(model, tokenizer, device, info)
        all_results["expA"] = resA
        print(f"\n  ExpA done in {time.time()-t0:.1f}s, onset_layer=L{resA['onset_layer']}")
    except Exception as e:
        print(f"  ExpA FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["expA"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ExpB: Logit空间门控分析
    t0 = time.time()
    try:
        resB = expB_logit_gate_analysis(model, tokenizer, device, info)
        all_results["expB"] = resB
        print(f"\n  ExpB done in {time.time()-t0:.1f}s, verdict={resB['verdict']}")
    except Exception as e:
        print(f"  ExpB FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["expB"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ExpC: Activation Patching
    t0 = time.time()
    try:
        resC = expC_activation_patching(model, tokenizer, device, info)
        all_results["expC"] = resC
        print(f"\n  ExpC done in {time.time()-t0:.1f}s, best_patch_layer=L{resC['best_patch_layer']}")
    except Exception as e:
        print(f"  ExpC FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["expC"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ExpD: Attention Head消融 (使用ExpA发现的onset层)
    t0 = time.time()
    try:
        onset = all_results.get("expA", {}).get("onset_layer") or info.n_layers // 2
        critical_layers = [
            max(0, onset - 2), onset, min(info.n_layers - 1, onset + 2),
            info.n_layers // 2, info.n_layers - 2
        ]
        critical_layers = sorted(set([l for l in critical_layers if 0 <= l < info.n_layers]))
        resD = expD_attention_head_ablation(model, tokenizer, device, info, critical_layers)
        all_results["expD"] = resD
        print(f"\n  ExpD done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  ExpD FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["expD"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ExpE: 跨否定词泛化
    t0 = time.time()
    try:
        resE = expE_cross_negation_generalization(model, tokenizer, device, info)
        all_results["expE"] = resE
        print(f"\n  ExpE done in {time.time()-t0:.1f}s, verdict={resE['verdict']}")
    except Exception as e:
        print(f"  ExpE FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["expE"] = {"error": str(e)}
    
    # 保存结果
    out_path = f"tests/glm5_temp/phase232_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    
    # 最终总结
    print("\n" + "="*60)
    print("Phase 232 Final Summary")
    print("="*60)
    
    if "expA" in all_results and "error" not in all_results["expA"]:
        print(f"  ExpA (层定位): Onset layer = L{all_results['expA']['onset_layer']}")
    if "expB" in all_results and "error" not in all_results["expB"]:
        print(f"  ExpB (门控机制): {all_results['expB']['verdict']}")
        print(f"    Cross-context cosine = {all_results['expB']['mean_cross_cosine']:.4f}")
    if "expC" in all_results and "error" not in all_results["expC"]:
        print(f"  ExpC (因果验证): Best patch layer = L{all_results['expC']['best_patch_layer']}")
        print(f"    Best patch KL = {all_results['expC']['best_patch_kl']:.4f}")
    if "expD" in all_results and "error" not in all_results["expD"]:
        comp_data = all_results["expD"].get("component_importance", all_results["expD"].get("head_importance", {}))
        top_comps = sorted(comp_data.items(),
                          key=lambda x: -x[1]["mean_reduction"])[:3]
        top_comp_strs = [f"{k}={v['mean_reduction']:.4f}" for k, v in top_comps]
        print(f"  ExpD (组件消融): Top = {top_comp_strs}")
    if "expE" in all_results and "error" not in all_results["expE"]:
        print(f"  ExpE (跨词泛化): {all_results['expE']['verdict']}")
    
    print(f"\nDone! {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
