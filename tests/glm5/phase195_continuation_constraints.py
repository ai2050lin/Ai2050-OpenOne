"""
Phase 195: Continuation-Space Constraint Analysis
====================================================

核心转向: 从"ΔS向量对象化"到"续写空间约束结构"

用户关键洞察:
- ΔS不是"语义本体", 只是局部更新算子
- 真正语义 = 对可能续写的约束 (constraint on possible continuations)
- "not" 不是方向, 而是翻转可接受续写
- "question" 不是模态向量, 而是打开多个可能状态
- "role_binding" 不是交换向量, 而是改变依赖图

★ 关键测试: question == conditional (cos=1.0) 是否是均值塌缩伪象?

5个实验:
Exp1: Entropy Dynamics — 语义功能是否改变预测不确定性?
      prediction: negation减少熵, question增加熵, role_binding大幅改变top tokens
Exp2: Token-Level Constraints — 每个功能promote/suppress哪些tokens?
      question和conditional是否真的promote相同tokens?
Exp3: Per-Sample KL Divergence — 逐样本分析, 避免均值塌缩
      question vs conditional的KL是否在每个样本上都相似?
Exp4: Cross-Function Token Overlap (Jaccard) — question vs conditional
      promoted/suppressed token集合的Jaccard重叠
Exp5: Constraint Independence — 不同语义约束是否独立?
      negation+question的组合约束是否=各自约束之和?

数据量: 25句对/功能 (Qwen3), 15句对/功能 (GLM4/DS7B)
"""

import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import gc
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

from model_utils import (get_model_info, get_layers, release_model, MODEL_CONFIGS)


# ===== 20个基础句 (transitive, 用于所有5种变换) =====
# 格式: (base, negation, past_tense, role_swapped, question, conditional)
# 使用SAME base sentences for all functions, 但conditional也包含一批DIFFERENT bases做robustness check
SENTENCE_QUADS = [
    # (base, negation, past_tense, role_swapped, question, conditional)
    ("The cat chases the dog",
     "The cat does not chase the dog",
     "The cat chased the dog",
     "The dog chases the cat",
     "Does the cat chase the dog?",
     "If the cat chases the dog"),
    ("The teacher questions the student",
     "The teacher does not question the student",
     "The teacher questioned the student",
     "The student questions the teacher",
     "Does the teacher question the student?",
     "If the teacher questions the student"),
    ("The leader guides the team",
     "The leader does not guide the team",
     "The leader guided the team",
     "The team guides the leader",
     "Does the leader guide the team?",
     "If the leader guides the team"),
    ("The parent protects the child",
     "The parent does not protect the child",
     "The parent protected the child",
     "The child protects the parent",
     "Does the parent protect the child?",
     "If the parent protects the child"),
    ("The writer inspires the reader",
     "The writer does not inspire the reader",
     "The writer inspired the reader",
     "The reader inspires the writer",
     "Does the writer inspire the reader?",
     "If the writer inspires the reader"),
    ("The coach trains the athlete",
     "The coach does not train the athlete",
     "The coach trained the athlete",
     "The athlete trains the coach",
     "Does the coach train the athlete?",
     "If the coach trains the athlete"),
    ("The manager evaluates the employee",
     "The manager does not evaluate the employee",
     "The manager evaluated the employee",
     "The employee evaluates the manager",
     "Does the manager evaluate the employee?",
     "If the manager evaluates the employee"),
    ("The doctor advises the patient",
     "The doctor does not advise the patient",
     "The doctor advised the patient",
     "The patient advises the doctor",
     "Does the doctor advise the patient?",
     "If the doctor advises the patient"),
    ("The buyer pays the seller",
     "The buyer does not pay the seller",
     "The buyer paid the seller",
     "The seller pays the buyer",
     "Does the buyer pay the seller?",
     "If the buyer pays the seller"),
    ("The host welcomes the guest",
     "The host does not welcome the guest",
     "The host welcomed the guest",
     "The guest welcomes the host",
     "Does the host welcome the guest?",
     "If the host welcomes the guest"),
    ("The student challenges the professor",
     "The student does not challenge the professor",
     "The student challenged the professor",
     "The professor challenges the student",
     "Does the student challenge the professor?",
     "If the student challenges the professor"),
    ("The soldier follows the general",
     "The soldier does not follow the general",
     "The soldier followed the general",
     "The general follows the soldier",
     "Does the soldier follow the general?",
     "If the soldier follows the general"),
    ("The artist influences the critic",
     "The artist does not influence the critic",
     "The artist influenced the critic",
     "The critic influences the artist",
     "Does the artist influence the critic?",
     "If the artist influences the critic"),
    ("The driver transports the passenger",
     "The driver does not transport the passenger",
     "The driver transported the passenger",
     "The passenger transports the driver",
     "Does the driver transport the passenger?",
     "If the driver transports the passenger"),
    ("The chef feeds the guest",
     "The chef does not feed the guest",
     "The chef fed the guest",
     "The guest feeds the chef",
     "Does the chef feed the guest?",
     "If the chef feeds the guest"),
    ("The singer entertains the crowd",
     "The singer does not entertain the crowd",
     "The singer entertained the crowd",
     "The crowd entertains the singer",
     "Does the singer entertain the crowd?",
     "If the singer entertains the crowd"),
    ("The judge questions the witness",
     "The judge does not question the witness",
     "The judge questioned the witness",
     "The witness questions the judge",
     "Does the judge question the witness?",
     "If the judge questions the witness"),
    ("The editor corrects the author",
     "The editor does not correct the author",
     "The editor corrected the author",
     "The author corrects the editor",
     "Does the editor correct the author?",
     "If the editor corrects the author"),
    ("The therapist helps the client",
     "The therapist does not help the client",
     "The therapist helped the client",
     "The client helps the therapist",
     "Does the therapist help the client?",
     "If the therapist helps the client"),
    ("The scientist studies the phenomenon",
     "The scientist does not study the phenomenon",
     "The scientist studied the phenomenon",
     "The phenomenon studies the scientist",
     "Does the scientist study the phenomenon?",
     "If the scientist studies the phenomenon"),
    ("The company hires the worker",
     "The company does not hire the worker",
     "The company hired the worker",
     "The worker hires the company",
     "Does the company hire the worker?",
     "If the company hires the worker"),
    ("The river nourishes the valley",
     "The river does not nourish the valley",
     "The river nourished the valley",
     "The valley nourishes the river",
     "Does the river nourish the valley?",
     "If the river nourishes the valley"),
    ("The government regulates the market",
     "The government does not regulate the market",
     "The government regulated the market",
     "The market regulates the government",
     "Does the government regulate the market?",
     "If the government regulates the market"),
    ("The engine drives the machine",
     "The engine does not drive the machine",
     "The engine drove the machine",
     "The machine drives the engine",
     "Does the engine drive the machine?",
     "If the engine drives the machine"),
    ("The teacher encourages the student",
     "The teacher does not encourage the student",
     "The teacher encouraged the student",
     "The student encourages the teacher",
     "Does the teacher encourage the student?",
     "If the teacher encourages the student"),
]

# Robustness check: 10个用DIFFERENT base的conditional句对
# 目的: 测试question=conditional是否是base sentence共享导致的伪象
CONDITIONAL_DIFFERENT_BASES = [
    ("It rains heavily tomorrow", "If it rains heavily tomorrow"),
    ("The price increases rapidly", "If the price increases rapidly"),
    ("She arrives early tomorrow", "If she arrives early tomorrow"),
    ("The system crashes completely", "If the system crashes completely"),
    ("He wins the race easily", "If he wins the race easily"),
    ("The temperature drops suddenly", "If the temperature drops suddenly"),
    ("They cancel the event", "If they cancel the event"),
    ("The machine breaks down", "If the machine breaks down"),
    ("She finds the solution", "If she finds the solution"),
    ("The experiment succeeds", "If the experiment succeeds"),
]

FUNC_TYPES = ["negation", "tense", "role_binding", "question", "conditional"]
TOP_K = 20  # top-k tokens to analyze


def load_model_with_flash(model_name):
    """加载模型, GLM4/DS7B用device_map=auto, Qwen3全GPU, 尝试flash attention"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试flash attention, 失败则fallback
    attn_impl = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=attn_impl,
        )
        print(f"[load] {model_name} loaded with flash_attention_2")
    except Exception as e:
        print(f"[load] flash_attention_2 failed ({e}), falling back to eager")
        attn_impl = "eager"
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
    print(f"[load] {model_name}: device={device}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device


def _load_weight_from_safetensors(model_path, key):
    """从safetensors文件中加载指定key的权重"""
    import glob
    from safetensors import safe_open
    sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    for sf_file in sf_files:
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            if key in sf.keys():
                w = sf.get_tensor(key)
                return w.float()
    return None


def get_logit_lens(model, tokenizer, text, n_layers, device, model_name=None):
    """
    Logit lens: 在每层residual stream上应用final LayerNorm + lm_head,
    得到每层的next-token预测分布

    修复: 处理device_map=auto下norm/lm_head在meta device上的情况
    - 从safetensors直接加载权重
    - 在CPU上计算避免accelerate hook

    Returns:
        layer_probs: list of numpy arrays [n_layers], each shape [vocab_size]
        layer_entropies: list of floats [n_layers]
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 确定输入设备
    input_device = next(model.parameters()).device
    input_ids = input_ids.to(input_device)
    attention_mask = attention_mask.to(input_device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False)

    # 获取final LayerNorm权重 (处理meta device)
    norm_weight = None
    norm_eps = 1e-6  # 默认RMSNorm eps

    # 从模型config获取eps
    if hasattr(model, 'config') and hasattr(model.config, 'rms_norm_eps'):
        norm_eps = model.config.rms_norm_eps
    elif hasattr(model, 'config') and hasattr(model.config, 'layer_norm_epsilon'):
        norm_eps = model.config.layer_norm_epsilon

    norm_module = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        norm_module = model.model.norm

    if norm_module is not None:
        w = norm_module.weight
        if not w.is_meta:
            norm_weight = w.detach().float().cpu()
        else:
            # 从safetensors加载
            model_path = MODEL_CONFIGS.get(model_name, {}).get("path")
            if model_path:
                norm_weight = _load_weight_from_safetensors(model_path, "model.norm.weight")
                if norm_weight is not None:
                    print(f"  [logit_lens] Loaded norm weight from safetensors, shape={norm_weight.shape}")

    # 获取lm_head权重 (处理meta device)
    lm_weight = None
    if hasattr(model, 'lm_head'):
        w = model.lm_head.weight
        if not w.is_meta:
            lm_weight = w.detach().float().cpu()
        else:
            # 从safetensors加载
            model_path = MODEL_CONFIGS.get(model_name, {}).get("path")
            if model_path:
                lm_weight = _load_weight_from_safetensors(model_path, "lm_head.weight")
                if lm_weight is not None:
                    print(f"  [logit_lens] Loaded lm_head weight from safetensors, shape={lm_weight.shape}")

    if norm_weight is None or lm_weight is None:
        print(f"  [WARN] Cannot get norm/lm_head weights for logit lens, skipping")
        return None, None

    # 最后一个有效token的位置
    last_pos = attention_mask.sum().item() - 1

    layer_probs = []
    layer_entropies = []

    for layer_idx, hidden in enumerate(outputs.hidden_states):
        # hidden: [1, seq_len, d_model]
        h_last = hidden[:, last_pos, :].float().cpu()  # [1, d_model]

        # 手动应用RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        rms = torch.sqrt(torch.mean(h_last ** 2, dim=-1, keepdim=True) + norm_eps)
        h_normed = (h_last / rms) * norm_weight

        # 手动应用lm_head: logits = h_normed @ lm_weight.T
        logits = F.linear(h_normed, lm_weight)  # [1, vocab_size]
        probs = F.softmax(logits.squeeze(0), dim=-1).detach().numpy()  # [vocab_size]

        # 计算熵
        probs_clipped = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped))

        layer_probs.append(probs)
        layer_entropies.append(float(entropy))

    return layer_probs, layer_entropies


def compute_top_k_info(probs, tokenizer, k=TOP_K):
    """获取top-k tokens的信息: token_id, token_text, probability"""
    top_k_ids = np.argsort(probs)[-k:][::-1]
    result = []
    for tid in top_k_ids:
        text = tokenizer.decode([tid]).strip()
        result.append({
            "id": int(tid),
            "text": text,
            "prob": float(probs[tid])
        })
    return result


def compute_kl_divergence(p, q):
    """KL(p || q) — p是base分布, q是transformed分布"""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def compute_jaccard(set_a, set_b):
    """Jaccard overlap of two sets"""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def run_experiment(model_name):
    """运行Phase 195完整实验"""
    t0_total = time.time()
    print(f"\n{'='*70}")
    print(f"Phase 195: Continuation-Space Constraint Analysis — {model_name}")
    print(f"{'='*70}")

    # 加载模型
    model, tokenizer, device = load_model_with_flash(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  n_layers={n_layers}, d_model={info.d_model}, vocab={info.vocab_size}")
    sys.stdout.flush()

    # 采样层数 (device_map=auto模型减少层数)
    if model_name == 'qwen3':
        n_pairs = 25
        sample_layers = list(range(n_layers))
    else:
        n_pairs = 15
        step = max(1, n_layers // 12)
        sample_layers = sorted(set(list(range(0, n_layers, step)) + [n_layers - 1]))
    print(f"  n_pairs={n_pairs}, sample_layers={len(sample_layers)} layers")
    sys.stdout.flush()

    results = {
        "model": model_name,
        "n_layers": n_layers,
        "n_pairs": n_pairs,
        "sample_layers": sample_layers,
        "exp1_entropy": {},
        "exp2_token_constraints": {},
        "exp3_per_sample_kl": {},
        "exp4_cross_function_overlap": {},
        "exp5_constraint_independence": {},
    }

    # ========== 数据收集 ==========
    # 对每个功能, 存储每层每个句对的probs和entropy
    # all_layer_probs[func_type][pair_idx][layer_idx] = np.array[vocab_size]
    all_layer_probs = defaultdict(lambda: defaultdict(dict))
    all_layer_entropies = defaultdict(lambda: defaultdict(dict))

    # 对每个功能, 存储每层的mean entropy
    func_mean_entropy = {}  # func_type -> [n_layers] (mean over pairs)

    # Base sentences (共享)
    base_probs = defaultdict(lambda: defaultdict(dict))  # pair_idx -> layer_idx -> probs
    base_entropies = defaultdict(lambda: defaultdict(dict))

    func_to_idx = {"negation": 1, "tense": 2, "role_binding": 3, "question": 4, "conditional": 5}

    print(f"\n--- Collecting logit lens data ---")
    sys.stdout.flush()

    for pi in range(min(n_pairs, len(SENTENCE_QUADS))):
        quad = SENTENCE_QUADS[pi]
        base_text = quad[0]

        # 处理base sentence
        base_p, base_e = get_logit_lens(model, tokenizer, base_text, n_layers, device, model_name)
        if base_p is not None:
            for li, layer_idx in enumerate(sample_layers):
                base_probs[pi][layer_idx] = base_p[layer_idx]
                base_entropies[pi][layer_idx] = base_e[layer_idx]

        if pi % 5 == 0:
            elapsed = time.time() - t0_total
            print(f"  Base {pi}/{n_pairs} ({elapsed:.0f}s): {base_text[:40]}...")
            sys.stdout.flush()

        torch.cuda.empty_cache()
        gc.collect()

        # 处理每个功能的transformed sentence
        for ft in FUNC_TYPES:
            idx = func_to_idx[ft]
            trans_text = quad[idx]

            trans_p, trans_e = get_logit_lens(model, tokenizer, trans_text, n_layers, device, model_name)
            if trans_p is not None:
                for li, layer_idx in enumerate(sample_layers):
                    all_layer_probs[ft][pi][layer_idx] = trans_p[layer_idx]
                    all_layer_entropies[ft][pi][layer_idx] = trans_e[layer_idx]

            torch.cuda.empty_cache()
            gc.collect()

        if pi % 3 == 0:
            elapsed = time.time() - t0_total
            print(f"    All functions done for pair {pi} ({elapsed:.0f}s)")
            sys.stdout.flush()

    # 处理conditional不同base的robustness check
    print(f"\n--- Conditional robustness check (different bases) ---")
    sys.stdout.flush()
    cond_diff_probs = defaultdict(lambda: defaultdict(dict))
    cond_diff_base_probs = defaultdict(lambda: defaultdict(dict))

    for ci, (base_text, cond_text) in enumerate(CONDITIONAL_DIFFERENT_BASES[:min(10, n_pairs)]):
        bp, be = get_logit_lens(model, tokenizer, base_text, n_layers, device, model_name)
        cp, ce = get_logit_lens(model, tokenizer, cond_text, n_layers, device, model_name)

        if bp is not None and cp is not None:
            for li, layer_idx in enumerate(sample_layers):
                cond_diff_base_probs[ci][layer_idx] = bp[layer_idx]
                cond_diff_probs[ci][layer_idx] = cp[layer_idx]

        torch.cuda.empty_cache()
        gc.collect()

        if ci % 3 == 0:
            elapsed = time.time() - t0_total
            print(f"  Cond-diff {ci}/10 ({elapsed:.0f}s): {base_text[:40]}...")
            sys.stdout.flush()

    # ========== Exp1: Entropy Dynamics ==========
    print(f"\n--- Exp1: Entropy Dynamics ---")
    sys.stdout.flush()

    for ft in FUNC_TYPES:
        entropy_changes = []  # [n_layers], each is mean over pairs
        for layer_idx in sample_layers:
            changes = []
            for pi in range(min(n_pairs, len(SENTENCE_QUADS))):
                if layer_idx in all_layer_entropies[ft][pi] and layer_idx in base_entropies[pi]:
                    delta_e = all_layer_entropies[ft][pi][layer_idx] - base_entropies[pi][layer_idx]
                    changes.append(delta_e)
            entropy_changes.append(float(np.mean(changes)) if changes else 0.0)

        results["exp1_entropy"][ft] = {
            "mean_entropy_change": entropy_changes,
            "layers": sample_layers,
        }

        # 找到entropy变化最大的层
        if entropy_changes:
            max_idx = np.argmax(np.abs(entropy_changes))
            max_layer = sample_layers[max_idx]
            max_change = entropy_changes[max_idx]
            print(f"  {ft}: max |ΔH| at L{max_layer} = {max_change:.4f}")
    sys.stdout.flush()

    # ========== Exp2: Token-Level Constraints ==========
    print(f"\n--- Exp2: Token-Level Constraints ---")
    sys.stdout.flush()

    # 对最后几层(最接近输出), 分析top-k token变化
    analysis_layers = sample_layers[-5:] if len(sample_layers) >= 5 else sample_layers[-3:]

    for ft in FUNC_TYPES:
        promoted_tokens = defaultdict(int)  # token_id -> count (promoted in how many pairs)
        suppressed_tokens = defaultdict(int)

        for layer_idx in analysis_layers:
            for pi in range(min(n_pairs, len(SENTENCE_QUADS))):
                if layer_idx not in all_layer_probs[ft][pi] or layer_idx not in base_probs[pi]:
                    continue

                trans_p = all_layer_probs[ft][pi][layer_idx]
                base_p = base_probs[pi][layer_idx]

                # Top-k tokens in transformed vs base
                trans_top = set(np.argsort(trans_p)[-TOP_K:][::-1].tolist())
                base_top = set(np.argsort(base_p)[-TOP_K:][::-1].tolist())

                # Promoted: in transformed top-k but not in base top-k
                for tid in trans_top - base_top:
                    promoted_tokens[tid] += 1
                # Suppressed: in base top-k but not in transformed top-k
                for tid in base_top - trans_top:
                    suppressed_tokens[tid] += 1

        # 取出现频率最高的promoted/suppressed tokens
        top_promoted = sorted(promoted_tokens.items(), key=lambda x: x[1], reverse=True)[:10]
        top_suppressed = sorted(suppressed_tokens.items(), key=lambda x: x[1], reverse=True)[:10]

        promoted_info = [(tokenizer.decode([tid]).strip(), count) for tid, count in top_promoted]
        suppressed_info = [(tokenizer.decode([tid]).strip(), count) for tid, count in top_suppressed]

        results["exp2_token_constraints"][ft] = {
            "top_promoted": promoted_info,
            "top_suppressed": suppressed_info,
            "analysis_layers": analysis_layers,
        }

        print(f"  {ft}:")
        print(f"    Promoted: {promoted_info[:5]}")
        print(f"    Suppressed: {suppressed_info[:5]}")
    sys.stdout.flush()

    # ========== Exp3: Per-Sample KL Divergence ==========
    print(f"\n--- Exp3: Per-Sample KL Divergence ---")
    sys.stdout.flush()

    # 对最后几层, 计算每个句对的KL(base || transformed)
    per_sample_kl = {}  # ft -> [n_pairs] -> [n_analysis_layers]
    for ft in FUNC_TYPES:
        pair_kls = []
        for pi in range(min(n_pairs, len(SENTENCE_QUADS))):
            layer_kls = []
            for layer_idx in analysis_layers:
                if layer_idx not in all_layer_probs[ft][pi] or layer_idx not in base_probs[pi]:
                    layer_kls.append(None)
                    continue
                kl = compute_kl_divergence(base_probs[pi][layer_idx],
                                            all_layer_probs[ft][pi][layer_idx])
                layer_kls.append(kl)
            pair_kls.append(layer_kls)

        per_sample_kl[ft] = pair_kls

        # 计算mean和std
        valid_kls = [kl for pair_kls_list in pair_kls for kl in pair_kls_list if kl is not None]
        mean_kl = float(np.mean(valid_kls)) if valid_kls else 0.0
        std_kl = float(np.std(valid_kls)) if valid_kls else 0.0
        print(f"  {ft}: mean KL = {mean_kl:.4f} ± {std_kl:.4f} (n={len(valid_kls)})")

    results["exp3_per_sample_kl"] = {
        ft: {
            "mean": float(np.mean([kl for pkl in pair_kls for kl in pkl if kl is not None])),
            "std": float(np.std([kl for pkl in pair_kls for kl in pkl if kl is not None])),
            "per_pair_means": [float(np.mean([kl for kl in pkl if kl is not None])) if any(kl is not None for kl in pkl) else 0.0 for pkl in pair_kls],
        }
        for ft, pair_kls in per_sample_kl.items()
    }

    # ★ 关键测试: question vs conditional 的逐样本KL相关性
    if "question" in per_sample_kl and "conditional" in per_sample_kl:
        q_means = results["exp3_per_sample_kl"]["question"]["per_pair_means"]
        c_means = results["exp3_per_sample_kl"]["conditional"]["per_pair_means"]
        # Pearson correlation
        if len(q_means) > 2 and len(c_means) > 2:
            q_arr = np.array(q_means)
            c_arr = np.array(c_means)
            if np.std(q_arr) > 1e-10 and np.std(c_arr) > 1e-10:
                corr = float(np.corrcoef(q_arr, c_arr)[0, 1])
            else:
                corr = 0.0
            print(f"\n  ★ Per-sample KL correlation (question vs conditional): r = {corr:.4f}")
            print(f"    question per-pair KL: {[f'{v:.3f}' for v in q_means[:5]]}...")
            print(f"    conditional per-pair KL: {[f'{v:.3f}' for v in c_means[:5]]}...")
            results["exp4_cross_function_overlap"]["per_sample_kl_corr"] = corr
    sys.stdout.flush()

    # ========== Exp4: Cross-Function Token Overlap (Jaccard) ==========
    print(f"\n--- Exp4: Cross-Function Token Overlap ---")
    sys.stdout.flush()

    # 对最后几层, 计算每对功能的top-k token Jaccard
    func_top_tokens = {}  # ft -> set of top token IDs (aggregated over pairs and layers)
    for ft in FUNC_TYPES:
        token_set = set()
        for layer_idx in analysis_layers:
            for pi in range(min(n_pairs, len(SENTENCE_QUADS))):
                if layer_idx not in all_layer_probs[ft][pi]:
                    continue
                top_ids = set(np.argsort(all_layer_probs[ft][pi][layer_idx])[-TOP_K:][::-1].tolist())
                token_set.update(top_ids)
        func_top_tokens[ft] = token_set

    # 计算Jaccard矩阵
    jaccard_matrix = {}
    for ft1 in FUNC_TYPES:
        for ft2 in FUNC_TYPES:
            if ft1 >= ft2:
                continue
            j = compute_jaccard(func_top_tokens[ft1], func_top_tokens[ft2])
            jaccard_matrix[f"{ft1}_vs_{ft2}"] = j

    # 也做per-layer per-pair的Jaccard
    detailed_jaccard = {}
    for layer_idx in analysis_layers:
        for ft1 in FUNC_TYPES:
            for ft2 in FUNC_TYPES:
                if ft1 >= ft2:
                    continue
                jaccards = []
                for pi in range(min(n_pairs, len(SENTENCE_QUADS))):
                    if (layer_idx not in all_layer_probs[ft1][pi] or
                        layer_idx not in all_layer_probs[ft2][pi]):
                        continue
                    top1 = set(np.argsort(all_layer_probs[ft1][pi][layer_idx])[-TOP_K:][::-1].tolist())
                    top2 = set(np.argsort(all_layer_probs[ft2][pi][layer_idx])[-TOP_K:][::-1].tolist())
                    jaccards.append(compute_jaccard(top1, top2))
                if jaccards:
                    key = f"L{layer_idx}_{ft1}_vs_{ft2}"
                    detailed_jaccard[key] = float(np.mean(jaccards))

    # ★ 关键: question vs conditional的Jaccard
    q_vs_c_key = "question_vs_conditional"
    q_c_jaccard = jaccard_matrix.get(q_vs_c_key, 0.0)
    print(f"  ★ question vs conditional Jaccard (aggregate): {q_c_jaccard:.4f}")
    for key, j in sorted(jaccard_matrix.items()):
        print(f"  {key}: {j:.4f}")

    results["exp4_cross_function_overlap"]["jaccard_matrix"] = jaccard_matrix
    results["exp4_cross_function_overlap"]["detailed_jaccard"] = detailed_jaccard

    # ★ Robustness check: conditional with different bases
    print(f"\n--- Robustness: Conditional with different bases ---")
    sys.stdout.flush()
    # 比较相同base的conditional vs 不同base的conditional的token overlap
    same_base_cond_tokens = func_top_tokens.get("conditional", set())
    diff_base_cond_tokens = set()
    for ci in range(min(10, len(CONDITIONAL_DIFFERENT_BASES))):
        for layer_idx in analysis_layers:
            if layer_idx in cond_diff_probs[ci]:
                top_ids = set(np.argsort(cond_diff_probs[ci][layer_idx])[-TOP_K:][::-1].tolist())
                diff_base_cond_tokens.update(top_ids)

    if same_base_cond_tokens and diff_base_cond_tokens:
        j_same_diff = compute_jaccard(same_base_cond_tokens, diff_base_cond_tokens)
        print(f"  Jaccard(same-base cond, diff-base cond): {j_same_diff:.4f}")
        results["exp4_cross_function_overlap"]["cond_same_vs_diff_base_jaccard"] = j_same_diff
    sys.stdout.flush()

    # ========== Exp5: Constraint Independence ==========
    print(f"\n--- Exp5: Constraint Independence ---")
    sys.stdout.flush()

    # 对每个层, 计算5个功能的KL矩阵
    # 如果功能独立, KL(A||B)应该约= KL(A||C) + KL(C||B) (三角不等式边界)
    for layer_idx in [analysis_layers[-1]]:  # 只分析最后一层
        kl_matrix = {}
        for ft1 in FUNC_TYPES:
            for ft2 in FUNC_TYPES:
                kls = []
                for pi in range(min(n_pairs, len(SENTENCE_QUADS))):
                    if (layer_idx not in all_layer_probs[ft1][pi] or
                        layer_idx not in all_layer_probs[ft2][pi]):
                        continue
                    kl = compute_kl_divergence(all_layer_probs[ft1][pi][layer_idx],
                                               all_layer_probs[ft2][pi][layer_idx])
                    kls.append(kl)
                if kls:
                    kl_matrix[f"{ft1}_vs_{ft2}"] = float(np.mean(kls))

        results["exp5_constraint_independence"][f"L{layer_idx}"] = kl_matrix

        # 检查三角不等式: KL(q||c) vs KL(q||n) + KL(n||c)
        qn = kl_matrix.get("question_vs_negation", 0)
        nc = kl_matrix.get("negation_vs_conditional", 0)
        qc = kl_matrix.get("question_vs_conditional", 0)
        print(f"  L{layer_idx}: KL(q||c)={qc:.4f}, KL(q||n)+KL(n||c)={qn+nc:.4f}")
        print(f"    三角不等式: KL(q||c) <= KL(q||n)+KL(n||c)? {qc <= qn + nc + 0.01}")
    sys.stdout.flush()

    # ========== ΔS 诊断: Per-Sample Cosine (测试均值塌缩) ==========
    print(f"\n--- Diagnostic: Per-Sample ΔS Cosine (test mean collapse) ---")
    sys.stdout.flush()

    # 计算每层的mean ΔS向量 (用于比较与Phase 194)
    # 和per-sample ΔS cosine (用于测试均值塌缩)
    for layer_idx in [sample_layers[len(sample_layers)//2]]:  # 中间层
        delta_s_vectors = {}  # ft -> list of ΔS vectors (per pair)
        for ft in FUNC_TYPES:
            vectors = []
            for pi in range(min(n_pairs, len(SENTENCE_QUADS))):
                if layer_idx not in all_layer_probs[ft][pi] or layer_idx not in base_probs[pi]:
                    continue
                # ΔS = transformed_probs - base_probs (在概率空间的"变化")
                delta = all_layer_probs[ft][pi][layer_idx] - base_probs[pi][layer_idx]
                vectors.append(delta)
            delta_s_vectors[ft] = vectors

        # Per-sample cosine between question and conditional ΔS
        if "question" in delta_s_vectors and "conditional" in delta_s_vectors:
            q_vecs = delta_s_vectors["question"]
            c_vecs = delta_s_vectors["conditional"]
            per_sample_cos = []
            for q_v, c_v in zip(q_vecs, c_vecs):
                q_norm = np.linalg.norm(q_v)
                c_norm = np.linalg.norm(c_v)
                if q_norm > 1e-10 and c_norm > 1e-10:
                    cos = float(np.dot(q_v, c_v) / (q_norm * c_norm))
                    per_sample_cos.append(cos)

            if per_sample_cos:
                mean_cos = float(np.mean(per_sample_cos))
                std_cos = float(np.std(per_sample_cos))
                min_cos = float(np.min(per_sample_cos))
                max_cos = float(np.max(per_sample_cos))
                print(f"  L{layer_idx}: per-sample cos(ΔS_q, ΔS_c):")
                print(f"    mean={mean_cos:.4f}, std={std_cos:.4f}, min={min_cos:.4f}, max={max_cos:.4f}")
                print(f"    ★ 如果mean_cos << 1.0 (Phase 194的1.0), 则cos=1.0是均值塌缩伪象!")
                results["exp4_cross_function_overlap"]["per_sample_delta_cos"] = {
                    "layer": layer_idx,
                    "mean": mean_cos,
                    "std": std_cos,
                    "min": min_cos,
                    "max": max_cos,
                    "n_samples": len(per_sample_cos),
                }

        # 也计算mean ΔS的cosine (与Phase 194对比)
        mean_delta_cos = {}
        for ft1 in FUNC_TYPES:
            for ft2 in FUNC_TYPES:
                if ft1 >= ft2:
                    continue
                v1_list = delta_s_vectors.get(ft1, [])
                v2_list = delta_s_vectors.get(ft2, [])
                if v1_list and v2_list:
                    mean_v1 = np.mean(v1_list, axis=0)
                    mean_v2 = np.mean(v2_list, axis=0)
                    n1, n2 = np.linalg.norm(mean_v1), np.linalg.norm(mean_v2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos = float(np.dot(mean_v1, mean_v2) / (n1 * n2))
                    else:
                        cos = 0.0
                    mean_delta_cos[f"{ft1}_vs_{ft2}"] = cos

        results["exp4_cross_function_overlap"]["mean_delta_cos_L" + str(layer_idx)] = mean_delta_cos
        for key, cos in sorted(mean_delta_cos.items()):
            if "question" in key or "conditional" in key:
                print(f"    mean cos(ΔS): {key} = {cos:.4f}")
    sys.stdout.flush()

    # ========== 保存结果 ==========
    timestamp = time.strftime("%Y%m%d_%H%M")
    output_path = Path(f"tests/glm5_temp/phase195_{model_name}_{timestamp}.json")

    # 转换numpy类型为python类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj

    results = convert(results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {output_path}")

    # ========== 释放模型 ==========
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"Phase 195 COMPLETE for {model_name} ({elapsed:.0f}s)")
    print(f"{'='*70}")
    sys.stdout.flush()

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
