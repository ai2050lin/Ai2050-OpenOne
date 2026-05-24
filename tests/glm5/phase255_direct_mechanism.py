"""
Phase 255: 直接破解编码机制 — 从"统计方法"到"直接读取"的范式转换
==================================================================

核心转变: 不再从激活值的统计特性推断编码机制, 而是直接读取权重和因果链

五方案:
  Part 1 (255a): Superposition程度快速检验 — 前置步骤, 决定后续路径
  Part 2 (255b): MLP键值分析 — 直接读取权重, 建立神经元知识词典
  Part 3 (255c): W_U结构分析 — 理解"编码字典"的组织方式
  Part 4 (255d): Logit Attribution — 精确计算每层/组件对预测的贡献
  Part 5 (255e): 97%维度功能探索 — 回答最重要的未解问题

用法:
  python tests/glm5/phase255_direct_mechanism.py --model qwen3 --part 1
  python tests/glm5/phase255_direct_mechanism.py --model qwen3 --part all
  python tests/glm5/phase255_direct_mechanism.py --model glm4 --part 1
  python tests/glm5/phase255_direct_mechanism.py --model deepseek7b --part 1
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 结果目录
RESULT_DIR = Path("results/direct_mechanism")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 工具函数
# ============================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_model_safe(model_name):
    """加载模型, 所有模型用bfloat16 + device_map=auto, 开flash attention"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} from {cfg['path']}...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 所有模型用bfloat16 + device_map="auto", 先尝试flash_attention_2
    for attn_impl in ["flash_attention_2", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            log_time(f"Loaded with attn_implementation={attn_impl}")
            break
        except Exception as e:
            log_time(f"flash_attention_2 failed ({str(e)[:80]}), falling back to eager")
            if attn_impl == "eager":
                raise

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"Model loaded: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device

def release_model_safe(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released, GPU memory freed")

def get_mlp_weights(layer, mlp_type):
    """获取MLP的W_gate, W_up, W_down (float32 numpy)"""
    mlp = layer.mlp
    if mlp_type == "merged_gate_up":
        w = mlp.gate_up_proj.weight
        if w.is_meta:
            # meta tensor: 需要从safetensors加载, 跳过此层
            return None, None, None
        W_gate_up = w.detach().cpu().float().numpy()
        mid = W_gate_up.shape[0] // 2
        W_gate = W_gate_up[:mid]
        W_up = W_gate_up[mid:]
    else:
        w_gate = mlp.gate_proj.weight
        w_up = mlp.up_proj.weight
        if w_gate.is_meta or w_up.is_meta:
            return None, None, None
        W_gate = w_gate.detach().cpu().float().numpy()
        W_up = w_up.detach().cpu().float().numpy()
    w_down = mlp.down_proj.weight
    if w_down.is_meta:
        return None, None, None
    W_down = w_down.detach().cpu().float().numpy()
    return W_gate, W_up, W_down

def decode_direction_with_WU(direction, W_U, tokenizer, k=10):
    """用W_U解码一个方向向量, 返回top-k token和分数"""
    # direction: [d_model], W_U: [vocab, d_model]
    logits = W_U @ direction  # [vocab]
    top_ids = np.argsort(logits)[-k:][::-1]
    results = []
    for tid in top_ids:
        tok_str = tokenizer.decode([tid]).strip()
        results.append({"token": tok_str, "id": int(tid), "score": float(logits[tid])})
    return results

def save_result(model_name, part, data):
    fpath = RESULT_DIR / f"phase255_part{part}_{model_name}.json"
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
    log_time(f"Results saved to {fpath}")


# ============================================================
# Part 1: Superposition程度快速检验
# ============================================================

def part1_superposition_check(model_name):
    """
    前置步骤: 检验MLP神经元的Superposition程度
    
    方法: 对中间层(L14-L18)的200个随机MLP神经元,
    找到让每个神经元激活值最高的100个输入词,
    计算这100个词的语义一致性得分(词嵌入平均相互cosine)
    
    判定:
      高一致性(cosine>0.7): superposition轻微, 可直接做键值分析
      中一致性(0.3-0.7): superposition中等, 键值分析结合SAE
      低一致性(<0.3): superposition严重, 必须先做SAE
    """
    import torch
    import torch.nn.functional as F

    log_time(f"=== Part 1: Superposition Check for {model_name} ===")
    
    model, tokenizer, device = load_model_safe(model_name)
    from model_utils import get_model_info, get_layers, get_W_U
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    layers = get_layers(model)
    
    log_time(f"d_model={info.d_model}, n_layers={info.n_layers}, intermediate={info.intermediate_size}")
    
    # 选取中间层
    target_layers = list(range(max(0, info.n_layers//2 - 2), min(info.n_layers, info.n_layers//2 + 3)))
    log_time(f"Target layers: {target_layers}")
    
    # 准备词表: 选取高频词(避免特殊token)
    vocab_size = info.vocab_size
    # 用W_U的范数作为词频代理(范数大的token通常更常见)
    token_norms = np.linalg.norm(W_U, axis=1)
    # 排除前256个(通常是特殊token和字节)
    valid_ids = np.arange(256, vocab_size)
    # 选norm最大的5000个词
    valid_norms = token_norms[valid_ids]
    top_valid = valid_ids[np.argsort(valid_norms)[-5000:]]
    log_time(f"Selected {len(top_valid)} high-norm tokens for probing")
    
    # 词嵌入矩阵 (用W_U的行作为词向量代理)
    # W_U: [vocab, d_model], 每行是一个token的解码方向
    
    results = {"model": model_name, "d_model": info.d_model, 
               "n_layers": info.n_layers, "intermediate_size": info.intermediate_size}
    layer_results = {}
    
    for li in target_layers:
        log_time(f"  Analyzing Layer {li}...")
        layer = layers[li]
        W_gate, W_up, W_down = get_mlp_weights(layer, info.mlp_type)
        if W_gate is None:
            log_time(f"  Layer {li} weights on meta device, skipping")
            continue
        intermediate_size = W_gate.shape[0]
        n_sample = min(200, intermediate_size)
        neuron_ids = np.random.choice(intermediate_size, n_sample, replace=False)
        
        neuron_consistency = []
        n_high = 0  # cos>0.7
        n_mid = 0   # 0.3-0.7
        n_low = 0   # <0.3
        
        for idx, ni in enumerate(neuron_ids):
            if idx % 50 == 0:
                log_time(f"    Neuron {idx}/{n_sample}...")
            
            # W_gate[ni]: [d_model] — 这个神经元的"键"
            # 计算哪些输入词最激活这个神经元
            key_vec = W_gate[ni]  # [d_model]
            # gate激活 = ReLU(key_vec · h_in), h_in是LayerNorm后的residual
            # 用W_U的行作为h_in的代理(粗略但快速)
            # 更精确: 用W_gate[ni]与W_U行的点积
            activation_scores = W_U[top_valid] @ key_vec  # [5000]
            
            # ReLU后取top-100
            activation_scores_relu = np.maximum(activation_scores, 0)
            top100_idx = np.argsort(activation_scores_relu)[-100:][::-1]
            top100_global_ids = top_valid[top100_idx]
            
            # 计算这100个词的语义一致性
            top100_embeddings = W_U[top100_global_ids]  # [100, d_model]
            # 归一化
            norms = np.linalg.norm(top100_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            top100_normed = top100_embeddings / norms
            
            # 两两cosine的平均
            cos_matrix = top100_normed @ top100_normed.T  # [100, 100]
            # 取上三角(不含对角线)
            mask = np.triu(np.ones_like(cos_matrix), k=1).astype(bool)
            mean_cos = float(cos_matrix[mask].mean()) if mask.sum() > 0 else 0.0
            
            neuron_consistency.append({
                "neuron_id": int(ni),
                "consistency": mean_cos,
                "top5_tokens": [tokenizer.decode([tid]).strip() for tid in top100_global_ids[:5]],
            })
            
            if mean_cos > 0.7: n_high += 1
            elif mean_cos > 0.3: n_mid += 1
            else: n_low += 1
        
        total = len(neuron_consistency)
        mean_consistency = np.mean([nc["consistency"] for nc in neuron_consistency])
        
        layer_results[str(li)] = {
            "n_sampled": total,
            "mean_consistency": float(mean_consistency),
            "n_high_superposition_slight": n_high,
            "n_mid_superposition_moderate": n_mid,
            "n_low_superposition_severe": n_low,
            "ratio_high": float(n_high / total),
            "ratio_mid": float(n_mid / total),
            "ratio_low": float(n_low / total),
            "sample_neurons": neuron_consistency[:20],  # 只保存前20个详情
        }
        
        log_time(f"  L{li}: mean_consistency={mean_consistency:.3f}, "
                 f"high={n_high}({n_high/total:.0%}), "
                 f"mid={n_mid}({n_mid/total:.0%}), "
                 f"low={n_low}({n_low/total:.0%})")
    
    results["results_by_layer"] = layer_results
    
    # 总结判定
    all_consistencies = []
    for lr in layer_results.values():
        all_consistencies.append(lr["mean_consistency"])
    overall_mean = float(np.mean(all_consistencies))
    
    if overall_mean > 0.7:
        verdict = "SLIGHT_SUPERPOSITION - 可直接做键值分析"
    elif overall_mean > 0.3:
        verdict = "MODERATE_SUPERPOSITION - 键值分析结合SAE"
    else:
        verdict = "SEVERE_SUPERPOSITION - 必须先做SAE"
    
    results["overall_mean_consistency"] = overall_mean
    results["verdict"] = verdict
    log_time(f"\n*** VERDICT: {verdict} (mean_consistency={overall_mean:.3f}) ***\n")
    
    save_result(model_name, 1, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 2: MLP键值分析 — 直接读取权重
# ============================================================

def part2_mlp_key_value(model_name):
    """
    直接读取MLP权重, 建立神经元知识词典
    
    原理: MLP计算 = W_down · ReLU(W_gate · h) * W_up · h
    - W_gate的每一行 = "键": 什么输入模式激活这个神经元
    - W_down的每一列 = "值": 激活后往输出写什么
    
    方法: 对每层的W_gate行和W_down列, 用W_U解码top-k token
    
    复用和差异化:
    - 找到响应"苹果"的神经元集合和响应"香蕉"的神经元集合
    - 交集 = 共享神经元(复用), 差集 = 专属神经元(差异化)
    """
    import torch

    log_time(f"=== Part 2: MLP Key-Value Analysis for {model_name} ===")
    
    model, tokenizer, device = load_model_safe(model_name)
    from model_utils import get_model_info, get_layers, get_W_U
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    layers = get_layers(model)
    
    # 采样层: 早期/中间/后期
    sample_layers = sorted(set([0, info.n_layers//4, info.n_layers//2, 
                                3*info.n_layers//4, info.n_layers-1]))
    log_time(f"Sample layers: {sample_layers}")
    
    # 概念词集: 用于测量复用和差异化
    concept_sets = {
        "fruits": ["apple", "banana", "orange", "strawberry", "grape",
                    "mango", "cherry", "peach", "pear", "lemon"],
        "animals": ["dog", "cat", "horse", "elephant", "lion",
                     "bird", "fish", "snake", "rabbit", "whale"],
        "vehicles": ["car", "bus", "train", "airplane", "bicycle",
                      "truck", "boat", "motorcycle", "taxi", "ship"],
        "tools": ["hammer", "wrench", "screwdriver", "drill", "saw",
                   "pliers", "axe", "chisel", "ruler", "knife"],
    }
    
    # 获取概念词的token id
    concept_token_ids = {}
    for cat, words in concept_sets.items():
        ids = []
        for w in words:
            toks = tokenizer.encode(w, add_special_tokens=False)
            if toks:
                ids.append((w, toks[0]))
        concept_token_ids[cat] = ids
    
    results = {"model": model_name, "d_model": info.d_model, 
               "n_layers": info.n_layers, "intermediate_size": info.intermediate_size}
    layer_results = {}
    
    for li in sample_layers:
        log_time(f"  Analyzing Layer {li}...")
        layer = layers[li]
        W_gate, W_up, W_down = get_mlp_weights(layer, info.mlp_type)
        if W_gate is None:
            log_time(f"  Layer {li} weights on meta device, skipping")
            continue
        intermediate_size = W_gate.shape[0]
        
        lr = {"intermediate_size": intermediate_size}
        
        # ---- 键分析: W_gate每行解码 ----
        # W_gate: [intermediate, d_model]
        # 对每个神经元, W_gate[n] · W_U.T → 激活词的logits
        log_time(f"    Decoding keys (W_gate) for {intermediate_size} neurons...")
        
        # 批量计算: W_gate @ W_U.T → [intermediate, vocab]
        # 可能内存太大, 分批处理
        batch_size = 500
        key_decode = {}  # neuron_id -> top-10 tokens
        
        for start in range(0, intermediate_size, batch_size):
            end = min(start + batch_size, intermediate_size)
            if start % 2000 == 0:
                log_time(f"      Key batch {start}-{end}...")
            logits_batch = W_gate[start:end] @ W_U.T  # [batch, vocab]
            for i in range(end - start):
                ni = start + i
                top_ids = np.argsort(logits_batch[i])[-10:][::-1]
                top_toks = [(tokenizer.decode([tid]).strip(), float(logits_batch[i][tid])) 
                           for tid in top_ids]
                key_decode[ni] = top_toks
        
        lr["n_neurons_decoded"] = len(key_decode)
        
        # ---- 值分析: W_down每列解码 ----
        # W_down: [d_model, intermediate]
        # 对每个神经元, W_U @ W_down[:, n] → 促进词的logits
        log_time(f"    Decoding values (W_down) for {intermediate_size} neurons...")
        
        value_decode = {}
        for start in range(0, intermediate_size, batch_size):
            end = min(start + batch_size, intermediate_size)
            if start % 2000 == 0:
                log_time(f"      Value batch {start}-{end}...")
            logits_batch = W_U @ W_down[:, start:end]  # [vocab, batch]
            for i in range(end - start):
                ni = start + i
                top_ids = np.argsort(logits_batch[:, i])[-10:][::-1]
                top_toks = [(tokenizer.decode([tid]).strip(), float(logits_batch[tid, i])) 
                           for tid in top_ids]
                value_decode[ni] = top_toks
        
        # ---- 复用与差异化分析 ----
        log_time(f"    Measuring reuse vs differentiation...")
        
        # 对每个概念词, 找到哪些神经元的键最响应它
        # 神经元n对词w的响应 = W_gate[n] · W_U[w] (键匹配)
        concept_neurons = {}  # (category, word) -> set of top neuron ids
        
        for cat, word_ids in concept_token_ids.items():
            for word, wid in word_ids:
                word_vec = W_U[wid]  # [d_model]
                # 计算所有神经元对这个词的键响应
                responses = W_gate @ word_vec  # [intermediate]
                # ReLU后的响应
                responses_relu = np.maximum(responses, 0)
                # 取top-100神经元
                top_neurons = set(np.argsort(responses_relu)[-100:][::-1].tolist())
                concept_neurons[(cat, word)] = top_neurons
        
        # 类别内复用分析
        reuse_analysis = {}
        for cat in concept_sets.keys():
            cat_words = [(cat, w) for w, _ in concept_token_ids[cat]]
            if len(cat_words) < 2:
                continue
            
            # 两两比较
            pairwise_reuse = []
            all_neuron_sets = [concept_neurons[k] for k in cat_words]
            
            for i in range(len(cat_words)):
                for j in range(i+1, len(cat_words)):
                    set_i = all_neuron_sets[i]
                    set_j = all_neuron_sets[j]
                    intersection = set_i & set_j
                    union = set_i | set_j
                    jaccard = len(intersection) / max(len(union), 1)
                    reuse_rate = len(intersection) / max(len(set_i), 1)
                    
                    pairwise_reuse.append({
                        "word_i": cat_words[i][1],
                        "word_j": cat_words[j][1],
                        "n_shared": len(intersection),
                        "jaccard": float(jaccard),
                        "reuse_rate": float(reuse_rate),
                    })
            
            # 类别平均复用率
            mean_jaccard = float(np.mean([pr["jaccard"] for pr in pairwise_reuse]))
            mean_reuse = float(np.mean([pr["reuse_rate"] for pr in pairwise_reuse]))
            
            # 共享神经元(被所有类别词激活的)
            shared_neurons = set.intersection(*all_neuron_sets) if all_neuron_sets else set()
            
            # 专属神经元(只被一个词激活的)
            exclusive_neurons = {}
            for i, k in enumerate(cat_words):
                others = set()
                for j, k2 in enumerate(cat_words):
                    if i != j:
                        others |= all_neuron_sets[j]
                exclusive = all_neuron_sets[i] - others
                exclusive_neurons[cat_words[i][1]] = len(exclusive)
            
            reuse_analysis[cat] = {
                "mean_jaccard": mean_jaccard,
                "mean_reuse_rate": mean_reuse,
                "n_shared_neurons": len(shared_neurons),
                "exclusive_neurons": exclusive_neurons,
                "pairwise": pairwise_reuse[:10],  # 只保存前10对
            }
            
            log_time(f"    {cat}: mean_jaccard={mean_jaccard:.3f}, "
                     f"shared_neurons={len(shared_neurons)}, "
                     f"reuse_rate={mean_reuse:.3f}")
        
        # 跨类别复用分析
        cross_category_reuse = {}
        cat_names = list(concept_sets.keys())
        for i in range(len(cat_names)):
            for j in range(i+1, len(cat_names)):
                cat_i, cat_j = cat_names[i], cat_names[j]
                # 取两个类别所有词的并集神经元
                neurons_i = set()
                neurons_j = set()
                for k, v in concept_neurons.items():
                    if k[0] == cat_i: neurons_i |= v
                    if k[0] == cat_j: neurons_j |= v
                
                intersection = neurons_i & neurons_j
                union = neurons_i | neurons_j
                jaccard = len(intersection) / max(len(union), 1)
                
                cross_category_reuse[f"{cat_i}_vs_{cat_j}"] = {
                    "jaccard": float(jaccard),
                    "n_shared": len(intersection),
                    "n_cat_i": len(neurons_i),
                    "n_cat_j": len(neurons_j),
                }
        
        # 保存一些典型神经元的键值对
        # 选共享神经元中的5个, 和每个类别专属的5个
        sample_neurons = {}
        if shared_neurons:
            sample_shared = list(shared_neurons)[:5]
            for ni in sample_shared:
                sample_neurons[f"shared_{ni}"] = {
                    "key": key_decode.get(ni, [])[:5],
                    "value": value_decode.get(ni, [])[:5],
                }
        
        for cat in list(concept_sets.keys())[:2]:
            cat_exclusive = exclusive_neurons.get(cat, {})
            top_exclusive = sorted(cat_exclusive.items(), key=lambda x: -x[1])[:3]
            for word, count in top_exclusive:
                # 找这个词专属的神经元
                word_key = (cat, word)
                if word_key in concept_neurons:
                    others = set()
                    for k2, v2 in concept_neurons.items():
                        if k2 != word_key:
                            others |= v2
                    word_excl = concept_neurons[word_key] - others
                    for ni in list(word_excl)[:2]:
                        sample_neurons[f"{cat}_{word}_{ni}"] = {
                            "key": key_decode.get(ni, [])[:5],
                            "value": value_decode.get(ni, [])[:5],
                        }
        
        lr["reuse_analysis"] = reuse_analysis
        lr["cross_category_reuse"] = cross_category_reuse
        lr["sample_neurons"] = sample_neurons
        
        # 释放权重内存
        del W_gate, W_up, W_down, key_decode, value_decode
        gc.collect()
        
        layer_results[str(li)] = lr
        log_time(f"  Layer {li} done")
    
    results["results_by_layer"] = layer_results
    save_result(model_name, 2, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 3: W_U结构分析
# ============================================================

def part3_wu_structure(model_name):
    """
    分析W_U(反嵌入矩阵)的结构
    
    内容:
    1. 有效秩分析 — W_U的有效维度
    2. 语义聚类 — token解码方向是否有聚类结构
    3. 概念词的解码方向相似度矩阵
    4. 与Phase 64质心结果的对比
    """
    import torch
    from scipy.sparse.linalg import svds

    log_time(f"=== Part 3: W_U Structure Analysis for {model_name} ===")
    
    model, tokenizer, device = load_model_safe(model_name)
    from model_utils import get_model_info, get_W_U
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)  # [vocab, d_model]
    
    log_time(f"W_U shape: {W_U.shape}")
    
    results = {"model": model_name, "d_model": info.d_model, 
               "vocab_size": info.vocab_size}
    
    # ---- 1. 有效秩分析 ----
    log_time("Computing effective rank of W_U...")
    # W_U: [vocab, d_model], 对W_U.T做SVD
    k = min(200, min(W_U.shape) - 2)
    U_svd, S_svd, Vt_svd = svds(W_U.astype(np.float32), k=k)
    # 按奇异值降序排列
    sorted_idx = np.argsort(S_svd)[::-1]
    S_sorted = S_svd[sorted_idx]
    
    # 有效秩 = exp(entropy of normalized singular values)
    S_norm = S_sorted / S_sorted.sum()
    S_norm = S_norm[S_norm > 0]
    entropy = -np.sum(S_norm * np.log(S_norm))
    effective_rank = float(np.exp(entropy))
    
    # 90%和99%方差对应的维度数
    cumvar = np.cumsum(S_sorted**2) / np.sum(S_sorted**2)
    k90 = int(np.searchsorted(cumvar, 0.90)) + 1
    k99 = int(np.searchsorted(cumvar, 0.99)) + 1
    
    results["effective_rank"] = effective_rank
    results["k90"] = k90
    results["k99"] = k99
    results["top_singular_values"] = S_sorted[:20].tolist()
    
    log_time(f"  Effective rank: {effective_rank:.1f}, k90: {k90}, k99: {k99}")
    
    # ---- 2. 概念词的解码方向相似度 ----
    log_time("Computing concept token similarity matrix...")
    
    concept_sets = {
        "fruits": ["apple", "banana", "orange", "strawberry", "grape"],
        "animals": ["dog", "cat", "horse", "elephant", "lion"],
        "vehicles": ["car", "bus", "train", "airplane", "bicycle"],
        "tools": ["hammer", "wrench", "screwdriver", "drill", "saw"],
        "colors": ["red", "blue", "green", "yellow", "black"],
        "emotions": ["happy", "sad", "angry", "afraid", "surprised"],
    }
    
    # 获取概念词的W_U行向量
    concept_vectors = {}
    concept_labels = []
    for cat, words in concept_sets.items():
        for w in words:
            toks = tokenizer.encode(w, add_special_tokens=False)
            if toks:
                wid = toks[0]
                concept_vectors[f"{cat}:{w}"] = W_U[wid].copy()
                concept_labels.append((cat, w))
    
    # 归一化
    cv_array = np.array([concept_vectors[l] for l in 
                         [f"{c}:{w}" for c, w in concept_labels]])
    cv_norms = np.linalg.norm(cv_array, axis=1, keepdims=True)
    cv_norms = np.maximum(cv_norms, 1e-10)
    cv_normed = cv_array / cv_norms
    
    # 两两cosine
    cos_matrix = cv_normed @ cv_normed.T
    
    # 类内平均cosine
    within_category = {}
    for cat in concept_sets.keys():
        cat_indices = [i for i, (c, w) in enumerate(concept_labels) if c == cat]
        if len(cat_indices) >= 2:
            cat_cos = []
            for i in cat_indices:
                for j in cat_indices:
                    if i < j:
                        cat_cos.append(float(cos_matrix[i, j]))
            within_category[cat] = float(np.mean(cat_cos))
    
    # 类间平均cosine
    between_category = {}
    cat_names = list(concept_sets.keys())
    for i in range(len(cat_names)):
        for j in range(i+1, len(cat_names)):
            ci = [k for k, (c, w) in enumerate(concept_labels) if c == cat_names[i]]
            cj = [k for k, (c, w) in enumerate(concept_labels) if c == cat_names[j]]
            bc = [float(cos_matrix[a, b]) for a in ci for b in cj]
            between_category[f"{cat_names[i]}_vs_{cat_names[j]}"] = float(np.mean(bc))
    
    results["within_category_cosine"] = within_category
    results["between_category_cosine"] = between_category
    results["concept_labels"] = [f"{c}:{w}" for c, w in concept_labels]
    results["cosine_matrix_sample"] = cos_matrix[:10, :10].tolist()
    
    mean_within = float(np.mean(list(within_category.values())))
    mean_between = float(np.mean(list(between_category.values())))
    log_time(f"  Mean within-category cosine: {mean_within:.4f}")
    log_time(f"  Mean between-category cosine: {mean_between:.4f}")
    log_time(f"  Separation: {mean_within - mean_between:.4f}")
    
    # ---- 3. 反义词/近义词在W_U中的关系 ----
    log_time("Analyzing antonym/synonym relationships...")
    
    relation_pairs = {
        "antonyms": [("hot", "cold"), ("big", "small"), ("fast", "slow"), 
                     ("happy", "sad"), ("light", "dark"), ("old", "young"),
                     ("strong", "weak"), ("rich", "poor")],
        "synonyms": [("big", "large"), ("small", "tiny"), ("fast", "quick"),
                     ("happy", "glad"), ("sad", "unhappy"), ("smart", "clever")],
        "unrelated": [("apple", "hammer"), ("dog", "car"), ("red", "slow"),
                      ("happy", "train"), ("big", "sad")],
    }
    
    relation_cosines = {}
    for rel_type, pairs in relation_pairs.items():
        cos_values = []
        for w1, w2 in pairs:
            t1 = tokenizer.encode(w1, add_special_tokens=False)
            t2 = tokenizer.encode(w2, add_special_tokens=False)
            if t1 and t2:
                v1 = W_U[t1[0]]
                v2 = W_U[t2[0]]
                cos_values.append(cosine_sim(v1, v2))
        relation_cosines[rel_type] = {
            "mean": float(np.mean(cos_values)),
            "values": cos_values,
        }
    
    results["relation_cosines"] = relation_cosines
    for rel_type, data in relation_cosines.items():
        log_time(f"  {rel_type} mean cosine: {data['mean']:.4f}")
    
    save_result(model_name, 3, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 4: Logit Attribution — 精确贡献分解
# ============================================================

def part4_logit_attribution(model_name):
    """
    精确计算每层/每个组件对特定预测的贡献
    
    原理: 残差流 = h_0 + sum(attn_l + mlp_l)
    每个组件对目标token的logit贡献 = W_U[target] · (component_output)
    
    这不是近似, 是精确分解
    """
    import torch
    import torch.nn.functional as F

    log_time(f"=== Part 4: Logit Attribution for {model_name} ===")
    
    model, tokenizer, device = load_model_safe(model_name)
    from model_utils import get_model_info, get_layers, get_W_U
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    W_U = get_W_U(model, model_name)  # [vocab, d_model]
    
    # 任务定义
    tasks = {
        "semantic_fruit": {
            "prompt": "Apple is a kind of",
            "target": "fruit",
            "description": "苹果→水果 (语义上位)",
        },
        "semantic_animal": {
            "prompt": "Dog is a kind of",
            "target": "animal",
            "description": "狗→动物 (语义上位)",
        },
        "antonym_hot": {
            "prompt": "The opposite of hot is",
            "target": "cold",
            "description": "hot的反义词",
        },
        "antonym_big": {
            "prompt": "The opposite of big is",
            "target": "small",
            "description": "big的反义词",
        },
        "translate_en": {
            "prompt": "Translate 'apple' to English:",
            "target": "apple",
            "description": "翻译(英文)",
        },
        "translate_fr": {
            "prompt": "Translate 'apple' to French: pomme. Translate 'banana' to French:",
            "target": "banane",
            "description": "翻译(法文)",
        },
        "logic": {
            "prompt": "If A is bigger than B, and B is bigger than C, then A is",
            "target": "bigger",
            "description": "逻辑推理",
        },
        "grammar": {
            "prompt": "The cats",
            "target": "are",
            "description": "语法(主谓一致)",
        },
    }
    
    results = {"model": model_name, "d_model": info.d_model, 
               "n_layers": info.n_layers}
    task_results = {}
    
    # 获取目标token的方向向量
    W_U_tensor = torch.tensor(W_U, dtype=torch.float32)  # [vocab, d_model]
    
    for task_name, task in tasks.items():
        log_time(f"  Task: {task_name} — {task['description']}")
        
        prompt = task["prompt"]
        target_word = task["target"]
        
        # 编码
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # 确定输入设备
        input_device = next(model.parameters()).device
        input_ids = input_ids.to(input_device)
        attention_mask = attention_mask.to(input_device)
        
        # 目标token
        target_ids = tokenizer.encode(target_word, add_special_tokens=False)
        if not target_ids:
            log_time(f"    Cannot encode target '{target_word}', skipping")
            continue
        target_id = target_ids[0]
        target_direction = W_U[target_id]  # [d_model]
        target_dir_t = torch.tensor(target_direction, dtype=torch.float32)
        
        # 收集每层的attn和mlp输出
        captured_outputs = {}
        
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured_outputs[key] = output[0].detach().float().cpu()
                else:
                    captured_outputs[key] = output.detach().float().cpu()
            return hook
        
        # 注册hooks
        hooks = []
        for li in range(info.n_layers):
            layer = layers[li]
            hooks.append(layer.self_attn.register_forward_hook(make_hook(f"attn_{li}")))
            hooks.append(layer.mlp.register_forward_hook(make_hook(f"mlp_{li}")))
        
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
        
        for h in hooks:
            h.remove()
        
        hidden_states = outputs.hidden_states  # tuple of [1, seq, d_model]
        
        # Logit Attribution
        # 总logit = W_U[target] · h_final[-1]
        h_final = hidden_states[-1][0, -1].float().cpu().numpy()  # [d_model]
        total_logit = float(np.dot(target_direction, h_final))
        
        # 初始embedding的贡献
        h_embed = hidden_states[0][0, -1].float().cpu().numpy()
        embed_logit = float(np.dot(target_direction, h_embed))
        
        # 每层attn和mlp的贡献
        attn_contributions = {}
        mlp_contributions = {}
        
        for li in range(info.n_layers):
            attn_key = f"attn_{li}"
            mlp_key = f"mlp_{li}"
            
            if attn_key in captured_outputs:
                attn_out = captured_outputs[attn_key][0, -1].float().cpu().numpy()
                attn_logit = float(np.dot(target_direction, attn_out))
                attn_contributions[li] = attn_logit
            
            if mlp_key in captured_outputs:
                mlp_out = captured_outputs[mlp_key][0, -1].float().cpu().numpy()
                mlp_logit = float(np.dot(target_direction, mlp_out))
                mlp_contributions[li] = mlp_logit
        
        # 验证: embed + sum(attn + mlp) ≈ total
        residual_sum = embed_logit
        for li in range(info.n_layers):
            residual_sum += attn_contributions.get(li, 0) + mlp_contributions.get(li, 0)
        
        # 找到贡献最大的层
        all_contribs = []
        for li in range(info.n_layers):
            a = attn_contributions.get(li, 0)
            m = mlp_contributions.get(li, 0)
            all_contribs.append((li, a, m, a + m))
        
        all_contribs.sort(key=lambda x: -abs(x[3]))
        top5_layers = all_contribs[:5]
        
        # 按贡献正负分类
        positive_layers = [(li, a, m) for li, a, m, t in all_contribs if t > 0]
        negative_layers = [(li, a, m) for li, a, m, t in all_contribs if t < 0]
        positive_layers.sort(key=lambda x: -x[2])
        negative_layers.sort(key=lambda x: x[2])
        
        task_results[task_name] = {
            "description": task["description"],
            "prompt": prompt,
            "target": target_word,
            "total_logit": total_logit,
            "embed_logit": embed_logit,
            "residual_sum": residual_sum,
            "verification_error": float(abs(total_logit - residual_sum)),
            "top5_contributing_layers": [
                {"layer": li, "attn": float(a), "mlp": float(m), "total": float(t)}
                for li, a, m, t in top5_layers
            ],
            "top5_positive": [
                {"layer": li, "attn": float(a), "mlp": float(m)}
                for li, a, m in positive_layers[:5]
            ],
            "top5_negative": [
                {"layer": li, "attn": float(a), "mlp": float(m)}
                for li, a, m in negative_layers[:5]
            ],
            "attn_contributions": {str(k): float(v) for k, v in attn_contributions.items()},
            "mlp_contributions": {str(k): float(v) for k, v in mlp_contributions.items()},
        }
        
        log_time(f"    Total logit: {total_logit:.3f}, Embed: {embed_logit:.3f}")
        log_time(f"    Top layers: {[(li, f'{t:.3f}') for li, a, m, t in top5_layers[:3]]}")
        log_time(f"    Verification error: {abs(total_logit - residual_sum):.4f}")
        
        del captured_outputs, hidden_states
        gc.collect()
    
    results["task_results"] = task_results
    
    # ---- 层级功能分工汇总 ----
    log_time("Summarizing layer functional specialization...")
    layer_task_matrix = defaultdict(dict)
    for task_name, tr in task_results.items():
        for li_str, contrib in tr["mlp_contributions"].items():
            layer_task_matrix[int(li_str)][task_name] = contrib
    
    results["layer_task_matrix"] = dict(layer_task_matrix)
    
    save_result(model_name, 4, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 5: 97%维度功能探索
# ============================================================

def part5_97percent_exploration(model_name):
    """
    回答最重要的未解问题: 语义轴仅占2-3.5%, 其余97%编码了什么?
    
    三步分析:
    1. 激活值统计分布 — 每个维度的均值/方差/稀疏度
    2. 位置编码维度 — 哪些维度与token位置高度相关
    3. 注意力信息维度 — 哪些维度与attention权重相关
    """
    import torch
    import torch.nn.functional as F

    log_time(f"=== Part 5: 97% Dimension Exploration for {model_name} ===")
    
    model, tokenizer, device = load_model_safe(model_name)
    from model_utils import get_model_info, get_layers, get_W_U
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    W_U = get_W_U(model, model_name)
    
    results = {"model": model_name, "d_model": info.d_model, 
               "n_layers": info.n_layers}
    
    # ---- 1. 激活值统计分布 ----
    log_time("Step 1: Activation statistics distribution...")
    
    # 用100个不同句子收集hidden state
    test_sentences = [
        "The apple is red and sweet.",
        "A large elephant walked slowly across the plain.",
        "She quickly finished her homework before dinner.",
        "The old house stood on top of the hill.",
        "Hot coffee tastes better on cold mornings.",
        "The scientist discovered a new species of butterfly.",
        "He drove his car to the grocery store.",
        "The happy children played in the sunny park.",
        "A tiny bird sang a beautiful melody.",
        "The complex machine required careful maintenance.",
        "The bright stars shone in the dark sky.",
        "She wore a heavy coat in the freezing weather.",
        "The ancient temple was built thousands of years ago.",
        "The distant mountains looked blue in the fading light.",
        "The intricate design puzzled even the expert craftsmen.",
        "It is certain that the sun will rise tomorrow.",
        "The dangerous storm approached the coastal town rapidly.",
        "The beautiful sunset painted the sky with orange and pink.",
        "A newborn baby cried loudly in the hospital.",
        "The moderate temperature made the hike pleasant.",
    ]
    # 扩展到100句: 用不同主语和结构变体
    expanded_sentences = list(test_sentences)
    prefixes = ["Interestingly,", "However,", "Therefore,", "Meanwhile,", "Suddenly,",
                "In fact,", "Moreover,", "Consequently,", "Nevertheless,", "Apparently,"]
    for p in prefixes:
        for s in test_sentences[:8]:
            expanded_sentences.append(f"{p} {s.lower()}")
    test_sentences = expanded_sentences[:100]
    
    log_time(f"  Collected {len(test_sentences)} test sentences")
    
    # 采样层
    sample_layers = sorted(set([0, info.n_layers//4, info.n_layers//2, 
                                3*info.n_layers//4, info.n_layers-1]))
    
    layer_stats = {}
    
    for li in sample_layers:
        log_time(f"  Collecting activations for Layer {li}...")
        
        all_hidden_states = []
        all_positions = []
        all_attn_weights = []
        
        captured = {}
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook
        
        layer = layers[li]
        hook = layer.register_forward_hook(make_hook(f"L{li}"))
        
        # 同时收集attention权重
        attn_hook = layer.self_attn.register_forward_hook(make_hook(f"attn_{li}"))
        
        for sent_idx, sent in enumerate(test_sentences):
            if sent_idx % 20 == 0:
                log_time(f"    Sentence {sent_idx}/{len(test_sentences)}...")
            
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
            input_device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True, output_attentions=True)
            
            # 最后一个token的hidden state
            hs = out.hidden_states[li][0, -1].float().cpu().numpy()  # [d_model]
            all_hidden_states.append(hs)
            all_positions.append(input_ids.shape[1] - 1)  # 位置
            
            # attention权重 (如果有)
            if out.attentions and len(out.attentions) > li:
                # 取最后一层attention, 平均所有头
                attn = out.attentions[li][0].float().cpu().numpy()  # [heads, seq, seq]
                attn_mean = attn.mean(axis=0)  # [seq, seq]
                # 最后一个token对所有其他token的attention
                last_attn = attn_mean[-1, :]  # [seq]
                all_attn_weights.append(last_attn)
            
            del out
            if sent_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        hook.remove()
        attn_hook.remove()
        
        # 统计每个维度的分布
        hs_array = np.array(all_hidden_states)  # [n_sents, d_model]
        
        dim_means = hs_array.mean(axis=0)  # [d_model]
        dim_stds = hs_array.std(axis=0)    # [d_model]
        dim_sparsity = (np.abs(hs_array) < 0.01).mean(axis=0)  # [d_model] 每维接近0的比例
        
        # 按方差分类
        high_var_dims = int((dim_stds > np.percentile(dim_stds, 90)).sum())
        low_var_dims = int((dim_stds < np.percentile(dim_stds, 10)).sum())
        dead_dims = int((dim_stds < 1e-4).sum())
        
        # ---- 2. 位置相关性分析 ----
        log_time(f"  Computing position correlation for Layer {li}...")
        
        positions = np.array(all_positions)  # [n_sents]
        pos_correlations = np.zeros(info.d_model)
        for d in range(info.d_model):
            if dim_stds[d] > 1e-6:
                pos_correlations[d] = abs(float(np.corrcoef(hs_array[:, d], positions)[0, 1]))
            else:
                pos_correlations[d] = 0.0
        
        # 与位置高度相关的维度数
        pos_correlated_dims = int((pos_correlations > 0.3).sum())
        pos_uncorrelated_dims = int((pos_correlations < 0.1).sum())
        
        # ---- 3. 与W_U行空间的对齐度 ----
        log_time(f"  Computing W_U alignment for Layer {li}...")
        
        # 用Phase 64的方法: 计算hidden state在W_U行空间的投影比
        # 但这里计算每个维度的对齐度
        # 更简单的方法: 计算hidden state的方差中有多少可以被W_U的前k个奇异向量解释
        
        from scipy.sparse.linalg import svds
        W_U_T = W_U.T.astype(np.float32)
        k_svd = min(100, min(W_U_T.shape) - 2)
        U_wut, s_wut, _ = svds(W_U_T, k=k_svd)
        
        # 每个hidden state在W_U行空间的投影
        hs_centered = hs_array - hs_array.mean(axis=0)
        proj_coeffs = hs_centered @ U_wut  # [n_sents, k_svd]
        proj_energy = np.sum(proj_coeffs ** 2, axis=1)  # [n_sents]
        total_energy = np.sum(hs_centered ** 2, axis=1)  # [n_sents]
        mean_recoding_ratio = float(np.mean(proj_energy / np.maximum(total_energy, 1e-10)))
        
        # ---- 维度功能分类 ----
        # 高方差 + 高位置相关 → 位置/句法信息
        # 高方差 + 低位相关 + 高W_U对齐 → 语义信息
        # 高方差 + 低位相关 + 低W_U对齐 → 计算中间态
        # 低方差 → 死维度或常量
        
        # W_U对齐度: 每个维度的变化有多少被W_U行空间捕获
        # 用dim_stds作为权重, 计算W_U行空间能解释多少维度变化
        dim_var = dim_stds ** 2
        total_var = dim_var.sum()
        
        # 语义维度: Phase 64发现k90≈53维即可解释12个语义轴
        # 这里用另一种方式: 每个hidden state在W_U空间的投影比
        # 已经算出了mean_recoding_ratio
        
        # 分类维度
        n_semantic_est = int(k90 * (total_var / (W_U.shape[1]))) if 'k90' in dir() else int(info.d_model * 0.035)
        n_pos_correlated = pos_correlated_dims
        n_dead = dead_dims
        n_high_var = high_var_dims
        n_low_var = low_var_dims
        
        # 按方差排序, 最大的维度更可能是语义/功能性的
        top_var_dims = np.argsort(dim_stds)[-n_high_var:]
        bottom_var_dims = np.argsort(dim_stds)[:n_low_var]
        
        # 位置相关维度中的高方差维度
        pos_high_var = int(((pos_correlations > 0.3) & (dim_stds > np.percentile(dim_stds, 50))).sum())
        
        layer_stats[str(li)] = {
            "mean_recoding_ratio": mean_recoding_ratio,
            "dim_mean_stats": {
                "mean_of_means": float(dim_means.mean()),
                "std_of_means": float(dim_means.std()),
                "mean_of_stds": float(dim_stds.mean()),
                "std_of_stds": float(dim_stds.std()),
            },
            "dim_categories": {
                "high_var_dims": high_var_dims,
                "low_var_dims": low_var_dims,
                "dead_dims": dead_dims,
                "pos_correlated_dims": pos_correlated_dims,
                "pos_uncorrelated_dims": pos_uncorrelated_dims,
                "pos_and_high_var": pos_high_var,
            },
            "sparsity": {
                "mean_sparsity": float(dim_sparsity.mean()),
                "high_sparsity_dims_90pct": int((dim_sparsity > 0.9).sum()),
                "medium_sparsity_dims": int(((dim_sparsity > 0.5) & (dim_sparsity <= 0.9)).sum()),
                "low_sparsity_dims": int((dim_sparsity <= 0.5).sum()),
            },
            "position_correlation": {
                "mean_pos_corr": float(pos_correlations.mean()),
                "max_pos_corr": float(pos_correlations.max()),
                "pos_corr_gt_03": pos_correlated_dims,
            },
            # 保存一些分布特征(压缩)
            "dim_std_histogram": np.histogram(dim_stds, bins=20)[0].tolist(),
            "pos_corr_histogram": np.histogram(pos_correlations, bins=20)[0].tolist(),
        }
        
        # 释放内存
        del hs_array, captured
        gc.collect()
        torch.cuda.empty_cache()
        
        log_time(f"  L{li}: recoding_ratio={mean_recoding_ratio:.3f}, "
                 f"dead_dims={dead_dims}, pos_corr={pos_correlated_dims}, "
                 f"mean_sparsity={dim_sparsity.mean():.3f}")
    
    results["layer_stats"] = layer_stats
    
    # ---- 维度功能分类汇总 ----
    log_time("Summarizing dimension functional classification...")
    
    # 跨层平均
    all_recoding_ratios = [ls["mean_recoding_ratio"] for ls in layer_stats.values()]
    all_dead = [ls["dim_categories"]["dead_dims"] for ls in layer_stats.values()]
    all_pos_corr = [ls["dim_categories"]["pos_correlated_dims"] for ls in layer_stats.values()]
    all_high_sparsity = [ls["sparsity"]["high_sparsity_dims_90pct"] for ls in layer_stats.values()]
    
    results["summary"] = {
        "mean_recoding_ratio": float(np.mean(all_recoding_ratios)),
        "mean_dead_dims": float(np.mean(all_dead)),
        "mean_pos_correlated_dims": float(np.mean(all_pos_corr)),
        "mean_high_sparsity_dims": float(np.mean(all_high_sparsity)),
        "d_model": info.d_model,
        "dimension_budget_estimate": {
            "semantic_3_5pct": round(info.d_model * 0.035),
            "pos_correlated_10pct": round(info.d_model * 0.10),
            "high_sparsity_20pct": round(info.d_model * 0.20),
            "remaining_computational": info.d_model - round(info.d_model * 0.035) - round(info.d_model * 0.10) - round(info.d_model * 0.20),
        },
    }
    
    log_time(f"\n*** DIMENSION BUDGET ESTIMATE ***")
    log_time(f"  Semantic (3.5%): ~{results['summary']['dimension_budget_estimate']['semantic_3_5pct']} dims")
    log_time(f"  Position/syntax (10%): ~{results['summary']['dimension_budget_estimate']['pos_correlated_10pct']} dims")
    log_time(f"  Sparse/variable (20%): ~{results['summary']['dimension_budget_estimate']['high_sparsity_20pct']} dims")
    log_time(f"  Computational (~66%): ~{results['summary']['dimension_budget_estimate']['remaining_computational']} dims")
    log_time(f"  Mean recoding_ratio: {np.mean(all_recoding_ratios):.3f}")
    
    save_result(model_name, 5, results)
    release_model_safe(model)
    return results


# ============================================================
# Main
# ============================================================

PART_FUNCTIONS = {
    1: part1_superposition_check,
    2: part2_mlp_key_value,
    3: part3_wu_structure,
    4: part4_logit_attribution,
    5: part5_97percent_exploration,
}

def main():
    parser = argparse.ArgumentParser(description="Phase 255: Direct Mechanism Decoding")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="Model to test")
    parser.add_argument("--part", type=str, required=True,
                       help="Part number (1-5) or 'all'")
    args = parser.parse_args()
    
    model_name = args.model
    
    if args.part == "all":
        parts = [1, 2, 3, 4, 5]
    else:
        parts = [int(args.part)]
    
    log_time(f"Phase 255: Direct Mechanism Decoding")
    log_time(f"Model: {model_name}, Parts: {parts}")
    log_time(f"=" * 60)
    
    for part_num in parts:
        if part_num not in PART_FUNCTIONS:
            log_time(f"Unknown part: {part_num}, skipping")
            continue
        
        log_time(f"\n{'#' * 60}")
        log_time(f"# Starting Part {part_num}")
        log_time(f"{'#' * 60}")
        
        try:
            result = PART_FUNCTIONS[part_num](model_name)
            log_time(f"Part {part_num} completed successfully!")
        except Exception as e:
            log_time(f"Part {part_num} FAILED: {e}")
            import traceback
            traceback.print_exc()
        
        # 强制GC
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        time.sleep(2)
    
    log_time(f"\nPhase 255 completed for {model_name}!")

if __name__ == "__main__":
    main()