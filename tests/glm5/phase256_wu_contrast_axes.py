"""
Phase 256: W_U对比轴解码 — 验证"语言编码=160-200个语义对比轴上的位置编码"
=========================================================================

核心假说: W_U的160-200个奇异向量对应160-200个语义对比轴(反义词在正负两端)

四方案:
  Part 1 (256a): W_U奇异向量解码 — 直接解码每个奇异向量的正负端词
  Part 2 (256b): 修正Superposition检验 — 用真实hidden state替代W_U行
  Part 3 (256c): 反义词cosine的共现频率控制 — 排除替代解释
  Part 4 (256d): 修正Logit Attribution — 用增量(attn_out+mlp_out)替代层总输出

用法:
  python tests/glm5/phase256_wu_contrast_axes.py --model qwen3 --part 1
  python tests/glm5/phase256_wu_contrast_axes.py --model qwen3 --part all
  python tests/glm5/phase256_wu_contrast_axes.py --model glm4 --part 1
  python tests/glm5/phase256_wu_contrast_axes.py --model deepseek7b --part 1
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase256_contrast_axes")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 工具函数
# ============================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.bool_): return bool(obj)
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
    """加载模型, bfloat16 + device_map=auto + flash attention"""
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
            log_time(f"  {attn_impl} failed: {e}, trying next...")
            continue

    model.eval()
    from model_utils import get_model_info
    info = get_model_info(model, model_name)
    log_time(f"{model_name}: class={info.model_class}, layers={info.n_layers}, "
             f"d_model={info.d_model}, vocab={info.vocab_size}")
    return model, tokenizer, info

def get_W_U_safe(model, model_name):
    """获取W_U, 处理meta tensor"""
    from model_utils import get_W_U
    return get_W_U(model, model_name)  # [vocab_size, d_model]

def release_model_safe(model):
    """释放模型"""
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released, GPU cleared")

def save_result(model_name, part, data):
    """保存结果"""
    fname = RESULT_DIR / f"{model_name}_part{part}.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
    log_time(f"Results saved to {fname}")

def safe_decode(tokenizer, token_id):
    """安全解码token"""
    try:
        r = tokenizer.decode([token_id])
        return r.strip() if r else f"<tok_{token_id}>"
    except:
        return f"<tok_{token_id}>"

def get_device_for_tensor(model):
    """获取模型输入设备"""
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Part 1: W_U奇异向量解码
# ============================================================

def part1_wu_svd_decoding(model_name):
    """
    解码W_U的160-200个奇异向量, 验证:
    1. 每个奇异向量的正负端是否对应语义对比轴
    2. 反义词对是否在同一个奇异向量的对立端
    3. 跨模型一致性
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)  # [vocab_size, d_model]
    log_time(f"W_U shape: {W_U.shape}")

    results = {"model": model_name, "d_model": info.d_model, "vocab_size": info.vocab_size}

    # ---- Step 1: W_U SVD ----
    log_time("Computing SVD of W_U...")
    t0 = time.time()
    # W_U shape: [vocab_size, d_model], 做 SVD(W_U) = U S Vt
    # U: [vocab_size, vocab_size], S: [min], Vt: [d_model, d_model] (截断)
    # 我们关心 Vt 的行 (每个是 d_model 维空间中的方向)
    # 以及 U 的列 (每个是 vocab_size 维空间中的方向)

    # 使用截断SVD节省内存
    n_components = min(300, min(W_U.shape) - 1)
    from scipy.sparse.linalg import svds
    U, S, Vt = svds(W_U.astype(np.float32), k=n_components)
    # 按奇异值从大到小排序
    order = np.argsort(S)[::-1]
    U, S, Vt = U[:, order], S[order], Vt[order, :]

    elapsed = time.time() - t0
    log_time(f"SVD done in {elapsed:.1f}s, top-10 singular values: {S[:10].tolist()}")

    # ---- Step 2: 有效秩分析 ----
    total_energy = np.sum(S ** 2)
    cumulative = np.cumsum(S ** 2) / total_energy
    k50 = int(np.searchsorted(cumulative, 0.50)) + 1
    k90 = int(np.searchsorted(cumulative, 0.90)) + 1
    k95 = int(np.searchsorted(cumulative, 0.95)) + 1
    k99 = int(np.searchsorted(cumulative, 0.99)) + 1
    # 有效秩 (entropy-based)
    p = (S ** 2) / total_energy
    effective_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))

    results["svd_analysis"] = {
        "n_components": n_components,
        "effective_rank": round(effective_rank, 1),
        "k50": k50, "k90": k90, "k95": k95, "k99": k99,
        "top10_singular_values": S[:10].tolist(),
        "singular_value_decay": S[:50].tolist(),
        "cumulative_variance_50": cumulative[:50].tolist(),
    }
    log_time(f"Effective rank: {effective_rank:.1f}, k50={k50}, k90={k90}, k95={k95}, k99={k99}")

    # ---- Step 3: 解码每个奇异向量的正负端 ----
    log_time("Decoding singular vectors (positive/negative ends)...")
    n_decode = min(200, n_components)

    # 预计算: 每个token在每个奇异向量上的得分
    # Vt[i, :] 是第i个奇异向量在 d_model 空间中的方向
    # W_U[j, :] 是第j个token在 d_model 空间中的解码方向
    # token j 在轴 i 上的投影 = W_U[j, :] · Vt[i, :] = (W_U @ Vt[i,:])
    # 批量: scores = W_U @ Vt.T  [vocab_size, n_components]

    log_time("Computing token scores on all axes...")
    scores = W_U @ Vt[:n_decode, :].T  # [vocab_size, n_decode]
    log_time(f"Token scores shape: {scores.shape}")

    # 过滤掉特殊token (前几个和最后几个通常是special tokens)
    # 用词频启发式: 只看有实际含义的token
    valid_start = min(100, W_U.shape[0] // 10)  # 跳过前100个(通常是special)
    valid_end = W_U.shape[0]

    singular_vector_decodings = []
    contrast_axis_count = 0  # 语义对比轴计数

    for i in range(n_decode):
        col = scores[valid_start:valid_end, i]
        top_pos_idx = np.argsort(col)[-20:][::-1] + valid_start
        top_neg_idx = np.argsort(col)[:20] + valid_start

        pos_words = [safe_decode(tokenizer, int(idx)) for idx in top_pos_idx]
        neg_words = [safe_decode(tokenizer, int(idx)) for idx in top_neg_idx]
        pos_scores = [float(col[idx - valid_start]) for idx in top_pos_idx]
        neg_scores = [float(col[idx - valid_start]) for idx in top_neg_idx]

        sv_entry = {
            "axis_id": i,
            "singular_value": float(S[i]),
            "variance_explained": float(S[i]**2 / total_energy),
            "positive_end": list(zip(pos_words, [round(s, 3) for s in pos_scores])),
            "negative_end": list(zip(neg_words, [round(s, 3) for s in neg_scores])),
        }
        singular_vector_decodings.append(sv_entry)

        # 判断是否是语义对比轴: 正负端词是否有明显的语义对立
        # 简单启发式: 如果正负端词都属于可识别的语义类别
        # (更严格的判断在Step 4中做)
        if i < 30:  # 只打印前30个
            log_time(f"  Axis {i} (σ={S[i]:.1f}): +{pos_words[:5]} / -{neg_words[:5]}")

    results["singular_vector_decodings"] = singular_vector_decodings

    # ---- Step 4: 反义词对在哪个轴上的检验 ----
    log_time("Testing antonym pairs on singular vectors...")

    # 定义反义词对 (英文+中文混合, 取决于模型词表)
    antonym_pairs = [
        # 基础对比
        ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("light", "dark"),
        ("good", "bad"), ("love", "hate"), ("rich", "poor"), ("strong", "weak"),
        ("happy", "sad"), ("beautiful", "ugly"), ("young", "old"), ("full", "empty"),
        ("loud", "quiet"), ("hard", "soft"), ("sharp", "dull"), ("wet", "dry"),
        ("clean", "dirty"), ("safe", "dangerous"), ("easy", "difficult"),
        ("open", "closed"), ("high", "low"), ("deep", "shallow"), ("wide", "narrow"),
        ("thick", "thin"), ("heavy", "light"), ("bright", "dim"), ("sweet", "bitter"),
        # 中文
        ("热", "冷"), ("大", "小"), ("快", "慢"), ("好", "坏"),
        ("强", "弱"), ("高", "低"), ("多", "少"), ("长", "短"),
        # 抽象对比
        ("create", "destroy"), ("accept", "reject"), ("increase", "decrease"),
        ("include", "exclude"), ("connect", "disconnect"), ("remember", "forget"),
    ]

    # 近义词对 (作为对比)
    synonym_pairs = [
        ("big", "large"), ("small", "tiny"), ("fast", "quick"), ("smart", "clever"),
        ("happy", "joyful"), ("sad", "unhappy"), ("beautiful", "pretty"),
        ("strong", "powerful"), ("begin", "start"), ("end", "finish"),
        ("help", "assist"), ("walk", "stroll"), ("talk", "speak"),
    ]

    # 无关词对 (作为基线)
    unrelated_pairs = [
        ("apple", "car"), ("table", "sky"), ("river", "book"),
        ("dance", "metal"), ("cloud", "shoe"), ("sleep", "hammer"),
    ]

    def find_best_axis(token_A, token_B, scores_matrix, tokenizer):
        """找到区分两个token最好的轴"""
        ids_A = tokenizer.encode(token_A, add_special_tokens=False)
        ids_B = tokenizer.encode(token_B, add_special_tokens=False)
        if not ids_A or not ids_B:
            return None, None, None, None

        id_A, id_B = ids_A[0], ids_B[0]
        if id_A >= scores_matrix.shape[0] or id_B >= scores_matrix.shape[0]:
            return None, None, None, None

        vec_A = scores_matrix[id_A, :]  # [n_decode]
        vec_B = scores_matrix[id_B, :]  # [n_decode]

        # 差异最大的轴
        diff = np.abs(vec_A - vec_B)
        best_axis = int(np.argmax(diff))

        # 也要看: 两者在同一轴上是否符号相反
        score_A = float(vec_A[best_axis])
        score_B = float(vec_B[best_axis])

        # cosine between W_U columns for A and B
        col_A = W_U[id_A]
        col_B = W_U[id_B]
        cos_AB = cosine_sim(col_A, col_B)

        return best_axis, score_A, score_B, cos_AB

    antonym_results = []
    for wA, wB in antonym_pairs:
        result = find_best_axis(wA, wB, scores, tokenizer)
        if result[0] is not None:
            best_axis, sA, sB, cos_AB = result
            # 判断是否符号相反 (对比轴的标志)
            opposite_sign = (sA * sB < 0)
            antonym_results.append({
                "word_A": wA, "word_B": wB,
                "best_axis": best_axis,
                "score_A": round(sA, 3), "score_B": round(sB, 3),
                "opposite_sign": opposite_sign,
                "cosine_in_W_U": round(cos_AB, 4),
            })

    synonym_results = []
    for wA, wB in synonym_pairs:
        result = find_best_axis(wA, wB, scores, tokenizer)
        if result[0] is not None:
            best_axis, sA, sB, cos_AB = result
            opposite_sign = (sA * sB < 0)
            synonym_results.append({
                "word_A": wA, "word_B": wB,
                "best_axis": best_axis,
                "score_A": round(sA, 3), "score_B": round(sB, 3),
                "opposite_sign": opposite_sign,
                "cosine_in_W_U": round(cos_AB, 4),
            })

    unrelated_results = []
    for wA, wB in unrelated_pairs:
        result = find_best_axis(wA, wB, scores, tokenizer)
        if result[0] is not None:
            best_axis, sA, sB, cos_AB = result
            opposite_sign = (sA * sB < 0)
            unrelated_results.append({
                "word_A": wA, "word_B": wB,
                "best_axis": best_axis,
                "score_A": round(sA, 3), "score_B": round(sB, 3),
                "opposite_sign": opposite_sign,
                "cosine_in_W_U": round(cos_AB, 4),
            })

    # 汇总统计
    antonym_opposite_rate = np.mean([r["opposite_sign"] for r in antonym_results]) if antonym_results else 0
    synonym_opposite_rate = np.mean([r["opposite_sign"] for r in synonym_results]) if synonym_results else 0
    unrelated_opposite_rate = np.mean([r["opposite_sign"] for r in unrelated_results]) if unrelated_results else 0

    antonym_mean_cos = np.mean([r["cosine_in_W_U"] for r in antonym_results]) if antonym_results else 0
    synonym_mean_cos = np.mean([r["cosine_in_W_U"] for r in synonym_results]) if synonym_results else 0
    unrelated_mean_cos = np.mean([r["cosine_in_W_U"] for r in unrelated_results]) if unrelated_results else 0

    results["antonym_axis_test"] = {
        "antonym_results": antonym_results,
        "synonym_results": synonym_results,
        "unrelated_results": unrelated_results,
        "summary": {
            "antonym_opposite_sign_rate": round(float(antonym_opposite_rate), 3),
            "synonym_opposite_sign_rate": round(float(synonym_opposite_rate), 3),
            "unrelated_opposite_sign_rate": round(float(unrelated_opposite_rate), 3),
            "antonym_mean_cosine": round(float(antonym_mean_cos), 4),
            "synonym_mean_cosine": round(float(synonym_mean_cos), 4),
            "unrelated_mean_cosine": round(float(unrelated_mean_cos), 4),
        }
    }

    log_time(f"\n*** ANTONYM AXIS TEST ***")
    log_time(f"  Antonym opposite_sign_rate: {antonym_opposite_rate:.3f}")
    log_time(f"  Synonym opposite_sign_rate:  {synonym_opposite_rate:.3f}")
    log_time(f"  Unrelated opposite_sign_rate: {unrelated_opposite_rate:.3f}")
    log_time(f"  Antonym mean cosine: {antonym_mean_cos:.4f}")
    log_time(f"  Synonym mean cosine:  {synonym_mean_cos:.4f}")
    log_time(f"  Unrelated mean cosine: {unrelated_mean_cos:.4f}")

    # ---- Step 5: 反义词对的轴分布 ----
    # 如果反义词确实在对比轴上, 那么它们应该在少数几个轴上聚集
    if antonym_results:
        antonym_axes = [r["best_axis"] for r in antonym_results]
        unique_axes = len(set(antonym_axes))
        # 反义词对是否集中在低维轴上?
        low_dim_count = sum(1 for a in antonym_axes if a < k90)
        results["antonym_axis_test"]["axis_distribution"] = {
            "unique_axes_used": unique_axes,
            "total_antonym_pairs": len(antonym_axes),
            "low_dim_axes_count": low_dim_count,
            "low_dim_ratio": round(float(low_dim_count / len(antonym_axes)), 3) if antonym_axes else 0,
            "axis_histogram": {str(k): int(v) for k, v in sorted(defaultdict(int, 
                {a: antonym_axes.count(a) for a in set(antonym_axes)}).items())},
        }
        log_time(f"  Antonym pairs use {unique_axes} unique axes out of {n_decode}")
        log_time(f"  {low_dim_count}/{len(antonym_axes)} pairs in top-{k90} axes")

    # ---- Step 6: 中文词汇的对比轴 ----
    # 检查中文词是否有独立的对比轴, 还是共享英文的对比轴
    chinese_concepts = [
        ("苹果", "香蕉"), ("猫", "狗"), ("红色", "蓝色"),
        ("快乐", "悲伤"), ("老师", "学生"), ("城市", "乡村"),
        ("春天", "冬天"), ("白天", "夜晚"),
    ]
    chinese_results = []
    for wA, wB in chinese_concepts:
        result = find_best_axis(wA, wB, scores, tokenizer)
        if result[0] is not None:
            best_axis, sA, sB, cos_AB = result
            chinese_results.append({
                "word_A": wA, "word_B": wB,
                "best_axis": best_axis,
                "score_A": round(sA, 3), "score_B": round(sB, 3),
                "cosine_in_W_U": round(cos_AB, 4),
            })

    results["chinese_concept_axes"] = chinese_results
    if chinese_results:
        chinese_axes = [r["best_axis"] for r in chinese_results]
        overlap_with_english = len(set(chinese_axes) & set([r["best_axis"] for r in antonym_results if r.get("best_axis") is not None]))
        log_time(f"  Chinese concept axes: {chinese_axes}")
        log_time(f"  Overlap with English antonym axes: {overlap_with_english}/{len(chinese_axes)}")

    save_result(model_name, 1, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 2: 修正Superposition检验 (用真实hidden state)
# ============================================================

def part2_corrected_superposition(model_name):
    """
    用实际推理中的hidden state重新评估superposition程度
    
    核心修正: Phase 255用W_U行作为h_in的代理, 但实际h_in是LayerNorm后的residual stream
    这里用真实推理来获取MLP激活值, 然后分析每个神经元的语义一致性
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)  # [vocab_size, d_model]
    layers = get_layers(model)

    results = {"model": model_name, "d_model": info.d_model}

    # ---- Step 1: 构建概念词列表 ----
    # 大量概念词, 用于真实推理
    concept_groups = {
        "fruits": ["苹果", "香蕉", "橙子", "草莓", "葡萄", "西瓜", "桃子", "梨", "芒果", "柠檬",
                    "apple", "banana", "orange", "strawberry", "grape", "watermelon", "peach", "mango"],
        "animals": ["猫", "狗", "鸟", "鱼", "马", "牛", "羊", "猪", "老虎", "狮子",
                     "cat", "dog", "bird", "fish", "horse", "cow", "tiger", "lion"],
        "tools": ["锤子", "扳手", "螺丝刀", "锯子", "钳子", "钻头", "斧头", "刀具",
                   "hammer", "wrench", "screwdriver", "saw", "pliers", "drill", "axe"],
        "vehicles": ["汽车", "火车", "飞机", "船", "自行车", "公交车", "卡车", "摩托车",
                      "car", "train", "airplane", "ship", "bicycle", "bus", "truck"],
        "colors": ["红色", "蓝色", "绿色", "黄色", "白色", "黑色", "紫色", "橙色",
                    "red", "blue", "green", "yellow", "white", "black", "purple"],
        "emotions": ["快乐", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶",
                      "happy", "sad", "angry", "fear", "surprised", "disgust"],
    }

    # 展平所有词
    all_words = []
    word_categories = {}
    for cat, words in concept_groups.items():
        for w in words:
            all_words.append(w)
            word_categories[w] = cat

    log_time(f"Total concept words: {len(all_words)} across {len(concept_groups)} categories")

    # ---- Step 2: 对每个词做推理, 捕获MLP激活值 ----
    # 选择中间层
    test_layers = get_sample_layers(info.n_layers, 5)
    log_time(f"Testing layers: {test_layers}")

    # 存储每个词在每个层的MLP post-activation
    # word_activations[word][layer_idx] = activation_vector [intermediate_size]
    word_activations = defaultdict(dict)

    input_device = get_device_for_tensor(model)

    for wi, word in enumerate(all_words):
        prompt = f"这是一个{word}。"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        # 用hook捕获MLP激活
        captured_mlp = {}
        def make_mlp_hook(li):
            def hook(module, input, output):
                # MLP的输出: 取最后一层MLP的post-activation
                # output可能是 (tensor,) 或 tensor
                if isinstance(output, tuple):
                    captured_mlp[li] = output[0][0, -1].detach().float().cpu().numpy()
                else:
                    captured_mlp[li] = output[0, -1].detach().float().cpu().numpy()
            return hook

        # 需要hook MLP的中间激活(SiLU/ReLU之后), 不是MLP的最终输出
        # 对于Qwen2/Qwen3: mlp.act_fn 是激活函数, 但hook位置取决于架构
        # 更可靠: hook down_proj的输入 (即激活后的中间表示)
        hooks = []
        for li in test_layers:
            layer = layers[li]
            mlp = layer.mlp
            # hook down_proj的输入 = SiLU(gate) * up
            # 在PyTorch中, down_proj接收的是act(gate_proj(x)) * up_proj(x)
            # 我们需要这个中间结果
            # 方法: hook mlp的forward, 在中间捕获
            # 但这需要修改forward, 太复杂
            # 替代方案: hook mlp.down_proj的input
            hooks.append(mlp.down_proj.register_forward_hook(make_mlp_hook(li)))

        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids, attention_mask=attn_mask)
            except Exception as e:
                log_time(f"  Forward failed for '{word}': {e}")

        for h in hooks:
            h.remove()

        for li in test_layers:
            if li in captured_mlp:
                word_activations[word][li] = captured_mlp[li]

        if (wi + 1) % 10 == 0:
            log_time(f"  Processed {wi+1}/{len(all_words)} words")

        # 清理
        del captured_mlp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_time(f"Collected activations for {len(word_activations)} words")

    # ---- Step 3: 对每个神经元, 找到激活最高的词, 计算语义一致性 ----
    layer_superposition_results = {}

    for li in test_layers:
        # 收集该层所有词的激活
        layer_acts = {}
        for word in all_words:
            if li in word_activations.get(word, {}):
                layer_acts[word] = word_activations[word][li]

        if not layer_acts:
            log_time(f"  L{li}: No activations collected, skipping")
            continue

        words_list = list(layer_acts.keys())
        act_matrix = np.stack([layer_acts[w] for w in words_list])  # [n_words, intermediate_size]
        n_neurons = act_matrix.shape[1]

        log_time(f"  L{li}: {len(words_list)} words, {n_neurons} neurons")

        # 采样200个神经元
        np.random.seed(42 + li)
        n_sample = min(200, n_neurons)
        sample_neurons = np.random.choice(n_neurons, n_sample, replace=False)

        consistencies = []
        for ni in sample_neurons:
            neuron_acts = act_matrix[:, ni]  # [n_words]

            # 找激活最高的top-30个词
            top_k = min(30, len(words_list))
            top_indices = np.argsort(neuron_acts)[-top_k:]
            top_words = [words_list[i] for i in top_indices]

            # 计算语义一致性: 这些词属于同一类别的比例
            top_categories = [word_categories.get(w, "unknown") for w in top_words]
            # 最常见的类别
            from collections import Counter
            cat_counts = Counter(top_categories)
            most_common_cat, most_common_count = cat_counts.most_common(1)[0]
            category_purity = most_common_count / top_k

            consistencies.append(category_purity)

        mean_consistency = float(np.mean(consistencies))
        high_consistency = sum(1 for c in consistencies if c > 0.7)
        medium_consistency = sum(1 for c in consistencies if 0.3 < c <= 0.7)
        low_consistency = sum(1 for c in consistencies if c <= 0.3)

        layer_superposition_results[str(li)] = {
            "mean_category_purity": round(mean_consistency, 3),
            "high_purity_count": high_consistency,
            "medium_purity_count": medium_consistency,
            "low_purity_count": low_consistency,
            "total_sampled": n_sample,
        }

        log_time(f"  L{li}: mean_purity={mean_consistency:.3f}, "
                 f"high={high_consistency}, med={medium_consistency}, low={low_consistency}")

    # ---- Step 4: 与Phase 255的W_U行方法对比 ----
    # Phase 255结果: Qwen3=0.073, GLM4=0.124, DS7B=0.199
    phase255_baseline = {"qwen3": 0.073, "glm4": 0.124, "deepseek7b": 0.199}
    baseline = phase255_baseline.get(model_name, 0.1)

    # 计算新方法的平均purity
    all_purities = [ls["mean_category_purity"] for ls in layer_superposition_results.values()]
    mean_new = float(np.mean(all_purities)) if all_purities else 0

    results["layer_superposition"] = layer_superposition_results
    results["comparison_with_phase255"] = {
        "phase255_W_U_method_consistency": baseline,
        "new_real_hidden_state_purity": round(mean_new, 3),
        "ratio_new_to_old": round(mean_new / baseline, 2) if baseline > 0 else 0,
        "method": "category_purity (top-30 activating words' category concentration)",
    }

    log_time(f"\n*** SUPERPOSITION COMPARISON ***")
    log_time(f"  Phase 255 (W_U proxy): consistency={baseline}")
    log_time(f"  New (real hidden state): purity={mean_new:.3f}")
    log_time(f"  Ratio: {mean_new/baseline:.2f}x" if baseline > 0 else "  Ratio: N/A")

    if mean_new > 0.3:
        log_time(f"  → Superposition is MODERATE, MLP key-value analysis may work with care")
    elif mean_new > 0.15:
        log_time(f"  → Superposition is SIGNIFICANT, SAE recommended")
    else:
        log_time(f"  → Superposition is SEVERE, SAE is necessary")

    save_result(model_name, 2, results)
    release_model_safe(model)
    return results


def get_sample_layers(n_layers, n_samples=5):
    """均匀采样层"""
    if n_layers <= n_samples:
        return list(range(n_layers))
    step = n_layers // n_samples
    layers = list(range(0, n_layers, step)) + [n_layers - 1]
    return sorted(set(layers))


# ============================================================
# Part 3: 反义词cosine的共现频率控制
# ============================================================

def part3_cooccurrence_control(model_name):
    """
    控制共现频率, 验证"反义词在W_U中最相似"不是共现频率的伪影
    
    方法:
    1. 用模型自身的attention作为共现的代理 (没有外部语料)
    2. 构造高共现vs低共现的反义词对
    3. 构造高共现vs低共现的近义词对
    4. 比较各组在W_U中的cosine
    """
    import torch

    model, tokenizer, info = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)  # [vocab_size, d_model]

    results = {"model": model_name}

    # ---- Step 1: 定义词对组 ----
    # 高共现反义词 (经常在同一句话中出现)
    high_cooc_antonyms = [
        ("hot", "cold"), ("black", "white"), ("day", "night"),
        ("up", "down"), ("left", "right"), ("yes", "no"),
        ("win", "lose"), ("open", "close"), ("start", "stop"),
        ("buy", "sell"), ("rise", "fall"), ("enter", "exit"),
    ]

    # 低共现反义词 (语义对立但很少在同一句话中出现)
    low_cooc_antonyms = [
        ("frugal", "extravagant"), ("taciturn", "loquacious"),
        ("meticulous", "careless"), ("pristine", "contaminated"),
        ("benevolent", "malicious"), ("ephemeral", "permanent"),
        ("obscure", "renowned"), ("barren", "fertile"),
        ("fragile", "robust"), ("mundane", "extraordinary"),
    ]

    # 高共现近义词 (经常在同一上下文中出现)
    high_cooc_synonyms = [
        ("big", "large"), ("small", "tiny"), ("fast", "quick"),
        ("smart", "clever"), ("happy", "glad"), ("sad", "unhappy"),
        ("begin", "start"), ("end", "finish"), ("help", "assist"),
    ]

    # 低共现近义词 (语义相近但很少在同一句话中出现)
    low_cooc_synonyms = [
        ("enormous", "gigantic"), ("minute", "minuscule"),
        ("intelligent", "brilliant"), ("melancholy", "forlorn"),
        ("sturdy", "resilient"), ("lucid", "transparent"),
    ]

    # 高共现无关词 (经常在同一上下文中, 但语义无关)
    high_cooc_unrelated = [
        ("salt", "pepper"), ("knife", "fork"), ("pen", "paper"),
        ("shoes", "socks"), ("bread", "butter"), ("car", "road"),
        ("sun", "moon"), ("rain", "wind"), ("table", "chair"),
    ]

    def compute_pair_cosines(pairs, pair_type):
        """计算一组词对在W_U中的cosine"""
        pair_results = []
        for wA, wB in pairs:
            ids_A = tokenizer.encode(wA, add_special_tokens=False)
            ids_B = tokenizer.encode(wB, add_special_tokens=False)
            if not ids_A or not ids_B:
                continue

            # 用第一个subtoken
            id_A, id_B = ids_A[0], ids_B[0]
            if id_A >= W_U.shape[0] or id_B >= W_U.shape[0]:
                continue

            vec_A = W_U[id_A]
            vec_B = W_U[id_B]
            cos = cosine_sim(vec_A, vec_B)

            pair_results.append({
                "word_A": wA, "word_B": wB,
                "cosine": round(cos, 4),
            })
        return pair_results

    # ---- Step 2: 计算所有组的cosine ----
    log_time("Computing cosines for all pair groups...")

    group_results = {}
    group_results["high_cooc_antonyms"] = compute_pair_cosines(high_cooc_antonyms, "antonym")
    group_results["low_cooc_antonyms"] = compute_pair_cosines(low_cooc_antonyms, "antonym")
    group_results["high_cooc_synonyms"] = compute_pair_cosines(high_cooc_synonyms, "synonym")
    group_results["low_cooc_synonyms"] = compute_pair_cosines(low_cooc_synonyms, "synonym")
    group_results["high_cooc_unrelated"] = compute_pair_cosines(high_cooc_unrelated, "unrelated")

    # ---- Step 3: 汇总分析 ----
    def mean_cosine(group):
        if not group:
            return 0
        return float(np.mean([p["cosine"] for p in group]))

    summary = {
        "high_cooc_antonyms_mean_cos": round(mean_cosine(group_results["high_cooc_antonyms"]), 4),
        "low_cooc_antonyms_mean_cos": round(mean_cosine(group_results["low_cooc_antonyms"]), 4),
        "high_cooc_synonyms_mean_cos": round(mean_cosine(group_results["high_cooc_synonyms"]), 4),
        "low_cooc_synonyms_mean_cos": round(mean_cosine(group_results["low_cooc_synonyms"]), 4),
        "high_cooc_unrelated_mean_cos": round(mean_cosine(group_results["high_cooc_unrelated"]), 4),
    }

    # 关键判断
    antonym_cooc_effect = summary["high_cooc_antonyms_mean_cos"] - summary["low_cooc_antonyms_mean_cos"]
    synonym_cooc_effect = summary["high_cooc_synonyms_mean_cos"] - summary["low_cooc_synonyms_mean_cos"]

    # 如果反义词的共现效应小 (高共现和低共现cosine相近), 说明共现不是主要解释
    # 如果反义词的共现效应大, 说明共现频率影响显著
    summary["antonym_cooc_effect"] = round(antonym_cooc_effect, 4)
    summary["synonym_cooc_effect"] = round(synonym_cooc_effect, 4)

    # 对比效应: 反义词vs近义词 在控制共现后的差异
    # 纯语义对比效应 (控制共现后)
    antonym_advantage_high_cooc = summary["high_cooc_antonyms_mean_cos"] - summary["high_cooc_synonyms_mean_cos"]
    antonym_advantage_low_cooc = summary["low_cooc_antonyms_mean_cos"] - summary["low_cooc_synonyms_mean_cos"]
    summary["antonym_advantage_high_cooc"] = round(antonym_advantage_high_cooc, 4)
    summary["antonym_advantage_low_cooc"] = round(antonym_advantage_low_cooc, 4)

    results["group_results"] = group_results
    results["summary"] = summary

    log_time(f"\n*** CO-OCCURRENCE CONTROL ***")
    log_time(f"  High co-oc antonyms: {summary['high_cooc_antonyms_mean_cos']:.4f}")
    log_time(f"  Low co-oc antonyms:  {summary['low_cooc_antonyms_mean_cos']:.4f}")
    log_time(f"  High co-oc synonyms: {summary['high_cooc_synonyms_mean_cos']:.4f}")
    log_time(f"  Low co-oc synonyms:  {summary['low_cooc_synonyms_mean_cos']:.4f}")
    log_time(f"  High co-oc unrelated: {summary['high_cooc_unrelated_mean_cos']:.4f}")
    log_time(f"  Antonym co-oc effect: {antonym_cooc_effect:.4f}")
    log_time(f"  Synonym co-oc effect: {synonym_cooc_effect:.4f}")
    log_time(f"  Antonym advantage (high co-oc): {antonym_advantage_high_cooc:.4f}")
    log_time(f"  Antonym advantage (low co-oc):  {antonym_advantage_low_cooc:.4f}")

    if abs(antonym_cooc_effect) < 0.05:
        log_time(f"  → Co-occurrence has MINIMAL effect on antonym cosine")
        log_time(f"  → 'Semantic contrast' hypothesis SUPPORTED")
    elif antonym_cooc_effect > 0.05:
        log_time(f"  → Co-occurrence has SIGNIFICANT effect on antonym cosine")
        log_time(f"  → Need to control for co-occurrence before concluding")
    else:
        log_time(f"  → Unexpected: low co-oc antonyms have HIGHER cosine")

    # ---- Step 4: 用模型自身attention作为共现代理 ----
    log_time("Using model attention as co-occurrence proxy...")

    # 选几个高共现和低共现的对, 用模型attention验证
    test_pairs_high = high_cooc_antonyms[:5]
    test_pairs_low = low_cooc_antonyms[:5]

    input_device = get_device_for_tensor(model)

    def measure_attention_cooccurrence(wA, wB):
        """测量两个词在同一上下文中模型attention的互相关"""
        prompt = f"The words {wA} and {wB} are"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        with torch.no_grad():
            try:
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_attentions=True)
                # 取中间层的attention, 平均所有头
                mid_layer = len(out.attentions) // 2
                attn = out.attentions[mid_layer][0].float().cpu().numpy()  # [heads, seq, seq]
                mean_attn = attn.mean(axis=0)  # [seq, seq]
                # 对角线外的平均attention (词间互相关)
                off_diag = mean_attn[~np.eye(mean_attn.shape[0], dtype=bool)]
                return float(np.mean(off_diag))
            except:
                return 0.0

    attn_cooc_results = {"high_cooc": [], "low_cooc": []}
    for wA, wB in test_pairs_high:
        attn_score = measure_attention_cooccurrence(wA, wB)
        attn_cooc_results["high_cooc"].append({"pair": f"{wA}-{wB}", "attn_score": round(attn_score, 4)})

    for wA, wB in test_pairs_low:
        attn_score = measure_attention_cooccurrence(wA, wB)
        attn_cooc_results["low_cooc"].append({"pair": f"{wA}-{wB}", "attn_score": round(attn_score, 4)})

    results["attention_cooccurrence_proxy"] = attn_cooc_results

    save_result(model_name, 3, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 4: 修正Logit Attribution (用增量)
# ============================================================

def part4_corrected_logit_attribution(model_name):
    """
    修正Logit Attribution: 用每层的attn_out + mlp_out增量, 而非层总输出
    
    数学基础:
      residual[l] = residual[0] + Σ(attn_out[i] + mlp_out[i])  for i=0..l
      logit(target) = W_U @ residual[-1] @ e_target
                    = W_U @ residual[0] @ e_target 
                      + Σ(W_U @ attn_out[i] @ e_target)
                      + Σ(W_U @ mlp_out[i] @ e_target)
    
    每层的attn_out和mlp_out是纯增量, 可以精确归因
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)  # [vocab_size, d_model]
    layers = get_layers(model)
    input_device = get_device_for_tensor(model)

    results = {"model": model_name, "n_layers": info.n_layers}

    # ---- 定义测试任务 ----
    tasks = [
        {
            "name": "semantic_superordinate_fruit",
            "prompt": "苹果是一种",
            "target": "水果",
            "description": "苹果→水果 (语义上位)",
        },
        {
            "name": "semantic_superordinate_animal",
            "prompt": "老虎是一种",
            "target": "动物",
            "description": "老虎→动物 (语义上位)",
        },
        {
            "name": "antonym_hot_cold",
            "prompt": "热的反义词是",
            "target": "冷",
            "description": "热→冷 (反义词)",
        },
        {
            "name": "translation_en",
            "prompt": "将'苹果'翻译成英文:",
            "target": "apple",
            "description": "中→英翻译",
        },
        {
            "name": "logical_reasoning",
            "prompt": "如果A大于B,B大于C,那么A",
            "target": "大于",
            "description": "逻辑推理",
        },
        {
            "name": "grammar_subject_verb",
            "prompt": "他们每天都",
            "target": "去",
            "description": "语法:主谓",
        },
    ]

    # ---- 对每个任务做修正的Logit Attribution ----
    task_results = {}

    for task in tasks:
        log_time(f"Task: {task['name']} — {task['description']}")

        prompt = task["prompt"]
        target_word = task["target"]

        # 编码
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        # 目标token
        target_ids = tokenizer.encode(target_word, add_special_tokens=False)
        if not target_ids:
            log_time(f"  Cannot encode target '{target_word}', skipping")
            continue
        target_id = target_ids[0]

        # 目标方向: W_U[target_id, :] (d_model维向量)
        target_direction = W_U[target_id]  # [d_model]
        target_norm = np.linalg.norm(target_direction)
        if target_norm > 0:
            target_direction = target_direction / target_norm

        # ---- Hook: 捕获attn_out和mlp_out ----
        # 注意: 需要hook的是self_attn和mlp的输出, 不是整个layer的输出
        captured_increments = {}

        def make_attn_hook(li):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured_increments[f"L{li}_attn"] = output[0][0, -1].detach().float().cpu().numpy()
                else:
                    captured_increments[f"L{li}_attn"] = output[0, -1].detach().float().cpu().numpy()
            return hook

        def make_mlp_hook(li):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured_increments[f"L{li}_mlp"] = output[0][0, -1].detach().float().cpu().numpy()
                else:
                    captured_increments[f"L{li}_mlp"] = output[0, -1].detach().float().cpu().numpy()
            return hook

        hooks = []
        for li in range(info.n_layers):
            layer = layers[li]
            # Hook attention output
            if hasattr(layer, 'self_attn'):
                hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(li)))
            # Hook MLP output
            if hasattr(layer, 'mlp'):
                hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(li)))

        # 前向传播
        with torch.no_grad():
            try:
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)
            except Exception as e:
                log_time(f"  Forward failed: {e}")
                for h in hooks:
                    h.remove()
                continue

        for h in hooks:
            h.remove()

        # ---- 计算Logit Attribution ----
        # 初始embedding的贡献
        h_embed = out.hidden_states[0][0, -1].float().cpu().numpy()  # [d_model]
        embed_logit = float(np.dot(target_direction, h_embed))

        # 每层attn和mlp的贡献
        attn_contributions = {}
        mlp_contributions = {}

        for li in range(info.n_layers):
            attn_key = f"L{li}_attn"
            mlp_key = f"L{li}_mlp"

            if attn_key in captured_increments:
                attn_out = captured_increments[attn_key]
                attn_contributions[li] = float(np.dot(target_direction, attn_out))

            if mlp_key in captured_increments:
                mlp_out = captured_increments[mlp_key]
                mlp_contributions[li] = float(np.dot(target_direction, mlp_out))

        # 最终logit (从输出验证)
        final_logits = out.logits[0, -1].float().cpu().numpy()
        actual_target_logit = float(final_logits[target_id])

        # 归因总和
        total_attribution = embed_logit + sum(attn_contributions.values()) + sum(mlp_contributions.values())

        # 验证误差
        verification_error = abs(total_attribution - actual_target_logit)
        verification_ratio = verification_error / max(abs(actual_target_logit), 1e-6)

        # ---- 找到关键层 ----
        # 合并attn和mlp贡献
        layer_total = {}
        for li in range(info.n_layers):
            layer_total[li] = attn_contributions.get(li, 0) + mlp_contributions.get(li, 0)

        # 按绝对贡献排序
        sorted_layers = sorted(layer_total.items(), key=lambda x: abs(x[1]), reverse=True)
        top5_layers = sorted_layers[:5]

        task_result = {
            "prompt": prompt,
            "target": target_word,
            "target_id": target_id,
            "actual_logit": round(actual_target_logit, 2),
            "embed_logit": round(embed_logit, 2),
            "total_attribution": round(total_attribution, 2),
            "verification_error": round(verification_error, 2),
            "verification_ratio": round(verification_ratio, 3),
            "attn_contributions": {str(k): round(v, 2) for k, v in sorted(attn_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]},
            "mlp_contributions": {str(k): round(v, 2) for k, v in sorted(mlp_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]},
            "top5_layers": [(str(l), round(c, 2)) for l, c in top5_layers],
            "positive_layers": sorted([(str(l), round(c, 2)) for l, c in layer_total.items() if c > 0.5], key=lambda x: -x[1])[:5],
            "negative_layers": sorted([(str(l), round(c, 2)) for l, c in layer_total.items() if c < -0.5], key=lambda x: x[1])[:5],
        }

        task_results[task["name"]] = task_result

        log_time(f"  Actual logit: {actual_target_logit:.2f}")
        log_time(f"  Total attribution: {total_attribution:.2f}")
        log_time(f"  Verification error: {verification_error:.2f} ({verification_ratio:.1%})")
        log_time(f"  Top-5 layers: {[(l, c) for l, c in top5_layers]}")
        log_time(f"  Positive: {task_result['positive_layers']}")
        log_time(f"  Negative: {task_result['negative_layers']}")

        # 清理
        del captured_increments, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["task_results"] = task_results

    # ---- 跨任务汇总 ----
    log_time("\n*** CROSS-TASK SUMMARY ***")
    for tname, tr in task_results.items():
        log_time(f"  {tname}: verification_error={tr['verification_ratio']:.1%}, "
                 f"top_layers={tr['top5_layers'][:3]}")

    # 判断修正是否有效
    mean_verif_ratio = np.mean([tr["verification_ratio"] for tr in task_results.values()])
    results["correction_assessment"] = {
        "mean_verification_ratio": round(float(mean_verif_ratio), 3),
        "correction_effective": mean_verif_ratio < 0.2,
        "note": "If verification_ratio < 20%, corrected attribution is reliable",
    }

    if mean_verif_ratio < 0.2:
        log_time(f"  ✓ Correction EFFECTIVE (mean error {mean_verif_ratio:.1%} < 20%)")
    else:
        log_time(f"  ✗ Correction INSUFFICIENT (mean error {mean_verif_ratio:.1%} >= 20%)")

    save_result(model_name, 4, results)
    release_model_safe(model)
    return results


# ============================================================
# Main
# ============================================================

PART_FUNCTIONS = {
    1: part1_wu_svd_decoding,
    2: part2_corrected_superposition,
    3: part3_cooccurrence_control,
    4: part4_corrected_logit_attribution,
}

def main():
    parser = argparse.ArgumentParser(description="Phase 256: W_U Contrast Axes Decoding")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="Model to test")
    parser.add_argument("--part", type=str, required=True,
                       help="Part number (1-4) or 'all'")
    args = parser.parse_args()

    model_name = args.model

    if args.part == "all":
        parts = [1, 2, 3, 4]
    else:
        parts = [int(args.part)]

    log_time(f"Phase 256: W_U Contrast Axes Decoding")
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

        gc.collect()
        import torch
        torch.cuda.empty_cache()
        time.sleep(2)

    log_time(f"\nPhase 256 completed for {model_name}!")

if __name__ == "__main__":
    main()
