"""
Phase 140: 语言算子力学 (Language Operator Mechanics)
====================================================
核心目标：从"统计现象学"进入"算子代数"

三大实验：
  Exp A: 微扰强度扫描 — 0.01%~5% 扰动下语义vs随机的差异
  Exp B: Jacobian SVD直接计算 — 真正的奇异值谱
  Exp C: 最小语言算子库 — 单token算子的交换性和干涉项

关键理论：
  - 语言不是"向量编码"，而是"算子在流形上的条件变形"
  - 最小算子: NOT, PAST, PLURAL, FUTURE, MODAL
  - 核心量: 干涉项 I_ij = Δh(O_i O_j) - Δh(O_i) - Δh(O_j)

时间：2026-05-12 15:00
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import json
import time
import gc
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS, get_W_U
)

# ============================================================
# 最小语言算子库 — 单token变化的语义算子
# ============================================================
# 关键改进：不再用 "always→never" 这种多token替换
# 而是用真正的"最小操作"：单token插入/替换

# 算子定义：每个算子是 (base_template, operator_token, position)
# position = "before_verb" / "after_subject" / "verb_suffix"

MINIMAL_OPERATORS = {
    "NOT": {
        "description": "否定算子 - 在动词前插入'not'",
        "pairs": [
            ("The cat can swim", "The cat can not swim"),
            ("The dog will run", "The dog will not run"),
            ("She has finished", "She has not finished"),
            ("He could answer", "He could not answer"),
            ("They should leave", "They should not leave"),
            ("The bird can fly", "The bird can not fly"),
            ("The child will sleep", "The child will not sleep"),
            ("She must study", "She must not study"),
            ("He would agree", "He would not agree"),
            ("They may enter", "They may not enter"),
            ("The fish can breathe", "The fish can not breathe"),
            ("The plant will grow", "The plant will not grow"),
            ("She can speak", "She can not speak"),
            ("He should rest", "He should not rest"),
            ("They might come", "They might not come"),
            ("The dog can bark", "The dog can not bark"),
            ("She will cook", "She will not cook"),
            ("He has eaten", "He has not eaten"),
            ("They must decide", "They must not decide"),
            ("The cat could jump", "The cat could not jump"),
        ],
    },
    "PAST": {
        "description": "时态算子 - 现在时→过去时(规则变化)",
        "pairs": [
            ("The cat walks home", "The cat walked home"),
            ("She plays the piano", "She played the piano"),
            ("He works hard", "He worked hard"),
            ("They talk loudly", "They talked loudly"),
            ("The dog jumps high", "The dog jumped high"),
            ("She cooks dinner", "She cooked dinner"),
            ("He reads books", "He read books"),
            ("They walk slowly", "They walked slowly"),
            ("The bird sings", "The bird sang"),
            ("She paints well", "She painted well"),
            ("He drives fast", "He drove fast"),
            ("They build houses", "They built houses"),
            ("The cat chases mice", "The cat chased mice"),
            ("She writes letters", "She wrote letters"),
            ("He runs fast", "He ran fast"),
            ("They teach children", "They taught children"),
            ("The dog fetches balls", "The dog fetched balls"),
            ("She dances well", "She danced well"),
            ("He fixes cars", "He fixed cars"),
            ("They watch TV", "They watched TV"),
        ],
    },
    "PLURAL": {
        "description": "复数算子 - 单数→复数(主语变化)",
        "pairs": [
            ("The cat sits on the mat", "The cats sit on the mat"),
            ("A dog runs in the park", "Dogs run in the park"),
            ("The bird flies away", "The birds fly away"),
            ("A child plays here", "Children play here"),
            ("The tree grows tall", "The trees grow tall"),
            ("A flower blooms", "Flowers bloom"),
            ("The star shines bright", "The stars shine bright"),
            ("A book lies open", "Books lie open"),
            ("The river flows south", "Rivers flow south"),
            ("A cloud drifts by", "Clouds drift by"),
            ("The car moves fast", "The cars move fast"),
            ("A bird builds a nest", "Birds build nests"),
            ("The door opens wide", "The doors open wide"),
            ("A student reads a lot", "Students read a lot"),
            ("The cat sleeps all day", "The cats sleep all day"),
            ("A fish swims deep", "Fish swim deep"),
            ("The leaf falls down", "The leaves fall down"),
            ("A horse runs fast", "Horses run fast"),
            ("The child learns math", "The children learn math"),
            ("A ship sails far", "Ships sail far"),
        ],
    },
    "FUTURE": {
        "description": "将来时算子 - 现在时→将来时",
        "pairs": [
            ("The cat sits here", "The cat will sit here"),
            ("She eats lunch", "She will eat lunch"),
            ("He goes home", "He will go home"),
            ("They play games", "They will play games"),
            ("The dog barks loud", "The dog will bark loud"),
            ("She reads a book", "She will read a book"),
            ("He drives to work", "He will drive to work"),
            ("They sing songs", "They will sing songs"),
            ("The bird flies high", "The bird will fly high"),
            ("She writes a letter", "She will write a letter"),
            ("He fixes the car", "He will fix the car"),
            ("They build a house", "They will build a house"),
            ("The cat sleeps now", "The cat will sleep now"),
            ("She cooks the meal", "She will cook the meal"),
            ("He runs the race", "He will run the race"),
            ("They watch the show", "They will watch the show"),
            ("The dog chases the cat", "The dog will chase the cat"),
            ("She paints the wall", "She will paint the wall"),
            ("He teaches the class", "He will teach the class"),
            ("They clean the room", "They will clean the room"),
        ],
    },
    "MODAL": {
        "description": "情态算子 - 插入情态动词",
        "pairs": [
            ("The cat catches mice", "The cat can catch mice"),
            ("She solves problems", "She can solve problems"),
            ("He lifts weights", "He can lift weights"),
            ("They finish work", "They can finish work"),
            ("The dog swims well", "The dog can swim well"),
            ("She speaks French", "She can speak French"),
            ("He drives safely", "He can drive safely"),
            ("They learn fast", "They can learn fast"),
            ("The bird builds nests", "The bird can build nests"),
            ("She writes poems", "She can write poems"),
            ("He fixes machines", "He can fix machines"),
            ("They grow crops", "They can grow crops"),
            ("The cat climbs trees", "The cat can climb trees"),
            ("She draws pictures", "She can draw pictures"),
            ("He runs marathons", "He can run marathons"),
            ("They read maps", "They can read maps"),
            ("The dog finds bones", "The dog can find bones"),
            ("She plays guitar", "She can play guitar"),
            ("He cooks pasta", "He can cook pasta"),
            ("They build bridges", "They can build bridges"),
        ],
    },
}

# 算子组合对 — 用于测试交换性和干涉项
COMPOSITION_PAIRS = [
    # NOT + PAST: "can not walk" vs "walked and not"  (否定+时态)
    ("NOT+PAST", [
        ("The cat can walk", "The cat can not walk", "The cat walked", "The cat can not have walked"),
        ("The dog can run", "The dog can not run", "The dog ran", "The dog can not have run"),
        ("She can sing", "She can not sing", "She sang", "She can not have sung"),
        ("He can swim", "He can not swim", "He swam", "He can not have swum"),
        ("They can fly", "They can not fly", "They flew", "They can not have flown"),
    ]),
    # NOT + FUTURE: 否定+将来时
    ("NOT+FUTURE", [
        ("The cat sits", "The cat will not sit", "The cat will sit", "The cat will not sit"),
        ("She eats", "She will not eat", "She will eat", "She will not eat"),
        ("He runs", "He will not run", "He will run", "He will not run"),
        ("They play", "They will not play", "They will play", "They will not play"),
        ("The dog barks", "The dog will not bark", "The dog will bark", "The dog will not bark"),
    ]),
    # PLURAL + PAST: 复数+时态
    ("PLURAL+PAST", [
        ("The cat walks", "The cats walk", "The cat walked", "The cats walked"),
        ("A dog runs", "Dogs run", "A dog ran", "Dogs ran"),
        ("The bird flies", "The birds fly", "The bird flew", "The birds flew"),
        ("A child plays", "Children play", "A child played", "Children played"),
        ("The tree grows", "The trees grow", "The tree grew", "The trees grew"),
    ]),
]


# ============================================================
# 工具函数
# ============================================================

def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_forward(model, input_ids, attention_mask, output_hidden_states=True):
    """安全前向传播，捕获异常"""
    try:
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=output_hidden_states)
    except Exception as e:
        print(f"    [WARN] Forward failed: {e}")
        return None


def get_hidden_states(model, tokenizer, device, sentence, layers_sample=None):
    """获取句子在各层的hidden states"""
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device)
    attn_mask = torch.ones(1, len(ids), device=device, dtype=torch.long)
    out = safe_forward(model, input_ids, attn_mask)
    if out is None:
        return None
    return out.hidden_states


# ============================================================
# Exp A: 微扰强度扫描
# ============================================================
# 关键假设：5%扰动可能已经把状态推出了流形
# 需要在0.01%~5%范围内精细扫描，看语义vs随机差异是否在低扰动下出现

def expA_perturbation_intensity_scan(model, tokenizer, device, model_info, model_name: str):
    """
    Exp A: 微扰强度扫描
    
    简化版：不用hook注入，而是比较不同强度下语义方向和随机方向的传播效率
    
    方法：
    1. 对base和operator句子做forward，得到各层hidden states
    2. 计算"语义delta" = h(operator) - h(base) 在每层的大小
    3. 注入不同强度的随机扰动到embedding层，测量传播
    4. 比较语义方向 vs 随机方向的传播效率随扰动强度的变化
    
    关键：不再用hook注入中间层扰动（太慢且不稳定），
    而是比较"自然语义差"和"人工随机扰动"在各层的传播效率
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_input_device(model)
    
    # 扰动强度列表 — 在embedding层注入
    intensities = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]
    intensity_labels = ["0.01%", "0.05%", "0.1%", "0.5%", "1%", "2%", "5%"]
    
    # 使用否定算子和时态算子
    operator_pairs_neg = MINIMAL_OPERATORS["NOT"]["pairs"][:10]
    operator_pairs_tense = MINIMAL_OPERATORS["PAST"]["pairs"][:10]
    
    results = {
        "model": model_name,
        "intensities": intensity_labels,
        "semantic_vs_random": {},  # {intensity: {layer: {semantic/random delta}}}
        "operator_sensitivity": {},  # {operator: {layer: mean_rel_change}}
    }
    
    # === Part 1: 算子敏感性分析 ===
    # 不注入扰动，直接比较base和operator的hidden state差
    print("\n  Part 1: 算子敏感性分析 (无扰动)...")
    
    for op_name, op_data in [("NOT", MINIMAL_OPERATORS["NOT"]), 
                              ("PAST", MINIMAL_OPERATORS["PAST"]),
                              ("PLURAL", MINIMAL_OPERATORS["PLURAL"]),
                              ("FUTURE", MINIMAL_OPERATORS["FUTURE"]),
                              ("MODAL", MINIMAL_OPERATORS["MODAL"])]:
        print(f"    算子 {op_name}...")
        layer_deltas = defaultdict(list)
        
        for base_sent, op_sent in op_data["pairs"][:15]:
            hs_base = get_hidden_states(model, tokenizer, input_device, base_sent)
            hs_op = get_hidden_states(model, tokenizer, input_device, op_sent)
            
            if hs_base is None or hs_op is None:
                continue
            
            min_layers = min(len(hs_base), len(hs_op))
            
            for li in range(min_layers):
                h_base = hs_base[li][0, -1, :].float().cpu().numpy()
                h_op = hs_op[li][0, -1, :].float().cpu().numpy()
                delta_norm = np.linalg.norm(h_op - h_base)
                base_norm = np.linalg.norm(h_base)
                rel_change = delta_norm / max(base_norm, 1e-10)
                layer_deltas[li].append(rel_change)
        
        results["operator_sensitivity"][op_name] = {}
        for li in sorted(layer_deltas.keys()):
            results["operator_sensitivity"][op_name][f"L{li}"] = {
                "mean_rel_change": float(np.mean(layer_deltas[li])),
                "std_rel_change": float(np.std(layer_deltas[li])),
                "n_pairs": len(layer_deltas[li]),
            }
    
    # === Part 2: 随机扰动 vs 语义方向的传播效率 ===
    # 在embedding层注入不同强度的随机扰动
    # 比较其传播效率与"自然语义差"的传播效率
    print("\n  Part 2: 随机扰动传播效率扫描...")
    
    test_sentences = [p[0] for p in operator_pairs_neg[:5]]
    
    for intensity_idx, (eps_rel, eps_label) in enumerate(zip(intensities, intensity_labels)):
        print(f"    扰动强度 {eps_label}...")
        
        random_layer_deltas = defaultdict(list)
        
        for sent_idx, sentence in enumerate(test_sentences):
            # 获取原始hidden states
            hs_orig = get_hidden_states(model, tokenizer, input_device, sentence)
            if hs_orig is None:
                continue
            
            # 在embedding层注入随机扰动
            ids = tokenizer.encode(sentence, add_special_tokens=False)
            input_ids = torch.tensor([ids], device=input_device)
            attn_mask = torch.ones(1, len(ids), device=input_device, dtype=torch.long)
            
            # 获取embedding
            embed_layer = model.get_input_embeddings()
            embed_orig = embed_layer(input_ids).detach().clone()  # [1, seq, d]
            embed_norm = float(embed_orig[0, -1, :].norm())
            eps_abs = eps_rel * embed_norm
            
            if eps_abs < 1e-10:
                continue
            
            # 注入3个随机方向的扰动（取平均）
            for dir_idx in range(3):
                torch.manual_seed(42 + sent_idx * 10 + dir_idx)
                rand_dir = torch.randn(d_model, device=input_device, dtype=embed_orig.dtype)
                rand_dir = rand_dir / rand_dir.norm() * eps_abs
                
                embed_perturbed = embed_orig.clone()
                embed_perturbed[0, -1, :] += rand_dir
                
                with torch.no_grad():
                    try:
                        out_perturbed = model(inputs_embeds=embed_perturbed, 
                                            attention_mask=attn_mask,
                                            output_hidden_states=True)
                    except Exception:
                        continue
                
                # 计算各层的相对变化
                for li in range(min(len(hs_orig), len(out_perturbed.hidden_states))):
                    h_orig = hs_orig[li][0, -1, :].float().cpu().numpy()
                    h_perturbed = out_perturbed.hidden_states[li][0, -1, :].float().cpu().numpy()
                    delta_norm = np.linalg.norm(h_perturbed - h_orig)
                    base_norm = np.linalg.norm(h_orig)
                    rel_change = delta_norm / max(base_norm, 1e-10)
                    random_layer_deltas[li].append(rel_change)
        
        # 保存随机扰动在各层的相对变化
        results["semantic_vs_random"][eps_label] = {
            "random_mean_rel_change": {},
        }
        for li in sorted(random_layer_deltas.keys()):
            results["semantic_vs_random"][eps_label]["random_mean_rel_change"][f"L{li}"] = {
                "mean": float(np.mean(random_layer_deltas[li])),
                "std": float(np.std(random_layer_deltas[li])),
                "n": len(random_layer_deltas[li]),
            }
        
        # 计算语义方向在各层的相对变化（从Part 1获取）
        for op_name in ["NOT", "PAST"]:
            if op_name in results["operator_sensitivity"]:
                results["semantic_vs_random"][eps_label][f"{op_name}_mean_rel_change"] = {}
                for layer_key, data in results["operator_sensitivity"][op_name].items():
                    results["semantic_vs_random"][eps_label][f"{op_name}_mean_rel_change"][layer_key] = data["mean_rel_change"]
        
        # 计算传播效率比 = 语义方向rel_change / 随机方向rel_change
        # 如果 < 1: 语义方向更稳定（沿流形切向）
        # 如果 > 1: 语义方向更敏感（横穿流形）
        for op_name in ["NOT", "PAST"]:
            sem_key = f"{op_name}_mean_rel_change"
            rand_key = "random_mean_rel_change"
            if sem_key in results["semantic_vs_random"][eps_label]:
                ratio_data = {}
                for layer_key in results["semantic_vs_random"][eps_label][sem_key]:
                    if layer_key in results["semantic_vs_random"][eps_label][rand_key]:
                        sem_val = results["semantic_vs_random"][eps_label][sem_key][layer_key]
                        rand_val = results["semantic_vs_random"][eps_label][rand_key][layer_key]["mean"]
                        if rand_val > 1e-10:
                            ratio_data[layer_key] = sem_val / rand_val
                results["semantic_vs_random"][eps_label][f"{op_name}_vs_random_ratio"] = ratio_data
        
        # 打印关键数据
        rand_data = results["semantic_vs_random"][eps_label]["random_mean_rel_change"]
        sample_layers_str = ""
        for li_str in sorted(rand_data.keys(), key=lambda x: int(x[1:])):
            li_int = int(li_str[1:])
            if li_int % max(1, n_layers // 6) == 0:
                sample_layers_str += f"{li_str}:{rand_data[li_str]['mean']:.4f} "
        print(f"      随机rel_change: {sample_layers_str}")
    
    return results


# ============================================================
# Exp B: LM Head Jacobian SVD 直接计算
# ============================================================
# 核心目标：真正计算 LM head 的奇异值谱
# 不再间接推断，直接做 SVD

def expB_lm_head_svd_with_W_U(W_U, wu_shape, model_name: str):
    """
    Exp B: LM Head SVD分析（直接传入W_U矩阵）
    
    LM head 将 hidden state [d_model] 映射到 logits [vocab_size]
    计算 W_U^T @ W_U 的特征值分解（比直接SVD更省内存）
    """
    vocab_size, d_model_wu = wu_shape
    print(f"  W_U shape: {vocab_size} x {d_model_wu}")
    
    # 计算 W_U^T @ W_U 特征值分解
    print("  计算 W_U^T @ W_U 特征值分解...")
    t0 = time.time()
    
    W_U_f32 = W_U.astype(np.float32)
    del W_U
    import gc; gc.collect()
    
    # 分块计算 W_U^T @ W_U 以节省内存
    # W_U^T: [d_model, vocab], W_U: [vocab, d_model]
    # 直接矩阵乘可能太大，分块处理
    print(f"  计算 WtW ({d_model_wu}x{d_model_wu})...")
    WtW = np.zeros((d_model_wu, d_model_wu), dtype=np.float32)
    chunk_size = 10000  # 每次处理10000行
    for i in range(0, vocab_size, chunk_size):
        end = min(i + chunk_size, vocab_size)
        chunk = W_U_f32[i:end, :]  # [chunk, d_model]
        WtW += chunk.T @ chunk
        if (i // chunk_size) % 5 == 0:
            print(f"    进度: {end}/{vocab_size} ({end/vocab_size*100:.0f}%)")
    
    del W_U_f32; gc.collect()
    
    # 特征值分解
    print("  特征值分解...")
    eigenvalues, eigenvectors = np.linalg.eigh(WtW)
    del WtW; gc.collect()
    
    # 特征值 → 奇异值 (降序)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    S = np.sqrt(np.maximum(eigenvalues, 0))
    
    print(f"  特征值分解完成: {time.time()-t0:.1f}s, top10 S = {S[:10].tolist()}")
    
    # 分析奇异值谱
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy
    
    # 找到90%/95%/99%能量所需的维度数
    dim_90 = int(np.searchsorted(cumulative_energy, 0.90)) + 1
    dim_95 = int(np.searchsorted(cumulative_energy, 0.95)) + 1
    dim_99 = int(np.searchsorted(cumulative_energy, 0.99)) + 1
    
    # 有效秩 (参与度>1%的奇异值数量)
    participation = S**2 / total_energy
    eff_rank = int(np.sum(participation > 0.01))
    
    # 谱衰减率
    top1 = S[0]
    top10_ratio = S[9] / S[0] if len(S) > 9 else None
    top100_ratio = S[99] / S[0] if len(S) > 99 else None
    top500_ratio = S[min(499, len(S)-1)] / S[0] if len(S) > 499 else None
    
    # 条件数 (使用最小非零奇异值)
    nonzero_S = S[S > 1e-10]
    cond = S[0] / nonzero_S[-1] if len(nonzero_S) > 0 else float('inf')
    
    results = {
        "model": model_name,
        "W_U_shape": wu_shape,
        "full_svd": False,
        "n_singular_values": len(S),
        "singular_values_top50": S[:50].tolist(),
        "singular_values_sampled": S[np.linspace(0, len(S)-1, min(50, len(S)), dtype=int)].tolist(),
        "total_energy": float(total_energy),
        "dim_90pct": dim_90,
        "dim_95pct": dim_95,
        "dim_99pct": dim_99,
        "effective_rank_1pct": eff_rank,
        "condition_number": float(cond),
        "top1_singular": float(top1),
        "top10_ratio": float(top10_ratio) if top10_ratio else None,
        "top100_ratio": float(top100_ratio) if top100_ratio else None,
        "top500_ratio": float(top500_ratio) if top500_ratio else None,
        "participation_ratio": float(np.sum(participation > 0.001)),
        "spectral_decay_type": "exponential" if top10_ratio and top10_ratio < 0.1 else "power_law" if top10_ratio and top10_ratio < 0.5 else "slow",
    }
    
    print(f"\n  === LM Head SVD 结果 ===")
    print(f"  条件数: {cond:.1f}")
    print(f"  90%能量维度: {dim_90}/{d_model_wu} ({dim_90/d_model_wu*100:.1f}%)")
    print(f"  95%能量维度: {dim_95}/{d_model_wu} ({dim_95/d_model_wu*100:.1f}%)")
    print(f"  99%能量维度: {dim_99}/{d_model_wu} ({dim_99/d_model_wu*100:.1f}%)")
    print(f"  有效秩(1%): {eff_rank}")
    top10_str = f"{top10_ratio:.4f}" if top10_ratio else "N/A"
    top100_str = f"{top100_ratio:.4f}" if top100_ratio else "N/A"
    print(f"  Top1: {top1:.2f}, Top10/Top1: {top10_str}, Top100/Top1: {top100_str}")
    
    return results


def expB_lm_head_svd(model, tokenizer, device, model_info, model_name: str):
    """
    Exp B: LM Head SVD分析
    
    LM head 将 hidden state [d_model] 映射到 logits [vocab_size]
    这是一个线性映射 W_U @ h + b
    
    直接计算 SVD: W_U = U S V^T
    看 S 的衰减模式
    """
    print("\n  获取LM head权重矩阵...")
    W_U = get_W_U(model, model_name)  # [vocab_size, d_model]
    vocab_size, d_model_wu = W_U.shape
    print(f"  W_U shape: {vocab_size} x {d_model_wu}")
    wu_shape = [vocab_size, d_model_wu]  # 保存shape，后面W_U会被释放
    
    # 直接计算W_U^T @ W_U的特征值分解(比SVD更省内存)
    # 因为 W_U^T W_U 的特征值 = W_U 奇异值的平方
    print("  计算 W_U^T @ W_U 特征值分解...")
    t0 = time.time()
    
    W_U_f32 = W_U.astype(np.float32)
    del W_U
    import gc; gc.collect()
    
    # W_U^T @ W_U: [d_model, d_model] — 远小于 [vocab, d_model]
    WtW = W_U_f32.T @ W_U_f32  # [d_model, d_model]
    del W_U_f32; gc.collect()
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(WtW)
    del WtW; gc.collect()
    
    # 特征值 → 奇异值 (降序)
    eigenvalues = eigenvalues[::-1]  # eigh返回升序
    eigenvectors = eigenvectors[:, ::-1]
    S = np.sqrt(np.maximum(eigenvalues, 0))  # 奇异值 = sqrt(特征值)
    Vt = eigenvectors.T  # 右奇异向量
    # 不计算U(太大), 只需要S和Vt
    
    full_svd = False
    
    print(f"  特征值分解完成: {time.time()-t0:.1f}s, top10 S = {S[:10].tolist()}")
    
    # 分析奇异值谱
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy
    
    # 找到90%/95%/99%能量所需的维度数
    dim_90 = int(np.searchsorted(cumulative_energy, 0.90)) + 1
    dim_95 = int(np.searchsorted(cumulative_energy, 0.95)) + 1
    dim_99 = int(np.searchsorted(cumulative_energy, 0.99)) + 1
    
    # 有效秩 (参与度>1%的奇异值数量)
    participation = S**2 / total_energy
    eff_rank = int(np.sum(participation > 0.01))
    
    # 谱衰减率
    top1 = S[0]
    top10_ratio = S[9] / S[0] if len(S) > 9 else None
    top100_ratio = S[99] / S[0] if len(S) > 99 else None
    top500_ratio = S[min(499, len(S)-1)] / S[0] if len(S) > 499 else None
    
    # 条件数 (使用最小非零奇异值)
    nonzero_S = S[S > 1e-10]
    cond = S[0] / nonzero_S[-1] if len(nonzero_S) > 0 else float('inf')
    
    results = {
        "model": model_name,
        "W_U_shape": wu_shape,
        "full_svd": full_svd,
        "n_singular_values": len(S),
        "singular_values_top50": S[:50].tolist(),
        "singular_values_sampled": S[np.linspace(0, len(S)-1, min(50, len(S)), dtype=int)].tolist(),
        "total_energy": float(total_energy),
        "dim_90pct": dim_90,
        "dim_95pct": dim_95,
        "dim_99pct": dim_99,
        "effective_rank_1pct": eff_rank,
        "condition_number": float(cond),
        "top1_singular": float(top1),
        "top10_ratio": float(top10_ratio) if top10_ratio else None,
        "top100_ratio": float(top100_ratio) if top100_ratio else None,
        "top500_ratio": float(top500_ratio),
        "participation_ratio": float(np.sum(participation > 0.001)),
        "spectral_decay_type": "exponential" if top10_ratio and top10_ratio < 0.1 else "power_law" if top10_ratio and top10_ratio < 0.5 else "slow",
    }
    
    print(f"\n  === LM Head SVD 结果 ===")
    print(f"  条件数: {cond:.1f}")
    print(f"  90%能量维度: {dim_90}/{d_model_wu} ({dim_90/d_model_wu*100:.1f}%)")
    print(f"  95%能量维度: {dim_95}/{d_model_wu} ({dim_95/d_model_wu*100:.1f}%)")
    print(f"  99%能量维度: {dim_99}/{d_model_wu} ({dim_99/d_model_wu*100:.1f}%)")
    print(f"  有效秩(1%): {eff_rank}")
    print(f"  Top1: {top1:.2f}, Top10/Top1: {top10_ratio:.4f}, Top100/Top1: {top100_ratio:.4f}")
    
    return results


# ============================================================
# Exp C: 语言算子代数 — 交换性和干涉项
# ============================================================
# 核心量：
#   干涉项 I_ij = Δh(O_i O_j(x)) - Δh(O_i(x)) - Δh(O_j(x))
#   如果 I_ij ≈ 0: 算子线性可加
#   如果 I_ij ≠ 0: 算子有干涉（非线性交互）

def expC_operator_algebra(model, tokenizer, device, model_info, model_name: str):
    """
    Exp C: 语言算子代数
    
    研究算子的：
    1. 每层的算子响应 (哪些层对哪些算子敏感)
    2. 算子干涉项 (算子是否线性可加)
    3. 算子方向的重叠度 (不同算子是否沿相同方向)
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_input_device(model)
    
    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    # === Part 1: 单算子响应分析 ===
    print("\n  Part 1: 单算子响应分析...")
    operator_responses = {}  # {op_name: {layer: {norm, direction_overlap}}}
    
    for op_name, op_data in MINIMAL_OPERATORS.items():
        print(f"    算子 {op_name}...")
        op_results = {}
        
        for pair_idx, (base_sent, op_sent) in enumerate(op_data["pairs"][:15]):
            hs_base = get_hidden_states(model, tokenizer, input_device, base_sent)
            hs_op = get_hidden_states(model, tokenizer, input_device, op_sent)
            
            if hs_base is None or hs_op is None:
                continue
            
            # 取共同长度
            min_seq = min(hs_base[0].shape[1], hs_op[0].shape[1])
            
            for li in sample_layers:
                if li + 1 >= len(hs_base) or li + 1 >= len(hs_op):
                    continue
                
                h_base = hs_base[li+1][0, -1, :].float().cpu().numpy()
                h_op = hs_op[li+1][0, -1, :].float().cpu().numpy()
                delta = h_op - h_base
                delta_norm = np.linalg.norm(delta)
                h_base_norm = np.linalg.norm(h_base)
                
                # 相对变化
                rel_change = delta_norm / max(h_base_norm, 1e-10)
                
                if f"L{li}" not in op_results:
                    op_results[f"L{li}"] = {"deltas": [], "rel_changes": [], "delta_norms": []}
                
                op_results[f"L{li}"]["deltas"].append(delta)
                op_results[f"L{li}"]["rel_changes"].append(rel_change)
                op_results[f"L{li}"]["delta_norms"].append(delta_norm)
        
        # 聚合
        operator_responses[op_name] = {}
        for layer_key, data in op_results.items():
            if not data["deltas"]:
                continue
            
            # 平均delta方向
            mean_delta = np.mean(data["deltas"], axis=0)
            mean_delta_norm = np.linalg.norm(mean_delta)
            
            # 方向一致性：各个pair的delta与平均delta的cosine
            if mean_delta_norm > 1e-10:
                cosines = [np.dot(d, mean_delta) / (np.linalg.norm(d) * mean_delta_norm + 1e-10) 
                          for d in data["deltas"] if np.linalg.norm(d) > 1e-10]
                direction_consistency = float(np.mean(cosines)) if cosines else 0.0
            else:
                direction_consistency = 0.0
            
            operator_responses[op_name][layer_key] = {
                "mean_rel_change": float(np.mean(data["rel_changes"])),
                "std_rel_change": float(np.std(data["rel_changes"])),
                "mean_delta_norm": float(np.mean(data["delta_norms"])),
                "direction_consistency": direction_consistency,
                "n_pairs": len(data["deltas"]),
            }
    
    # === Part 2: 算子间方向重叠 ===
    print("\n  Part 2: 算子间方向重叠...")
    operator_overlap = {}
    
    # 计算每层每个算子的平均方向
    operator_directions = {}  # {op_name: {layer: unit_vector}}
    for op_name in MINIMAL_OPERATORS:
        operator_directions[op_name] = {}
        for layer_key, data in operator_responses.get(op_name, {}).items():
            # 重新计算平均方向
            op_data = MINIMAL_OPERATORS[op_name]
            deltas = []
            for base_sent, op_sent in op_data["pairs"][:10]:
                hs_base = get_hidden_states(model, tokenizer, input_device, base_sent)
                hs_op = get_hidden_states(model, tokenizer, input_device, op_sent)
                if hs_base is None or hs_op is None:
                    continue
                li = int(layer_key[1:])
                if li + 1 >= len(hs_base) or li + 1 >= len(hs_op):
                    continue
                delta = hs_op[li+1][0, -1, :].float().cpu().numpy() - hs_base[li+1][0, -1, :].float().cpu().numpy()
                deltas.append(delta)
            
            if deltas:
                mean_delta = np.mean(deltas, axis=0)
                norm = np.linalg.norm(mean_delta)
                if norm > 1e-10:
                    operator_directions[op_name][layer_key] = mean_delta / norm
    
    # 计算算子间的余弦相似度
    op_names = list(MINIMAL_OPERATORS.keys())
    for i, op1 in enumerate(op_names):
        for j, op2 in enumerate(op_names):
            if j <= i:
                continue
            overlap_key = f"{op1}_vs_{op2}"
            operator_overlap[overlap_key] = {}
            for layer_key in operator_directions.get(op1, {}):
                if layer_key in operator_directions.get(op2, {}):
                    d1 = operator_directions[op1][layer_key]
                    d2 = operator_directions[op2][layer_key]
                    cos = float(np.dot(d1, d2))
                    operator_overlap[overlap_key][layer_key] = cos
    
    # === Part 3: 干涉项分析 ===
    print("\n  Part 3: 干涉项分析...")
    interference_results = {}
    
    for comp_name, comp_pairs in COMPOSITION_PAIRS:
        print(f"    组合 {comp_name}...")
        comp_interference = []
        
        for quad in comp_pairs:
            base, op1_sent, op2_sent, both_sent = quad
            
            hs_base = get_hidden_states(model, tokenizer, input_device, base)
            hs_op1 = get_hidden_states(model, tokenizer, input_device, op1_sent)
            hs_op2 = get_hidden_states(model, tokenizer, input_device, op2_sent)
            hs_both = get_hidden_states(model, tokenizer, input_device, both_sent)
            
            if any(hs is None for hs in [hs_base, hs_op1, hs_op2, hs_both]):
                continue
            
            for li in sample_layers:
                if any(li+1 >= len(hs) for hs in [hs_base, hs_op1, hs_op2, hs_both]):
                    continue
                
                h_base = hs_base[li+1][0, -1, :].float().cpu().numpy()
                h_op1 = hs_op1[li+1][0, -1, :].float().cpu().numpy()
                h_op2 = hs_op2[li+1][0, -1, :].float().cpu().numpy()
                h_both = hs_both[li+1][0, -1, :].float().cpu().numpy()
                
                # Δh(O_1) = h(O_1(x)) - h(x)
                delta_o1 = h_op1 - h_base
                # Δh(O_2) = h(O_2(x)) - h(x)
                delta_o2 = h_op2 - h_base
                # Δh(O_1 O_2) = h(O_1 O_2(x)) - h(x)
                delta_both = h_both - h_base
                # 干涉项: I = Δh(O_1 O_2) - Δh(O_1) - Δh(O_2)
                interference = delta_both - delta_o1 - delta_o2
                
                delta_o1_norm = np.linalg.norm(delta_o1)
                delta_o2_norm = np.linalg.norm(delta_o2)
                delta_both_norm = np.linalg.norm(delta_both)
                interference_norm = np.linalg.norm(interference)
                
                # 归一化干涉项
                if delta_both_norm > 1e-10:
                    relative_interference = interference_norm / delta_both_norm
                else:
                    relative_interference = float('nan')
                
                # 线性预测 vs 实际
                if delta_both_norm > 1e-10:
                    linear_pred_norm = np.linalg.norm(delta_o1 + delta_o2)
                    linearity_ratio = delta_both_norm / max(linear_pred_norm, 1e-10)
                else:
                    linearity_ratio = float('nan')
                
                comp_interference.append({
                    "layer": f"L{li}",
                    "delta_o1_norm": float(delta_o1_norm),
                    "delta_o2_norm": float(delta_o2_norm),
                    "delta_both_norm": float(delta_both_norm),
                    "interference_norm": float(interference_norm),
                    "relative_interference": float(relative_interference) if not np.isnan(relative_interference) else None,
                    "linearity_ratio": float(linearity_ratio) if not np.isnan(linearity_ratio) else None,
                })
        
        if comp_interference:
            interference_results[comp_name] = {
                "pair_results": comp_interference,
                "summary": {},
            }
            
            # 按层聚合
            by_layer = defaultdict(list)
            for item in comp_interference:
                by_layer[item["layer"]].append(item)
            
            for layer_key, items in by_layer.items():
                rel_ints = [i["relative_interference"] for i in items if i["relative_interference"] is not None]
                lin_ratios = [i["linearity_ratio"] for i in items if i["linearity_ratio"] is not None]
                
                interference_results[comp_name]["summary"][layer_key] = {
                    "mean_relative_interference": float(np.mean(rel_ints)) if rel_ints else None,
                    "mean_linearity_ratio": float(np.mean(lin_ratios)) if lin_ratios else None,
                    "n_pairs": len(items),
                }
            
            print(f"      {comp_name}: {len(comp_interference)} 层-对组合")
    
    results = {
        "model": model_name,
        "operator_responses": operator_responses,
        "operator_overlap": operator_overlap,
        "interference_results": interference_results,
    }
    
    return results


# ============================================================
# 主函数
# ============================================================

def run_all_experiments(model_name: str):
    """运行所有实验"""
    print(f"\n{'='*60}")
    print(f"Phase 140: 语言算子力学 — {model_name}")
    print(f"{'='*60}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    print(f"\n模型信息: {model_info.model_class}, {model_info.n_layers}层, d={model_info.d_model}")
    
    all_results = {
        "model_name": model_name,
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "vocab_size": model_info.vocab_size,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Exp A: 微扰强度扫描
    print(f"\n{'='*40}")
    print("Exp A: 微扰强度扫描")
    print(f"{'='*40}")
    t0 = time.time()
    all_results["expA"] = expA_perturbation_intensity_scan(model, tokenizer, device, model_info, model_name)
    print(f"Exp A 完成: {time.time()-t0:.1f}s")
    
    # Exp B: LM Head SVD
    # 在做SVD前先获取W_U，然后释放模型以节省内存
    print(f"\n{'='*40}")
    print("Exp B: LM Head SVD")
    print(f"{'='*40}")
    t0 = time.time()
    
    # 先获取W_U（需要模型）
    print("  获取LM head权重矩阵(在释放模型前)...")
    W_U_for_expB = get_W_U(model, model_name)
    wu_shape_expB = list(W_U_for_expB.shape)
    print(f"  W_U shape: {wu_shape_expB}")
    
    # 释放模型
    release_model(model)
    print("  模型已释放，开始计算SVD...")
    
    all_results["expB"] = expB_lm_head_svd_with_W_U(W_U_for_expB, wu_shape_expB, model_name)
    print(f"Exp B 完成: {time.time()-t0:.1f}s")
    
    # 重新加载模型（Exp C需要）
    model, tokenizer, device = load_model(model_name)
    
    # Exp C: 语言算子代数
    print(f"\n{'='*40}")
    print("Exp C: 语言算子代数")
    print(f"{'='*40}")
    t0 = time.time()
    all_results["expC"] = expC_operator_algebra(model, tokenizer, device, model_info, model_name)
    print(f"Exp C 完成: {time.time()-t0:.1f}s")
    
    # 释放模型
    release_model(model)
    
    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase140_operator_mechanics.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)
    
    model_name = sys.argv[1].lower()
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    results = run_all_experiments(model_name)
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M")
    filename = f"tests/glm5_temp/phase140_{model_name}_operator_mechanics_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {filename}")
