"""
Phase 139: Jacobian Geometry Analysis
======================================

批评的核心修正:
1. "混沌"推断不成立 — 没有测Lyapunov指数, 只有有限深度30-40步
2. 每层参数不同, 不是标准动力系统, 更不是经典混沌
3. 真正的数学对象可能是"非正规传播系统" (non-normal propagator)
   即使特征值稳定, 也会出现巨大瞬时放大
4. 核心实验: Jacobian SVD → 三种可能:
   A) 稳定收缩: 所有σ < 1
   B) 稀疏放大: 少数σ >> 1, 大多数σ < 1 (最可能)
   C) 临界系统: σ ≈ 1 到处 (支持criticality)
5. 需要停止宏大隐喻, 建立最小数学闭环

Phase 139三个实验:
  Exp 1: Jacobian Singular Value Spectrum
         - 每层J_l = ∂h_{l+1}[pos]/∂h_l[pos]的奇异值谱 (randomized SVD)
         - 有效秩, 条件数, 放大方向比例
         - 非正规度 ||JJ^T - J^TJ||_F / ||J||_F²
  Exp 2: Semantic vs Random Perturbation Propagation
         - 否定方向 vs 时态方向 vs 随机方向的传播差异
         - 语义方向是否沿"稳定方向"传播?
  Exp 3: Layer-to-Output Sensitivity
         - ∂logits/∂h_l[pos]的谱结构 (通过backward pass)
         - 哪些h_l方向最影响最终输出?
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
import gc
import math
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS, get_W_U
)


# ============================================================
# 句子设计 — 多样化, 覆盖不同语法结构
# ============================================================

# 否定对 (用于Exp 2的语义方向)
NEGATION_PAIRS = [
    ("The dog always bites the man", "The dog never bites the man"),
    ("The cat always chases the mouse", "The cat never chases the mouse"),
    ("The sun always rises early", "The sun never rises early"),
    ("The river always flows south", "The river never flows south"),
    ("The wind always blows hard", "The wind never blows hard"),
    ("The bird always sings loud", "The bird never sings loud"),
    ("The fire always burns hot", "The fire never burns hot"),
    ("The doctor always helps patients", "The doctor never helps patients"),
    ("The teacher always reads books", "The teacher never reads books"),
    ("The soldier always fights hard", "The soldier never fights hard"),
    ("The farmer always grows crops", "The farmer never grows crops"),
    ("The artist always paints well", "The artist never paints well"),
]

# 时态对 (用于Exp 2的语义方向)
TENSE_PAIRS = [
    ("The dog bites the man", "The dog bit the man"),
    ("The cat chases the mouse", "The cat chased the mouse"),
    ("The sun rises early", "The sun rose early"),
    ("The river flows south", "The river flowed south"),
    ("The wind blows hard", "The wind blew hard"),
    ("The bird sings loud", "The bird sang loud"),
    ("The fire burns hot", "The fire burnt hot"),
    ("The doctor helps patients", "The doctor helped patients"),
    ("The teacher reads books", "The teacher read books"),
    ("The soldier fights hard", "The soldier fought hard"),
    ("The farmer grows crops", "The farmer grew crops"),
    ("The artist paints well", "The artist painted well"),
]

# 通用测试句子 (用于Exp 1的Jacobian谱)
GENERAL_SENTENCES = [
    "The dog always bites the man",
    "The cat never chases the mouse",
    "The sun rises early every morning",
    "The children play happily outside",
    "The birds fly south for winter",
    "The river flows into the ocean",
    "The wind blows hard today",
    "The teacher reads books quietly",
    "The soldier fights bravely always",
    "The farmer grows crops every year",
    "The artist paints beautiful pictures",
    "The doctor helps sick patients",
    "The fire burns hot and bright",
    "The student studies hard for exams",
    "The singer sings softly and clearly",
]


# ============================================================
# 工具函数
# ============================================================

def get_device_for_input(model) -> torch.device:
    """获取输入tensor应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_logit_entropy(logits: np.ndarray) -> float:
    """计算logits的softmax熵"""
    logits_shifted = logits - np.max(logits)
    exp_l = np.exp(logits_shifted)
    probs = exp_l / np.sum(exp_l)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


# ============================================================
# 核心类: 单层运行器 (高效计算Jacobian)
# ============================================================

class JacobianComputer:
    """
    计算每层Jacobian J_l = ∂h_{l+1}[pos]/∂h_l[pos]
    
    方法: 使用register_forward_pre_hook在目标层注入扰动,
         使用register_forward_hook捕获输出, 运行完整模型。
    
    这是可靠的方法, 兼容所有模型架构。
    """
    
    def __init__(self, model, tokenizer, device, model_info):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_info = model_info
        self.layers = get_layers(model)
        self.n_layers = model_info.n_layers
        self.d_model = model_info.d_model
        
        # 缓存的基线hidden states
        self.base_hidden_states = None
        self.base_logits = None
        self._input_ids = None
        self._attention_mask = None
    
    def prepare(self, input_ids, attention_mask):
        """运行完整forward, 缓存hidden states"""
        self._input_ids = input_ids
        self._attention_mask = attention_mask
        
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        self.base_hidden_states = [hs.detach().clone() for hs in out.hidden_states]
        self.base_logits = out.logits[0, -1].float().cpu().numpy()
    
    def compute_jacobian_column(self, layer_idx, pos_idx, v, eps):
        """
        计算Jacobian的一列: J_l @ v ≈ (h_{l+1}(h_l+εv) - h_{l+1}(h_l)) / ε
        
        方法:
        1. 在目标层l注册pre_hook, 给input加eps*v
        2. 在目标层l注册forward_hook, 捕获output
        3. 运行完整模型
        4. 计算 (output_perturbed - output_base) / eps
        
        Args:
            layer_idx: 层索引 l
            pos_idx: 位置索引 (-1表示最后一个token)
            v: 扰动方向 [d_model] numpy数组
            eps: 扰动大小
        
        Returns:
            Jv: [d_model] numpy数组, 近似Jacobian-vector product
        """
        # 基线输出
        h_l1_base = self.base_hidden_states[layer_idx + 1][0, pos_idx, :].float().cpu().numpy()
        
        # 将v转换为tensor
        v_tensor = torch.tensor(v, dtype=torch.float32, device=self.device)
        eps_v = eps * v_tensor
        
        # 捕获输出
        captured_output = {"h_l1": None}
        pre_hook_called = [False]
        
        # Pre-hook: 在层l的input加扰动
        def pre_hook(module, args):
            if not pre_hook_called[0]:
                pre_hook_called[0] = True
                # args[0] = hidden_states
                if len(args) > 0 and isinstance(args[0], torch.Tensor):
                    hs = args[0]
                    perturbed = hs.clone()
                    perturbed[0, pos_idx, :] += eps_v.to(perturbed.dtype)
                    return (perturbed,) + args[1:]
            return args
        
        # Forward hook: 捕获层l的output
        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                captured_output["h_l1"] = output[0].detach().clone()
            else:
                captured_output["h_l1"] = output.detach().clone()
        
        # 注册hooks
        hooks = []
        hooks.append(self.layers[layer_idx].register_forward_pre_hook(pre_hook))
        hooks.append(self.layers[layer_idx].register_forward_hook(capture_hook))
        
        try:
            with torch.no_grad():
                self.model(
                    input_ids=self._input_ids,
                    attention_mask=self._attention_mask
                )
        except Exception as e:
            print(f"    Forward failed at L{layer_idx}: {e}")
            for h in hooks:
                h.remove()
            return None
        
        for h in hooks:
            h.remove()
        
        if captured_output["h_l1"] is None:
            return None
        
        h_l1_perturbed = captured_output["h_l1"][0, pos_idx, :].float().cpu().numpy()
        Jv = (h_l1_perturbed - h_l1_base) / eps
        
        return Jv
    
    def cleanup(self):
        """释放缓存"""
        self.base_hidden_states = None
        self.base_logits = None
        self._input_ids = None
        self._attention_mask = None
        torch.cuda.empty_cache()


# ============================================================
# Exp 1: Jacobian Singular Value Spectrum
# ============================================================

def exp1_jacobian_spectrum(model, tokenizer, device, model_info, model_name: str):
    """
    核心实验: 计算每层Jacobian J_l = ∂h_{l+1}[pos]/∂h_l[pos]的奇异值谱
    
    方法:
    1. 对每层, 用randomized probing计算J @ V (V为随机正交矩阵)
    2. SVD(JV)给出近似top-k奇异值
    3. 分析: 谱形, 有效秩, 条件数, 放大方向比例, 非正规度
    
    关键区分:
    - 稳定收缩: 所有σ < 1
    - 稀疏放大: 少数σ >> 1, 大多数σ < 1 (最可能)
    - 临界系统: σ ≈ 1 到处
    """
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_device_for_input(model)
    
    # 采样层: 每~5层一个 + 首 + 尾
    sample_step = max(1, n_layers // 7)
    sample_layers = sorted(set(
        list(range(0, n_layers, sample_step)) + [n_layers - 1]
    ))
    
    # 随机探针数量
    n_probes = min(128, d_model)  # 不超过d_model
    
    # 测试句子 (15个, 增加统计稳健性)
    test_sentences = GENERAL_SENTENCES
    
    # 扰动大小: h_l范数的1%
    eps_ratio = 0.01
    
    results = {"per_sentence": {}, "aggregated": {}}
    
    for sent_idx, sentence in enumerate(test_sentences):
        print(f"\n  Sentence {sent_idx+1}/{len(test_sentences)}: '{sentence}'")
        
        ids = tokenizer.encode(sentence, add_special_tokens=False)
        seq_len = len(ids)
        input_ids = torch.tensor([ids], device=input_device)
        attention_mask = torch.ones(1, seq_len, device=input_device, dtype=torch.long)
        
        # 准备Jacobian计算器
        runner = JacobianComputer(model, tokenizer, device, model_info)
        runner.prepare(input_ids, attention_mask)
        
        sent_results = {}
        
        for li in sample_layers:
            lk = f"L{li}"
            print(f"    Computing Jacobian spectrum at {lk}...")
            
            # 获取h_l的范数 (用于确定eps)
            h_l = runner.base_hidden_states[li]  # 层l的输入
            h_l_norm = float(h_l[0, -1, :].norm())
            eps = eps_ratio * h_l_norm
            
            if eps < 1e-8:
                print(f"      SKIP: h_l norm too small ({h_l_norm:.6f})")
                continue
            
            # 生成随机正交探针
            torch.manual_seed(42 + li)
            V_raw = torch.randn(n_probes, d_model)
            V_orth, _ = torch.linalg.qr(V_raw.T)  # [d_model, n_probes]
            V_orth = V_orth.T  # [n_probes, d_model]
            V_np = V_orth.numpy()
            
            # 计算J @ V (逐列)
            Y = np.zeros((d_model, n_probes))
            n_success = 0
            
            for k in range(n_probes):
                v = V_np[k]  # [d_model]
                Jv = runner.compute_jacobian_column(li, -1, v, eps)
                
                if Jv is not None:
                    Y[:, k] = Jv
                    n_success += 1
                else:
                    Y[:, k] = 0  # 填零
            
            if n_success < n_probes // 2:
                print(f"      WARNING: only {n_success}/{n_probes} probes succeeded")
                continue
            
            # SVD of Y = J @ V
            # Y的奇异值近似J的top-k奇异值
            U_y, S_y, Vh_y = np.linalg.svd(Y, full_matrices=False)
            
            # 归一化奇异值: σ_norm = σ / σ_base
            # σ_base: 如果h_l没有结构, 扰动放大的"基准"
            # 简化: 直接用σ_y, 它已经反映了J对随机方向的放大
            
            # 有效秩
            S_y_safe = np.maximum(S_y, 1e-10)
            effective_rank = float(np.sum(S_y_safe)**2 / np.sum(S_y_safe**2))
            
            # 条件数 (top-k中)
            condition_number = float(S_y_safe[0] / S_y_safe[-1])
            
            # 放大方向比例: σ > 1 的比例
            fraction_amplifying = float(np.mean(S_y > 1.0))
            
            # 谱统计
            spectrum_stats = {
                "sigma_max": float(S_y[0]),
                "sigma_min": float(S_y[-1]),
                "sigma_median": float(np.median(S_y)),
                "sigma_mean": float(np.mean(S_y)),
                "sigma_std": float(np.std(S_y)),
                "sigma_p90": float(np.percentile(S_y, 90)),
                "sigma_p10": float(np.percentile(S_y, 10)),
                "effective_rank": effective_rank,
                "condition_number": condition_number,
                "fraction_amplifying": fraction_amplifying,
                "n_probes": n_probes,
                "n_success": n_success,
                "eps": float(eps),
                "h_l_norm": float(h_l_norm),
            }
            
            # 完整谱 (用于绘图)
            # 降采样到最多64个点
            n_spectrum = min(64, len(S_y))
            step = max(1, len(S_y) // n_spectrum)
            spectrum_downsampled = S_y[::step].tolist()
            spectrum_stats["spectrum_downsampled"] = spectrum_downsampled
            
            # 非正规度: ||JJ^T - J^TJ||_F / ||J||_F^2
            # 近似: J ≈ Y @ V^T (V正交)
            # JJ^T ≈ Y @ Y^T
            # J^TJ ≈ V^T @ (Y^T @ Y) @ V  — 但V是[d_model, n_probes]的宽矩阵
            # 简化计算: 用Frobenius范数的近似
            JJT_fro_sq = float(np.sum(Y @ Y.T)**2) if d_model <= 4096 else None
            JtJ_fro_sq = None
            
            # 更简单的非正规度近似:
            # ||JJ^T||_F^2 - ||J^TJ||_F^2 = 0 当且仅当J正规
            # ||J^TJ||_F^2 ≈ ||V^T Y^T Y V||_F^2
            # 简化: 用 ||Y^T Y - Y Y^T||_F / ||Y||_F^2 (对n_probes × n_probes和d_model × d_model矩阵)
            # 进一步简化: 只看 Y^T Y 的对称性
            YtY = Y.T @ Y  # [n_probes, n_probes]
            YYt = Y @ Y.T  # [d_model, d_model]
            
            # 实际上对于矩形矩阵Y, Y^TY和YY^T总是半正定的
            # 非正规度应该衡量J = YV^T的正规性
            # ||JJ^T - J^TJ||_F^2 = ||YY^T - V(Y^TY)V^T||_F^2
            
            # 简化: 用 Y^TY 的特征值 vs S_y^2 的差异作为非正规度代理
            # 对于正规矩阵, Y^TY 的特征值应该等于 S_y^2
            eig_YtY = np.linalg.eigvalsh(YtY)
            eig_YtY_sorted = np.sort(eig_YtY)[::-1]
            S_y_sq = S_y[:n_probes]**2
            
            # 非正规度代理: Y^TY的特征值与S_y^2的差异
            if len(eig_YtY_sorted) == len(S_y_sq):
                non_normality_proxy = float(
                    np.linalg.norm(eig_YtY_sorted - S_y_sq) / max(np.linalg.norm(S_y_sq), 1e-10)
                )
            else:
                non_normality_proxy = -1.0  # 无法计算
            
            spectrum_stats["non_normality_proxy"] = non_normality_proxy
            
            # 更准确的非正规度 (对小d_model可行)
            if d_model <= 4096:
                try:
                    # J ≈ Y @ V_orth_numpy.T (V_orth: [n_probes, d_model])
                    # JJ^T = Y @ V^T @ V @ Y^T = Y @ Y^T (V正交)
                    # J^TJ = V^T @ Y^T @ Y @ V
                    
                    V_orth_np = V_np  # [n_probes, d_model]
                    Vt = V_orth_np.T  # [d_model, n_probes]
                    
                    JJT_approx = Y @ Y.T  # [d_model, d_model]
                    JtJ_approx = Vt @ YtY @ V_orth_np  # [d_model, d_model]
                    
                    diff_norm = float(np.linalg.norm(JJT_approx - JtJ_approx, 'fro'))
                    J_fro_sq = float(np.linalg.norm(Y, 'fro')**2)
                    
                    non_normality = diff_norm / max(J_fro_sq, 1e-10)
                    spectrum_stats["non_normality"] = non_normality
                except Exception as e:
                    spectrum_stats["non_normality"] = -1.0
                    print(f"      Non-normality computation failed: {e}")
            
            sent_results[lk] = spectrum_stats
            
            # 打印关键指标
            print(f"      σ_max={S_y[0]:.3f}, σ_min={S_y[-1]:.3f}, "
                  f"σ_med={np.median(S_y):.3f}, "
                  f"eff_rank={effective_rank:.1f}, "
                  f"frac_amp={fraction_amplifying:.3f}, "
                  f"non_norm={spectrum_stats.get('non_normality', -1):.4f}")
        
        results["per_sentence"][sentence] = sent_results
        
        # 清理
        runner.cleanup()
        del runner
        torch.cuda.empty_cache()
    
    # 聚合
    results["aggregated"] = _aggregate_spectrum(results["per_sentence"], sample_layers)
    
    return results


def _aggregate_spectrum(per_sentence, sample_layers):
    """聚合所有句子的谱分析结果"""
    agg = {}
    
    for li in sample_layers:
        lk = f"L{li}"
        
        # 收集该层所有句子的指标
        metrics = defaultdict(list)
        spectra = []
        
        for sent, sent_data in per_sentence.items():
            if lk not in sent_data:
                continue
            d = sent_data[lk]
            
            for key in ["sigma_max", "sigma_min", "sigma_median", "sigma_mean", "sigma_std",
                         "effective_rank", "condition_number", "fraction_amplifying",
                         "non_normality", "non_normality_proxy"]:
                if key in d and d[key] >= 0:
                    metrics[key].append(d[key])
            
            if "spectrum_downsampled" in d:
                spectra.append(d["spectrum_downsampled"])
        
        if not metrics.get("sigma_max"):
            continue
        
        agg[lk] = {}
        for key, vals in metrics.items():
            agg[lk][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }
        
        # 平均谱
        if spectra:
            min_len = min(len(s) for s in spectra)
            trimmed = [s[:min_len] for s in spectra]
            avg_spectrum = np.mean(trimmed, axis=0).tolist()
            std_spectrum = np.std(trimmed, axis=0).tolist()
            agg[lk]["avg_spectrum"] = avg_spectrum
            agg[lk]["std_spectrum"] = std_spectrum
    
    return agg


# ============================================================
# Exp 2: Semantic vs Random Perturbation Propagation
# ============================================================

def exp2_semantic_vs_random(model, tokenizer, device, model_info, model_name: str):
    """
    对比语义方向 vs 随机方向的传播行为
    
    核心问题:
    - 语义扰动(否定/时态)是否沿"稳定方向"传播?
    - 随机扰动是否沿"不稳定方向"传播?
    - 如果是, 说明语言计算有特定的"传播通道"
    
    方法:
    1. 对否定对和时态对, 计算每层的"语义方向" d_semantic = h(alt) - h(base)
    2. 在base句子上, 沿d_semantic方向和随机方向注入扰动
    3. 测量传播比和方向保持
    4. 对比: 语义方向 vs 随机方向
    """
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_device_for_input(model)
    
    # 采样层
    sample_step = max(1, n_layers // 7)
    sample_layers = sorted(set(
        list(range(0, n_layers, sample_step)) + [n_layers - 1]
    ))
    
    # 传播测量层 (从扰动层到最后)
    measure_layers = sorted(set(
        list(range(0, n_layers, max(1, n_layers // 10))) + [n_layers - 1]
    ))
    
    n_random_dirs = 5  # 随机方向数量
    
    results = {"negation": [], "tense": []}
    
    # --- 否定方向 ---
    for pair_idx, (sent_base, sent_alt) in enumerate(NEGATION_PAIRS[:8]):
        print(f"\n  Negation pair {pair_idx+1}/8: '{sent_base}' → '{sent_alt}'")
        
        ids_base = tokenizer.encode(sent_base, add_special_tokens=False)
        ids_alt = tokenizer.encode(sent_alt, add_special_tokens=False)
        
        if len(ids_base) != len(ids_alt):
            print(f"    SKIP: token数不同")
            continue
        
        seq_len = len(ids_base)
        input_ids_base = torch.tensor([ids_base], device=input_device)
        input_ids_alt = torch.tensor([ids_alt], device=input_device)
        attention_mask = torch.ones(1, seq_len, device=input_device, dtype=torch.long)
        
        # 运行两个句子
        with torch.no_grad():
            out_base = model(input_ids=input_ids_base, attention_mask=attention_mask,
                             output_hidden_states=True)
            out_alt = model(input_ids=input_ids_alt, attention_mask=attention_mask,
                            output_hidden_states=True)
        
        hs_base = [hs.detach().clone() for hs in out_base.hidden_states]
        hs_alt = [hs.detach().clone() for hs in out_alt.hidden_states]
        
        # 找到差异位置
        diff_pos = [i for i in range(seq_len) if ids_base[i] != ids_alt[i]]
        if not diff_pos:
            continue
        
        pair_result = {"sentence_base": sent_base, "sentence_alt": sent_alt, "layers": {}}
        
        for li in sample_layers:
            lk = f"L{li}"
            
            # 语义方向: 在差异位置的平均方向
            semantic_dirs = []
            for pos in diff_pos:
                d_sem = (hs_alt[li][0, pos, :] - hs_base[li][0, pos, :]).float().cpu().numpy()
                d_sem_norm = np.linalg.norm(d_sem)
                if d_sem_norm > 1e-8:
                    semantic_dirs.append(d_sem / d_sem_norm)
            
            if not semantic_dirs:
                continue
            
            # 平均语义方向 (归一化)
            avg_semantic = np.mean(semantic_dirs, axis=0)
            avg_semantic = avg_semantic / max(np.linalg.norm(avg_semantic), 1e-10)
            
            # 在last token位置注入扰动, 测量传播
            h_l = hs_base[li]  # 层l的输入
            eps = 0.05 * float(h_l[0, -1, :].norm())  # 5%扰动
            
            # --- 语义方向传播 ---
            sem_propagation = _measure_propagation_single(
                model, tokenizer, input_device, model_info,
                input_ids_base, attention_mask,
                hs_base, li, -1, avg_semantic, eps, measure_layers
            )
            
            # --- 随机方向传播 ---
            random_propagations = []
            for dir_idx in range(n_random_dirs):
                torch.manual_seed(42 + dir_idx + li * 100)
                random_dir = np.random.randn(d_model)
                random_dir = random_dir / np.linalg.norm(random_dir)
                
                rand_prop = _measure_propagation_single(
                    model, tokenizer, input_device, model_info,
                    input_ids_base, attention_mask,
                    hs_base, li, -1, random_dir, eps, measure_layers
                )
                if rand_prop is not None:
                    random_propagations.append(rand_prop)
            
            # 聚合随机方向
            if random_propagations:
                rand_agg = {
                    "prop_ratio_mean": float(np.mean([p["final_prop_ratio"] for p in random_propagations])),
                    "prop_ratio_std": float(np.std([p["final_prop_ratio"] for p in random_propagations])),
                    "dir_preserve_mean": float(np.mean([p["final_dir_preserve"] for p in random_propagations])),
                    "dir_preserve_std": float(np.std([p["final_dir_preserve"] for p in random_propagations])),
                    "logit_shift_mean": float(np.mean([p["logit_shift"] for p in random_propagations])),
                }
            else:
                rand_agg = {"prop_ratio_mean": 0, "prop_ratio_std": 0,
                           "dir_preserve_mean": 0, "dir_preserve_std": 0, "logit_shift_mean": 0}
            
            pair_result["layers"][lk] = {
                "semantic": sem_propagation,
                "random_agg": rand_agg,
                "eps": float(eps),
            }
        
        results["negation"].append(pair_result)
        
        del hs_base, hs_alt
        torch.cuda.empty_cache()
    
    # --- 时态方向 ---
    for pair_idx, (sent_base, sent_alt) in enumerate(TENSE_PAIRS[:8]):
        print(f"\n  Tense pair {pair_idx+1}/8: '{sent_base}' → '{sent_alt}'")
        
        ids_base = tokenizer.encode(sent_base, add_special_tokens=False)
        ids_alt = tokenizer.encode(sent_alt, add_special_tokens=False)
        
        if len(ids_base) != len(ids_alt):
            print(f"    SKIP: token数不同")
            continue
        
        seq_len = len(ids_base)
        input_ids_base = torch.tensor([ids_base], device=input_device)
        input_ids_alt = torch.tensor([ids_alt], device=input_device)
        attention_mask = torch.ones(1, seq_len, device=input_device, dtype=torch.long)
        
        with torch.no_grad():
            out_base = model(input_ids=input_ids_base, attention_mask=attention_mask,
                             output_hidden_states=True)
            out_alt = model(input_ids=input_ids_alt, attention_mask=attention_mask,
                            output_hidden_states=True)
        
        hs_base = [hs.detach().clone() for hs in out_base.hidden_states]
        hs_alt = [hs.detach().clone() for hs in out_alt.hidden_states]
        
        diff_pos = [i for i in range(seq_len) if ids_base[i] != ids_alt[i]]
        if not diff_pos:
            continue
        
        pair_result = {"sentence_base": sent_base, "sentence_alt": sent_alt, "layers": {}}
        
        for li in sample_layers:
            lk = f"L{li}"
            
            semantic_dirs = []
            for pos in diff_pos:
                d_sem = (hs_alt[li][0, pos, :] - hs_base[li][0, pos, :]).float().cpu().numpy()
                d_sem_norm = np.linalg.norm(d_sem)
                if d_sem_norm > 1e-8:
                    semantic_dirs.append(d_sem / d_sem_norm)
            
            if not semantic_dirs:
                continue
            
            avg_semantic = np.mean(semantic_dirs, axis=0)
            avg_semantic = avg_semantic / max(np.linalg.norm(avg_semantic), 1e-10)
            
            h_l = hs_base[li]
            eps = 0.05 * float(h_l[0, -1, :].norm())
            
            sem_propagation = _measure_propagation_single(
                model, tokenizer, input_device, model_info,
                input_ids_base, attention_mask,
                hs_base, li, -1, avg_semantic, eps, measure_layers
            )
            
            random_propagations = []
            for dir_idx in range(n_random_dirs):
                torch.manual_seed(100 + dir_idx + li * 100)
                random_dir = np.random.randn(d_model)
                random_dir = random_dir / np.linalg.norm(random_dir)
                
                rand_prop = _measure_propagation_single(
                    model, tokenizer, input_device, model_info,
                    input_ids_base, attention_mask,
                    hs_base, li, -1, random_dir, eps, measure_layers
                )
                if rand_prop is not None:
                    random_propagations.append(rand_prop)
            
            if random_propagations:
                rand_agg = {
                    "prop_ratio_mean": float(np.mean([p["final_prop_ratio"] for p in random_propagations])),
                    "prop_ratio_std": float(np.std([p["final_prop_ratio"] for p in random_propagations])),
                    "dir_preserve_mean": float(np.mean([p["final_dir_preserve"] for p in random_propagations])),
                    "dir_preserve_std": float(np.std([p["final_dir_preserve"] for p in random_propagations])),
                    "logit_shift_mean": float(np.mean([p["logit_shift"] for p in random_propagations])),
                }
            else:
                rand_agg = {"prop_ratio_mean": 0, "prop_ratio_std": 0,
                           "dir_preserve_mean": 0, "dir_preserve_std": 0, "logit_shift_mean": 0}
            
            pair_result["layers"][lk] = {
                "semantic": sem_propagation,
                "random_agg": rand_agg,
                "eps": float(eps),
            }
        
        results["tense"].append(pair_result)
        
        del hs_base, hs_alt
        torch.cuda.empty_cache()
    
    return results


def _measure_propagation_single(model, tokenizer, input_device, model_info,
                                 input_ids, attention_mask, hs_base,
                                 perturb_layer, pos_idx, direction, eps, measure_layers):
    """测量单个方向扰动的传播行为 (使用full-model hook方式)"""
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    # 构造扰动
    h_l_perturbed = hs_base[perturb_layer].clone()
    dir_tensor = torch.tensor(direction, dtype=h_l_perturbed.dtype, device=h_l_perturbed.device)
    h_l_perturbed[0, pos_idx, :] += eps * dir_tensor
    
    # 基线logits
    with torch.no_grad():
        out_base = model(input_ids=input_ids, attention_mask=attention_mask)
    logits_base = out_base.logits[0, -1].float().cpu().numpy()
    
    # 用hook注入扰动, 捕获后续层
    captured_hs = {}
    inject_done = [False]
    
    def make_inject_hook():
        def hook(module, input, output):
            if not inject_done[0]:
                inject_done[0] = True
                if isinstance(output, tuple):
                    return (h_l_perturbed.to(output[0].device).to(output[0].dtype),) + output[1:]
                return h_l_perturbed.to(output.device).to(output.dtype)
            return output
        return hook
    
    def make_capture_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured_hs[key] = output[0].detach().clone()
            else:
                captured_hs[key] = output.detach().clone()
        return hook
    
    # 注册hooks
    hooks = []
    hooks.append(layers[perturb_layer].register_forward_hook(make_inject_hook()))
    
    for mli in measure_layers:
        if mli > perturb_layer:
            hooks.append(layers[mli].register_forward_hook(make_capture_hook(f"L{mli}")))
    
    try:
        with torch.no_grad():
            out_perturbed = model(input_ids=input_ids, attention_mask=attention_mask)
    except Exception as e:
        for h in hooks:
            h.remove()
        return None
    
    for h in hooks:
        h.remove()
    
    logits_perturbed = out_perturbed.logits[0, -1].float().cpu().numpy()
    logit_shift = float(np.linalg.norm(logits_perturbed - logits_base))
    
    # 传播分析
    final_prop_ratio = 0
    final_dir_preserve = 0
    
    for mli in measure_layers:
        clk = f"L{mli}"
        if clk not in captured_hs:
            continue
        
        delta_h = captured_hs[clk] - hs_base[mli + 1]  # [1, seq_len, d_model]
        delta_h_last = delta_h[0, pos_idx, :].float().cpu().numpy()
        delta_norm = np.linalg.norm(delta_h_last)
        
        prop_ratio = delta_norm / max(eps, 1e-10)
        
        # 方向保持
        if delta_norm > 1e-10:
            dir_preserve = float(np.dot(delta_h_last, direction) / (delta_norm * np.linalg.norm(direction)))
        else:
            dir_preserve = 0.0
        
        if mli == measure_layers[-1] or mli == n_layers - 1:
            final_prop_ratio = prop_ratio
            final_dir_preserve = dir_preserve
    
    return {
        "final_prop_ratio": final_prop_ratio,
        "final_dir_preserve": final_dir_preserve,
        "logit_shift": logit_shift,
    }


# ============================================================
# Exp 3: Layer-to-Output Sensitivity (Backward Analysis)
# ============================================================

def exp3_output_sensitivity(model, tokenizer, device, model_info, model_name: str):
    """
    计算每层的"输出敏感度谱" ∂logits/∂h_l[pos]
    
    方法:
    对每层l, 在h_l[pos]上注入随机扰动, 测量logits的变化方向
    多次重复, 用SVD提取主要的"logit影响方向"
    
    这揭示了: h_l中的哪些方向最影响最终输出
    """
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_device_for_input(model)
    vocab_size = model_info.vocab_size
    
    # 采样层
    sample_step = max(1, n_layers // 7)
    sample_layers = sorted(set(
        list(range(0, n_layers, sample_step)) + [n_layers - 1]
    ))
    
    n_probes = min(64, d_model)  # 输出敏感度用较少探针
    
    test_sentences = GENERAL_SENTENCES[:8]  # 8个句子
    
    results = {"per_sentence": {}, "aggregated": {}}
    
    for sent_idx, sentence in enumerate(test_sentences):
        print(f"\n  Sentence {sent_idx+1}/{len(test_sentences)}: '{sentence}'")
        
        ids = tokenizer.encode(sentence, add_special_tokens=False)
        seq_len = len(ids)
        input_ids = torch.tensor([ids], device=input_device)
        attention_mask = torch.ones(1, seq_len, device=input_device, dtype=torch.long)
        
        # 基线forward
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        
        hs_base = [hs.detach().clone() for hs in out.hidden_states]
        logits_base = out.logits[0, -1].float().cpu().numpy()
        
        sent_results = {}
        
        for li in sample_layers:
            lk = f"L{li}"
            
            # h_l范数
            h_l = hs_base[li]
            h_l_norm = float(h_l[0, -1, :].norm())
            eps = 0.05 * h_l_norm
            
            # 随机探针
            torch.manual_seed(42 + li)
            V_raw = np.random.randn(n_probes, d_model)
            V_raw = V_raw / np.linalg.norm(V_raw, axis=1, keepdims=True)
            
            # 计算每列: Δlogits = logits(h_l + εv) - logits(h_l)
            layers = get_layers(model)
            delta_logits = np.zeros((vocab_size, n_probes))
            n_success = 0
            
            for k in range(n_probes):
                v = V_raw[k]
                h_l_perturbed = hs_base[li].clone()
                v_tensor = torch.tensor(v, dtype=h_l_perturbed.dtype, device=h_l_perturbed.device)
                h_l_perturbed[0, -1, :] += eps * v_tensor
                
                # 注入扰动
                inject_done = [False]
                
                def make_inject(perturbed_hs):
                    def hook(module, input, output):
                        if not inject_done[0]:
                            inject_done[0] = True
                            if isinstance(output, tuple):
                                return (perturbed_hs.to(output[0].device).to(output[0].dtype),) + output[1:]
                            return perturbed_hs.to(output.device).to(output.dtype)
                        return output
                    return hook
                
                hook = layers[li].register_forward_hook(make_inject(h_l_perturbed))
                
                try:
                    with torch.no_grad():
                        out_p = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_p = out_p.logits[0, -1].float().cpu().numpy()
                    delta_logits[:, k] = (logits_p - logits_base) / eps
                    n_success += 1
                except:
                    pass
                
                hook.remove()
            
            if n_success < n_probes // 2:
                continue
            
            # SVD of Δlogits
            U_d, S_d, Vh_d = np.linalg.svd(delta_logits, full_matrices=False)
            
            # 只保留前n_probes个奇异值
            S_d = S_d[:n_probes]
            
            sent_results[lk] = {
                "sensitivity_spectrum": S_d.tolist(),
                "sigma_max": float(S_d[0]) if len(S_d) > 0 else 0,
                "sigma_min": float(S_d[-1]) if len(S_d) > 0 else 0,
                "effective_rank": float(np.sum(S_d)**2 / np.sum(S_d**2)) if len(S_d) > 0 else 0,
                "n_probes": n_probes,
                "n_success": n_success,
            }
            
            print(f"    {lk}: σ_max={S_d[0]:.3f}, σ_min={S_d[-1]:.3f}, "
                  f"eff_rank={sent_results[lk]['effective_rank']:.1f}")
        
        results["per_sentence"][sentence] = sent_results
        
        del hs_base
        torch.cuda.empty_cache()
    
    # 聚合
    results["aggregated"] = _aggregate_sensitivity(results["per_sentence"], sample_layers)
    
    return results


def _aggregate_sensitivity(per_sentence, sample_layers):
    """聚合输出敏感度结果"""
    agg = {}
    
    for li in sample_layers:
        lk = f"L{li}"
        metrics = defaultdict(list)
        spectra = []
        
        for sent, sent_data in per_sentence.items():
            if lk not in sent_data:
                continue
            d = sent_data[lk]
            
            for key in ["sigma_max", "sigma_min", "effective_rank"]:
                if key in d:
                    metrics[key].append(d[key])
            
            if "sensitivity_spectrum" in d:
                spectra.append(d["sensitivity_spectrum"])
        
        if not metrics.get("sigma_max"):
            continue
        
        agg[lk] = {}
        for key, vals in metrics.items():
            agg[lk][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
        
        if spectra:
            min_len = min(len(s) for s in spectra)
            trimmed = [s[:min_len] for s in spectra]
            avg_spectrum = np.mean(trimmed, axis=0).tolist()
            agg[lk]["avg_sensitivity_spectrum"] = avg_spectrum
    
    return agg


# ============================================================
# 简化输出
# ============================================================

def simplify_results(results):
    """简化结果以便JSON存储"""
    import copy
    
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(x) for x in obj]
        return obj
    
    return to_serializable(copy.deepcopy(results))


# ============================================================
# 打印结果摘要
# ============================================================

def print_summary(results, model_name):
    """打印结果摘要"""
    print(f"\n{'='*70}")
    print(f"Phase 139 结果摘要: {model_name}")
    print(f"{'='*70}")
    
    # Exp 1: Jacobian谱
    exp1 = results.get("exp1_jacobian", {})
    agg1 = exp1.get("aggregated", {})
    print(f"\n--- Exp 1: Jacobian Singular Value Spectrum ---")
    print(f"{'Layer':<8} {'σ_max':>8} {'σ_min':>8} {'σ_med':>8} {'eff_rank':>10} "
          f"{'frac_amp':>10} {'non_norm':>10}")
    print("-" * 70)
    
    for lk in sorted(agg1.keys()):
        d = agg1[lk]
        sigma_max = d.get("sigma_max", {}).get("mean", 0)
        sigma_min = d.get("sigma_min", {}).get("mean", 0)
        sigma_med = d.get("sigma_median", {}).get("mean", 0)
        eff_rank = d.get("effective_rank", {}).get("mean", 0)
        frac_amp = d.get("fraction_amplifying", {}).get("mean", 0)
        non_norm = d.get("non_normality", {}).get("mean", -1)
        
        print(f"{lk:<8} {sigma_max:>8.3f} {sigma_min:>8.3f} {sigma_med:>8.3f} "
              f"{eff_rank:>10.1f} {frac_amp:>10.3f} {non_norm:>10.4f}")
    
    # 判断谱形
    print(f"\n谱形判断:")
    all_sigmas_max = [agg1[lk].get("sigma_max", {}).get("mean", 0) for lk in sorted(agg1.keys())]
    all_sigmas_min = [agg1[lk].get("sigma_min", {}).get("mean", 0) for lk in sorted(agg1.keys())]
    all_frac_amp = [agg1[lk].get("fraction_amplifying", {}).get("mean", 0) for lk in sorted(agg1.keys())]
    
    if all_sigmas_max and all_sigmas_min:
        if max(all_sigmas_max) < 1.0:
            print("  → 稳定收缩: 所有σ_max < 1")
        elif min(all_frac_amp) < 0.2 and max(all_sigmas_max) > 2.0:
            print("  → 稀疏放大: 少数σ >> 1, 大多数σ < 1 (frac_amp < 0.2)")
        elif all(s > 0.8 and s < 1.2 for s in all_sigmas_max):
            print("  → 临界系统: σ ≈ 1 到处")
        else:
            print(f"  → 混合模式: σ_max范围 [{min(all_sigmas_max):.2f}, {max(all_sigmas_max):.2f}], "
                  f"frac_amp范围 [{min(all_frac_amp):.3f}, {max(all_frac_amp):.3f}]")
    
    # Exp 2: 语义 vs 随机
    exp2 = results.get("exp2_semantic", {})
    print(f"\n--- Exp 2: Semantic vs Random Perturbation ---")
    
    for category in ["negation", "tense"]:
        pairs = exp2.get(category, [])
        if not pairs:
            continue
        
        print(f"\n  {category.upper()}:")
        sem_props = []
        rand_props = []
        sem_dirs = []
        rand_dirs = []
        
        for pair in pairs:
            for lk, layer_data in pair.get("layers", {}).items():
                sem = layer_data.get("semantic", {})
                rand = layer_data.get("random_agg", {})
                
                if sem:
                    sem_props.append(sem.get("final_prop_ratio", 0))
                    sem_dirs.append(sem.get("final_dir_preserve", 0))
                if rand:
                    rand_props.append(rand.get("prop_ratio_mean", 0))
                    rand_dirs.append(rand.get("dir_preserve_mean", 0))
        
        if sem_props and rand_props:
            print(f"    Semantic: prop_ratio={np.mean(sem_props):.3f}±{np.std(sem_props):.3f}, "
                  f"dir_preserve={np.mean(sem_dirs):.3f}±{np.std(sem_dirs):.3f}")
            print(f"    Random:   prop_ratio={np.mean(rand_props):.3f}±{np.std(rand_props):.3f}, "
                  f"dir_preserve={np.mean(rand_dirs):.3f}±{np.std(rand_dirs):.3f}")
            
            ratio = np.mean(sem_props) / max(np.mean(rand_props), 1e-10)
            dir_diff = np.mean(sem_dirs) - np.mean(rand_dirs)
            print(f"    Semantic/Random ratio: {ratio:.3f}, dir_diff: {dir_diff:.3f}")
    
    # Exp 3: 输出敏感度
    exp3 = results.get("exp3_sensitivity", {})
    agg3 = exp3.get("aggregated", {})
    print(f"\n--- Exp 3: Layer-to-Output Sensitivity ---")
    for lk in sorted(agg3.keys()):
        d = agg3[lk]
        s_max = d.get("sigma_max", {}).get("mean", 0)
        s_min = d.get("sigma_min", {}).get("mean", 0)
        eff_r = d.get("effective_rank", {}).get("mean", 0)
        print(f"  {lk}: σ_max={s_max:.3f}, σ_min={s_min:.3f}, eff_rank={eff_r:.1f}")


# ============================================================
# 主函数
# ============================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    model_name = model_name.lower()
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Phase 139: Jacobian Geometry Analysis — {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.model_class}, {model_info.n_layers} layers, "
          f"d_model={model_info.d_model}")
    
    all_results = {"model_info": {
        "name": model_name,
        "class": model_info.model_class,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
    }}
    
    # === Exp 1: Jacobian Singular Value Spectrum ===
    print(f"\n{'='*70}")
    print("Exp 1: Jacobian Singular Value Spectrum")
    print(f"{'='*70}")
    t0 = time.time()
    exp1_results = exp1_jacobian_spectrum(model, tokenizer, device, model_info, model_name)
    t1 = time.time()
    print(f"Exp 1 done in {t1-t0:.1f}s")
    all_results["exp1_jacobian"] = exp1_results
    
    # === Exp 2: Semantic vs Random Perturbation ===
    print(f"\n{'='*70}")
    print("Exp 2: Semantic vs Random Perturbation")
    print(f"{'='*70}")
    t0 = time.time()
    exp2_results = exp2_semantic_vs_random(model, tokenizer, device, model_info, model_name)
    t1 = time.time()
    print(f"Exp 2 done in {t1-t0:.1f}s")
    all_results["exp2_semantic"] = exp2_results
    
    # === Exp 3: Layer-to-Output Sensitivity ===
    print(f"\n{'='*70}")
    print("Exp 3: Layer-to-Output Sensitivity")
    print(f"{'='*70}")
    t0 = time.time()
    exp3_results = exp3_output_sensitivity(model, tokenizer, device, model_info, model_name)
    t1 = time.time()
    print(f"Exp 3 done in {t1-t0:.1f}s")
    all_results["exp3_sensitivity"] = exp3_results
    
    # 打印摘要
    print_summary(all_results, model_name)
    
    # 保存
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"phase139_{model_name}_jacobian_geometry.json")
    
    simplified = simplify_results(all_results)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")
    
    # 释放模型
    release_model(model)


if __name__ == "__main__":
    main()
