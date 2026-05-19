"""
Phase 225: Jacobian特征向量通用性验证
======================================

核心问题：Phase 223已证明Jacobian谱形状完全通用(Spearman=1.000)。
         但谱形状稳定可能只是架构的平凡属性。
         关键缺口：对应的特征向量子空间是否也通用？
         即：不同句子/上下文下的Jacobian是否具有相同的主方向？

方法：Bootstrap子集采样 + 状态转移矩阵（与Phase 221一致，但关注特征向量而非谱值）

三个实验:
  Exp1 (P0, 核心): Jacobian特征向量子空间Bootstrap稳定性
    - 收集50对句子(4类型)的Δh
    - Bootstrap采样25次 → T_l → top-20右奇异向量
    - 测量跨Bootstrap的主角(principal angles)
    - < 30° = STABLE, < 55° = MODERATE, ≥ 55° = UNSTABLE
    - 与随机基准(~88°)对比

  Exp2 (P1): 约束类型条件Jacobian子空间对齐
    - 分别为SVA/Tense/Voice/Topic计算T_l
    - 测量跨类型主角: 若≈Bootstrap内 → 真正通用

  Exp3 (P2): S_J vs S_Δh vs S_WU三子空间对齐
    - S_J  = T_l的top-30右奇异向量(Jacobian输入主方向)
    - S_Δh = {Δh_l}的top-30 PCA向量(约束几何空间)
    - S_WU = W_U的top-30右奇异向量(可解码方向)
    - J∩Δh vs J∩WU 差异直接检验暗物质假说

用法: python tests/glm5/phase225_jacobian_eigenvector_universality.py qwen3/glm4/deepseek7b/all
结果: tests/glm5_temp/phase225_<model>_results.json

执行: 2026-05-18
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
from pathlib import Path
from model_utils import (get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS,
                          get_sample_layers)

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 日志 =====
_last_log_time = time.time()

def log_status(msg):
    global _last_log_time
    t = time.strftime("%H:%M:%S")
    gpu_mem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    print(f"[{t}] GPU={gpu_mem:.1f}GB | {msg}", flush=True)
    _last_log_time = time.time()

def maybe_log(msg):
    global _last_log_time
    if time.time() - _last_log_time > 30:
        log_status(msg)

# ===== 模型加载 =====
def load_model_bf16(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_status(f"Loading {model_name} (bf16 + auto + sdpa)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    log_status(f"  Loaded: device={device}, GPU={gpu_mem:.1f}GB")

    info = get_model_info(model, model_name)
    log_status(f"Model: {info.model_class}, layers={info.n_layers}, d_model={info.d_model}, mlp_type={info.mlp_type}")
    return model, tokenizer, device, info

# ===== 数据集：50对，4类型 =====

SVA_PAIRS = [  # 20对，主谓数一致
    ("The cat chases", "The cats chase"),
    ("The dog runs", "The dogs run"),
    ("The bird sings", "The birds sing"),
    ("The girl reads", "The girls read"),
    ("The boy walks", "The boys walk"),
    ("The tree falls", "The trees fall"),
    ("The car moves", "The cars move"),
    ("The student writes", "The students write"),
    ("The teacher speaks", "The teachers speak"),
    ("The horse gallops", "The horses gallop"),
    ("The flower blooms", "The flowers bloom"),
    ("The river flows", "The rivers flow"),
    ("The star shines", "The stars shine"),
    ("The wind blows", "The winds blow"),
    ("The rain falls", "The rains fall"),
    ("The clock ticks", "The clocks tick"),
    ("The train stops", "The trains stop"),
    ("The boat sails", "The boats sail"),
    ("The plane flies", "The planes fly"),
    ("The child plays", "The children play"),
]

TENSE_PAIRS = [  # 10对，时态变化（现在→过去）
    ("The cat chases the mouse", "The cat chased the mouse"),
    ("The dog runs fast", "The dog ran fast"),
    ("The bird sings daily", "The bird sang daily"),
    ("The girl reads books", "The girl read books"),
    ("The boy walks home", "The boy walked home"),
    ("The tree falls slowly", "The tree fell slowly"),
    ("The car moves quickly", "The car moved quickly"),
    ("The student writes essays", "The student wrote essays"),
    ("The teacher speaks clearly", "The teacher spoke clearly"),
    ("The horse gallops freely", "The horse galloped freely"),
]

VOICE_PAIRS = [  # 10对，主动→被动
    ("The cat chases the mouse", "The mouse is chased by the cat"),
    ("The dog bites the man", "The man is bitten by the dog"),
    ("The girl reads the book", "The book is read by the girl"),
    ("The boy builds the house", "The house is built by the boy"),
    ("The teacher grades the papers", "The papers are graded by the teacher"),
    ("The chef cooks the food", "The food is cooked by the chef"),
    ("The artist paints the picture", "The picture is painted by the artist"),
    ("The worker fixes the machine", "The machine is fixed by the worker"),
    ("The scientist studies the problem", "The problem is studied by the scientist"),
    ("The author writes the novel", "The novel is written by the author"),
]

TOPIC_PAIRS = [  # 10对，跨领域SVA（与SVA结构相同但话题不同）
    ("The algorithm processes", "The algorithms process"),
    ("The protein folds", "The proteins fold"),
    ("The planet orbits", "The planets orbit"),
    ("The neuron fires", "The neurons fire"),
    ("The market rises", "The markets rise"),
    ("The virus spreads", "The viruses spread"),
    ("The equation holds", "The equations hold"),
    ("The theory predicts", "The theories predict"),
    ("The experiment confirms", "The experiments confirm"),
    ("The measurement shows", "The measurements show"),
]

ALL_PAIRS_BY_TYPE = {
    "sva": SVA_PAIRS,
    "tense": TENSE_PAIRS,
    "voice": VOICE_PAIRS,
    "topic": TOPIC_PAIRS,
}
ALL_PAIRS = SVA_PAIRS + TENSE_PAIRS + VOICE_PAIRS + TOPIC_PAIRS  # 50对

# ===== 工具函数 =====

def get_hidden_states_last_tok(model, tokenizer, device, text, n_layers):
    """获取所有层hidden states（最后token，返回numpy list）"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_hidden_states=True)

    h_states = []
    for l in range(n_layers + 1):
        h = out.hidden_states[l][0, -1].detach().float().cpu().numpy()
        h_states.append(h)

    del out
    return h_states


def principal_angles_deg(A, B):
    """
    计算两子空间的主角（度数，升序）
    A: [d, k], B: [d, p], 列向量不必正交化（函数内部处理）
    """
    try:
        from scipy.linalg import subspace_angles
        # subspace_angles期望full-rank列矩阵（若k>p，取min(k,p)）
        k = min(A.shape[1], B.shape[1])
        angles = subspace_angles(A[:, :k], B[:, :k])
        return np.degrees(np.sort(angles))
    except Exception:
        # 备用：SVD法
        Qa = np.linalg.qr(A)[0]
        Qb = np.linalg.qr(B)[0]
        M = Qa.T @ Qb
        sv = np.linalg.svd(M, compute_uv=False)
        sv = np.clip(sv, -1.0, 1.0)
        return np.degrees(np.sort(np.arccos(sv)))


def random_subspace_angle_baseline(d, k, n_trials=15):
    """随机子空间的期望主角（高维时≈88-89°）"""
    angles_all = []
    for _ in range(n_trials):
        A = np.linalg.qr(np.random.randn(d, k))[0]
        B = np.linalg.qr(np.random.randn(d, k))[0]
        angles = principal_angles_deg(A, B)
        angles_all.append(float(np.mean(angles)))
    return float(np.mean(angles_all)), float(np.std(angles_all))


def build_transition_matrix(dh_in, dh_out):
    """
    最小二乘估计状态转移矩阵 T ≈ J_l
    T = Y X^+  (where X=[dh_in.T], Y=[dh_out.T])

    Returns: (T [d,d], eff_rank float, sv_top [ndarray])
    """
    X = dh_in.T    # [d, n]
    Y = dh_out.T   # [d, n]

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    thresh = S[0] * max(len(S), X.shape[0]) * np.finfo(float).eps * 10
    k_use = int(np.sum(S > max(thresh, 1e-8)))
    k_use = max(k_use, 1)

    T = Y @ Vt[:k_use].T @ np.diag(1.0 / S[:k_use]) @ U[:, :k_use].T

    # 有效秩（谱熵）
    sv_T = np.linalg.svd(T, compute_uv=False)
    sv_norm = sv_T / (np.sum(sv_T) + 1e-12)
    sv_norm = sv_norm[sv_norm > 1e-15]
    entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-30))
    eff_rank = float(np.exp(entropy))

    return T, eff_rank, sv_T


def get_top_right_sv(T, k):
    """T的top-k右奇异向量（输入空间主方向），返回[d, k]"""
    _, _, Vt = np.linalg.svd(T, full_matrices=False)
    k = min(k, Vt.shape[0])
    return Vt[:k].T  # [d, k]


# ===== Exp1: Bootstrap特征向量子空间稳定性 =====
def experiment_ev_universality(model, tokenizer, device, model_info,
                                all_dh_cache,
                                n_bootstrap=25, bootstrap_size=18, k_sv=20):
    """
    核心实验: 跨Bootstrap样本的Jacobian特征向量子空间稳定性

    使用预收集的all_dh_cache（避免重复前向传播）。
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    log_status(f"[EVUniversality] n_layers={n_layers}, d_model={d_model}, "
               f"n_bootstrap={n_bootstrap}, bootstrap_size={bootstrap_size}, k_sv={k_sv}")

    sample_layers = sorted(set([
        max(0, int(n_layers * f)) for f in [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
    ]))
    log_status(f"[EVUniversality] Key layers: {sample_layers}")

    results = {"per_layer": [], "overall": {}}

    for l in sample_layers:
        if l >= n_layers:
            continue

        deltas_l  = np.array(all_dh_cache[l])      # [n, d]
        deltas_l1 = np.array(all_dh_cache[l + 1])  # [n, d]
        n_pairs = len(deltas_l)

        if n_pairs < bootstrap_size + 3:
            log_status(f"  L{l}: insufficient pairs ({n_pairs}), skipping")
            continue

        rand_mean, rand_std = random_subspace_angle_baseline(d_model, k_sv, n_trials=10)

        ev_sets = []
        sv_shapes = []

        for b in range(n_bootstrap):
            idx = np.random.choice(n_pairs, size=min(bootstrap_size, n_pairs), replace=False)
            dl  = deltas_l[idx]
            dl1 = deltas_l1[idx]
            try:
                T, eff_rank, sv_T = build_transition_matrix(dl, dl1)
                V = get_top_right_sv(T, k_sv)
                ev_sets.append(V)
                sv_shapes.append(sv_T[:10].tolist() if len(sv_T) >= 10 else sv_T.tolist())
            except Exception as e:
                pass

        if len(ev_sets) < 5:
            log_status(f"  L{l}: too few successful bootstraps ({len(ev_sets)})")
            continue

        # Pairwise principal angles（最多取前20组避免O(n^2)爆炸）
        n_sets = min(len(ev_sets), 20)
        angles_all = []
        top5_angles = []

        for i in range(n_sets):
            for j in range(i + 1, n_sets):
                try:
                    angles = principal_angles_deg(ev_sets[i], ev_sets[j])
                    angles_all.append(float(np.mean(angles)))
                    top5_angles.append(float(np.mean(angles[:5])) if len(angles) >= 5 else float(np.mean(angles)))
                except Exception:
                    pass

        mean_angle     = float(np.mean(angles_all))  if angles_all  else 90.0
        top5_mean_angle = float(np.mean(top5_angles)) if top5_angles else 90.0
        normalized     = mean_angle / max(rand_mean, 1e-6)

        verdict = "STABLE" if mean_angle < 30 else ("MODERATE" if mean_angle < 55 else "UNSTABLE")

        result = {
            "layer": l,
            "mean_angle_deg": mean_angle,
            "top5_mean_angle_deg": top5_mean_angle,
            "random_baseline_deg": rand_mean,
            "normalized_angle": normalized,
            "verdict": verdict,
            "n_bootstrap_sets": len(ev_sets),
            "n_pairs": n_pairs,
            "sv_top10_mean": [float(x) for x in np.mean(np.array([s[:10] for s in sv_shapes if len(s) >= 10]), axis=0)] if sv_shapes else [],
        }
        results["per_layer"].append(result)

        log_status(f"  L{l}: mean={mean_angle:.1f}° (top5={top5_mean_angle:.1f}°), "
                   f"rand={rand_mean:.1f}°, norm={normalized:.2f}, n_pairs={n_pairs}, {verdict}")

    if results["per_layer"]:
        angles_all = [r["mean_angle_deg"] for r in results["per_layer"]]
        norms_all  = [r["normalized_angle"] for r in results["per_layer"]]
        verdicts   = [r["verdict"] for r in results["per_layer"]]
        n_stable   = verdicts.count("STABLE")
        n_moderate = verdicts.count("MODERATE")
        n_unstable = verdicts.count("UNSTABLE")
        overall_v  = ("STABLE" if n_stable > n_moderate + n_unstable
                      else "MODERATE" if n_stable + n_moderate >= n_unstable
                      else "UNSTABLE")
        results["overall"] = {
            "mean_angle_all_layers": float(np.mean(angles_all)),
            "mean_normalized_angle": float(np.mean(norms_all)),
            "n_stable": n_stable,
            "n_moderate": n_moderate,
            "n_unstable": n_unstable,
            "verdict": overall_v,
        }
        log_status(f"[EVUniversality] OVERALL: mean={np.mean(angles_all):.1f}°, "
                   f"norm={np.mean(norms_all):.2f}, "
                   f"STABLE={n_stable} MODERATE={n_moderate} UNSTABLE={n_unstable}, {overall_v}")

    return results


# ===== Exp2: 约束类型条件Jacobian子空间对齐 =====
def experiment_type_jacobians(model, tokenizer, device, model_info,
                               type_dh_cache, k_sv=15):
    """
    分别为每种约束类型计算T_l，测量跨类型主角。
    若跨类型角度 ≈ Bootstrap内角度 → 真正通用（与约束类型无关）
    若跨类型角度 >> Bootstrap内角度 → 约束类型特异
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    log_status(f"[TypeJacobian] k_sv={k_sv}")

    sample_layers = sorted(set([
        max(0, int(n_layers * f)) for f in [0.15, 0.35, 0.55, 0.75, 0.9]
    ]))

    results = {"per_layer": [], "overall": {}}
    ctypes = list(type_dh_cache.keys())

    for l in sample_layers:
        if l >= n_layers:
            continue

        type_sv = {}
        for ctype in ctypes:
            dh_l  = np.array(type_dh_cache[ctype][l])
            dh_l1 = np.array(type_dh_cache[ctype][l + 1])
            if len(dh_l) < 5:
                continue
            try:
                T, _, _ = build_transition_matrix(dh_l, dh_l1)
                V = get_top_right_sv(T, k_sv)
                type_sv[ctype] = V
            except Exception:
                pass

        if len(type_sv) < 2:
            continue

        rand_mean, _ = random_subspace_angle_baseline(d_model, k_sv, n_trials=5)
        type_names = list(type_sv.keys())
        cross_angles = {}

        for i in range(len(type_names)):
            for j in range(i + 1, len(type_names)):
                t1, t2 = type_names[i], type_names[j]
                try:
                    angles = principal_angles_deg(type_sv[t1], type_sv[t2])
                    cross_angles[f"{t1}_vs_{t2}"] = {
                        "mean_angle": float(np.mean(angles)),
                        "top5_angle": float(np.mean(angles[:5])) if len(angles) >= 5 else float(np.mean(angles)),
                    }
                except Exception:
                    pass

        mean_cross = float(np.mean([v["mean_angle"] for v in cross_angles.values()])) if cross_angles else 90.0

        result = {
            "layer": l,
            "cross_type_angles": cross_angles,
            "mean_cross_angle": mean_cross,
            "random_baseline_deg": rand_mean,
        }
        results["per_layer"].append(result)

        pairs_str = ", ".join(f"{k}={v['mean_angle']:.1f}°" for k, v in cross_angles.items())
        log_status(f"  L{l}: {pairs_str} | rand={rand_mean:.1f}°")

    if results["per_layer"]:
        cross_means = [r["mean_cross_angle"] for r in results["per_layer"]]
        rand_means  = [r["random_baseline_deg"] for r in results["per_layer"]]
        results["overall"] = {
            "mean_cross_angle_all_layers": float(np.mean(cross_means)),
            "mean_random_baseline": float(np.mean(rand_means)),
            "mean_normalized": float(np.mean(cross_means) / max(np.mean(rand_means), 1e-6)),
        }
        log_status(f"[TypeJacobian] OVERALL: mean_cross={np.mean(cross_means):.1f}°, "
                   f"rand={np.mean(rand_means):.1f}°, norm={results['overall']['mean_normalized']:.2f}")

    return results


# ===== Exp3: S_J vs S_Δh vs S_WU 三子空间对齐 =====
def experiment_three_subspace_alignment(model, tokenizer, device, model_info,
                                         all_dh_cache, model_name, k_sv=30):
    """
    三子空间对齐：
    - S_J ∩ S_Δh: 约束方向是否在Jacobian主方向上？
    - S_J ∩ S_WU: Jacobian主方向是否可被W_U观测？
    若 J∩Δh << J∩WU → 约束在J主方向但W_U看不见 → 暗物质假说
    若 J∩WU << random → Jacobian作用于W_U盲区 → 强暗物质
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    log_status(f"[3SubAlign] k_sv={k_sv}, n_layers={n_layers}")

    sample_layers = sorted(set([
        max(0, int(n_layers * f)) for f in [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    ]))

    # 获取W_U子空间（用scipy svds避免vocab×d_model的全SVD内存爆炸）
    S_WU = None
    log_status("[3SubAlign] Computing W_U subspace (svds)...")
    try:
        from scipy.sparse.linalg import svds
        W_U = get_W_U(model, model_name)  # [vocab, d]
        W_U_f = W_U.astype(np.float32)
        k_wu = min(k_sv + 10, min(W_U_f.shape) - 2)
        _, _, Vt_wu = svds(W_U_f, k=k_wu)
        # svds返回最小奇异值在前，翻转得到最大在前
        S_WU = Vt_wu[::-1].T[:, :k_sv]  # [d, k_sv]
        log_status(f"[3SubAlign] W_U shape={W_U.shape}, S_WU={S_WU.shape}")
        del W_U, W_U_f
        gc.collect()
    except Exception as e:
        log_status(f"[3SubAlign] W_U error: {e}")
        S_WU = None

    rand_mean, _ = random_subspace_angle_baseline(d_model, k_sv, n_trials=5)
    results = {"per_layer": [], "overall": {}}

    for l in sample_layers:
        if l >= n_layers:
            continue

        dh_l  = np.array(all_dh_cache[l])
        dh_l1 = np.array(all_dh_cache[l + 1])

        if len(dh_l) < 10:
            continue

        try:
            # S_J: Jacobian右奇异向量
            T, eff_rank, _ = build_transition_matrix(dh_l, dh_l1)
            S_J = get_top_right_sv(T, k_sv)  # [d, k_sv]

            # S_Δh: 约束方向PCA（主成分在d_model空间中，是dh_centered.T的左奇异向量U）
            dh_centered = dh_l - dh_l.mean(axis=0)  # [n, d]
            U_dh, _, _ = np.linalg.svd(dh_centered.T, full_matrices=False)  # [d, min(d,n)]
            k_dh = min(k_sv, U_dh.shape[1])
            S_Dh = U_dh[:, :k_dh]  # [d, k_dh] ✓

            angles_J_Dh  = principal_angles_deg(S_J, S_Dh)
            mean_J_Dh    = float(np.mean(angles_J_Dh))
            top5_J_Dh    = float(np.mean(angles_J_Dh[:5])) if len(angles_J_Dh) >= 5 else mean_J_Dh

            mean_J_WU    = None
            top5_J_WU    = None
            mean_Dh_WU   = None

            if S_WU is not None:
                k_min = min(S_J.shape[1], S_WU.shape[1])
                angles_J_WU  = principal_angles_deg(S_J[:, :k_min], S_WU[:, :k_min])
                mean_J_WU    = float(np.mean(angles_J_WU))
                top5_J_WU    = float(np.mean(angles_J_WU[:5])) if len(angles_J_WU) >= 5 else mean_J_WU

                k_min2 = min(S_Dh.shape[1], S_WU.shape[1])
                angles_Dh_WU = principal_angles_deg(S_Dh[:, :k_min2], S_WU[:, :k_min2])
                mean_Dh_WU   = float(np.mean(angles_Dh_WU))

            result = {
                "layer": l,
                "eff_rank_J": eff_rank,
                "mean_angle_J_Dh":  mean_J_Dh,
                "top5_angle_J_Dh":  top5_J_Dh,
                "mean_angle_J_WU":  mean_J_WU,
                "top5_angle_J_WU":  top5_J_WU,
                "mean_angle_Dh_WU": mean_Dh_WU,
                "random_baseline_deg": rand_mean,
            }
            results["per_layer"].append(result)

            log_status(f"  L{l}: eff_rank={eff_rank:.0f}, "
                       f"J∩Δh={mean_J_Dh:.1f}°(top5={top5_J_Dh:.1f}°), "
                       f"J∩WU={mean_J_WU:.1f}°, Δh∩WU={mean_Dh_WU:.1f}°, "
                       f"rand={rand_mean:.1f}°")

        except Exception as e:
            log_status(f"  L{l} error: {e}")
            import traceback; traceback.print_exc()

    if results["per_layer"]:
        j_dh  = [r["mean_angle_J_Dh"] for r in results["per_layer"]]
        j_wu  = [r["mean_angle_J_WU"]  for r in results["per_layer"] if r["mean_angle_J_WU"] is not None]
        dh_wu = [r["mean_angle_Dh_WU"] for r in results["per_layer"] if r["mean_angle_Dh_WU"] is not None]
        results["overall"] = {
            "mean_J_Dh_all_layers":  float(np.mean(j_dh))  if j_dh  else None,
            "mean_J_WU_all_layers":  float(np.mean(j_wu))  if j_wu  else None,
            "mean_Dh_WU_all_layers": float(np.mean(dh_wu)) if dh_wu else None,
            "random_baseline_deg":   rand_mean,
            "dark_matter_supported": bool(np.mean(j_wu) > rand_mean * 0.85) if j_wu else None,
        }
        log_status(f"[3SubAlign] OVERALL: J∩Δh={np.mean(j_dh):.1f}°, "
                   f"J∩WU={np.mean(j_wu):.1f}° (rand={rand_mean:.1f}°)")

    return results


# ===== 共享数据收集 =====
def collect_all_hidden_states(model, tokenizer, device, n_layers):
    """
    一次性收集所有50对句子的Δh（各层），供三个实验共用。
    Returns:
        all_dh_cache: {layer: [Δh array, ...]}  (50对)
        type_dh_cache: {type: {layer: [Δh array, ...]}}  (按类型)
    """
    log_status(f"[DataCollect] Collecting {len(ALL_PAIRS)} pairs, all layers...")

    all_dh_cache  = {l: [] for l in range(n_layers + 1)}
    type_dh_cache = {ctype: {l: [] for l in range(n_layers + 1)}
                     for ctype in ALL_PAIRS_BY_TYPE}

    for pair_idx, (s1, s2) in enumerate(ALL_PAIRS):
        # 确定对应类型
        ctype = "sva"
        offset = 0
        for ct, pairs in ALL_PAIRS_BY_TYPE.items():
            if pair_idx < offset + len(pairs):
                ctype = ct
                break
            offset += len(pairs)

        maybe_log(f"[DataCollect] pair {pair_idx+1}/{len(ALL_PAIRS)}")
        try:
            h1 = get_hidden_states_last_tok(model, tokenizer, device, s1, n_layers)
            h2 = get_hidden_states_last_tok(model, tokenizer, device, s2, n_layers)

            for l in range(n_layers + 1):
                delta = h1[l] - h2[l]
                all_dh_cache[l].append(delta)
                type_dh_cache[ctype][l].append(delta)

        except Exception as e:
            log_status(f"  Error pair {pair_idx} ({s1[:20]}...): {e}")

    n_valid = len(all_dh_cache[0])

    log_status(f"[DataCollect] Collected {n_valid} valid pairs (layer 0)")
    return all_dh_cache, type_dh_cache


# ===== 主函数 =====
def run_model(model_name: str):
    log_status("=" * 60)
    log_status(f"Phase 225: Jacobian特征向量通用性验证 - {model_name}")
    log_status("=" * 60)

    model, tokenizer, device, model_info = load_model_bf16(model_name)
    n_layers = model_info.n_layers

    results = {
        "model": model_name,
        "model_info": {
            "n_layers": n_layers,
            "d_model": model_info.d_model,
            "model_class": model_info.model_class,
            "mlp_type": model_info.mlp_type,
        }
    }

    # 共享数据收集（一次前向传播，供三实验复用）
    log_status(f"\n{'='*40}")
    log_status("Data Collection (shared across all experiments)")
    log_status(f"{'='*40}")
    all_dh_cache, type_dh_cache = collect_all_hidden_states(
        model, tokenizer, device, n_layers
    )

    # Exp 1
    log_status(f"\n{'='*40}")
    log_status("Exp 1: Jacobian特征向量子空间稳定性 (P0)")
    log_status(f"{'='*40}")
    try:
        results["exp1_ev_universality"] = experiment_ev_universality(
            model, tokenizer, device, model_info,
            all_dh_cache,
            n_bootstrap=25, bootstrap_size=18, k_sv=20,
        )
    except Exception as e:
        log_status(f"Exp1 error: {e}")
        import traceback; traceback.print_exc()
        results["exp1_ev_universality"] = {"error": str(e)}

    # Exp 2
    log_status(f"\n{'='*40}")
    log_status("Exp 2: 约束类型条件Jacobian对齐 (P1)")
    log_status(f"{'='*40}")
    try:
        results["exp2_type_jacobians"] = experiment_type_jacobians(
            model, tokenizer, device, model_info,
            type_dh_cache, k_sv=15,
        )
    except Exception as e:
        log_status(f"Exp2 error: {e}")
        import traceback; traceback.print_exc()
        results["exp2_type_jacobians"] = {"error": str(e)}

    # Exp 3
    log_status(f"\n{'='*40}")
    log_status("Exp 3: S_J vs S_Δh vs S_WU 三子空间对齐 (P2)")
    log_status(f"{'='*40}")
    try:
        results["exp3_three_subspace"] = experiment_three_subspace_alignment(
            model, tokenizer, device, model_info,
            all_dh_cache, model_name, k_sv=30,
        )
    except Exception as e:
        log_status(f"Exp3 error: {e}")
        import traceback; traceback.print_exc()
        results["exp3_three_subspace"] = {"error": str(e)}

    # 保存结果
    out_path = OUTPUT_DIR / f"phase225_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    log_status(f"Results saved to {out_path}")

    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    log_status("Phase 225 complete!")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                run_model(name)
            except Exception as e:
                log_status(f"!!! {name} failed: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
    else:
        run_model(model_name)


if __name__ == "__main__":
    main()
