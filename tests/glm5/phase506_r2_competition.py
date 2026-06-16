"""
Phase 506 R2: 竞争轨迹 + 读出转移 + vc连接
=============================================
Exp5: 逐层竞争轨迹 (D, T, C per layer, logit lens)
Exp1: 全层读出相关标量预测 (ridge regression R²_D/T/C)
Exp3: W·R_pre → v_c 连接 (cosine alignment)

跨模型: qwen3, glm4, deepseek7b
所有模型: bf16 + device_map="auto" + sdpa (flash)

Usage:
  python tests/glm5/phase506_r2_competition.py qwen3
  python tests/glm5/phase506_r2_competition.py glm4
  python tests/glm5/phase506_r2_competition.py deepseek7b
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc, json, time, os
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from model_utils import (get_model_info, release_model, get_W_U, MODEL_CONFIGS)

OUTPUT_DIR = Path("results/glm5")
TEMP_DIR = Path("tests/glm5_temp")

# === 数据配置 ===
TRAIN_N = 12; TEST_N = 8  # R1的设置; R2保持一致以便对比

CATEGORIES = {
    "fruit": {
        "objects": ["apple","banana","orange","grape","pear","peach","mango","plum",
                    "cherry","lemon","apricot","kiwi","pineapple","melon","coconut","lime",
                    "fig","pomegranate","papaya","avocado"],
        "relation": "is a type of fruit"
    },
    "animal": {
        "objects": ["dog","cat","horse","elephant","tiger","dolphin","eagle","snake",
                    "rabbit","whale","lion","bear","fox","wolf","deer","monkey",
                    "shark","frog","penguin","owl"],
        "relation": "is a type of animal"
    },
    "action": {
        "objects": ["run","eat","build","throw","buy","learn","measure","communicate",
                    "swim","write","sing","draw","fly","climb","teach","drive",
                    "cook","dance","fight","sleep"],
        "relation": "is a type of action"
    },
    "emotion": {
        "objects": ["joy","anger","fear","sadness","surprise","disgust","pride","shame",
                    "guilt","envy","hope","love","hate","boredom","anxiety","jealousy",
                    "gratitude","regret","curiosity","embarrassment"],
        "relation": "is a type of emotion"
    },
    "clothing": {
        "objects": ["shirt","dress","jacket","pants","coat","skirt","sweater","blouse",
                    "scarf","vest","hat","glove","sock","boot","belt","tie",
                    "jeans","shorts","hoodie","raincoat"],
        "relation": "is a type of clothing"
    },
}
ALL_CLASS = list(CATEGORIES.keys())
NEUTRAL_RELATION = "is a thing"


def load_bf16_auto(name):
    """BF16 + device_map=auto + sdpa(flash) 加载"""
    cfg = MODEL_CONFIGS[name]
    print(f"[load] Loading {name} (bf16 + auto + sdpa)...")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",  # flash attention via SDPA
    )
    m.eval()

    dev = next(m.parameters()).device
    gmem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    gn = cn = 0
    if hasattr(m, 'hf_device_map'):
        dm = m.hf_device_map
        gn = sum(1 for v in dm.values() if 'cuda' in str(v))
        cn = sum(1 for v in dm.values() if 'cpu' in str(v))

    print(f"[load] {name}: GPU={gn} CPU={cn} mem={gmem:.1f}GB class={type(m).__name__} ({time.time()-t0:.0f}s)")
    return m, tok, dev


def get_norm_g(model, name):
    """获取最终RMSNorm/LayerNorm的gain权重"""
    for attr in ['model.norm', 'model.final_layernorm', 'model.decoder.final_layer_norm']:
        obj = model
        for p in attr.split('.'):
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj and hasattr(obj, 'weight'):
            w = obj.weight.detach()
            if str(w.device) != 'meta':
                return w.float().cpu().numpy()
    # Fallback: 从safetensors加载
    cfg = MODEL_CONFIGS[name]
    for sf in sorted(Path(cfg["path"]).glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                if 'norm' in key.lower() and 'weight' in key.lower() and not any(x in key for x in ['layer', 'input', 'post']):
                    return f.get_tensor(key).float().cpu().numpy()
    return None


def get_token_ids(tokenizer, words):
    """获取词列表的token ID"""
    ids = []
    for w in words:
        tid = tokenizer.encode(w, add_special_tokens=False)
        if tid:
            ids.append(tid[0])
    return ids


def forward_with_all_hidden(model, tokenizer, prompt, device):
    """
    前向推理，返回所有层的隐藏状态
    返回: dict {layer_idx: {"ans": ndarray, "pre": ndarray}}
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hs = outputs.hidden_states  # tuple of (1, seq_len, d_model)
    n_layers = len(hs)  # 包括embedding层

    result = {}
    for l in range(n_layers):
        seq_len = hs[l].shape[1]
        ans = hs[l][0, -1, :].float().cpu().numpy().astype(np.float64)
        pre = hs[l][0, -2, :].float().cpu().numpy().astype(np.float64) if seq_len >= 2 else ans.copy()
        result[l] = {"ans": ans, "pre": pre}

    return result


def ridge_map(X, Y, ridge=0.1):
    """Ridge回归: W = (X^T X + λI)^{-1} X^T Y"""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    XtX = X.T @ X
    return np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), X.T @ Y).astype(np.float64)


def r2_score(y_true, y_pred):
    """R² score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def rms_norm(vec):
    """RMS norm"""
    return float(np.sqrt(np.mean(vec ** 2)))


def run_phase506_r2(model, tokenizer, model_name, device):
    """Phase 506 R2 主函数"""
    info = get_model_info(model, model_name)
    L = info.n_layers  # transformer层数
    d = info.d_model
    print(f"[info] L={L}, d={d}, class={info.model_class}")

    # 加载W_U和gain
    W_U = get_W_U(model, model_name).astype(np.float64)  # [vocab, d_model]
    g = get_norm_g(model, model_name)
    if g is None:
        print("[ERROR] Cannot get final layer norm gain!")
        return None
    g = g.astype(np.float64)
    print(f"[info] W_U shape={W_U.shape}, g norm={np.linalg.norm(g):.2f}")

    # 构建每个类别的q_c方向和token ID
    cat_meta = {}
    for cat_name, cfg in CATEGORIES.items():
        target_ids = get_token_ids(tokenizer, [cat_name])
        other_cats = [c for c in ALL_CLASS if c != cat_name]
        competitor_ids = get_token_ids(tokenizer, other_cats)

        # q_c = g ⊙ (w_target - w_competitor)
        wDt = np.mean([W_U[i] for i in target_ids if i < len(W_U)], axis=0) if target_ids else np.zeros(d)
        wDc = np.mean([W_U[i] for i in competitor_ids if i < len(W_U)], axis=0) if competitor_ids else np.zeros(d)
        qc = (wDt - wDc) * g

        cat_meta[cat_name] = {
            "target_ids": target_ids,
            "competitor_ids": competitor_ids,
            "qc": qc,
            "qcn": float(np.linalg.norm(qc)),
        }
        print(f"  {cat_name}: target_ids={target_ids}, |qc|={np.linalg.norm(qc):.2f}")

    # === Exp5: 逐层竞争轨迹 ===
    print(f"\n{'='*60}")
    print("Exp5: 逐层竞争轨迹")
    print(f"{'='*60}")

    trajectory = {}  # {cat: {layer: {metric: value}}}

    for cat_name, cfg in CATEGORIES.items():
        print(f"\n--- {cat_name} ---")
        objs = cfg["objects"]
        target_ids = cat_meta[cat_name]["target_ids"]
        competitor_ids = cat_meta[cat_name]["competitor_ids"]
        qc = cat_meta[cat_name]["qc"]
        qcn = cat_meta[cat_name]["qcn"]

        # 收集所有对象的隐藏状态
        rich_h_all = []  # [obj_idx, layer, ans/pre, d_model]
        neutral_h_all = []

        for oi, obj in enumerate(objs[:TRAIN_N + TEST_N]):
            rich_prompt = f"The {obj} {cfg['relation']}"
            neutral_prompt = f"The {obj} {NEUTRAL_RELATION}"

            rich_states = forward_with_all_hidden(model, tokenizer, rich_prompt, device)
            neutral_states = forward_with_all_hidden(model, tokenizer, neutral_prompt, device)

            rich_h_all.append(rich_states)
            neutral_h_all.append(neutral_states)

            if (oi + 1) % 4 == 0:
                print(f"  [{oi+1}/{TRAIN_N+TEST_N}] {obj} done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

        # 逐层计算竞争轨迹
        cat_traj = {}
        for l in range(L + 1):  # 包括embedding层
            D_rich_list, T_rich_list, C_rich_list = [], [], []
            D_neutral_list, T_neutral_list, C_neutral_list = [], [], []
            vc_list = []
            proj_qc_rich_list, proj_qc_neutral_list = [], []

            for oi in range(len(rich_h_all)):
                h_rich = rich_h_all[oi][l]["ans"]  # [d_model]
                h_neutral = neutral_h_all[oi][l]["ans"]

                # Logit lens: 直接用W_U计算logits
                logits_rich = h_rich @ W_U.T
                logits_neutral = h_neutral @ W_U.T

                # Target和competitor logits
                t_rich = np.mean([logits_rich[i] for i in target_ids if i < len(logits_rich)])
                c_rich = np.mean([logits_rich[i] for i in competitor_ids if i < len(logits_rich)])
                t_neutral = np.mean([logits_neutral[i] for i in target_ids if i < len(logits_neutral)])
                c_neutral = np.mean([logits_neutral[i] for i in competitor_ids if i < len(logits_neutral)])

                D_rich_list.append(t_rich - c_rich)
                T_rich_list.append(t_rich)
                C_rich_list.append(c_rich)
                D_neutral_list.append(t_neutral - c_neutral)
                T_neutral_list.append(t_neutral)
                C_neutral_list.append(c_neutral)

                # v_c = h_rich - h_neutral
                vc = h_rich - h_neutral
                vc_list.append(vc)

                # 投影到qc方向 (DCF-like)
                h_rms = rms_norm(h_rich)
                h_neutral_rms = rms_norm(h_neutral)
                proj_qc_rich_list.append(float(np.dot(h_rich, qc) / (h_rms + 1e-10)))
                proj_qc_neutral_list.append(float(np.dot(h_neutral, qc) / (h_neutral_rms + 1e-10)))

            # 平均vc
            vc_mean = np.mean(vc_list, axis=0)
            vc_norm = np.linalg.norm(vc_mean)
            cos_vc_qc = float(np.dot(vc_mean, qc) / (vc_norm * qcn + 1e-10)) if vc_norm > 1e-10 else 0

            cat_traj[l] = {
                "D_rich": float(np.mean(D_rich_list)),
                "T_rich": float(np.mean(T_rich_list)),
                "C_rich": float(np.mean(C_rich_list)),
                "D_neutral": float(np.mean(D_neutral_list)),
                "T_neutral": float(np.mean(T_neutral_list)),
                "C_neutral": float(np.mean(C_neutral_list)),
                "proj_qc_rich": float(np.mean(proj_qc_rich_list)),
                "proj_qc_neutral": float(np.mean(proj_qc_neutral_list)),
                "cos_vc_qc": cos_vc_qc,
                "vc_norm": float(vc_norm),
            }

        trajectory[cat_name] = cat_traj

        # 打印关键层摘要
        key_layers = [0, L // 4, L // 2, 3 * L // 4, L - 3, L - 1]
        key_layers = sorted(set(min(l, L) for l in key_layers))
        print(f"  Competition trajectory ({cat_name}):")
        for l in key_layers:
            t = cat_traj[l]
            print(f"    L{l:>3}: D_rich={t['D_rich']:+7.2f} D_neutral={t['D_neutral']:+7.2f} "
                  f"ΔD={t['D_rich']-t['D_neutral']:+7.2f} cos(vc,qc)={t['cos_vc_qc']:+.4f} "
                  f"|vc|={t['vc_norm']:.1f}")

    # === Exp1: Ridge regression for readout-relevant scalars ===
    print(f"\n{'='*60}")
    print("Exp1: Ridge regression (全层扫描)")
    print(f"{'='*60}")

    # 每6层采样 + 首尾关键层
    ridge_layers = sorted(set(
        [0, L // 4, L // 3, L // 2, 2 * L // 3, 3 * L // 4, L - 5, L - 3, L - 1]
        + list(range(0, L + 1, max(L // 6, 1)))
    ))
    ridge_layers = [min(l, L) for l in ridge_layers]
    # 去重并排序
    ridge_layers = sorted(set(ridge_layers))
    print(f"  Ridge scan layers: {ridge_layers}")

    ridge_results = {}

    for cat_name, cfg in CATEGORIES.items():
        print(f"\n--- {cat_name} ---")
        objs = cfg["objects"]
        target_ids = cat_meta[cat_name]["target_ids"]
        competitor_ids = cat_meta[cat_name]["competitor_ids"]
        qc = cat_meta[cat_name]["qc"]

        train_objs = objs[:TRAIN_N]
        test_objs = objs[TRAIN_N:TRAIN_N + TEST_N]

        # 收集训练和测试数据 (一次性前向推理)
        train_rich = {l: {"pre": [], "ans": []} for l in ridge_layers}
        train_neutral = {l: {"ans": []} for l in ridge_layers}
        test_rich = {l: {"pre": [], "ans": []} for l in ridge_layers}
        test_neutral = {l: {"ans": []} for l in ridge_layers}

        for obj in train_objs:
            rich_states = forward_with_all_hidden(model, tokenizer,
                                                  f"The {obj} {cfg['relation']}", device)
            neutral_states = forward_with_all_hidden(model, tokenizer,
                                                     f"The {obj} {NEUTRAL_RELATION}", device)
            for l in ridge_layers:
                if l < len(rich_states):
                    train_rich[l]["pre"].append(rich_states[l]["pre"])
                    train_rich[l]["ans"].append(rich_states[l]["ans"])
                    train_neutral[l]["ans"].append(neutral_states[l]["ans"])

        for obj in test_objs:
            rich_states = forward_with_all_hidden(model, tokenizer,
                                                  f"The {obj} {cfg['relation']}", device)
            neutral_states = forward_with_all_hidden(model, tokenizer,
                                                     f"The {obj} {NEUTRAL_RELATION}", device)
            for l in ridge_layers:
                if l < len(rich_states):
                    test_rich[l]["pre"].append(rich_states[l]["pre"])
                    test_rich[l]["ans"].append(rich_states[l]["ans"])
                    test_neutral[l]["ans"].append(neutral_states[l]["ans"])

        # 逐层拟合ridge
        cat_ridge = {}
        for l in ridge_layers:
            if l >= len(rich_states) or len(train_rich[l]["pre"]) < 2:
                continue

            X_train = np.array(train_rich[l]["pre"])
            Y_train = np.array(train_rich[l]["ans"])
            W = ridge_map(X_train, Y_train, ridge=0.1)

            # 训练集评估
            Y_train_pred = np.array([W @ x for x in X_train])

            # 测试集
            X_test = np.array(test_rich[l]["pre"])
            Y_test = np.array(test_rich[l]["ans"])
            WR_test = np.array([W @ x for x in X_test])

            # 真实D, T, C (从answer hidden state)
            D_true, T_true, C_true = [], [], []
            for y in Y_test:
                logits = y @ W_U.T
                t_val = np.mean([logits[i] for i in target_ids if i < len(logits)])
                c_val = np.mean([logits[i] for i in competitor_ids if i < len(logits)])
                D_true.append(t_val - c_val)
                T_true.append(t_val)
                C_true.append(c_val)

            # 预测D, T, C (从WR)
            D_pred, T_pred, C_pred = [], [], []
            for wr in WR_test:
                logits = wr @ W_U.T
                t_val = np.mean([logits[i] for i in target_ids if i < len(logits)])
                c_val = np.mean([logits[i] for i in competitor_ids if i < len(logits)])
                D_pred.append(t_val - c_val)
                T_pred.append(t_val)
                C_pred.append(c_val)

            # R² scores
            r2_vec = r2_score(Y_test.flatten(), WR_test.flatten())
            r2_D = r2_score(np.array(D_true), np.array(D_pred))
            r2_T = r2_score(np.array(T_true), np.array(T_pred))
            r2_C = r2_score(np.array(C_true), np.array(C_pred))

            # Exp3: WR → vc 连接
            # vc = h_rich - h_neutral for test objects
            vc_test = np.array(test_rich[l]["ans"]) - np.array(test_neutral[l]["ans"])
            vc_mean = np.mean(vc_test, axis=0)
            WR_mean = np.mean(WR_test, axis=0)

            vc_norm = np.linalg.norm(vc_mean)
            wr_norm = np.linalg.norm(WR_mean)
            qcn = np.linalg.norm(qc)

            cos_WR_qc = float(np.dot(WR_mean, qc) / (wr_norm * qcn + 1e-10)) if wr_norm > 1e-10 else 0
            cos_vc_qc = float(np.dot(vc_mean, qc) / (vc_norm * qcn + 1e-10)) if vc_norm > 1e-10 else 0
            cos_WR_vc = float(np.dot(WR_mean, vc_mean) / (wr_norm * vc_norm + 1e-10)) if wr_norm > 1e-10 and vc_norm > 1e-10 else 0

            # Proj_{qc}(WR) vs Proj_{qc}(vc)
            proj_qc_WR = np.dot(WR_mean, qc) / (qcn + 1e-10)
            proj_qc_vc = np.dot(vc_mean, qc) / (qcn + 1e-10)

            # <WR, qc> 标量预测
            D_pred_qc = np.array([np.dot(wr, qc) for wr in WR_test])
            D_true_qc = np.array([np.dot(y, qc) for y in Y_test])
            r2_D_qc = r2_score(D_true_qc, D_pred_qc)

            cat_ridge[l] = {
                "r2_vector": round(r2_vec, 4),
                "r2_D": round(r2_D, 4),
                "r2_T": round(r2_T, 4),
                "r2_C": round(r2_C, 4),
                "r2_D_qc": round(r2_D_qc, 4),  # 预测 <ans, qc>
                "cos_WR_qc": round(cos_WR_qc, 6),
                "cos_vc_qc": round(cos_vc_qc, 6),
                "cos_WR_vc": round(cos_WR_vc, 6),
                "proj_qc_WR": round(float(proj_qc_WR), 4),
                "proj_qc_vc": round(float(proj_qc_vc), 4),
                "mean_D_true": round(float(np.mean(D_true)), 2),
                "mean_D_pred": round(float(np.mean(D_pred)), 2),
            }

            sig = "✅" if r2_D_qc > 0 else "❌"
            print(f"  {sig} L{l:>3}: R²_vec={r2_vec:+.3f} R²_D={r2_D:+.3f} R²_D_qc={r2_D_qc:+.3f} "
                  f"cos(WR→qc)={cos_WR_qc:+.4f} cos(vc→qc)={cos_vc_qc:+.4f} "
                  f"cos(WR→vc)={cos_WR_vc:+.4f}")

        ridge_results[cat_name] = cat_ridge

    # === 汇总 ===
    summary = {
        "model": model_name,
        "L": L,
        "d_model": d,
        "model_class": info.model_class,
        "n_train": TRAIN_N,
        "n_test": TEST_N,
        "trajectory": trajectory,
        "ridge": ridge_results,
        "ridge_layers": ridge_layers,
    }

    return summary


def print_summary(results):
    """打印结果摘要"""
    model_name = results["model"]
    L = results["L"]

    print(f"\n{'='*70}")
    print(f"Phase 506 R2 Summary — {model_name}")
    print(f"{'='*70}")

    # --- Exp5: 竞争轨迹关键特征 ---
    print("\n[Exp5] Competition Trajectory:")
    print(f"  {'Cat':>10s} | {'L0 D_rich':>9s} | {'L_mid D':>9s} | {'L-1 D':>9s} | "
          f"{'L0 cos(vc,qc)':>13s} | {'L-1 cos(vc,qc)':>13s} | {'D_emerges':>9s}")
    for cat, traj in results["trajectory"].items():
        # 找D从负转正的层 (emergence layer)
        d_rich_vals = [traj[l]["D_rich"] for l in range(L + 1)]
        d_neutral_vals = [traj[l]["D_neutral"] for l in range(L + 1)]
        delta_d = [r - n for r, n in zip(d_rich_vals, d_neutral_vals)]

        emerge_l = "N/A"
        for l in range(1, L + 1):
            if delta_d[l] > 0 and delta_d[l - 1] <= 0:
                emerge_l = f"L{l}"
                break
        if emerge_l == "N/A" and delta_d[-1] > 0:
            emerge_l = f"L0+"

        mid = L // 2
        print(f"  {cat:>10s} | {traj[0]['D_rich']:+9.2f} | {traj[mid]['D_rich']:+9.2f} | "
              f"{traj[L]['D_rich']:+9.2f} | {traj[0]['cos_vc_qc']:+13.4f} | "
              f"{traj[L]['cos_vc_qc']:+13.4f} | {emerge_l:>9s}")

    # --- Exp1: Ridge最佳层 ---
    print("\n[Exp1] Ridge Regression Best Layer per Category:")
    print(f"  {'Cat':>10s} | {'Best L':>6s} | {'R²_vec':>7s} | {'R²_D':>7s} | {'R²_D_qc':>8s} | "
          f"{'cos(WR→qc)':>11s} | {'cos(vc→qc)':>11s}")
    for cat, lr in results["ridge"].items():
        if not lr:
            continue
        # Best by R²_D_qc
        best_l = max(lr.items(), key=lambda x: x[1]["r2_D_qc"])
        bl_key = str(best_l[0]) if not isinstance(best_l[0], str) else best_l[0]
        print(f"  {cat:>10s} | L{bl_key:>4s} | {best_l[1]['r2_vector']:+7.3f} | "
              f"{best_l[1]['r2_D']:+7.3f} | {best_l[1]['r2_D_qc']:+8.3f} | "
              f"{best_l[1]['cos_WR_qc']:+11.4f} | {best_l[1]['cos_vc_qc']:+11.4f}")

    # --- 全层R²_D_qc ---
    print("\n[Exp1] R²_D_qc across layers (avg over categories):")
    for l in results["ridge_layers"]:
        vals = [results["ridge"][c][l]["r2_D_qc"] for c in results["ridge"] if l in results["ridge"][c]]
        if vals:
            avg = np.mean(vals)
            n_pos = sum(1 for v in vals if v > 0)
            print(f"  L{l:>3}: avg R²_D_qc={avg:+.3f} (+cats={n_pos}/{len(vals)})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/glm5/phase506_r2_competition.py <model_name>")
        sys.exit(1)

    mn = sys.argv[1]
    if mn not in MODEL_CONFIGS:
        print(f"Unknown model: {mn}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    print("=" * 70)
    print(f"Phase 506 R2: Competition Trajectory + Readout Transport — {mn}")
    print(f"train={TRAIN_N} test={TEST_N} | bf16 + auto + sdpa")
    print("=" * 70)

    t0 = time.time()
    model, tokenizer, device = load_bf16_auto(mn)

    try:
        results = run_phase506_r2(model, tokenizer, mn, device)
        if results is None:
            print("ERROR: No results!")
            return

        print_summary(results)

        # 保存结果
        out = OUTPUT_DIR / f"phase506_r2_{mn}.json"

        # 将numpy类型转为Python原生类型
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=convert)
        print(f"\nSaved: {out}")

        elapsed = time.time() - t0
        print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    finally:
        release_model(model)
        print("Model released.")


if __name__ == "__main__":
    main()
