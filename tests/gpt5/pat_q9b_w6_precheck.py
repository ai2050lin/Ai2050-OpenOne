# -*- coding: utf-8 -*-
"""Q9-B 增补 W6-b 零成本预检：语域方向在 down_proj 输出列上的投影分布形状

目的（预注册增补 §6.1「零成本预检」）：不跑前向，仅用冻结 npz 的 v_reg 与真实权重，
判断「语域写入沿 MLP 输出列是集中的（WSP 型 LOCALIZED）还是均匀的（DISTRIBUTED）」。

可算量：
  c_j = u · W_out[:, j]                （W_out = down_proj.weight, shape (d_model, d_ff)）
  能量份额 S_k = Σ_{top-k} c_j² / Σ_j c_j²
  Gini(c²), 以及 |c_j| 与列范数的相关（排除"只是列范数大"的混淆）
零分布：同层随机单位方向 200 次，给出 S_k 的 null 分位数与 z。

用法: python tests/gpt5/pat_q9b_w6_precheck.py
输出: tests/gpt5/result/pat_q9b_w6_precheck.json
"""
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "tests" / "gpt5" / "result"
NPZ = RES / "pat_q9a_qwen.npz"
MODEL = Path("D:/develop/model/Qwen/Qwen3-4B")
MID = list(range(18, 28))          # 预注册中段带 Qwen L18–27
CTRL = [0, 35]                     # 对照层（首/末）
K_LIST = [8, 32, 64, 128, 512]
N_NULL = 200
SEED = 20261102

REG = ["colloq", "acad", "code"]
out_lines = []


def say(s):
    print(s, flush=True)
    out_lines.append(s)


def gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def load_tensor(layer):
    """只读单层 down_proj 权重，避免整模型加载"""
    idx_path = MODEL / "model.safetensors.index.json"
    wm = json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]
    key = f"model.layers.{layer}.mlp.down_proj.weight"
    shard = wm[key]
    with safe_open(str(MODEL / shard), framework="pt") as f:
        t = f.get_tensor(key)
    return t.float().numpy()      # (d_model, d_ff)


say(f"[cfg] model={MODEL.name} mid={MID} ctrl={CTRL} n_null={N_NULL} seed={SEED}")

z = np.load(NPZ)
R, A, M = z["R"].astype(np.float32), z["A"].astype(np.float32), z["M"].astype(np.float32)
y_reg, y_lang, is_disc = z["y_reg"], z["y_lang"], z["is_disc"]
N, L, d = R.shape
Rext = np.zeros((N, L + 1, d), np.float32)
Rext[:, :L] = R
Rext[:, L] = R[:, L - 1] + A[:, L - 1] + M[:, L - 1]
say(f"[field] N={N} L={L} d={d} disc={int(is_disc.sum())}")

rng = np.random.default_rng(SEED)
results = []

for l in MID + CTRL:
    W = load_tensor(l)                      # (d, d_ff)
    d_ff = W.shape[1]
    X = Rext[:, l + 1]                      # resid_out[l]（l=L 时用重构值；L=36 不在本表）
    if l >= L:
        X = Rext[:, L]
    sel = is_disc
    dirs = {}
    m_col = X[sel & (y_reg == 0)].mean(0) - X[sel & (y_reg == 1)].mean(0)
    dirs["reg_all"] = m_col / (np.linalg.norm(m_col) + 1e-9)
    for lang, lv in [("zh", 0), ("en", 1)]:
        s = sel & (y_reg == 0) & (y_lang == lv)
        a = sel & (y_reg == 1) & (y_lang == lv)
        if s.sum() >= 5 and a.sum() >= 5:
            mm = X[s].mean(0) - X[a].mean(0)
            dirs[f"reg_{lang}"] = mm / (np.linalg.norm(mm) + 1e-9)

    col_norm = np.linalg.norm(W, axis=0)
    U = rng.standard_normal((N_NULL, d)).astype(np.float32)
    U /= np.linalg.norm(U, axis=1, keepdims=True)

    for name, u in dirs.items():
        u = u.astype(np.float32)
        c = W.T @ u                          # (d_ff,)
        c2 = c ** 2
        tot = c2.sum()
        row = {"layer": l, "dir": name, "d_ff": int(d_ff),
               "share": {k: float(np.sort(c2)[::-1][:k].sum() / tot) for k in K_LIST},
               "gini_c2": gini(c2),
               "corr_absC_colnorm": float(np.corrcoef(np.abs(c), col_norm)[0, 1])}
        # 零分布：随机方向
        cn = U @ W                          # (N_NULL, d_ff)
        cn2 = cn ** 2
        totn = cn2.sum(1, keepdims=True)
        null_share = {}
        for k in K_LIST:
            part = np.sort(cn2, axis=1)[:, ::-1][:, :k].sum(1) / totn[:, 0]
            null_share[k] = {"mean": float(part.mean()), "p95": float(np.quantile(part, 0.95)),
                             "sd": float(part.std())}
        row["null"] = null_share
        row["z_64"] = float((row["share"][64] - null_share[64]["mean"]) / (null_share[64]["sd"] + 1e-12))
        # 参与比（有效列数）与达到 50%/80% 能量所需列数
        pr = float(tot ** 2 / (c2 ** 2).sum())
        order = np.sort(c2)[::-1]
        cum = np.cumsum(order) / tot
        k50 = int(np.searchsorted(cum, 0.50) + 1)
        k80 = int(np.searchsorted(cum, 0.80) + 1)
        row["participation_ratio"] = pr
        row["k50"], row["k80"] = k50, k80
        results.append(row)
        ratio = row["share"][64] / max(null_share[64]["mean"], 1e-9)
        say(f"[c] l={l:2d} {name:8s} S64={row['share'][64]:.4f} null={null_share[64]['mean']:.4f} "
            f"x{ratio:5.2f} z={row['z_64']:7.2f} S8={row['share'][8]:.4f} gini={row['gini_c2']:.3f} "
            f"PR={pr:6.0f}/{d_ff} k50={k50} k80={k80} corr|c|,norm={row['corr_absC_colnorm']:+.2f}")

# 判决（预注册 §6.2 W6-b：top-64 能量份额是否显著高于随机构型）
mid_rows = [r for r in results if r["layer"] in MID and r["dir"] == "reg_all"]
sh = [r["share"][64] for r in mid_rows]
zs = [r["z_64"] for r in mid_rows]
say("")
say(f"[mid-band reg_all] S64 = {np.mean(sh):.4f} ± {np.std(sh):.4f}  (null 期望 ≈ {64/9728:.4f})")
say(f"[mid-band reg_all] z_64 = {np.mean(zs):.2f} ± {np.std(zs):.2f}")
SHARE_LOC = 0.30
verdict = "LOCALIZED_WRITE_CANDIDATE" if (np.mean(sh) >= SHARE_LOC and np.mean(zs) >= 3) else "DISTRIBUTED_WRITE_CANDIDATE"
say(f"[verdict] {verdict}  (门槛预注册: S64 ≥ {SHARE_LOC} 且 mean z ≥ 3)")

out = {"model": str(MODEL), "npz": str(NPZ.name), "mid_layers": MID, "ctrl_layers": CTRL,
       "n_null": N_NULL, "seed": SEED, "k_list": K_LIST,
       "mean_share64_reg_all": float(np.mean(sh)), "mean_z64": float(np.mean(zs)),
       "verdict": verdict, "rows": results, "log": out_lines}
(RES / "pat_q9b_w6_precheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8", newline="")
say(f"[save] {RES / 'pat_q9b_w6_precheck.json'}")
