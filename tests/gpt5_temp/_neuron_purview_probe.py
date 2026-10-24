# -*- coding: utf-8 -*-
"""原生神经元视野探针（Goodfire essay 驱动的可行性验证，零 GPU）

目的：验证能否只用「封存 npz + 封存 execution.json」还原出
      「原生神经元 (layer,dim) → 最强激活的具体数据片段」这一证据形式，
      并量化单义程度（class purity），不新增任何模型前向。

数据：tests/glm5/result/rdc_query_construction_20260913/phase3169/g5a6_oov_panel/
      collect_{model}.npz   H[NT=3, PAIRS=730, layers, dims] float16
      execution.json        TPL / ENT_SEEN_FULL / ENT_OOV_FULL / CLASSES_*

行语义（来自 phase3169_g5a6_oov_panel.py）：
      pi = entity_index * NC + class_index，NC=10
      ENTS = 按类别顺序拼接的实体列表；CLS_OF = 每个实体的真实类别下标
      文本 = TPL[t].format(e=ENTS[i], c=CLASSES[c])   t = H 的第 0 维

输出：tests/gpt5_temp/_neuron_purview.json + 控制台简表
纪律：只读封存产物；不写 registry；数字全部实测，不引入记忆值。
"""
import io
import json
import os
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE = os.path.join(ROOT, "tests", "glm5", "result",
                     "rdc_query_construction_20260913", "phase3169", "g5a6_oov_panel")
_exec = json.load(io.open(os.path.join(PHASE, "execution.json"), encoding="utf-8"))

TPL = _exec["tpl"]                      # {"0": "{e}是一种{c}。", ...}
CLASSES_SEEN = _exec["classes_seen"]
CLASSES_OOV = _exec["classes_oov"]
ENT_SEEN = _exec["ent_seen_full"]
ENT_OOV = _exec["ent_oov_full"]

CLASSES = list(CLASSES_SEEN) + list(CLASSES_OOV)
ENT = {}
ENT.update({c: list(v) for c, v in ENT_SEEN.items()})
ENT.update({c: list(v) for c, v in ENT_OOV.items()})
ENTS, CLS_OF = [], []
for ci, c in enumerate(CLASSES):
    for e in ENT[c]:
        ENTS.append(e)
        CLS_OF.append(ci)
CLS_OF = np.asarray(CLS_OF, dtype=np.int32)
NE, NC, NT = len(ENTS), len(CLASSES), len(TPL)
assert NC == _exec["nc_full"] == 10, NC
assert NE == _exec["ne_full"] == 73, NE
assert NT == _exec["nt"] == 3, NT
PAIRS = NE * NC
assert PAIRS == _exec["pairs_full"] == 730, PAIRS

MODELS = ["qwen3-4b", "qwen3-14b", "glm4-9b"]


def npz_path(model):
    p = os.path.join(PHASE, "collect_%s.npz" % model)
    if not os.path.isfile(p):
        raise SystemExit("missing npz: %s" % p)
    return p


def load_H(model):
    with zipfile.ZipFile(npz_path(model)) as z:
        names = z.namelist()
        assert "H.npy" in names, names
        with z.open("H.npy") as f:
            H = np.load(f, allow_pickle=False)
    if H.ndim != 4:
        raise SystemExit("H ndim=%s" % (H.ndim,))
    assert H.shape[0] == NT and H.shape[1] == PAIRS, H.shape
    return H            # (NT, PAIRS, NH, D) float16


def prompt(t, pi):
    i = pi // NC
    c = pi % NC
    return TPL[str(t)].format(e=ENTS[i], c=CLASSES[c])


def purview(v, k=20):
    """对一条 (NT,PAIRS) 激活面取 top-k，量它偏向哪个因子。

    class_purity  = top-k 里众数类别的占比（1.0 = 完全同一类别）
    tpl_purity    = top-k 里众数模板的占比
    ent_purity    = top-k 里众数实体的占比
    diag_ratio    = 对角行（c == 实体真实类别）均值 / 非对角均值
    """
    flat = v.reshape(-1)
    idx = np.argpartition(-flat, k)[:k]
    t_of = idx // PAIRS
    pi_of = idx % PAIRS
    cls = pi_of % NC
    ent = pi_of // NC
    cnt_c = np.bincount(cls, minlength=NC)
    cnt_t = np.bincount(t_of, minlength=NT)
    cnt_e = np.bincount(ent, minlength=NE)
    diag = np.asarray([CLS_OF[p // NC] == (p % NC) for p in pi_of])
    return dict(
        class_purity=float(cnt_c.max() / k),
        class_mode=int(cnt_c.argmax()),
        tpl_purity=float(cnt_t.max() / k),
        ent_purity=float(cnt_e.max() / k),
        diag_frac=float(diag.mean()),
        top=[dict(t=int(t_of[j]), pi=int(pi_of[j]), act=round(float(flat[idx[j]]), 3),
                  text=prompt(int(t_of[j]), int(pi_of[j])))
             for j in np.argsort(-flat[idx])[:6]],
    )


def diag_ratio(v):
    d = np.zeros(PAIRS, dtype=bool)
    for pi in range(PAIRS):
        d[pi] = CLS_OF[pi // NC] == (pi % NC)
    a = v[:, d].mean()
    b = v[:, ~d].mean()
    return float(a / b) if abs(b) > 1e-9 else float("nan")


def pick_dims(H, layer, n_var=6, n_rnd=6, seed=0):
    """方差前 n_var + 确定性伪随机 n_rnd（避免只挑极端神经元）。"""
    slab = H[:, :, layer, :].astype(np.float32)          # (NT,PAIRS,D)
    var = slab.reshape(-1, slab.shape[-1]).var(axis=0)
    top = list(np.argsort(-var)[:n_var])
    rng = np.random.RandomState(seed)
    rnd = [int(x) for x in rng.choice(slab.shape[-1], n_rnd, replace=False)]
    seen, out = set(), []
    for d in top + rnd:
        d = int(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out, slab


def main():
    report = {"schema": "neuron_purview_probe_v1", "source_phase": 3169,
              "row_rule": "pi = entity_index*NC + class_index (NC=10); axis0 = template",
              "models": {}}
    for model in MODELS:
        H = load_H(model)
        nt, npairs, nlayers, dims = H.shape
        layers = sorted({int(round((nlayers - 1) * f)) for f in (0.25, 0.5, 0.75)})
        rows = []
        print("\n=== %s  H=%s  layers采样=%s ===" % (model, H.shape, layers))
        print("  layer  dim    var-rank  class_pur  ent_pur  tpl_pur  diag  ratio  top1 片段")
        for layer in layers:
            chosen, slab = pick_dims(H, layer)
            for rank, d in enumerate(chosen):
                v = slab[:, :, d].astype(np.float32)
                p = purview(v)
                r = diag_ratio(v)
                rec = dict(layer=layer, dim=d, rank=rank, var_pick=rank < 6,
                           **{k: p[k] for k in ("class_purity", "ent_purity", "tpl_purity")},
                           class_mode=CLASSES[p["class_mode"]], diag_frac=p["diag_frac"],
                           diag_ratio=round(r, 3) if r == r else None, top=p["top"])
                rows.append(rec)
                if rank < 8:
                    print("  %5d %5d   %-8s  %.2f      %.2f     %.2f    %.2f  %5.2f  %s"
                          % (layer, d, "var" if rank < 6 else "rnd", p["class_purity"],
                             p["ent_purity"], p["tpl_purity"], p["diag_frac"],
                             r if r == r else 0.0, p["top"][0]["text"]))
        cp = np.asarray([r["class_purity"] for r in rows])
        report["models"][model] = dict(
            H_shape=list(H.shape), layers=layers, n_neurons=len(rows),
            class_purity_mean=round(float(cp.mean()), 4),
            class_purity_median=round(float(np.median(cp)), 4),
            frac_purity_ge_0_8=round(float((cp >= 0.8).mean()), 4),
            frac_purity_ge_0_6=round(float((cp >= 0.6).mean()), 4),
            rows=rows,
        )
        print("  → class_purity  mean=%.3f median=%.3f  ≥0.8占比=%.2f"
              % (cp.mean(), np.median(cp), (cp >= 0.8).mean()))
        del H

    out = os.path.join(ROOT, "tests", "gpt5_temp", "_neuron_purview.json")
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
