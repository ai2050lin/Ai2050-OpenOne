# -*- coding: utf-8 -*-
"""Q9-B 主实验（Qwen3-4B）：读写分离 patch 检验 W1–W3 + W6 权重空间语域消融

预注册：PAT_Q9B_PREREG_v1.md + PAT_Q9B_WSP_ADDENDUM_v1.md（含修订 A1 四数据集电池）

阶段（STAGE 环境变量）:
  patch —— W1 中段 MLP 语域方向 patch（充分性）、W2 消融（必要性）、W3 注意力同轴对照
  w6    —— W6-a 全投影权重移除 + W6-b 列级剂量（top-|c| vs 随机列）+ 四数据集能力保持门

用法:
  STAGE=patch [SMOKE=1] python tests/gpt5/pat_q9b.py
  STAGE=w6    [SMOKE=1] python tests/gpt5/pat_q9b.py
输出: tests/gpt5/result/pat_q9b_qwen_<stage>.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
from pat_q9a import build_corpus, REG  # noqa: E402  同批材料，冻结

RES = ROOT / "tests" / "gpt5" / "result"
NPZ = RES / "pat_q9a_qwen.npz"
MODEL = os.environ.get("MODEL", "D:/develop/model/Qwen/Qwen3-4B")
STAGE = os.environ.get("STAGE", "patch")
SMOKE = os.environ.get("SMOKE", "0") == "1"
MID = list(range(18, 28))
READ_LAYER = 35           # resid_in[35]，下游于全部 patch 层，且避开 final-norm 末项
SEED = 20261102
N_RAND = 5 if SMOKE else 10

logs = []


def say(s):
    print(s, flush=True)
    logs.append(s)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()
    return tok, m


def fit_probe(X, y):
    clf = LogisticRegression(C=1.0, max_iter=3000)
    clf.fit(X, y)
    return clf


def main():
    t0 = time.time()
    z = np.load(NPZ)
    R, A, M = z["R"].astype(np.float32), z["A"].astype(np.float32), z["M"].astype(np.float32)
    y_reg, y_lang, is_disc = z["y_reg"], z["y_lang"], z["is_disc"]
    N, L, d = R.shape
    Rext = np.zeros((N, L + 1, d), np.float32)
    Rext[:, :L] = R
    Rext[:, L] = R[:, L - 1] + A[:, L - 1] + M[:, L - 1]
    say(f"[field] N={N} L={L} d={d} disc={int(is_disc.sum())} conf={int((~is_disc).sum())}")

    # ---- 冻结方向：逐层 resid_out[l] 的 colloq−acad 均值差（预注册 §1.1）
    V = {}
    for l in MID:
        X = Rext[:, l + 1]
        mu = X[is_disc & (y_reg == 0)].mean(0) - X[is_disc & (y_reg == 1)].mean(0)
        V[l] = mu / (np.linalg.norm(mu) + 1e-9)

    # ---- 冻结读出探针：resid_in[READ_LAYER] 上的 3-way，disc 拟合
    clf = fit_probe(R[is_disc, READ_LAYER].astype(np.float64), y_reg[is_disc])
    base_acc_disc = float((clf.predict(R[is_disc, READ_LAYER].astype(np.float64)) == y_reg[is_disc]).mean())
    say(f"[probe] 3-way disc acc={base_acc_disc:.4f} (l={READ_LAYER})")

    # ---- 模型与材料
    tok, m = load_model()
    texts, meta = build_corpus()
    conf_idx = [i for i, mt in enumerate(meta) if mt["split"] == "conf"]
    if SMOKE:
        conf_idx = conf_idx[:24]
    say(f"[material] conf n={len(conf_idx)}")

    # ---- 采集基线（干预通过临时钩子完成，这里只保留无钩子的干净前向）

    def _forward(ids):
        with torch.no_grad():
            out = m(input_ids=ids, output_hidden_states=True)
        vec = out.hidden_states[READ_LAYER][0, -1].float().cpu().numpy()
        lg = out.logits[0, -1].float().cpu().numpy()
        return vec, lg

    def run(text):
        return _forward(tok(text, return_tensors="pt").input_ids.cuda())

    # 幅度标定（预注册 §2）：1 单位 = 该层 mlp_out 在 v 上投影的中位数
    SCALE = {}
    for l in MID:
        pr = M[is_disc, l] @ V[l]
        SCALE[l] = float(np.median(np.abs(pr)))

    def run_hooked(text, kind, vecs, alpha=1.0, sgn=1, target="mlp", scales=None):
        hs = []
        scales = scales or SCALE

        def mk(l):
            def fn(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                v = torch.tensor(vecs[l], device=t.device, dtype=t.dtype)
                t = t.clone()
                if kind == "add":
                    t[:, -1, :] = t[:, -1, :] + (alpha * scales[l] * sgn) * v
                else:
                    p = (t[:, -1, :] * v).sum(-1, keepdim=True)
                    t[:, -1, :] = t[:, -1, :] - p * v
                return (t,) + out[1:] if isinstance(out, tuple) else t
            return fn

        for l in MID:
            if target == "mlp":
                hs.append(m.model.layers[l].mlp.register_forward_hook(mk(l)))
            elif target == "attn":
                hs.append(m.model.layers[l].self_attn.register_forward_hook(mk(l)))
            else:  # both：同一 delta 施加到两个组件的输出（联合消融/联合 patch）
                hs.append(m.model.layers[l].mlp.register_forward_hook(mk(l)))
                hs.append(m.model.layers[l].self_attn.register_forward_hook(mk(l)))
        try:
            return _forward(tok(text, return_tensors="pt").input_ids.cuda())
        finally:
            for h in hs:
                h.remove()

    # ---- 基线
    base_pairs = [run(texts[i]) for i in conf_idx]
    bvec = np.array([b[0] for b in base_pairs], np.float64)
    blg = np.array([b[1] for b in base_pairs], np.float64)
    bf = clf.decision_function(bvec)
    bpred = clf.classes_[bf.argmax(1)]
    bmargin = bf[:, 0] - bf[:, 1]
    blogp = blg - np.log(np.exp(blg - blg.max(1, keepdims=True)).sum(1, keepdims=True)) - blg.max(1, keepdims=True)
    say(f"[base] conf 3-way acc={float((bpred == y_reg[conf_idx]).mean()):.4f} "
        f"margin(colloq-acad)={bmargin.mean():+.2f}")

    def metrics(pairs, sgn=None, note=""):
        vecs = np.array([p[0] for p in pairs], np.float64)
        lgs = np.array([p[1] for p in pairs], np.float64)
        f = clf.decision_function(vecs)
        pred = clf.classes_[f.argmax(1)]
        m = f[:, 0] - f[:, 1]
        logq = lgs - np.log(np.exp(lgs - lgs.max(1, keepdims=True)).sum(1, keepdims=True)) - lgs.max(1, keepdims=True)
        kl = (np.exp(blogp) * (blogp - logq)).sum(1)
        out = {"acc": float((pred == y_reg[conf_idx]).mean()),
               "d_margin": float(np.mean(m - bmargin)),
               "kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)),
               "top1_change": float((lgs.argmax(1) != blg.argmax(1)).mean())}
        if sgn is not None:
            tgt = 0 if sgn > 0 else 1
            out["flip_toward_s"] = float(((pred == tgt) & (bpred != tgt)).mean())
        say(f"[{note}] acc={out['acc']:.3f} dmargin={out['d_margin']:+.2f} "
            f"KL={out['kl_mean']:.3f}(med {out['kl_median']:.3f}) top1chg={out['top1_change']:.3f}"
            + (f" flip={out['flip_toward_s']:.3f}" if sgn is not None else ""))
        return out

    res = {"model": MODEL, "stage": STAGE, "smoke": SMOKE, "mid": MID, "read_layer": READ_LAYER,
           "n_conf": len(conf_idx), "base_conf_acc": float((bpred == y_reg[conf_idx]).mean()),
           "base_disc_acc": base_acc_disc, "scale": {str(k): v for k, v in SCALE.items()}}

    rng = np.random.default_rng(SEED)

    def rand_V(seed_i):
        r2 = np.random.default_rng(SEED + 100 + seed_i)
        out = {}
        for l in MID:
            u = r2.standard_normal(d).astype(np.float32)
            out[l] = u / np.linalg.norm(u)
        return out

    if STAGE == "patch":
        res["W1"] = {}
        for alpha in (0.5, 1.0, 2.0, 4.0):
            for sgn in (1, -1):
                pairs = [run_hooked(texts[i], "add", V, alpha=alpha, sgn=sgn) for i in conf_idx]
                res["W1"][f"a{alpha}_s{sgn:+d}"] = metrics(pairs, sgn, f"W1 a={alpha} s={sgn:+d}")

        null_kl, null_dm = [], []
        for r in range(N_RAND):
            Vr = rand_V(r)
            pairs = [run_hooked(texts[i], "add", Vr, alpha=1.0, sgn=1) for i in conf_idx]
            mm = metrics(pairs, None, f"W1-null r={r}")
            null_kl.append(mm["kl_mean"])
            null_dm.append(mm["d_margin"])
        res["W1_null"] = {"kl": null_kl, "d_margin": null_dm,
                          "kl_p95": float(np.quantile(null_kl, 0.95)),
                          "dm_p95": float(np.quantile(np.abs(null_dm), 0.95))}
        say(f"[W1-null] KL mean={np.mean(null_kl):.3f} p95={np.quantile(null_kl, 0.95):.3f} | "
            f"|d_margin| p95={np.quantile(np.abs(null_dm), 0.95):.2f}")

        # ---------- W2 必要性：消融 ----------
        pairs = [run_hooked(texts[i], "ablate", V) for i in conf_idx]
        res["W2"] = metrics(pairs, None, "W2 ablate")
        null_acc, null_kl2 = [], []
        for r in range(N_RAND):
            Vr = rand_V(r)
            pp = [run_hooked(texts[i], "ablate", Vr) for i in conf_idx]
            mm = metrics(pp, None, f"W2-null r={r}")
            null_acc.append(mm["acc"])
            null_kl2.append(mm["kl_mean"])
        res["W2_null"] = {"acc": null_acc, "kl": null_kl2, "acc_p05": float(np.quantile(null_acc, 0.05)),
                          "kl_p95": float(np.quantile(null_kl2, 0.95))}

        # ---------- W3 注意力同轴对照 ----------
        res["W3"] = {}
        for sgn in (1, -1):
            pairs = [run_hooked(texts[i], "add", V, alpha=1.0, sgn=sgn, target="attn") for i in conf_idx]
            res["W3"][f"s{sgn:+d}"] = metrics(pairs, sgn, f"W3 attn s={sgn:+d}")

        # W3b 注意力必要性：沿 v 消融 attn 输出（与 W2 的 MLP 消融对照）
        pairs = [run_hooked(texts[i], "ablate", V, target="attn") for i in conf_idx]
        res["W3b_ablate_attn"] = metrics(pairs, None, "W3b ablate attn")
        # W2b 联合消融：mlp+attn 同时减去 v 投影（方向的整体必要性）
        pairs = [run_hooked(texts[i], "ablate", V, target="both") for i in conf_idx]
        res["W2b_ablate_both"] = metrics(pairs, None, "W2b ablate both")

    elif STAGE == "w6":
        # ---------- W6 权重编辑 ----------
        import torch.nn.functional as F  # noqa: F401
        base_pred2 = bpred.copy()
        base_acc = float((base_pred2 == y_reg[conf_idx]).mean())
        say(f"[W6] base conf acc={base_acc:.3f}")

        def col_projections(l):
            W = m.model.layers[l].mlp.down_proj.weight.data.float()   # (d, d_ff)
            u = torch.tensor(V[l], device=W.device, dtype=W.dtype)
            c = (W * u[:, None]).sum(0)                               # (d_ff,)
            return W, u, c

        def edit(mode, beta=1.0, k=None, top=True):
            """就地编辑 down_proj；返回差量记录，供精确回滚（R-MTH-32 纪律：编辑可逆）"""
            recs = []
            rng2 = np.random.default_rng(SEED + (k or 0))
            with torch.no_grad():
                for l in MID:
                    W = m.model.layers[l].mlp.down_proj.weight.data.float()
                    u = torch.tensor(V[l], device=W.device, dtype=W.dtype)
                    c = W.T @ u
                    if mode == "full":
                        dW = beta * torch.outer(u, c)
                        W = W - dW
                        rec = (l, None, dW)
                    else:
                        if top:
                            idx = torch.topk(c.abs(), k).indices
                        else:
                            idx = torch.from_numpy(rng2.choice(W.shape[1], size=k, replace=False)).to(W.device)
                        dW = beta * torch.outer(u, c[idx])
                        W[:, idx] = W[:, idx] - dW
                        rec = (l, idx, dW)
                    m.model.layers[l].mlp.down_proj.weight.data.copy_(
                        W.to(m.model.layers[l].mlp.down_proj.weight.dtype))
                    recs.append(rec)
            return recs

        def undo(recs):
            with torch.no_grad():
                for l, idx, dW in recs:
                    W = m.model.layers[l].mlp.down_proj.weight.data.float()
                    if idx is None:
                        W = W + dW
                    else:
                        W[:, idx] = W[:, idx] + dW
                    m.model.layers[l].mlp.down_proj.weight.data.copy_(
                        W.to(m.model.layers[l].mlp.down_proj.weight.dtype))

        base_lg = np.array([b[1] for b in base_pairs], np.float64)
        base_logp = base_lg - np.log(np.exp(base_lg - base_lg.max(1, keepdims=True)).sum(1, keepdims=True)) - base_lg.max(1, keepdims=True)

        def probe_flip():
            preds, kls = [], []
            for i in conf_idx:
                vec, lg = run(texts[i])
                preds.append(int(clf.predict(vec[None, :].astype(np.float64))[0]))
                lgf = lg.astype(np.float64)
                logq = lgf - np.log(np.exp(lgf - lgf.max()).sum()) - lgf.max()
                kls.append(float((np.exp(base_logp[conf_idx.index(i)]) * (base_logp[conf_idx.index(i)] - logq)).sum()))
            preds = np.array(preds)
            tgt = 0
            return {"acc": float((preds == y_reg[conf_idx]).mean()),
                    "flip_to_colloq": float(((preds == tgt) & (base_pred2 != tgt)).mean()),
                    "kl_mean": float(np.mean(kls))}

        w6b = {}
        for k in ([64, 512] if SMOKE else [64, 512, 2048]):
            recs = edit("cols", k=k, top=True)
            top_r = probe_flip()
            undo(recs)
            recs = edit("cols", k=k, top=False)
            rand_r = probe_flip()
            undo(recs)
            w6b[f"k{k}"] = {"top": top_r, "rand": rand_r}
            say(f"[W6-b] k={k} top: acc={top_r['acc']:.3f} flip={top_r['flip_to_colloq']:.3f} KL={top_r['kl_mean']:.4f} | "
                f"rand: acc={rand_r['acc']:.3f} flip={rand_r['flip_to_colloq']:.3f} KL={rand_r['kl_mean']:.4f}")

        cap_path = ROOT / "data" / "lang_bench" / "q9b_capgate.v1.jsonl"
        cap = [json.loads(x) for x in cap_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if SMOKE:
            cap = cap[:16]
        letters = ["A", "B", "C", "D"]
        lead = tok("答案：", return_tensors="pt").input_ids[0]
        lmap = {ch: tok(ch, add_special_tokens=False).input_ids[0] for ch in letters}

        lmap_sp = {ch: tok(" " + ch, add_special_tokens=False).input_ids for ch in letters}

        def cap_acc():
            """四选项强制选择：在 prompt 末位（答案：之后）读 A/B/C/D 的 logit（含空格变体取大）"""
            correct = 0
            for q in cap:
                prompt = (f"{q['question']}\nA. {q['choices'][0]}\nB. {q['choices'][1]}\n"
                          f"C. {q['choices'][2]}\nD. {q['choices'][3]}\n答案：")
                ids = tok(prompt, return_tensors="pt").input_ids.cuda()
                with torch.no_grad():
                    lg = m(input_ids=ids).logits[0, -1].float()
                lp = []
                for ch in letters:
                    v = float(lg[lmap[ch]])
                    for t in lmap_sp[ch]:
                        v = max(v, float(lg[t]))
                    lp.append(v)
                correct += int(np.argmax(lp) == letters.index(q["answer"]))
            return correct / len(cap), correct

        # W6-a 全投影移除 + 能力保持门（先编辑后测，再回滚测基线）
        recs_full = edit("full", beta=1.0)
        full_r = probe_flip()
        say(f"[W6-a] 全移除: acc={full_r['acc']:.3f} flip={full_r['flip_to_colloq']:.3f} KL={full_r['kl_mean']:.4f}")
        acc_edited, n_ok = cap_acc()
        say(f"[capgate] 编辑后 acc={acc_edited:.3f} ({n_ok}/{len(cap)})")
        undo(recs_full)
        acc_base, n_ok_b = cap_acc()
        say(f"[capgate] base acc={acc_base:.3f} ({n_ok_b}/{len(cap)})")

        res["W6"] = {"dose": w6b, "full": full_r, "capgate": {"base": acc_base, "edited": acc_edited,
                                                              "n_base": n_ok_b, "n_edited": n_ok, "n": len(cap)}}

    res["logs"] = logs
    out = RES / f"pat_q9b_qwen_{STAGE}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline="")
    say(f"[save] {out}  耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
