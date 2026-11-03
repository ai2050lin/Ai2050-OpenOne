# -*- coding: utf-8 -*-
"""Q9-A 独立磁盘复核：从冻结 npz/json 确定性重算全部关键量，对照 result/账本/规则/MEMO/wlog/MEMORY。
不依赖主脚本 import；统计实现独立重写。置换用 default_rng(20261101) 相同调用序列逐位重放。
输出 tests/gpt5_temp/q9a_disk_verify.txt，末尾 TOTAL FAILS: n。
"""
import json, hashlib
import numpy as np
from pathlib import Path

ROOT = Path(r"D:/AI2050/Ai2050-OpenOne")
SEED = 20261101
N_PERM = 200
OUT = []
FAILS = []
NCHK = [0]


def say(s):
    OUT.append(str(s))


def chk(name, ok, got="", exp=""):
    NCHK[0] += 1
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
    say(f"[{tag}] {name}  got={got}  exp={exp}")


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def probe_acc(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(ytr)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


# ============ A0_load ============
say("== A0_load ==")
data = {}
for tag, Lx, dx in [("qwen", 36, 2560), ("glm", 40, 4096)]:
    z = np.load(ROOT / f"tests/gpt5/result/pat_q9a_{tag}.npz")
    data[tag] = z
    chk(f"A0.{tag}.keys", set(z.keys()) == {"R", "A", "M", "RB", "AB", "MB", "posA", "y_reg", "y_lang", "is_disc"},
        sorted(z.keys()))
    for k in ["R", "A", "M", "RB", "AB", "MB"]:
        chk(f"A0.{tag}.{k}.shape_dtype", z[k].shape == (360, Lx, dx) and z[k].dtype == np.float16,
            f"{z[k].shape}/{z[k].dtype}")
    y = z["y_reg"]
    chk(f"A0.{tag}.labels", sorted(np.unique(y).tolist()) == [0, 1, 2] and
        all(int(np.sum(y == c)) == 120 for c in (0, 1, 2)) and
        int(np.sum(z["y_lang"] == 0)) == 180 and int(np.sum(z["y_lang"] == 1)) == 180 and
        int(np.sum(z["is_disc"])) == 180,
        f"y_counts={[int(np.sum(y == c)) for c in (0, 1, 2)]}")

# ============ A_result ============
say("== A_result ==")
HARD = {
    "qwen": dict(res="191456_pat_q9a_qwen.json", sha8="3f470a5b", L=36, d=2560,
                 idrel=0.005509285721927881, idrelB=0.004711492918431759,
                 acc0=2 / 3, acc1=1.0,
                 accB={"0": 5 / 6, "1": 59 / 60, "18": 1.0, "36": 1.0},
                 perm0=(0.31916666666666665, 0.17777756076375645),
                 bag2=1.0, v3=dict(in_zh=1.0, zh2en=0.5, in_en=1.0, en2zh=53 / 60,
                                   gap=0.1166666666666667, verdict="PARTIAL", layer=1),
                 band_zh_mid=dict(band=[18, 27], med_attn=0.760302722454071, med_mlp=1.961750864982605,
                                  ratio=0.3875633478164673, p=0.009950248756218905),
                 band_en_0=dict(band=[0, 9], ratio=0.19153359532356262, p=0.009950248756218905),
                 v1="STYLE_HAS_COMPUTED_INCREMENT"),
    "glm": dict(res="192208_pat_q9a_glm.json", sha8="6df1b476", L=40, d=4096,
                idrel=0.005718400701880455, idrelB=0.005637466441839933,
                acc0=2 / 3, acc1=0.9833333333333333,
                accB={"0": 5 / 6, "1": 0.85, "20": 1.0, "40": 1.0},
                perm0=(0.32583333333333336, 0.19457753381792733),
                bag2=1.0, v3=dict(in_zh=1.0, zh2en=8 / 15, in_en=1.0, en2zh=1.0,
                                  verdict="STYLE_ABSTRACT", layer=2),                band_zh_mid=dict(band=[20, 30], med_attn=0.1557258665561676, med_mlp=0.6961708068847656,
                                 ratio=0.22368916869163513, p=0.004975124378109453),
                band_en_0=dict(band=[0, 10], ratio=0.6452641487121582, p=0.01990049751243781),
                v1="AMBIGUOUS"),
}
RES = {}
for tag, H in HARD.items():
    p = ROOT / "results/2026-11-01" / H["res"]
    doc = json.loads(p.read_text(encoding="utf-8"))
    RES[tag] = doc
    body = json.dumps({k: v for k, v in doc.items() if k != "sha8"}, ensure_ascii=False, indent=1)
    chk(f"A.{tag}.self_sha8", sha8(body.encode("utf-8")) == H["sha8"], sha8(body.encode("utf-8")), H["sha8"])
    pl = doc["payload"]
    chk(f"A.{tag}.Ld", pl["L"] == H["L"] and pl["d"] == H["d"] and pl["n_texts"] == 360 and not pl["smoke"],
        f"L={pl['L']} d={pl['d']}")
    chk(f"A.{tag}.identity", close(pl["identity_rel_max"], H["idrel"], 1e-12) and pl["identity_ok"] is True
        and close(pl["identity_rel_max_B"], H["idrelB"], 1e-12),
        f"{pl['identity_rel_max']:.6g}/{pl['identity_rel_max_B']:.6g}")
    ar = pl["acc_resid"]
    chk(f"A.{tag}.acc_resid01", close(ar[0], H["acc0"], 1e-12) and close(ar[1], H["acc1"], 1e-12)
        and all(close(a, 1.0, 1e-12) for a in ar[2:]), f"l0={ar[0]:.4f} l1={ar[1]:.4f}")
    chk(f"A.{tag}.v1", pl["v1"]["verdict"] == H["v1"], pl["v1"]["verdict"])
    for k, v in H["accB"].items():
        chk(f"A.{tag}.acc_posB[{k}]", close(pl["acc_posB"][k], v, 1e-12), pl["acc_posB"][k], v)
    chk(f"A.{tag}.perm0", close(pl["perm_null"]["0"]["mean"], H["perm0"][0], 1e-12) and
        close(pl["perm_null"]["0"]["sd"], H["perm0"][1], 1e-12),
        f"{pl['perm_null']['0']['mean']:.6g}/{pl['perm_null']['0']['sd']:.6g}")
    chk(f"A.{tag}.bag2", close(pl["p5_baselines"]["0"]["bag2"], H["bag2"], 1e-12) and
        close(pl["p5_baselines"][str(H["L"] - 1)]["bag2"], H["bag2"], 1e-12),
        json.dumps(pl["p5_baselines"]))
    v3 = pl["v3"]
    chk(f"A.{tag}.v3", v3["verdict"] == H["v3"]["verdict"] and close(v3["in_zh"], H["v3"]["in_zh"], 1e-12)
        and close(v3["zh2en"], H["v3"]["zh2en"], 1e-12) and close(v3["en2zh"], H["v3"]["en2zh"], 1e-12)
        and v3["layer"] == H["v3"]["layer"], json.dumps(v3))
    p3 = pl["p3_component"]
    bz = p3["zh"]["bands"][2]
    chk(f"A.{tag}.p3.zh_mid", bz["band"] == H["band_zh_mid"]["band"] and
        close(bz["ratio"], H["band_zh_mid"]["ratio"], 1e-9) and
        close(bz["p_paired"], H["band_zh_mid"]["p"], 1e-12), json.dumps(bz))
    be = p3["en"]["bands"][0]
    chk(f"A.{tag}.p3.en_first", be["band"] == H["band_en_0"]["band"] and
        close(be["ratio"], H["band_en_0"]["ratio"], 1e-9) and
        close(be["p_paired"], H["band_en_0"]["p"], 1e-12), json.dumps(be))

# ============ B_recompute（npz float64/float32 独立重算） ============
say("== B_recompute ==")


def build_Rext(R, A, M, L):
    Rx = np.zeros((R.shape[0], L + 1, R.shape[2]), np.float16)
    Rx[:, :L] = R
    Rx[:, L] = (R[:, L - 1].astype(np.float32) + A[:, L - 1] + M[:, L - 1]).astype(np.float16)
    return Rx


for tag, H in HARD.items():
    z = data[tag]
    R, A, M, RB, AB, MB = z["R"], z["A"], z["M"], z["RB"], z["AB"], z["MB"]
    y_reg, y_lang, is_disc = z["y_reg"], z["y_lang"], z["is_disc"]
    L = H["L"]
    tr, te = is_disc, ~is_disc
    pl = RES[tag]["payload"]

    # B1 恒等式（A 位与 B 位，fp32 同主脚本口径）
    err = (R[:, 1:].astype(np.float32) - (R[:, :-1] + A[:, :-1] + M[:, :-1]).astype(np.float32))
    den = np.linalg.norm(R[:, 1:].astype(np.float32), axis=-1) + 1e-6
    rel = float((np.linalg.norm(err, axis=-1) / den).max())
    chk(f"B1.{tag}.identity_A", close(rel, pl["identity_rel_max"], 1e-9), rel, pl["identity_rel_max"])
    errB = (RB[:, 1:].astype(np.float32) - (RB[:, :-1] + AB[:, :-1] + MB[:, :-1]).astype(np.float32))
    denB = np.linalg.norm(RB[:, 1:].astype(np.float32), axis=-1) + 1e-6
    relB = float((np.linalg.norm(errB, axis=-1) / denB).max())
    chk(f"B1.{tag}.identity_B", close(relB, pl["identity_rel_max_B"], 1e-9), relB, pl["identity_rel_max_B"])

    # B2 acc_resid 逐层重算
    Rext = build_Rext(R, A, M, L)
    accs = [probe_acc(Rext[tr, l].astype(np.float32), y_reg[tr],
                      Rext[te, l].astype(np.float32), y_reg[te]) for l in range(L + 1)]
    dmax = max(abs(a - b) for a, b in zip(accs, pl["acc_resid"]))
    chk(f"B2.{tag}.acc_resid_replay", dmax <= 1e-9, f"max|d|={dmax:.3g} (L+1={L + 1} 层)")

    # B3 acc_posB 重算
    RBx = build_Rext(RB, AB, MB, L)
    for l in (0, 1, L // 2, L):
        a = probe_acc(RBx[tr, l].astype(np.float32), y_reg[tr],
                      RBx[te, l].astype(np.float32), y_reg[te])
        chk(f"B3.{tag}.acc_posB[{l}]", close(a, pl["acc_posB"][str(l)], 1e-9), a, pl["acc_posB"][str(l)])

    # B4 perm_null l=0 逐位重放（首个 rng 消费者）
    rng = np.random.default_rng(SEED)
    Xtr = Rext[tr, 0].astype(np.float32)
    Xte = Rext[te, 0].astype(np.float32)
    acc_list = []
    for _ in range(N_PERM):
        yp = rng.permutation(y_reg[tr])
        acc_list.append(probe_acc(Xtr, yp, Xte, y_reg[te]))
    mu, sd = float(np.mean(acc_list)), float(np.std(acc_list))
    chk(f"B4.{tag}.perm_l0_replay", close(mu, pl["perm_null"]["0"]["mean"], 1e-9) and
        close(sd, pl["perm_null"]["0"]["sd"], 1e-9), f"{mu:.6g}/{sd:.6g}",
        f"{pl['perm_null']['0']['mean']:.6g}/{pl['perm_null']['0']['sd']:.6g}")

    # B5 P3 组件归因重算 + 配对置换 p 重放（rng 快进 3 层置换后按 zh→en 逐带重放）
    rng2 = np.random.default_rng(SEED)
    for _ in range(3 * N_PERM):
        rng2.permutation(y_reg[tr])
    col, acad, code = 0, 1, 2
    p3r = {}
    for lang in ("zh", "en"):
        ml = y_lang == (0 if lang == "zh" else 1)
        disc_l, conf_l = tr & ml, te & ml
        proj_a, proj_m = [], []
        for l in range(L):
            X = Rext[:, l + 1].astype(np.float32)
            grand = X[disc_l].mean(0)
            mc = X[disc_l & (y_reg == col)].mean(0)
            ma_ = X[disc_l & (y_reg == acad)].mean(0)
            v = (mc - ma_)
            v /= (np.linalg.norm(v) + 1e-9)
            proj_a.append(A[conf_l, l].astype(np.float32) @ v)
            proj_m.append(M[conf_l, l].astype(np.float32) @ v)
        proj_a = np.array(proj_a)
        proj_m = np.array(proj_m)
        bands = [(0, L // 4), (L // 4, L // 2), (L // 2, 3 * L // 4), (3 * L // 4, L)]
        band_stat = []
        for b0, b1 in bands:
            ia = float(np.median(np.abs(proj_a[b0:b1])))
            im = float(np.median(np.abs(proj_m[b0:b1])))
            diff = (np.abs(proj_a[b0:b1]) - np.abs(proj_m[b0:b1])).mean(1)
            obs = diff.mean()
            cnt = 0
            for _ in range(N_PERM):
                sgn = rng2.choice([-1, 1], size=diff.shape)
                if abs((sgn * diff).mean()) >= abs(obs):
                    cnt += 1
            band_stat.append({"band": [b0, b1], "med_attn": ia, "med_mlp": im,
                              "ratio": ia / (im + 1e-12), "p_paired": (cnt + 1) / (N_PERM + 1)})
        p3r[lang] = band_stat
    for lang in ("zh", "en"):
        for bi, (brec, bexp) in enumerate(zip(p3r[lang], pl["p3_component"][lang]["bands"])):
            ok = (brec["band"] == bexp["band"] and close(brec["med_attn"], bexp["med_attn"], 1e-4)
                  and close(brec["med_mlp"], bexp["med_mlp"], 1e-4)
                  and close(brec["ratio"], bexp["ratio"], 1e-6)
                  and close(brec["p_paired"], bexp["p_paired"], 1e-12))
            chk(f"B5.{tag}.p3.{lang}.band{bi}", ok,
                f"ratio={brec['ratio']:.4f} p={brec['p_paired']:.4f}",
                f"ratio={bexp['ratio']:.4f} p={bexp['p_paired']:.4f}")

# B6 迁移曲线（最近质心，独立实现，对照冻结曲线 json）
say("== B6_transfer_curve ==")
curve = json.loads((ROOT / "tests/gpt5/result/pat_q9a_transfer_curve.json").read_text(encoding="utf-8"))


def nc_bin(Xtr_, ytr_, Xte_, yte_):
    c1 = Xtr_[ytr_ == 1].mean(0)
    c0 = Xtr_[ytr_ == 0].mean(0)
    c1 = c1 / np.linalg.norm(c1)
    c0 = c0 / np.linalg.norm(c0)
    return float(((Xte_ @ c1 > Xte_ @ c0).astype(int) == yte_).mean())


def nc3(Xtr_, ytr_, Xte_, yte_):
    labs = np.unique(ytr_)
    C = []
    for c in labs:
        mu = Xtr_[ytr_ == c].mean(0)
        C.append(mu / (np.linalg.norm(mu) + 1e-9))
    C = np.array(C)
    Xn = Xte_ / (np.linalg.norm(Xte_, axis=1, keepdims=True) + 1e-9)
    return float((labs[(Xn @ C.T).argmax(1)] == yte_).mean())


for tag in ("qwen", "glm"):
    z = data[tag]
    R, A, M = z["R"].astype(np.float64), z["A"].astype(np.float64), z["M"].astype(np.float64)
    y_reg, y_lang, is_disc = z["y_reg"], z["y_lang"], z["is_disc"]
    N, L, d = R.shape
    Rext64 = np.zeros((N, L + 1, d))
    Rext64[:, :L] = R
    Rext64[:, L] = R[:, L - 1] + A[:, L - 1] + M[:, L - 1]
    tr, te = is_disc, ~is_disc
    yb = np.where(y_reg == 0, 1, 0)
    sz = np.isin(y_reg, [0, 1]) & (y_lang == 0)
    se = np.isin(y_reg, [0, 1]) & (y_lang == 1)
    rows = []
    for l in range(0, L + 1, 4):
        z2e = nc_bin(Rext64[tr & sz, l], yb[tr & sz], Rext64[te & se, l], yb[te & se])
        e2z = nc_bin(Rext64[tr & se, l], yb[tr & se], Rext64[te & sz, l], yb[te & sz])
        a3 = nc3(Rext64[tr, l], y_reg[tr], Rext64[te, l], y_reg[te])
        rows.append([l, round(a3, 3), round(z2e, 3), round(e2z, 3)])
    exp_rows = curve[tag] if isinstance(curve[tag], list) else curve[tag].get("rows")
    ok = len(rows) == len(exp_rows) and all(
        r[0] == e[0] and all(abs(x - y) <= 1e-9 for x, y in zip(r[1:], e[1:]))
        for r, e in zip(rows, exp_rows))
    chk(f"B6.{tag}.curve", ok, f"rows={len(rows)} head={rows[:3]}", f"head={exp_rows[:3]}")
    # 关键形状断言：语域轴早期跨语言抽象化（qwen l=8 双向 1.0；glm l=8 ≥0.95、l=20 才 1.0）
    l8 = next(r for r in rows if r[0] == 8)
    if tag == "qwen":
        chk(f"B6.{tag}.abstract_at_l8", l8[2] == 1.0 and l8[3] == 1.0, f"l8 row={l8}")
    else:
        chk(f"B6.{tag}.abstract_at_l8", l8[2] >= 0.95 and l8[3] >= 0.95, f"l8 row={l8}")
        l20 = next(r for r in rows if r[0] == 20)
        chk(f"B6.{tag}.abstract_at_l20_glm", l20[2] == 1.0 and l20[3] == 1.0, f"l20 row={l20}")

# ============ C_ledger ============
say("== C_ledger ==")
lp = ROOT / "research/gpt5/atlas/atlas_ledger.json"
led = json.loads(lp.read_text(encoding="utf-8"))
meas, negs = led["measurements"], led["negatives"]
q9m = [m for m in meas if m.get("phase") == "Q9-A"]
q9n = [n for n in negs if n.get("phase") == "Q9-A"]
chk("C.q9a_meas_count", len(q9m) == 1, len(q9m))
chk("C.q9a_neg_count", len(q9n) == 3, len(q9n))
mjson = json.dumps(q9m, ensure_ascii=False)
chk("C.q9a_res_sha_ref", "3f470a5b" in mjson and "6df1b476" in mjson)
negnames = sorted(n["name"] for n in q9n)
chk("C.q9a_neg_names", negnames == sorted([
    "pat_q9a_neg_attention_is_the_register_carrier",
    "pat_q9a_neg_style_is_pure_token_identity",
    "pat_q9a_neg_register_axis_language_abstract_at_every_layer"]), negnames)
# 链式 sha：Q9-A 入账步重建（n=468, errata=11, prev=4524aa96 → 5944a2d0）
doc468 = dict(led)
doc468["measurements"] = meas[:468]
doc468["negatives"] = negs[:42]
doc468["errata_ledger"] = led["errata_ledger"][:11]
doc468["ledger_sha256_8"] = "4524aa96"
h468 = sha8(json.dumps(doc468, ensure_ascii=False, indent=1).encode("utf-8"))
chk("C.chain_step_5944a2d0", h468 == "5944a2d0", h468, "5944a2d0")
# 本轮勘误步（2026-11-02）：errata 11→12，prev=f1671034 → 38b4bd0b（硬断言）
docerr = dict(led)
docerr["errata_ledger"] = led["errata_ledger"][:12]
docerr["ledger_sha256_8"] = "f1671034"
herr = sha8(json.dumps(docerr, ensure_ascii=False, indent=1).encode("utf-8"))
chk("C.chain_erratum_38b4bd0b", herr == "38b4bd0b" and led.get("ledger_sha256_8") == "38b4bd0b",
    f"{herr}/{led.get('ledger_sha256_8')}", "38b4bd0b")
# Q9-A 之后的并发入账步：记录性复核，FAIL 降级为记录
if h468 == "5944a2d0" and len(meas) == 470 and len(negs) == 44:
    doc469 = dict(led)
    doc469["measurements"] = meas[:469]
    doc469["negatives"] = negs[:43]
    doc469["errata_ledger"] = led["errata_ledger"][:11]
    doc469["ledger_sha256_8"] = "5944a2d0"
    x1 = sha8(json.dumps(doc469, ensure_ascii=False, indent=1).encode("utf-8"))
    doc470 = dict(led)
    doc470["measurements"] = meas[:470]
    doc470["negatives"] = negs[:44]
    doc470["errata_ledger"] = led["errata_ledger"][:11]
    doc470["ledger_sha256_8"] = x1
    h470 = sha8(json.dumps(doc470, ensure_ascii=False, indent=1).encode("utf-8"))
    NCHK[0] += 1
    say(f"[{'PASS' if h470 == 'f1671034' else 'NOTE'}] C.chain_concurrent(470 条, errata=11) "
        f"recomputed={h470} recorded_prev_head=f1671034（并发写者步，仅记录）")
    if h470 != "f1671034":
        say("    （NOTE：并发步配方或字段集可能不同，不影响 Q9-A 步与勘误步断言；不记 FAIL）")

# ============ D_rules ============
say("== D_rules ==")
rled = json.loads((ROOT / "data/rules/ledger.json").read_text(encoding="utf-8"))
rules = rled["rules"]
chk("D.total_124", len(rules) == 124, len(rules))
for rid in ["R-MET-37", "R-MET-38", "R-MET-39", "R-MTH-28"]:
    cnt = sum(1 for r in rules if r.get("id") == rid)
    chk(f"D.{rid}_count1", cnt == 1, cnt)
md = (ROOT / "data/rules/LEDGER.md").read_text(encoding="utf-8")
for rid in ["R-MET-37", "R-MET-38", "R-MET-39", "R-MTH-28"]:
    chk(f"D.md.{rid}", md.count(rid) >= 1, md.count(rid))
for bak in ["research/gpt5/atlas/atlas_ledger.json.bak_q9a", "data/rules/ledger.json.bak_q9a",
            "data/rules/LEDGER.md.bak_q9a"]:
    chk(f"D.bak.{Path(bak).name}", (ROOT / bak).exists())

# ============ E_memo ============
say("== E_memo ==")
memo_b = (ROOT / "research/gpt5/docs/AGI_GPT5_MEMO.md").read_bytes()
memo = memo_b.decode("utf-8", "replace")
chk("E.marker_unique", memo.count("## Phase Q9-A:") == 1, memo.count("## Phase Q9-A:"))
for s in ["3f470a5b", "6df1b476", "5944a2d0", "MID_BAND_MLP_CARRIES", "STYLE_HAS_COMPUTED_INCREMENT",
          "R-MET-37", "R-MTH-28"]:
    chk(f"E.contains.{s}", s in memo, memo.count(s))

# ============ F_wlogs ============
say("== F_wlogs ==")
w1 = ROOT / ".workbuddy/memory/2026-11-01.md"
t1 = w1.read_text(encoding="utf-8")
chk("F.wlog1101.q9a", t1.count("Q9-A") >= 1 and "5944a2d0" in t1,
    f"q9a={t1.count('Q9-A')} head={'5944a2d0' in t1}")
vs = ROOT / "tests/gpt5/result/pat_q9a_verify_summary.json"
chk("F.verify_summary", vs.exists() and
    set(json.loads(vs.read_text(encoding="utf-8")).keys()) == {"qwen", "glm"})

# ============ G_memory ============
say("== G_memory ==")
memt = (ROOT / ".workbuddy/memory/MEMORY.md").read_text(encoding="utf-8")
chk("G.memory.q9a_line", "Q9-A" in memt and "5944a2d0" in memt,
    f"q9a={memt.count('Q9-A')} head={'5944a2d0' in memt}")

say("")
say(f"TOTAL CHECKS: {NCHK[0]}")
say(f"TOTAL FAILS: {len(FAILS)}")
for f in FAILS:
    say(f"  FAIL: {f}")

rep = ROOT / "tests/gpt5_temp/q9a_disk_verify.txt"
rep.write_text("\n".join(OUT), encoding="utf-8", newline="\n")
print("written", rep, f"checks={NCHK[0]} fails={len(FAILS)}")
