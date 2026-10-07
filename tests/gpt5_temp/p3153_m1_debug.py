# -*- coding: utf-8 -*-
"""3153 M1 定位 v2：从 3153 主脚本 exec 头部常量/函数，提取 3152 als_complete 对比。"""
import io, os, re
import numpy as np

ROOT = r"D:\AI2050\Ai2050-OpenOne"
S3153 = os.path.join(ROOT, "tests", "glm5", "phase3153_g1p3_failure_mode_anatomy.py")
S3152 = os.path.join(ROOT, "tests", "glm5", "phase3152_g1p2_tri_model_k1.py")
OUT = os.path.join(ROOT, "tests", "gpt5_temp", "p3153_m1_debug.txt")
out = []
def log(s):
    out.append(s)

s3 = io.open(S3153, encoding="utf-8").read()
# exec 头部：从 T0 到 ALLP 行
a = s3.index("T0 = time.time()")
b = s3.index("if MODEL in ('qwen3-4b'")
head = s3[a:b]
# 去掉 3153 头部对 BASE/log 的依赖（保留但 log 落到临时目录无害）
ns = {"os": os, "sys": __import__("sys"), "json": __import__("json"),
      "time": __import__("time"),
      "hashlib": __import__("hashlib"), "np": np,
      "__name__": "p3153_head"}
os.environ.pop("P3153_SMOKE", None)
exec(compile(head, "p3153_head", "exec"), ns)
MODEL_NS = ns
log("exec head ok: NE=%d NC=%d NP_=%d NT=%d NE_KEEP=%d" %
    (ns["NE"], ns["NC"], ns["NP_"], ns["NT"], ns["NE_KEEP"]))
log("ENTS[0]/[24]: %s / %s  CLASSES=%s" % (ns["ENTS"][0], ns["ENTS"][24], ns["CLASSES"]))

# 提取两个 als_complete
def extract_func(path, name):
    s = io.open(path, encoding="utf-8").read()
    m = re.search(r"^def %s\(.*?(?=^def |^# =)" % name, s, re.M | re.S)
    return m.group(0)
f3152 = extract_func(S3152, "als_complete")
f3153 = extract_func(S3153, "als_complete")
log("als_complete identical: %s" % (f3152 == f3153))
if f3152 != f3153:
    for i, (la, lb) in enumerate(zip(f3152.splitlines(), f3153.splitlines())):
        if la != lb:
            log("DIFF L%d:\n  3152: %r\n  3153: %r" % (i, la, lb))
als_3152 = (lambda: (lambda e: e)(exec(f3152, {"np": np}, (g := {})) or g["als_complete"]))()
als_3153 = (lambda: (lambda e: e)(exec(f3153, {"np": np}, (g2 := {})) or g2["als_complete"]))()

# 数据：3152 RDIR 拼写从 3153 头部拿
RDIR = ns["RDIR"]
NPZ = os.path.join(RDIR, "phase3152", "g1p2_tri_model_k1", "qwen3-4b", "collect.npz")
log("npz exists: %s" % os.path.exists(NPZ))
z = np.load(NPZ)
H = z["H"]
KSTAR = 3
Y = H[:, :, KSTAR, :].reshape(ns["NT"] * ns["NP_"], -1).astype(np.float32)
D = Y.shape[1]

split_s1, rows_of, phi_main, ridge_primal = ns["split_s1"], ns["rows_of"], ns["phi_main"], ns["ridge_primal"]
NP_, NT, NE_KEEP, keep_e, PAIRS, NC = (
    ns["NP_"], ns["NT"], ns["NE_KEEP"], ns["keep_e"], ns["PAIRS"], ns["NC"])

train_set, test_set = split_s1(7)
tr_rows = rows_of(train_set)
te_rows = rows_of(test_set)
tr_set = set(train_set)
Xtr, _, rowvec, cols = phi_main(train_set)
W = ridge_primal(Xtr, Y[tr_rows], lam=1e-3)
B4te = np.stack([rowvec(r // NP_, r % NP_) for r in te_rows]) @ W
B4tr = Xtr @ W
Ytr = Y[tr_rows]
ref = Ytr.mean(0)
Dk = float(((Ytr - ref) ** 2).sum(1).mean()) + 1e-9

t = 0
Rg = np.zeros((NE_KEEP, NC, D), np.float32)
mask = np.zeros((NE_KEEP, NC), bool)
for pi, (i, c) in enumerate(PAIRS):
    if (i, c) in tr_set:
        Rg[keep_e.index(i), c] = \
            H[t, pi, KSTAR, :].astype(np.float32) - \
            B4tr[tr_rows.index(t * NP_ + pi)]
        mask[keep_e.index(i), c] = True
log("Rg masked norm=%.6f mu_norm=%.6f n_cells=%d" %
    (float(np.linalg.norm(Rg[mask])),
     float(np.linalg.norm((Rg * mask[..., None]).sum((0, 1)) / mask.sum())),
     int(mask.sum())))
log("in-sample rel (Rg^2/Dk mean)=%.6f" %
    float((np.linalg.norm(Rg[mask], axis=1) ** 2).mean() / Dk))

s_arg = 7 + 1000 * 7 + KSTAR + t
R1 = als_3152(Rg, mask, 5, 100, 1e-2, s_arg)
R3 = als_3153(Rg, mask, 5, 100, 1e-2, s_arg)
log("Rhat identical: %s" % bool(np.array_equal(R1, R3)))
sel = [j for j, r in enumerate(te_rows) if r // NP_ == t]
Yte = Y[te_rows]
e_b4 = float((((B4te[sel] - Yte[sel]) ** 2).sum(1) / Dk).mean())
for nm, Rh in [("als3152", R1), ("als3153", R3)]:
    M1 = np.zeros((len(te_rows), D), np.float32)
    for j, r in enumerate(te_rows):
        if r // NP_ == t:
            i2, c2 = PAIRS[r % NP_]
            M1[j] = B4te[j] + Rh[keep_e.index(i2), c2]
    e = float((((M1[sel] - Yte[sel]) ** 2).sum(1) / Dk).mean())
    log("tpl0 M1(%s) mean=%.4f B4=%.4f margin=%.4f" % (nm, e, e_b4, e - e_b4))
    log("  Rhat test-cell norm=%.6f train-cell norm=%.6f mu_norm=%.6f" %
        (float(np.linalg.norm(Rh[~mask])), float(np.linalg.norm(Rh[mask])),
         float(np.linalg.norm(Rh.mean((0, 1))))))

io.open(OUT, "w", encoding="utf-8").write("\n".join(out))
print("written")
