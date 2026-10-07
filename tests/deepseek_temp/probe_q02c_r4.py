# -*- coding: utf-8 -*-
"""Q02c：dump 3152 四产物的关键字段，定位 0.3316/0.3898/0.3986 的出处。"""
import os, json, hashlib
ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUTD = os.path.join(ROOT, "tests", "deepseek_temp", "_review")
def sha8(b): return hashlib.sha256(b).hexdigest()[:8]
def rd(p):
    with open(os.path.join(ROOT, p), "rb") as f: return f.read()
R = []
def A(s=""): R.append(str(s))

BASE = "tests/glm5/result/rdc_query_construction_20260913/phase3152/g1p2_tri_model_k1"
KEYS = ["phase", "name", "mode", "created", "panel_rows", "n_pairs", "NL", "hidden",
        "kstar", "readout", "kgrid", "m1_curve_layers", "curve_argmin_b4",
        "k1_model_report", "gates", "v1_verdict_kstar", "m1_grid_opt_rank", "m1_overfit_note"]
TARGETS = ["0.3316", "0.3898", "0.3986", "0.0461", "0.0079", "0.0807"]

for arm in ["qwen3-4b", "qwen3-14b", "glm4k1", "summary"]:
    fn = "result_summary.json" if arm == "summary" else "result.json"
    p = "%s/%s/%s" % (BASE, arm, fn)
    try:
        b = rd(p)
    except Exception as e:
        A("== %s MISSING %s" % (arm, e)); continue
    d = json.loads(b.decode("utf-8-sig"))
    A("=== %s (%d B, sha8 %s) ===" % (arm, len(b), sha8(b)))
    A("  顶层键 = %s" % list(d.keys()))
    for k in KEYS:
        if k in d:
            A("  %-18s = %s" % (k, json.dumps(d[k], ensure_ascii=False)[:900]))
    txt = json.dumps(d, ensure_ascii=False)
    for t in TARGETS:
        c = txt.count(t)
        if c:
            A("  ** 数字 %s 出现 %d 次" % (t, c))
    A("")

op = os.path.join(OUTD, "probe_q02c_r4.txt")
open(op, "wb").write(("\n".join(R) + "\n").encode("utf-8"))
print("WROTE", op, len(R))
print("OK")
