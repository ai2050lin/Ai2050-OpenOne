# -*- coding: utf-8 -*-
"""Q02b：定位 E_read 的键名与值 + held-out 面板指纹（零 GPU，纯读盘）。"""
import os, re, json, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUTD = os.path.join(ROOT, "tests", "deepseek_temp", "_review")
def sha8(b): return hashlib.sha256(b).hexdigest()[:8]
def rd(p):
    with open(os.path.join(ROOT, p), "rb") as f: return f.read()
R = []
def A(s=""): R.append(str(s))

def walk(o, prefix="", out=None, depth=0, maxd=5):
    if out is None: out = []
    if depth > maxd: return out
    if isinstance(o, dict):
        for k, v in o.items():
            kk = (prefix + "." + str(k)) if prefix else str(k)
            if isinstance(v, dict):
                out.append((kk, "dict(%d)" % len(v))); walk(v, kk, out, depth + 1, maxd)
            elif isinstance(v, list):
                out.append((kk, "list(%d)" % len(v)))
                if v and isinstance(v[0], (dict, list)):
                    walk(v[0], kk + "[0]", out, depth + 1, maxd)
            else:
                out.append((kk, repr(v)[:70]))
    return out

BASE = "tests/glm5/result/rdc_query_construction_20260913/phase3152/g1p2_tri_model_k1"
TARGET = re.compile(r"readout|b4|k1|read_out|读出", re.I)

for arm in ["qwen3-4b", "qwen3-14b", "glm4k1", "summary"]:
    fn = "result_summary.json" if arm == "summary" else "result.json"
    p = "%s/%s/%s" % (BASE, arm, fn)
    try:
        b = rd(p)
    except Exception as e:
        A("  %-10s MISSING (%s)" % (arm, e)); continue
    A("=== %s -> %s  %d B  sha8=%s ===" % (arm, p.split("/")[-2] + "/" + fn, len(b), sha8(b)))
    try:
        d = json.loads(b.decode("utf-8-sig"))
    except Exception as e:
        A("  parse err %s" % e); continue
    rows = walk(d)
    A("  顶层键 = %s" % (list(d.keys())[:30] if isinstance(d, dict) else type(d)))
    sel = [(k, v) for k, v in rows if TARGET.search(k)]
    A("  匹配 readout/B4/k1 的路径（%d 条）:" % len(sel))
    for k, v in sel[:40]:
        A("     %-58s %s" % (k[:58], v))
    A("  --- 全部叶子键（前 60） ---")
    for k, v in rows[:60]:
        A("     %-58s %s" % (k[:58], v))
    A("")

# ---- 面板 / npz 指纹 ----
A("=== collect.npz 指纹（held-out 面板载体） ===")
for arm in ["qwen3-4b", "qwen3-14b"]:
    p = "%s/%s/collect.npz" % (BASE, arm)
    try:
        b = rd(p); A("  %-62s %10d B  sha8=%s" % (p.replace("tests/glm5/result/rdc_query_construction_20260913/", ""), len(b), sha8(b)))
    except Exception as e:
        A("  %s MISSING" % p)
p2 = "tests/glm5/result/rdc_query_construction_20260913/phase3151/g1p1_combo_additive_vs_interaction/collect.npz"
try:
    b = rd(p2); A("  %-62s %10d B  sha8=%s" % ("phase3151/.../collect.npz", len(b), sha8(b)))
except Exception as e:
    A("  phase3151 collect.npz MISSING")

op = os.path.join(OUTD, "probe_q02b_r4.txt")
open(op, "wb").write(("\n".join(R) + "\n").encode("utf-8"))
print("WROTE", op, len(R))
print("OK")
