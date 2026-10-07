# -*- coding: utf-8 -*-
"""Q02 探针：定位三个 KPI 的数据来源与现状（零 GPU，纯读盘）。"""
import os, re, json, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUTD = os.path.join(ROOT, "tests", "deepseek_temp", "_review")
def sha8(b): return hashlib.sha256(b).hexdigest()[:8]
def rd(p):
    try: return open(os.path.join(ROOT, p), "rb").read()
    except Exception as e: return None
R = []
def A(s=""): R.append(str(s))

memo = rd("research/gpt5/docs/AGI_GPT5_MEMO.md").decode("utf-8-sig")
ML = memo.split("\n")

# ---- [1] KPI 名在 MEMO 中的出现次数 ----
A("=== [1] KPI 名在 gpt5 MEMO 中的出现次数 ===")
for k in ["E_read", "E_ar", "C_steer", "readout", "读出层", "B4"]:
    A("  %-10s %d" % (k, memo.count(k)))

# ---- [2] 读出层误差数字出现的行 ----
A("")
A("=== [2] 读出层误差数字 0.3316 / 0.3986 / 0.3898 出现处 ===")
for num in ["0.3316", "0.3986", "0.3898", "0.0461", "0.0079", "0.0807"]:
    hits = [(i + 1, l) for i, l in enumerate(ML) if num in l]
    A("  %s : %d 处" % (num, len(hits)))
    for i, l in hits[:4]:
        A("    L%-6d %s" % (i, l.strip()[:190]))

# ---- [3] 3152 / 3153 节 ----
A("")
A("=== [3] MEMO Phase 3152 / 3153 节 ===")
idx = [i for i, l in enumerate(ML) if re.match(r"^##\s", l)]
for want in ["3152", "3153"]:
    for j, i in enumerate(idx):
        if re.search(r"Phase\s+%s\b" % want, ML[i]) or re.search(r"^##\s+%s\b" % want, ML[i]):
            s = i; e = idx[j + 1] if j + 1 < len(idx) else len(ML)
            A("  --- %s (行 %d..%d, %d 行) ---" % (ML[s][:110], s + 1, e, e - s))
            for l in ML[s:min(e, s + 45)]:
                A("    |%s" % l[:210])
            break

# ---- [4] phase3152/3153 产物目录 ----
A("")
A("=== [4] tests/glm5/result 中含 3152 / 3153 的目录与 json ===")
base = os.path.join(ROOT, "tests", "glm5", "result")
for dp, dn, fns in os.walk(base):
    if any(x in dp for x in ["__pycache__", ".git"]):
        continue
    if re.search(r"315[23]", os.path.basename(dp)):
        A("  DIR %s" % os.path.relpath(dp, ROOT).replace("\\", "/"))
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            try:
                b = open(p, "rb").read()
                A("      %-56s %9d B  %s" % (fn, len(b), sha8(b)))
            except Exception as e:
                A("      %-56s ERR %s" % (fn, e))

# ---- [5] 找 held-out 面板 / readout 相关产物 ----
A("")
A("=== [5] 文件名含 heldout / holdout / panel / readout 的 json（限 gpt5 与 glm5 产物） ===")
cnt = 0
for base in [os.path.join(ROOT, "tests", "glm5", "result"), os.path.join(ROOT, "research", "gpt5")]:
    for dp, dn, fns in os.walk(base):
        if any(x in dp for x in ["__pycache__", ".git", "node_modules"]):
            continue
        for fn in fns:
            if fn.endswith(".json") and re.search(r"held|holdout|panel|readout|kpi", fn, re.I):
                p = os.path.join(dp, fn)
                try:
                    b = open(p, "rb").read()
                    A("  %-88s %8d B  %s" % (os.path.relpath(p, ROOT).replace("\\", "/"), len(b), sha8(b)))
                    cnt += 1
                except Exception:
                    pass
                if cnt > 40:
                    break
        if cnt > 40:
            break
    if cnt > 40:
        break
A("  合计 %d" % cnt)

# ---- [6] 已有 metric/kpi 类文件 ----
A("")
A("=== [6] 已存在 metric / kpi 类文件 ===")
for base in [os.path.join(ROOT, "research"), os.path.join(ROOT, "tests", "glm5")]:
    for dp, dn, fns in os.walk(base):
        if any(x in dp for x in ["__pycache__", ".git"]):
            continue
        for fn in fns:
            if re.search(r"metric|kpi", fn, re.I) and fn.endswith(".json"):
                p = os.path.join(dp, fn)
                b = open(p, "rb").read()
                A("  %-92s %8d B  %s" % (os.path.relpath(p, ROOT).replace("\\", "/"), len(b), sha8(b)))

op = os.path.join(OUTD, "probe_q02_r4.txt")
open(op, "wb").write(("\n".join(R) + "\n").encode("utf-8"))
print("WROTE", op, len(R))
print("OK")
