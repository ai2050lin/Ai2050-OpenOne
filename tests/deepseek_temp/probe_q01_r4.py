# -*- coding: utf-8 -*-
"""Q01 元层单一真源对账 —— 纯读盘取证（零 GPU，不改任何判定）。
产物：tests/deepseek_temp/_review/probe_q01_r4.txt
"""
import os, re, json, hashlib
from collections import Counter

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUTD = os.path.join(ROOT, "tests", "deepseek_temp", "_review")
os.makedirs(OUTD, exist_ok=True)
R = []
def A(s=""):
    R.append(str(s))
def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]
def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")

# ---------- [1] ledger 类文件清单 ----------
A("=== [1] ledger 类文件清单 ===")
found = []
for base in ["research", "tests/glm5"]:
    bp = os.path.join(ROOT, base)
    if not os.path.isdir(bp):
        continue
    for dp, dn, fns in os.walk(bp):
        if any(x in dp for x in [".venv", "node_modules", "__pycache__", ".git"]):
            continue
        for fn in fns:
            if "ledger" in fn.lower() and fn.endswith(".json"):
                p = os.path.join(dp, fn)
                b = open(p, "rb").read()
                found.append((rel(p), len(b), sha8(b)))
for r, n, h in sorted(found):
    A("  %-92s %8d B  %s" % (r, n, h))
A("  合计 %d 个" % len(found))

# ---------- [2] atlas_ledger 元字段 ----------
A("")
A("=== [2] atlas_ledger.json 元字段 ===")
lp = os.path.join(ROOT, "research/gpt5/atlas/atlas_ledger.json")
raw = open(lp, "rb").read()
A("  bytes     = %d" % len(raw))
A("  sha8(raw) = %s" % sha8(raw))
A("  has_crlf  = %s" % (b"\r\n" in raw))
A("  has_bom   = %s" % (raw[:3] == b"\xef\xbb\xbf"))
d = json.loads(raw.decode("utf-8-sig"))
A("  顶层键    = %s" % list(d.keys()))
for k, v in d.items():
    if isinstance(v, list):
        A("    %-26s list(%d)" % (k, len(v)))
    elif isinstance(v, dict):
        A("    %-26s dict %s" % (k, json.dumps(v, ensure_ascii=False)[:400]))
    else:
        A("    %-26s %r" % (k, v))
ms = d.get("measurements", [])
if ms:
    A("  measurements[0] keys = %s" % list(ms[0].keys()))
    A("  measurements 首条 = %s" % json.dumps(ms[0], ensure_ascii=False)[:400])
    A("  末三条:")
    for m in ms[-3:]:
        A("    %s" % json.dumps(m, ensure_ascii=False)[:400])
    ph = [m.get("phase", m.get("phase_num")) for m in ms]
    phs = [x for x in ph if x is not None]
    A("  measurements n=%d phase 范围 %s..%s" % (len(ph), min(phs) if phs else None, max(phs) if phs else None))
    # 找 N 线条目
    nline = [m for m in ms if "N" in json.dumps(m, ensure_ascii=False)[:200]]
    A("  含 N 线条目数（粗筛）= %d" % len(nline))

# ---------- [3] 自声明哈希语义复现 ----------
A("")
A("=== [3] 自声明 ledger_sha256_8 语义复现（目标 41d65a13） ===")
target = d.get("ledger_sha256_8")
A("  自声明 ledger_sha256_8 = %r" % target)
cands = {}
def add(name, bb):
    cands[name] = sha8(bb)
for ea in (True, False):
    for ind in (None, 1, 2):
        try:
            s = json.dumps(d, ensure_ascii=ea, indent=ind)
        except Exception:
            continue
        add("dumps(d,ea=%s,ind=%s)" % (ea, ind), s.encode("utf-8"))
        add("dumps(d,ea=%s,ind=%s)+CRLF" % (ea, ind), s.replace("\n", "\r\n").encode("utf-8"))
d2 = {k: v for k, v in d.items() if k != "ledger_sha256_8"}
for ea in (True, False):
    for ind in (None, 1, 2):
        add("dumps(d-minus-self,ea=%s,ind=%s)" % (ea, ind), json.dumps(d2, ensure_ascii=ea, indent=ind).encode("utf-8"))
        add("dumps(measurements,ea=%s,ind=%s)" % (ea, ind), json.dumps(ms, ensure_ascii=ea, indent=ind).encode("utf-8"))
add("raw", raw)
if raw[:3] == b"\xef\xbb\xbf":
    add("raw(strip BOM)", raw[3:])
hit = [k for k, v in cands.items() if v == target]
A("  候选数 = %d" % len(cands))
A("  命中目标 %r 的候选 = %s" % (target, hit if hit else "（无 -> 自声明哈希非当前内容语义，需另立 schema）"))
A("  部分候选抽样:")
for k in list(cands)[:12]:
    A("    %-46s %s" % (k, cands[k]))

# ---------- [4] proposition_ledger.json ----------
A("")
A("=== [4] proposition_ledger.json ===")
pls = [p for p in found if "proposition" in p[0].lower()]
A("  候选: %s" % ([p[0] for p in pls] if pls else "（未找到）"))
for r, n, h in pls:
    p = os.path.join(ROOT, r)
    b = open(p, "rb").read()
    try:
        pd = json.loads(b.decode("utf-8-sig"))
    except Exception as e:
        A("  parse err: %s" % e)
        continue
    A("  --- %s (%d B, sha8 %s) ---" % (r, n, h))
    if isinstance(pd, dict):
        for k, v in pd.items():
            if isinstance(v, list):
                A("    %-24s list(%d)" % (k, len(v)))
            elif isinstance(v, dict):
                A("    %-24s dict keys=%s" % (k, list(v.keys())[:12]))
            else:
                A("    %-24s %r" % (k, v))
        for k, v in pd.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                ks = set().union(*[set(x.keys()) for x in v[:80]])
                A("    [%s] 元素键 = %s" % (k, sorted(ks)))
                for gk in ["grade", "level", "tier", "rank", "grade_raw", "等级"]:
                    if gk in ks:
                        c = Counter(x.get(gk) for x in v)
                        A("    [%s] 按 %s 计数 = %s" % (k, gk, dict(c)))
                        break
    elif isinstance(pd, list):
        A("    顶层是 list(%d)" % len(pd))
        if pd and isinstance(pd[0], dict):
            A("    元素键 = %s" % sorted(set().union(*[set(x.keys()) for x in pd[:80]])))

# ---------- [5] MEMO 3103 节 ----------
A("")
A("=== [5] MEMO 3103 节（命题账本） ===")
mp = os.path.join(ROOT, "research/gpt5/docs/AGI_GPT5_MEMO.md")
mt = open(mp, "rb").read().decode("utf-8-sig")
ML = mt.split("\n")
idx = [i for i, l in enumerate(ML) if re.match(r"^##\s", l)]
pos = None
for j, i in enumerate(idx):
    if re.search(r"3103", ML[i]):
        pos = j
        break
if pos is not None:
    s = idx[pos]
    e = idx[pos + 1] if pos + 1 < len(idx) else len(ML)
    e = min(e, s + 130)
    A("  标题 = %s" % ML[s][:170])
    A("  行范围 = %d..%d" % (s + 1, e))
    for l in ML[s:e]:
        A("    |%s" % l[:220])
else:
    A("  未找到含 3103 的 ## 标题")

# ---------- [6] TESTPLAN 分级 ----------
A("")
A("=== [6] RDC_TESTPLAN_v1.md 命题账本/分级相关行 ===")
tp = os.path.join(ROOT, "research/gpt5/docs/RDC_TESTPLAN_v1.md")
if os.path.exists(tp):
    tb = open(tp, "rb").read()
    A("  bytes=%d sha8=%s" % (len(tb), sha8(tb)))
    tt = tb.decode("utf-8-sig")
    for i, l in enumerate(tt.split("\n"), 1):
        if re.search(r"\bB\s*=\s*\d|\bC\s*=\s*\d|命题账本|分级|A\+B|55\s*%|63\s*%|\bA\s*=\s*\d", l):
            A("  T%-5d %s" % (i, l[:210]))
else:
    A("  MISSING")

# ---------- 落盘 ----------
op = os.path.join(OUTD, "probe_q01_r4.txt")
open(op, "wb").write(("\n".join(R) + "\n").encode("utf-8"))
print("WROTE %s  lines=%d" % (op, len(R)))
print("OK")
