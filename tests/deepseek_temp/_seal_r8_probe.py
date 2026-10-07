# -*- coding: utf-8 -*-
"""R8 探针：密封执行前的现场盘点（只读）。"""
import os, re, json, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT  = os.path.join(ROOT, r"tests\deepseek_temp\_seal_r8_probe.txt")
r = []
def add(s): r.append(s)
def sha8(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8] if os.path.exists(p) else "MISSING"

MEMO = os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md")
QUEUE = os.path.join(ROOT, r"research\deepseek\atlas\phase_queue_v1.json")
MDICT = os.path.join(ROOT, r"research\deepseek\atlas\metric_dict.json")
LEDGER_G = os.path.join(ROOT, r"research\gpt5\atlas\atlas_ledger.json")

# ---------- 1. memo ----------
raw = open(MEMO, "rb").read()
txt = raw.decode("utf-8-sig")
lf = txt.replace("\r\n", "\n")
ls = lf.split("\n")
add("=== 1) MEMO ===")
add("  path=%s" % MEMO)
add("  bytes=%d chars=%d lines=%d BOM=%s CRLF=%d bare_lf=%d sha8=%s"
    % (len(raw), len(txt), len(ls), raw.startswith(b"\xef\xbb\xbf"),
       txt.count("\r\n"), txt.count("\n") - txt.count("\r\n"), sha8(MEMO)))
add("  --- 所有 ## Phase 标题 ---")
for i, l in enumerate(ls, 1):
    if re.match(r"^## Phase", l):
        add("  L%5d %s" % (i, l[:110]))

# ---------- 2. queue ----------
add("")
add("=== 2) QUEUE %s sha8=%s exists=%s ===" % (os.path.basename(QUEUE), sha8(QUEUE), os.path.exists(QUEUE)))
if os.path.exists(QUEUE):
    d = json.loads(open(QUEUE, "rb").read().decode("utf-8-sig"))
    add("  top keys = %s" % list(d.keys()))
    def walk(o, pre=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (dict, list)):
                    add("  %s%s : %s(%d)" % (pre, k, type(v).__name__, len(v)))
                    walk(v, pre + "  ")
                else:
                    add("  %s%s = %r" % (pre, k, v))
        elif isinstance(o, list):
            for idx, it in enumerate(o[:60]):
                if isinstance(it, dict):
                    keys = list(it.keys())
                    add("  %s[%d] keys=%s" % (pre, idx, keys))
                    add("       %s" % json.dumps(it, ensure_ascii=False)[:400])
                else:
                    add("  %s[%d] %r" % (pre, idx, it))
    walk(d)

# ---------- 3. C1..C6 verbatim from memo ----------
add("")
add("=== 3) C1–C6 原文（memo 内检索）===")
pat = re.compile(r"(^|[^\w])C\s*[1-6]([^\d]|$)")
for i, l in enumerate(ls, 1):
    if pat.search(l) and re.search(r"(接受|拒绝|更正|更正表|seal|待|C1|C6)", l):
        add("  L%5d|%s" % (i, l[:400]))

# ---------- 4. meta_single_source_v4 ----------
add("")
add("=== 4) meta_single_source_v4.json ===")
for cand in [os.path.join(ROOT, r"tests\deepseek\result\meta_single_source_v4.json"),
             os.path.join(ROOT, r"research\deepseek\atlas\meta_single_source_v4.json"),
             os.path.join(ROOT, r"research\gpt5\atlas\meta_single_source_v4.json")]:
    add("  %s exists=%s sha8=%s" % (cand, os.path.exists(cand), sha8(cand)))
    if os.path.exists(cand):
        dd = json.loads(open(cand, "rb").read().decode("utf-8-sig"))
        add("    keys=%s" % list(dd.keys()))
        if "discrepancies" in dd:
            add("    discrepancies=%s" % json.dumps(dd["discrepancies"], ensure_ascii=False, indent=1)[:3000])

# ---------- 5. 目录清单 ----------
add("")
for sub in [r"research\deepseek\atlas", r"tests\deepseek\result", r"tests\deepseek", r"research\gpt5\atlas", r"research\gpt5\docs"]:
    p = os.path.join(ROOT, sub)
    add("=== DIR %s exists=%s ===" % (sub, os.path.isdir(p)))
    if os.path.isdir(p):
        for fn in sorted(os.listdir(p)):
            fp = os.path.join(p, fn)
            if os.path.isfile(fp):
                add("  f  %-58s %8d B  %s" % (fn, os.path.getsize(fp), sha8(fp)))
            else:
                add("  d  %s/" % fn)

# ---------- 6. ledger ----------
add("")
add("=== 6) atlas_ledger.json (G 线共享) ===")
if os.path.exists(LEDGER_G):
    lg = json.loads(open(LEDGER_G, "rb").read().decode("utf-8-sig"))
    add("  sha8=%s top keys=%s" % (sha8(LEDGER_G), list(lg.keys())))
    if "measurements" in lg:
        add("  n_measurements=%d" % len(lg["measurements"]))
        add("  last=%s" % json.dumps(lg["measurements"][-1], ensure_ascii=False)[:600])

open(OUT, "w", encoding="utf-8").write("\n".join(r))
print("WROTE %s  (%d lines)" % (OUT, len(r)))
