# -*- coding: utf-8 -*-
"""R5 探针 C：dump 每个 id 命中的真实上下文（判假阳性）。只读。"""
import os, re, json

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, "tests", "deepseek_temp", "_review", "probe_q09q12c_r5.txt")
R = []
def A(s=""):
    R.append(s)

MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
t = open(MEMO, "rb").read().decode("utf-8-sig")
L = t.split("\n")
idx3103 = next(i for i, l in enumerate(L, 1) if l.startswith("## ") and "3103" in l)

def ctx(txt, pat, half=90):
    out = []
    for m in re.finditer(pat, txt):
        s = max(0, m.start() - half); e = min(len(txt), m.end() + half)
        out.append(txt[s:e].replace("\n", " | "))
    return out

docs_dir = os.path.join(ROOT, "research", "gpt5", "docs")
targets = {
 "MEMO>3103": "\n".join(L[idx3103:]),
}
for g in ["LOOP_DIAGNOSIS_AND_EXIT_v1.md", "MEMO_AUDIT_2750_3148.md", "RDC_TESTPLAN_v1.md",
          "RDC_RESEARCH_CONSTITUTION_v1.md", "UNIFIED_REVIEW_ADJUDICATION_v1.md", "PARADIGM_SHIFT_VERDICT_v1.md"]:
    fp = os.path.join(docs_dir, g)
    if os.path.exists(fp):
        targets[g] = open(fp, "rb").read().decode("utf-8-sig")

ids_to_check = ["R01","R10","R11","R44","R48","R51","R52","R53","R54","R55","R57","PA-01","PA-02","PA-03","PA-04","PA-05"]
for pid in ids_to_check:
    pat = r"(?<![A-Za-z0-9])%s(?![0-9])" % re.escape(pid)
    A("=" * 100)
    A("### id=%s" % pid)
    for name, txt in targets.items():
        cs = ctx(txt, pat)
        if cs:
            A("  -- %s (%d 次) --" % (name, len(cs)))
            for c in cs[:6]:
                A("     %s" % c[:230])
    A("")

open(OUT, "w", encoding="utf-8").write("\n".join(R))
print("WROTE", OUT, len(R), "lines")
