# -*- coding: utf-8 -*-
"""R8 独立复核（新进程）：三处落盘 + 产物指纹。"""
import os, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT  = os.path.join(ROOT, r"tests\deepseek\result\_r8_indep_verify.txt")
rows = []
def sh(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
def txt(p): return open(p, "rb").read().decode("utf-8-sig")

M = os.path.join(ROOT, r".workbuddy\memory\MEMORY.md")
W = os.path.join(ROOT, r".workbuddy\memory\2026-10-03.md")
S = r"C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md"
mt, wt, st = txt(M), txt(W), txt(S)

rows.append("1) MEMORY  %s  chars=%d  (<4000=%s)" % (sh(M), len(mt), len(mt) < 4000))
for k in ["Phase 1–35", "150241da", "P35 = A 闸门 seal 执行", "A 闸门已关闭（R8）", "fired@readout", "42 教训", "只落补丁"]:
    rows.append("     %s %s" % ("OK " if k in mt else "NO!", k))
rows.append("2) WLOG    %s  chars=%d  R8block=%s  教训42=%s" % (sh(W), len(wt),
            "## R8：A 闸门 seal 执行与关闭" in wt, "教训 42" in wt))
rows.append("3) SKILL   %s  chars=%d  lesson42=%s  a-g=%s" % (sh(S), len(st),
            "42. **seal 落盘前必须做" in st,
            all(x in st for x in ["(a) **先取证归属再动手**", "(g) 收尾仍走五步"])))
rows.append("4) artifacts:")
for rel, lbl in [(r"research\deepseek\atlas\a_gate_closure_v1.json", "a_gate_closure_v1.json"),
                 (r"research\deepseek\atlas\ledger_corrections_v1.json", "ledger_corrections_v1.json"),
                 (r"research\deepseek\atlas\phase_queue_v1.json", "phase_queue_v1.json"),
                 (r"tests\deepseek\result\meta_single_source_v4.json", "meta_single_source_v4.json"),
                 (r"tests\deepseek\result\a_gate_closure_r8.html", "a_gate_closure_r8.html"),
                 (r"tests\deepseek\result\verify_r8.txt", "verify_r8.txt"),
                 (r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md", "AGI_DEEPSEEK_MEMO.md")]:
    p = os.path.join(ROOT, rel)
    rows.append("     %-30s %8d B  %s" % (lbl, os.path.getsize(p), sh(p)))
summ = [l for l in txt(os.path.join(ROOT, r"tests\deepseek\result\verify_r8.txt")).split("\n") if "汇总" in l]
rows.append("5) verify_r8 = %s" % (summ[0].strip() if summ else "?"))
rows.append("6) 跨线受保护（再确认）:")
for rel, exp in [(r"research\gpt5\docs\AGI_GPT5_MEMO.md", "2a84776b"),
                 (r"research\gpt5\atlas\atlas_ledger.json", "bbda63df"),
                 (r"tests\glm5\result\rdc_query_construction_20260913\phase3103\omega_p101_formula_audit\proposition_ledger.json", "9a3c6ff4")]:
    p = os.path.join(ROOT, rel); g = sh(p)
    rows.append("     %-8s exp=%s got=%s %s" % ("OK" if g == exp else "DRIFT!", exp, g, os.path.basename(rel)))

t = "\n".join(rows)
open(OUT, "w", encoding="utf-8").write(t)
print(t)
