# -*- coding: utf-8 -*-
import os, hashlib
ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, r"tests\deepseek_temp\_tail_r6.txt")
r = []
def add(s): r.append(s)

P = os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md")
raw = open(P, "rb").read()
txt = raw.decode("utf-8-sig")
ls = txt.split("\n")
add("MEMO lines=%d bytes=%d sha8=%s" % (len(ls), len(raw), hashlib.sha256(raw).hexdigest()[:8]))
add("== tail from L4765 ==")
for i, l in enumerate(ls[4764:], 4765):
    add("%6d|%s" % (i, l))
add("")

add("== 8 docs metadata ==")
DOCS = os.path.join(ROOT, r"research\gpt5\docs")
names = ["RDC_TESTPLAN_v1.md", "RDC_RESEARCH_CONSTITUTION_v1.md", "META_SINGLE_SOURCE_Q01.md",
         "METRIC_DICT_Q02.md", "LOOP_DIAGNOSIS_AND_EXIT_v1.md", "DEADLINE_DUAL_TRACK_Q09.md",
         "PROP_CITATION_AUDIT_Q12.md", "SEAL_REQUEST_A_GATE.md"]
for n in names:
    fp = os.path.join(DOCS, n)
    if not os.path.exists(fp):
        add("  MISSING %s" % n); continue
    b = open(fp, "rb").read()
    t = b.decode("utf-8-sig")
    lf = t.count("\n"); crlf = t.count("\r\n")
    hd = [x for x in t.split("\n") if x.startswith("#")]
    add("  %-34s %7d B sha8=%s bom=%s crlf=%d bare_lf=%d h=%d" % (
        n, len(b), hashlib.sha256(b).hexdigest()[:8], b.startswith(b"\xef\xbb\xbf"), crlf, lf - crlf, len(hd)))
    for h in hd[:14]:
        add("        %s" % h[:110])
add("")

add("== json artifacts metadata ==")
ATL = os.path.join(ROOT, r"research\gpt5\atlas")
for n in ["deadline_dual_track_v1.json", "prop_citation_audit_v1.json", "meta_single_source_v4.json",
          "metric_dict.json", "phase_queue_v1.json", "seal_request_v1.json"]:
    fp = os.path.join(ATL, n)
    if os.path.exists(fp):
        b = open(fp, "rb").read()
        add("  %-34s %7d B sha8=%s" % (n, len(b), hashlib.sha256(b).hexdigest()[:8]))
    else:
        add("  MISSING %s" % n)

txt = "\n".join(r)
open(OUT, "w", encoding="utf-8").write(txt)
print("WROTE chars=%d" % len(txt))
