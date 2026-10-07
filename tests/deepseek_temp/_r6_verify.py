# -*- coding: utf-8 -*-
"""R6 独立复核：三处落盘 + MEMO 并入完整性（逐行包含性证明）+ 归位结果。"""
import os, re, hashlib, shutil, json

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, r"tests\deepseek_temp\_r6_verify.txt")
RES = os.path.join(ROOT, r"tests\deepseek\result")
rows = []
def A(s): rows.append(s)
def sh8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8] if os.path.exists(p) else "MISSING"
def rd(p):
    return open(p, "rb").read()

# 0) 先把本轮 4 个 R6 报告归入 result/
A("== 0) R6 报告归位 ==")
for fn in ["_merge_docs_r6_report.txt", "_reorg_r6_report.txt", "_patch_paths_r6_report.txt", "_skill40_r6_report.txt"]:
    src = os.path.join(ROOT, r"tests\deepseek_temp", fn); dst = os.path.join(RES, fn)
    if os.path.exists(src) and not os.path.exists(dst):
        h = sh8(src); shutil.move(src, dst)
        assert sh8(dst) == h
        A("  MOV %s -> result/  %s" % (fn, h))
    else:
        A("  SKIP %s (src=%s dst=%s)" % (fn, os.path.exists(src), os.path.exists(dst)))

# 1) 三处落盘
A("")
A("== 1) 记忆/工作日志/技能 ==")
M = os.path.join(ROOT, r".workbuddy\memory\MEMORY.md")
W = os.path.join(ROOT, r".workbuddy\memory\2026-10-03.md")
S = r"C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md"
mt = rd(M).decode("utf-8")
A("  MEMORY.md %s  chars=%d  (<3600 = %s)" % (sh8(M), len(mt), len(mt) < 3600))
for a in ["## 0 约定", "AGI_DEEPSEEK_MEMO.md", "AGI_GPT5_MEMO.md", "P23–30", "aeb4e786",
          "tests\\deepseek\\result", "fired_all_models", "model_specific", "RULE-UNDECIDABLE",
          "NS-COLLISION", "40 教训", "C1–C6"]:
    A("    %s : %s" % ("OK " if a in mt else "NO!", a))
wt = rd(W).decode("utf-8")
A("  2026-10-03.md %s  bytes=%d  R6块=%s  指纹=%s" % (sh8(W), len(rd(W)), "## R6：规范归一" in wt,
    all(k in wt for k in ["aeb4e786", "71b85673", "7678cbe0", "03887e51"])))
st = rd(S).decode("utf-8-sig")
A("  SKILL.md %s  chars=%d  L39=%s  L40=%s" % (sh8(S), len(st), "39. **一次脚本多处写入" in st, "40. **跨目录归位" in st))

# 2) MEMO
A("")
A("== 2) AGI_DEEPSEEK_MEMO.md ==")
P = os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md")
raw = rd(P); t = raw.decode("utf-8-sig")
bare = t.count("\n") - t.count("\r\n")
ph = re.findall(r"^## Phase (\d+):", t, re.M)
A("  bytes=%d lines=%d sha8=%s bom=%s bare_lf=%d" % (len(raw), t.count("\n") + 1, sh8(P), raw.startswith(b"\xef\xbb\xbf"), bare))
A("  Phase 标题 = %d 个 %s" % (len(ph), ph))
A("  Phase22 已规范化 = %s ; 旧标记残留 = %d" % (
    "## Phase 22: A 闸门 R3–R5b：循环诊断 → 宪法/冻结队列 → Q01/Q02 → Q09/Q12 → seal 请求包 [2026-10-03 02:20]" in t,
    t.count("（非 Phase 编号）")))
assert bare == 0 and raw.startswith(b"\xef\xbb\xbf")

# 3) 全文并入的逐行包含性证明
A("")
A("== 3) 8 件文档全文包含性（逐行，标题降一级后必须逐字出现）==")
BK = os.path.join(ROOT, r"tests\deepseek_temp\_archive_r6\gpt5_docs")
def demote(ln):
    m = re.match(r"^(\s*)(#{1,6})(\s.*)$", ln)
    return (m.group(1) + "#" + m.group(2) + m.group(3)) if (m and len(m.group(2)) < 6) else ln
DOCS = ["RDC_TESTPLAN_v1.md", "LOOP_DIAGNOSIS_AND_EXIT_v1.md", "RDC_RESEARCH_CONSTITUTION_v1.md",
        "META_SINGLE_SOURCE_Q01.md", "METRIC_DICT_Q02.md", "DEADLINE_DUAL_TRACK_Q09.md",
        "PROP_CITATION_AUDIT_Q12.md", "SEAL_REQUEST_A_GATE.md"]
allok = True
for fn in DOCS:
    src = os.path.join(BK, fn)
    assert os.path.exists(src), "backup missing %s" % fn
    lines = rd(src).decode("utf-8-sig").replace("\r\n", "\n").split("\n")
    body = [l for l in lines if l.strip()]
    # 跳过文档自身一级标题
    ti = next((i for i, l in enumerate(lines) if l.startswith("# ")), None)
    body = [l for i, l in enumerate(lines) if l.strip() and i != ti]
    miss = [l for l in body if demote(l) not in t]
    frac = (len(body) - len(miss)) / float(len(body))
    allok = allok and not miss
    A("  %-32s lines=%3d  命中率=%.4f  未命中=%d %s" % (fn, len(body), frac, len(miss), (miss[:2] if miss else "")))
A("  全文并入完整 = %s" % allok)

# 4) 归位结果
A("")
A("== 4) 归位 ==")
GD = os.path.join(ROOT, r"research\gpt5\docs"); GA = os.path.join(ROOT, r"research\gpt5\atlas")
A("  gpt5/docs 8 件已删 = %s" % (not any(os.path.exists(os.path.join(GD, f)) for f in DOCS)))
A("  gpt5/atlas 6 件已迁 = %s" % (not any(os.path.exists(os.path.join(GA, f)) for f in
    ["deadline_dual_track_v1.json", "prop_citation_audit_v1.json", "meta_single_source_v4.json",
     "metric_dict.json", "phase_queue_v1.json", "seal_request_v1.json"])))
for rel, exp in [(r"research\deepseek\atlas\metric_dict.json", "03887e51"),
                 (r"research\deepseek\atlas\phase_queue_v1.json", "675836fd"),
                 (r"tests\deepseek\result\deadline_dual_track_v1.json", "4d1853d3"),
                 (r"tests\deepseek\result\prop_citation_audit_v1.json", "4c2ea9d2"),
                 (r"tests\deepseek\result\meta_single_source_v4.json", "f9d7ede6"),
                 (r"tests\deepseek\result\seal_request_v1.json", "d871a4b4")]:
    fp = os.path.join(ROOT, rel); g = sh8(fp)
    A("  %-8s exp=%s got=%s  %s" % ("OK" if g == exp else "DRIFT!", exp, g, rel))
A("  backup 8 件 = %d ; _review 目录已消失 = %s" % (
    len([f for f in os.listdir(BK) if f.endswith(".md")]),
    (not os.path.isdir(os.path.join(ROOT, r"tests\deepseek\_review"))) and (not os.path.isdir(os.path.join(ROOT, r"tests\deepseek_temp\_review")))))
nd = len([f for f in os.listdir(os.path.join(ROOT, r"tests\deepseek")) if f.endswith(".py") and os.path.isfile(os.path.join(ROOT, r"tests\deepseek", f))])
nt = len([f for f in os.listdir(os.path.join(ROOT, r"tests\deepseek_temp")) if f.endswith(".py") and os.path.isfile(os.path.join(ROOT, r"tests\deepseek_temp", f))])
A("  tests/deepseek/*.py=%d  tests/deepseek_temp/*.py=%d  tests/deepseek/result 条目=%d" % (nd, nt, len(os.listdir(RES))))
# 死路径复核
bad = []
for d in [os.path.join(ROOT, r"tests\deepseek"), os.path.join(ROOT, r"tests\deepseek_temp")]:
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if f.endswith(".py") and os.path.isfile(fp):
            x = rd(fp).decode("utf-8")
            if "deepseek_temp/_review" in x or "deepseek_temp\\_review" in x or "deepseek/_review/" in x:
                bad.append(f)
A("  死路径残留（全部脚本，含本轮临时脚本）= %s" % bad)

txt = "\n".join(rows)
open(OUT, "w", encoding="utf-8").write(txt)
print(txt)
print("VERIFY_DONE")
