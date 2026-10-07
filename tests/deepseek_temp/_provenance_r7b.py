# -*- coding: utf-8 -*-
"""R7b：归属判据细化 + deepseek memo 哈希差异排查。"""
import os, hashlib, re, json

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT  = os.path.join(ROOT, r"tests\deepseek_temp\_provenance_r7b.txt")
r = []
def add(s): r.append(s)

DSK = os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md")
G5  = os.path.join(ROOT, r"research\gpt5\docs\AGI_GPT5_MEMO.md")
raw = open(DSK, "rb").read()
dsk = raw.decode("utf-8-sig", "replace")
g5_raw = open(G5, "rb").read()
g5 = g5_raw.decode("utf-8-sig", "replace")

add("== A. deepseek memo 一致性 ==")
add("  bytes=%d  sha256=%s" % (len(raw), hashlib.sha256(raw).hexdigest()))
add("  sha8=%s  crlf=%d  bare_lf=%d  BOM=%s"
    % (hashlib.sha256(raw).hexdigest()[:8], dsk.count("\r\n"),
       dsk.count("\n") - dsk.count("\r\n"), raw.startswith(b"\xef\xbb\xbf")))
ph = re.findall(r"^## Phase\s*([0-9]+)", dsk, re.M)
add("  phase headings = %d -> %s" % (len(ph), ph))
add("  == 尾部 12 行 ==")
ls = dsk.split("\r\n")
for i, l in enumerate(ls[-12:], len(ls) - 11):
    add("  %5d|%s" % (i, l[:200]))
add("")

# 全部 ## 标题
add("  == 全部 '## ' 标题 ==") 
for i, l in enumerate(ls, 1):
    if l.startswith("## "):
        add("  %5d|%s" % (i, l[:110]))
add("")

add("== B. deepseek memo 中的 Phase 22–31 各节标题（用于定位引用）==")
add("")

# 每个候选文件：在哪一节被引用
add("== C. 逐文件：引用位置定位 ==")
CAND = ["ATLAS_LEDGER_SPEC.md", "card_set_v2.json", "cleanup_ledger_20260930.json",
        "ATLAS_PLAN_map_cracking_v2.md", "EMBED_ANCHOR_VERDICT_v1.md",
        "FINGERPRINT_PARADIGM_PLAN.md", "FIRST_PRINCIPLES_3090_3149.md",
        "MAIN_AXIS_VERDICT_v1.md", "MASTER_PLAN_map_linkage_v1.md",
        "MEMO_AUDIT_2750_3148.md", "PARADIGM_SHIFT_VERDICT_v1.md",
        "UNIFIED_REVIEW_ADJUDICATION_v1.md",
        "fingerprint_competition_review_20260921.md",
        "hdmcc_knowledge_map_review_20260921.md",
        "lpf_multiaxis_gating_roadmap_v1.md", "plan_v3_omega_dynamic_manifold.md",
        "plan_v4_micro_macro_merge.md", "plan_v5_dynamic_manifold_control.md",
        "plan_v6_reuse_topology.md", "research_synthesis_20260921.md",
        "atlas_ledger.json", "metric_dict_v1_backup.json"]

def phase_of(text, needle):
    """返回该 needle 出现在哪些 ## Phase 节（标题）下。"""
    hits = []
    cur = "(前言)"
    for l in text.split("\n"):
        if l.startswith("## "):
            cur = l[3:60]
        if needle in l:
            hits.append(cur)
    return sorted(set(hits))

for c in CAND:
    add("  --- %s ---" % c)
    d_ph = phase_of(dsk, c)
    g_ph = phase_of(g5, c)
    add("      deepseek memo: %d 处 in %s" % (dsk.count(c), d_ph[:6]))
    add("      gpt5 memo    : %d 处 in %s" % (g5.count(c), g_ph[:6]))
    add("      files-on-disk-refs (research/gpt5/docs|atlas):")
    for sub in [r"research\gpt5\docs", r"research\gpt5\atlas", r"research\gpt5\code"]:
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            fp = os.path.join(d, fn)
            if not os.path.isfile(fp) or not fn.endswith((".md", ".json")): continue
            try:
                t = open(fp, "rb").read().decode("utf-8-sig", "replace")
            except Exception:
                continue
            if c in t and fn != c:
                add("          %s" % os.path.join(sub, fn))
add("")

# '线：' 自声明
add("== D. '线' 自声明（各文档头部）==")
for fn in sorted(os.listdir(os.path.join(ROOT, r"research\gpt5\docs"))):
    if not fn.endswith(".md"): continue
    fp = os.path.join(ROOT, r"research\gpt5\docs", fn)
    t = open(fp, "rb").read().decode("utf-8-sig", "replace")
    decl = []
    for l in t.split("\n")[:20]:
        if re.search(r"(线[:：]|路线[:：]|归属|line[:：])", l) and len(l) < 220:
            decl.append(l.strip())
    if decl:
        add("  [%s]" % fn)
        for d in decl[:3]:
            add("      %s" % d)

txt = "\n".join(r)
open(OUT, "w", encoding="utf-8").write(txt)
print(txt[:200])
