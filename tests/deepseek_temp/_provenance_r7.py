# -*- coding: utf-8 -*-
"""R7 侦察：确定 research/gpt5 下剩余文件的归属（deepseek 线 vs 其他 AI 线）。
判据（证据法，不凭记忆）：
  P1 该文件是否被 AGI_DEEPSEEK_MEMO.md 引用（文件名出现）
  P2 该文件是否被 AGI_GPT5_MEMO.md 引用
  P3 文件内容是否含 deepseek 线专属 token（Phase 3150/3151/3152/3153/3154、Q01..Q30、RDC_TESTPLAN 等）
  P4 mtime / 大小 / sha8
"""
import os, hashlib, json, time, re

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT  = os.path.join(ROOT, r"tests\deepseek_temp\_provenance_r7.txt")

def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
def stat(p):
    st = os.stat(p)
    return st.st_size, time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))

def tree(root, exts=None, maxdepth=2):
    rows = []
    for dp, dns, fns in os.walk(root):
        rel = os.path.relpath(dp, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > maxdepth:
            dns[:] = []
            continue
        for fn in sorted(fns):
            if exts and not fn.lower().endswith(tuple(exts)):
                continue
            fp = os.path.join(dp, fn)
            try:
                sz, mt = stat(fp)
            except Exception:
                sz, mt = -1, "ERR"
            rows.append((os.path.relpath(fp, ROOT), sz, mt, sha8(fp)))
    return sorted(rows)

r = []
def add(s): r.append(s)

# ---- 0. 两份 memo 的文本（用于引用扫描）----
DSK = os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md")
G5  = os.path.join(ROOT, r"research\gpt5\docs\AGI_GPT5_MEMO.md")
dsk_txt = open(DSK, "rb").read().decode("utf-8-sig", "replace") if os.path.exists(DSK) else ""
g5_txt  = open(G5,  "rb").read().decode("utf-8-sig", "replace") if os.path.exists(G5)  else ""
add("== memos ==")
add("  deepseek memo: %d B sha8=%s" % (len(dsk_txt.encode()), sha8(DSK)))
add("  gpt5 memo    : %d B sha8=%s" % (len(g5_txt.encode()), sha8(G5)))
add("")

# ---- 1. research/deepseek 现状 ----
add("== research/deepseek tree ==")
for p, sz, mt, h in tree(os.path.join(ROOT, "research", "deepseek"), maxdepth=3):
    add("  %-70s %8d B  %s  %s" % (p, sz, mt, h))
add("")

# ---- 2. research/gpt5 现状 ----
add("== research/gpt5 tree ==")
for p, sz, mt, h in tree(os.path.join(ROOT, "research", "gpt5"), maxdepth=3):
    add("  %-70s %8d B  %s  %s" % (p, sz, mt, h))
add("")

# ---- 3. tests/deepseek + tests/deepseek_temp + result 现状 ----
for sub in [r"tests\deepseek", r"tests\deepseek_temp", r"tests\deepseek\result"]:
    add("== %s ==" % sub)
    for p, sz, mt, h in tree(os.path.join(ROOT, sub), maxdepth=2):
        add("  %-78s %8d B  %s  %s" % (p, sz, mt, h))
    add("")

# ---- 4. 逐文件归属判据 ----
add("== provenance evidence for research/gpt5 remaining files ==")
DSK_TOKENS = re.compile(r"(Phase\s*31[0-9][0-9]|Phase\s*32[0-9][0-9]|Q0[1-9]\b|Q1[0-9]\b|Q2[0-9]\b|Q30\b|RDC_TESTPLAN|RDC_RESEARCH_CONSTITUTION|LOOP_DIAGNOSIS|deadline_dual_track|prop_citation_audit|seal_request|meta_single_source|metric_dict|rdc_query_construction|K1|K2\b|K3\b|死线)")
for p, sz, mt, h in tree(os.path.join(ROOT, "research", "gpt5"), exts=[".md", ".json"], maxdepth=3):
    if not (p.endswith(".md") or p.endswith(".json")):
        continue
    if p.endswith("AGI_GPT5_MEMO.md") or "AGI_GPT5_MEMO_" in p:
        tag = "G5-MEMO"
        add("  [%-22s] %s" % (tag, p)); continue
    fp = os.path.join(ROOT, p)
    try:
        t = open(fp, "rb").read().decode("utf-8-sig", "replace")
    except Exception:
        t = ""
    base = os.path.basename(p)
    in_dsk = base in dsk_txt
    in_g5  = base in g5_txt
    tok = sorted(set(DSK_TOKENS.findall(t)))[:8]
    # 是否在 deepseek 线自身的目录树里被别的文件引用（弱信号，跳过）
    add("  %s" % p)
    add("      size=%d mtime=%s sha8=%s" % (sz, mt, h))
    add("      in_deepseek_memo=%s  in_gpt5_memo=%s  n_dsk_tokens=%d %s"
        % (in_dsk, in_g5, len(tok), tok))
    add("      head: %s" % " / ".join([x.strip() for x in t.split("\n") if x.strip()][:4])[:220])
add("")

# ---- 5. gpt5/docs 里的 md 是否被 deepseek memo 的 Phase 23–31 正文提到（路径级）----
add("== references of 'research/gpt5' inside deepseek memo ==")
for m in sorted(set(re.findall(r"research[/\\]gpt5[/\\][A-Za-z0-9_./\\-]+", dsk_txt))):
    add("  %s" % m)
add("")

txt = "\n".join(r)
open(OUT, "w", encoding="utf-8").write(txt)
print(txt)
