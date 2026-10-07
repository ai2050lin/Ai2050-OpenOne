# -*- coding: utf-8 -*-
"""R7 独立磁盘复核（新进程）：归属整理后的全量不变量。
判据全部独立重算，不读取生成器日志。
"""
import os, hashlib, re, json

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT  = os.path.join(ROOT, r"tests\deepseek\result\verify_r7.txt")
MEMO = os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md")
SNAP = os.path.join(ROOT, r"tests\deepseek_temp\_archive_r7\AGI_DEEPSEEK_MEMO.pre_r7.bin")
ARCH = os.path.join(ROOT, r"tests\deepseek_temp\_archive_r7\gpt5_docs")
G5D  = os.path.join(ROOT, r"research\gpt5\docs")
G5A  = os.path.join(ROOT, r"research\gpt5\atlas")
DSKA = os.path.join(ROOT, r"research\deepseek\atlas")

P, F, rows = 0, 0, []
def chk(cond, label, detail=""):
    global P, F
    ok = bool(cond)
    P, F = P + ok, F + (not ok)
    rows.append("%s  %s%s" % ("PASS" if ok else "FAIL", label, ("  | " + detail) if detail else ""))

def sha8(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
def sha256f(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

# ---- 1. memo 结构 ----
raw = open(MEMO, "rb").read()
txt = raw.decode("utf-8-sig")
chk(raw.startswith(b"\xef\xbb\xbf"), "memo BOM")
chk((txt.count("\n") - txt.count("\r\n")) == 0, "memo bare_lf == 0",
    "bare=%d" % (txt.count("\n") - txt.count("\r\n")))
chk(len(raw) == 676086, "memo bytes == 676086", "got %d" % len(raw))
chk(sha256f(MEMO)[:8] == "b257f54f", "memo sha8 == b257f54f", sha256f(MEMO)[:8])
ph = re.findall(r"^## Phase\s*([0-9]+)", txt.replace("\r\n", "\n"), re.M)
chk(ph == [str(i) for i in range(1, 35)], "Phase 标题 1..34 连续", "%d 个" % len(ph))

# ---- 2. 前缀（= R7 前快照）逐字节不变 ----
snap = open(SNAP, "rb").read()
chk(raw[:len(snap)] == snap, "memo 前缀逐字节 == R7 前快照",
    "snap=%d B/%s" % (len(snap), hashlib.sha256(snap).hexdigest()[:8]))
chk(len(raw) > len(snap), "memo 变长（append-only）", "+%d B" % (len(raw) - len(snap)))

# ---- 3. Phase 32–34 全文包含性（对照 R7 归档原件，独立降级重算）----
def demote(t):
    out, fence = [], False
    for l in t.split("\n"):
        if l.lstrip().startswith("```"): fence = not fence
        if not fence and re.match(r"^(#{1,5}) ", l): l = "#" + l
        out.append(l)
    return "\n".join(out)
lf = txt.replace("\r\n", "\n")
EXP = {"MAIN_AXIS_VERDICT_v1.md": "827fc48d", "EMBED_ANCHOR_VERDICT_v1.md": "792f9181",
       "MEMO_AUDIT_2750_3148.md": "5473f968"}
for fn, h in EXP.items():
    bp = os.path.join(ARCH, fn)
    chk(os.path.exists(bp), "备份存在 %s" % fn)
    chk(sha8(bp) == h, "备份哈希 %s" % fn, "%s vs %s" % (sha8(bp), h))
    d = open(bp, "rb").read().decode("utf-8-sig", "replace").replace("\r\n", "\n")
    lines = [x for x in demote(d).split("\n") if x.strip()]
    miss = [x for x in lines if x not in lf]
    chk(not miss, "包含性 %s" % fn, "%d/%d hit" % (len(lines) - len(miss), len(lines)))

# ---- 4. 原件已从 gpt5/docs 移除 ----
for fn in EXP:
    chk(not os.path.exists(os.path.join(G5D, fn)), "原件已移除 research/gpt5/docs/%s" % fn)

# ---- 5. JSON 迁移 ----
dstj = os.path.join(DSKA, "metric_dict_v1_backup.json")
chk(os.path.exists(dstj), "metric_dict_v1_backup.json 已在 deepseek/atlas")
chk(sha8(dstj) == "469c0ad1", "迁移后哈希一致", sha8(dstj))
chk(not os.path.exists(os.path.join(G5A, "metric_dict_v1_backup.json")), "源侧已移除")

# ---- 6. 其他线文件一律未触碰 ----
UNTOUCHED = {
    "research/gpt5/docs/AGI_GPT5_MEMO.md": "2a84776b",
    "research/gpt5/docs/AGI_GPT5_ICSPB.md": "47866afe",
    "research/gpt5/docs/ATLAS_PLAN_map_cracking_v2.md": "48806a5d",
    "research/gpt5/docs/MASTER_PLAN_map_linkage_v1.md": "34230a9d",
    "research/gpt5/docs/FINGERPRINT_PARADIGM_PLAN.md": "e905abaa",
    "research/gpt5/docs/fingerprint_competition_review_20260921.md": "8778e714",
    "research/gpt5/docs/hdmcc_knowledge_map_review_20260921.md": "8bdd0066",
    "research/gpt5/docs/lpf_multiaxis_gating_roadmap_v1.md": "afa60f18",
    "research/gpt5/docs/plan_v3_omega_dynamic_manifold.md": "56f1cc9d",
    "research/gpt5/docs/plan_v4_micro_macro_merge.md": "f91bbf87",
    "research/gpt5/docs/plan_v5_dynamic_manifold_control.md": "321fab42",
    "research/gpt5/docs/plan_v6_reuse_topology.md": "55b8c09c",
    "research/gpt5/docs/research_synthesis_20260921.md": "3c62b2ea",
    "research/gpt5/docs/FIRST_PRINCIPLES_3090_3149.md": "fce9c394",
    "research/gpt5/docs/PARADIGM_SHIFT_VERDICT_v1.md": "f9078b0b",
    "research/gpt5/docs/UNIFIED_REVIEW_ADJUDICATION_v1.md": "a4320a0a",
    "research/gpt5/atlas/atlas_ledger.json": "bbda63df",
    "research/gpt5/atlas/ATLAS_LEDGER_SPEC.md": "a3a5ffaf",
    "research/gpt5/atlas/card_set_v2.json": "73c3a4ae",
    "research/gpt5/atlas/cleanup_ledger_20260930.json": "6e5a22a2",
}
for rel, h in UNTOUCHED.items():
    fp = os.path.join(ROOT, rel.replace("/", os.sep))
    g = sha8(fp) if os.path.exists(fp) else "MISSING"
    chk(g == h, "未触碰 %s" % os.path.basename(rel), "%s" % g)

# ---- 7. deepseek 资产 ----
for rel, h in [("research/deepseek/atlas/metric_dict.json", "03887e51"),
               ("research/deepseek/atlas/phase_queue_v1.json", "675836fd")]:
    fp = os.path.join(ROOT, rel.replace("/", os.sep))
    chk(os.path.exists(fp) and sha8(fp) == h, "deepseek 资产 %s" % os.path.basename(rel))

summary = "RESULT: PASS=%d FAIL=%d %s" % (P, F, "ALL_PASS" if F == 0 else "HAS_FAILURE")
body = "\n".join(rows) + "\n\n" + summary + "\n"
open(OUT, "w", encoding="utf-8").write(body)
print(body)
