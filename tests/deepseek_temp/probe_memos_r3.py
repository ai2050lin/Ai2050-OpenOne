# -*- coding: utf-8 -*-
"""复核轮 R3：探针 gpt5 / deepseek 两份 MEMO 的结构与规模，输出报告文件。
非 Phase 轮：脚本落 tests/deepseek/_review/，报告落 tests/deepseek_temp/_review/。
"""
import hashlib
import os
import re

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, "tests", "deepseek_temp", "_review", "probe_memos_r3.txt")

FILES = [
    "research/gpt5/docs/AGI_GPT5_MEMO.md",
    "research/gpt5/docs/AGI_GPT5_MEMO_SUMMARY.md",
    "research/gpt5/docs/AGI_GPT5_MEMO_VISUAL.md",
    "research/gpt5/docs/MEMO_AUDIT_2750_3148.md",
    "research/gpt5/docs/UNIFIED_REVIEW_ADJUDICATION_v1.md",
    "research/gpt5/docs/FIRST_PRINCIPLES_3090_3149.md",
    "research/gpt5/docs/research_synthesis_20260921.md",
    "research/gpt5/docs/PARADIGM_SHIFT_VERDICT_v1.md",
    "research/gpt5/docs/MAIN_AXIS_VERDICT_v1.md",
    "research/gpt5/docs/RDC_TESTPLAN_v1.md",
    "research/gpt5/docs/plan_v6_reuse_topology.md",
    "research/gpt5/docs/FINGERPRINT_PARADIGM_PLAN.md",
    "research/deepseek/docs/AGI_DEEPSEEK_MEMO.md",
    "research/glm5/docs/AGI_GLM5_MEMO_SUMMARY.md",
    "research/glm5/docs/EVALUATION_AND_PLAN_V3_20260916.md",
    "research/glm5/docs/RESEARCH_REVIEW_20260914.md",
]

L = []
L.append("=" * 100)
L.append("R3 探针：MEMO 结构/规模  (只读，不改任何文件)")
L.append("=" * 100)

for rel in FILES:
    p = os.path.join(ROOT, rel.replace("/", os.sep))
    if not os.path.exists(p):
        L.append("%-56s MISSING" % rel)
        continue
    b = open(p, "rb").read()
    t = b.decode("utf-8-sig", errors="replace")
    lines = t.split("\n")
    h2 = [x for x in lines if x.startswith("## ")]
    h3 = [x for x in lines if x.startswith("### ")]
    L.append("%-56s %9d B %6d lines sha8=%s BOM=%s h2=%3d h3=%4d"
             % (rel, len(b), len(lines), hashlib.sha256(b).hexdigest()[:8],
                b[:3] == b"\xef\xbb\xbf", len(h2), len(h3)))

L.append("")
L.append("-" * 100)
L.append("[A] gpt5 MEMO 全部 ## 标题（按出现顺序，带行号）")
L.append("-" * 100)
p = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
t = open(p, "rb").read().decode("utf-8-sig", errors="replace")
lines = t.split("\n")
for i, ln in enumerate(lines, 1):
    if ln.startswith("## "):
        L.append("%7d  %s" % (i, ln[:150]))

L.append("")
L.append("-" * 100)
L.append("[B] gpt5 MEMO 中 'Phase NNNN' 编号分布（按百位分桶）")
L.append("-" * 100)
ph = re.findall(r"Phase\s+(\d{2,5})", t)
nums = sorted({int(x) for x in ph})
L.append("distinct phase numbers: %d ; min=%s max=%s"
         % (len(nums), nums[0] if nums else "-", nums[-1] if nums else "-"))
buckets = {}
for n in nums:
    buckets.setdefault(n // 100 * 100, []).append(n)
for k in sorted(buckets):
    v = buckets[k]
    L.append("  %5d-%5d : %4d 个 (%d..%d)" % (k, k + 99, len(v), v[0], v[-1]))

L.append("")
L.append("-" * 100)
L.append("[C] 判决/结论类关键词计数（gpt5 MEMO）")
L.append("-" * 100)
KW = ["判决", "封存", "已证实", "已确立", "否证", "反例", "FAIL", "PASS",
      "预注册", "不可证伪", "post-hoc", "事后", "循环论证", "同义反复",
      "局限", "限界", "TODO", "待办", "悬置", "降级", "撤回", "纠错"]
for k in KW:
    L.append("  %-12s %5d" % (k, t.count(k)))

L.append("")
L.append("-" * 100)
L.append("[D] 各 ## 节的行数（判断哪一节在膨胀）")
L.append("-" * 100)
idx = [i for i, ln in enumerate(lines) if ln.startswith("## ")]
for j, s in enumerate(idx):
    e = idx[j + 1] if j + 1 < len(idx) else len(lines)
    L.append("%7d  %6d lines  %s" % (s + 1, e - s, lines[s][:120]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("wrote", OUT, len(L), "lines")
