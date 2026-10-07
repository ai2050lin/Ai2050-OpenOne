# -*- coding: utf-8 -*-
"""R5 探针 B：为 Q09（K2/K3 可测量性）与 Q12（D/E 引用位点）取证。只读。"""
import os, re, json, hashlib, glob

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUTD = os.path.join(ROOT, "tests", "deepseek_temp", "_review")
os.makedirs(OUTD, exist_ok=True)
OUT = os.path.join(OUTD, "probe_q09q12b_r5.txt")
R = []
def A(s=""):
    R.append(s)

MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
t = open(MEMO, "rb").read().decode("utf-8-sig")
L = t.split("\n")

# ---------- [D] K2/K3 现有可测量性 ----------
A("=== [D] K2/K3 现有可测量性 ===")
for k in ["top-50", "top50", "覆盖率", "cover", "可分离", "不可分离", "phi_l", "φ_ℓ", "条件门"]:
    A("  %-14s %d" % (k, t.count(k)))
A("")
A("--- 含 '覆盖率' 且含 'top' 的行 ---")
n = 0
for i, l in enumerate(L, 1):
    if ("覆盖率" in l or "cover" in l) and ("top" in l or "50" in l):
        A("  L%-6d %s" % (i, l.strip()[:180]))
        n += 1
        if n >= 20:
            break
A("")
A("--- 3154 是否已运行（找 phase3154 目录/结果） ---")
for pat in ["tests/glm5/result/rdc_query_construction_20260913/phase3154*",
            "tests/glm5/result/**/phase315*"]:
    hits = glob.glob(os.path.join(ROOT, pat), recursive=True)
    A("  %s -> %d" % (pat, len(hits)))
    for h in hits[:12]:
        A("     %s" % os.path.relpath(h, ROOT))

# ---------- [E] 命题账本全表 + 引用位点 ----------
A("")
A("=== [E] 命题账本全表与引用位点 ===")
led_path = glob.glob(os.path.join(ROOT, "tests", "glm5", "result", "**", "proposition_ledger.json"), recursive=True)[0]
led = json.loads(open(led_path, "rb").read().decode("utf-8-sig"))

def cmp_split(g):
    parts = [x.strip() for x in re.split(r"[+/、,\s]+", str(g or "")) if x.strip()]
    return parts or ["?"]

props = []
for key in ["propositions_review", "propositions_new"]:
    for p in (led.get(key) or []):
        q = dict(p)
        q["_src"] = key
        q["_comp"] = cmp_split(p.get("grade"))
        props.append(q)
A("总条数=%d" % len(props))

de = [p for p in props if ("D" in p["_comp"]) or ("E" in p["_comp"])]
A("含 D 或 E 分量（并集）条数=%d" % len(de))
A("  其中 纯D/E（无A/B/C）=%d" % len([p for p in de if not (set(p["_comp"]) & {"A", "B", "C"})]))

# 引用位点扫描：按 id 精确匹配
A("")
A("--- 按 id 精确匹配的引用位点（扫描范围=3103 之后的 MEMO 行） ---")
idx3103 = None
for i, l in enumerate(L, 1):
    if l.startswith("## ") and "3103" in l:
        idx3103 = i
        break
tail = "\n".join(L[idx3103:])
docs_dir = os.path.join(ROOT, "research", "gpt5", "docs")
gov = ["RDC_TESTPLAN_v1.md", "RDC_RESEARCH_CONSTITUTION_v1.md", "LOOP_DIAGNOSIS_AND_EXIT_v1.md",
       "UNIFIED_REVIEW_ADJUDICATION_v1.md", "PARADIGM_SHIFT_VERDICT_v1.md", "META_SINGLE_SOURCE_Q01.md",
       "METRIC_DICT_Q02.md", "MEMO_AUDIT_2750_3148.md", "FIRST_PRINCIPLES_3090_3149.md"]
govtext = {}
for g in gov:
    fp = os.path.join(docs_dir, g)
    if os.path.exists(fp):
        govtext[g] = open(fp, "rb").read().decode("utf-8-sig")

idpat = re.compile(r"\b(PA-\d{2}|R\d{2})\b")
# 先列出所有 id，确认哪些 id 在账本内
ids = [str(p.get("id")) for p in props]
A("账本 id: %s" % ", ".join(ids))
A("")
cited = []
for p in de:
    pid = str(p.get("id"))
    sites = []
    if re.search(r"(?<![A-Za-z0-9])%s(?![0-9])" % re.escape(pid), tail):
        sites.append("MEMO>3103")
    for g, gt in govtext.items():
        if re.search(r"(?<![A-Za-z0-9])%s(?![0-9])" % re.escape(pid), gt):
            sites.append(g)
    if sites:
        cited.append((pid, p.get("grade"), p.get("claim", "")[:60], sites))
A("--- D/E 级命题中，其 id 在「新推理链」文本中被引用的（%d 条） ---" % len(cited))
for pid, gr, cl, s in cited:
    A("   [%s] %-6s %s  << %s" % (pid, gr, cl, ", ".join(s)))
if not cited:
    A("   （无：D/E 级命题的 id 在 3103 之后文本中零次出现）")

A("")
A("--- 交叉核对：所有 id（含 A/B/C 级）在各文本中的出现次数 ---")
def cnt(txt, pid):
    return len(re.findall(r"(?<![A-Za-z0-9])%s(?![0-9])" % re.escape(pid), txt))
A("  id    MEMO>3103  " + "  ".join("%-28s" % g[:26] for g in govtext))
for p in props:
    pid = str(p.get("id"))
    row = "  %-6s %8d  " % (pid, cnt(tail, pid))
    row += "  ".join("%-28d" % cnt(gt, pid) for gt in govtext.values())
    A(row)

# consumer_check
A("")
A("--- consumer_check 内容 ---")
A(json.dumps(led.get("consumer_check"), ensure_ascii=False, indent=1)[:2500])

# 导出全表供生成器使用
exp = os.path.join(OUTD, "prop_ledger_flat_r5.json")
json.dump({"source": os.path.relpath(led_path, ROOT),
           "sha8": hashlib.sha256(open(led_path, "rb").read()).hexdigest()[:8],
           "n": len(props),
           "props": [{"id": p.get("id"), "grade": p.get("grade"), "comp": p["_comp"],
                      "claim": p.get("claim", ""), "phases": p.get("phases", ""),
                      "review_verdict": p.get("review_verdict", ""), "retain": p.get("retain", ""),
                      "src": p["_src"]} for p in props],
           "consumer_check": led.get("consumer_check")},
          open(exp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
A("")
A("EXPORTED %s" % os.path.relpath(exp, ROOT))

open(OUT, "w", encoding="utf-8").write("\n".join(R))
print("WROTE", OUT, len(R), "lines")
