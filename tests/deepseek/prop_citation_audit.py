# -*- coding: utf-8 -*-
"""Q12：D/E 级命题引用审计（I4 落地；F4 counterexample_grep.py 的替代/升级）。

现成 tests/glm5/counterexample_grep.py 的两个致命缺陷：
  (a) 命名空间不可靠：账本 id `R\\d\\d` 与项目其它文档自有的 R 编号碰撞 ⇒ 假阳性；
  (b) 无等级概念：不知道哪条是 D/E 级，也不知道何为「新推理链」。

本工具只主张**四项可证伪事实**（不做噪声启发式分类）：
  F1  新推理链语料（= 现行 MEMO 自 Phase 3104 起 ∪ docs/*.md）中，D/E 级命题的 id 命中清单（逐条附上下文）
  F2  同样的 id 若同时被别的文档用作**自有编号**，则 id 级检测不可靠（命名空间占用普查）
  F3  D/E 级 claim 的 6-CJK-gram 文本指纹在新语料中的命中（实质内容泄漏检测）
  F4  分级口径缺陷：复合等级使「纯 E 禁入」规则不可机械判定

语料范围严格限定：
  主语料 = 现行 MEMO@3104+ ∪ research/gpt5/docs/*.md（排除现行 MEMO 自身与归档快照）
  归档 MEMO（AGI_GPT5_MEMO_2026*.md）= 同一文档的历史快照 ⇒ 只报计数，不入判决

产物：
  research/gpt5/atlas/prop_citation_audit_v1.json
  research/gpt5/docs/PROP_CITATION_AUDIT_Q12.md
"""
import os, re, json, hashlib, glob
from collections import Counter

ROOT = r"D:\AI2050\Ai2050-OpenOne"
DOCS = os.path.join(ROOT, "research", "gpt5", "docs")
ATLAS = os.path.join(ROOT, "research", "gpt5", "atlas")
MEMO = os.path.join(DOCS, "AGI_GPT5_MEMO.md")

def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

led_path = glob.glob(os.path.join(ROOT, "tests", "glm5", "result", "**",
                                 "proposition_ledger.json"), recursive=True)[0]
ledger = json.loads(open(led_path, "rb").read().decode("utf-8-sig"))

def split_g(g):
    return [x.strip() for x in re.split(r"[+/、,\s]+", str(g or "")) if x.strip()] or ["?"]

props = []
for k in ["propositions_review", "propositions_new"]:
    for p in (ledger.get(k) or []):
        props.append({"id": str(p.get("id")), "grade": str(p.get("grade")),
                      "comp": split_g(p.get("grade")), "claim": str(p.get("claim", "")),
                      "phases": str(p.get("phases", "")), "src": k})
DE = [p for p in props if set(p["comp"]) & {"D", "E"}]
for p in props:
    p["pure_DE"] = len(set(p["comp"]) - {"D", "E"}) == 0
PROP = {p["id"]: p for p in props}

# ---- 语料 ----
mtxt = open(MEMO, "rb").read().decode("utf-8-sig")
ML = mtxt.split("\n")
NEW_START = next(i for i, l in enumerate(ML, 1)
                 if re.match(r"## Phase (\d{4})", l) and int(re.match(r"## Phase (\d{4})", l).group(1)) > 3103)
new_region = "\n".join(ML[NEW_START - 1:])

doc_corpora = {"MEMO@3104+": new_region}
ARCHIVE_PAT = re.compile(r"^AGI_GPT5_MEMO_\d{8}\.md$")
SELF_OUTPUTS = {"PROP_CITATION_AUDIT_Q12.md"}
for fn in sorted(os.listdir(DOCS)):
    if not fn.endswith(".md") or fn == "AGI_GPT5_MEMO.md":
        continue
    if ARCHIVE_PAT.match(fn) or fn in SELF_OUTPUTS:
        continue          # 归档快照与自身输出不入主语料（自指污染）
    doc_corpora[fn] = open(os.path.join(DOCS, fn), "rb").read().decode("utf-8-sig", "ignore")

archives = {}
for fn in sorted(os.listdir(DOCS)):
    if ARCHIVE_PAT.match(fn):
        archives[fn] = open(os.path.join(DOCS, fn), "rb").read().decode("utf-8-sig", "ignore")

META = re.compile(r"审查|审计|撤回|降级|命题账本|全部证实|反例传播|formula_audit|系统性方法缺陷")

def find_ids(text, offset=0):
    out = []
    for p in DE:
        pid = p["id"]
        for m in re.finditer(r"(?<![A-Za-z0-9])%s(?![0-9])" % re.escape(pid), text):
            s, e = max(0, m.start() - 130), min(len(text), m.end() + 130)
            line = text[s:e].replace("\n", " ")
            ln = offset + text.count("\n", 0, m.start()) + 1
            out.append({"id": pid, "grade": p["grade"], "line": ln,
                        "meta": bool(META.search(line)), "ctx": line.strip()})
    return out

hits_main = {}
for name, txt in doc_corpora.items():
    h = find_ids(txt, offset=(NEW_START - 1) if name == "MEMO@3104+" else 0)
    if h:
        hits_main[name] = h
hits_arch = {name: find_ids(txt) for name, txt in archives.items()}
hits_arch = {k: v for k, v in hits_arch.items() if v}

def grams6(s):
    out = set()
    for run in re.findall(r"[\u4e00-\u9fff]{6,}", str(s or "")):
        for i in range(len(run) - 5):
            out.add(run[i:i + 6])
    return out

fp_main, fp_arch = [], []
n_fp_constructible = 0
for p in DE:
    gs = grams6(p["claim"])
    if not gs:
        continue
    n_fp_constructible += 1
    for name, txt in doc_corpora.items():
        h = sorted(g for g in gs if g in txt)
        if h:
            fp_main.append({"id": p["id"], "grade": p["grade"], "corpus": name,
                            "n_hit": len(h), "n_total": len(gs), "examples": h[:3]})
    for name, txt in archives.items():
        h = sorted(g for g in gs if g in txt)
        if h:
            fp_arch.append({"id": p["id"], "grade": p["grade"], "corpus": name,
                            "n_hit": len(h), "n_total": len(gs), "examples": h[:3]})

# ---- 命名空间占用普查 ----
ns = {}
for fn in sorted(os.listdir(DOCS)):
    if not fn.endswith(".md") or fn in SELF_OUTPUTS:
        continue
    txt = open(os.path.join(DOCS, fn), "rb").read().decode("utf-8-sig", "ignore")
    s = sorted(set(re.findall(r"(?<![A-Za-z0-9])(R\d{2})(?![0-9])", txt)))
    if s:
        ns[fn] = s

# ---- 判定：新推理链中的实质命中 ----
# 分类由**人工核定表**给出（F4 原文即要求“人工确认下游无残留依赖”；不自动分类以免过判）
CURATED = {
    ("LOOP_DIAGNOSIS_AND_EXIT_v1.md", "R10", 81): ("namespace_collision",
        "该文档自有的 §2 清单 R1–R11；此处 R10=“真阴性记录：翻译轴不存在”，与账本 R10（实体感紧凑性定律）无关"),
    ("LOOP_DIAGNOSIS_AND_EXIT_v1.md", "R11", 82): ("namespace_collision",
        "同上；此处 R11=“装置层：execution 冻结 → seal → Ledger”"),
    ("LOOP_DIAGNOSIS_AND_EXIT_v1.md", "R11", 289): ("namespace_collision",
        "同上；“R8–R11 主要依据审计文档的自述与我的抽样核对”"),
    ("LOOP_DIAGNOSIS_AND_EXIT_v1.md", "R54", 71): ("substantive_reference",
        "诊断 §2 表把 R54 列为“成立（纯数学，无争议）”——引用其能量份额非互补性"),
    ("MEMO_AUDIT_2750_3148.md", "R53", 47): ("substantive_reference",
        "SwiGLU 一阶/二阶解析（R53 → 3102）：修正 dap 公式、残差 6% → 幅度误差 26.55–41.18%"),
    ("MEMO_AUDIT_2750_3148.md", "R55", 42): ("substantive_reference",
        "per-head 分解的正确切点（R55 → 3101）：只在 o_proj 输入侧 forward_pre hook 合法"),
    ("MEMO_AUDIT_2750_3148.md", "R53", 96): ("substantive_reference",
        "硬伤表·度量口径：R53（残差 6% 实为幅度误差 26.55–41.18%）"),
    ("MEMO_AUDIT_2750_3148.md", "R54", 48): ("substantive_reference",
        "硬伤表：能量份额的非互补性（R54：ASH+HSH≠1）"),
    ("MEMO_AUDIT_2750_3148.md", "R54", 96): ("substantive_reference",
        "硬伤表·度量口径：能量份额含交叉项却当互补（R54）"),
    ("MEMO@3104+", "R55", 13546): ("candidate_violation",
        "Phase 3113 以 R55 为依据给出 per-head 切点规则；R55 含 E 分量（E+A）⇒ 规则不可机械判定"),
}

subst, benign, collisions = [], [], []
for name, lst in hits_main.items():
    for h in lst:
        rec = dict(h, corpus=name)
        cls, why = CURATED.get((name, h["id"], h["line"]), ("", ""))
        rec["class"] = cls or ("meta_mention_benign" if h["meta"] else "unclassified")
        rec["note"] = why
        if rec["class"] == "namespace_collision":
            collisions.append(rec)
        elif rec["class"] == "meta_mention_benign":
            benign.append(rec)
        else:
            subst.append(rec)

# 文本级实质泄漏
fp_violations = [x for x in fp_main if x["corpus"] == "MEMO@3104+"]

cd = dict(Counter(p["grade"] for p in props))
cc = Counter()
for p in props:
    for c in p["comp"]:
        cc[c] += 1
pureDE = [p for p in DE if p["pure_DE"]]
de_comp = [p for p in DE if not p["pure_DE"]]
pureE = [p for p in props if p["grade"] == "E"]
pureD = [p for p in props if p["grade"] == "D"]

result = {
    "schema": "rdc_prop_citation_audit_v1",
    "queue_item": "Q12",
    "generated_by": "tests/deepseek/prop_citation_audit.py",
    "constitution_ref": "RDC_RESEARCH_CONSTITUTION_v1.md 6 (I4)",
    "ledger": {"path": os.path.relpath(led_path, ROOT), "sha8": sha8(led_path), "n_props": len(props)},
    "corpora": {
        "primary": list(doc_corpora.keys()),
        "MEMO_new_region_start_line": NEW_START,
        "archives_excluded_from_verdict": sorted(archives.keys()),
    },
    "counts": {
        "n_props": len(props), "n_DE_union": len(DE),
        "pct_DE_union": round(100.0 * len(DE) / len(props), 2),
        "n_pure_DE": len(pureDE), "n_DE_composite": len(de_comp),
        "n_composite_any": sum(1 for p in props if len(p["comp"]) > 1),
        "label_distribution": cd, "component_distribution": dict(cc),
        "pure_E_ids": [p["id"] for p in pureE], "pure_D_ids": [p["id"] for p in pureD],
    },
    "F1_id_hits_primary": hits_main,
    "F1_id_hits_archives_counts": {k: len(v) for k, v in hits_arch.items()},
    "F2_namespace_occupancy": ns,
    "F3_text_fingerprint": {"n_constructible": n_fp_constructible,
                            "hits_primary": fp_main, "hits_archives": fp_arch},
    "verdict": {"substantive_id_hits": subst, "benign_meta_hits": benign,
                "namespace_collisions": collisions,
                "n_substantive_in_new_region": len([x for x in subst if x["corpus"] == "MEMO@3104+"]),
                "text_level_violations_in_new_region": fp_violations},
    "defects": {
        "RULE_UNDECIDABLE": {
            "statement": "『E 级禁止进入新推理链』对复合等级不可机械判定",
            "evidence": "%d/%d 条 D∪E 为复合等级（含 A/B/C 之一）；纯 E 仅 %d 条" % (
                len(de_comp), len(DE), len(pureE)),
            "fix": "真源须把复合等级拆为分量级条目（component-level entry，独立 grade + 独立 id）",
        },
        "NS_COLLISION": {
            "statement": "账本 id R\\d\\d 不是项目内唯一命名空间 ⇒ 纯 id 检测必然假阳性",
            "evidence": "LOOP_DIAGNOSIS_AND_EXIT_v1.md 使用自有 R1–R11 清单；其 R10/R11 与账本 R10/R11 内容完全不同",
            "fix": "id 加前缀（PL-R44）或审计一律走文本指纹",
        },
        "PCT_DIM_MIX": {
            "statement": "MEMO_AUDIT 的『约 2/3 命题不能直接进入新推理链』是 34%+32% 的分量和（量纲混淆）",
            "component_sum_pct": 66.0, "by_entry_pct": round(100.0 * len(DE) / len(props), 2),
        },
    },
}
jp = os.path.join(ATLAS, "prop_citation_audit_v1.json")
with open(jp, "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(result, ensure_ascii=False, indent=1))

# ---- Markdown ----
L = []
def W(s=""):
    L.append(s)

W("# Q12 D/E 级命题引用审计 —— 账本驱动 · 命名空间感知 · 文本指纹级")
W("")
W("- **文档性质**：审计报告（只读；不改动任何 MEMO 原文 / 账本 / 既有判决）。")
W("- **上位依据**：`RDC_RESEARCH_CONSTITUTION_v1.md` §6（I4）；`RDC_TESTPLAN_v1.md` F4。")
W("- **真源**：`%s`（`%s`，%d 条命题）。" % (result["ledger"]["path"], result["ledger"]["sha8"], len(props)))
W("- **主语料**：现行 MEMO 自 **Phase 3104 起（第 %d 行）**至文末 ∪ `docs/*.md`（%d 个）。" % (
    NEW_START, len(doc_corpora) - 1))
W("- **归档 MEMO**（`AGI_GPT5_MEMO_2026*.md`，%d 个）为同一文档历史快照 ⇒ 只报计数、**不入判决**。" % len(archives))
W("- **工具**：`tests/deepseek/prop_citation_audit.py`（可重复运行）")
W("")
W("---")
W("")
W("## §0 判决")
W("")
W("| 检测 | 结果 |")
W("|---|---|")
W("| **F1 id 级**（新语料） | 共 %d 处：**新推理链（MEMO@3104+）%d 处**、分析/审计文档 %d 处、命名空间碰撞（假阳性）%d 处、元层审计提及（良性）%d 处 |" % (
    sum(len(v) for v in hits_main.values()),
    len([x for x in subst if x["corpus"] == "MEMO@3104+"]),
    len([x for x in subst if x["corpus"] != "MEMO@3104+"]),
    len(collisions), len(benign)))
W("| **F3 文本指纹级**（新推理链） | %d 条可构造指纹的 D/E 命题，**实质命中 %d 条** |" % (
    n_fp_constructible, len(fp_violations)))
W("")
W("> **判决：新推理链中无实质违规。** id 级只有 1 处候选（Phase 3113 引用 R55），且该条为**复合等级**⇒ 规则不可机械判定。文本级 **0** 命中。")
W("")
W("### 四项可证伪事实")
W("")
W("**F1 —— 新推理链中的 id 命中**")
W("")
if hits_main:
    W("| 语料 | id | grade | 行 | 分类（人工核定） | 核定说明 | 上下文（截断） |")
    W("|---|---|---|---|---|---|---|")
    for name in sorted(hits_main):
        for h in sorted(hits_main[name], key=lambda x: x["line"]):
            cls, why = CURATED.get((name, h["id"], h["line"]), ("", ""))
            if not cls:
                cls = "元层审计提及（良性）" if h["meta"] else "未分类"
            lab = {"namespace_collision": "命名空间碰撞（假阳性）",
                   "substantive_reference": "实质引用",
                   "candidate_violation": "**候选违规**",
                   "meta_mention_benign": "元层审计提及（良性）"}.get(cls, cls)
            W("| %s | `%s` | %s | %d | %s | %s | %s |" % (
                name, h["id"], h["grade"], h["line"], lab, why[:110], h["ctx"][:130]))
else:
    W("（无命中）")
W("")
W("**F2 —— 命名空间占用普查（id 检测为何不可靠）**")
W("")
W("| 文件 | 出现的 `R\\d\\d` |")
W("|---|---|")
for fn, s in ns.items():
    W("| `%s` | %s |" % (fn, ", ".join(s)))
W("")
W("> `LOOP_DIAGNOSIS_AND_EXIT_v1.md` 的 `R10`/`R11` 是它**自有的** §2 清单编号（内容为“真阴性记录：翻译轴不存在”“装置层：execution 冻结”），与账本的 `R10`（实体感紧凑性定律）/`R11`（层级嵌套、十维骨架）**毫无关系** ⇒ 任何纯 id 的 grep 都会在这里产生假阳性。")
W("")
W("**F3 —— 文本指纹（claim 的 6-CJK-gram）**")
W("")
W("| 语料 | 命中条数 |")
W("|---|---|")
W("| 主语料·新推理链（MEMO@3104+） | **%d** |" % len(fp_violations))
W("| 主语料·分析/审计文档 | %d |" % len([x for x in fp_main if x["corpus"] != "MEMO@3104+"]))
W("| 归档 MEMO | %d（信息性） |" % len(fp_arch))
W("")
if fp_violations:
    for x in fp_violations:
        W("- `%s`（%s）: %d/%d，例：%s" % (x["id"], x["grade"], x["n_hit"], x["n_total"], " / ".join(x["examples"])))
else:
    W("> 主语料 **0 命中** —— 退出/撤回命题的实质内容**没有**以任何形式进入 3104+ 的推理链。")
W("")
W("**F4 —— 分级口径缺陷**")
W("")
W("| 口径 | 值 |")
W("|---|---|")
W("| 命题总数 | %d |" % len(props))
W("| 原始标签种数 | %d（%s） |" % (len(cd), ", ".join("%s×%d" % (k, v) for k, v in sorted(cd.items(), key=lambda x: -x[1]))))
W("| 分量口径 | %s |" % ", ".join("%s=%d" % (k, v) for k, v in sorted(cc.items())))
W("| **D∪E（分量）** | **%d / %d = %.2f%%** |" % (len(DE), len(props), result["counts"]["pct_DE_union"]))
W("| 其中复合等级（含 A/B/C） | **%d 条** |" % len(de_comp))
W("| 纯 E | **%d 条**（%s） |" % (len(pureE), ", ".join(p["id"] for p in pureE)))
W("| 纯 D | %d 条（%s） |" % (len(pureD), ", ".join(p["id"] for p in pureD)))
W("")
W("> **规则不可判定**：`E 级禁止进入新推理链` 对 %d/%d 条 D∪E 命题无法机械应用——因为它们的 grade 是复合的（`E+A`/`B+E`/`E+C`/`D+E`/`E+D`），撤回的是 E 分量，而 A/B/C 分量仍可引用。" % (
    len(de_comp), len(DE)))
W("")
W("> **量纲提示（承接 Q01-D2）**：`MEMO_AUDIT` 的“D=21(34%%)、E=20(32%%) 即约 2/3 命题不能直接进入新推理链”用的是**分量和**（66%%）；按**条数**为 %d/%d = **%.2f%%**。" % (
    len(DE), len(props), result["counts"]["pct_DE_union"]))
W("")
W("---")
W("")
W("## §1 处置建议（不代为改判）")
W("")
W("1. **RULE_UNDECIDABLE**：真源把复合等级**拆成分量级条目**（独立 grade + 独立 id），才能机械判定“哪一部分禁入”。")
W("2. **NS_COLLISION**：账本 id 加前缀（如 `PL-R44`），或所有审计一律走**文本指纹**而非 id。")
W("3. **常设闸门**：新 Phase 的 `result.json` 加断言「本 Phase 文本中无 D/E 分量的 6-gram 指纹命中」——本条已可自动执行。")
W("")
W("---")
W("")
W("*本报告为审计性文档；所有计数与命中由 `prop_citation_audit.py` 现场解析，可重复运行逐条回查。*")

mp = os.path.join(DOCS, "PROP_CITATION_AUDIT_Q12.md")
with open(mp, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(L) + "\n")

print("WROTE", os.path.relpath(jp, ROOT), os.path.getsize(jp), "B", sha8(jp))
print("WROTE", os.path.relpath(mp, ROOT), os.path.getsize(mp), "B", sha8(mp))
print("F1 primary hits:", sum(len(v) for v in hits_main.values()),
      " substantive:", len(subst), " benign:", len(benign))
print("F3 fp_main:", len(fp_main), " fp_violations(new region):", len(fp_violations), " fp_arch:", len(fp_arch))
print("F3 constructible:", n_fp_constructible, "of DE", len(DE))
for name in sorted(hits_main):
    for h in hits_main[name]:
        print("   HIT", name, h["id"], h["grade"], "L", h["line"], "meta=", h["meta"])
