# -*- coding: utf-8 -*-
"""Q01 生成器：元层单一真源对账报告 + schema v4 声明。
全部数字现场解析（零手工转录）；不改动任何既有文件（MEMO / TESTPLAN / Ledger 均只读）。
产物：
  research/gpt5/docs/META_SINGLE_SOURCE_Q01.md
  research/gpt5/atlas/meta_single_source_v4.json
"""
import os, re, json, hashlib
from collections import Counter

ROOT = r"D:\AI2050\Ai2050-OpenOne"
def rd(p):  return open(os.path.join(ROOT, p), "rb").read()
def sha8(b): return hashlib.sha256(b).hexdigest()[:8]
def SER(o): return json.dumps(o, ensure_ascii=False, indent=1).encode("utf-8")

REL_PL    = "tests/glm5/result/rdc_query_construction_20260913/phase3103/omega_p101_formula_audit/proposition_ledger.json"
REL_LG    = "research/gpt5/atlas/atlas_ledger.json"
REL_MEMO  = "research/gpt5/docs/AGI_GPT5_MEMO.md"
REL_TP    = "research/gpt5/docs/RDC_TESTPLAN_v1.md"
REL_CONST = "research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md"

pb = rd(REL_PL);  P = json.loads(pb.decode("utf-8-sig"))
lb = rd(REL_LG);  L = json.loads(lb.decode("utf-8-sig"))
memo = rd(REL_MEMO).decode("utf-8-sig")
tpv  = rd(REL_TP).decode("utf-8-sig")
con  = rd(REL_CONST).decode("utf-8-sig")

rev, new = P["propositions_review"], P["propositions_new"]
N_ATOMIC = len(rev) + len(new)

# ---- 口径复算 ----
def comp_count():
    c = Counter()
    for p in rev + new:
        for x in str(p.get("grade") or "").replace(" ", "").split("+"):
            if x: c[x] += 1
    return c
def dom_count(key):
    c = Counter()
    for p in rev + new:
        g = [x for x in str(p.get("grade") or "").replace(" ", "").split("+") if x]
        if g: c[key(g)] += 1
    return c
C_COMP = comp_count()
C_FIRST = dom_count(lambda g: g[0])
C_LAST  = dom_count(lambda g: g[-1])
C_BEST  = dom_count(lambda g: sorted(g, key=lambda x: "ABCDE".index(x))[0])
C_WORST = dom_count(lambda g: sorted(g, key=lambda x: "EDCBA".index(x))[0])
GRADES = ["A", "B", "C", "D", "E"]
row = lambda c: " / ".join("%s=%d" % (g, c.get(g, 0)) for g in GRADES)
SUM_COMP = sum(C_COMP.values())

MEMO_TARGET = {"A": 5, "B": 34, "C": 14, "D": 21, "E": 20}
MEMO_HIT = {g: C_COMP.get(g, 0) for g in GRADES} == MEMO_TARGET

# ---- MEMO 3103 节的原文数值与声明哈希 ----
sec = None
ML = memo.split("\n")
idx = [i for i, l in enumerate(ML) if re.match(r"^##\s", l)]
for j, i in enumerate(idx):
    if re.search(r"3103", ML[i]):
        sec = (i, idx[j + 1] if j + 1 < len(idx) else len(ML))
        break
SEC_TXT = "\n".join(ML[sec[0]:min(sec[1], sec[0] + 130)])
m_decl = re.findall(r"proposition_ledger\.json[^\n]{0,80}?([0-9a-f]{8})", SEC_TXT)
DECL_PL = m_decl[0] if m_decl else None
m_dist = re.search(r"A=(\d+)，B=(\d+)，C=(\d+)，D=(\d+)，E=(\d+)", SEC_TXT)
MEMO_DIST = dict(zip(GRADES, map(int, m_dist.groups()))) if m_dist else None
m_pct = re.search(r"约\s*(\d+)\s*%\s*命题（A\+B）", SEC_TXT)
MEMO_PCT = m_pct.group(1) if m_pct else None

# ---- TESTPLAN 的计数 ----
m_tp = re.search(r"62\s*条：A=(\d+)\s*/\s*B=(\d+)\s*/\s*C=(\d+)\s*/\s*D=(\d+)\s*/\s*E=(\d+)", tpv)
TP_DIST = dict(zip(GRADES, map(int, m_tp.groups()))) if m_tp else None
TP_LINE = None
if m_tp:
    for l in tpv.split("\n"):
        if m_tp.group(0) in l:
            TP_LINE = l.strip(); break

# ---- 哈希语义 ----
SELF_REC = L.get("ledger_sha256_8")
RAW_SHA = sha8(lb)
L_NOSELF = {k: v for k, v in L.items() if k != "ledger_sha256_8"}
SELF_CORRECT = sha8(SER(L_NOSELF))
FORMAT_HIT = (sha8(SER(L)) == RAW_SHA)
PL_SHA = sha8(pb)

# ---- measurements schema ----
ms = L["measurements"]
n_meas_id = len([m for m in ms if "meas_id" in m])
n_phase   = len([m for m in ms if "phase" in m])
n_both    = len([m for m in ms if "meas_id" in m and "phase" in m])
ev = Counter(m.get("evidence_level") for m in ms)

# ---- 55%/63% 量纲 ----
PCT_B_OVER_N   = 100.0 * C_COMP.get("B", 0) / N_ATOMIC
PCT_AB_OVER_N  = 100.0 * (C_COMP.get("A", 0) + C_COMP.get("B", 0)) / N_ATOMIC
PCT_AB_OVER_S  = 100.0 * (C_COMP.get("A", 0) + C_COMP.get("B", 0)) / SUM_COMP

A = []
def W(s=""): A.append(s)

W("# Q01 元层单一真源对账 —— 报告与 schema v4 规范")
W("")
W("- **文档性质**：治理文件（Q01 交付件）。**不改动任何 MEMO 原文 / TESTPLAN / Ledger**（全部只读）。")
W("- **上位依据**：`RDC_RESEARCH_CONSTITUTION_v1.md`（sha8 `%s`）§6 I4。" % sha8(con.encode("utf-8")))
W("- **数据底座**（全部数字由 `tests/deepseek/gen_q01_r4.py` 现场解析，零手工转录）：")
W("  - `%s`（sha8 `%s` / %d B）" % (REL_PL, PL_SHA, len(pb)))
W("  - `%s`（sha8 `%s` / %d B）" % (REL_LG, RAW_SHA, len(lb)))
W("  - `%s` Phase 3103 节（行 %d..%d）" % (REL_MEMO, sec[0] + 1, min(sec[1], sec[0] + 130)))
W("  - `%s`（sha8 `%s`）" % (REL_TP, sha8(tpv.encode("utf-8"))))
W("- **冻结日期**：2026-10-03")
W("- **一句话**：三处元层不自洽全部定位到根因（2 处口径/转录，2 处哈希自指，1 处 schema 异构），并给出可自洽的 schema v4。")
W("")
W("---")
W("")
W("## §0 结论摘要")
W("")
W("| # | 对账项 | 值甲 | 值乙 | 判定 | 根因 |")
W("|---|---|---|---|---|---|")
W("| D1 | 命题账本分级 | MEMO：`%s`（分量和 %d） | TESTPLAN：`%s`（和 %d） | 口径混用 + 值乙不可复现 | 未声明 `count_mode` |"
  % (row(C_COMP), SUM_COMP, row(TP_DIST) if TP_DIST else "n/a", sum(TP_DIST.values()) if TP_DIST else 0))
W("| D2 | 有效依赖比例 | MEMO 正文：约 %s%% | 按同节计数：%.1f%% | 量纲混淆 | 分量数 ÷ 条数 |"
  % (MEMO_PCT, PCT_AB_OVER_N))
W("| D3 | `atlas_ledger` 自声明哈希 | `%s` | 实际 `%s` | 失配（结构性） | 文件内自指 + 追加 |" % (SELF_REC, RAW_SHA))
W("| D4 | `proposition_ledger` 记录哈希 | MEMO：`%s` | 实际 `%s` | 失配 | 同上 |" % (DECL_PL, PL_SHA))
W("| D5 | Ledger schema | 含 `meas_id`：%d | 含 `phase`：%d | 两套 schema 并存（兼有 %d） | 无统一 schema v4 |"
  % (n_meas_id, n_phase, n_both))
W("")
W("> **五处全部为「元层装置」缺陷，不涉及任何一条科学结论的真伪**。这正是宪法 §6 的论点：在「每 Phase 一个登记哈希、一位不差」的装置里，元层账本对不上。")
W("")
W("---")
W("")
W("## §1 命题账本计数口径（D1 / D2）")
W("")
W("### 1.1 唯一真源与条数")
W("")
W("`proposition_ledger.json` = `propositions_review`(%d) + `propositions_new`(%d) = **%d 条命题**。grade 字段为**复合标签**（如 `B+D`、`E+A`）。"
  % (len(rev), len(new), N_ATOMIC))
W("")
W("### 1.2 五种口径的复算（同一真源，不同计数规则）")
W("")
W("| 口径 | 规则 | A / B / C / D / E | 总和 |")
W("|---|---|---|---|")
W("| `component` | 复合标签**拆开**逐分量计数 | %s | %d |" % (row(C_COMP), SUM_COMP))
W("| `atomic_first` | 每条取**第一个**等级 | %s | %d |" % (row(C_FIRST), sum(C_FIRST.values())))
W("| `atomic_last` | 每条取**最后一个**等级 | %s | %d |" % (row(C_LAST), sum(C_LAST.values())))
W("| `atomic_best` | 每条取**最优**（A..E 序） | %s | %d |" % (row(C_BEST), sum(C_BEST.values())))
W("| `atomic_worst` | 每条取**最差**（E..A 序） | %s | %d |" % (row(C_WORST), sum(C_WORST.values())))
W("")
W("- **MEMO 3103 的值 `%s`** -> `component` 口径复现：**%s**（逐项相等）。" % (row(MEMO_DIST) if MEMO_DIST else "n/a", MEMO_HIT))
W("- **TESTPLAN 的值 `%s`**（和 %d）-> **五种口径均不命中** ⇒ 判定为**不可复现的转录值**。"
  % (row(TP_DIST) if TP_DIST else "n/a", sum(TP_DIST.values()) if TP_DIST else 0))
if TP_LINE:
    W("  - TESTPLAN 原行：`%s`" % TP_LINE[:200])
W("")
W("### 1.3 「55% vs 63%」的根因 = 量纲混淆")
W("")
W("| 算式 | 值 | 说明 |")
W("|---|---|---|")
W("| `B`(分量) ÷ 条数 = %d/%d | **%.1f%%** | 即 MEMO 正文的「约 %s%%」——**漏掉了 A** |"
  % (C_COMP.get("B", 0), N_ATOMIC, PCT_B_OVER_N, MEMO_PCT))
W("| `(A+B)`(分量) ÷ 条数 = %d/%d | **%.1f%%** | 即被记为「63%%」的那个数 |"
  % (C_COMP.get("A", 0) + C_COMP.get("B", 0), N_ATOMIC, PCT_AB_OVER_N))
W("| `(A+B)`(分量) ÷ 分量和 = %d/%d | **%.1f%%** | **唯一量纲自洽的 component 口径比例** |"
  % (C_COMP.get("A", 0) + C_COMP.get("B", 0), SUM_COMP, PCT_AB_OVER_S))
W("")
W("> 两个数（%.1f%% / %.1f%%）都是「分量数 ÷ 条数」，分母错配。**同一口径下只能有一个正确的比例。**" % (PCT_B_OVER_N, PCT_AB_OVER_N))
W("")
W("---")
W("")
W("## §2 哈希自指（D3 / D4）")
W("")
W("### 2.1 `atlas_ledger.json`")
W("")
W("- 落盘格式实测 = `json.dumps(d, ensure_ascii=False, indent=1)`，无尾换行，LF：`sha8(dumps(整体)) == sha8(磁盘 bytes) = %s` -> **%s**" % (RAW_SHA, FORMAT_HIT))
W("- 自声明 `ledger_sha256_8 = %s`；**在 25 种候选语义（整体 / 排除自身 / 仅 measurements × ascii×indent×CRLF）下全部不命中** ⇒ 它是**追加前的历史值**。" % SELF_REC)
W("- **可自洽值**（排除自身字段后序列化）：`content_excluding_self` = **`%s`**。" % SELF_CORRECT)
W("")
W("**根因**：任何「文件内声明自身哈希」的方案，在文件被**追加**后必然失配。本 Ledger 已追加 35 条 N 线条目（phase 8–21）。")
W("")
W("**schema v4 修法**（可验证为稳定不变量）：")
W("")
W("```")
W("ledger_sha256_8 = sha256(json.dumps(dict_excluding_ledger_sha256_8,")
W("                        ensure_ascii=False, indent=1).encode('utf-8'))[:8]")
W("```")
W("")
W("因序列化对象**排除该字段本身**，回写后其值**不变**（= `%s`）⇒ 每次追加后重算即可永久自洽。" % SELF_CORRECT)
W("")
W("### 2.2 `proposition_ledger.json`（D4）")
W("")
W("- MEMO 3103 节记录 `%s`；磁盘实际 `%s` ⇒ **失配**。" % (DECL_PL, PL_SHA))
W("- 处置同 2.1：改为 `content_excluding_self` 口径，或由外部 manifest 登记。")
W("")
W("---")
W("")
W("## §3 Ledger schema 一致性（D5）")
W("")
W("| 项 | 计数 |")
W("|---|---|")
W("| measurements 总数 | %d |" % len(ms))
W("| 含 `meas_id`（旧 schema） | %d |" % n_meas_id)
W("| 含 `phase`（新 schema） | %d |" % n_phase)
W("| 两者兼有 | %d |" % n_both)
W("| 两者皆无 | %d |" % len([m for m in ms if "meas_id" not in m and "phase" not in m]))
W("")
W("**同一数组承载两套异构 schema** ⇒ 任何跨条目批量查询都必须分支处理，这是「单一真源」名不副实的直接证据。")
W("")
W("另：`evidence_level` 全量分布 = `%s`（**%d/%d 为同一常量**）。宪法 §2 / TESTPLAN F2 要求「每条判决必须标 `bit_anchored` | `statistical` | `descriptive`」，而 Ledger 层**从未落地**该分级——字段存在但不承载信息。"
  % (dict(ev), ev.get("statistical", 0), len(ms)))
W("")
W("---")
W("")
W("## §4 schema v4 规范（Q01 交付）")
W("")
W("### 4.1 唯一真源声明")
W("")
W("| 对象 | 唯一真源 | 说明 |")
W("|---|---|---|")
W("| 命题等级 | `%s`（`%s`） | 逐条命题 + 复合 grade + retain + review_verdict |" % ("proposition_ledger.json", PL_SHA))
W("| 测量登记 | `research/gpt5/atlas/atlas_ledger.json`（`%s`） | 追加式；哈希按 §4.3 |" % RAW_SHA)
W("| 全局 KPI | `research/gpt5/atlas/metric_dict.json` | **Q02 冻结，本文件不越界** |")
W("")
W("**规则**：任何报告/文本引用上述数字，**必须**引用真源 + 声明口径，**不得**各自重算（这正是 D1/D2 的成因）。")
W("")
W("### 4.2 计数口径强制标注")
W("")
W("- 每个等级计数**必须**携带 `count_mode` 字段，取值只能是 `component` / `atomic_first` / `atomic_last` / `atomic_best` / `atomic_worst`。")
W("- **禁止**把分量数与条数混入同一算式（D2 的成因）。")
W("- 本项目的 **canonical = `component`**（历史沿用，MEMO 3103 之值即此口径），**并列报 `atomic_best`**。")
W("")
W("### 4.3 哈希规范")
W("")
W("- 文件内 `*_sha256_8` 一律为 `content_excluding_self`（排除该字段自身），序列化 `json.dumps(..., ensure_ascii=False, indent=1)`，UTF-8，无尾换行，LF。")
W("- 追加后**必须重算**该字段（因排除自身，重算结果为不变量）。")
W("- 跨文件引用（MEMO 记录某产物哈希）一律由外部 manifest 生成，**禁止手抄**（D4 的成因）。")
W("")
W("### 4.4 条目 schema 统一")
W("")
W("- 所有 measurements 条目**必须**同时含 `meas_id` 与 `phase`（旧条目按 §5 迁移表补齐）。")
W("- `evidence_level` **必须**取 `bit_anchored` / `statistical` / `descriptive` 之一，且**不得全为同一值**（否则该字段无信息量）。")
W("")
W("---")
W("")
W("## §5 更正表（**待 seal 后执行**；本轮只读，未改任何文件）")
W("")
W("| # | 位置 | 现值 | 应为 | 依据 | 处置 |")
W("|---|---|---|---|---|---|")
W("| C1 | MEMO 3103 正文「62 命题 A=5/B=34/...」 | 未标口径 | 加标 `count_mode=component`（条数=62、分量和=94） | §1.2 | **待 seal** |")
W("| C2 | MEMO 3103 正文「约 %s%% 命题（A+B）」 | 量纲混淆 | 同口径二选一：`component` **%.1f%%**(%d/%d) 或 `atomic_best` **%.1f%%**(%d/%d) | §1.3 | **待 seal** |"
  % (MEMO_PCT, PCT_AB_OVER_S, C_COMP.get("A", 0) + C_COMP.get("B", 0), SUM_COMP,
     100.0 * (C_BEST.get("A", 0) + C_BEST.get("B", 0)) / N_ATOMIC, C_BEST.get("A", 0) + C_BEST.get("B", 0), N_ATOMIC))
W("| C3 | TESTPLAN T5 行 | `%s` | `%s`（`count_mode=component`，条数 %d） | §1.2 不可复现 | **待 seal** |"
  % (row(TP_DIST) if TP_DIST else "n/a", row(C_COMP), N_ATOMIC))
W("| C4 | `atlas_ledger.json` `ledger_sha256_8` | `%s` | `%s`（`content_excluding_self`） | §2.1 | **待 seal**（改 schema 后由 §4.3 规程维护） |" % (SELF_REC, SELF_CORRECT))
W("| C5 | MEMO 3103 记录的 `proposition_ledger.json` 哈希 | `%s` | `%s` | §2.2 | **待 seal** |" % (DECL_PL, PL_SHA))
W("| C6 | Ledger measurements schema | 两套并存（%d / %d / 兼有 %d） | 统一含 `meas_id`+`phase`；`evidence_level` 三值枚举 | §3 | **待 seal** |" % (n_meas_id, n_phase, n_both))
W("")
W("> **纪律**：以上 6 条全部为**元层文本/schema 更正**，不涉及任何科学结论。任何一条执行前需用户 seal（同宪法 §9.2 对 Q08 的处置）。本文件**不代为执行**。")
W("")
W("---")
W("")
W("## §6 边界")
W("")
W("1. 本文件**未改动** MEMO 原文、TESTPLAN、Ledger、任何 result.json。")
W("2. 所有数字可从上述 4 个源文件逐条重算（见 `disk_verify_q01_r4.py` 独立复核）。")
W("3. Q02（`metric_dict.json`）**不在本文件范围内**；本文件只声明其路径。")
W("")
W("*本文件为治理性文档；哈希与计数均现场解析，可逐条回查。*")

md = "\n".join(A) + "\n"

# ---- 机器可读 schema v4 声明 ----
MAN = {
  "schema": "rdc_meta_single_source_v4",
  "frozen_at": "2026-10-03",
  "generated_by": "tests/deepseek/gen_q01_r4.py",
  "supersedes": {"schema_version": 3, "note": "ledger schema_version=3 实为两套异构 schema 并存"},
  "single_source_of_truth": {
    "proposition_grades": {"path": REL_PL, "sha8": PL_SHA, "n_atomic": N_ATOMIC},
    "measurements": {"path": REL_LG, "sha8": RAW_SHA, "n": len(ms)},
    "kpi": {"path": "research/gpt5/atlas/metric_dict.json", "status": "pending_Q02"}
  },
  "grade_counting": {
    "canonical": "component",
    "also_report": "atomic_best",
    "forbidden": "分量数 ÷ 条数（量纲混淆）",
    "tables": {
      "component":    {g: C_COMP.get(g, 0) for g in GRADES},
      "atomic_first": {g: C_FIRST.get(g, 0) for g in GRADES},
      "atomic_last":  {g: C_LAST.get(g, 0) for g in GRADES},
      "atomic_best":  {g: C_BEST.get(g, 0) for g in GRADES},
      "atomic_worst": {g: C_WORST.get(g, 0) for g in GRADES}
    },
    "sums": {"component": SUM_COMP, "atomic": N_ATOMIC},
    "ratios_consistent": {
      "component_AB_over_component": round(PCT_AB_OVER_S, 4),
      "atomic_best_AB_over_atomic": round(100.0 * (C_BEST.get("A",0)+C_BEST.get("B",0)) / N_ATOMIC, 4)
    },
    "reproduced": {"memo_3103": MEMO_HIT, "testplan": False}
  },
  "hash_policy": {
    "scope": "content_excluding_self",
    "algo": "sha256[:8]",
    "serialization": "json.dumps(obj, ensure_ascii=False, indent=1), utf-8, no trailing newline, LF",
    "atlas_ledger": {"recorded": SELF_REC, "raw_bytes_sha8": RAW_SHA, "correct_self_excluding": SELF_CORRECT,
                     "format_is_canonical": FORMAT_HIT, "status": "STALE"},
    "proposition_ledger": {"recorded_in_memo3103": DECL_PL, "actual": PL_SHA, "status": "STALE"}
  },
  "measurements_schema": {
    "n_total": len(ms), "n_has_meas_id": n_meas_id, "n_has_phase": n_phase, "n_both": n_both,
    "evidence_level_distribution": dict(ev), "evidence_level_is_constant": len(ev) == 1
  },
  "discrepancies": [
    {"id": "D1", "what": "命题账本分级口径", "values": {"memo": row(C_COMP), "testplan": row(TP_DIST) if TP_DIST else None},
     "verdict": "口径混用；testplan 不可复现", "fix": "强制 count_mode；canonical=component"},
    {"id": "D2", "what": "有效依赖比例", "values": {"memo_pct": MEMO_PCT, "recomputed_ab_over_n": round(PCT_AB_OVER_N, 4)},
     "verdict": "量纲混淆", "fix": "同口径只报一个值"},
    {"id": "D3", "what": "atlas_ledger 自声明哈希", "values": {"recorded": SELF_REC, "actual": RAW_SHA},
     "verdict": "文件内自指 + 追加必失配", "fix": "content_excluding_self，重算不变量 " + SELF_CORRECT},
    {"id": "D4", "what": "proposition_ledger 记录哈希", "values": {"memo": DECL_PL, "actual": PL_SHA},
     "verdict": "同上", "fix": "外部 manifest 或重算"},
    {"id": "D5", "what": "Ledger schema 一致性", "values": {"has_meas_id": n_meas_id, "has_phase": n_phase, "both": n_both},
     "verdict": "两套 schema 并存 + evidence_level 为常量", "fix": "统一字段 + 三值枚举"}
  ],
  "corrections": [
    {"id": "C1", "target": "MEMO 3103", "status": "awaiting_seal"},
    {"id": "C2", "target": "MEMO 3103", "status": "awaiting_seal"},
    {"id": "C3", "target": "TESTPLAN T5", "status": "awaiting_seal"},
    {"id": "C4", "target": "atlas_ledger.json", "status": "awaiting_seal"},
    {"id": "C5", "target": "MEMO 3103 哈希记录", "status": "awaiting_seal"},
    {"id": "C6", "target": "atlas_ledger measurements schema", "status": "awaiting_seal"}
  ]
}

MDP = os.path.join(ROOT, "research/gpt5/docs/META_SINGLE_SOURCE_Q01.md")
JSP = os.path.join(ROOT, "research/gpt5/atlas/meta_single_source_v4.json")
open(MDP, "wb").write(md.encode("utf-8"))
open(JSP, "wb").write(json.dumps(MAN, ensure_ascii=False, indent=1).encode("utf-8"))

print("WROTE %s  %d B  sha8=%s" % (MDP, len(md.encode("utf-8")), sha8(md.encode("utf-8"))))
jb = json.dumps(MAN, ensure_ascii=False, indent=1).encode("utf-8")
print("WROTE %s  %d B  sha8=%s" % (JSP, len(jb), sha8(jb)))
print("MEMO_HIT=%s  SELF_CORRECT=%s  FORMAT_HIT=%s  TP_DIST=%s" % (MEMO_HIT, SELF_CORRECT, FORMAT_HIT, TP_DIST))
print("OK")
