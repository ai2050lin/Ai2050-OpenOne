# -*- coding: utf-8 -*-
"""R6: 把 research/gpt5/docs 下 8 件 deepseek 治理/交付文档全文并入 AGI_DEEPSEEK_MEMO.md
（Phase 23-30），追加 Phase 31 记录本轮路由归一；规范化 Phase 22 遗留标题标记。
只读 gpt5 目录（备份另存），不删除原件（删除由下一步脚本执行）。"""
import os, re, hashlib, shutil

ROOT = r"D:\AI2050\Ai2050-OpenOne"
MEMO = os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md")
DOCS = os.path.join(ROOT, r"research\gpt5\docs")
BK = os.path.join(ROOT, r"tests\deepseek_temp\_archive_r6\gpt5_docs")
REPORT = os.path.join(ROOT, r"tests\deepseek_temp\_merge_docs_r6_report.txt")
STAMP = "[2026-10-03 02:35]"

PLAN = [
    (23, "RDC_TESTPLAN_v1.md", "RDC 破解路线裁决与测试方案 v1", "前置设计：逐条裁决 + 3150-3153 测试方案 + §8.2 三条死线原文"),
    (24, "LOOP_DIAGNOSIS_AND_EXIT_v1.md", "循环诊断与出路 v1", "R3：为什么 402 个 Phase 只产出局部特征"),
    (25, "RDC_RESEARCH_CONSTITUTION_v1.md", "RDC 研究宪法 v1", "R3：I1-I11 全部落为判据，让验收函数可失败"),
    (26, "META_SINGLE_SOURCE_Q01.md", "Q01 元层单一真源对账", "R4：D1-D5 五处不自洽 + schema v4 + C1-C6 更正表"),
    (27, "METRIC_DICT_Q02.md", "Q02 KPI 口径冻结", "R4：metric_dict v2，E_read / E_ar / C_steer"),
    (28, "DEADLINE_DUAL_TRACK_Q09.md", "Q09 死线双轨重述", "R5：I3 双轨制 + K1 现场重算 + K2/K3 从未被测量"),
    (29, "PROP_CITATION_AUDIT_Q12.md", "Q12 D/E 级命题引用审计", "R5：账本驱动 · 命名空间感知 · 文本指纹级"),
    (30, "SEAL_REQUEST_A_GATE.md", "A 闸门 seal 请求", "R5b：Q08 层位选择 + C1-C6 + I1/I9"),
]
JSON_PLAN = [
    ("deadline_dual_track_v1.json", r"tests\deepseek\result\deadline_dual_track_v1.json"),
    ("prop_citation_audit_v1.json", r"tests\deepseek\result\prop_citation_audit_v1.json"),
    ("meta_single_source_v4.json", r"tests\deepseek\result\meta_single_source_v4.json"),
    ("seal_request_v1.json", r"tests\deepseek\result\seal_request_v1.json"),
    ("metric_dict.json", r"research\deepseek\atlas\metric_dict.json"),
    ("phase_queue_v1.json", r"research\deepseek\atlas\phase_queue_v1.json"),
]


def sh8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def demote_line(ln):
    m = re.match(r"^(\s*)(#{1,6})(\s.*)$", ln)
    if m and len(m.group(2)) < 6:
        return m.group(1) + "#" + m.group(2) + m.group(3)
    return ln


def demote_block(text):
    out = []
    fence = False
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith("```") or s.startswith("~~~"):
            fence = not fence
            out.append(ln)
            continue
        out.append(ln if fence else demote_line(ln))
    return out


log = []
def A(s): log.append(s)

# ---- 0) backup ----
os.makedirs(BK, exist_ok=True)
A("== 0) backup 8 docs ==")
info = {}
for N, fn, title, desc in PLAN:
    src = os.path.join(DOCS, fn)
    raw = open(src, "rb").read()
    dst = os.path.join(BK, fn)
    shutil.copy2(src, dst)
    assert open(dst, "rb").read() == raw, "backup mismatch %s" % fn
    info[fn] = (raw, len(raw), sh8(raw))
    A("  OK %-34s %7d B %s" % (fn, len(raw), sh8(raw)))

# ---- 1) read memo + normalize Phase 22 heading ----
old_raw = open(MEMO, "rb").read()
old_txt = old_raw.decode("utf-8-sig")
assert "\r\n" in old_txt
A("")
A("== 1) memo before: bytes=%d lines=%d sha8=%s ==" % (len(old_raw), old_txt.count("\n") + 1, sh8(old_raw)))

OLDH = "## Phase 22: A 闸门 R3-R5b：循环诊断 → 宪法/冻结队列 → Q01/Q02 → Q09/Q12 → seal 请求包（非 Phase 编号）[2026-10-03 02:20]"
NEWH = "## Phase 22: A 闸门 R3-R5b：循环诊断 → 宪法/冻结队列 → Q01/Q02 → Q09/Q12 → seal 请求包 [2026-10-03 02:20]"
# 上面 OLDH 用半角连字符占位，实际用全角/长横线匹配
OLDH = OLDH.replace("R3-R5b", "R3\u2013R5b")
NEWH = NEWH.replace("R3-R5b", "R3\u2013R5b")
c = old_txt.count(OLDH)
A("  Phase22 heading OLD count=%d" % c)
assert c == 1, "heading count=%d" % c
n_before = old_txt.count("（非 Phase 编号）")
old_txt = old_txt.replace(OLDH, NEWH)
n_after = old_txt.count("（非 Phase 编号）")
A("  （非 Phase 编号）marker: %d -> %d" % (n_before, n_after))
assert n_after == n_before - 1

fixed_raw = b"\xef\xbb\xbf" + old_txt.encode("utf-8")
A("  memo fixed: bytes=%d (delta=%d)" % (len(fixed_raw), len(fixed_raw) - len(old_raw)))

# ---- 2) build appended lines ----
L = []
for N, fn, title, desc in PLAN:
    raw, nb, h = info[fn]
    t = raw.decode("utf-8-sig").replace("\r\n", "\n")
    lines = t.split("\n")
    ti = next((i for i, l in enumerate(lines) if l.startswith("# ")), None)
    body_lines = (lines[:ti] + lines[ti + 1:]) if ti is not None else lines
    body = "\n".join(demote_block("\n".join(body_lines))).strip("\n")
    L.append("## Phase %d: 并入 `%s` —— %s（全文）%s" % (N, fn, title, STAMP))
    L.append("")
    L.append("**源文档**：`research/gpt5/docs/%s`（sha8 `%s`，%d B，LF-only）—— 本节为其**全文**（标题统一降一级）。" % (fn, h, nb))
    L.append("**出处处置**：已从 `research/gpt5/docs/` 迁出（备份 `tests/deepseek_temp/_archive_r6/gpt5_docs/%s`），以避免与 G 线混合；后续引用以本 Phase 为准。" % fn)
    L.append("**要点**：%s。" % desc)
    L.append("")
    L.append(body)
    L.append("")

# ---- Phase 31 narrative ----
L.append("## Phase 31: 目录路由归一与 G 线目录迁出（R6）%s" % STAMP)
L.append("")
L.append("### 0 登记约定变更 v4（用户指令，2026-10-03）")
L.append("- **研究日志唯一落点**：`research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md` —— 本对话（deepseek 线）的全部研究日志只写本文件，**不再写入其他任何 `.md`**。")
L.append("- **`AGI_GPT5_MEMO.md` 归属**：该文件用于**其他 AI 路线**的研发日志；本对话不读、不改、不追加。")
L.append("- **三类落点（仅本对话遵守，避免与其他路线混合）**：① 测试脚本 → `tests/deepseek/`；② 临时脚本 → `tests/deepseek_temp/`；③ 测试结果 → `tests/deepseek/result/`。")
L.append("- **历史产物**：`tests/deepseek/Phase1..21/` 与 `tests/deepseek_temp/Phase1..21/` 保持原样（其相对路径已被本 MEMO 多处引用，不做迁移）。")
L.append("- **Phase 22 标题规范化**：删去遗留的「（非 Phase 编号）」标记（append-only 的唯一例外，仅此一处格式修正，已逐字复核）。")
L.append("")
L.append("### 1 并入清单（8 件 → 本文件 Phase 23-30）")
L.append("| Phase | 源文件 | 字节 | sha8 |")
L.append("|---|---|---|---|")
for N, fn, title, desc in PLAN:
    raw, nb, h = info[fn]
    L.append("| %d | `%s` | %d | `%s` |" % (N, fn, nb, h))
L.append("")
L.append("### 2 迁出清单（`research/gpt5/` → deepseek 目录）")
L.append("| 原路径 | 新位置 | 字节 | sha8 |")
L.append("|---|---|---|---|")
for N, fn, title, desc in PLAN:
    raw, nb, h = info[fn]
    L.append("| `research/gpt5/docs/%s` | 并入本文件 Phase %d；原件备份 `tests/deepseek_temp/_archive_r6/gpt5_docs/` | %d | `%s` |" % (fn, N, nb, h))
for fn, tgt in JSON_PLAN:
    fp = os.path.join(ROOT, r"research\gpt5\atlas", fn)
    if os.path.exists(fp):
        b = open(fp, "rb").read()
        L.append("| `research/gpt5/atlas/%s` | `%s` | %d | `%s` |" % (fn, tgt.replace("\\", "/"), len(b), sh8(b)))
    else:
        L.append("| `research/gpt5/atlas/%s` | (缺失) | - | - |" % fn)
L.append("")
L.append("### 3 脚本 / 结果分桶（本对话产物）")
L.append("- **测试脚本**（可复跑 / 复核 / 生成交付件）→ `tests/deepseek/`：`disk_verify_*_r3/r4/r5.py`、`prop_citation_audit.py`、`append_memo_r6.py`、`gen_*_r3/r4/r5.py`、`classify_phases_r3.py`、`verify_final_r2.py`。")
L.append("- **临时脚本**（探针 / 一次性补丁）→ `tests/deepseek_temp/`：`probe_*`、`patch_*`、`do_*_r2.py`、`compact_memory_r3.py`、`_inv_/_tail_/_list_*`。")
L.append("- **测试结果**（txt/json/html 报告）→ `tests/deepseek/result/`（新建）：`disk_verify_*.txt`、`probe_*.txt`、`*_r3/r4/r5.html`、`loop_stats_r3.json`、`phase_classify_r3.json`、`prop_ledger_flat_r5.json`、`memo_append_r2.md` 等。")
L.append("- 本目录约定只约束**本对话**；其他路线（G 线等）不受影响。")
L.append("")
L.append("### 4 一句话 ×3")
L.append("1. **G 线目录已不再承载 deepseek 线产物**（8 件 `.md` 全文并入本文件，6 个 JSON 迁入 deepseek 目录）。")
L.append("2. **研究日志唯一落点 = 本文件**；测试脚本 / 临时脚本 / 测试结果三桶分离，仅本对话遵守。")
L.append("3. **全程未改任何既有判定**：并入为**全文搬运**（标题降级），Phase 22 仅修一处遗留标题标记。")
L.append("")

# ---- 3) write ----
base_lf = old_txt.replace("\r\n", "\n").rstrip("\n")
new_lf = base_lf + "\n" + "\n".join(L)
payload = b"\xef\xbb\xbf" + new_lf.replace("\n", "\r\n").encode("utf-8")
open(MEMO, "wb").write(payload)

# ---- 4) verify ----
new_raw = open(MEMO, "rb").read()
nt = new_raw.decode("utf-8-sig")
bare = nt.count("\n") - nt.count("\r\n")
prefix_ok = new_raw.startswith(fixed_raw.rstrip(b"\r\n"))
phases = re.findall(r"^## Phase (\d+):", nt, re.M)
A("")
A("== 3) memo after ==")
A("  bytes=%d lines=%d sha8=%s bom=%s bare_lf=%d" % (
    len(new_raw), nt.count("\n") + 1, sh8(new_raw), new_raw.startswith(b"\xef\xbb\xbf"), bare))
A("  prefix(固定 Phase22 后) 逐字节保持 = %s" % prefix_ok)
A("  Phase 标题数=%d  末尾 9 个 = %s" % (len(phases), phases[-9:]))
assert prefix_ok, "prefix drift!"
assert bare == 0, "bare_lf=%d" % bare
assert new_raw.startswith(b"\xef\xbb\xbf")
for want in ["23", "24", "25", "26", "27", "28", "29", "30", "31"]:
    assert want in phases, "missing Phase %s" % want

txt = "\n".join(log)
open(REPORT, "w", encoding="utf-8").write(txt)
print(txt)
print("MERGE_OK")
