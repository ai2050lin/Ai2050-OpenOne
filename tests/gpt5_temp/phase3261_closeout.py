# -*- coding: utf-8 -*-
"""Phase 3261 closeout: five writes (result assert / ledger #413 / MEMO / dual wlog / MEMORY)."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
RES = os.path.join(RBASE, "phase3261", "g5h24_evidence_regrading", "result.json")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
WLOGS = [os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
         os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md")]
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")

FREEZE = "d8364fd8"    # formal design sha; SMOKE freeze was cd5d8712
RES_SHA = "dc6c1518"
CHAIN_PREV = "c26305dd"

R = json.load(open(RES, encoding="utf-8"))
assert R["verdict"] == "evidence_regrading_registered" and R["gate_all"], R["verdict"]
assert R["design_sha8"] == FREEZE and R["res_sha8"] == RES_SHA
assert R["arm_a"]["entries"] == 417 and R["arm_a"]["ambiguous_count"] == 16
assert R["arm_a"]["derived_distribution"] == {
    "statistical": 314, "GOVERNANCE": 73, "bit_anchored": 14, "AMBIGUOUS": 16}
assert R["arm_b"]["torch"] == "installed"
print("write1 result assert OK")

# ---------------- write 2: ledger #413 (append-only, chained)
led = json.load(open(LED, encoding="utf-8"))
has413 = any(e.get("phase") == 3261 for e in led["measurements"])
if not has413:
    assert len(led["measurements"]) == 417 and led["ledger_sha256_8"] == CHAIN_PREV, \
        (len(led["measurements"]), led["ledger_sha256_8"])
    entry = {
        "phase": 3261,
        "name": "g5h24_evidence_regrading",
        "title": ("evidence_level normalization re-grading draft (report-only, F2 rule) "
                  "+ GPU feasibility note"),
        "verdict": "evidence_regrading_registered",
        "evidence_level": "report_only_regrading",
        "sources": ["tests/glm5/phase3150_p0_freeze_carrierdecode.py (F2_verdict_grading)",
                    "research/gpt5/atlas/atlas_ledger.json (n=417 post c26305dd)",
                    "tests/glm5/result/rdc_query_construction_20260913/phase3261/g5h24_evidence_regrading/result.json"],
        "detail": ("all 417 entries re-graded against sealed F2_verdict_grading triad: "
                   "statistical 314 / GOVERNANCE 73 (4th non-evidence class for process "
                   "records) / bit_anchored 14 / AMBIGUOUS 16 (flagged for manual review); "
                   "55+ free-text values mapped; identity rule for existing triad values; "
                   "ZERO ledger writes - application would be a destructive migration "
                   "(cf. 3260 op2) needing explicit confirmation, recommend documenting "
                   "the mapping instead. GPU feasibility: RTX 5080 16GB present AND torch "
                   "installed in managed venv -> GPU backlog items (checkpoint training-"
                   "step trajectory, APPX-1..5) are environment-feasible; no env mutation "
                   "performed by this phase"),
        "design_sha8": FREEZE,
        "res_sha8": RES_SHA,
    }
    led["measurements"].append(entry)
    led["ledger_sha256_8"] = CHAIN_PREV
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode("utf-8")).hexdigest()[:8]
    led["ledger_sha256_8"] = new_post
    with open(LED, "w", encoding="utf-8", newline="") as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    print("write2 ledger #413 appended, post", new_post)
else:
    new_post = led["ledger_sha256_8"]
    print("write2 ledger #413 already present (idempotent), post", new_post)

# ---------------- write 3: MEMO section
memo = open(MEMO, encoding="utf-8").read()
SEC = "## Phase 3261 (G5-H24):"
if SEC not in memo:
    block = (
        "## Phase 3261 (G5-H24): evidence_level normalization re-grading (report-only) "
        "+ GPU feasibility [2026-10-21]\n\n"
        "- **verdict `evidence_regrading_registered`（GATE 5/5，零 GPU，零写入账本）**。\n"
        "- **Arm (a) 重判定**：全 417 条对封存 F2_verdict_grading 三级规则（phase3150 逐字锚定）"
        "重判定——**statistical 314 / GOVERNANCE 73 / bit_anchored 14 / AMBIGUOUS 16**"
        "（歧义标记待人工复核）；55+ 自由文本值全映射；triad 内恒等映射；GOVERNANCE 为第 4 类"
        "非证据级（治理/工程/report 记录，零下游前提权重）。产物 `evidence_regrading_draft_v1.json`"
        "（REPORT_ONLY_DRAFT）——**施加＝破坏性迁移（3260 op2 处置），需显式确认；建议以映射文档"
        "取代改写**。\n"
        "- **Arm (b) GPU 可行性**：RTX 5080 16GB 在位 **且 managed venv torch 已安装**——"
        "GPU 挂账项（checkpoint 训练步轨迹、APPX-1..5）环境可行；本 phase 零环境变更，"
        "启动与否待预注册协议设计。\n"
        "- ledger #413 / freeze **`%s`**（SMOKE `cd5d8712`）/ res **`%s`**。\n"
        "- **3262 已预注册**：(a) GPU 臂启动评估（checkpoint 在位性核查 + checkpoint 训练步轨迹"
        "预注册协议设计；APPX-1..5 逐项协议草案）——环境障碍已排除，剩余为协议与算时决策；"
        "(b) 两项用户确认提案（水果类/K4 删除、lcorr op4 施加）继续待裁决。\n" % (FREEZE, RES_SHA)
    )
    memo = memo.rstrip("\n") + "\n\n" + block
    with open(MEMO, "w", encoding="utf-8", newline="") as f:
        f.write(memo)
    print("write3 MEMO section appended")
else:
    print("write3 MEMO already present (idempotent)")

# ---------------- write 4: dual wlogs
for p in WLOGS:
    t = open(p, encoding="utf-8").read()
    if "3261 G5-H24" not in t:
        line = ("- 3261 G5-H24 evidence_regrading_registered（GATE 5/5，零 GPU 零写入）：全 417 条对"
                "封存 F2 三级规则重判定——statistical 314 / GOVERNANCE 73 / bit_anchored 14 / "
                "AMBIGUOUS 16；55+ 自由值全映射、triad 恒等；GOVERNANCE＝第 4 非证据类；施加＝破坏性迁移"
                "需确认，建议映射文档取代改写。GPU 可行性：RTX 5080 + torch installed——GPU 挂账环境可行，"
                "启动待协议设计。freeze %s / res %s / ledger #413 post %s。"
                % (FREEZE, RES_SHA, new_post))
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(t.rstrip("\n") + "\n" + line + "\n")
        print("write4 wlog appended:", os.path.basename(p))
    else:
        print("write4 wlog already present (idempotent):", os.path.basename(p))

# ---------------- write 5: MEMORY (add block + compress, idempotent)
mem = open(MEMORY, encoding="utf-8").read()
if "**3261" not in mem:
    new_block = ("- **3261 证据级重判定**：417 条对 F2 三级规则重推——stat 314/GOV 73/bit 14/"
                 "AMBIGUOUS 16；零写入，施加=破坏性迁移待确认；GPU 挂账环境可行（torch 在）。#413。")
    mem = mem.rstrip("\n") + "\n" + new_block + "\n"

if len(mem) > 3000:
    o = "- **3257 终审**：final_audit_passed——三端实机+磁盘终审（anchors 28/28、gap 锚 6/6）；N 线预演；收线包可交付。#409。"
    n = "- **3257 终审**：final_audit_passed——三端实机+磁盘终审；N 线预演；收线包可交付。#409。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3255**：gaps 增 appendix + GapLedgerPanel（实机 17/17）；MEMO 定谳双轨道几何——INFORMATION_GAP 词表侧不可补，XC_p5_b8 −0.449 即收线终点。#407。"
    n = "- **3255**：gaps+GapLedgerPanel（17/17）；MEMO 定谳双轨道几何——INFORMATION_GAP 词表侧不可补，XC_p5_b8 −0.449 收线终点。#407。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3254 落册**：gap v1.12 `aa8cff21`（APPX-6）+ registry v1.34 `3deabf25`（anchors ×4+log）；REVERT 逐位、审计 6/6。#406。"
    n = "- **3254 落册**：gap v1.12（APPX-6）+ registry v1.34（anchors ×4+log）；REVERT 逐位、审计 6/6。#406。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3256 封章**：收线链 CLOSED（74.36%）；registry v1.35 seal 注记；N 线 manifest。#408。"
    n = "- **3256 封章**：收线链 CLOSED（74.36%）；registry v1.35 注记；N 线 manifest。#408。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3246**：FLIP=INFORMATION_GAP；ORC +0.329；XC_b4 −0.772。#398。"
    n = "- **3246**：FLIP=INFORMATION_GAP；ORC +0.329。#398。"
    if o in mem:
        mem = mem.replace(o, n)
with open(MEMORY, "w", encoding="utf-8", newline="") as f:
    f.write(mem)
print("write5 MEMORY len", len(mem))
assert len(mem) <= 3000, len(mem)
print("CLOSEOUT OK #413 post", new_post)
