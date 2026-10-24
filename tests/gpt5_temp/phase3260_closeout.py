# -*- coding: utf-8 -*-
"""Phase 3260 closeout: five writes (result assert / ledger #412 / MEMO / dual wlog / MEMORY)."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
RES = os.path.join(RBASE, "phase3260", "g5h23_lcorr_rederivation", "result.json")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
WLOGS = [os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
         os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md")]
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")

FREEZE = "f962faf0"    # formal design sha; SMOKE freeze was 918d5816
RES_SHA = "209fe48d"
CHAIN_PREV = "7b6e6230"
LCORR_SHA = "beb775bd"

R = json.load(open(RES, encoding="utf-8"))
assert R["verdict"] == "lcorr_rederivation_registered" and R["gate_all"], R["verdict"]
assert R["design_sha8"] == FREEZE and R["res_sha8"] == RES_SHA
assert R["arm_a"]["op_dispositions"] == {
    "op1_set_field": "OBSOLETE_SUPERSEDED",
    "op2_unify_schema": "REQUIRES_DESTRUCTIVE_MIGRATION",
    "op3_enumerate_field": "SUPERSEDED_BY_PRACTICE",
    "op4_append_agate": "READY_PENDING_CONFIRMATION"}
assert R["arm_a"]["sealed_draft"]["sha8"] == LCORR_SHA
print("write1 result assert OK")

# ---------------- write 2: ledger #412 (append-only, chained)
led = json.load(open(LED, encoding="utf-8"))
has412 = any(e.get("phase") == 3260 for e in led["measurements"])
if not has412:
    assert len(led["measurements"]) == 416 and led["ledger_sha256_8"] == CHAIN_PREV, \
        (len(led["measurements"]), led["ledger_sha256_8"])
    entry = {
        "phase": 3260,
        "name": "g5h23_lcorr_rederivation",
        "title": "lcorr re-derivation v2 (report-only) + backlog inventory v3 + user proposal",
        "verdict": "lcorr_rederivation_registered",
        "evidence_level": "report_only_rederivation",
        "sources": ["research/deepseek/atlas/ledger_corrections_v1.json (beb775bd)",
                    "research/gpt5/atlas/atlas_ledger.json (n=416 post 7b6e6230)",
                    "tests/glm5/result/rdc_query_construction_20260913/phase3260/g5h23_lcorr_rederivation/result.json"],
        "detail": ("op1 OBSOLETE_SUPERSEDED (chained convention supersedes one-shot set_field, "
                   "g1 recompute 7b6e6230); op2 REQUIRES_DESTRUCTIVE_MIGRATION (meas_id 270/phase 311/"
                   "both 165 at n=416; byte-anchored history must not be rewritten -> document "
                   "convention instead); op3 SUPERSEDED_BY_PRACTICE (55+ distinct evidence_level "
                   "values, statistical 310/416; successor = F2 re-grading backlog item); "
                   "op4 READY_PENDING_CONFIRMATION (A-gate governance entry still absent, closure "
                   "record a_gate_closure_v1.json verified); zero writes to shared ledger and "
                   "deepseek sealed files; backlog v3 closed N-line-apply + lcorr re-derivation"),
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
    print("write2 ledger #412 appended, post", new_post)
else:
    new_post = led["ledger_sha256_8"]
    print("write2 ledger #412 already present (idempotent), post", new_post)

# ---------------- write 3: MEMO section
memo = open(MEMO, encoding="utf-8").read()
SEC = "## Phase 3260 (G5-H23):"
if SEC not in memo:
    block = (
        "## Phase 3260 (G5-H23): lcorr re-derivation (report-only) + backlog v3\n\n"
        "- **verdict `lcorr_rederivation_registered`（GATE 5/5，零 GPU，零写入共享账本/deepseek）**。\n"
        "- **Arm (a) 重推导**：对现账 n=416 post `7b6e6230` 重推封存补丁 `beb775bd` 的 4 操作——"
        "op1 **OBSOLETE_SUPERSEDED**（链式约定取代一次性 set_field，g1 recompute 通过）；"
        "op2 **REQUIRES_DESTRUCTIVE_MIGRATION**（meas_id 270 / phase 311 / both 165；字节锚定历史不可改写，"
        "建议以文档约定取代迁移）；op3 **SUPERSEDED_BY_PRACTICE**（evidence_level 55+ 值实践、statistical 310/416，"
        "三值枚举不可表达治理/report 臂——后继动作为 F2_verdict_grading 规范化重判定，入挂账 v3）；"
        "op4 **READY_PENDING_CONFIRMATION**（A-gate 治理登记仍不在账，closure record "
        "`a_gate_closure_v1.json` 复核，施加待用户/所有者确认，协议同 3258 append-only 先例）。"
        "产物 `lcorr_rederivation_v2.json`（REPORT_ONLY_DRAFT）。\n"
        "- **Arm (b) 挂账清单 v3 + 用户提案**：closed_since_v2 ×2（N 线施加 3258/3259；lcorr 重推导 3260）；"
        "open_gpu ×3（checkpoint 轨迹 / APPX-1..5 / evidence_level 规范化）；"
        "pending_user_confirmation ×2（水果类/K4 MEMO 列表删除；lcorr v2 施加）。"
        "一页式提案 `user_confirmation_proposal.md`。\n"
        "- **零写入断言**：共享账本、lcorr、deepseek MEMO 三文件 sha8 前后相等（g5）；lcorr 封存字节不动（g2）。\n"
        "- ledger #412 / freeze **`%s`**（SMOKE `918d5816`）/ res **`%s`**。\n"
        "- **3261 已预注册**：等待用户对两项确认提案的裁决（水果类/K4 删除、lcorr op4 施加）；"
        "GPU 挂账维持；无默认新臂。\n" % (FREEZE, RES_SHA)
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
    if "3260 G5-H23" not in t:
        line = ("- 3260 G5-H23 lcorr_rederivation_registered（GATE 5/5，零 GPU 零写入）：封存补丁 beb775bd "
                "4 操作对现账 n=416 重推导——op1 作废（链式取代）/op2 破坏性迁移不建议/op3 枚举被 55+ 值实践超越"
                "（改挂 evidence_level 规范化）/op4 A-gate 治理登记 READY_PENDING_CONFIRMATION；"
                "零写入断言 g5 三文件 sha8 前后相等；挂账 v3（closed ×2/open_gpu ×3/pending ×2）+ "
                "用户提案一页式。freeze %s / res %s / ledger #412 post %s。"
                % (FREEZE, RES_SHA, new_post))
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(t.rstrip("\n") + "\n" + line + "\n")
        print("write4 wlog appended:", os.path.basename(p))
    else:
        print("write4 wlog already present (idempotent):", os.path.basename(p))

# ---------------- write 5: MEMORY (add block + compress, idempotent)
mem = open(MEMORY, encoding="utf-8").read()
if "**3260" not in mem:
    new_block = ("- **3260 lcorr 重推导**：4 操作对现账 n=416 重推——op1 作废/op2 不迁移/op3 被实践超越"
                 "（改挂 F2 重判定）/op4 A-gate 待确认；零写入；v2 稿+挂账 v3+提案落 phase3260。#412。")
    mem = mem.rstrip("\n") + "\n" + new_block + "\n"

if len(mem) > 3000:
    o = "- **3257 终审**：final_audit_passed——三端实机+磁盘终审（anchors 28/28、gap 锚 6/6、#408 recompute）；N 线预演。收线包可交付。#409。"
    n = "- **3257 终审**：final_audit_passed——三端实机+磁盘终审（anchors 28/28、gap 锚 6/6）；N 线预演；收线包可交付。#409。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3256 封章**：收线链 CLOSED（修复 74.36%）；registry v1.35 `e8335331` seal 注记；N 线 manifest。#408。"
    n = "- **3256 封章**：收线链 CLOSED（74.36%）；registry v1.35 seal 注记；N 线 manifest。#408。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ("- **3255**：/atlas/gaps 增 appendix + GapLedgerPanel 入 AtlasLens（实机 17/17）；"
         "MEMO 定谳**双轨道几何**——INFORMATION_GAP 词表侧不可补，XC_p5_b8 −0.449 即收线终点。#407。")
    n = ("- **3255**：gaps 增 appendix + GapLedgerPanel（实机 17/17）；MEMO 定谳双轨道几何——"
         "INFORMATION_GAP 词表侧不可补，XC_p5_b8 −0.449 即收线终点。#407。")
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3254 落册**：gap v1.12 `aa8cff21`（APPX-6）+ registry v1.34 `3deabf25`（anchors ×4+log）；REVERT 逐位、审计 6/6；features/failures 零改动。#406。"
    n = "- **3254 落册**：gap v1.12 `aa8cff21`（APPX-6）+ registry v1.34 `3deabf25`（anchors ×4+log）；REVERT 逐位、审计 6/6。#406。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "零写入；v2 稿+挂账 v3+提案落 phase3260。#412。"
    n = "零写入；v2 稿/v3/提案落 phase3260。#412。"
    if o in mem:
        mem = mem.replace(o, n)
with open(MEMORY, "w", encoding="utf-8", newline="") as f:
    f.write(mem)
print("write5 MEMORY len", len(mem))
if len(mem) > 3000:
    o = ("- **3250**：deg 5–6 停点；**XC_p5_b8 现役**（−0.449，r2 0.241/KL 0.987）；"
         "混合核≡poly3；fidelity v2+UI 19/19。#402。")
    n = "- **3250**：deg 5–6 停点；**XC_p5_b8 现役**（−0.449）；混合核≡poly3；fidelity v2+UI 19/19。#402。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3251**：词级对齐小、随规模衰减、非"
    n = "- **3251**：词级对齐小、随规模衰减、"
    if o in mem:
        mem = mem.replace(o, n)
assert len(mem) <= 3000, len(mem)
print("CLOSEOUT OK #412 post", new_post)
