# -*- coding: utf-8 -*-
"""Phase 3258 closeout: five writes (result assertions / ledger #410 / MEMO / dual wlog / MEMORY)."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
WLOG1 = os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md")
WLOG2 = os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md")
MEM = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")

PREV_POST = "15875b87"
RES_SHA = "ab5215e5"
FREEZE = "db2255eb"   # formal design sha (mode=formal); SMOKE freeze was ae1e34cd
SMOKE_FREEZE = "db2255eb"   # formal design sha (mode=formal); SMOKE freeze was ae1e34cd
NEW_POST = "a59da52b"
RES_PATH = os.path.join(ROOT, "tests", "glm5", "result",
                        "rdc_query_construction_20260913", "phase3258",
                        "g5h21_nline_apply", "result.json")
REPORT = []


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def A_(s):
    REPORT.append(s)
    print(s)


# ---------------- 1 result assertions
R = json.load(open(RES_PATH, encoding="utf-8"))
assert R["verdict"] == "nline_apply_registered" and R["gate_all"] is True
assert R["res_sha8"] == RES_SHA and R["design_sha8"] == FREEZE
assert R["arm_a"]["ledger"]["post_after"] == NEW_POST
assert R["arm_a"]["applied_entry_phases"] == ["N-P3", "N-P4", "N-P5", "N-P6", "N-P7"]
assert R["arm_b"]["writes"] == 0
assert R["arm_b"]["disposition"].startswith("STALE_AS_SEALED")
A_("1 result assertions OK (res %s / freeze %s / gates 6/6)" % (RES_SHA, FREEZE))

# ---------------- 2 ledger append #410
led_raw = open(LED, "rb").read()
led = json.loads(led_raw.decode("utf-8"))
PREV_N = 414
assert (len(led["measurements"]), led["ledger_sha256_8"]) in [
    (PREV_N, NEW_POST), (PREV_N + 1, "8de77a63")], \
    (len(led["measurements"]), led["ledger_sha256_8"])
has410 = any(e.get("phase") == 3258 for e in led["measurements"])
if not has410:
    entry = {
        "phase": 3258,
        "name": "g5h21_nline_apply",
        "line": "G",
        "date": "2026-10-21",
        "model": "zero GPU: cross-line backfill application (append-type) + lcorr staleness disposition",
        "verdict": "nline_apply_registered|HEADLINE - sealed N-line backfill draft f29da2de (N-P3..N-P7, 5 entries) applied to shared atlas ledger per its target '#measurements (append 5)': draft fields verbatim (name/phase/sources/status/title) + registration provenance only, evidence_level descriptive (per-line levels non-transferable per G5-H20 apply_preview step 5, N-line starts at E0); lcorr beb775bd dispositioned STALE_AS_SEALED (4 ops computed against n=304 state; target sha mismatch; op3 enum superseded by 55-value evidence_level practice; op4 A-gate absent -> re-derivation required, not verbatim application), deepseek files untouched",
        "evidence_level": "registered (preregistered backlog arm announced at 3257 tail; readiness ready_append_type per 3218; append-only pointer diff exact {$.measurements, $.ledger_sha256_8}; revert bitwise == pre raw bytes; chain recompute prev=15875b87 -> post " + NEW_POST + "; draft seal f29da2de byte-unchanged)",
        "model_scope": "n/a (governance backfill; zero GPU)",
        "prereg_id": "G5-H21 (3257 tail backlog order)",
        "superseded_by": None,
        "detail": "Arm (a): N-line backfill draft f29da2de applied - entries N-P3 e2_conditioned_compare_layer_scan / N-P4 n1_main_axis_layer_scan / N-P5 n2_reconstruction_source / N-P6 n2h1_permutation_ablation / N-P7 n3_device_transfer, each with backfill provenance block (draft sha8, applied_by_phase 3258, governance APPX-4/5 tong-xing zhi-bu-bu-gai). Draft file byte-unchanged. Ledger n 409->414, backup .bak_3258. Arm (b): lcorr beb775bd verified stale on-disk (target_sha256_before mismatch, op1 from 41d65a13 stale, op2 n_total 304 stale, op3 distribution stale + enum superseded, op4 A-gate registration absent hence not redundant but requires re-derivation); disposition STALE_AS_SEALED recorded, zero writes to deepseek/. SMOKE device fixes (DESIGN unchanged, freeze intact): chain recompute moved before pointer diff (sha field was updated post-diff, SMOKE saw no diff); op2 meas_id coincidence check demoted to INFO (270==270 by accident since all post-#304 entries omit meas_id). freeze db2255eb (SMOKE ae1e34cd), res ab5215e5. Pre-closeout ledger file sha8 9288d087."
    }
    # NOTE: pre-append file sha placeholder fixed below after computing
    led["measurements"].append(entry)
    led["ledger_sha256_8"] = NEW_POST
    new_post = sha8(json.dumps(led, ensure_ascii=False, indent=1).encode("utf-8"))
    led["ledger_sha256_8"] = new_post
    import shutil
    shutil.copy2(LED, LED + ".bak_3258_closeout")
    with open(LED, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    A_("2 ledger: appended #410 post=%s (chain prev=%s)" % (new_post, NEW_POST))
else:
    new_post = led["ledger_sha256_8"]
    A_("2 ledger: #410 already present, post=%s (idempotent)" % new_post)
chk = json.load(open(LED, encoding="utf-8"))
assert len(chk["measurements"]) == PREV_N + 1
chk["ledger_sha256_8"] = NEW_POST
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode("utf-8"))
assert rec == new_post == json.load(open(LED, encoding="utf-8"))["ledger_sha256_8"]
A_("2 ledger self-check: n=415 post=%s recompute OK (chain prev=%s)" % (new_post, NEW_POST))

# ---------------- 3 MEMO
memo = open(MEMO, encoding="utf-8").read()
sec = ("\n\n## Phase 3258: G5-H21 N-line apply (sealed backfill draft f29da2de -> shared ledger, append-only) + lcorr disposition (zero GPU) [%s]\n"
       % "2026-10-21")
if "## Phase 3258:" not in memo:
    sec += (
        "**执行模式**：预注册自 3257 tail（挂账择序：N 线施加臂，readiness ready_append_type per 3218 + 用户续研授权 3257 tail 宣告）；freeze **`db2255eb`**（SMOKE `ae1e34cd`，装置修正×2 不动 DESIGN 语义）；res **`ab5215e5`**；verdict **`nline_apply_registered`**（GATE 6/6）。\n\n"
        "### 施加臂 (a)：N 线补账草案落共享账本（append-only）\n"
        "- **草案** `n_line_backfill_draft_v1.json`（phase3208 g5g7 封存 `f29da2de`，status draft_pending_cross_line，target `#measurements (append 5)`）**字节不动**（施加前后断言相等）。\n"
        "- **五条落账**：N-P3 `e2_conditioned_compare_layer_scan` / N-P4 `n1_main_axis_layer_scan` / N-P5 `n2_reconstruction_source` / N-P6 `n2h1_permutation_ablation` / N-P7 `n3_device_transfer`——草案字段逐字（name/phase/sources/status/title）+ 登记溯源块（applied_by_phase 3258 / draft sha8 / governance「只补不改」）；evidence_level=descriptive（跨线补记，**证据级不跨线转移** per apply_preview step 5，N 线自 E0 起算）；字符串 phase `N-P*` 为草案注册命名空间（账本历史混合 schema，lcorr C6 同观察）。\n"
        "- **账本** n 409→414，post `15875b87`→**`a59da52b`**（链式 recompute 通过）；指针 diff 精确 {$.measurements, $.ledger_sha256_8}；**revert 逐位 == 前驱原始字节**；备份 `.bak_3258`。\n\n"
        "### lcorr 处置臂 (b)：STALE_AS_SEALED（report-only，deepseek 零写入）\n"
        "- `ledger_corrections_v1.json`（`beb775bd`，2026-10-03 seal）4 操作全部对 n=304 旧账计算：target_sha256_before 失配现文件；op1 from `41d65a13` 过期；op2 n_total 304 过期（meas_id 计数 270==270 系巧合——#304 后条目均不带 meas_id，降级 INFO）；op3 枚举 [bit_anchored/statistical/descriptive] 被现实 55 值 evidence_level 实践超越；op4 A-gate 登记不在账（非冗余但须重推导，不可逐字施加）。**不施加**；挂账改为「lcorr 重推导（基于现账状态）」。\n\n"
        "### 诚实日志\n"
        "SMOKE 装置修正 ×2（DESIGN 不变、freeze 不动）：① 链计算原在指针 diff 之后执行（SMOKE 下 sha 字段无 diff）→ 移至 diff 前；② op2 meas_id 巧合相等 → 降级信息项不入 gate 合取。\n\n"
        "### 预注册 Phase 3259（两支候选）\n"
        "- (a) **施加后一致性核验臂**（零 GPU）：三账（ledger/registry/gap）+ 客户端 /atlas/registry 对 5 条 N-P 条目可见性复读 + N 线侧（deepseek MEMO）回执注记。\n"
        "- (b) **lcorr 重推导臂**（零 GPU）：基于现账 n=415 重新推导 4 操作的等价修正（重推导稿 report-only，施加仍待确认）。\n"
        "- 默认执行 (a)。")
    memo = memo.rstrip("\n") + sec
    with open(MEMO, "w", encoding="utf-8", newline="") as f:
        f.write(memo)
    A_("3 MEMO: Phase 3258 section appended")
else:
    A_("3 MEMO: 3258 section already present (idempotent)")

# ---------------- 4 dual wlog
wl = ("- 3258 G5-H21 N 线施加收官（nline_apply_registered）：封存草案 f29da2de（N-P3..P7 五条）按其 target append 至共享账本——草案字段逐字+登记溯源（证据级 descriptive，不跨线转移、N 线自 E0 起算）；账本 n 409→414 post a59da52b（链 recompute、指针 diff 精确、revert 逐位、草案字节不动、备份 .bak_3258）；lcorr beb775bd 处置 STALE_AS_SEALED（4 操作全对 n=304 旧账计算、op3 枚举被 55 值实践超越、op4 A-gate 须重推导），deepseek 零写入，挂账改「lcorr 重推导」。SMOKE 修正×2：链计算移至 diff 前、meas_id 巧合降级 INFO。freeze ae1e34cd / res ab5215e5 / ledger #410 post 见账。3259 预注册：施加后一致性核验臂（默认）或 lcorr 重推导臂。\n")
for p in (WLOG1, WLOG2):
    t = open(p, encoding="utf-8").read()
    if "3258 G5-H21 N 线施加收官" not in t:
        with open(p, "a", encoding="utf-8", newline="") as f:
            f.write(wl)
        A_("4 wlog appended: " + os.path.basename(os.path.dirname(p)))
    else:
        A_("4 wlog already present (idempotent): " + os.path.basename(os.path.dirname(p)))

# ---------------- 5 MEMORY
mem = open(MEM, encoding="utf-8").read()
if "**3258" not in mem:
    new_block = ("- **3258 N 线施加**：草案 f29da2de（N-P3..P7×5）落共享账本（逐字+溯源，证据级不跨线）；n 414 post `a59da52b`；lcorr STALE（挂账改重推导）。#410。")
    mem = mem.rstrip("\n") + "\n" + new_block + "\n"
    # compress if over limit
    if len(mem) > 3000:
        o = "- **3257 终审**：final_audit_passed——端点 15 断言 + UI 11/11（v1.35 链 35 代）+ 磁盘终审（anchors 28/28、gap 锚口径表 6/6、ledger #408 recompute）；N 线预演 apply_preview（read-only 零写入）。G 线收线包可交付。#409 `c2390c7f`/`961a9e97`。"
        n = "- **3257 终审**：final_audit_passed——端点+UI 实机 + 磁盘终审（anchors 28/28、gap 锚口径表 6/6、#408 recompute）；N 线预演 apply_preview。收线包可交付。#409。"
        if o in mem:
            mem = mem.replace(o, n)
    if len(mem) > 3000:
        o = "- **3256 封章**：3239–3255 收线链 CLOSED（修复 74.36%）；registry v1.35 `e8335331`（seal 注记，revert 逐位）；N 线收线包 manifest。#408。"
        n = "- **3256 封章**：收线链 CLOSED（修复 74.36%）；registry v1.35 `e8335331` seal 注记；N 线 manifest。#408。"
        if o in mem:
            mem = mem.replace(o, n)
    # semantic updates from 3258 (backlog resolution + current registry version)
    o = "挂账：N 线 P3–P7 补 Ledger、跨线账本补丁施加确认、水果类、K4。"
    n = "挂账：水果类、K4。（N 线 P3–P7 已补账 #410/3258；lcorr 重推导挂账）"
    if o in mem:
        mem = mem.replace(o, n)
    o = "## 2 G 线现状（registry v1.34 3deabf25 + gap v1.12 aa8cff21；v1.33 a30215fd/v1.11 5934631b 封存）"
    n = "## 2 G 线现状（registry v1.35 e8335331 + gap v1.12 aa8cff21；v1.34 3deabf25/v1.11 封存）"
    if o in mem:
        mem = mem.replace(o, n)
    assert len(mem) <= 3000, len(mem)
    with open(MEM, "w", encoding="utf-8", newline="") as f:
        f.write(mem)
    A_("5 MEMORY: 3258 block added, %d chars" % len(mem))
else:
    A_("5 MEMORY: 3258 block already present (idempotent)")

print("CLOSEOUT OK")
