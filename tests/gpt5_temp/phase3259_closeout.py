# -*- coding: utf-8 -*-
"""Phase 3259 closeout: five writes (result assertions / ledger #411 / MEMO / dual wlog / MEMORY)."""
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
WLOG1 = os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md")
WLOG2 = os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md")
MEM = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")

CHAIN_PREV = "8de77a63"
RES_SHA = "f79e477d"
FREEZE = "3de50b8d"     # formal design sha (mode=formal); SMOKE freeze 49925445
RES_PATH = os.path.join(ROOT, "tests", "glm5", "result",
                        "rdc_query_construction_20260913", "phase3259",
                        "g5h22_postapply_consistency", "result.json")
REPORT = []


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def A_(s):
    REPORT.append(s)
    print(s)


# ---------------- 1 result assertions
R = json.load(open(RES_PATH, encoding="utf-8"))
assert R["verdict"] == "postapply_consistency_confirmed" and R["gate_all"] is True
assert R["res_sha8"] == RES_SHA and R["design_sha8"] == FREEZE
assert R["arm_a"]["np_entries"] == ["N-P3", "N-P4", "N-P5", "N-P6", "N-P7"]
assert R["arm_a"]["np_verbatim"] is True
assert all(v == e for v, e in zip(
    sorted(R["arm_a"]["sealed_untouched"].values()),
    sorted(["e8335331", "3deabf25", "aa8cff21", "f29da2de", "beb775bd"])))
assert R["arm_b"]["write"] == "append-only receipt"
A_("1 result assertions OK (res %s / freeze %s / gates 5/5)" % (RES_SHA, FREEZE))

# ---------------- 2 ledger append #411
led_raw = open(LED, "rb").read()
led = json.loads(led_raw.decode("utf-8"))
PREV_N = 415
has411 = any(e.get("phase") == 3259 for e in led["measurements"])
assert ((len(led["measurements"]) == PREV_N
         and led["ledger_sha256_8"] == CHAIN_PREV)
        or (len(led["measurements"]) == PREV_N + 1 and has411)), \
    (len(led["measurements"]), led["ledger_sha256_8"])
if not has411:
    pre_file_sha8 = sha8(led_raw)
    entry = {
        "phase": 3259,
        "name": "g5h22_postapply_consistency",
        "line": "G",
        "date": "2026-10-21",
        "model": "zero GPU: post-apply consistency recheck + deepseek MEMO receipt",
        "verdict": "postapply_consistency_confirmed|HEADLINE - 3258 N-line apply verified end-to-end after the fact: ledger chain recompute (n=415 post " + CHAIN_PREV + " from prev a59da52b), 5 N-P entries verbatim vs sealed draft f29da2de (name/phase/sources/status/title + backfill provenance), #410 present exactly once, sealed accounts byte-untouched (registry v1.35 e8335331 / v1.34 3deabf25 / gap v1.12 aa8cff21 / draft f29da2de / lcorr beb775bd); live client /atlas/ledger?line=N returns exactly the 5 backfill entries newest-first with post " + CHAIN_PREV + " (11 asserts PASS); deepseek MEMO receipt appended (append-only, marker '## 跨线施加回执（G 线 Phase 3258）') recording backlog resolution: 'N-line P3-P7 ledger backfill' CLOSED, 'cross-line patch confirmation' -> lcorr re-derivation backlog (STALE_AS_SEALED)",
        "evidence_level": "registered (preregistered 3258 tail default arm (a); all checks disk-recomputed or live-endpoint; zero writes to any sealed account; single append-only receipt to deepseek MEMO)",
        "model_scope": "n/a (consistency audit; zero GPU)",
        "prereg_id": "G5-H22 (3258 tail)",
        "superseded_by": None,
        "detail": "Arm (a) disk: ledger n=415 post 8de77a63 chain recompute from a59da52b OK; N-P3..N-P7 entries verbatim (draft fields + backfill.applied_by_phase=3258 + line=N); sealed set sha8 re-measured all match (e8335331/3deabf25/aa8cff21/f29da2de/beb775bd). Live: /atlas/ledger total=415, line=N count=5 (N-P7..N-P3 newest-first), names verbatim, prereg_id cites f29da2de, evidence_level descriptive; head top = #410 g5h21_nline_apply; /atlas/registry v1.35 + /atlas/gaps v1.12 appendix 6 unchanged; line=G unaffected (11/11 PASS, zero FAIL). Arm (b): deepseek AGI_DEEPSEEK_MEMO.md receipt appended (append-only): backlog 'N-line P3-P7 ledger backfill' CLOSED via 3258; 'cross-line ledger patch confirmation' updated -> lcorr beb775bd STALE_AS_SEALED (3258), backlog renamed 'lcorr re-derivation (current-state basis)', application still pending confirmation. SMOKE device fixes (DESIGN unchanged, freeze intact): PASS count by line-start (summary line '11 PASS' double-counted); receipt gate None in SMOKE (write arm deferred). freeze 3de50b8d (SMOKE 49925445), res f79e477d. Pre-closeout ledger file sha8 " + pre_file_sha8[:8] + "."
    }
    led["measurements"].append(entry)
    led["ledger_sha256_8"] = CHAIN_PREV
    new_post = sha8(json.dumps(led, ensure_ascii=False, indent=1).encode("utf-8"))
    led["ledger_sha256_8"] = new_post
    shutil.copy2(LED, LED + ".bak_3259_closeout")
    with open(LED, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    A_("2 ledger: appended #411 post=%s (chain prev=%s)" % (new_post, CHAIN_PREV))
else:
    new_post = led["ledger_sha256_8"]
    A_("2 ledger: #411 already present, post=%s (idempotent)" % new_post)
chk = json.load(open(LED, encoding="utf-8"))
assert len(chk["measurements"]) == PREV_N + 1
chk["ledger_sha256_8"] = CHAIN_PREV
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode("utf-8"))
assert rec == new_post == json.load(open(LED, encoding="utf-8"))["ledger_sha256_8"], \
    "ledger post recompute mismatch"
A_("2 ledger self-check: n=416 post=%s recompute OK (chain prev=%s)" % (new_post, CHAIN_PREV))

# ---------------- 3 MEMO
memo = open(MEMO, encoding="utf-8").read()
if "## Phase 3259:" not in memo:
    sec = ("\n\n## Phase 3259: G5-H22 施加后一致性核验（三账+客户端实机）+ deepseek MEMO 回执（零 GPU） [2026-10-21]\n"
           "**执行模式**：预注册自 3258 tail 默认臂 (a)；freeze **`3de50b8d`**（SMOKE `49925445`）；res **`f79e477d`**；verdict **`postapply_consistency_confirmed`**（GATE 5/5）。\n\n"
           "### 核验臂 (a)：3258 施加事后全链复读\n"
           "- **账本**：n=415 post `8de77a63` 链式 recompute（自 prev `a59da52b`）OK；N-P3..N-P7 五条与封存草案 `f29da2de` 逐字一致（name/phase/sources/status/title + backfill 溯源）；#410 恰一次。\n"
           "- **封存集字节不动**：registry v1.35 `e8335331` / v1.34 `3deabf25` / gap v1.12 `aa8cff21` / 草案 `f29da2de` / lcorr `beb775bd` 五文件 sha8 实测全匹配。\n"
           "- **客户端实机 11 断言全过**：/atlas/ledger total=415、line=N 恰 5 条（N-P7..N-P3 最新在前、name 逐字、prereg_id 引 f29da2de、evidence_level descriptive）、head 顶 = #410；/atlas/registry v1.35 + /atlas/gaps v1.12 appendix 6 无扰；line=G 不受扰。\n\n"
           "### 回执臂 (b)：deepseek MEMO 回执（append-only）\n"
           "AGI_DEEPSEEK_MEMO.md 尾部追加「跨线施加回执（G 线 Phase 3258）」：挂账「N 线 P3–P7 补 Ledger」**结清**；「跨线账本补丁施加确认」更新为 lcorr `beb775bd` **STALE_AS_SEALED**（3258 实测），挂账改「lcorr 重推导（基于现账状态）」，施加仍待确认。本线封存文件零改动。\n\n"
           "### 诚实日志\n"
           "SMOKE 装置修正 ×2（DESIGN 不变、freeze 不动）：① PASS 计数按行首统计（摘要行「11 PASS」被重复计入）；② 回执 gate 于 SMOKE 下置 None（写臂延迟到 formal）。\n\n"
           "### 预注册 Phase 3260（两支候选）\n"
           "- (a) **lcorr 重推导臂**（零 GPU report-only）：基于现账 n=416 重推导 4 操作等价修正稿（report-only，施加待确认）。\n"
           "- (b) **G 线挂账盘点 + 水果类/K4 用户确认提案**：挂账清单 v3 + 待确认项一页式说明（供用户决策）。\n"
           "- 默认执行 (a)。")
    memo = memo.rstrip("\n") + sec
    with open(MEMO, "w", encoding="utf-8", newline="") as f:
        f.write(memo)
    A_("3 MEMO: Phase 3259 section appended")
else:
    A_("3 MEMO: 3259 section already present (idempotent)")

# ---------------- 4 dual wlog
wl = ("- 3259 G5-H22 施加后一致性核验收官（postapply_consistency_confirmed）：账本 n=415 post 8de77a63 链 recompute；N-P3..P7 五条与封存草案 f29da2de 逐字一致；封存集五文件 sha8 实测不动；客户端实机 11 断言（line=N 恰 5 条最新在前 + head 顶 #410 + registry/gaps 无扰 + line=G 不受扰）；deepseek MEMO 回执 append-only（挂账「N 线 P3–P7 补 Ledger」结清，lcorr STALE_AS_SEALED→挂账改重推导）。SMOKE 修正×2：PASS 按行首计数、回执 gate SMOKE 置 None。freeze 3de50b8d (SMOKE 49925445) / res f79e477d / ledger #411。3260 预注册：lcorr 重推导臂（默认）或挂账盘点+用户确认提案。\n")
for p in (WLOG1, WLOG2):
    t = open(p, encoding="utf-8").read()
    if "3259 G5-H22" not in t:
        with open(p, "a", encoding="utf-8", newline="") as f:
            f.write(wl)
        A_("4 wlog appended: " + os.path.basename(os.path.dirname(p)))
    else:
        A_("4 wlog already present (idempotent): " + os.path.basename(os.path.dirname(p)))

# ---------------- 5 MEMORY
mem = open(MEM, encoding="utf-8").read()
if "**3259" not in mem:
    new_block = ("- **3259 一致性核验**：3258 事后复读全过（链/逐字/封存/端点）；deepseek 回执挂账结清。#411。")
    mem = mem.rstrip("\n") + "\n" + new_block + "\n"
    if len(mem) > 3000:
        o2 = "- **3257 终审**：final_audit_passed——端点+UI 实机 + 磁盘终审（anchors 28/28、gap 锚口径表 6/6、#408 recompute）；N 线预演 apply_preview。收线包可交付。#409。"
        n2 = "- **3257 终审**：final_audit_passed——三端实机+磁盘终审（anchors 28/28、gap 锚 6/6、#408 recompute）；N 线预演。收线包可交付。#409。"
        if o2 in mem:
            mem = mem.replace(o2, n2)
    # semantic updates: N-line apply completed at 3258 (drop from backlog lines)
    if len(mem) > 3000:
        o3 = "drift_3252 锚逐位。MET→3254 落册。"
        n3 = "MET→3254 落册。"
        if o3 in mem:
            mem = mem.replace(o3, n3)
    o4 = "；N 线正式施加；水果类/K4 删除待确认。"
    n4 = "；水果类/K4 删除待确认。"
    if o4 in mem:
        mem = mem.replace(o4, n4)
    o6 = "U 形轨迹；近邻稳定 ×3。"
    n6 = "U 形；近邻稳 ×3。"
    if o6 in mem:
        mem = mem.replace(o6, n6)
    o5 = "- 挂账：APPX-5（重基线后）；N 线施加；5 appendix GPU 项；UI 线 v1.10 冻结；水果类/K4 删除待确认。"
    n5 = "- 挂账：APPX-5（重基线后）；5 appendix GPU 项；UI 线 v1.10 冻结；水果类/K4 删除待确认。"
    if o5 in mem:
        mem = mem.replace(o5, n5)
    assert len(mem) <= 3000, len(mem)
    with open(MEM, "w", encoding="utf-8", newline="") as f:
        f.write(mem)
    A_("5 MEMORY: 3259 block added, %d chars" % len(mem))
else:
    A_("5 MEMORY: 3259 block already present (idempotent)")

print("CLOSEOUT OK")
