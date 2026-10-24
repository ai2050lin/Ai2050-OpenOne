# -*- coding: utf-8 -*-
"""Phase 3272 closeout: five writes."""
import hashlib
import json
import os

BASE = r"D:\AI2050\Ai2050-OpenOne"
LED = os.path.join(BASE, "research", "gpt5", "atlas", "atlas_ledger.json")
WLOG = os.path.join(BASE, "research", "gpt5", "docs", "WORKLOG_gpt5.md")
MIRR = os.path.join(BASE, "research", "deepseek", "docs",
                    "WORKLOG_gpt5_mirror.md")
MEMO = os.path.join(BASE, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
MEMORIA = os.path.join(BASE, ".workbuddy", "memory", "MEMORY.md")
DAILY = os.path.join(BASE, ".workbuddy", "memory", "2026-10-21.md")
OUT = os.path.join(BASE, "tests", "glm5", "result",
                   "rdc_query_construction_20260913", "phase3272",
                   "g5h35_standby_maintenance")
RES = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
FREEZE = RES["design_sha8"]
RES_SHA = RES["res_sha8"]
DRY = RES["arms"]["b_dryrun_refresh"]["dry_run_post_current"]
MARK = "## Phase 3272 (G5-H35):"


def append_once(path, text, marker):
    cur = open(path, encoding="utf-8").read()
    if marker in cur:
        print("skip (idempotent):", os.path.basename(path))
        return False
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(text)
    return True


# ---- 1. ledger append (#424) ----
L = json.load(open(LED, encoding="utf-8"))
ms = L["measurements"]
if ms[-1]["phase"] == 3272:
    print("skip (idempotent): ledger")
else:
    assert len(ms) == 428 and L["ledger_sha256_8"] == "3384cfbd"
    rec = {"phase": 3272, "name": "standby_maintenance",
           "verdict": "standby_maintenance_registered",
           "evidence_level": "report_only_maintenance",
           "res_sha8": RES_SHA,
           "note": "report-only: op4 DRY-RUN post refreshed vs current head "
                   "-> " + DRY + " (supersedes stale 329b1398); drift check "
                   "5 objects clean; MD addendum append-only",
           "prev_post": L["ledger_sha256_8"]}
    Ld = dict(L)
    Ld["measurements"] = ms + [rec]
    Ld["ledger_sha256_8"] = L["ledger_sha256_8"]
    post = hashlib.sha256(json.dumps(Ld, ensure_ascii=False, indent=1)
                          .encode("utf-8")).hexdigest()[:8]
    Ld["ledger_sha256_8"] = post
    with open(LED, "w", encoding="utf-8", newline="") as f:
        json.dump(Ld, f, ensure_ascii=False, indent=1)
    print("write1 ledger: #424 post", post,
          "(n=%d)" % len(Ld["measurements"]))

# ---- 2. wlogs ----
wtxt = ("\n- **3272（G5-H35）standby_maintenance**（report-only，GATE 5/5，零 GPU 零账本测量写入）："
        "①漂移核查——5 对象（gap v1.13/registry v1.36/lcorr v2/deepseek MEMO/协议 v2）sha8 全清、无并发 append；"
        "②**op4 DRY-RUN 刷新——3271 的 329b1398 系 n=427 基线已过期（SUPERSEDED_BY_MAINTENANCE），对现役头 #423 重算 = "
        "`%s`（n 428→429）**；③PENDING_ITEMS_IMPACT_PREVIEW.md 追加 dated 勘误注记（append-only，基线字节不动）。"
        "freeze %s / res %s / ledger #424 post '%s'（n=429）。\n"
        % (DRY, FREEZE, RES_SHA,
           json.load(open(LED, encoding="utf-8"))["ledger_sha256_8"]))
append_once(WLOG, "\n" + MARK + " 待命维护\n" + wtxt, MARK)
append_once(MIRR, "\n" + MARK + " 待命维护\n" + wtxt, MARK)

# ---- 3. MEMO ----
sec = (MARK + " 待命维护（report-only）\n\n"
       + "- op4 DRY-RUN post 对现役头刷新：`%s`（supersedes 过期值 329b1398）；"
         "5 对象零漂移；预览报告 MD 追加 dated 勘误注记。\n" % DRY)
append_once(MEMO, "\n" + sec + "\n", MARK)

# ---- 4. MEMORY ----
m = open(MEMORIA, encoding="utf-8").read()
if "**3272" not in m:
    m = m.rstrip("\n") + ("\n- **3272**：op4 DRY-RUN post 刷新（现役头）=`%s`"
                          "（329b1398 过期）；5 对象零漂移。#424。" % DRY)
open(MEMORIA, "w", encoding="utf-8", newline="").write(m)
print("MEMORY len:", len(m))
assert len(m) <= 3000, "MEMORY OVER"

# ---- 5. daily log ----
line = ("\n- 3272 G5-H35 待命维护收官（standby_maintenance_registered，GATE 5/5，report-only 零 GPU 零账本测量写入）："
        "①漂移核查——gap v1.13 814ce808 / registry v1.36 78d37211 / lcorr v2 ef72c733 / deepseek MEMO d3f62d35 / 协议 v2 ce73984d 全清，无并发 append（尾条 3271）；"
        "②**op4 DRY-RUN post 刷新**——3271 注册的 329b1398 基于 n=427 基线，#423 append 后过期（SUPERSEDED_BY_MAINTENANCE）；对现役头 3384cfbd 重算 = **d165ee30**（n 428→429，确定性 x2 复算）；"
        "③PENDING_ITEMS_IMPACT_PREVIEW.md 追加 dated 勘误注记（append-only，基线字节不动）。零封存账目/待决对象写入。"
        "freeze b66ca350 / res 06293923 / ledger #424 post %s（n=429）。维持 STANDBY：5 项裁决权全在所有者；后续任何账本 append 都会使 dry-run post 再过期——施加 op4 时以当时头现场重算为准，预览值仅作数量级参考。\n"
        % json.load(open(LED, encoding="utf-8"))["ledger_sha256_8"])
append_once(DAILY, line, "3272 G5-H35")
print("closeout done")
