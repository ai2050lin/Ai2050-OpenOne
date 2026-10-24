# -*- coding: utf-8 -*-
"""Phase 3273 closeout: five writes (minimal)."""
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
                   "rdc_query_construction_20260913", "phase3273",
                   "g5h36_minimal_refresh")
RES = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
MARK = "## Phase 3273 (G5-H36):"


def append_once(path, text, marker):
    cur = open(path, encoding="utf-8").read()
    if marker in cur:
        print("skip (idempotent):", os.path.basename(path))
        return
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(text)


# 1 ledger
L = json.load(open(LED, encoding="utf-8"))
ms = L["measurements"]
if ms[-1]["phase"] == 3273:
    print("skip: ledger")
else:
    assert len(ms) == 429 and L["ledger_sha256_8"] == "3f210467"
    rec = {"phase": 3273, "name": "minimal_standby_refresh",
           "verdict": "minimal_standby_refresh_registered",
           "evidence_level": "report_only_minimal_refresh",
           "res_sha8": RES["res_sha8"],
           "note": "repetitive-by-design minimal refresh (declared 3272); "
                   "drift clean, no concurrent append",
           "prev_post": L["ledger_sha256_8"]}
    Ld = dict(L)
    Ld["measurements"] = ms + [rec]
    Ld["ledger_sha256_8"] = L["ledger_sha256_8"]
    post = hashlib.sha256(json.dumps(Ld, ensure_ascii=False, indent=1)
                          .encode("utf-8")).hexdigest()[:8]
    Ld["ledger_sha256_8"] = post
    with open(LED, "w", encoding="utf-8", newline="") as f:
        json.dump(Ld, f, ensure_ascii=False, indent=1)
    print("write1 ledger: #425 post", post,
          "(n=%d)" % len(Ld["measurements"]))

# 2 wlogs
wtxt = ("\n- **3273（G5-H36）minimal_standby_refresh**（report-only 最小化，GATE 3/3，重复性显式标注）："
        "链头 n=429 复算过、5 对象零漂移、无并发 append。freeze 57f49d5b / res 42d43ab7 / "
        "ledger #425 post '%s'（n=430）。维持 STANDBY。\n"
        % json.load(open(LED, encoding="utf-8"))["ledger_sha256_8"])
append_once(WLOG, "\n" + MARK + " 最小化待命快照\n" + wtxt, MARK)
append_once(MIRR, "\n" + MARK + " 最小化待命快照\n" + wtxt, MARK)

# 3 MEMO
append_once(MEMO, "\n" + MARK + " 最小化待命快照（report-only）\n\n"
            + "- 漂移零/无并发；重复性声明（3272 预告）。\n\n", MARK)

# 4 MEMORY: merge 3271/3272 lines, add 3273
m = open(MEMORIA, encoding="utf-8").read()
o = ("- **3271**：5 项待决影响预览；op4 DRY-RUN post 329b1398；"
     "APPX-4 STALE（N-P 在账）/APPX-5→op4。#423。\n"
     "- **3272**：op4 DRY-RUN post 刷新（现役头）=`d165ee30`（329b1398 过期）；"
     "5 对象零漂移。#424。")
n = ("- **3271/3272**：5 项待决影响预览（APPX-4 STALE/APPX-5→op4）；"
     "op4 DRY-RUN 现役值 d165ee30（329b1398 过期）。#423/#424。")
if "**3273" not in m:
    if o in m:
        m = m.replace(o, n)
    m = m.rstrip("\n") + ("\n- **3273**：最小化待命快照；漂移零/无并发。#425。")
open(MEMORIA, "w", encoding="utf-8", newline="").write(m)
print("MEMORY len:", len(m))
assert len(m) <= 3000, "MEMORY OVER"

# 5 daily
append_once(DAILY,
            "\n- 3273 G5-H36 最小化待命快照收官（minimal_standby_refresh_registered，GATE 3/3，report-only）："
            "重复性显式标注（3272 预告）——链头 #424 3f210467（n=429）复算过、5 对象零漂移、无并发 append。"
            "MEMORY 3271/3272 行合并压缩。freeze 57f49d5b / res 42d43ab7 / ledger #425 post %s（n=430）。"
            "维持 STANDBY；5 项裁决权全在所有者。\n"
            % json.load(open(LED, encoding="utf-8"))["ledger_sha256_8"],
            "3273 G5-H36")
print("closeout done")
