# -*- coding: utf-8 -*-
"""Phase 3275 closeout: five writes."""
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
                   "rdc_query_construction_20260913", "phase3275",
                   "g5h38_fruit_k4_erratum")
RES = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
MARK = "## Phase 3275 (G5-H38):"


def append_once(path, text, marker):
    cur = open(path, encoding="utf-8").read()
    if marker in cur:
        print("skip (idempotent):", os.path.basename(path))
        return
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(text)


# 1 ledger receipt (#427, n 432->433)
L = json.load(open(LED, encoding="utf-8"))
ms = L["measurements"]
if ms[-1]["phase"] == 3275:
    print("skip: ledger receipt already present")
    new_post = L["ledger_sha256_8"]
else:
    assert len(ms) == 432 and L["ledger_sha256_8"] == "98c772a8", (
        len(ms), L["ledger_sha256_8"])
    rec = {"phase": 3275, "name": "g5h38_fruit_k4_erratum",
           "verdict": "fruit_k4_erratum_registered",
           "evidence_level": "owner_approved_erratum",
           "res_sha8": RES["res_sha8"],
           "note": "append-only erratum on deepseek MEMO: standing items "
                   "水果类/K4 closed per 3210/3216 triple anchors + owner "
                   "approval 2026-10-21; 35 lines / 52 occurrences annotated, "
                   "prefix bytes immutable (6d0ed4fb -> fb813f57)",
           "prev_post": L["ledger_sha256_8"]}
    Ld = dict(L)
    Ld["measurements"] = ms + [rec]
    Ld["ledger_sha256_8"] = L["ledger_sha256_8"]
    new_post = hashlib.sha256(json.dumps(Ld, ensure_ascii=False, indent=1)
                              .encode("utf-8")).hexdigest()[:8]
    Ld["ledger_sha256_8"] = new_post
    with open(LED, "w", encoding="utf-8", newline="") as f:
        json.dump(Ld, f, ensure_ascii=False, indent=1)
    print("write1 ledger: #427 post", new_post,
          "(n=%d)" % len(Ld["measurements"]))

# 2 wlogs
wtxt = ("- **3275（G5-H38）fruit_k4_erratum_registered**（owner 批准勘误相，GATE 9/9，"
        "append-only）：deepseek MEMO 尾部追加 dated 勘误注记——standing 名目「水果类」"
        "「K4」结清（三锚：3210 G5-G9 e678dc42/3ae23744/23008571 占位项定性+disposition、"
        "3216 挂账总账 v2 closed×5、owner 批准 2026-10-21）；逐行注记 35 行 52 处"
        "（科学发现 7 行/协议定义 3 行保留、挂账清单行 25 行→已结清）；字节级前缀不动"
        "（6d0ed4fb→fb813f57，+2162 bytes，备份 .bak_3275，原始 35 行逐位不变）。"
        "freeze c2dd8005 / res %s / ledger #427 post '%s'（n=433）。"
        "剩余待决 2 项：APPX-1 三选一、APPX-2/3 点名。\n"
        % (RES["res_sha8"], new_post))
append_once(WLOG, "\n" + MARK + " 水果类/K4 挂账结清勘误\n\n" + wtxt, MARK)
append_once(MIRR, "\n" + MARK + " 水果类/K4 挂账结清勘误\n\n" + wtxt, MARK)

# 3 G-line MEMO
memo_txt = ("\n" + MARK + " 水果类/K4 挂账结清勘误（owner 批准，append-only）\n\n"
            "- 裁决：owner 于 2026-10-21 批准「水果类/K4 MEMO 挂账删除」，机制 = "
            "append-only 勘误注记（不重写历史字节；物理删除未批准）。\n"
            "- 执行：deepseek MEMO 尾部追加 dated 勘误注记（6d0ed4fb→fb813f57，"
            "+2162 bytes，备份 .bak_3275）——standing 名目「水果类」「K4」结清，"
            "三锚 = 3210 G5-G9（design e678dc42 / res 3ae23744 / seal 23008571，"
            "D3 占位项定性 + 两项 disposition）+ 3216 挂账总账 v2 closed×5 + "
            "owner 批准；R1-P6 之 K4 处置要求同批改判关闭。\n"
            "- 逐行注记 35 行 52 处：科学发现/统计事实 7 行与协议判据定义 3 行保留；"
            "挂账清单行 25 行 → 已结清（stale-as-closed）；原始 35 行逐位不变"
            "（g7b）。\n"
            "- GATE 9/9；勘误 gate 教训（进诚实日志）：g7 初版按「+1 行」预算，"
            "实际勘误文自身含多行「水果类」→ 改为按勘误文实测计数 + g7b 原行逐位"
            "断言；SMOKE 未拦截（预算型断言），字节回滚后重跑正式收官。\n"
            "- freeze c2dd8005 / res %s / ledger #427。剩余待决 2 项：APPX-1 "
            "三选一、APPX-2/3 点名（协议 ce73984d）。\n\n" % RES["res_sha8"])
append_once(MEMO, memo_txt, MARK)

# 4 MEMORY: compress 3274 line, add 3275, clear 水果类/K4 pending
m = open(MEMORIA, encoding="utf-8").read()
o1 = "挂账：水果类、K4。（N 线 P3–P7 已补账 #410/3258；lcorr op4 已施加 #426/3274）"
n1 = "挂账：无（水果类/K4 已结清 3275；N 线 P3–P7 已补账 #410/3258；lcorr op4 已施加 #426/3274）"
o2 = ("- **3274**：**lcorr op4 施加**（owner 批准 2026-10-21）：A-gate 治理登记入账 "
      "post `498a80f6`（n 431）；gap v1.14 `c6d97133` + registry v1.37 `bfb7d7e5`"
      "（APPX-4/5 结清）；lcorr/closure 字节不动。#426。")
n2 = ("- **3274**：**lcorr op4 施加**（owner 批准）：A-gate 治理登记入账 `498a80f6`"
      "（n 431）；gap v1.14 `c6d97133`+registry v1.37 `bfb7d7e5`（APPX-4/5 结清）。"
      "#426。\n"
      "- **3275**：**水果类/K4 结清勘误**（owner 批准，append-only）：deepseek MEMO "
      "尾 dated 注记（3210/3216 三锚，35 行 52 处注记），前缀字节不动；MEMO 现 "
      "`fb813f57`。#427。")
o3 = ("挂账：APPX-1 量化待确认；水果类/K4 删除待确认；APPX-2/3 待点名"
      "（协议 ce73984d）。APPX-4/5 已结清（3274）。")
n3 = "挂账：APPX-1 量化待确认；APPX-2/3 待点名（协议 ce73984d）。APPX-4/5、水果类/K4 已结清（3274/3275）。"
if "**3275" not in m:
    if o1 in m:
        m = m.replace(o1, n1)
    if o2 in m:
        m = m.replace(o2, n2)
    if o3 in m:
        m = m.replace(o3, n3)
    open(MEMORIA, "w", encoding="utf-8", newline="").write(m)
print("MEMORY len(chars):", len(m))
assert len(m) <= 3000, "MEMORY OVER: %d" % len(m)

# 5 daily
append_once(DAILY,
            "\n- 3275 G5-H38 水果类/K4 结清勘误收官（fruit_k4_erratum_registered，"
            "GATE 9/9，owner 批准 append-only 勘误相）：deepseek MEMO 尾 dated 注记——"
            "standing 名目「水果类」「K4」结清（三锚：3210 e678dc42/3ae23744/23008571 "
            "占位项定性+disposition、3216 closed×5、owner 批准 2026-10-21）；逐行注记 "
            "35 行 52 处（科学发现 7/协议定义 3 保留、挂账清单行 25→已结清）；字节级前缀"
            "不动（6d0ed4fb→fb813f57，+2162 bytes，备份 .bak_3275，g7b 原行逐位断言）。"
            "诚实日志：g7 初版 +1 行预算笔误（勘误文自身含多行「水果类」）→ 改实测计数；"
            "字节回滚后重跑正式。freeze c2dd8005 / res %s / ledger #427 post %s"
            "（n=433）。剩余待决 2 项：APPX-1 三选一、APPX-2/3 点名（协议 ce73984d）。\n"
            % (RES["res_sha8"], new_post),
            "3275 G5-H38")
print("closeout done; #427 post:", new_post)
