# -*- coding: utf-8 -*-
"""Phase 3274 closeout: five writes + deepseek receipt + impact-preview addendum."""
import hashlib
import json
import os

BASE = r"D:\AI2050\Ai2050-OpenOne"
LED = os.path.join(BASE, "research", "gpt5", "atlas", "atlas_ledger.json")
WLOG = os.path.join(BASE, "research", "gpt5", "docs", "WORKLOG_gpt5.md")
MIRR = os.path.join(BASE, "research", "deepseek", "docs",
                    "WORKLOG_gpt5_mirror.md")
MEMO = os.path.join(BASE, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
DSMEMO = os.path.join(BASE, "research", "deepseek", "docs",
                      "AGI_DEEPSEEK_MEMO.md")
PREVIEW = os.path.join(BASE, "research", "gpt5", "docs",
                       "PENDING_ITEMS_IMPACT_PREVIEW.md")
MEMORIA = os.path.join(BASE, ".workbuddy", "memory", "MEMORY.md")
DAILY = os.path.join(BASE, ".workbuddy", "memory", "2026-10-21.md")
OUT = os.path.join(BASE, "tests", "glm5", "result",
                   "rdc_query_construction_20260913", "phase3274",
                   "g5h37_lcorr_op4_apply")
RES = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
MARK = "## Phase 3274 (G5-H37):"


def append_once(path, text, marker):
    cur = open(path, encoding="utf-8").read()
    if marker in cur:
        print("skip (idempotent):", os.path.basename(path))
        return
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(text)


# 1 ledger receipt (#426, n 431->432)
L = json.load(open(LED, encoding="utf-8"))
ms = L["measurements"]
if ms[-1]["phase"] == 3274 and ms[-1].get("prev_post") == "498a80f6":
    print("skip: ledger receipt already present")
    new_post = L["ledger_sha256_8"]
else:
    assert len(ms) == 431 and L["ledger_sha256_8"] == "498a80f6", (
        len(ms), L["ledger_sha256_8"])
    # verify op4 entry in place (payload verbatim)
    op4 = ms[-1]
    assert op4["phase"] == "A-gate"
    assert op4["name"] == "rdc_a_gate_closure_q01_q02_q08_q09_q12"
    assert op4["verdict"] == ("k1_fired_at_readout__kspecific__"
                              "k2_k3_not_measurable__A_gate_closed")
    rec = {"phase": 3274, "name": "g5h37_lcorr_op4_apply",
           "verdict": "lcorr_op4_applied",
           "evidence_level": "owner_approved_apply",
           "res_sha8": RES["res_sha8"],
           "note": "op4 A-gate governance registration applied "
                   "(post 498a80f6, n 430->431, owner-approved 2026-10-21); "
                   "APPX-4/5 resolved via gap v1.14 + registry v1.37",
           "prev_post": L["ledger_sha256_8"]}
    Ld = dict(L)
    Ld["measurements"] = ms + [rec]
    Ld["ledger_sha256_8"] = L["ledger_sha256_8"]
    new_post = hashlib.sha256(json.dumps(Ld, ensure_ascii=False, indent=1)
                              .encode("utf-8")).hexdigest()[:8]
    Ld["ledger_sha256_8"] = new_post
    with open(LED, "w", encoding="utf-8", newline="") as f:
        json.dump(Ld, f, ensure_ascii=False, indent=1)
    print("write1 ledger: #426 post", new_post,
          "(n=%d)" % len(Ld["measurements"]))

# 2 wlogs
wtxt = ("- **3274（G5-H37）lcorr_op4_applied**（owner 批准施加相，GATE 10/10）："
        "op4（A-gate 治理登记，payload 逐字 5 字段，closure 24c60160 交叉验证）施加至"
        "共享账本——**post '498a80f6'（n 430→431）**，append-only 指针差断言 + revert 逐位 + "
        "备份 .bak_3274；3272 dry-run d165ee30 确认过期未复用（现场重算）。ARM b："
        "gap v1.14 'c6d97133'（APPX-4/5 结清注记）+ registry v1.37 'bfb7d7e5'"
        "（upgrade_log 16）；v1.13/v1.36 字节封存。lcorr v1 beb775bd / v2 ef72c733 / "
        "closure 24c60160 零改动。freeze 91bce3ab / res fe3ca7a4 / ledger #426 post "
        "'%s'（n=432）。5 项待决 → 剩 3 项（水果类/K4、APPX-1 三选一、APPX-2/3 点名）。\n"
        % new_post)
append_once(WLOG, "\n" + MARK + " lcorr op4 施加 + APPX-4/5 结清\n\n" + wtxt, MARK)
append_once(MIRR, "\n" + MARK + " lcorr op4 施加 + APPX-4/5 结清\n\n" + wtxt, MARK)

# 3 G-line MEMO
memo_txt = ("\n" + MARK + " lcorr op4 施加 + APPX-4/5 结清（owner 批准）\n\n"
            "- 裁决：owner 于 2026-10-21 批准 lcorr op4 施加"
            "（PENDING_ITEMS_IMPACT_PREVIEW ②/⑤ 路径）。\n"
            "- Arm a：op4 = sealed patch beb775bd op append_measurement（3260 重推导 "
            "READY）——A-gate 治理登记（phase='A-gate'，name=rdc_a_gate_closure_"
            "q01_q02_q08_q09_q12，verdict=k1_fired_at_readout__kspecific__k2_k3_not_"
            "measurable__A_gate_closed，evidence_level=bit_anchored，note='Q08=甲 "
            "seal 后登记；属治理记录非科学测量'）payload 逐字 + 登记溯源块，施加至共享账本；"
            "post 498a80f6（n 430→431）；append-only 指针差断言（$.measurements/"
            "$.ledger_sha256_8）+ revert 逐位 + 备份 .bak_3274；dry-run d165ee30 "
            "过期未复用（现场重算纪律）。\n"
            "- Arm b：gap v1.13→v1.14（APPX-4 STALE→RESOLVED：N-P3..P7 自 3258 在账；"
            "APPX-5→RESOLVED via op4）+ registry v1.36→v1.37（upgrade_log +1）；"
            "REVERT 逐位回 814ce808/78d37211；gaps/features/failures/anchors 零触碰。\n"
            "- 封存断言：lcorr v1 beb775bd / v2 ef72c733 / closure 24c60160 / "
            "gap v1.13 / registry v1.36 全字节不动。\n"
            "- GATE 10/10；freeze 91bce3ab / res fe3ca7a4 / ledger #426。"
            "剩余待决 3 项：水果类/K4 删除、APPX-1 三选一、APPX-2/3 点名（协议 ce73984d）。\n\n")
append_once(MEMO, memo_txt, MARK)

# 3b deepseek MEMO append-only receipt
DSMARK = "2026-10-21 G 线 3274 回执"
ds_txt = ("\n- **%s（append-only 注记）**：lcorr `ledger_corrections_v1.json`"
          "（beb775bd）op4（A-gate 治理登记，closure 24c60160）经 owner 裁决批准"
          "（2026-10-21），由 G 线 Phase 3274 施加至共享账本 "
          "`research/gpt5/atlas/atlas_ledger.json`（post 498a80f6，n 430→431，"
          "payload 逐字 + 登记溯源块，revert-proof）；op1/op2/op3 以 3260 重推导 v2 稿"
          "（ef72c733）结论归档（作废/不迁移/被实践超越），不再执行。挂账「lcorr 重推导"
          "施加确认」**结清**（APPX-4/5 同步于 gap v1.14 注记 RESOLVED）。本线封存文件"
          "（lcorr/草案/账目）字节不动。\n" % DSMARK)
append_once(DSMEMO, "\n" + ds_txt, DSMARK)

# 3c impact preview addendum (append-only)
PVMARK = "addendum-3274"
pv_txt = ("\n\n<!-- %s -->\n"
          "## 施加回执（Phase 3274，2026-10-21）\n\n"
          "② lcorr op4 已施加：owner 裁决批准（2026-10-21），实际 post = `498a80f6`"
          "（n 430→431，现场重算；本文件记载的 329b1398/d165ee30 均已过期，仅作历史参考）。"
          "⑤ APPX-4/5 已结清：gap v1.14 `c6d97133` + registry v1.37 `bfb7d7e5` 注记 "
          "RESOLVED。剩余待决 3 项：① 水果类/K4、③ APPX-1 三选一、④ APPX-2/3 点名。\n"
          % PVMARK)
append_once(PREVIEW, pv_txt, PVMARK)

# 4 MEMORY: compress 3267-3273 lines, add 3274
m = open(MEMORIA, encoding="utf-8").read()
o1 = ("- **3268**：收线待命快照（report-only）——三账+端点实机核验、APPX 矩阵 7/7、"
      "STANDBY 声明（5 项待批准）。#420。\n"
      "- **3269**：待命复核——三账字节级不变；STANDBY_CONFIRMED。#421。\n"
      "- **3270**：OWNER_STATUS_REPORT.md 权威汇总（docs/）。#422。\n"
      "- **3267**：APPX-2/3 协议 v2（K2/K3 死线冻结；p3155 带=proxy 非 criterion）。#419。")
n1 = ("- **3267–3270**：APPX-2/3 协议 v2（ce73984d）；收线待命快照/复核；"
      "OWNER_STATUS_REPORT.md。#419–#422。")
o2 = ("- **3271/3272**：5 项待决影响预览（APPX-4 STALE/APPX-5→op4）；"
      "op4 DRY-RUN 现役值 d165ee30（329b1398 过期）。#423/#424。\n"
      "- **3273**：最小化待命快照；漂移零/无并发。#425。")
n2 = ("- **3271–3273**：5 项待决影响预览+待命维护+最小快照（STANDBY）。#423–#425。\n"
      "- **3274**：**lcorr op4 施加**（owner 批准 2026-10-21）：A-gate 治理登记入账 "
      "post `498a80f6`（n 431）；gap v1.14 `c6d97133` + registry v1.37 `bfb7d7e5`"
      "（APPX-4/5 结清）；lcorr/closure 字节不动。#426。")
if "**3274" not in m:
    if o1 in m:
        m = m.replace(o1, n1)
    if o2 in m:
        m = m.replace(o2, n2)
    m = m.replace("registry v1.36 78d37211 + gap v1.13 814ce808；v1.35/v1.12 封存",
                  "registry v1.37 bfb7d7e5 + gap v1.14 c6d97133；v1.36/v1.13 封存")
    m = m.replace("（N 线 P3–P7 已补账 #410/3258；lcorr 重推导挂账）",
                  "（N 线 P3–P7 已补账 #410/3258；lcorr op4 已施加 #426/3274）")
    m = m.replace("挂账：APPX-1 量化待确认；水果类/K4 删除待确认；APPX-4/5 见 3271 预览。",
                  "挂账：APPX-1 量化待确认；水果类/K4 删除待确认；APPX-2/3 待点名"
                  "（协议 ce73984d）。APPX-4/5 已结清（3274）。")
    open(MEMORIA, "w", encoding="utf-8", newline="").write(m)
print("MEMORY len(chars):", len(m))
assert len(m) <= 3000, "MEMORY OVER: %d" % len(m)

# 5 daily
append_once(DAILY,
            "\n- 3274 G5-H37 lcorr op4 施加收官（lcorr_op4_applied，GATE 10/10，owner 批准施加相）："
            "op4（A-gate 治理登记，payload 逐字，closure 24c60160 交叉验证）施加至共享账本 "
            "post 498a80f6（n 430→431），append-only/revert-proof/备份 .bak_3274；3272 dry-run "
            "d165ee30 过期未复用（现场重算）。ARM b：gap v1.14 c6d97133（APPX-4/5 结清注记）+ "
            "registry v1.37 bfb7d7e5（upgrade_log 16），v1.13/v1.36 字节封存。lcorr v1 beb775bd/"
            "v2 ef72c733/closure 24c60160 零改动；deepseek MEMO 回执 + 影响预览施加回执注记。"
            "freeze 91bce3ab / res fe3ca7a4 / ledger #426 post %s（n=432）。"
            "剩余待决 3 项：水果类/K4 删除、APPX-1 三选一、APPX-2/3 点名（协议 ce73984d）。\n"
            % new_post,
            "3274 G5-H37")
print("closeout done; #426 post:", new_post)
