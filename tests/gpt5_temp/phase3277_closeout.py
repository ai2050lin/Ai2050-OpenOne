# -*- coding: utf-8 -*-
"""Phase 3277 closeout: five-write (ledger #429 / wlog+mirror / G-line MEMO /
MEMORY with row-compress / daily log) + impact-preview addendum. Idempotent.
NOTE: all templates use __PLACEHOLDER__ replacement (bare-% lesson, 3267/3276).
"""
import hashlib
import json
import os
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
WLOG = os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
PV = os.path.join(ROOT, "research", "gpt5", "docs", "PENDING_ITEMS_IMPACT_PREVIEW.md")
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
DAILY = os.path.join(ROOT, ".workbuddy", "memory", "2026-10-21.md")
RES = {"res_sha8": "3ddd53b8", "gap15": "58094dab", "reg38": "f8772d65"}
PREV_POST = "ee5b19f3"
PREV_N = 434
TODAY = datetime.date.today().isoformat()


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


# sanity: anchors on disk (content-sha semantics, not file bytes)
_rj = json.load(open(os.path.join(ROOT, "tests", "glm5", "result",
                                  "rdc_query_construction_20260913",
                                  "phase3277", "g5h40_appx23_landing",
                                  "result.json"), encoding="utf-8"))
assert _rj["res_sha8"] == RES["res_sha8"]
_led0 = json.load(open(LED, encoding="utf-8"))
assert _led0["ledger_sha256_8"] == PREV_POST and len(_led0["measurements"]) == PREV_N

# ---------------- 1. ledger append #429 (idempotent) ----------------
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
already = any(str(r.get("phase")) == "3277" for r in ms)
if already:
    new_post = led["ledger_sha256_8"]
    print("ledger: 3277 present, head", new_post, "n", len(ms))
else:
    assert len(ms) == PREV_N and led["ledger_sha256_8"] == PREV_POST
    rec = {
        "n": len(ms) + 1,
        "phase": "3277",
        "name": "g5h40_appx23_landing",
        "line": "G",
        "date": TODAY,
        "verdict": "appx23_landed",
        "res_sha8": RES["res_sha8"],
        "note": ("APPX-2/3 registry landing (protocol G5/H5): gap "
                 "v1.14->v1.15 (" + RES["gap15"] + ") appendix APPX-8 "
                 "(cross-model conservation of S-family separability 14/14 "
                 "pairs 48.22-73.60 deg; K3 single-coordinate deathline "
                 "fires 3/3, coverage 1.92-5.85 pct << 30; K2 does not fire "
                 "3/3) + registry v1.37->v1.38 (" + RES["reg38"] + ") "
                 "upgrade_log 16->17. REVERT bitwise c6d97133/bfb7d7e5 "
                 "verified; gaps/features/failures untouched."),
        "prev_post": PREV_POST,
    }
    led["measurements"] = ms + [rec]
    probe = json.loads(json.dumps(led, ensure_ascii=False))
    probe["ledger_sha256_8"] = PREV_POST
    new_post = hashlib.sha256(json.dumps(probe, ensure_ascii=False, indent=1)
                              .encode("utf-8")).hexdigest()[:8]
    led["ledger_sha256_8"] = new_post
    with open(LED, "w", encoding="utf-8", newline="") as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    print("ledger #429 appended, post", new_post, "n", len(led["measurements"]))

# ---------------- 2. wlog + mirror (idempotent) ----------------
wtxt = ("## Phase 3277 (G5-H40): APPX-2/3 registry landing\n\n"
        "- **3277（G5-H40）appx23_landed**（协议 G5/H5 落册相，GATE 6/6）："
        "**Arm a 评审**——证据级 statistical（3276 X-gate 5/5 + GATE 9/9）；"
        "锚完备（p3276 res 3ddd53b8 + 协议 ce73984d 在盘）；与 APPX-1..7 "
        "无冲突；链 n=434 post ee5b19f3 复算一致。**Arm b 落册**——gap "
        "v1.14→v1.15（__GAP15__）appendix +APPX-8（S 族可分离性跨模型守恒 "
        "14/14 对 48.22–73.60°，gemma-3 S_syntax 48.22 为下界；K3 单坐标"
        "死线 3/3 触发，η² top-50 覆盖 1.92–5.85% ≪ 30% → 弃单坐标机制"
        "目标；K2 3/3 不触发，cos 降 1.18–1.39% 且一致率 0.509–0.639，"
        "低幅但行为承重的诚实注记）；registry v1.37→v1.38（__REG38__）"
        "upgrade_log 16→17（不带 updated 键，3256 教训）。REVERT 逐位回 "
        "c6d97133/bfb7d7e5 验证；gaps/features/failures 指针差审计全等；"
        "共享账本零写（#429 于本 closeout）。freeze bbb5d5fe / res "
        "__RES__ / ledger #429 post '__POST__'（n=435）。剩余待决 1 项："
        "APPX-1 量化三选一。\n")
if "G5-H40" not in open(WLOG, encoding="utf-8").read():
    wtxt = (wtxt.replace("__GAP15__", RES["gap15"])
                .replace("__REG38__", RES["reg38"])
                .replace("__RES__", RES["res_sha8"])
                .replace("__POST__", new_post))
    with open(WLOG, "a", encoding="utf-8", newline="") as f:
        f.write(wtxt)
    print("wlog appended")

mir = os.path.join(ROOT, "tests", "gpt5_temp", "WORKLOG_gpt5_mirror.md")
if "G5-H40" not in open(mir, encoding="utf-8").read() if os.path.exists(mir) else True:
    with open(mir, "a", encoding="utf-8", newline="") as f:
        f.write(wtxt)
    print("mirror appended")

# ---------------- 3. G-line MEMO append (idempotent) ----------------
mtxt = ("\n### 3277（G5-H40）APPX-2/3 registry landing（appx23_landed）\n"
        "协议 G5/H5 落册相：gap v1.14→v1.15（__GAP15__）appendix +APPX-8"
        "——S 族可分离性跨模型守恒（APPX-2，14/14 对 separable "
        "48.22–73.60°，含 4b 现役锚）+ K3 单坐标死线定谳（APPX-3，3/3 "
        "触发，覆盖 1.92–5.85% ≪ 30%，单坐标机制目标废弃转 ≤10 维联合+"
        "端口类；K2 3/3 不触发）。registry v1.37→v1.38（__REG38__）。"
        "REVERT 逐位 c6d97133/bfb7d7e5 验证。freeze bbb5d5fe / res __RES__"
        " / ledger #429 post __POST__（n=435）。剩余待决 1 项：APPX-1。\n")
if "G5-H40" not in open(MEMO, encoding="utf-8").read():
    mtxt = (mtxt.replace("__GAP15__", RES["gap15"])
                .replace("__REG38__", RES["reg38"])
                .replace("__RES__", RES["res_sha8"])
                .replace("__POST__", new_post))
    with open(MEMO, "a", encoding="utf-8", newline="") as f:
        f.write(mtxt)
    print("MEMO appended")

# ---------------- 4. MEMORY: row-compress + 3277 entry (idempotent) ----------------
mem = open(MEMORY, encoding="utf-8").read()
if "3277" not in mem:
    lines = mem.split("\n")
    out = []
    for l in lines:
        if l.startswith("- 3238–3257 收线链"):
            l = ("- 3238–3257 收线链（#390–#409）：3246 FLIP=INFORMATION_GAP；"
                 "3247–3250 XC 现役链（终 XC_p5_b8 −0.449）；3254 落册 gap "
                 "v1.12（APPX-6）；3255 双轨道几何定谳（收线终点）；3256 封章 "
                 "CLOSED（74.36%）+registry v1.35；3257 终审 final_audit_passed。"
                 "细节回 MEMO。")
        elif l.startswith("- **3271–3273**"):
            l = "- **3271–3273**：影响预览+待命维护+最小快照（STANDBY）。#423–#425。"
        elif l.startswith("- **3275**"):
            l = ("- **3275**：水果类/K4 结清勘误（append-only dated 注记，MEMO 现 "
                 "fb813f57）。#427。")
        elif l.startswith("- **3276**"):
            l = ("- **3276**：APPX-2/3 执行相（协议 ce73984d）：S 族×K_readout "
                 "separable 跨 6 模型守恒；K3 死线 3/3 触发（单坐标覆盖 <6%→废弃"
                 "单坐标目标）、K2 3/3 不触发。res 8d5bba36。#428。")
        out.append(l)
    out.append("- **3277**：APPX-2/3 落册（G5/H5）：gap v1.15 `58094dab`"
               "（APPX-8）+ registry v1.38 `f8772d65`；现役链头 #429 "
               "`" + new_post + "`（n=435）。剩余待决 1 项：APPX-1 三选一。#429。")
    m = "\n".join(out)
    if len(m) > 3000:
        # hard compress: drop phase index detail line 20 (reconstructable from MEMO)
        out = [l for l in out]
        for i, l in enumerate(out):
            if l.startswith("3154 KOUT"):
                out[i] = ("3154/3159/3160/3169/3171/3177–3187/3188–3197/3198–3202/"
                          "3206/3225–3231 phase 索引细节回 MEMO。")
                break
        m = "\n".join(out)
    assert len(m) <= 3000, len(m)
    with open(MEMORY, "w", encoding="utf-8", newline="") as f:
        f.write(m)
    print("MEMORY updated, len", len(m))
else:
    print("MEMORY already has 3277")

# ---------------- 5. daily log append (idempotent) ----------------
dtxt = ("- 3277 G5-H40 APPX-2/3 registry landing 收官（appx23_landed，GATE 6/6，"
        "协议 G5/H5 落册相）：Arm a 评审（statistical 级；锚 p3276 res 3ddd53b8 + "
        "协议 ce73984d；无冲突；链复算一致）→ Arm b 落册——gap v1.14→v1.15 "
        "（58094dab）appendix +APPX-8（S 族可分离性跨模型守恒 14/14 对 "
        "48.22–73.60°；K3 单坐标死线 3/3 触发覆盖 1.92–5.85%≪30%→弃单坐标目标；"
        "K2 3/3 不触发）+ registry v1.37→v1.38（f8772d65）upgrade_log 17。REVERT "
        "逐位 c6d97133/bfb7d7e5；指针差审计全等；账本零写。freeze bbb5d5fe / res "
        "3ddd53b8 / ledger #429 post __POST__（n=435）。剩余待决 1 项：APPX-1 "
        "三选一。\n")
dl = open(DAILY, encoding="utf-8").read() if os.path.exists(DAILY) else ""
if "3277 G5-H40" not in dl:
    with open(DAILY, "a", encoding="utf-8", newline="") as f:
        f.write(dtxt.replace("__POST__", new_post))
    print("daily log appended")

# ---------------- 6. impact preview addendum (idempotent) ----------------
pv = open(PV, encoding="utf-8").read()
if "addendum-3277" not in pv:
    txt = ("\n<!-- addendum-3277 -->\n④ APPX-2/3 已落册：Phase 3276 执行相 "
           "（res 8d5bba36）+ Phase 3277 registry 相（G5/H5，gap v1.15 "
           "58094dab APPX-8 + registry v1.38 f8772d65；ledger #429）。"
           "剩余待决 1 项：③ APPX-1 三选一。\n")
    with open(PV, "a", encoding="utf-8", newline="") as f:
        f.write(txt)
    print("impact preview addendum appended")

print("CLOSEOUT DONE, head post", new_post)
