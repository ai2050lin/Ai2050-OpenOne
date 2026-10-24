# -*- coding: utf-8 -*-
"""Phase 3279 closeout: five-write (ledger #431 / wlog+mirror / G-line MEMO /
MEMORY compact rewrite / daily log). Idempotent. Placeholder templates
(no bare % formatting — known defect).
"""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
WLOG = os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md")
WLOG_M = os.path.join(ROOT, "tests", "gpt5_temp", "WORKLOG_gpt5_mirror.md")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
DAILY = os.path.join(ROOT, ".workbuddy", "memory", "2026-10-21.md")

RES_DIR = os.path.join(ROOT, "tests", "glm5", "result",
                       "rdc_query_construction_20260913",
                       "phase3279", "g5h42_appx1_landing")

PREV_POST = "c4d25687"
PREV_N = 436


def s8f(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


res = json.load(open(os.path.join(RES_DIR, "result.json"), encoding="utf-8"))
RES_SHA = res["res_sha8"]
assert res["verdict"] == "appx1_landed_blocked_verbatim"
assert res["gate_all"] is True

# ---------------------------------------------------------------- 1. ledger
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
if any(str(m.get("phase")) == "3279" for m in ms):
    new_post = led["ledger_sha256_8"]
    print("ledger: 3279 present, head", new_post, "n", len(ms))
else:
    assert led["ledger_sha256_8"] == PREV_POST and len(ms) == PREV_N
    rec = {
        "n": len(ms) + 1,
        "phase": "3279",
        "name": "g5h42_appx1_landing",
        "line": "G",
        "date": today(),
        "verdict": "appx1_landed_blocked_verbatim",
        "res_sha8": RES_SHA,
        "note": ("APPX-1 landing (blocked verdict recorded verbatim per "
                 "F-ledger principle): gap v1.15 -> v1.16 appendix +APPX-9 "
                 "(shared-dim localization dim0 cos=0.8938 adjudication-"
                 "connective / dim1 cos=0.7149 truth-judgment, neonatal "
                 "layers 37-40; nf4 gB fidelity PASS, gC top1 FAIL 30.791 "
                 "vs 26.641+3.0, eff=2 and cos-diff<=0.035 conserve; fp16 "
                 "first run archived), registry v1.38 -> v1.39 upgrade_log "
                 "18. APPX-1 standing item NOT closed (owner decision "
                 "reserved). GATE 6/6."),
        "prev_post": PREV_POST,
    }
    led["measurements"] = ms + [rec]
    probe = json.loads(json.dumps(led))
    probe["ledger_sha256_8"] = PREV_POST
    new_post = hashlib.sha256(json.dumps(probe, ensure_ascii=False, indent=1)
                              .encode("utf-8")).hexdigest()[:8]
    led["ledger_sha256_8"] = new_post
    with open(LED, "w", encoding="utf-8", newline="") as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    print("ledger #431 appended, post", new_post, "n", len(led["measurements"]))

# ---------------------------------------------------------------- 2. wlog + mirror
wtxt = (
    "\n## Phase 3279 (G5-H42) g5h42_appx1_landing — __DATE__\n"
    "- APPX-1 落册相（landing-pattern 先例 3276→3277）：评审 Arm a（statistical 级，"
    "p3278 res 588d36c9 在盘，verdict appx1_blocked 逐字入账——F 账本原则）→ "
    "Arm b 落册。\n"
    "- gap v1.15 `58094dab`→**v1.16 `12961b7b`**（appendix +APPX-9：R_logic×"
    "K_entity 共享维定位 dim0 0.8938 转折连词族 / dim1 0.7149 真值判断族，末段层 "
    "37–40 新生；nf4 G-B 保真 PASS、G-C top1 FAIL 30.791 vs 26.641+3.0，eff=2 与 "
    "cos diff≤0.035 守恒；APPX-1 挂账不关闭留 owner）+ registry v1.38 `f8772d65`"
    "→**v1.39 `ab14e586`**（upgrade_log 18）。REVERT 逐位 58094dab/f8772d65；"
    "指针差审计全等；账本零写断言。\n"
    "- 诚实日志：SMOKE 拦截 g1 chain_prev 笔误（ee5b19f3 是 #428 post；#430 链前值"
    "为 #429 的 76783ddd）。freeze cccbd727 / res __RES__ / ledger #431 post "
    "'__POST__'（n=437）。剩余待决 1 项：APPX-1 处置（关闭 vs 全精度复测）。\n"
).replace("__DATE__", today()).replace("__RES__", RES_SHA)
for p in (WLOG, WLOG_M):
    cur = open(p, encoding="utf-8").read()
    if "g5h42_appx1_landing" not in cur:
        with open(p, "a", encoding="utf-8", newline="") as f:
            f.write(wtxt.replace("__POST__", new_post))
print("wlog+mirror done")

# ---------------------------------------------------------------- 3. G-line MEMO
mtxt = (
    "\n### Phase 3279 (G5-H42) APPX-1 落册相 — __DATE__\n"
    "- verdict `appx1_landed_blocked_verbatim`（GATE 6/6）。Arm a 评审：statistical 级"
    "（p3278 gA 锚复现 + gB 保真 PASS；gC FAIL 按 F 账本原则如实入账）；锚完备；与 "
    "APPX-1..8 无冲突；链 n=436 post c4d25687 复算一致。\n"
    "- Arm b 落册：gap v1.15→**v1.16 `12961b7b`** appendix +APPX-9（APPX-1 执行结果"
    "逐字入账：Arm a 零 GPU 三锚复现+两共享维定位（dim0 cos=0.8938 R_logic row0×"
    "K_ent rows0+1=转折连词族；dim1 cos=0.7149 rows5/3/7×row2=真值判断族，溯源 "
    "3157 面板极性臂；末段层 37–40 双维新生）；Arm b nf4 bf16：G-B 保真 PASS"
    "（rowspace 0.0001°）、G-C top1 FAIL（30.791 vs 26.641+3.0，R_logic 侧承载），"
    "eff_ge05=2 与共享维 cos diff≤0.035 守恒——量化扰动真实但非定性）；registry "
    "v1.38→**v1.39 `ab14e586`**（upgrade_log 18，无 updated 键）。REVERT 逐位；"
    "v1.15/v1.38 密封字节不动。\n"
    "- **APPX-1 挂账不在本条关闭**（关闭 vs 留待全精度硬件复测，owner 裁决权保留）。"
    "诚实日志：SMOKE 拦截 chain_prev 笔误。res __RES__ / ledger #431 post '__POST__'"
    "（n=437）。\n"
).replace("__DATE__", today()).replace("__RES__", RES_SHA)
cur = open(MEMO, encoding="utf-8").read()
if "g5h42_appx1_landing" not in cur:
    with open(MEMO, "a", encoding="utf-8", newline="") as f:
        f.write(mtxt.replace("__POST__", new_post))
print("memo done")

# ---------------------------------------------------------------- 4. MEMORY compact rewrite
mem = open(MEMORY, encoding="utf-8").read()
if "APPX-1 落册" not in mem:
    mtxt2 = (
        "# RDC/LPF 双线项目纪律（工作区索引）\n"
        "> 权威日志：deepseek 线 `research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`"
        "（P1–40 收官）；gpt5/G 线 `research\\gpt5\\docs\\AGI_GPT5_MEMO.md`（现役）。"
        "细节回查 MEMO/技能/用户级记忆。\n"
        "- 「好的，继续」= AI 主导续研；关键发现重复 3 次。Ledger "
        "`research\\gpt5\\atlas\\atlas_ledger.json`（跨线共享，他线并发只 append）。\n"
        "\n"
        "## 0 现役纪律（G 线五段式）\n"
        "- 每轮：freeze（DESIGN 静态、hash 排除 created）→ SMOKE → 正式 → verdict → "
        "closeout 五写 → 独立复核（verify FAIL 只修 verify）→ present → 预注册下一 Phase。\n"
        "- sha8 锚：锚值实测探针禁凭记忆；内容 sha ≠ 文件 sha；registry/gap vN immutable，"
        "升级 append only，legacy 字节保全断言；链头复算含全部 measurements 且 "
        "field=prev_post；closeout rec 的 phase 用字符串。\n"
        "- E 级 taxonomy：E0/E1/E2/E3；失败必入 F 账本；读得出≠控得住、指纹≠机制。"
        "写盘 newline=''（Windows 换行陷阱）；closeout 模板禁裸 % 格式化。\n"
        "\n"
        "## 1 深线（AGI_DEEPSEEK，P1–40 收官）\n"
        "主轴三段/写入端分布式/REACH 域；P35 A 闸门 seal；Q03 E_read=0.373350；"
        "Q06 C_steer=0。挂账：无（水果类/K4 结清 3275；N 线补账 #410/3258；"
        "lcorr op4 施加 #426/3274）。\n"
        "\n"
        "## 2 G 线现状（registry v1.39 ab14e586 + gap v1.16 12961b7b；"
        "v1.38 f8772d65/v1.15 58094dab 封存）\n"
        "- 26 特征×3 模型全 scope：E2_held_out×20+、E3×3、E1×3；F1–F20"
        "（F20=读出几何≠行为控制）。\n"
        "- GAP-1/2/3 closed；GAP-4 机制链收官；GAP-5 入册 gap v1.7–v1.10；"
        "E 线 CLOSED@3201；D 线收官@3198。APPX 附录 10 项（APPX-9=APPX-1 执行结果"
        "blocked 逐字）。\n"
        "\n"
        "## 3 phase 索引\n"
        "3154–3231 细节回 MEMO；3238–3257 收线链收官（3256 封章 CLOSED 74.36%"
        "+registry v1.35；3257 终审）；3267–3270 协议 v2+STANDBY+OWNER_STATUS_REPORT"
        "（#419–#422）；3238-align：registry 自动发现+/atlas/gaps；UI v1.10 冻结。\n"
        "\n"
        "## 4 现役与挂账\n"
        "- 链头：ledger #431 post '__POST__'（n=437）；gap v1.16 `12961b7b` + "
        "registry v1.39 `ab14e586` 现役。\n"
        "- 挂账仅 1：**APPX-1 处置（关闭 vs 留待全精度硬件复测）留 owner**；其余全结清"
        "（APPX-2/3 #428–#429、APPX-4/5 #426、水果类/K4 #427）。\n"
        "\n"
        "## 5 本机缺陷（Windows，全量见用户级记忆）\n"
        "bash shim 坏/反引号被吃/内联 stdout 丢 ⇒ Python 文件化写 .txt 再 Read；"
        "Edit 幻影 ⇒ 关键写后回读；沙箱 curl localhost 须 --noproxy '*'；V×V 大 Gram "
        "内存爆改 D 空间 GPU eigh（rowspace 验证）。\n"
        "\n"
        "## 6 近期 phase 锚（细节回 daily log/MEMO）\n"
        "- 3263–3266：P-TRAJ GPU 相（**双轨道几何=预训练属性**）；APPX-7 落册 "
        "v1.13/v1.36；manifest v2。#415–#418。\n"
        "- 3271–3273：STANDBY 快照。#423–#425。\n"
        "- 3274–3275：lcorr op4 施加（A-gate 入账 `498a80f6`）+ 水果类/K4 勘误"
        "（deepseek MEMO 现 fb813f57）。#426–#427。\n"
        "- 3276–3277：APPX-2/3 执行+落册：S 族×K_readout separable 跨 6 模型守恒"
        "（48.2–73.6°）；K3 死线 3/3 触发弃单坐标、K2 3/3 不触发；gap v1.15 "
        "`58094dab`/registry v1.38 `f8772d65`。res 8d5bba36。#428–#429。\n"
        "- 3278：APPX-1 执行（nf4，owner 批准）：2 共享维定位（0.894 转折连词族/"
        "0.715 真值族，层 37–40 新生）；nf4 G-C top1 FAIL（30.79 vs 26.64+3）→ "
        "BLOCKED；fp16 首跑存档。res 588d36c9。#430。\n"
        "- 3279：APPX-1 落册（blocked 逐字，F 账本原则；挂账不关闭留 owner）："
        "gap v1.16 `12961b7b`（APPX-9）+ registry v1.39 `ab14e586`。res __RES__。"
        "#431。\n"
    ).replace("__RES__", RES_SHA).replace("__POST__", new_post)
    assert len(mtxt2) <= 3000, len(mtxt2)
    with open(MEMORY, "w", encoding="utf-8", newline="") as f:
        f.write(mtxt2)
    print("MEMORY rewritten, len", len(mtxt2))
else:
    print("MEMORY skip (already has 3279)")

# ---------------------------------------------------------------- 5. daily log
dtxt = (
    "- 3279 G5-H42 APPX-1 落册相收官（appx1_landed_blocked_verbatim，GATE 6/6，"
    "landing 先例 3276→3277）：Arm a 评审（statistical；锚 p3278 res 588d36c9；"
    "F 账本原则入账 blocked verdict；无冲突；链复算一致）→ Arm b 落册——gap "
    "v1.15→v1.16 `12961b7b` appendix +APPX-9（Arm a 三锚复现+2 共享维定位 "
    "0.8938/0.7149 层 37–40 新生；Arm b nf4 G-B PASS/G-C top1 FAIL 30.791 vs "
    "26.641+3.0，eff=2 与 cos diff≤0.035 守恒）+ registry v1.38→v1.39 `ab14e586`"
    "（upgrade_log 18）。REVERT 逐位 58094dab/f8772d65；指针差全等；账本零写。"
    "APPX-1 挂账不关闭留 owner。诚实日志：SMOKE 拦截 chain_prev 笔误（ee5b19f3 为 "
    "#428 post，#430 链前值=76783ddd）。freeze cccbd727 / res 0d702e47 / ledger "
    "#431 post '__POST__'（n=437）。剩余待决 1 项：APPX-1 处置。\n"
).replace("__POST__", new_post)
cur = open(DAILY, encoding="utf-8").read() if os.path.exists(DAILY) else ""
if "3279 G5-H42" not in cur:
    with open(DAILY, "a", encoding="utf-8", newline="") as f:
        f.write(dtxt)
print("daily log done")
print("CLOSEOUT DONE, head post", new_post)
