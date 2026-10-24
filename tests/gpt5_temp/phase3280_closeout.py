# -*- coding: utf-8 -*-
"""Phase 3280 closeout: five-write (ledger #432 / wlog+mirror / G-line MEMO /
MEMORY compact rewrite / daily log). Idempotent. Placeholder templates.
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
DAILY = os.path.join(ROOT, ".workbuddy", "memory", "2026-10-22.md")
RES_DIR = os.path.join(ROOT, "tests", "glm5", "result",
                       "rdc_query_construction_20260913",
                       "phase3280", "g5h43_consumer_attribution")

PREV_POST = "0ee33c4c"
PREV_N = 437


def today():
    import datetime
    return datetime.date.today().isoformat()


res = json.load(open(os.path.join(RES_DIR, "result.json"), encoding="utf-8"))
RES_SHA = res["res_sha8"]
assert res["verdict"] == "BLOCKED" and res["gate_all"] is False

# ---------------------------------------------------------------- 1. ledger
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
if any(str(m.get("phase")) == "3280" for m in ms):
    new_post = led["ledger_sha256_8"]
    print("ledger: 3280 present, head", new_post, "n", len(ms))
else:
    assert led["ledger_sha256_8"] == PREV_POST and len(ms) == PREV_N
    rec = {
        "n": len(ms) + 1,
        "phase": "3280",
        "name": "g5h43_consumer_attribution",
        "line": "G",
        "date": today(),
        "verdict": "consumer_attribution_blocked_v1",
        "res_sha8": RES_SHA,
        "note": ("P0 consumer attribution v1 (Cammarata-alignment plan; F20 "
                 "as attack surface). gA anchors reproduced (45.845/73.528/"
                 "69.823 tol 0.05deg); gB static OV write analysis sane and "
                 "specific (top heads S_rel~0.042 vs null p95 0.012, "
                 "layer 23-30 cluster, R_logic x K_entity share top heads); "
                 "dynamic top-64 late-block share 0.92/0.91 with shared "
                 "carrier heads (34,15),(35,0),(29,27),(35,22). K-A overlap "
                 "9/11 (need both >=10) and K-A2 spearman 0.137/0.179 "
                 "(need >=0.25) FAIL — dynamic absolute-norm confounded by "
                 "per-head output magnitude; share-normalized correction "
                 "preregistered as 3281. Honest BLOCKED."),
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
    print("ledger #432 appended, post", new_post, "n", len(led["measurements"]))

# ---------------------------------------------------------------- 2. wlog + mirror
wtxt = (
    "\n## Phase 3280 (G5-H43) g5h43_consumer_attribution — __DATE__\n"
    "- P0 消费方归因 v1（Cammarata 对齐计划启动；F20 变主攻面）。Arm a 静态 OV 写入"
    "分析（1152 头 × 2 载荷子空间，零前向）：top 头 S_rel≈0.042 vs null p95 0.012"
    "（~9× 特异性），层 23–30 聚集，R_logic×K_entity top8 高度共享"
    "（(23,9)/(24,19)/(23,10)/(23,11)/(28,6)/(28,7)）。gA 锚逐值复现"
    "（45.845/73.528/69.823）。\n"
    "- Arm b 一次 bf16 前向（3157 面板 128 行，o_proj pre-hook 逐头投影）："
    "动态 top-64 末段层占比 0.92/0.91，共享承载头 (34,15)/(35,0)/(29,27)/(35,22)"
    "——**载轴中继头地图第一版（K-B 报告制，强信号）**。K-A overlap 9/11（需双"
    "≥10）与 K-A2 spearman 0.137/0.179（需 ≥0.25）FAIL——动态绝对范数被头输出"
    "幅度混淆，份额归一化修正（D_norm=||proj||/||c||）预注册为 3281。\n"
    "- 诚实日志：SMOKE 拦截 main 调用缺失/K_readout tied-embeddings/einsum "
    "下标/hook 签名/null 泄漏共 7 处装置笔误。freeze 8171ded4 / res __RES__ / "
    "ledger #432 post '__POST__'（n=438）。下一 Phase 预注册：3281 份额归一化"
    "修正相。\n"
).replace("__DATE__", today()).replace("__RES__", RES_SHA)
for p in (WLOG, WLOG_M):
    cur = open(p, encoding="utf-8").read()
    if "g5h43_consumer_attribution" not in cur:
        with open(p, "a", encoding="utf-8", newline="") as f:
            f.write(wtxt.replace("__POST__", new_post))
print("wlog+mirror done")

# ---------------------------------------------------------------- 3. G-line MEMO
mtxt = (
    "\n### Phase 3280 (G5-H43) 消费方归因 v1（P0）— __DATE__\n"
    "- verdict `consumer_attribution_blocked_v1`（诚实判定；gA/gB 过，K-A/K-A2 "
    "FAIL）。转向声明：按 Cammarata《Already Clean》对齐，从几何描述转向电路"
    "归因——F20（读出几何≠行为控制）从边界变主攻。\n"
    "- **Arm a 静态发现**：qwen3-4b 1152 头中，向 R_logic/K_entity 载荷子空间"
    "写入最强的头（W_OV=O_h·V_h 相对强度 S_rel）聚集于层 23–30，top 值 ~9× 于"
    "随机基 p95；两子空间的 top8 写入头高度共享——**存在专用中继头族**。\n"
    "- **Arm b 动态发现**：3157 面板一次前向逐头投影，top-64 动态写入头 92%/91% "
    "落在末段层 28–35，且两子空间共享承载头 (34,15)/(35,0)/(29,27)/(35,22)"
    "——与双轨道几何「载轴末段成形」一致。K-A/K-A2 因动态绝对范数混淆未达预"
    "注册强度 → BLOCKED 如实入账；份额归一化修正（D_norm=||proj||/||c||）"
    "预注册 3281。\n"
    "- res __RES__ / ledger #432 post '__POST__'（n=438）。\n"
).replace("__DATE__", today()).replace("__RES__", RES_SHA)
cur = open(MEMO, encoding="utf-8").read()
if "Phase 3280 (G5-H43)" not in cur:
    with open(MEMO, "a", encoding="utf-8", newline="") as f:
        f.write(mtxt.replace("__POST__", new_post))
print("memo done")

# ---------------------------------------------------------------- 4. MEMORY compact rewrite
mem = open(MEMORY, encoding="utf-8").read()
if "consumer" not in mem and "3280" not in mem:
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
        "Q06 C_steer=0。挂账：无。\n"
        "\n"
        "## 2 G 线现状（registry v1.39 ab14e586 + gap v1.16 12961b7b）\n"
        "- 26 特征×3 模型全 scope：E2_held_out×20+、E3×3、E1×3；F1–F20；GAP-1/2/3 "
        "closed、GAP-4/5 收官；E 线 CLOSED@3201；D 线收官@3198。APPX 附录 10 项。\n"
        "- **转向（2026-10-22）**：owner 采纳 Cammarata《Already Clean》对齐计划，"
        "P0 消费方归因启动（F20 变主攻）。\n"
        "\n"
        "## 3 phase 索引\n"
        "3154–3231 细节回 MEMO；3238–3257 收线链收官（3256 封章 74.36%）；"
        "3267–3270 协议 v2+STANDBY；3238-align：registry 自动发现；UI v1.10 冻结。\n"
        "\n"
        "## 4 现役与挂账\n"
        "- 链头：ledger #432 post '__POST__'（n=438）。\n"
        "- APPX-1 处置（关闭 vs 全精度复测）留 owner；其余全结清。\n"
        "- **下一 Phase 预注册：3281 消费方归因修正相（份额归一化 "
        "D_norm=||proj||/||c||，同判据）**。\n"
        "\n"
        "## 5 本机缺陷（Windows，全量见用户级记忆）\n"
        "bash shim 坏/反引号被吃/内联 stdout 丢 ⇒ Python 文件化写 .txt 再 Read；"
        "Edit 幻影 ⇒ 关键写后回读；沙箱 curl localhost 须 --noproxy '*'；V×V 大 Gram "
        "内存爆改 D 空间 GPU eigh（rowspace 验证）；qwen3-4b tied embeddings"
        "（lm_head 不在 index）。\n"
        "\n"
        "## 6 近期 phase 锚（细节回 daily log/MEMO）\n"
        "- 3263–3266：P-TRAJ GPU 相（双轨道几何=预训练属性）；APPX-7 落册；manifest v2。\n"
        "- 3274–3277：op4 施加+水果类/K4 结清+APPX-2/3 执行落册（gap v1.15/reg v1.38）。\n"
        "- 3278–3279：APPX-1 执行+落册（2 共享维定位；nf4 G-C FAIL → BLOCKED 逐字；"
        "gap v1.16 12961b7b/reg v1.39 ab14e586）。#430–#431。\n"
        "- **3280**：P0 消费方归因 v1（BLOCKED 诚实入账）：静态 top 写入头层 23–30 "
        "聚集 ~9× 特异性、两子空间共享中继头族；动态 top-64 末段占比 0.92/0.91、"
        "共享承载头 (34,15)/(35,0)/(29,27)/(35,22)；K-A/K-A2 因动态绝对范数混淆 "
        "FAIL→3281 份额归一化修正。res __RES__。#432。\n"
        "- 通道测量纪律：概念=低维子空间非单方向；跨层 cos 不可用（坐标系漂移）；"
        "范数预测可读性不预测一致性；SwiGLU 规范自由度⇒范数非内禀量。\n"
    ).replace("__RES__", RES_SHA).replace("__POST__", new_post)
    assert len(mtxt2) <= 3000, len(mtxt2)
    with open(MEMORY, "w", encoding="utf-8", newline="") as f:
        f.write(mtxt2)
    print("MEMORY rewritten, len", len(mtxt2))
else:
    print("MEMORY skip")

# ---------------------------------------------------------------- 5. daily log
dtxt = (
    "- 3280 G5-H43 消费方归因 v1（P0，Cammarata 对齐计划启动；BLOCKED 诚实入账）："
    "Arm a 静态 OV 写入分析——qwen3-4b 1152 头 × R_logic/K_entity 载荷子空间，"
    "top 写入头 S_rel≈0.042 vs null p95 0.012（~9×），层 23–30 聚集，两子空间 "
    "top8 共享中继头族（(23,9)/(24,19)/(23,10)/(23,11)）；gA 锚逐值复现"
    "（45.845/73.528/69.823）。Arm b 一次 bf16 前向（3157 面板 128 行 pre-hook "
    "逐头投影）——动态 top-64 末段层占比 0.92/0.91，共享承载头 (34,15)/(35,0)/"
    "(29,27)/(35,22)（K-B 报告制强信号，与载轴末段成形一致）。K-A overlap 9/11、"
    "K-A2 spearman 0.137/0.179 未达预注册强度——动态绝对范数被头输出幅度混淆，"
    "份额归一化（D_norm=||proj||/||c||）预注册 3281。诚实日志：SMOKE 拦截 main "
    "缺失/tied-embeddings/einsum/hook 签名/null 泄漏 7 处笔误。freeze 8171ded4 / "
    "res __RES__ / ledger #432 post '__POST__'（n=438）。\n"
).replace("__RES__", RES_SHA).replace("__POST__", new_post)
cur = open(DAILY, encoding="utf-8").read()
if "3280 G5-H43" not in cur:
    with open(DAILY, "a", encoding="utf-8", newline="") as f:
        f.write(dtxt)
print("daily log done")
print("CLOSEOUT DONE, head post", new_post)
