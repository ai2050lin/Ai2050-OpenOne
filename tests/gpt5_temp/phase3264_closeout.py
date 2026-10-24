# -*- coding: utf-8 -*-
"""Phase 3264 closeout: five writes (result assert / ledger #416 / MEMO / dual wlog / MEMORY)."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
RES = os.path.join(RBASE, "phase3264", "g5h27_traj_second_order", "result.json")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
WLOGS = [os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
         os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md")]
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")

FREEZE = "1c734572"    # formal design sha; SMOKE freeze recorded in smoke/
RES_SHA = "f223a9a0"
CHAIN_PREV = "b740bf76"

R = json.load(open(RES, encoding="utf-8"))
assert R["verdict"] == "traj_second_order_registered" and R["gate_all"], R["verdict"]
assert R["design_sha8"] == FREEZE and R["res_sha8"] == RES_SHA
assert all(R["gates"].values()) and len(R["gates"]) == 8
assert R["arm_b"]["replication"] is True
print("write1 result assert OK")

# ---------------- write 2: ledger #416 (append-only, chained)
led = json.load(open(LED, encoding="utf-8"))
has416 = any(e.get("phase") == 3264 for e in led["measurements"])
if not has416:
    assert len(led["measurements"]) == 420 and led["ledger_sha256_8"] == CHAIN_PREV, \
        (len(led["measurements"]), led["ledger_sha256_8"])
    entry = {
        "phase": 3264,
        "name": "g5h27_traj_second_order",
        "title": ("trajectory second-order expansion (report-only, zero GPU): "
                  "wscp phase transition is abrupt + readout-side localized; "
                  "APPX-7 candidate (D scale law) drafted"),
        "verdict": "traj_second_order_registered",
        "evidence_level": "statistical",
        "sources": ["phase3263/g5h26_p_traj/result.json (f0f95c27) + "
                    "trajectory.npz (75 arrays)",
                    "phase3264/g5h27_traj_second_order/"
                    "trajectory_second_order_v1.json (8d1a7b09)"],
        "detail": ("GATE 8/8 incl Track-I bitwise invariant re-assert across all "
                   "24 fine-tune states (independent re-check of 3263). Arm A "
                   "per-step curves: 4/6 runs monotone_stable (D 16.57-16.93), "
                   "2/6 wscp runs biphasic - D peaks at s32 (+4%), then s128 "
                   "kout_top1 jumps +36-37% while D falls back. Arm B "
                   "attribution: layer 0-16 diff bitwise ZERO; readout-side "
                   "segment (>=17) carries the entire offset (max 0.1947 L28); "
                   "u*_kout angular drift is ABRUPT not gradual (cos >= 0.99952 "
                   "through s32 -> 0.96959 in single s32->s128 step); 2747/2748 "
                   "twins replicate the same-type phase transition; wscp runs "
                   "are top-2 behavior-gain runs (5.41/5.49). Arm C: APPX-7 "
                   "candidate draft - D is scale-dependent (1.7b band 15.98-"
                   "17.28, 0/24 states in the 3253 cross-model 30-40x band), "
                   "u* geometry scale- and training-invariant -> dissociation. "
                   "Zero GPU, zero model load, zero ledger write (g8)"),
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
    print("write2 ledger #416 appended, post", new_post)
else:
    new_post = led["ledger_sha256_8"]
    print("write2 ledger #416 already present (idempotent), post", new_post)

# ---------------- write 3: MEMO section
memo = open(MEMO, encoding="utf-8").read()
SEC = "## Phase 3264 (G5-H27):"
if SEC not in memo:
    block = (
        "## Phase 3264 (G5-H27): 轨迹二阶展开——wscp 相变为突变 + readout 侧定域；"
        "APPX-7 候选（D 规模律）[2026-10-21]\n\n"
        "- **verdict `traj_second_order_registered`（GATE 8/8，report-only 零 GPU 零模型加载）**。"
        "freeze **`1c734572`** / res **`f223a9a0`**；ledger #416。仅读 3263 产物"
        "（result f0f95c27 + trajectory.npz 75 arrays）。\n"
        "- **Arm A per-step 曲线**：4/6 runs 单调平稳（D 全程 16.57–16.93、kout_top1 微升 "
        "≤0.024、cos ≥ 0.9993）；**2/6 wscp runs 双相**——D 先升至 s32 峰值（17.20/17.28，"
        "+4%），s128 反落（15.98/16.06）且 kout_top1 跳升 +36–37%（0.445→0.606/0.611）——"
        "跳变集中于单步 s32→s128，非渐进。\n"
        "- **Arm B wscp_2748@128 个案归因**：逐层 diff 剖面**层 0–16 逐位零**（embedding+"
        "载轴段与 base 完全相同），偏移全部定域 readout 侧段（≥17，max 0.1947@L28）；"
        "u*_kout 角漂**突变**（s1→s32 cos ≥ 0.99952，s32→s128 单步降至 0.96959）；"
        "轨道 I 不变量独立复断言全过（24 态 mid_emb σ₁ 逐位 == base、layer0 top1 = 1.0、"
        "cos_emb = 1.0）；2747/2748 孪生同型复现；wscp 双 run 恰为行为增益 top-2（5.41/5.49，"
        "n=6 无门）。\n"
        "- **Arm C APPX-7 候选草案**：D 具规模依赖——1.7b 全程带 15.98–17.28（24 态 0 入带 "
        "vs 3253 跨模型 30–40× 带）；u* 几何规模+训练双不变 → **轨道 I 与 D 解离**。"
        "落册需独立 registry 相（3264 不落）。\n"
        "- 产物：trajectory_second_order_v1.json（8d1a7b09）。零写入账本（g8）；3263 封存不动。\n"
        "- **3265 已预注册**：(a) APPX-7 registry 落册评估臂（若用户批准）或转其他挂账；"
        "(b) 两项用户确认提案（水果类/K4 删除、lcorr op4 施加）+ APPX-1 量化口径 继续待裁决"
        "——「继续」不构成批准。\n"
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
    if "3264 G5-H27" not in t:
        line = ("- 3264 G5-H27 traj_second_order_registered（GATE 8/8，report-only 零 GPU）："
                "wscp 相变**突变 + readout 侧定域**（层 0–16 逐位零；u*_kout cos 0.99952→0.96959 "
                "单步；kout_top1 单步 +37%%）；2747/2748 孪生复现；行为增益 top-2 与 readout 重构"
                "同相；APPX-7 候选（D 规模律：1.7b 带 15.98–17.28 全程 0 入 30–40× 跨模型带，"
                "轨道 I 与 D 解离）。freeze %s / res %s / ledger #416 post %s。"
                % (FREEZE, RES_SHA, new_post))
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(t.rstrip("\n") + "\n" + line + "\n")
        print("write4 wlog appended:", os.path.basename(p))
    else:
        print("write4 wlog already present (idempotent):", os.path.basename(p))

# ---------------- write 5: MEMORY (add block + compress, idempotent)
mem = open(MEMORY, encoding="utf-8").read()
if "**3264" not in mem:
    new_block = ("- **3264 二阶展开**：wscp 相变突变+readout 侧定域（层 0–16 逐位零）；APPX-7 候选"
                 "（D 规模律，轨道 I 与 D 解离）。#416。")
    mem = mem.rstrip("\n") + "\n" + new_block + "\n"

if len(mem) > 3000:
    o = "- **3263 P-TRAJ 执行**：G 线首个 GPU 相；25 态位精确恢复；**双轨道几何=预训练属性**（layer0 top1=1.0、cos_emb=1.0 全 25 态；u*_kout cos≥0.97；D 带 16-17 恒定）。#415。"
    n = "- **3263 P-TRAJ**：G 线首个 GPU 相；**双轨道几何=预训练属性**（top1=1.0/cos_emb=1.0 全 25 态）。#415。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3262**：GPU 臂就绪（1.7b 24 delta 在盘；4b 原生 OFF_MACHINE；torch 口径修正）。#414。"
    n = "- **3262**：GPU 臂就绪（24 delta 在盘；4b OFF_MACHINE）。#414。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3261**：417 条 F2 重判定（statistical 314/GOVERNANCE 73/bit_anchored 14/AMBIGUOUS 16）；RTX 5080+torch 实证（前误报）。#413。"
    n = "- **3261**：417 条 F2 重判定（314/73/14/16）。#413。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3259**：N-P 逐字复读、封存集不动、端点 11 断言；deepseek MEMO 回执。#411。"
    n = "- **3259**：施加后复读一致；deepseek 回执。#411。"
    if o in mem:
        mem = mem.replace(o, n)
with open(MEMORY, "w", encoding="utf-8", newline="") as f:
    f.write(mem)
print("write5 MEMORY len", len(mem))
assert len(mem) <= 3000, len(mem)
print("CLOSEOUT OK #416 post", new_post)
