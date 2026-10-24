# -*- coding: utf-8 -*-
"""Phase 3265 closeout: five writes (result assert / ledger #417 / MEMO / dual wlog / MEMORY)."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
RES = os.path.join(RBASE, "phase3265", "g5h28_appx7_landing", "result.json")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
WLOGS = [os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
         os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md")]
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")

FREEZE = "84fd1381"
RES_SHA = "18437872"
CHAIN_PREV = "113e6af3"
G13_SHA = "814ce808"
R36_SHA = "78d37211"

R = json.load(open(RES, encoding="utf-8"))
assert R["verdict"] == "appx7_landed" and R["gate_all"], R["verdict"]
assert R["design_sha8"] == FREEZE and R["res_sha8"] == RES_SHA
assert all(R["gates"].values()) and len(R["gates"]) == 6
assert R["revert_check"] == {"gap": True, "registry": True}
assert R["gap_v113"]["file_sha8"] == G13_SHA
assert R["registry_v136"]["file_sha8"] == R36_SHA
assert R["gap_v113"]["appx7"]["id"] == "APPX-7"
print("write1 result assert OK")

# ---------------- write 2: ledger #417 (append-only, chained)
led = json.load(open(LED, encoding="utf-8"))
has417 = any(e.get("phase") == 3265 for e in led["measurements"])
if not has417:
    assert len(led["measurements"]) == 421 and led["ledger_sha256_8"] == CHAIN_PREV, \
        (len(led["measurements"]), led["ledger_sha256_8"])
    entry = {
        "phase": 3265,
        "name": "g5h28_appx7_landing",
        "title": ("APPX-7 landed: gap v1.12 -> v1.13 + registry v1.35 -> v1.36 "
                  "(dominance scale law & u* geometry dissociation)"),
        "verdict": "appx7_landed",
        "evidence_level": "statistical",
        "sources": ["phase3263/g5h26_p_traj/result.json (f0f95c27)",
                    "phase3264/g5h27_traj_second_order/trajectory_second_order_v1"
                    ".json (8d1a7b09)",
                    "phase3265/g5h28_appx7_landing/gap_ledger_v1_13.json (814ce808) "
                    "+ atlas_registry_v1_36.json (78d37211)"],
        "detail": ("GATE 6/6. Appraisal: q1 LAND (statistical, same grade as "
                   "APPX-6 precedent); q2 anchors COMPLETE (p3263 f0f95c27 + "
                   "p3264 f223a9a0 + artifact 8d1a7b09 on disk); q3 "
                   "NO_CONFLICT (new scale dimension; APPX-6 checkpoint-"
                   "trajectory GPU backlog RESOLVED by 3263/3264); q4 CLEAN. "
                   "Landing: gap appendix 6->7 (APPX-7: D scale-dependent, "
                   "1.7b band 15.98-17.28, 0/24 in 30-40x cross-model band; "
                   "Track I bitwise invariant -> dissociation; wscp phase "
                   "transition abrupt + readout-side localized); registry "
                   "upgrade_log 14->15 (anchors unchanged 28); REVERT bitwise "
                   "to aa8cff21 / e8335331; gaps/features/failures/census/"
                   "taxonomy untouched (g5); sealed v1.12/v1.35 byte-identical. "
                   "Zero ledger write at phase time (g6); entry at closeout"),
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
    print("write2 ledger #417 appended, post", new_post)
else:
    new_post = led["ledger_sha256_8"]
    print("write2 ledger #417 already present (idempotent), post", new_post)

# ---------------- write 3: MEMO section
memo = open(MEMO, encoding="utf-8").read()
SEC = "## Phase 3265 (G5-H28):"
if SEC not in memo:
    block = (
        "## Phase 3265 (G5-H28): APPX-7 落册——gap v1.12→v1.13 + registry v1.35→v1.36 "
        "（支配比规模律与 u* 几何解离）[2026-10-21]\n\n"
        "- **verdict `appx7_landed`（GATE 6/6）**。freeze **`84fd1381`** / res "
        "**`18437872`**；ledger #417。落册文件：**gap_ledger_v1_13.json `814ce808`**（"
        "appendix 6→7）+ **atlas_registry_v1_36.json `78d37211`**（upgrade_log 14→15，"
        "anchors 28 不变）。\n"
        "- **评估臂全过**：q1 LAND（statistical，与 APPX-6 落册先例同级——3263 GATE 7/7 "
        "GPU 测量 + 3264 GATE 8/8 独立复核）；q2 锚完备（p3263 f0f95c27 / p3264 f223a9a0 / "
        "产物 8d1a7b09 在盘实测）；q3 无冲突（新增规模维度，不改 APPX-1..6 任何断言；"
        "APPX-6 挂账族 checkpoint 轨迹采样由此 RESOLVED）；q4 链 CLEAN。\n"
        "- **APPX-7 条目**：锚 p3263+p3264（anchor_sha8 f223a9a0）；核心——轨道 I 逐位不变"
        "×24 态 / D 规模依赖（1.7b 带 15.98–17.28，0/24 入 3253 跨模型 30–40× 带）/ wscp 相变"
        "突变 + readout 侧定域（层 0–16 逐位零）；后续 GPU 挂账（可选）：跨规模 D 采样"
        "（qwen2.5-3b/7b）标定规模律曲线，不阻塞收线。\n"
        "- **纪律**：REVERT 逐位 ×2（重建→aa8cff21 / e8335331）；gaps/features/failures/"
        "census/taxonomy 指针级零改动（g5）；封存 v1.12/v1.35 字节不动；registry 链不加 "
        "`updated` 键（3256 SMOKE 教训内化）；SMOKE 拦截 note 字符串括号缺失 ×1。\n"
        "- **3266 已预注册**：(a) 收线整合臂——registry v1.36 客户端可见性核查 + 端点 "
        "/atlas/gaps appendix 实机复读 + N 线 handoff manifest 增补 v1.13 指针（report-only）；"
        "(b) 三项用户确认提案（水果类/K4、lcorr op4、APPX-1 量化口径）继续待裁决。\n"
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
    if "3265 G5-H28" not in t:
        line = ("- 3265 G5-H28 appx7_landed（GATE 6/6）：评估臂 LAND（statistical 同级先例）/ "
                "锚完备 / 无冲突 / 链 CLEAN；gap v1.12→v1.13（appendix +APPX-7：D 规模律与 "
                "u* 几何解离，814ce808）+ registry v1.35→v1.36（upgrade_log +1，78d37211，"
                "anchors 28 不变）；REVERT 逐位 ×2（aa8cff21/e8335331）；封存不动；零账本写入。"
                "freeze %s / res %s / ledger #417 post %s。"
                % (FREEZE, RES_SHA, new_post))
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(t.rstrip("\n") + "\n" + line + "\n")
        print("write4 wlog appended:", os.path.basename(p))
    else:
        print("write4 wlog already present (idempotent):", os.path.basename(p))

# ---------------- write 5: MEMORY (add block + compress, idempotent)
mem = open(MEMORY, encoding="utf-8").read()
if "**3265" not in mem:
    new_block = ("- **3265 APPX-7 落册**：gap v1.13 `814ce808` + registry v1.36 `78d37211`"
                 "（REVERT 逐位 ×2）；D 规模律与 u* 几何解离入册；checkpoint 轨迹挂账 RESOLVED。#417。")
    mem = mem.rstrip("\n") + "\n" + new_block + "\n"

if len(mem) > 3000:
    o = "- **3264 二阶展开**：wscp 相变突变+readout 侧定域（层 0–16 逐位零）；APPX-7 候选（D 规模律，轨道 I 与 D 解离）。#416。"
    n = "- **3264 二阶展开**：wscp 相变突变+readout 侧定域；APPX-7 候选。#416。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3256 封章**：3239–3255 收线链 CLOSED；registry v1.35 e8335331；N 线收线包。#408。"
    n = "- **3256 封章**：收线链 CLOSED；registry v1.35。#408。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3247**：XC_poly_b4 现役（−0.651）。#399。"
    n = "- **3247**：XC_poly_b4 现役。#399。"
    if o in mem:
        mem = mem.replace(o, n)
with open(MEMORY, "w", encoding="utf-8", newline="") as f:
    f.write(mem)
print("write5 MEMORY len", len(mem))
assert len(mem) <= 3000, len(mem)
print("CLOSEOUT OK #417 post", new_post)
