# -*- coding: utf-8 -*-
"""Phase 3266 closeout: five writes (result assert / ledger #418 / MEMO / dual wlog / MEMORY)."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
RES = os.path.join(RBASE, "phase3266", "g5h29_closure_integration", "result.json")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
WLOGS = [os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
         os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md")]
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")

FREEZE = "6cc272b4"
RES_SHA = "d6b15de7"
CHAIN_PREV = "ec34b29b"
MV2_SHA = "57531ec3"

R = json.load(open(RES, encoding="utf-8"))
assert R["verdict"] == "closure_integration_registered" and R["gate_all"]
assert R["design_sha8"] == FREEZE and R["res_sha8"] == RES_SHA
assert all(R["gates"].values()) and len(R["gates"]) == 6
assert R["arm_c_manifest_v2"]["file_sha8"] == MV2_SHA
assert R["arm_a_gaps_live"]["version"] == "1.13"
assert R["arm_b_registry_live"]["version"] == "1.36"
print("write1 result assert OK")

# ---------------- write 2: ledger #418 (append-only, chained)
led = json.load(open(LED, encoding="utf-8"))
has418 = any(e.get("phase") == 3266 for e in led["measurements"])
if not has418:
    assert len(led["measurements"]) == 422 and led["ledger_sha256_8"] == CHAIN_PREV, \
        (len(led["measurements"]), led["ledger_sha256_8"])
    entry = {
        "phase": 3266,
        "name": "g5h29_closure_integration",
        "title": ("closure integration: v1.36/v1.13 client visibility live-"
                  "verified + N-line handoff manifest v2 (report-only)"),
        "verdict": "closure_integration_registered",
        "evidence_level": "statistical",
        "sources": ["phase3265/g5h28_appx7_landing/gap_ledger_v1_13.json "
                    "(814ce808) + atlas_registry_v1_36.json (78d37211)",
                    "phase3266/g5h29_closure_integration/"
                    "nline_handoff_manifest_v2.json (57531ec3)"],
        "detail": ("GATE 6/6, report-only. Arm a: /atlas/gaps coroutine live "
                   "re-read -- v1.13 visible, appendix 7, APPX-7 anchor "
                   "p3263+p3264/f223a9a0, chain head [1.13, 1.12, 1.11], "
                   "gaps-5 unchanged (regression). Arm b: /atlas/registry live "
                   "re-read -- v1.36 visible, upgrade_log 15 (tail phase 3265 "
                   "APPX-7 annotation), v1.35 still resolvable with 14 "
                   "(regression). Arm c: N-line handoff manifest v2 "
                   "(nline_handoff_v2, 57531ec3) -- refreshed snapshot "
                   "registry v1.36 / gap v1.13 / ledger #417; gpu_backlog "
                   "updated (P-TRAJ RESOLVED by 3263/3264; APPX-7 follow-up "
                   "cross-scale D sampling OPTIONAL non-blocking); "
                   "u_star_trilogy extended 3->6 (3263/3264/3265); v1 "
                   "manifest (bc212d52) sealed byte-untouched. Zero ledger "
                   "write at phase time (g6); entry at closeout"),
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
    print("write2 ledger #418 appended, post", new_post)
else:
    new_post = led["ledger_sha256_8"]
    print("write2 ledger #418 already present (idempotent), post", new_post)

# ---------------- write 3: MEMO section
memo = open(MEMO, encoding="utf-8").read()
SEC = "## Phase 3266 (G5-H29):"
if SEC not in memo:
    block = (
        "## Phase 3266 (G5-H29): 收线整合——v1.36/v1.13 客户端可见性实机复读 + "
        "N 线 handoff manifest v2 [2026-10-21]\n\n"
        "- **verdict `closure_integration_registered`（GATE 6/6，report-only）**。"
        "freeze **`6cc272b4`** / res **`d6b15de7`**；ledger #418。"
        "manifest v2 **`57531ec3`**（v1 bc212d52 封存不动）。\n"
        "- **Arm a /atlas/gaps 实机**（协程直调，3255 先例口径）：**v1.13 可见**——"
        "appendix 7、APPX-7 锚 p3263+p3264/f223a9a0、链头 [1.13, 1.12, 1.11]、"
        "gaps-5 回归不变。\n"
        "- **Arm b /atlas/registry 实机**：**v1.36 可见**——upgrade_log 15"
        "（尾条 phase 3265 APPX-7 注记）、v1.35 仍可解析（14 条，回归）。\n"
        "- **Arm c manifest v2**（nline_handoff_v2）：快照刷新至 registry v1.36 / "
        "gap v1.13（appendix 7）/ ledger #417（n=422, ec34b29b）；gpu_backlog 更新——"
        "**P-TRAJ RESOLVED（3263/3264）**、APPX-7 后续跨规模 D 采样 OPTIONAL 不阻塞、"
        "APPX-1..3 维持（APPX-1 量化待决）；u_star_trilogy 3→6（+3263/3264/3265）；"
        "boundary/apply_protocol 沿 v1 逐字。\n"
        "- 零账本写入（g6）；封存三元组（manifest v1/gap v1.12/registry v1.35）"
        "字节不动（g2）。\n"
        "- **3267 已预注册**：(a) 待裁决项处置臂——若用户批准水果类/K4 删除则执行 MEMO "
        "挂账勘误（append-only 注记）；lcorr op4 与 APPX-1 量化口径须用户显式批准方可动；"
        "(b) 否则转 GPU 可行项（APPX-2/3 协议细化）或收线待命。\n"
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
    if "3266 G5-H29" not in t:
        line = ("- 3266 G5-H29 closure_integration_registered（GATE 6/6，report-only）："
                "/atlas/gaps 实机 v1.13 可见（appendix 7、APPX-7、gaps-5 回归）+ "
                "/atlas/registry 实机 v1.36 可见（upgrade_log 15、v1.35 回归）；"
                "N 线 handoff manifest v2（57531ec3：快照 v1.36/v1.13/#417，P-TRAJ "
                "RESOLVED，trilogy 3→6，v1 封存不动）。freeze %s / res %s / ledger "
                "#418 post %s。" % (FREEZE, RES_SHA, new_post))
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(t.rstrip("\n") + "\n" + line + "\n")
        print("write4 wlog appended:", os.path.basename(p))
    else:
        print("write4 wlog already present (idempotent):", os.path.basename(p))

# ---------------- write 5: MEMORY (add block + compress, idempotent)
mem = open(MEMORY, encoding="utf-8").read()
if "**3266" not in mem:
    new_block = ("- **3266 收线整合**：v1.36/v1.13 端点实机可见；N 线 manifest v2 "
                 "（P-TRAJ RESOLVED，trilogy 3→6）。#418。")
    mem = mem.rstrip("\n") + "\n" + new_block + "\n"

if len(mem) > 3000:
    o = "- **3258 N 线施加**：草案 f29da2de（N-P3..P7×5）落账本（逐字+溯源）；post `a59da52b`；lcorr STALE（改重推导）。#410。"
    n = "- **3258 N 线施加**：N-P3..P7×5 落账本（post a59da52b）；lcorr STALE。#410。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3261**：417 条证据级重判定（stat 314/GOV 73/bit 14/AMBIG 16）；零写入，施加=破坏性迁移待确认。#413。"
    n = "- **3261**：417 条证据级重判定（314/73/14/16）；零写入。#413。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3265 APPX-7 落册**：gap v1.13 `814ce808` + registry v1.36 `78d37211`（REVERT 逐位 ×2）；D 规模律与 u* 几何解离入册；checkpoint 轨迹挂账 RESOLVED。#417。"
    n = "- **3265 APPX-7 落册**：gap v1.13 `814ce808` + registry v1.36 `78d37211`；D 规模律解离入册。#417。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- 3237 收官（F21+GAP-5，registry v1.32/gap v1.11）；3225–3232 E3 锚（细节回 MEMO）。"
    n = "- 3237 收官（F21+GAP-5）；3225–3232 E3 锚（细节回 MEMO）。"
    if o in mem:
        mem = mem.replace(o, n)
with open(MEMORY, "w", encoding="utf-8", newline="") as f:
    f.write(mem)
print("write5 MEMORY len", len(mem))
assert len(mem) <= 3000, len(mem)
print("CLOSEOUT OK #418 post", new_post)
