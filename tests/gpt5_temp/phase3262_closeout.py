# -*- coding: utf-8 -*-
"""Phase 3262 closeout: five writes (result assert / ledger #414 / MEMO / dual wlog / MEMORY)."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
RES = os.path.join(RBASE, "phase3262", "g5h25_gpu_arm_readiness", "result.json")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
WLOGS = [os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
         os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md")]
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")

FREEZE = "5c311013"    # formal design sha; SMOKE freeze was 5243fb82
RES_SHA = "acf1b81c"
CHAIN_PREV = "4f5ed748"

R = json.load(open(RES, encoding="utf-8"))
assert R["verdict"] == "gpu_arm_readiness_registered" and R["gate_all"], R["verdict"]
assert R["design_sha8"] == FREEZE and R["res_sha8"] == RES_SHA
assert R["arm_a"]["port_archive_1_7b"].startswith("FULLY_IN_PLACE (24 deltas")
assert R["arm_a"]["native_4b_archive"].startswith("OFF_MACHINE")
assert R["gates"]["g5_torch_cuda"] and R["gates"]["g6_zero_write"]
print("write1 result assert OK")

# ---------------- write 2: ledger #414 (append-only, chained)
led = json.load(open(LED, encoding="utf-8"))
has414 = any(e.get("phase") == 3262 for e in led["measurements"])
if not has414:
    assert len(led["measurements"]) == 418 and led["ledger_sha256_8"] == CHAIN_PREV, \
        (len(led["measurements"]), led["ledger_sha256_8"])
    entry = {
        "phase": 3262,
        "name": "g5h25_gpu_arm_readiness",
        "title": ("GPU arm readiness: checkpoint presence audit (24-delta 1.7b "
                  "archive FULLY_IN_PLACE; 4b native OFF_MACHINE) + preregistered "
                  "P-TRAJ / APPX-1..3 protocol drafts (report-only)"),
        "verdict": "gpu_arm_readiness_registered",
        "evidence_level": "governance_readiness_report",
        "sources": ["tests/glm5/phase2755_rdc_cross_model_signature.py (a2f3b950: "
                    "restore rule W=W0+delta+residual bit-exact, STEPS=128, "
                    "CHECKPOINTS=[1,8,32,128])",
                    "tests/glm5/result/rdc_query_construction_20260913/phase2755/"
                    "qwen25_3b (24 delta npz, 2.427 GB, sha8 x24)",
                    "tests/glm5/result/rdc_query_construction_20260913/phase3262/"
                    "g5h25_gpu_arm_readiness/result.json"],
        "detail": ("GATE 6/6 zero-GPU zero-write. Arm(a) presence: qwen3-1.7b port "
                   "archive (phase2755/qwen25_3b) FULLY_IN_PLACE - 6 runs x 4 steps "
                   "x delta npz (001/008/032/128), 24 sha8 recorded, qwen3-1.7b "
                   "model + index + 2 shards on disk; qwen3-4b native fields/delta "
                   "npz point to C:\\AI2050-RDC-Archive which is ABSENT on this "
                   "machine (JSON manifests only; 2748 local checkpoint96 = "
                   "terminal step-96 only) -> 4b native intermediate-step trajectory "
                   "NOT locally recoverable. torch probe CORRECTED scope: managed "
                   "python 3.11.9 torch 2.11.0+cu128 CUDA True RTX 5080 (3261 note's "
                   "'torch env missing' was probe-scope false report; sealed 3261 "
                   "files untouched). Arm(b) drafts: P-TRAJ -> execution 3263 (25 "
                   "states = base + 6 runs x 4 steps, bit-exact restore, R1 "
                   "formation timing / R2 track-I stability / R3 behavior "
                   "alignment); APPX-1 (14b nf4 quantization acceptability pending), "
                   "APPX-2 (per-model vocab rebuild, feasible), APPX-3 (K2/K3 "
                   "direct, feasible); APPX-4/5 remain cross-line governance"),
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
    print("write2 ledger #414 appended, post", new_post)
else:
    new_post = led["ledger_sha256_8"]
    print("write2 ledger #414 already present (idempotent), post", new_post)

# ---------------- write 3: MEMO section
memo = open(MEMO, encoding="utf-8").read()
SEC = "## Phase 3262 (G5-H25):"
if SEC not in memo:
    block = (
        "## Phase 3262 (G5-H25): GPU arm readiness — checkpoint 在位性核查 + "
        "预注册协议草案（report-only）[2026-10-21]\n\n"
        "- **verdict `gpu_arm_readiness_registered`（GATE 6/6，零 GPU，零写入账本）**。\n"
        "- **Arm (a) 在位性核查**：qwen3-1.7b 移植归档（phase2755/qwen25_3b）"
        "**FULLY_IN_PLACE——6 runs × 4 步（001/008/032/128）= 24 个 delta npz 全 sha8 "
        "实测（2.427 GB）+ qwen3-1.7b 模型/索引/双 shard 在盘**；qwen3-4b 原生 "
        "fields/delta npz 实体指向 `C:\\AI2050-RDC-Archive`——**本机不存在（OFF_MACHINE）**，"
        "本地仅 JSON manifest（field_sha256+shapes）；2748 本地 checkpoint96 仅终态 "
        "step-96 ⇒ 4b 原生中间步轨迹本机不可恢复。torch 探测口径修正：managed python "
        "3.11.9 torch 2.11.0+cu128 CUDA True（RTX 5080）——3261 note「torch env missing」"
        "系探测口径误报，封存 3261 文件不动。恢复规则逐字锚定 phase2755 源（sha8 `a2f3b950`）："
        "**W_t = W_base + delta_t + residual_t 位精确（np.array_equal）**，STEPS=128。\n"
        "- **Arm (b) 协议草案**（PREREGISTERED_DRAFT，零执行）：**P-TRAJ → 执行相 3263**"
        "（25 态 = base + 6×4，位精确恢复+复基底断言，tokenizer per-row gate，采样 "
        "last-token 全层 hidden；预注册读数 **R1 形成时点**（载轴支配 ovr 带于何训练步出现）、"
        "**R2 轨道 I 稳定性**（embedding 秩1 对齐随步稳定假说）、**R3 行为对齐**（2755 "
        "behavior_scores B1 6/6）；成本 RTX 5080 上远低于 1h）；APPX-1（14b 需 nf4 量化，"
        "可接受性待决）、APPX-2（逐模型词表重建，可行）、APPX-3（K2/K3 直接测量，可行）；"
        "APPX-4/5 = 跨线治理非 GPU，维持待确认。\n"
        "- 产物：`checkpoint_presence_audit_v1.json`（`9bbc3664`）/ "
        "`gpu_trajectory_protocol_draft_v1.json`（`b569ce24`）。\n"
        "- ledger #414 / freeze **`%s`**（SMOKE `5243fb82`）/ res **`%s`**。\n"
        "- **3263 已预注册**：(a) **P-TRAJ 正式执行**（GPU 轨迹臂首发：25 态采样 + "
        "R1/R2/R3 读数 + bit-exact restore 断言，五段式照常）；(b) APPX-1 量化口径裁决"
        "与 APPX-2/3 执行排序待定；(c) 两项用户确认提案（水果类/K4 删除、lcorr op4 施加）"
        "继续待裁决——「继续」不构成批准。\n" % (FREEZE, RES_SHA)
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
    if "3262 G5-H25" not in t:
        line = ("- 3262 G5-H25 gpu_arm_readiness_registered（GATE 6/6，零 GPU 零写入）："
                "1.7b 移植归档 FULLY_IN_PLACE（24 delta npz 2.427GB 全 sha8）+ qwen3-1.7b "
                "在盘；4b 原生 npz 归档 OFF_MACHINE（C:\\AI2050-RDC-Archive 不存在，"
                "manifest-only，ckpt96 仅终态）；torch 口径修正（managed 3.11.9 cu128 "
                "CUDA True）。协议草案：P-TRAJ→3263 执行（25 态位精确恢复，R1/R2/R3）；"
                "APPX-1 量化待决 / APPX-2/3 可行 / APPX-4/5 治理维持。freeze %s / res %s / "
                "ledger #414 post %s。"
                % (FREEZE, RES_SHA, new_post))
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(t.rstrip("\n") + "\n" + line + "\n")
        print("write4 wlog appended:", os.path.basename(p))
    else:
        print("write4 wlog already present (idempotent):", os.path.basename(p))

# ---------------- write 5: MEMORY (add block + compress, idempotent)
mem = open(MEMORY, encoding="utf-8").read()
if "**3262" not in mem:
    new_block = ("- **3262 GPU 臂就绪**：1.7b 归档 24 delta 全在盘（4b 原生 OFF_MACHINE）；"
                 "torch 口径修正；P-TRAJ 草案→3263 执行（R1/R2/R3）。#414。")
    mem = mem.rstrip("\n") + "\n" + new_block + "\n"

if len(mem) > 3000:
    o = "- **3261 证据级重判定**：417 条对 F2 三级规则重推——stat 314/GOV 73/bit 14/AMBIGUOUS 16；零写入，施加=破坏性迁移待确认；GPU 挂账环境可行（torch 在）。#413。"
    n = "- **3261**：417 条证据级重判定（stat 314/GOV 73/bit 14/AMBIG 16）；零写入，施加=破坏性迁移待确认。#413。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3260**：lcorr 重推导 v2 草案 + backlog v3 + 用户提案（REPORT_ONLY，零写入三文件 sha 断言）；lcorr 4 op 全过期。#412。"
    n = "- **3260**：lcorr v2 草案 + backlog v3 + 用户提案（REPORT_ONLY）；lcorr 4 op 全过期。#412。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3259**：施加后复读一致（registry/gap/ledger 端点）；deepseek MEMO 回执落款。#411。"
    n = "- **3259**：施加后复读一致；deepseek MEMO 回执。#411。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3258 N 线施加**：N-P3..P7 施加（n 414 post a59da52b）；lcorr STALE_AS_SEALED。#410。"
    n = "- **3258**：N-P3..P7 施加（n 414）；lcorr STALE_AS_SEALED。#410。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3257 终审**：final_audit_passed——三端实机+磁盘终审；N 线预演；收线包可交付。#409。"
    n = "- **3257**：终审 final_audit_passed；收线包可交付。#409。"
    if o in mem:
        mem = mem.replace(o, n)
with open(MEMORY, "w", encoding="utf-8", newline="") as f:
    f.write(mem)
print("write5 MEMORY len", len(mem))
assert len(mem) <= 3000, len(mem)
print("CLOSEOUT OK #414 post", new_post)
