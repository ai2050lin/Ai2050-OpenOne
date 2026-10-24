# -*- coding: utf-8 -*-
"""Phase 3263 closeout: five writes (result assert / ledger #415 / MEMO / dual wlog / MEMORY)."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
RES = os.path.join(RBASE, "phase3263", "g5h26_p_traj", "result.json")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
WLOGS = [os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
         os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md")]
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")

FREEZE = "bdcd77ef"    # formal design sha; SMOKE freeze was 27fd3d74
RES_SHA = "f0f95c27"
CHAIN_PREV = "ec0dc42c"

R = json.load(open(RES, encoding="utf-8"))
assert R["verdict"] == "p_traj_executed" and R["gate_all"], R["verdict"]
assert R["design_sha8"] == FREEZE and R["res_sha8"] == RES_SHA
assert all(R["gates"].values()) and len(R["gates"]) == 7
assert R["readouts"]["R1"]["status"] == "MIXED"
assert R["readouts"]["R2"]["status"] == "TRACK1_STABLE"
assert R["readouts"]["R2"]["layer0_top1_min"] == 1.0
assert abs(R["readouts"]["R3"]["spearman_D128_vs_improvement"] + 0.4285714285714286) < 1e-12
assert len(R["states"]) == 25
print("write1 result assert OK")

# ---------------- write 2: ledger #415 (append-only, chained)
led = json.load(open(LED, encoding="utf-8"))
has415 = any(e.get("phase") == 3263 for e in led["measurements"])
if not has415:
    assert len(led["measurements"]) == 419 and led["ledger_sha256_8"] == CHAIN_PREV, \
        (len(led["measurements"]), led["ledger_sha256_8"])
    entry = {
        "phase": 3263,
        "name": "g5h26_p_traj",
        "title": ("P-TRAJ checkpoint training-step trajectory executed (first G-line "
                  "GPU measurement): u* dual-track geometry is a PRETRAINING "
                  "property — both tracks invariant across 25 states"),
        "verdict": "p_traj_executed",
        "evidence_level": "statistical",
        "sources": ["phase3262/g5h25_gpu_arm_readiness/gpu_trajectory_protocol_draft_v1"
                    ".json (b569ce24, prereg)",
                    "phase2755/qwen25_3b delta npz x24 (sha8 re-verified) + "
                    "qwen3-1.7b model",
                    "phase3263/g5h26_p_traj/result.json (25 states, collect caches "
                    "193MB x25, restore bit-exact)"],
        "detail": ("GATE 7/7 incl bit-exact fp32 restore + base re-collect bitwise "
                   "equal + 24 delta sha8 re-verified. Readouts: R1 MIXED - "
                   "dominance D=ovr(mid,out)/ovr(mid,emb) stays 16.1-17.2 from base "
                   "through step128 in ALL 6 runs (0/6 reach the 3253 band 30-40x; "
                   "band is model-scale-dependent, not training-formed); R2 "
                   "TRACK1_STABLE - layer0 top1=1.0000 and cos(u*_emb, base)=1.0000 "
                   "at ALL 25 states (embedding rank-1 track fully invariant under "
                   "fine-tuning); R3 n=6 characterization spearman(D128, "
                   "improvement)=-0.43 / spearman(sigma1_128, improvement)=+0.49. "
                   "Headline: u* dual-track geometry does NOT form during training "
                   "- both tracks are pretraining properties; largest excursion is "
                   "wscp_2748@128 (top1[kout] 0.61, cos_out 0.9696) in the run with "
                   "the largest behavior gain (5.49). k_mid=16 constant everywhere"),
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
    print("write2 ledger #415 appended, post", new_post)
else:
    new_post = led["ledger_sha256_8"]
    print("write2 ledger #415 already present (idempotent), post", new_post)

# ---------------- write 3: MEMO section
memo = open(MEMO, encoding="utf-8").read()
SEC = "## Phase 3263 (G5-H26):"
if SEC not in memo:
    block = (
        "## Phase 3263 (G5-H26): P-TRAJ 训练步轨迹执行——u* 双轨道几何为预训练属性 "
        "（G 线首次 GPU 测量）[2026-10-21]\n\n"
        "- **verdict `p_traj_executed`（GATE 7/7，25/25 态采样，G 线首个 GPU 相）**。"
        "freeze **`bdcd77ef`**（SMOKE `27fd3d74`）/ res **`f0f95c27`**；ledger #415。\n"
        "- **装置**：qwen3-1.7b（models/hf 在盘），block layers[16].mlp（37,748,736 "
        "参数）；按 3262 封存协议 b569ce24：**25 态 = base + 6 runs × 4 步（1/8/32/128）**，"
        "位精确恢复 W_t = W_base + delta_t + residual_t（fp32 参数空间、2755 FP32-block "
        "bridge 先例——现行 torch 下混精度须桥接前向，SMOKE 修正×1）；逐态 batch=1 bf16 "
        "收集（3169 逐字惯例，NH=29 层 last-token，fp16 存储，缓存 193MB ×25 可续跑）；"
        "u* 链逐字沿用 3190/3252/3253（ridge 1e-3、3 seeds、876 OOV 残差行、SVD top-8）；"
        "tokenizer 门全过（D2 类词首 token 唯一 + 尾 token『。』2190/2190）；24 delta sha8 "
        "全复验；**位精确复原 + base 复采逐位相等（g6）**。\n"
        "- **R1 = MIXED（0/6 runs 达带）**：支配比 D = ovr(mid,out)/ovr(mid,emb) 从 base "
        "16.60 全程 16.1–17.2 无一达到 3253 的 30–40× 带（该带系 4b/14b/glm4 跨模型观测，"
        "随模型规模而异，非训练成形——6 runs 任意步均未入带，预注册处置=characterization "
        "either way）。\n"
        "- **R2 = TRACK1_STABLE（headline）**：**全部 25 态 layer0 top1 = 1.0000 且 "
        "cos(u*_emb, base) = 1.0000**——embedding 秩 1 轨道（轨道 I）对微调完全不变；"
        "u*_kout 方向 cos ≥ 0.9696（25 态 min），k_mid=16 恒定。**双轨道几何（3255 综论）"
        "在训练步维度既不形成也不改变——均为预训练属性**；唯一显著偏移 wscp_2748@128 "
        "（top1[kout] 0.6109 / cos_out 0.9696），恰为行为增益最大 run（improvement 5.49）。\n"
        "- **R3（n=6 特征化）**：spearman(D128, improvement) = −0.43；"
        "spearman(σ1_mid_out@128, improvement) = +0.49——读数与行为增益方向一致但 n=6 "
        "不设门。\n"
        "- 产物：result.json + trajectory.npz（25×top1 曲线 + u*_kout/emb 向量）+ "
        "collect 缓存 ×25。零写入账本（g7）。\n"
        "- **3264 已预注册**：(a) 轨迹二阶展开臂（report-only，零 GPU）——per-step top1/"
        "D 曲线的定量刻画 + wscp_2748@128 偏移个案归因 + R1 带的模型规模依赖性注记"
        "（APPX-7 候选评估）；(b) 两项用户确认提案（水果类/K4 删除、lcorr op4 施加）继续"
        "待裁决——「继续」不构成批准。\n"
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
    if "3263 G5-H26" not in t:
        line = ("- 3263 G5-H26 p_traj_executed（GATE 7/7，G 线首次 GPU 测量，42min RTX 5080）："
                "25 态 = base + 6 runs × 4 训练步，位精确恢复+复采逐位相等；R2 TRACK1_STABLE"
                "——layer0 top1=1.0000、cos_emb=1.0000 全 25 态，u*_kout cos≥0.9696，"
                "**双轨道几何均为预训练属性，训练既不形成也不改变**；R1 MIXED（D 带 16-17 "
                "不达 3253 的 30-40× 跨模型带，模型规模依赖）；R3 n=6 spearman −0.43/+0.49。"
                "freeze %s / res %s / ledger #415 post %s。"
                % (FREEZE, RES_SHA, new_post))
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(t.rstrip("\n") + "\n" + line + "\n")
        print("write4 wlog appended:", os.path.basename(p))
    else:
        print("write4 wlog already present (idempotent):", os.path.basename(p))

# ---------------- write 5: MEMORY (add block + compress, idempotent)
mem = open(MEMORY, encoding="utf-8").read()
if "**3263" not in mem:
    new_block = ("- **3263 P-TRAJ 执行**：G 线首个 GPU 相；25 态位精确恢复；**双轨道几何=预训练属性**"
                 "（layer0 top1=1.0、cos_emb=1.0 全 25 态；u*_kout cos≥0.97；D 带 16-17 恒定）。#415。")
    mem = mem.rstrip("\n") + "\n" + new_block + "\n"

if len(mem) > 3000:
    o = "- **3262 GPU 臂就绪**：1.7b 归档 24 delta 全在盘（4b 原生 OFF_MACHINE）；torch 口径修正；P-TRAJ 草案→3263 执行（R1/R2/R3）。#414。"
    n = "- **3262**：GPU 臂就绪（1.7b 24 delta 在盘；4b 原生 OFF_MACHINE；torch 口径修正）。#414。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3258**：N-P3..P7 施加（n 414）；lcorr STALE_AS_SEALED。#410。"
    n = "- **3258**：N-P3..P7 施加；lcorr STALE。#410。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3250**：deg 5–6 停点；**XC_p5_b8 现役**（−0.449，r2 0.241/KL 0.987）；混合核≡poly3；fidelity v2+UI 19/19。#402。"
    n = "- **3250**：**XC_p5_b8 现役**（−0.449）；fidelity v2+UI 19/19。#402。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3248**：degree 未饱和；K_p3_T3 最优。#400。"
    n = "- **3248**：degree 未饱和。#400。"
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = "- **3247**：XC_poly_b4 现役（−0.651，闭合 +0.142）。#399。"
    n = "- **3247**：XC_poly_b4 现役（−0.651）。#399。"
    if o in mem:
        mem = mem.replace(o, n)
with open(MEMORY, "w", encoding="utf-8", newline="") as f:
    f.write(mem)
print("write5 MEMORY len", len(mem))
assert len(mem) <= 3000, len(mem)
print("CLOSEOUT OK #415 post", new_post)
