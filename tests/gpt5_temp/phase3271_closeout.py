# -*- coding: utf-8 -*-
"""Phase 3271 closeout: five writes + PENDING_ITEMS_IMPACT_PREVIEW.md."""
import hashlib
import json
import os
import time

BASE = r"D:\AI2050\Ai2050-OpenOne"
LED = os.path.join(BASE, "research", "gpt5", "atlas", "atlas_ledger.json")
WLOG = os.path.join(BASE, "research", "gpt5", "docs", "WORKLOG_gpt5.md")
MIRR = os.path.join(BASE, "research", "deepseek", "docs",
                    "WORKLOG_gpt5_mirror.md")
MEMO = os.path.join(BASE, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
MEMO_DS = os.path.join(BASE, "research", "deepseek", "docs",
                       "AGI_DEEPSEEK_MEMO.md")
MEMORIA = os.path.join(BASE, ".workbuddy", "memory", "MEMORY.md")
DAILY = os.path.join(BASE, ".workbuddy", "memory", "2026-10-21.md")
OUT = os.path.join(BASE, "tests", "glm5", "result",
                   "rdc_query_construction_20260913", "phase3271",
                   "g5h34_impact_preview")
RES = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
FREEZE = RES["design_sha8"]
RES_SHA = RES["res_sha8"]
DRY = RES["arms"]["item2_lcorr_op4"]["dry_run_post"]
MARK = "## Phase 3271 (G5-H34):"


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def append_once(path, text, marker):
    cur = open(path, encoding="utf-8").read()
    if marker in cur:
        print("skip (idempotent):", os.path.basename(path))
        return False
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(text)
    return True


# ---- 1. ledger append (#423) ----
L = json.load(open(LED, encoding="utf-8"))
ms = L["measurements"]
if ms[-1]["phase"] == 3271:
    print("skip (idempotent): ledger")
else:
    rec = {"phase": 3271,
           "name": "pending_items_impact_preview",
           "verdict": "pending_impact_preview_registered",
           "evidence_level": "report_only_impact_preview",
           "res_sha8": RES_SHA,
           "note": "report-only: 5 pending items impact preview; "
                   "zero GPU, zero writes to pending objects or sealed "
                   "accounts; op4 DRY-RUN prospective post " + DRY,
           "prev_post": L["ledger_sha256_8"]}
    Ld = dict(L)
    Ld["measurements"] = ms + [rec]
    Ld["ledger_sha256_8"] = L["ledger_sha256_8"]
    post = hashlib.sha256(json.dumps(Ld, ensure_ascii=False, indent=1)
                          .encode("utf-8")).hexdigest()[:8]
    Ld["ledger_sha256_8"] = post
    raw = open(LED, "rb").read()
    with open(LED, "w", encoding="utf-8", newline="") as f:
        json.dump(Ld, f, ensure_ascii=False, indent=1)
    back = open(LED, "rb").read()
    assert json.loads(back.decode("utf-8")) == Ld
    print("write1 ledger: #423 post", post, "(n=%d)" % len(Ld["measurements"]))
    assert len(ms) == 427

# ---- 6. markdown deliverable (before wlog so it can be referenced) ----
md_path = os.path.join(BASE, "research", "gpt5", "docs",
                       "PENDING_ITEMS_IMPACT_PREVIEW.md")
a = RES["arms"]
md = """# 待决项影响预览（Phase 3271，report-only）

生成：3271 G5-H34 `pending_impact_preview_registered`（freeze `%s` / res `%s`）。
全部数值磁盘实测（含 op4 零写入 DRY-RUN 链计算）；5 项裁决状态均未改变。

## ① 水果类/K4 MEMO 挂账删除
- 对象：`research/deepseek/docs/AGI_DEEPSEEK_MEMO.md`（基线 sha8 `d3f62d35`）。
- 行位实测：「水果类」%s 处、「K4」%s 处；其中挂账清单行（7153/7257/7360/7409/2279/2530/2834/3124 等）与科学发现段（1106/1160/1318/3097 等）须区别对待。
- 机制：**append-only 勘误注记**（MEMO 尾部追加 dated resolution note + 新 dated 小节逐行注记），不重写历史字节；若所有者要求物理删除则属破坏性操作须二次确认。
- 批准后触及：约 %d 行的注记 + 1 段新注记；账本 +1 收据。

## ② lcorr op4 施加（A-gate 治理登记）
- payload（lcorr v2 `ef72c733`，`READY_PENDING_CONFIRMATION`）：phase="A-gate"、name=rdc_a_gate_closure_q01_q02_q08_q09_q12、verdict=k1_fired_at_readout__kspecific__k2_k3_not_measurable__A_gate_closed、evidence_level=bit_anchored、note=Q08=甲 seal 后登记；属治理记录非科学测量。
- **DRY-RUN（零写入）链计算：批准并施加后 post = `%s`（n 427→428）**；机制同 3258（append + recompute + revert-proof）。
- 联动：批准本项即结清 **APPX-5 的最后开放路径**（见 ⑤）。

## ③ APPX-1 量化口径（14b R×K_entity 定位）
- 定义（gap v1.13）：eff_ge05=2 仅 14b；定位共享方向并解释其来源；status=appendix_open。
- 语境事实：Qwen3-14B-bnb-nf4 在盘（nf4_on_disk=True）；RTX 5080 16GB 无法 bf16 装载 14B。
- 裁决空间：批准 nf4 基线执行 / 拒绝量化（转未来硬件）/ 维持 open。批准后 = 新 GPU 执行相（3267 协议模式），封存账目不动。

## ④ APPX-2/3 执行相
- 协议：`appx23_protocol_v2.json`（`ce73984d`，PREREGISTERED_DRAFT_V2）。
- 前置复查（本轮实测）：5 候选模型 tokenizer+config+weights 全在盘。
- GPU 成本声明（3262 readiness note，非排期）：APPX-2 约 15–60 min/模型（S1–S3）；APPX-3 基于现有 3 模型 collects，GPU-light。

## ⑤ APPX-4/5 跨线治理
- **APPX-4 = STALE_STATUS（本轮新定谳）**：N-P3..P7 自 3258 起实测在账（ledger 5 条 N-P 条目），gap v1.13 的 status=open_crossline 系滞后注记；处置 = gap v1.14 appendix 注记 RESOLVED（registry 臂），v1.13 字节不动。
- **APPX-5 = 已收敛至 op4**：ledger_corrections_v1.json（`beb775bd`）四操作 3260 全处置（op1 作废/op2 不迁移/op3 被实践超越/op4 READY）——唯一活口即 ②。

## 裁决矩阵（owner 视角）
| 项 | 动作 | 触及 | 风险 |
|---|---|---|---|
| ① 水果类/K4 | 批准→append-only 注记 | deepseek MEMO ~%d 行注记 | 低（不重写历史） |
| ② lcorr op4 | 批准→ledger append | atlas_ledger +1（post `%s`） | 低（revert-proof） |
| ③ APPX-1 | 三选一 | 新 GPU 相或无 | 中（量化保真度） |
| ④ APPX-2/3 | 点名启动 | GPU 相 ×N + gap/registry 落册 | 中（跨模型口径） |
| ⑤ APPX-4/5 | 随②自动结清+注记 | gap v1.14（registry 臂） | 低 |
""" % (FREEZE, RES_SHA, len(a["item1_fruit_k4"]["fruit_class_lines"]),
       len(a["item1_fruit_k4"]["k4_lines"]),
       len(a["item1_fruit_k4"]["fruit_class_lines"])
       + len(a["item1_fruit_k4"]["k4_lines"]),
       DRY,
       len(a["item1_fruit_k4"]["fruit_class_lines"])
       + len(a["item1_fruit_k4"]["k4_lines"]),
       DRY)
if not os.path.exists(md_path):
    with open(md_path, "w", encoding="utf-8", newline="") as f:
        f.write(md)
    print("write MD:", md_path, s8(md_path))
else:
    print("skip (idempotent): MD")
with open(os.path.join(OUT, "pending_impact_preview_snapshot.md"), "w",
          encoding="utf-8", newline="") as f:
    f.write(md)

# ---- 2. wlogs ----
wline = ("\n- **3271（G5-H34）pending_items_impact_preview**（report-only，GATE 7/7，零 GPU 零账本测量写入）："
         "5 项待裁决逐一影响预览——①水果类/K4：deepseek MEMO 行位实测（水果类 %d 处/K4 %d 处），机制=append-only 勘误注记非字节重写；"
         "②lcorr op4：payload 就绪，DRY-RUN 链计算批准后 post=`%s`（n 427→428）；③APPX-1：nf4 在盘、三选一裁决空间；"
         "④APPX-2/3：协议 `ce73984d` 前置复查 5 模型全在盘；⑤APPX-4 **STALE_STATUS 新定谳**（N-P3..P7 自 3258 在账）+ APPX-5 收敛至 op4。"
         "交付 `PENDING_ITEMS_IMPACT_PREVIEW.md`。freeze %s / res %s / ledger #423 post "
         + "'%s'（n=428）。\n")
SMOKE_TAG = ""
wtxt = wline % (len(a["item1_fruit_k4"]["fruit_class_lines"]),
                len(a["item1_fruit_k4"]["k4_lines"]), DRY, FREEZE, RES_SHA,
                json.load(open(LED, encoding="utf-8"))["ledger_sha256_8"])
append_once(WLOG, "\n" + MARK + " 待决项影响预览\n" + wtxt, MARK)
append_once(MIRR, "\n" + MARK + " 待决项影响预览\n" + wtxt, MARK)

# ---- 3. MEMO section ----
sec = MARK + " 待决项影响预览（report-only）\n\n" \
    + "- 5 项待所有者裁决事项逐一影响预览；op4 DRY-RUN post `%s`（零写入链计算）；" \
      "APPX-4 STALE_STATUS 定谳（N-P3..P7 在账）与 APPX-5→op4 收敛为本轮新口径发现。\n" \
    % DRY
append_once(MEMO, "\n" + sec + "\n", MARK)

# ---- 4. MEMORY (append + compress if over) ----
m = open(MEMORIA, encoding="utf-8").read()
if "**3271" not in m:
    m = m.rstrip("\n") + ("\n- **3271**：5 项待决影响预览；op4 DRY-RUN post 329b1398；"
                          "APPX-4 STALE（N-P 在账）/APPX-5→op4。#423。")
if len(m) > 3000:
    o = "- **3270**：OWNER_STATUS_REPORT.md（b6039d3e）六节权威汇总；全部磁盘实测。#422。"
    n = "- **3270**：OWNER_STATUS_REPORT.md 权威汇总（docs/）。#422。"
    if o in m:
        m = m.replace(o, n)
open(MEMORIA, "w", encoding="utf-8", newline="").write(m)
print("MEMORY len:", len(m))
assert len(m) <= 3000, "MEMORY OVER LIMIT"

# ---- 5. daily log ----
line = ("\n- 3271 G5-H34 待决项影响预览收官（pending_impact_preview_registered，GATE 6/7→7/7，report-only 零 GPU 零账本测量写入）："
        "5 项 owner 待裁决逐一影响预览——①水果类/K4 挂账删除：deepseek MEMO（d3f62d35）行位实测 水果类 26 处/K4 26 处，机制=append-only 勘误注记（不重写历史字节）；"
        "②lcorr op4（ef72c733 READY_PENDING_CONFIRMATION）：**DRY-RUN 零写入链计算批准后 post=329b1398**（n 427→428，机制同 3258）；"
        "③APPX-1：nf4 在盘（True）、三选一裁决空间；④APPX-2/3：协议 ce73984d 前置复查 5 模型全在盘；"
        "⑤**APPX-4 STALE_STATUS 新定谳**（N-P3..P7 自 3258 实测在账，gap v1.13 status 滞后）+ APPX-5 收敛至 op4；两者处置=gap v1.14 appendix 注记（registry 臂）。"
        "交付 research/gpt5/docs/PENDING_ITEMS_IMPACT_PREVIEW.md。勘误：SMOKE 拦截 g1 链式口径（head 复算应含全部 ms 且 field=prev_post，p3267 修法）+ g5 SMOKE N/A 语义。"
        "freeze 12d3194a / res 6cf92c2b / ledger #423 post %s（n=428）。维持 STANDBY：5 项裁决权全在所有者；批准任一项或点名 APPX-2/3 执行相即恢复实质推进。\n"
        % json.load(open(LED, encoding="utf-8"))["ledger_sha256_8"])
append_once(DAILY, line, "3271 G5-H34")
print("closeout done")
