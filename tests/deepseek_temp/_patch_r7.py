# -*- coding: utf-8 -*-
"""R7：写入 MEMORY.md（全量重建）/ wlog 追加 / 技能教训 41。
每处：写前 assert count==1（或全长重建）→ 写 → 回读比对。
所有文件均为 LF（无 CRLF）。
"""
import os, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
MEM  = os.path.join(ROOT, r".workbuddy\memory\MEMORY.md")
WLOG = os.path.join(ROOT, r".workbuddy\memory\2026-10-03.md")
SK   = r"C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md"
REP  = os.path.join(ROOT, r"tests\deepseek\result\_patch_r7_report.txt")
log = []
def L(s): log.append(s); print(s)
def sha8(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

# ---------------- 1. MEMORY.md 全量重建 ----------------
NEW_MEMORY = """# RDC/LPF 项目纪律（工作区长期记忆）
> 权威记录 `research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`（**唯一研究日志**，Phase 1–34）。本文件仅跨轮索引，细节回查 MEMO / 技能。

## 0 约定（v4，2026-10-03，**仅本对话遵守**）
- 日志唯一落点 = `AGI_DEEPSEEK_MEMO.md`（append-only、BOM+CRLF、`bare_lf 0`、标题 `## Phase {N}: 短标题 [hh:mm]`）；**不再新建其他 `.md`**。`AGI_GPT5_MEMO.md` 属其他 AI 线，**不读不改不追加**。
- 三桶：测试脚本 `tests\\deepseek\\`；临时脚本 `tests\\deepseek_temp\\`；测试结果 `tests\\deepseek\\result\\`。历史 `Phase1..21\\` 不动。
- MEMO：P1–21 N 线；**P22 = A 闸门 R3–R5b**；**P23–30 = 8 件并入文档**；**P31 = 路由归一 R6**；**P32–34 = 归属整理并入（N1/E1/MEMO 审计）**。现 `b257f54f`（676,086 B / 6902 行）。
- Ledger `atlas_ledger.json` n=**304** `bbda63df`（**N 线 P3–P7 待补**）。「好的，继续」= AI 主导续研；**关键发现重复 3 次**。

## 1 装置铁律（63 坑全文见 skill `rdc-main-axis-probe`）
(a) 份额只用**精确可加向量预算**；(b) SMOKE 必做**必看数字**；(o) 多 Edit 静默丢 ⇒ Python 补丁 + `assert count==1` + 回读；(ad) 实现与 seal **逐字一致**；(ae) 数字**一律 result 现场渲染**；(af) **冻结锚重验**。
其余：patch 剔末层；写入窗=相邻最大增量；`jump/max|effect|≥0.5`；剔实例 token 同剔类别 token；双剂量 `x*·r_ℓ`；固定基报 `overlap`；paired `margin` 负=候选优。

## 2 N 线主线（P4→P21）
- P4–P7 主轴三段（嵌入=词典/层=开关/**权重绑定定 is-a 落点**）；单头 share_max **3.0%**；读位槽 **G−1=5 维**；跨族近正交 ⇒ 无通用类别算子。
- P8–P11 写入端**分布式**（向量预算 MLP 0.472）；L6 内**无阈值增益** ⇒ **栈=软门**；深端塌陷主因**方向失配**。
- P12–P16 `rho(xhalf,depth)=−0.783`；**P16 否证 ⇒ P12/13/14 深度表述撤回**；域 `REACH={ℓ:ρ≥0.10}`。
- P17–P21 `w_ℓ`+`com_V` ≈26 ≫ median(REACH)；`spearman(w,|b|)` 与 P17 **反号 ⇒ P17 P6 对象错配**；跨精度 7/9、A0_bf16 **逐位复现 P8 锚**。

## 3 挂账与限界
- **核心三条限界**：① 否定臂基线不成立；② 「未见类别」仍崩（水果 0.04/0.05）；③ **激活级干预 ≠ 权重级证明**（其余见 MEMO / 技能 26–41）。
- R1/R2：10 条成立；P1 纠错 1（K_d）+降级 4+挂账 5。G 线索引：3151 k3_only / 3152 k1_not_triggered / 3153 coverage_partial；判决符号 rev-3151b（负=优）。

## 4 本机缺陷（Windows）
- bash shim `ls/rm/tail/dirname/cd` 坏、内联 `python -c` stdout 常丢、**反引号被吃** ⇒ Python 文件化 + 写 `.txt` 再 Read。
- **`Edit` 与脚本日志均可能幻影** ⇒ 中文大段走 Python 补丁（**禁凭记忆写匹配串，先 dump 真实磁盘**）；**落盘证据只能靠独立进程 re-hash**。`Read` 大文件视图陈旧 ⇒ 改没改用 Grep/Python。
- **`AGI_DEEPSEEK_MEMO.md` 未被 git 跟踪** ⇒ 改动前先落快照（`tests\\deepseek_temp\\_archive_r7\\*.pre_r7.bin`）才能事后定位 drift。

## 5 下一步 / 死线
- **⚠ 待 seal**（`tests\\deepseek\\result\\seal_request_v1.json`）：Q08（K1 改判：读法甲/乙）、Q01 更正表 **C1–C6**、I1/I9 ⇒ 之后才进 B 闸门 Q03（需 GPU）。
- 零 GPU：**Q01/Q02/Q09/Q12 已完成**（复核 44/0、43/0、76/0）。挂账：下一实验 Phase（P8 `share_v` × P16/P17 `w_ℓ` 同精度对接）；N 线 P3–P7 补 Ledger；N2h1-α-1 权重级；N2h1-β 水果类；N3-β→ε；R1 补强；K4。

## 6 元层诊断与 A 闸门（MEMO Phase 22–34）
- **诊断**：gpt5 MEMO 402 Phase/1.90 MB；判决 685 次但**唯一标签 48（复用率 1.10）⇒ 不可累积**；**397 Phase 仅 26（6.55%）报告共享指标变化**。
- **K1 改判（Q08）**：挂 k* ⇒ above5=**1/3**；**行为读出层 0.3316/0.3986/0.3898 = 门 6.6×** ⇒ 旧 `k1_not_triggered…operator_line_kept` 应改判。
- **K1 双轨（Q09）**：k* **`model_specific`**；读出层 **`fired_all_models`**（池化 margin **+0.9409**、E_read **0.3734**）。**K2/K3 从未被测量**（合取触发 + 无装置）。
- **Q01/Q02**：真源 62 条按**分量**复现 A5/B34/C14/D21/E20（和 94）；TESTPLAN B6/C10 不可复现；「55%/63%」= **量纲混淆**（41.5%=39/94）；`metric_dict` v2 `03887e51`，E_read=0.33162/0.39860/0.38984（0/3 过门）。
- **Q12**：新推理链 id 命中 1（Phase 3113→R55）+ 指纹 0 ⇒ 无实质违规；缺陷 **RULE-UNDECIDABLE**（D∪E 61.29%，26 复合）、**NS-COLLISION**（`R\\d\\d` 非唯一命名空间）。
- **归档（R6+R7）**：11 件 `.md` 全文 → MEMO P23–34（备份 `tests\\deepseek_temp\\_archive_r6|_r7\\gpt5_docs\\`）；`metric_dict(_v1_backup)/phase_queue` → `research\\deepseek\\atlas\\`；审计 JSON → `tests\\deepseek\\result\\`；独立复核 R6 44/0、**R7 44/0**。
- **归属判据（R7 确立）**：**谁的备忘录登记它，就是谁的线**（本线 MEMO 引用 / 他线 MEMO 0 引用 = 最强信号；须区分「直接登记」与「经并入文档间接引用」）。gpt5 目录余 20 件判为**其他线，一律未动**；`atlas_ledger.json`（n=304）为**跨线共享**，不动。

## 7 技能
`rdc-main-axis-probe`（15 臂 + 63 坑）、`rdc-phase-closeout`（**41 教训**）、`rdc-dual-arm-phase-template`。
"""

old_mem = open(MEM, "rb").read().decode("utf-8-sig")
ANCH = ["AGI_DEEPSEEK_MEMO.md", "AGI_GPT5_MEMO.md", "Phase 1–34", "b257f54f", "676,086 B",
        "P32–34", "归属判据", "真源 62 条", "Q12", "fired_all_models", "model_specific",
        "41 教训", "seal_request_v1.json", "atlas_ledger.json", "未被 git 跟踪",
        "量纲混淆", "从未被测量", "RULE-UNDECIDABLE", "跨线共享", "43/0"]
miss = [a for a in ANCH if a not in NEW_MEMORY]
assert not miss, "MEMORY 缺锚点: %s" % miss
assert len(NEW_MEMORY) < 3600, "MEMORY 过长: %d" % len(NEW_MEMORY)
open(MEM, "w", encoding="utf-8", newline="\n").write(NEW_MEMORY)
back = open(MEM, "rb").read().decode("utf-8-sig")
assert back == NEW_MEMORY, "MEMORY 回读不等"
L("MEMORY.md: %d -> %d chars  sha8=%s  锚点 %d/%d" % (len(old_mem), len(NEW_MEMORY), sha8(MEM), len(ANCH), len(ANCH)))

# ---------------- 2. wlog 追加 ----------------
R7 = """
## R7：deepseek 线归属确认 + gpt5 目录整理（2026-10-03）

- **归属判据（本轮确立，证据法）**：**谁的备忘录登记了它，就是谁的线**。逐件三处取证：①在本线 `AGI_DEEPSEEK_MEMO.md` 的出现次数与所在 Phase 节；②在 `AGI_GPT5_MEMO.md` 的出现次数；③文件自述「线：」。
- **判为 deepseek 线（本对话）＝ 3 件 .md + 1 JSON**（本线 MEMO 引用，gpt5 MEMO 0 引用）：
  - `MAIN_AXIS_VERDICT_v1.md`(`827fc48d`) → N1 主轴三段裁决（本线 Phase 4 依据；其引用的 N1/N1b/N1c 探针全为 N 线）
  - `EMBED_ANCHOR_VERDICT_v1.md`(`792f9181`) → E1 词嵌入锚点裁决（本线 Phase 2 依据）
  - `MEMO_AUDIT_2750_3148.md`(`5473f968`) → MEMO 审计（本线 18 处引用；RDC 计划之源）
  - `metric_dict_v1_backup.json`(`469c0ad1`) → 迁 `research\\deepseek\\atlas\\`
- **并入**：3 件全文 → MEMO `## Phase 32–34`（标题降一级、跳过代码围栏、剥掉文档自身 H1，与 P23–30 同法）。MEMO 623,627 B/6320 行/`7c7d2ae4` → **676,086 B/6902 行/`b257f54f`**（BOM+CRLF、bare_lf 0、**前缀逐字节不变**）；逐行包含性 **148/148 + 117/117 + 156/156**（miss=0）。
- **判为其他线（未动，20 件）**：`ATLAS_LEDGER_SPEC`/`card_set_v2`/`cleanup_ledger_20260930`、`ATLAS_PLAN_map_cracking_v2`/`MASTER_PLAN_map_linkage_v1`/`FINGERPRINT_PARADIGM_PLAN`/`fingerprint_competition_review_20260921`/`hdmcc_knowledge_map_review_20260921`/`lpf_multiaxis_gating_roadmap_v1`/`plan_v3–v6`/`research_synthesis_20260921`、`FIRST_PRINCIPLES_3090_3149`/`PARADIGM_SHIFT_VERDICT_v1`/`UNIFIED_REVIEW_ADJUDICATION_v1`（后两件自述线=RDC/T4，分别由 gpt5 Phase 3150/3151 登记）；`AGI_GPT5_MEMO*.md`/`backup/`/`code/`/`data/` 全部不动。
- **跨线共享（未动）**：`atlas_ledger.json`(`bbda63df`, n=304) —— 两条线共用的证据账本，搬走会打断他线。
- **独立复核** `tests/deepseek/verify_r7.py` → **PASS 44 / FAIL 0 ALL_PASS**（20 件其他线文件指纹逐一未变、3 件备份哈希、迁移哈希、memo 前缀=快照、Phase 1–34 连续）。
- **⬛ drift 如实记录**：MEMO 在 R6 复核（623,625 B/`aeb4e786`）之后、R7 开始之前被**外部写入**为 623,627 B/`7c7d2ae4`（**恰好 +1 个 CRLF**，PowerShell 独立读数一致，mtime 02:54:39）。局部化失败（非行尾、非连续空行）；Phase 23–30 对照 R6 归档逐行完好；该文件**未被 git 跟踪** ⇒ 无法 diff。处置：落前快照 `tests\\deepseek_temp\\_archive_r7\\AGI_DEEPSEEK_MEMO.pre_r7.bin`，以新哈希为当前值。
- **通俗版决策讲解页** `tests/deepseek/result/decisions_plain_r7.html`(`90debc19`)：归属整理表 + 主线三项决策（Q08 / C1–C6 / I1·I9）白话讲解，数字全部现场渲染。
- **技能** `rdc-phase-closeout` 新增**教训 41**（判「某文件属于哪条线」用备忘录登记法；先证归属再搬运）。
- **一句话 ×3**：① 归属判据 = 备忘录登记法；② 只有 3 件属于 deepseek 线，其余 20 件一律未动；③ 主线仍只等那一行 seal。
"""
old_w = open(WLOG, "rb").read().decode("utf-8-sig")
assert old_w.count("## R7：deepseek 线归属确认") == 0, "wlog 已含 R7 块"
new_w = old_w.rstrip("\n") + "\n" + R7
open(WLOG, "w", encoding="utf-8", newline="\n").write(new_w)
bw = open(WLOG, "rb").read().decode("utf-8-sig")
assert bw == new_w and "## R7：deepseek 线归属确认" in bw, "wlog 回读失败"
L("WLOG: %d -> %d chars  sha8=%s" % (len(old_w), len(new_w), sha8(WLOG)))

# ---------------- 3. 技能教训 41 ----------------
L41 = """41. **判「某文件属于哪条线」要用「备忘录登记法」，先证归属再搬运（R7 实证，2026-10-03）**：
```text
(a) 判据 = 谁的备忘录登记了它，就是谁的线。三处取证：①本线 MEMO 中出现次数 + 所在 Phase 节；
    ②他线 MEMO 中出现次数；③文件自述「线：」。
(b) 「本线 MEMO 引用 N 次 / 他线 MEMO 0 次」是最强单条信号 —— 他线自产的文件必被他线日志登记。
(c) 必须区分「直接登记」与「经并入文档间接引用」：后者（如本线并入了他线的文档，而该文档引用了 X）
    会把归属判反。计数要落在「谁写的、谁登记的」，不是「谁提到过」。
(d) 文件自述的「线：」可能过时（命名曾长期混用），只能当辅助证据，不能当判决依据。
(e) 跨线共享件（如多条线共写的证据账本）**不要动**：搬走会打断他线的脚本路径。
(f) 搬运前先备份 + 逐件哈希比对，删原件前再验一次；搬完做「其他线文件未被触碰」的指纹清单复核（R7：20 件逐一）。
(g) 未被 git 跟踪的大文件，动手前先落「前快照」（.bin），否则事后无法定位 drift。
(h) drift 要如实报告并尽量局部化：给「+N 字节 / +M 行 / mtime / 两进程独立读数是否一致」；
    局部化失败就明说失败，不要含糊成「内容未变」。
```

"""
old_s = open(SK, "rb").read().decode("utf-8-sig")
NEEDLE = "删前再校验一次。\n```\n\n\n## 参照实现（Phase 3125"
assert old_s.count(NEEDLE) == 1, "SKILL 锚点 count=%d" % old_s.count(NEEDLE)
new_s = old_s.replace(NEEDLE, "删前再校验一次。\n```\n\n" + L41 + "## 参照实现（Phase 3125")
assert new_s.count("41. **判「某文件属于哪条线」") == 1
open(SK, "w", encoding="utf-8", newline="\n").write(new_s)
bks = open(SK, "rb").read().decode("utf-8-sig")
assert bks == new_s and "教训 40" not in bks.split("41. **判")[0][-80:], "SKILL 回读失败"
L("SKILL: %d -> %d chars  sha8=%s" % (len(old_s), len(new_s), sha8(SK)))

open(REP, "w", encoding="utf-8").write("\n".join(log) + "\n")
L("REPORT -> %s" % REP)
