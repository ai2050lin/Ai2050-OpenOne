# -*- coding: utf-8 -*-
"""R6c: ① 重写工作区 MEMORY.md（压缩到注入阈值内 + 登记 v4 规范）；
② 追加 wlog R6 块；③ 技能 rdc-phase-closeout 追加教训 40。逐处回读校验。"""
import os, hashlib, io

ROOT = r"D:\AI2050\Ai2050-OpenOne"
MEM = os.path.join(ROOT, r".workbuddy\memory\MEMORY.md")
WLOG = os.path.join(ROOT, r".workbuddy\memory\2026-10-03.md")
SKILL = r"C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md"
REPORT = os.path.join(ROOT, r"tests\deepseek_temp\_r6c_report.txt")
log = []
def A(s): log.append(s)
def sh8b(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

# ---------- 1) MEMORY.md ----------
NEW_MEM = r"""# RDC/LPF 项目纪律（工作区长期记忆）
> 权威记录 `research\deepseek\docs\AGI_DEEPSEEK_MEMO.md`（**唯一研究日志**，Phase 1–31）。本文件仅跨轮索引，细节回查 MEMO / 技能。

## 0 约定（v4，2026-10-03，**仅本对话遵守**）
- 日志唯一落点 = `AGI_DEEPSEEK_MEMO.md`（append-only、BOM+CRLF、`bare_lf 0`、标题 `## Phase {N}: 短标题 [hh:mm]`）；**不再新建其他 `.md`**。`AGI_GPT5_MEMO.md` 属其他 AI 线，**不读不改不追加**。
- 三桶：测试脚本 `tests\deepseek\`；临时脚本 `tests\deepseek_temp\`；测试结果 `tests\deepseek\result\`。历史 `Phase1..21\` 不动。
- MEMO：P1–21 N 线；**P22 = A 闸门 R3–R5b**；**P23–30 = 8 件并入文档全文**；**P31 = 路由归一 R6**。现 `aeb4e786`（6319 行）。
- Ledger `atlas_ledger.json` n=**304** `bbda63df`（**N 线 P3–P7 待补**）。「好的，继续」= AI 主导续研；**关键发现重复 3 次**。

## 1 装置铁律（63 坑全文见 skill `rdc-main-axis-probe`）
(a) 份额只用**精确可加向量预算**；(b) SMOKE 必做**必看数字**；(o) 多 Edit 静默丢 ⇒ Python 补丁 + `assert count==1` + 回读；(ad) 实现与 seal **逐字一致**；(ae) 数字**一律 result 现场渲染**；(af) **冻结锚重验**。
其余：patch 剔末层；写入窗=相邻最大增量；`jump/max|effect|≥0.5`；剔实例 token 同剔类别 token；双剂量 `x*·r_ℓ`；固定基报 `overlap`；paired `margin` 负=候选优。

## 2 N 线主线（P4→P21）
- P4–P7 主轴三段（嵌入=词典/层=开关/**权重绑定定 is-a 落点**）；单头 share_max **3.0%**；读位槽 **G−1=5 维**；跨族近正交 ⇒ 无通用类别算子。
- P8–P11 写入端**分布式**（向量预算 MLP 0.472）；L6 内**无阈值增益** ⇒ **栈=软门**；深端塌陷主因**方向失配**。
- P12–P16 `rho(xhalf,depth)=−0.783`；**P16 否证 ⇒ P12/13/14 深度表述撤回**；域 `REACH={ℓ:ρ≥0.10}`。
- P17–P21 `w_ℓ`+`com_V` ≈26 ≫ median(REACH)；`spearman(w,|b|)` 与 P17 **反号 ⇒ P17 P6 对象错配**；跨精度 7/9、A0_bf16 **逐位复现 P8 锚** ⇒「分布式搬运」非 nf4 产物。

## 3 挂账与限界
- **核心三条限界**：① 否定臂基线不成立；② 「未见类别」仍崩（水果 0.04/0.05）；③ **激活级干预 ≠ 权重级证明**（其余 10 条见 MEMO / 技能 26–40）。
- R1/R2：10 条成立；P1 纠错 1（K_d）+降级 4+挂账 5。G 线索引：3151 k3_only / 3152 k1_not_triggered / 3153 coverage_partial；判决符号 rev-3151b（负=优）。

## 4 本机缺陷（Windows）
- bash shim `ls/rm/tail/dirname/cd` 坏、内联 `python -c` stdout 常丢、**反引号被吃** ⇒ Python 文件化 + 写 `.txt` 再 Read。
- **`Edit` 与脚本日志均可能幻影** ⇒ 中文大段走 Python 补丁（**禁凭记忆写匹配串，先 dump 真实磁盘**）；**落盘证据只能靠独立进程 re-hash**。`Read` 大文件视图陈旧 ⇒ 改没改用 Grep/Python。

## 5 下一步 / 死线
- **⚠ 待 seal**（`tests\deepseek\result\seal_request_v1.json`）：Q08（K1 改判：读法甲/乙）、Q01 更正表 **C1–C6**、I1/I9 ⇒ 之后才进 B 闸门 Q03（需 GPU）。
- 零 GPU：**Q01/Q02/Q09/Q12 已完成**（复核 44/0、43/0、76/0）。并列挂账：下一实验 Phase（P8 `share_v` × P16/P17 `w_ℓ` 同精度对接）；N 线 P3–P7 补 Ledger；N2h1-α-1 权重级；N2h1-β 水果类；N3-β→ε；R1 补强；K4。

## 6 元层诊断与 A 闸门（MEMO Phase 22–31）
- **诊断**：gpt5 MEMO 402 Phase/1.90 MB；判决 685 次但**唯一标签 48（复用率 1.10）⇒ 不可累积**；**397 Phase 仅 26（6.55%）报告共享指标变化**。
- **K1 改判（Q08）**：挂 k* ⇒ above5=**1/3**；**行为读出层 0.3316/0.3986/0.3898 = 门 6.6×** ⇒ 旧 `k1_not_triggered…operator_line_kept` 应改判。
- **K1 双轨（Q09）**：k* **`model_specific`**；读出层 **`fired_all_models`**（池化 margin **+0.9409**、E_read **0.3734**）。**K2/K3 从未被测量**（合取触发 + 无装置）。
- **Q01/Q02**：真源 62 条按**分量**复现 A5/B34/C14/D21/E20（和 94）；TESTPLAN B6/C10 不可复现；「55%/63%」= **量纲混淆**（41.5%=39/94）；`metric_dict` v2 `03887e51`，E_read=0.33162/0.39860/0.38984（0/3 过门）。
- **Q12**：新推理链 id 命中 1（Phase 3113→R55）+ 指纹 0 ⇒ 无实质违规；缺陷 **RULE-UNDECIDABLE**（D∪E 61.29%，26 复合）、**NS-COLLISION**（`R\d\d` 非唯一命名空间）。
- **归档**：8 件 `.md` 全文 → MEMO P23–30（备份 `tests\deepseek_temp\_archive_r6\gpt5_docs\`）；`metric_dict/phase_queue` → `research\deepseek\atlas\`，审计 JSON → `tests\deepseek\result\`。

## 7 技能
`rdc-main-axis-probe`（15 臂 + 63 坑）、`rdc-phase-closeout`（**40 教训**）、`rdc-dual-arm-phase-template`。
"""
assert "## 0 约定" in NEW_MEM and "aeb4e786" in NEW_MEM
ANCH = ["AGI_DEEPSEEK_MEMO.md", "AGI_GPT5_MEMO.md", "P23–30", "aeb4e786",
        "tests\\deepseek\\result", "fired_all_models", "model_specific",
        "RULE-UNDECIDABLE", "NS-COLLISION", "rdc-phase-closeout", "40 教训", "C1–C6"]
miss = [a for a in ANCH if a not in NEW_MEM]
assert not miss, "miss=%s" % miss
old = open(MEM, "rb").read()
open(MEM, "w", encoding="utf-8", newline="").write(NEW_MEM)
back = open(MEM, "rb").read().decode("utf-8")
assert back == NEW_MEM, "MEMORY readback mismatch"
A("[MEMORY] %d -> %d chars  sha8 %s -> %s  anchors %d/%d" % (
    len(old.decode("utf-8")), len(NEW_MEM), sh8b(MEM) if False else "n/a", sh8b(MEM), len(ANCH) - len(miss), len(ANCH)))
assert len(NEW_MEM) < 3600, "still too long: %d" % len(NEW_MEM)

# ---------- 2) wlog ----------
WLOG_BLOCK = """

## R6：规范归一 —— 研究日志唯一化 + deepseek 目录全归位（2026-10-03）

- 用户指令（v4）：本对话全部研究日志只写 `research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`；`AGI_GPT5_MEMO.md` 属其他 AI 线（不读不改不追加）；测试脚本 → `tests\\deepseek\\`、临时脚本 → `tests\\deepseek_temp\\`、测试结果 → `tests\\deepseek\\result\\`；**仅本对话遵守**，避免与其他路线混合。
- **并入**：`research/gpt5/docs` 下 8 件（RDC_TESTPLAN_v1 `71b85673` / RDC_RESEARCH_CONSTITUTION_v1 `01df6398` / LOOP_DIAGNOSIS_AND_EXIT_v1 `013d08a9` / META_SINGLE_SOURCE_Q01 `95366ca9` / METRIC_DICT_Q02 `ba518fea` / DEADLINE_DUAL_TRACK_Q09 `0d8633c7` / PROP_CITATION_AUDIT_Q12 `32933855` / SEAL_REQUEST_A_GATE `7678cbe0`）**全文**并入 MEMO → `## Phase 23–30`（标题降一级）；原件备份 `tests\\deepseek_temp\\_archive_r6\\gpt5_docs\\` 后删除。
- **迁出**：`research/gpt5/atlas` 的 `metric_dict.json`(`03887e51`)、`phase_queue_v1.json`(`675836fd`) → `research\\deepseek\\atlas\\`；`deadline_dual_track_v1`(`4d1853d3`)、`prop_citation_audit_v1`(`4c2ea9d2`)、`meta_single_source_v4`(`f9d7ede6`)、`seal_request_v1`(`d871a4b4`) → `tests\\deepseek\\result\\`。
- **分桶**：17 个耐用脚本 → `tests/deepseek/`；27 个 probe/patch → `tests/deepseek_temp/`；58 个结果文件 + `memo_review_20261001/` → `tests/deepseek/result/`（新建）；两处 `_review/` 目录删除。
- **死路径补丁**：归位后 17 个耐用脚本内嵌的 `_review` 输出目录已不存在 ⇒ 32 处改写为 `tests/deepseek/result`（复核：残留死路径 0、`propositions_review` 哨兵 6 处未误伤）。
- **MEMO 指纹**：507,779 B / 4832 行 / `1f0bac0e` → **623,625 B / 6319 行 / `aeb4e786`**（BOM+CRLF、bare_lf 0、固定区前缀逐字节不变、Phase 标题 24 个）；Phase 22 标题删去遗留「（非 Phase 编号）」标记（append-only 的唯一格式例外）。
- **一句话 ×3**：① 研究日志唯一落点 = `AGI_DEEPSEEK_MEMO.md`；② 三类落点仅本对话遵守，G 线目录不再承载 deepseek 产物；③ 全程未改任何既有判定（并入为全文搬运）。
"""
wt = open(WLOG, "rb").read().decode("utf-8-sig")
assert "## R6：规范归一" not in wt, "R6 block already present"
if not wt.endswith("\n"):
    wt += "\n"
open(WLOG, "w", encoding="utf-8", newline="").write(wt + WLOG_BLOCK)
wb = open(WLOG, "rb").read().decode("utf-8")
assert wb == wt + WLOG_BLOCK, "wlog readback mismatch"
A("[WLOG] %d -> %d chars  sha8 %s" % (len(wt), len(wb), sh8b(WLOG)))

# ---------- 3) skill lesson 40 ----------
LESSON40 = """
### 教训 40：跨目录归位后，脚本内嵌的输出路径会变成死路径

**现象**：把 `tests/deepseek_temp/_review/` 整桶并入 `tests/deepseek/result/` 之后，17 个"耐用脚本"里仍写着 `_review` 输出目录 ⇒ 一旦复跑就会**重建已废弃的目录**，直接违反新落点约定（"脚本与它的产物必须同源"）。

**对策**：
- **(a) 两步分离**：归位脚本只搬文件；补丁脚本只改脚本文本。不要在一次执行里既搬又改，否则失败时无法判断是哪一半坏了。
- **(b) 白名单精确串，禁全局 regex**：用 `"tests","deepseek_temp","_review"`、`tests/deepseek_temp/_review`、`tests\\deepseek_temp\\_review`、`tests/deepseek/_review/` 逐条替换；**绝不要对 `_review` 做全局替换**——`propositions_review` / `propositions_new` 这类变量名会被误伤。
- **(c) 替换顺序**：先替换"带引号的整串"，再替换裸路径；否则后者的改写会把前者的模式重新切碎。
- **(d) 反向复核**：改完必须 grep 确认「残留死路径 = 0」，并统计一个已知的哨兵词（`propositions_review`）计数**不变**，用以证明没有误伤。
- **(e) 产物哈希会变**：改写 `generated_by` 之类**会写进产物正文**的字符串时，重跑生成的产物哈希必然改变；冻结产物的哈希一律以归位**前**的 MEMO 记录为准。
- **(f) 大段并入的标题必须降级**：把整篇 `.md` 并入 MEMO 成 `## Phase {N}` 时，正文标题要降一级（`#`→`##`），并**跳过代码围栏内的 `#`**；同时**剥掉文档自身的一级标题**，否则它会与 `## Phase` 同级，污染 MEMO 的 Phase 目录。
- **(g) 只读冻结件先备份再动**：并入后要删除的原件，必须先复制到备份目录并**逐字节比对哈希**，再删；删除前再校验一次哈希。
"""
st = open(SKILL, "rb").read().decode("utf-8-sig")
assert "### 教训 39" in st, "lesson 39 missing"
assert "### 教训 40" not in st, "lesson 40 already present"
if not st.endswith("\n"):
    st += "\n"
open(SKILL, "w", encoding="utf-8", newline="").write(st + LESSON40)
sb = open(SKILL, "rb").read().decode("utf-8")
assert sb == st + LESSON40, "skill readback mismatch"
A("[SKILL ] %d -> %d chars  sha8 %s" % (len(st), len(sb), sh8b(SKILL)))

txt = "\n".join(log)
open(REPORT, "w", encoding="utf-8").write(txt)
print(txt)
print("R6C_OK")
