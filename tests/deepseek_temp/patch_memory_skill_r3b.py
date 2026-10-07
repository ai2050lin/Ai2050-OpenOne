# -*- coding: utf-8 -*-
"""R3 续研：把本轮（宪法 + 冻结队列）写入工作区 MEMORY.md、当日日志与 closeout 技能。
幂等：每条变更先测 old/new 双向 count，已存在则跳过。"""
import os, io

root = r"D:\AI2050\Ai2050-OpenOne"
rep = []


def rep_write(path, old, new, note):
    """精确单次替换；old 不存在且 new 已存在则视为已完成（幂等）。"""
    b = open(path, "rb").read()
    t = b.decode("utf-8")
    c_old = t.count(old)
    c_new = t.count(new)
    if c_old == 0 and c_new >= 1:
        rep.append("[SKIP] %s（已完成）" % note)
        return False
    assert c_old == 1, "%s: old count=%d (期望 1)" % (note, c_old)
    t2 = t.replace(old, new)
    assert t2.count(new) >= 1
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(t2)
    rep.append("[OK]   %s（old=%d new=%d）" % (note, c_old, t2.count(new)))
    return True


def rep_append(path, block, note, marker, header=None):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(header or "# 工作日志\n")
        rep.append("[NEW]  %s 新建" % note)
    t = open(path, "rb").read().decode("utf-8")
    if marker in t:
        rep.append("[SKIP] %s（已完成）" % note)
        return False
    if not t.endswith("\n"):
        t += "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(t + block)
    rep.append("[OK]   %s（追加 %d 字符）" % (note, len(block)))
    return True


# ---------- 1. MEMORY.md ----------
MP = os.path.join(root, ".workbuddy/memory/MEMORY.md")
rep_write(MP,
          "- **⚠ 待用户裁定（§9）**：是否按 I2 正式改判 K1；是否采纳 I1（唯一全局 KPI）/ I9（30-Phase 固定队列）。",
          "- **⚠ 待用户裁定**：是否按 I2 正式改判 K1（Q08 修正文本已备，待 seal）；是否 seal I1（唯一全局 KPI）/ I9（冻结队列）——**宪法与队列已生成**（§9）。",
          "MEMORY §7 待裁定行")
rep_write(MP,
          "`rdc-main-axis-probe`（15 臂+**63 坑**）、`rdc-phase-closeout`（**35 教训**/十四次链）、`rdc-dual-arm-phase-template`。",
          "`rdc-main-axis-probe`（15 臂+**63 坑**）、`rdc-phase-closeout`（**36 教训**/十四次链）、`rdc-dual-arm-phase-template`。",
          "MEMORY §8 技能计数")
rep_append(MP,
           "- **2026-10-03 续研（I1 落地）**：**397 Phase 中仅 26 个（6.55%）报告共享指标变化**（窄词表 18）⇒ 93.45% 是「目录条目」。产物：`RDC_RESEARCH_CONSTITUTION_v1.md`（`01df6398`，I1–I11 宪法）+ `atlas/phase_queue_v1.json`（`675836fd`，冻结 Q01–Q30）；生成器/分类器/独立复核在 `tests/deepseek[_temp]/_review/`（复核 44/0）。**新纪律：判据挂行为读出层；词汇/份额类代理须报敏感区间。**\n",
           "MEMORY §9 追记",
           "2026-10-03 续研（I1 落地）")

# ---------- 2. 当日日志 ----------
DL = os.path.join(root, ".workbuddy/memory/2026-10-03.md")
daily_block = "\n## Phase 21 之后（R3 续研：验收函数落地，非 Phase）\n\n- **把 I1 操作化为可计算量**：`tests/deepseek/_review/classify_phases_r3.py` 扫描 gpt5 MEMO，启发式判定「共享指标变化」⇒ **397 个 Phase 节中仅 26 个（6.55%）报告了「指标名 + 小数 A→B」**（窄词表 18，区间 [18,26]）⇒ **93.45% 是目录条目**。\n- **两份治理交付件**：\n  - `research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md`（10,388 B / sha8 `01df6398` / LF-only）：I1–I11 全部落为判据（KPI 三数、行为判决层、死线双轨、可识别性门、未识别变量程序、单一真源、冻结队列、外部评审规则、工程去摩擦）。\n  - `research/gpt5/atlas/phase_queue_v1.json`（9,669 B / sha8 `675836fd` / 30 项 Q01–Q30 / 唯一议程来源）。\n- **生成与复核**：`gen_constitution_r3.py`（数据驱动，0 占位残留）；`disk_verify_continue_r3.py` 独立复核 **PASS 44 / FAIL 0**（源文件级重算：Phase 节 397、宽词表 advance 精确复现 26、ledger 自声明 `41d65a13` ≠ 实际 `bbda63df`、MEMO 中 E_ar/E_read/C_steer 各 0 次）。\n- **一处工程修正**：生成器文本模式写盘产生 CRLF（10388 vs 10555 差 166 行）⇒ 改 `newline=\"\\n\"` 强制 LF，使 hash 跨平台可复现。\n- **未改动**：MEMO 原文、既有判决、Ledger 均未触碰；K1 改判仍为待 seal 的决策点（Q08）。\n"
rep_append(DL, daily_block, "当日日志 2026-10-03", "R3 续研：验收函数落地",
           header="# 工作日志 2026-10-03\n")

# ---------- 3. 技能 rdc-phase-closeout ----------
SK = r"C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md"
lesson36 = """36. **「验收函数」必须可失败：把 I1–I3 落成判据（R3 续研实证，2026-10-03）**：
    - **① 判据挂行为层，不挂假设自选的层**：同一条 5% 门，挂在「候选声称作用层位 k*」得 above5=1/3 未触发；挂在行为读出层则三模型误差 **0.3316/0.3986/0.3898 = 门的 6.6×**。判定层由被检验假设自选 ⇒ 必然保住主线。
    - **② 死线用聚合量 + 单模型否决权，禁用跨模型合取**：跨模型异质 ⇒ 合取式永不触发。
    - **③ 每 Phase 必报同一全局 KPI（`E_read`/`E_ar(k)`/`C_steer`），未降低者记 `catalog` 而非 `advance`**：启发式可验证——gpt5 MEMO **397 个 Phase 中仅 26 个（6.55%）**报告了「指标名 + 小数 A→B」的共享指标变化 ⇒ 93.45% 是目录条目。
    - **④ 代理指标（词汇/份额类）必须报敏感区间**：同一分类器宽词表得 26、窄词表得 18 ⇒ 只可报「区间 + 结论不随词表变」，不可报单点。
    - **⑤ 议程必须冻结**：删「残差 → 自动派生下一 Phase」模板（gpt5「接续」字段 **380** 处），改为固定 30-Phase 队列 + backlog；**唯一允许生成新队列的点**。
    - **⑥ 写盘一律 `newline="\\n"` 强制 LF**：文本模式在 Windows 产生 CRLF，会让同一内容的 sha256 随平台变化（实证：10388 vs 10555 B，差 166 行）。

"""
rep_write(SK,
          "## 参照实现（Phase 3125，210/210 全绿；Phase 3128，19/19 全绿）",
          lesson36 + "## 参照实现（Phase 3125，210/210 全绿；Phase 3128，19/19 全绿）",
          "skill 教训 36")

# ---------- 报告 + 复核 ----------
mb = open(MP, "rb").read()
sb = open(SK, "rb").read()
db = open(DL, "rb").read()
import hashlib
rep.append("")
rep.append("MEMORY.md : %d B / %d chars / sha8 %s / LF-only=%s" % (
    len(mb), len(mb.decode("utf-8")), hashlib.sha256(mb).hexdigest()[:8], b"\r" not in mb))
rep.append("daily     : %d B / sha8 %s" % (len(db), hashlib.sha256(db).hexdigest()[:8]))
rep.append("skill     : %d B / sha8 %s" % (len(sb), hashlib.sha256(sb).hexdigest()[:8]))
# 回读断言
mt = mb.decode("utf-8")
assert "6.55" in mt and "01df6398" in mt and "675836fd" in mt and "36 教训" in mt
assert "2026-10-03 续研" in mt
st = sb.decode("utf-8")
assert "36. **「验收函数」必须可失败" in st and "63. " not in st.split("36. **")[0][-40:]
rep.append("回读断言: 全部通过")
open(os.path.join(root, "tests/deepseek_temp/_review/_patch_memory_skill_r3b.txt"), "w",
     encoding="utf-8").write("\n".join(rep))
print("OK")
for x in rep:
    print(x)
