# -*- coding: utf-8 -*-
"""R5 压缩补丁：把 MEMORY.md 压到注入阈值以下（保留全部必需锚点）。"""
import os, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
MEM = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
REP = os.path.join(ROOT, "tests", "deepseek_temp", "_review", "_patch_r5b_report.txt")

log = []
def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

OLD_BLOCK = """- **A 闸门 Q01（元层对账）**：五处不自洽全定位根因（原 3 + 新 2：MEMO 记账本哈希 `add57ba7`≠实际 `9a3c6ff4`；`measurements` 两套 schema 且 `evidence_level` **304/304 恒定**）。口径：真源 62 条按**分量计**复现 A5/B34/C14/D21/E20（和 94）；**TESTPLAN 的 B6/C10 五口径全不可复现**；「55%/63%」= **量纲混淆**（同量纲唯一值 41.5%=39/94）。哈希自指结构性失效（自洽值 `0dc6e57a`）。产物 `META_SINGLE_SOURCE_Q01.md`。**C1–C6 待 seal**。
- **Q02 KPI 口径冻结**：`metric_dict.json` v1→v2（`03887e51`）；E_read=`b4_rel_readout_mean3seed` **0.33162/0.39860/0.38984（0/3 过门）**、E_ar/C_steer 口径冻结未测；`metrics`7+`meta_rules` 原样 ⇒ 兼容 P3150 断言。
- **Q09 死线双轨（I3）**：K1/K2/K3 触发**全为全称量词合取** ⇒ **K2/K3 从未被测量**（`phase3154` 不存在；无 top-50 覆盖率量）。K1 双轨（3151/3152 result 现场读）：**k* 层 `model_specific`**（轨 A 通过，qwen3-4b 否决）、**读出层 `fired_all_models`**（池化 margin **+0.9409** > -2·MDE 0.1933、E_read 池化 **0.3734**、3/3 否决）⇒ 与旧「未触发、算子代数线保住」**相反**。层位选择属 Q08。产物 `DEADLINE_DUAL_TRACK_Q09.md`。
- **Q12 D/E 引用审计（I4）**：账本驱动 + 命名空间感知 + 文本指纹级。**新推理链（MEMO@3104+）id 命中仅 1 处**（Phase 3113→R55 复合等级）+ **文本指纹 0 命中** ⇒ **无实质违规**。两制度缺陷：**RULE-UNDECIDABLE**（D∪E **38/62=61.29%**，其中 **26 复合**、纯 E 仅 3 ⇒「E 禁入」不可机械判定）、**NS-COLLISION**（`R\\d\\d` 非唯一命名空间，纯 id grep 必假阳性）。工具 `prop_citation_audit.py`；产物 `PROP_CITATION_AUDIT_Q12.md`。复核 **44/0、43/0、76/0**。"""

NEW_BLOCK = """- **Q01 元层单一真源 / Q02 KPI 口径**：五处不自洽全定位根因（含记账本哈希 `add57ba7`≠实际 `9a3c6ff4`；`measurements` 两套 schema、`evidence_level` 304/304 恒定）；真源 62 条按**分量**复现 A5/B34/C14/D21/E20，**TESTPLAN 的 B6/C10 五口径不可复现**、「55%/63%」=**量纲混淆**（同量纲唯一值 41.5%）、哈希自指失效（自洽值 `0dc6e57a`）⇒ `META_SINGLE_SOURCE_Q01.md`，**C1–C6 待 seal**。`metric_dict.json` v1→v2(`03887e51`)：E_read **0.33162/0.39860/0.38984（0/3 过门）**、E_ar/C_steer 未测；`metrics`7+`meta_rules` 原样 ⇒ 兼容 P3150。
- **Q09 死线双轨（I3）**：K1/K2/K3 触发**全为合取** ⇒ **K2/K3 从未被测量**（`phase3154` 不存在；无 top-50 覆盖率量）。K1 双轨（3151/3152 现场读）：**k* 层 `model_specific`**（轨 A 通过、qwen3-4b 否决）、**读出层 `fired_all_models`**（池化 margin **+0.9409**、E_read 池化 **0.3734**、3/3 否决）⇒ 与旧「未触发」**相反**；层位属 Q08。产物 `DEADLINE_DUAL_TRACK_Q09.md`。
- **Q12 D/E 引用审计（I4）**：账本驱动+命名空间感知+文本指纹。**新链 id 命中仅 1**（Phase 3113→R55 复合）、**指纹 0** ⇒ **无实质违规**。缺陷：**RULE-UNDECIDABLE**（D∪E 38/62=61.29%、26 复合、纯 E 仅 3 ⇒「E 禁入」不可判定）、**NS-COLLISION**（`R\\d\\d` 非唯一命名空间 ⇒ 纯 id grep 必假阳性）。工具 `prop_citation_audit.py`。复核 **44/0、43/0、76/0**。"""

OLD_DIAG = "- **诊断核心**：gpt5 MEMO 1.90 MB/14,901 行/**402 Phase**；判决 **685** 次但**判决串 47→48 唯一标签（复用率 1.10）⇒ 不可累积**；A 级 5/62(8%)；「接续」**380** 处。**改判主张**：`k1_not_triggered…operator_line_kept` **应改判**——判据挂在 k*（≈7.5%）⇒ above5=**1/3** 未触发；但**行为读出层 0.3316/0.3986/0.3898 = 5% 门的 6.6×**；3153 读出层：交互格 **0.0–0.3%**、加性残差 34–38%、**模板内高秩散布 56.8–60.4%**（谱相关 0.968–0.991）。四机制与 I1–I11 见诊断/宪法。"
NEW_DIAG = "- **诊断核心**：MEMO 1.90 MB/14,901 行/**402 Phase**；判决 **685** 次但**判决串 47→48 唯一标签（1.10）⇒ 不可累积**；A 级 5/62(8%)；「接续」**380** 处。**K1 改判主张**：判据挂在 k*（≈7.5%）⇒ above5=**1/3**；但**读出层 0.3316/0.3986/0.3898 = 5% 门的 6.6×**；3153 读出层交互格 **0.0–0.3%**、加性残差 34–38%、**高秩散布 56.8–60.4%**（谱相关 0.968–0.991）。四机制与 I1–I11 见诊断/宪法。"

OLD_PROD = "产物：`LOOP_DIAGNOSIS_AND_EXIT_v1.md`（`013d08a9`）+ `RDC_RESEARCH_CONSTITUTION_v1.md`（`01df6398`）+ `atlas/phase_queue_v1.json`（`675836fd`，Q01–Q30）；数据 `loop_stats_r3.json`；页 `loop_diagnosis_r3.html` / `q01q02_gate_r4.html` / `q09q12_gate_r5.html`。"
NEW_PROD = "产物：`LOOP_DIAGNOSIS_AND_EXIT_v1.md`(`013d08a9`) + `RDC_RESEARCH_CONSTITUTION_v1.md`(`01df6398`) + `atlas/phase_queue_v1.json`(`675836fd`, Q01–Q30)；数据 `loop_stats_r3.json`；页 `loop_diagnosis_r3`/`q01q02_gate_r4`/`q09q12_gate_r5`.html。"

t = open(MEM, "rb").read().decode("utf-8")
b0 = len(t)
for old, new in [(OLD_BLOCK, NEW_BLOCK), (OLD_DIAG, NEW_DIAG), (OLD_PROD, NEW_PROD)]:
    c = t.count(old)
    assert c == 1, "count=%d for %r" % (c, old[:60])
    t = t.replace(old, new)
open(MEM, "wb").write(t.encode("utf-8"))
b1 = len(t)

keys = ["Q01 元层单一真源 / Q02 KPI 口径", "Q09 死线双轨", "Q12 D/E 引用审计",
        "fired_all_models", "+0.9409", "RULE-UNDECIDABLE", "NS-COLLISION", "0dc6e57a",
        "03887e51", "38 教训", "76/0", "C1–C6 待 seal", "bbda63df", "4ba5e22f", "675836fd", "01df6398"]
miss = [k for k in keys if k not in t]
log.append("MEMORY %d -> %d chars (delta %+d) sha8=%s  BOM=%s  CR=%s" % (
    b0, b1, b1 - b0, sha8(MEM), open(MEM, "rb").read()[:3] == b"\xef\xbb\xbf", b"\r" in open(MEM, "rb").read()))
log.append("missing keys: %s" % (miss if miss else "none"))
log.append("under 5250: %s" % (b1 < 5250))
open(REP, "w", encoding="utf-8", newline="\n").write("\n".join(log) + "\n")
print("\n".join(log))
assert not miss, "缺锚点: %s" % miss
assert b1 < 5250, "仍过长: %d" % b1
