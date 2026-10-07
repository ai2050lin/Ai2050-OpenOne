# -*- coding: utf-8 -*-
"""R3 复核轮：记忆与技能写入（幂等；每步断言 + 落盘复核）。"""
import hashlib
import os

ROOT = r"D:\AI2050\Ai2050-OpenOne"
MEM = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
DAILY = os.path.join(ROOT, ".workbuddy", "memory", "2026-10-02.md")
SKILL = r"C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md"
REPORT = os.path.join(ROOT, "tests", "deepseek_temp", "_review", "_patch_memory_skill_r3.txt")

log = []


def w(p, b):
    with open(p, "wb") as f:
        f.write(b)


# ---------------- 1. MEMORY.md：追加 §9 ----------------
b = open(MEM, "rb").read()
t = b.decode("utf-8")
if "## 9 元层循环诊断" in t:
    log.append("MEMORY: §9 already present, skip (idempotent)")
else:
    SEC9 = """
## 9 元层循环诊断（2026-10-02，R3 复核轮，非 Phase）
产物：`research/gpt5/docs/LOOP_DIAGNOSIS_AND_EXIT_v1.md` + `tests/deepseek_temp/_review/loop_diagnosis_r3.html`（数据 `loop_stats_r3.json`；生成器 `tests/deepseek/_review/gen_loop_diagnosis_r3.py`）。
- **规模**：gpt5 MEMO 1.90 MB / 14,901 行 / **402 Phase**（403 h2 / 2087 h3，sha8 2a84776b）；deepseek 497 KB / 21 Phase（sha8 4ba5e22f）。
- **判决语言失衡**：判决 **685** / 预注册 580 / 否证 86 / 降级 51 / 撤回 12；**判决串 47 次 → 48 个唯一标签（复用率 1.10）= 判决不可累积**；A 级命题 5/62（8%）；「接续」字段 **380** 处（模板写死「与总目标一致 ⇒ 自动进入下一 Phase」）。
- **核心改判主张**：`k1_not_triggered…operator_line_kept` **应改判**。K1 要求「未见组合 logit 误差 >5%」，但判据被挂在「候选声称作用层位 k*」（≈7.5% 深度）⇒ above5 = **1/3** 未触发；而**行为读出层误差 0.3316 / 0.3986 / 0.3898 = 5% 门的 6.6×**。3153 读出层分解：交互格仅 **0.0–0.3%**、加性残差 34–38%、**模板内高秩散布 56.8–60.4%**（跨模型谱形相关 0.968–0.991，402 Phase 一直当噪声处理）。
- **循环四机制**：M1 验收函数错位（奖励「找到一个过门局部结构」，高维系统里几乎恒真）；M2 死线免疫（K1/K2/K3 全是跨模型**合取**，而项目自述「跨模型从不一致」⇒ 永不触发）；M3 自催化议程（议程由上一 Phase 残差自动派生，全局目标不投票；N 线 N2h1-α-1…14 共 **14 个连续 Phase 同一主题**，其中 19/20/21 连续三轮是同一量的 nf4↔bf16 复核）；M4 外部证伪被吸收（两篇外部长文均判「无需范式修正」，TDA/SAE/ODE 三处方全延后）。
- **元层三处对不上（须先修）**：① 账本分级 **B=34**（MEMO 3103）vs **B=6**（TESTPLAN §0），C=14 vs 10；② MEMO 自述「约 55%（A+B）」vs 同节计数 39/62=**63%**；③ `atlas_ledger.json` 自声明 `ledger_sha256_8`=**41d65a13** vs 实际 **bbda63df**。
- **出路（待用户裁定）**：I1 唯一全局 KPI——`E_read`（现 0.3316/0.3986/0.3898）、`E_ar(k)`（现缺失）、`C_steer`（现未测）三数每 Phase 必报，未降低任一项者标 `catalog`；I2 判决层由**行为**定义（据此改判 K1）；I3 死线改双轨（聚合统计 + **单模型否决权** ⇒ `model_specific` 不得升为机制）；I5 把 57–60% 「模板内散布」当**未识别变量**做独立因子恢复（循环出口候选）；I6 可识别性门（每个新发现须报 ≥2 个互不兼容解释 + held-out 预测分歧，据 ICLR'25）；I9 冻结 30-Phase 固定队列、删「残差自动派生下一 Phase」。
"""
    need_lf = "\r\n" not in t
    add = SEC9.replace("\n", "\n" if need_lf else "\r\n")
    if not t.endswith("\n"):
        add = ("\n" if need_lf else "\r\n") + add
    w(MEM, b + add.encode("utf-8"))
    log.append("MEMORY: §9 appended")

b2 = open(MEM, "rb").read()
t2 = b2.decode("utf-8")
assert t2.count("## 9 元层循环诊断") == 1, "§9 count != 1"
assert t2.startswith(b.decode("utf-8")[:200]), "prefix changed (must be append-only)"
log.append("MEMORY now %d B / %d chars / sha8 %s" % (len(b2), len(t2), hashlib.sha256(b2).hexdigest()[:8]))

# ---------------- 2. 技能：插入教训 35 ----------------
sb = open(SKILL, "rb").read()
st = sb.decode("utf-8")
if "35. **收尾链之外还要有「元层自洽」检查" in st:
    log.append("SKILL: lesson 35 already present, skip (idempotent)")
else:
    lines = st.split("\n")
    idx = None
    for i, l in enumerate(lines):
        if l.startswith("## 参照实现"):
            idx = i
            break
    assert idx is not None, "anchor '## 参照实现' not found"
    LESSON = """35. **收尾链之外还要有「元层自洽」检查（R3 循环诊断实证，2026-10-02）**：
    单项 Phase 全绿 ≠ 证据底座可审计。交付前对**跨文档引用**做三项对账：
    - ① **同一账本的分级分布在不同文档里必须同值**（实证：`MEMO 3103` 记 A=5/B=34/C=14/D=21/E=20，而 `RDC_TESTPLAN §0` 记 A=5/B=6/C=10/D=21/E=20，两者同称「62 条」）；
    - ② **同节内的百分比与计数必须自洽**（实证：MEMO 3103 正文「约 55%（A+B）」vs 同节计数 39/62=63%）；
    - ③ **自声明哈希必须随写入更新**（实证：`atlas_ledger.json` 自声明 `ledger_sha256_8=41d65a13`，实际 `bbda63df`）。
    - **判决标签必须复用**：实测 gpt5 线 47 次判决串产生 **48 个唯一标签（复用率 1.10）** ⇒ 判决不可跨 Phase 比较、无法累积。新 Phase 的 verdict 应优先复用既有标签；新标签须在 `metric_dict` 登记。
    - **死线的触发逻辑不得用跨模型合取**：项目自述「跨模型从不一致」⇒ 合取式死线永不触发（实测 K1 = 1/3 未触发）；且**判定层不得由被检验假设自选**（K1 挂在 k* 而非行为读出层 ⇒ 读出层误差 6.6× 于门仍未触发）。
    - **唯一全局 KPI 优于每 Phase 局部判决**：每 Phase 必报同一个必须可能失败的数字，否则只能产出「目录条目」。
"""
    lines.insert(idx, LESSON)
    w(SKILL, "\n".join(lines).encode("utf-8"))
    log.append("SKILL: lesson 35 inserted before '## 参照实现' (line %d)" % (idx + 1))

sb2 = open(SKILL, "rb").read()
st2 = sb2.decode("utf-8")
assert st2.count("35. **收尾链之外还要有「元层自洽」检查") == 1, "lesson 35 count != 1"
log.append("SKILL now %d B / sha8 %s" % (len(sb2), hashlib.sha256(sb2).hexdigest()[:8]))

# ---------------- 3. 当日 wlog：追加（append-only） ----------------
db = open(DAILY, "rb").read()
dt = db.decode("utf-8")
if "## R3 元层复核" in dt:
    log.append("DAILY: R3 section already present, skip (idempotent)")
else:
    nl = "\r\n" if "\r\n" in dt else "\n"
    SEC = nl.join([
        "",
        "## R3 元层复核：402 个 Phase 的循环诊断（非 Phase，22:2x）",
        "",
        "用户要求核对 `research/gpt5/docs/AGI_GPT5_MEMO.md`（1.90 MB / 14,901 行 / **402 个独立 Phase** / sha8 2a84776b）中哪些结论正确、哪些错误、如何跳出「每 Phase 只找到局部特征」的循环。",
        "",
        "- **做法**：全文结构统计 + 关键节精读（2750 / 3100–3153）+ 判决串与死线触发条件的正则审计 + 与领域现行标准对账。脚本 `tests/deepseek/_review/gen_loop_diagnosis_r3.py`（数据全现场解析）→ `loop_stats_r3.json` → `gen_loop_html_r3.py` 渲染 HTML。**零手工转录**。",
        "- **产出**：`research/gpt5/docs/LOOP_DIAGNOSIS_AND_EXIT_v1.md`（正文）+ `tests/deepseek_temp/_review/loop_diagnosis_r3.html`（22,633 B / sha8 e715b4f7）+ `loop_stats_r3.json`。",
        "- **核对结论**：绝大多数**局部**结论站得住（读出端简并/端口类 5 次确证、两种编码拓扑、层位分工、范数占比≠因果占比、真阴性记录、装置层位级锚）；真正的问题不在单条正确性，而在**可累积性与整体性**。",
        "- **核心改判主张**：`k1_not_triggered…operator_line_kept` 应改判——K1 判据被挂在「候选声称作用层位 k*」（≈7.5% 深度）⇒ above5=1/3 未触发；但**行为读出层误差 0.3316/0.3986/0.3898 = 5% 门的 6.6×**；3153 读出层交互格仅 0.0–0.3%，加性残差 34–38%，**模板内高秩散布 56.8–60.4%**（跨模型谱形相关 0.968–0.991，402 Phase 一直当噪声）。",
        "- **循环四机制**：M1 验收函数错位；M2 死线免疫（K1/K2/K3 全是跨模型**合取**，而项目自述「跨模型从不一致」⇒ 永不触发）；M3 自催化议程（「接续」字段 **380** 处，模板写死「与总目标一致 ⇒ 自动进入下一 Phase」）；M4 外部证伪被吸收（两篇外部长文均判「无需范式修正」，三处方全延后）。",
        "- **量化铁证**：判决 **685** 次但**判决串 47 次 → 48 个唯一标签（复用率 1.10）** ⇒ 判决不可累积；A 级命题 5/62（8%）。",
        "- **元层三处对不上**：账本分级 B=34（MEMO 3103）vs B=6（TESTPLAN §0）/ C=14 vs 10；「约 55%（A+B）」vs 计数 63%；`atlas_ledger.json` 自声明哈希 41d65a13 vs 实际 bbda63df。",
        "- **出路**：I1 唯一全局 KPI（`E_read`/`E_ar(k)`/`C_steer` 每 Phase 必报）· I2 判决层由行为定义（据此改判 K1）· I3 死线双轨 + 单模型否决权 · I5 把 57–60%「模板内散布」当**未识别变量**做因子恢复（循环出口候选）· I6 可识别性门（ICLR'25）· I9 冻结 30-Phase 队列、删「残差自动派生下一 Phase」。",
        "- **待用户裁定**：是否按 I2 正式改判 K1；是否采纳 I1/I9 的制度改动。",
        "",
    ])
    w(DAILY, db + SEC.encode("utf-8"))
    log.append("DAILY: R3 section appended")

db2 = open(DAILY, "rb").read()
assert db2[:len(db)] == db, "daily not append-only!"
log.append("DAILY now %d B (was %d)" % (len(db2), len(db)))

os.makedirs(os.path.dirname(REPORT), exist_ok=True)
open(REPORT, "w", encoding="utf-8").write("\n".join(log))
print("OK")
for x in log:
    print(" ", x)
