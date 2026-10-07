# -*- coding: utf-8 -*-
"""R5 落盘：MEMORY.md（压缩重排 + Q09/Q12）+ 当日 wlog + 技能教训 38。幂等，逐条 assert count==1。"""
import os, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
MEM = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
DL = os.path.join(ROOT, ".workbuddy", "memory", "2026-10-03.md")
SK = r"C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md"
REP = os.path.join(ROOT, "tests", "deepseek_temp", "_review", "_patch_r5_report.txt")

log = []
def L(s=""):
    log.append(s)
def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

def patch(path, pairs, note):
    t = open(path, "rb").read().decode("utf-8")
    before = len(t)
    for old, new in pairs:
        c = t.count(old)
        assert c == 1, "[%s] count=%d for %r" % (note, c, old[:70])
        t = t.replace(old, new)
    open(path, "wb").write(t.encode("utf-8"))
    after = len(t)
    L("[%s] %d -> %d chars (%+d) sha8=%s" % (note, before, after, after - before, sha8(path)))
    return t

# ---------------- 1) MEMORY.md ----------------
P_MEM = [
 # §7 进度
 ("- **A 闸门进度**：**Q01/Q02 已完成**（§9 末）；Q09（死线双轨）/Q12（D/E 引用审计）可零 GPU 直做；Q03–Q07 为 KPI 装置。",
  "- **A 闸门进度**：**Q01/Q02/Q09/Q12 已完成**（§9 末，零 GPU）；Q03–Q07 为 KPI 装置（需 GPU）；Q08/Q10/Q11 待 seal 或需 GPU。"),
 # §8 技能计数
 ("（**37 教训**/十四次链）", "（**38 教训**/十四次链）"),
 # §5 本机缺陷补一条
 ("（P21：identity-probe 取权重 / argmax 并列内翻转 / 跨模型地板不可比 / 复核脚本 `sha8`-vs-64hex 错配⇒修脚本不动产物）。",
  "（P21：identity-probe 取权重 / argmax 并列内翻转 / 跨模型地板不可比 / 复核脚本 `sha8`-vs-64hex 错配⇒修脚本不动产物）。**Q09/Q12 新增**：合取式死线 ⇒ 两条从未被测量；`R\\d\\d` 非唯一命名空间 ⇒ 纯 id grep 必假阳性（改走文本指纹）。"),
 # §9 首行产物清单补页
 ("页 `loop_diagnosis_r3.html` / `q01q02_gate_r4.html`。",
  "页 `loop_diagnosis_r3.html` / `q01q02_gate_r4.html` / `q09q12_gate_r5.html`。"),
 # §9 诊断核心压缩
 ("四机制（验收错位/死线免疫/自催化/吸收证伪）与 I1–I11 全文见诊断与宪法。",
  "四机制与 I1–I11 见诊断/宪法。"),
 # §9 Q01/Q02 两长条 -> 四条紧凑（Q01/Q02 压缩 + Q09/Q12 新增）
 ("""- **A 闸门 Q01（元层对账）**：五处不自洽全定位根因（原 3 + 新 2：MEMO 记 `proposition_ledger`=`add57ba7`≠实际 `9a3c6ff4`；`measurements` 两套 schema 且 `evidence_level` **304/304 恒为常量**）。**口径**：真源 62 条（review57+new5），按**分量计**复现 MEMO A5/B34/C14/D21/E20（和 94）；**TESTPLAN 的 B6/C10 五口径全不可复现**；「55%/63%」= 分量数÷条数**量纲混淆**（同量纲唯一成立值 41.5%=39/94）。**哈希**：自声明 `41d65a13`=追加前旧值（18 候选全不命中）⇒ 自指结构性失效；自洽值 `0dc6e57a`（content_excluding_self，回写不变）。产物 `META_SINGLE_SOURCE_Q01.md`+`meta_single_source_v4.json`。**C1–C6 待 seal**。
- **A 闸门 Q02（KPI 冻结）**：`metric_dict.json` **v1→v2**（`469c0ad1`→`03887e51`，v1 备份在册）；新增 `global_kpis`——E_read=`k1_model_report.b4_rel_readout_mean3seed` **0.33162/0.39860/0.38984（0/3 过 5% 门）**、E_ar(k)/C_steer 口径冻结未测；`metrics`7 项 + `meta_rules` **原样** ⇒ 兼容 P3150 历史断言。**复核 44/0 + 43/0**。""",
  """- **Q01 元层单一真源对账**：五处不自洽全定位根因（原 3 + 新 2：MEMO 记账本哈希 `add57ba7`≠实际 `9a3c6ff4`；`measurements` 两套 schema 且 `evidence_level` **304/304 恒定**）。口径：真源 62 条按**分量计**复现 A5/B34/C14/D21/E20（和 94）；**TESTPLAN 的 B6/C10 五口径全不可复现**；「55%/63%」= **量纲混淆**（同量纲唯一值 41.5%=39/94）。哈希自指结构性失效（自洽值 `0dc6e57a`）。产物 `META_SINGLE_SOURCE_Q01.md`。**C1–C6 待 seal**。
- **Q02 KPI 口径冻结**：`metric_dict.json` v1→v2（`03887e51`）；E_read=`b4_rel_readout_mean3seed` **0.33162/0.39860/0.38984（0/3 过门）**、E_ar/C_steer 口径冻结未测；`metrics`7+`meta_rules` 原样 ⇒ 兼容 P3150 断言。
- **Q09 死线双轨（I3）**：K1/K2/K3 触发**全为全称量词合取** ⇒ **K2/K3 从未被测量**（`phase3154` 不存在；无 top-50 覆盖率量）。K1 双轨（3151/3152 result 现场读）：**k* 层 `model_specific`**（轨 A 通过，qwen3-4b 否决）、**读出层 `fired_all_models`**（池化 margin **+0.9409** > -2·MDE 0.1933、E_read 池化 **0.3734**、3/3 否决）⇒ 与旧「未触发、算子代数线保住」**相反**。层位选择属 Q08。产物 `DEADLINE_DUAL_TRACK_Q09.md`。
- **Q12 D/E 引用审计（I4）**：账本驱动 + 命名空间感知 + 文本指纹级。**新推理链（MEMO@3104+）id 命中仅 1 处**（Phase 3113→R55 复合等级）+ **文本指纹 0 命中** ⇒ **无实质违规**。两制度缺陷：**RULE-UNDECIDABLE**（D∪E **38/62=61.29%**，其中 **26 复合**、纯 E 仅 3 ⇒「E 禁入」不可机械判定）、**NS-COLLISION**（`R\\d\\d` 非唯一命名空间，纯 id grep 必假阳性）。工具 `prop_citation_audit.py`；产物 `PROP_CITATION_AUDIT_Q12.md`。复核 **44/0、43/0、76/0**。"""),
]
t = patch(MEM, P_MEM, "MEMORY.final")
cc = len(t)
L("MEMORY chars=%d  (<5300: %s)" % (cc, cc < 5300))
assert cc < 5300, "MEMORY 仍过长: %d" % cc
for k in ["Q09 死线双轨", "Q12 D/E 引用审计", "fired_all_models", "RULE-UNDECIDABLE",
          "NS-COLLISION", "38 教训", "0dc6e57a", "03887e51"]:
    assert k in t, "MEMORY 缺 %r" % k

# ---------------- 2) 当日 wlog ----------------
BLOCK = """
## R5：A 闸门 Q09（死线双轨重述）+ Q12（D/E 命题引用审计），零 GPU
- **Q09**：K1/K2/K3 原文（TESTPLAN §8.2）触发条件**全为全称量词合取**（K1 对 3 模型、K3 对 3 族）；且 **K2 同文件内两种操作化互斥**（§8.2「cos 降>50%」vs 3154 预注册「交互份额>50%」）。重述为**轨 A 聚合统计+bootstrap CI** + **轨 B 单模型否决权**，明令禁合取、禁把「任一模型不达标」当加固证据。
- **K1 重算（3151/3152 result 现场读，零手工转录）**：k* 层轨 A 通过（池化 margin **-0.025202** ≤ -2·MDE_pooled **0.011391**）但**轨 B 触发**（qwen3-4b 模型级 -0.002765 > -0.003690）⇒ **`model_specific`**；读出层轨 A+轨 B 双触发（池化 margin **+0.940920**、池化 MDE 0.193309、E_read 池化 **0.373350**、3/3 否决）⇒ **`fired_all_models`**。与旧结论「未触发、算子代数线保住」**相反**。层位选择属 Q08（待 seal）。
- **K2/K3 从未被测量**：`phase3154*` 目录 0 命中（停在 3103…3153 的预注册）；全 MEMO 无「top-50 覆盖率」量（现有「覆盖率」指微场普查 283 词，非同源）。⇒ **三条死线中两条从未开跑**。
- **Q12**：`prop_citation_audit.py`（账本驱动 + 命名空间感知 + 文本指纹级，替代只做关键词行匹配的 F4 `counterexample_grep.py`）。**新推理链（MEMO@3104+）id 命中仅 1 处**（L13546/Phase 3113 → R55，E+A 复合等级）；**文本指纹 0 命中**（29 条可构造指纹的 D/E 命题）⇒ **无实质违规**。
- **两制度缺陷**：① **RULE-UNDECIDABLE** —— D∪E **38/62=61.29%**，其中 **26 条复合等级**（E+A/B+E/E+C/D+E/E+D）、**纯 E 仅 3 条**（R03/R15/R46）⇒「E 级禁入」不可机械判定，真源须拆分量级条目。② **NS-COLLISION** —— 账本 `R\\d\\d` 非项目唯一命名空间：`LOOP_DIAGNOSIS_AND_EXIT_v1.md` 自有 R1–R11，其 R10「真阴性记录：翻译轴不存在」/R11「装置层」与账本 R10（实体感紧凑性定律）/R11（层级嵌套）**毫无关系** ⇒ 纯 id grep 必假阳性，审计须走文本指纹。③ **PCT-DIM-MIX**（承接 Q01-D2）：MEMO_AUDIT「约 2/3 命题不能进新推理链」= 34%+32% 的**分量和**；按条数 61.29%。
- **产物**：`research/gpt5/docs/DEADLINE_DUAL_TRACK_Q09.md`(`0d8633c7`) + `atlas/deadline_dual_track_v1.json`(`4d1853d3`，注：Q09 生成器重跑后哈希以脚本末次为准) + `research/gpt5/docs/PROP_CITATION_AUDIT_Q12.md` + `atlas/prop_citation_audit_v1.json`；汇报页 `tests/deepseek_temp/_review/q09q12_gate_r5.html`(`ee439e4c`)。
- **独立复核** `disk_verify_q09q12_r5.py` = **PASS 76 / FAIL 0 → ALL_PASS**（从 result 独立重算池化量/CI、从账本独立重算等级与命中、受保护文件指纹 MEMO `2a84776b`/账本 `9a3c6ff4`/TESTPLAN `71b85673`、LF-only）。一处**复核脚本自身**断言写反（k* 轨 A「不触发=通过」误断言为 True）⇒ 修脚本，**未动交付件**。
- **未改动**任何 MEMO 原文 / 账本 / TESTPLAN / 队列 / 宪法；Q09/Q12 在队列中仍 `pending`（生效需 seal）。
"""
rep_append = None
def append_block(path, block, marker, note):
    if os.path.exists(path):
        t = open(path, "rb").read().decode("utf-8")
    else:
        t = "# 工作日志 2026-10-03\n"
    if marker in t:
        L("[%s] 已存在，跳过" % note)
        return
    t = t.rstrip("\n") + "\n" + block
    open(path, "wb").write(t.encode("utf-8"))
    L("[%s] 追加 +%d chars, now %d chars sha8=%s" % (note, len(block), len(t), sha8(path)))

append_block(DL, BLOCK, "R5：A 闸门 Q09", "wlog 2026-10-03")

# ---------------- 3) 技能教训 38 ----------------
S38 = """
38. **元层「禁入」类规则必须先验证它是否可机械判定（Q09/Q12 实证，2026-10-03）**：
    - **(a) 规则的触发条件里出现全称量词 = 合取 = 永不触发。** K1 写「3 模型全 above 且 M1 全败」、K3 写「3 族覆盖率**均**<30%」——与「跨模型从不一致」叠加后触发概率趋 0 **且这是可预测的**。重述必须落到**聚合统计量 + bootstrap CI**，并把「单模型反例」改成**降级信号**（`model_specific` 不得升机制），而不是加固证据。
    - **(b) 动笔重述前先查「这条规则有没有装置」。** 本轮发现 K2/K3 **从未被测量**（`phase3154*` 目录 0 命中、全库无该量）。**在报告里把「无数据」与「判为不成立」严格分开**——这是最容易自欺的一处。
    - **(c) id 级审计不可信，除非先做命名空间普查。** 账本 id `R\\d\\d` 与别的文档自有编号（`LOOP_DIAGNOSIS` 的 R1–R11）碰撞 ⇒ 纯 grep 必假阳性。审计应**默认走内容指纹**（如 claim 的 6-CJK-gram），id 只作辅助且必须逐条附上下文 + 人工核定表（F4 原文即要求「人工确认」）。
    - **(d) 审计脚本自身会污染语料。** 把审计报告写进被扫目录 ⇒ 第二次运行扫到自己的输出（本轮实测 id 命中从 24 虚增到 362）。**必须显式排除自身输出与历史快照（归档 MEMO）**，并在报告里写明排除清单。
    - **(e) 复合分级使「哪一部分禁入」不可判定。** D∪E 61% 中有 26 条是 `E+A`/`B+E` 这类复合标签：撤回的是 E 分量，A/B/C 分量仍可引用。真源须拆**分量级条目**（独立 grade + 独立 id）。
"""
t = open(SK, "rb").read().decode("utf-8")
if "38. **元层「禁入」类规则" in t:
    L("[skill] 教训 38 已存在，跳过")
else:
    anchor = "## 参照实现"
    assert t.count(anchor) == 1, "skill 锚点不唯一"
    t2 = t.replace(anchor, S38.strip("\n") + "\n\n" + anchor)
    open(SK, "wb").write(t2.encode("utf-8"))
    L("[skill] 教训 38 追加 +%d chars, now %d B sha8=%s" % (len(S38), os.path.getsize(SK), sha8(SK)))

# 回读复核
mt = open(MEM, "rb").read().decode("utf-8")
dt = open(DL, "rb").read().decode("utf-8")
st = open(SK, "rb").read().decode("utf-8")
L("")
L("RE-READ: MEMORY chars=%d sha8=%s" % (len(mt), sha8(MEM)))
L("RE-READ: wlog  chars=%d sha8=%s  has-R5=%s" % (len(dt), sha8(DL), "R5：A 闸门 Q09" in dt))
L("RE-READ: skill bytes=%d sha8=%s  has-38=%s" % (os.path.getsize(SK), sha8(SK), "38. **元层「禁入」类规则" in st))
for k in ["Q09 死线双轨", "Q12 D/E 引用审计", "76/0", "38 教训"]:
    L("  MEMORY has %r: %s" % (k, k in mt))

open(REP, "w", encoding="utf-8", newline="\n").write("\n".join(log) + "\n")
print("\n".join(log))
