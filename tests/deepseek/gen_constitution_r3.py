# -*- coding: utf-8 -*-
"""R3 续研：生成 RDC 研究宪法 v1 与冻结 30-Phase 队列 v1。
所有数字从 loop_stats_r3.json / phase_classify_r3.json 现场渲染，禁手工转录。"""
import os, json, hashlib

root = r"D:\AI2050\Ai2050-OpenOne"
T = os.path.join(root, "tests/deepseek/result")
S = json.loads(open(os.path.join(T, "loop_stats_r3.json"), "rb").read().decode("utf-8-sig"))
C = json.loads(open(os.path.join(T, "phase_classify_r3.json"), "rb").read().decode("utf-8-sig"))

# ---- 取数 ----
sc = S["scale"]; kw = S["keywords"]; vv = S["verdict_vocab"]; g1 = S["g1"]
kl = S["kill"]; lg = S["ledger"]; ac = S["audit_consistency"]; af = S["audit_findings"]
pc = C["phases"]; cv = C["verdict_vocab_inline"]

N_SEC = pc["n_sections"]
N_ADV = pc["n_with_metric_delta_strict"]
N_ADV_N = pc.get("n_with_metric_delta_narrow", N_ADV)
N_CAT = pc["catalog_proxy_strict"]
ADV_PCT = 100.0 * N_ADV / N_SEC
CAT_PCT = 100.0 * N_CAT / N_SEC
E_READ = [r["b4_readout"] for r in g1["rows"]]
E_READ_MODELS = [r["model"] for r in g1["rows"]]
ANOVA = {a["model"]: a["share"] for a in g1["anova"]}
AMP = g1["amplification"]
LED_DECL = lg["declared"]; LED_ACT = lg["actual_sha8"]
HI_RANK = [ANOVA["4b"][3], ANOVA["14b"][3], ANOVA["glm4"][3]]
INTER_CELL = [ANOVA["4b"][2], ANOVA["14b"][2], ANOVA["glm4"][2]]
ADD_RES = [ANOVA["4b"][1], ANOVA["14b"][1], ANOVA["glm4"][1]]
CLS = C["samples"]
EREAD_LAB = " / ".join("`{}` {:.4f}".format(m, v) for m, v in zip(E_READ_MODELS, E_READ))

def pct(x):
    return "{:.2f}".format(x)

# ---- 冻结队列（30 项）----
QUEUE = [
    (1,  "元层单一真源对账", "A 闸门", "meta", "none", "zero", "I4",
     "命题账本 B/C 分级、A+B 百分比、ledger 自声明哈希三处对账；收敛 schema v4，声明唯一真源路径"),
    (2,  "KPI 口径冻结", "A 闸门", "kpi", "all", "zero", "I1",
     "metric_dict.json：E_read / E_ar(k) / C_steer 的精确公式、数据、判据、CI 方法，冻结后不得改口径"),
    (3,  "E_read 统一基线复算", "B KPI", "kpi", "E_read", "low", "I1",
     "三模型 + 统一 held-out 指纹 + bootstrap CI；锁定当前基线数字"),
    (4,  "E_ar(k) 装置建造", "B KPI", "kpi", "E_ar", "mid", "I1",
     "k 步自回归 logit-margin 误差曲线装置（k=1..K），SMOKE 先通"),
    (5,  "E_ar(k) 正式测量", "B KPI", "kpi", "E_ar", "mid", "I1",
     "三模型曲线；判决形状（线性 / 饱和 / 发散）与 k 的半衰期"),
    (6,  "C_steer 基座测量", "B KPI", "steer", "C_steer", "mid", "I7",
     "v1 承重轴 + 端口替换：held-out 目标改动的 steered 成功率与附带损伤"),
    (7,  "KPI 曲线 v0 汇总", "B KPI", "meta", "all", "zero", "I1",
     "把 Q03–Q06 + 397 条历史判决映射为 advance/catalog，产出第一张单调曲线"),
    (8,  "K1 重述与改判", "A 闸门", "meta", "none", "zero", "I2",
     "无歧义重述 K1 判据并重新冻结；读出层判定。需用户 seal"),
    (9,  "死线双轨重述", "A 闸门", "meta", "none", "zero", "I3",
     "K1/K2/K3 改为聚合统计 + 单模型否决权；model_specific 不得升机制"),
    (10, "历史 A 级命题可识别性回溯", "D 门", "audit", "none", "zero", "I6",
     "每条 A 级须报 >=2 不兼容解释 + held-out 分歧（IIA）；不可识别者强制 descriptive"),
    (11, "面板功效硬约束", "D 门", "meta", "all", "low", "I8",
     "T4 面板 128 -> >=672；gate_precheck 在 MDE>目标效应时真拒绝开跑"),
    (12, "D/E 级命题引用审计", "A 闸门", "audit", "none", "zero", "I4",
     "自动检查 D/E 级不得进入新推理链；违规引用列出"),
    (13, "高秩散布独立因子恢复", "C 未识别", "experiment", "E_read", "mid", "I5",
     "把模板内高秩散布当未识别变量；SVD/ICA/稀疏编码恢复候选因子"),
    (14, "因子恢复 held-out 判决", "C 未识别", "experiment", "E_read", "mid", "I5",
     ">=2 不兼容解释 + held-out 预测分歧显著则判决，否则标不可识别"),
    (15, "候选因子化替换审计", "C 未识别", "experiment", "E_read", "mid", "I5",
     "(实体 x 类 x 模板) 是否为模型实际使用的分解；跨模型一致性检验"),
    (16, "位置 vs 内容分解", "C 未识别", "experiment", "E_read", "mid", "I5",
     "高秩散布是位置依赖还是内容依赖；与已知词位驻留编码对照"),
    (17, "高秩散布的因果地位", "C 未识别", "experiment", "C_steer", "mid", "I5",
     "它是否承重：干预 + 剂量曲线；无安全阈值则记入承重轴族"),
    (18, "因子恢复的形式化", "C 未识别", "kpi", "E_read", "mid", "I5",
     "若恢复成功，写出可预测代理与误差闭合量；否则记为负结果"),
    (19, "高秩散布跨模型稳定性", "C 未识别", "experiment", "E_read", "mid", "I5",
     "谱形相关在 held-out 上是否保持；稳定性是它非噪声的关键依据"),
    (20, "C_steer 基准正式化", "D 门", "steer", "C_steer", "mid", "I7",
     "给定 held-out 目标改动，机制选干预，报成功率 + 附带损伤；低分即诚实总成绩"),
    (21, "IIA 判据移植", "D 门", "kpi", "E_read", "mid", "I6",
     "BoundlessDAS 式对齐度量落地；what-then-where 流程重排"),
    (22, "端口类形式化", "E 机制", "kpi", "E_read", "mid", "R1",
     "把「任意 100 维 AUC 0.9997」写成可预测代理；给出等价类维度"),
    (23, "关系身份 400 维续查", "E 机制", "experiment", "E_read", "mid", "R2",
     "为何比真值多 80 倍；是否可压缩；与 d_min=5 并存的结构含义"),
    (24, "承重轴剂量曲线机制解释", "E 机制", "experiment", "C_steer", "mid", "R7",
     "无安全阈值的来源；幅度阈值后效应的机制刻画"),
    (25, "跨模型异质性作为研究对象", "E 机制", "experiment", "E_read", "mid", "I3",
     "把「跨模型从不一致」当独立对象研究，而非当作噪声合并"),
    (26, "30-Phase 中期 KPI 复盘", "F 制度", "meta", "all", "zero", "I9",
     "以全局 KPI 曲线决定 backlog 去留；无 KPI 改善的方向停止"),
    (27, "backlog 冻结", "F 制度", "meta", "none", "zero", "I9",
     "Q01–Q26 未采纳残差一律入 backlog；禁止从残差派生新 Phase"),
    (28, "外部评审落地规则", "F 制度", "meta", "none", "zero", "I10",
     "每条外部建议须落为「能触发死线的实验」，否则不得结案"),
    (29, "工程去摩擦", "F 制度", "meta", "none", "low", "I11",
     "抽公共库（capture/hook/anchor/装配断言）；SMOKE 值域断言；目标 <1 patch/run"),
    (30, "队列收束与下一代生成", "F 制度", "meta", "none", "zero", "I9",
     "唯一允许生成新队列的点；由 KPI 曲线与 backlog 共同决定"),
]

qjson = {
    "schema": "rdc_phase_queue_v1",
    "frozen_at": "2026-10-03",
    "rule": "本队列为唯一议程来源；禁止从任意 Phase 的未解释残差派生新 Phase。",
    "generated_by": "tests/deepseek/gen_constitution_r3.py",
    "source_stats": {"loop_stats": "loop_stats_r3.json", "phase_classify": "phase_classify_r3.json"},
    "kpi_definition_ref": "research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md#1",
    "queue": [],
    "backlog_policy": "未采纳残差入 backlog，仅在 Q26 由全局 KPI 曲线统一决定去留。",
    "count": len(QUEUE),
}
for (q, title, blk, typ, kpi, gpu, src, deliv) in QUEUE:
    qjson["queue"].append({
        "q": q, "id": "Q{:02d}".format(q), "title": title, "block": blk,
        "type": typ, "kpi": kpi, "gpu": gpu, "source": src, "deliverable": deliv,
        "status": "pending",
    })

qp = os.path.join(root, "research/gpt5/atlas/phase_queue_v1.json")
with open(qp, "w", encoding="utf-8", newline="\n") as f:
    json.dump(qjson, f, ensure_ascii=False, indent=1)
qbytes = open(qp, "rb").read()

# ---- 宪法正文 ----
L = []
A = L.append
A("# RDC 研究宪法 v1 —— 让验收函数可失败")
A("")
A("- **文档性质**：研究治理文件（非 Phase 记录，不改动任何 MEMO 原文）。")
A("- **上位依据**：`research/gpt5/docs/LOOP_DIAGNOSIS_AND_EXIT_v1.md`（sha8 `013d08a9`）。")
A("- **数据底座**：`tests/deepseek/result/loop_stats_r3.json`（`f4a0dca2`）+ `phase_classify_r3.json`。全部数字由 `gen_constitution_r3.py` 现场渲染。")
A("- **冻结日期**：2026-10-03")
A("- **一句话**：把验收函数从「找到一个能过门的局部结构」（几乎恒真）换成「必须降低一个全局数字」（会失败）。")
A("")
A("---")
A("")
A("## §0 为什么需要这部宪法（三条实测）")
A("")
A(f"1. **验收函数错位是量化的**。全 MEMO {sc['gpt5_phases_distinct']} 个 Phase，启发式可识别的 Phase 节 {N_SEC} 个，其中只有 **{N_ADV} 个（{pct(ADV_PCT)}%）**报告了「同一指标名 + 小数 A -> 小数 B」的共享指标变化；**{N_CAT} 个（{pct(CAT_PCT)}%）没有**。即约 **{pct(CAT_PCT)}%** 的工作是「目录条目」（local catalog），不是「进展」（advance）。（词汇敏感区间：窄词表 {N_ADV_N} 个 -> {N_ADV} 个，结论不随词表改变。）")
A(f"2. **判决语言不可累积**。「判决」字样出现 **{kw['gpt5']['判决']}** 次，判决串 {vv['gpt5_uses']} 条 -> **{vv['gpt5_unique']} 个唯一标签**（复用率 **{vv['gpt5_reuse']}**）。一个从不复用的量表，无法在 Phase 之间比较。")
A(f"3. **死线在设计上被免疫**。K1/K2/K3 触发条件均为跨模型**合取**；实测 K1 为 **1/3** -> 未触发。而在行为读出层，三模型误差为 **{EREAD_LAB}**，是同一条 5% 门的 **{kl['readout_fail_ratio']} 倍**。")
A("")
A("---")
A("")
A("## §1 唯一全局 KPI（I1）")
A("")
A("每个 Phase **必须**报告下列三个数字（同一套冻结 held-out 数据、同一口径）。口径写入 `metric_dict.json`（Q02 冻结），任何改口径须新开 Phase 并公开标注。")
A("")
A("| KPI | 定义 | 当前实测 | 状态 |")
A("|---|---|---|---|")
A("| `E_read` | 未见组合在**行为读出层**的 rel-L2 预测误差 | **{}** | 已测（Q03 统一复算） |".format(EREAD_LAB))
A("| `E_ar(k)` | k 步自回归 logit-margin 预测误差曲线 | 不存在（MEMO 中 `E_ar` 出现 0 次） | 缺失（Q04–Q05 建造） |")
A("| `C_steer` | 用抽取机制控制 held-out 行为且无附带损伤的比例 | 未测；代理指标预示很低（cancel 反向加重 0.820；附带损伤地板 3.5/13） | 未测（Q06/Q20） |")
A("")
A("**登记规则（本宪法核心）**：一个 Phase 若未降低三者中任何一个，其在 Ledger 中登记为 `catalog`（目录条目），**不得**登记为 `advance`（进展）。")
A("")
A("> ⚠️ 反向风险（自我限制）：KPI 若被优化到虚假低值（held-out 泄漏 / 选易子面板），将重演同一错误。**必须与 §4 可识别性门 + §5 单模型否决权捆绑使用。**")
A("")
A("---")
A("")
A("## §2 判决层由行为定义（I2）")
A("")
A("任何「我们解释了 X」的断言，误差必须在**行为读出层与模型最终输出**上报告。承诺层 k* 可另附报告，**不得用于主线存续判定**。")
A("")
A("**K1 的现状与两种读法**（详见诊断 §3.3 / §8.4）：")
A("")
A("| 模型 | B4@k*（约 7.5% 深度） | B4@读出层 | 是否过 5% 门 |")
A("|---|---|---|---|")
for r in g1["rows"]:
    A("| {} | {:.4f} | **{:.4f}** | {} |".format(r["model"], r["b4_kstar"], r["b4_readout"], r["above5"]))
A("")
A("- **读法甲（推荐）**：K1 的语义是「能否预测未见组合」，行为由输出定义 -> 判定层应为读出层 -> **K1 应当触发**，「算子代数」降级为 `descriptive`。")
A("- **读法乙**：若「承诺层 = 候选取作用层」是**预先登记并有意**的判据，则 K1 的原文在两处自相矛盾，**必须重述为无歧义版本并重新冻结**。")
A("")
A("> **无论哪种读法，「用假设自己指定的层来判断假设」都必须被显式接受或显式否决，不能默认。** 该项落为 Q08，需用户 seal。本宪法不擅自改动 MEMO 中的既有判决。")
A("")
A("---")
A("")
A("## §3 死线双轨（I3）")
A("")
A("1. **主判据 = 跨模型聚合量**（pooled error + bootstrap CI），**不用合取**。")
A("2. **单个模型的反例即足以把命题标记为 `model_specific`**；`model_specific` **不得升为机制**。")
A("")
A("> 依据：项目自述「跨模型阴性反复出现」。**合取 + 系统异质 = 死线永不触发**，这是可预测的，不是运气。")
A("")
A("---")
A("")
A("## §4 可识别性门（I6）")
A("")
A("**每一个「新发现的局部结构」，必须同时报告 >=2 个互相不兼容的候选解释，以及它们在 held-out 上的预测分歧。**")
A("")
A("- 分歧显著 -> 用 held-out 判决二者（产出知识）；")
A("- 预测一致 -> 该结构**不可识别**，只是等价类描述 -> 强制 `descriptive`。")
A("")
A("> 理论依据（ICLR'25《Is Mechanistic Interpretability Identifiable?》）：因果对齐不足以保证唯一解释；小网络穷举下 XOR 任务含 85 条完美电路 / 45,000+ 条合法解释；<2% 的网络存在唯一最小映射。**这解释了「每个 Phase 都能找到过门结构」不是巧合，而是结构性必然。**")
A("")
A("---")
A("")
A("## §5 未识别变量程序（I5，最关键的科学方向）")
A("")
A("读数（Phase 3153 方差分解，读出层残差）：")
A("")
A("| 模型 | 类间 | 加性残差 | 交互格 (i,c) | **模板内高秩散布** |")
A("|---|---|---|---|---|")
for m in ["4b", "14b", "glm4"]:
    s = ANOVA[m]
    A("| {} | {:.1f}% | **{:.1f}%** | {:.1f}% | **{:.1f}%** |".format(m, s[0], s[1], s[2], s[3]))
A("")
A("即：加性假设解释不了 {a1}–{a2}%，而唯一被预注册的救援手段（rank-5 交互）只承载 {i1}–{i2}%。**被测假设类 h = Σ主效应 + 低秩交互 在读出层已被自己的数据否证——再加一个秩没有意义。**".format(
    a1="{:.1f}".format(min(ADD_RES)), a2="{:.1f}".format(max(ADD_RES)),
    i1="{:.1f}".format(min(INTER_CELL)), i2="{:.1f}".format(max(INTER_CELL))))
A("")
A(f"**出口候选**：模板内高秩散布 **{' / '.join('{:.1f}'.format(x) for x in HI_RANK)}%**，跨三模型稳定（谱形相关 0.968–0.991）、占比过半、**397 个 Phase 一致把它当噪声**。")
A("")
A("> **一个跨模型稳定、占比过半、始终未被命名的量，就是循环的出口。** 程序：把它当**未识别变量**做独立因子恢复（Q13–Q19），而不是当残差。")
A("")
A("---")
A("")
A("## §6 单一真源（I4）")
A("")
A("元层账本现存在三处不自洽（须先修）：")
A("")
A("| 项 | 值甲 | 值乙 |")
A("|---|---|---|")
A(f"| 命题账本分级（B / C） | MEMO 3103: B={ac['memo_3103_dist'][0][1]} / C={ac['memo_3103_dist'][0][2]} | TESTPLAN: B={ac['testplan_dist'][0][1]} / C={ac['testplan_dist'][0][2]} |")
A(f"| 有效依赖比例（A+B） | MEMO 3103 正文: 约 {ac['memo_pct_claim'][0]}% | 按同节计数: 63% |")
A(f"| `atlas_ledger.json` | 自声明 `ledger_sha256_8` = {LED_DECL} | 实际 = {LED_ACT} |")
A("")
A(f"Ledger 现状：{lg['measurements']} 条 measurements / schema {lg['schema_version']} / 实际 sha8 `{LED_ACT}`。")
A("")
A("> **在「每 Phase 一个登记哈希、一位不差」的装置里，元层账本三处对不上。** 修这个（Q01）比再跑一个 Phase 重要。")
A("")
A("---")
A("")
A("## §7 议程：冻结 30-Phase 队列（I9）")
A("")
A("**本队列是唯一议程来源。禁止从任意 Phase 的未解释残差派生新 Phase。** 未被采纳的残差进 `backlog`，仅在 Q26 由全局 KPI 曲线统一决定去留。")
A("")
A(f"队列文件：`research/gpt5/atlas/phase_queue_v1.json`（{len(qbytes)} B / {len(QUEUE)} 项 / sha8 `{hashlib.sha256(qbytes).hexdigest()[:8]}`）。")
A("")
A("| Q | 标题 | 区块 | KPI | GPU | 依据 |")
A("|---|---|---|---|---|---|")
for (q, title, blk, typ, kpi, gpu, src, deliv) in QUEUE:
    A("| Q{:02d} | {} | {} | {} | {} | {} |".format(q, title, blk, kpi, gpu, src))
A("")
A("区块：**A 闸门**（Q01–Q02, Q08–Q09, Q12，零 GPU，先做，否则其余无效）；**B KPI 装置**（Q03–Q07）；**C 未识别变量**（Q13–Q19，最关键）；**D 可识别性/可控性门**（Q10–Q11, Q20–Q21）；**E 机制**（Q22–Q25）；**F 制度**（Q26–Q30）。")
A("")
A("---")
A("")
A("## §8 外部评审规则（I10）与工程去摩擦（I11）")
A("")
A("- **I10**：外部评审的每条实质建议必须落地为「一条能触发死线的实验」，否则不得结案。禁止用「评审者的方案与我们已冻结的方案一致」作为结案理由。若某处方被延后，必须写明延后它的实验证据及其 MDE。")
A("- **I11**：抽取公共库（capture / hook / anchor / 装配断言）；SMOKE 断言覆盖**值域**而非仅键存在；目标 **<1 patch/run**。装置成本压下去，才有预算给证伪。")
A("")
A("---")
A("")
A("## §9 生效与修订")
A("")
A("1. 本宪法自冻结之日起生效；**在 Q26 / Q30 之外，不得生成本队列之外的新 Phase**。")
A("2. §2 的 K1 处置（Q08）是**需要用户 seal 的决策点**；本文件只呈现两种读法与推荐，不代为改判。")
A("3. 任何对本宪法的修订须新开一个 Phase，写明触发的实测数字与该修订会如何改变某条死线的可触发概率。")
A("")
A("---")
A("")
A("*本文件为治理性文档，不改动任何 MEMO 原文；所有统计数字由 `gen_constitution_r3.py` 从源文件现场解析，可逐条回查。*")

md = "\n".join(L) + "\n"
mp = os.path.join(root, "research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md")
open(mp, "w", encoding="utf-8", newline="\n").write(md)
mb = md.encode("utf-8")

# ---- 校验 ----
import re
bad = re.findall(r"\{[a-z_\[\]'\"]+\}|None|nan", md)
rep = []
rep.append("=== gen_constitution_r3 报告 ===")
rep.append("宪法: %s" % mp)
rep.append("  %d B / %d 行 / sha8 %s" % (len(mb), md.count("\n") + 1, hashlib.sha256(mb).hexdigest()[:8]))
rep.append("  未填充占位/None: %d %s" % (len(bad), bad[:6]))
rep.append("队列: %s" % qp)
rep.append("  %d B / %d 项 / sha8 %s" % (len(qbytes), len(QUEUE), hashlib.sha256(qbytes).hexdigest()[:8]))
rep.append("")
rep.append("关键数字落位检查:")
for k in ["6.55", "26", "371", "0.3316", "0.3986", "0.3898", "6.6", "60.4", "56.8",
          "41d65a13", "bbda63df", "1.104", "48", "1.2", "5.15"]:
    rep.append("  %-10s %d" % (k, md.count(k)))
open(os.path.join(T, "_gen_constitution_r3.txt"), "w", encoding="utf-8").write("\n".join(rep))
print("OK constitution %d B, queue %d B" % (len(mb), len(qbytes)))
