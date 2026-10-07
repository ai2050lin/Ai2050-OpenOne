# -*- coding: utf-8 -*-
"""R5d: MEMORY.md 压缩重写（全量替换，最稳）。"""
import os, hashlib

P = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md"
REP = r"D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_review\_patch_r5d_report.txt"

raw = open(P, "rb").read()
has_bom = raw.startswith(b"\xef\xbb\xbf")
body = raw.decode("utf-8-sig")
crlf = "\r\n" in body
old_chars = len(body)
old_sha8 = hashlib.sha256(raw).hexdigest()[:8]

L = []
A = L.append

A("# RDC/LPF 项目纪律（工作区长期记忆）")
A("> 权威记录：`research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`（N 线）、`research\\gpt5\\docs\\AGI_GPT5_MEMO.md`（G 线）。本文件仅跨轮索引，细节一律回查 MEMO / 技能。")
A("")
A("## 0 约定与落点")
A("- deepseek 线只写 deepseek 备忘录（append-only、UTF-8+**BOM**+**CRLF**、`bare_lf 0`）。")
A("- 产物 v2：脚本→`tests\\deepseek\\Phase{N}\\`；报告/seal/`verify_*`/memo·wlog 节源码→`tests\\deepseek_temp\\Phase{N}\\`；非 Phase→`_review\\`/`_infra\\`。**禁堆根下**。")
A("- 收尾链（十四次 P8–P21）：探针→seal→amend→exec→SMOKE→正式→判决→Ledger→MEMO→基线→wlog→MEMORY→技能→独立复核→present。")
A("- 标题 `## Phase {N}: 短标题（线-P）[hh:mm]`；纠错 append 不回改（**唯一例外：交付前生成件缺陷⇒逐字节回滚基线重跑链**）。")
A("- Ledger `atlas_ledger.json` n=**304** / sha8 **`bbda63df`**（P8–P21 各 1；**N 线 P3–P7 待补**）。基线 `post-append-phase21`=497,406 B/4,774 行（`4ba5e22f`）。")
A("- 工作方式：「好的，继续」= AI 主导不停、深度自主续研；结构化输出；**关键发现重复 3 次**。")
A("")
A("## 1 装置铁律（26 条全文在 skill `rdc-main-axis-probe` 坑集，现 **63 坑**）")
A("最易违反：(a) 份额只用**精确可加向量预算**。(b) SMOKE 必做**必看数字**。(o) 同消息多 Edit 静默丢⇒Python 补丁+`assert count==1`+回读。(ad) 实现与 seal **逐字一致**。(ae) 数字**一律 result 现场渲染**。(ac) 插值读数不设单一硬门⇒分层。(af) **冻结锚重验**、如实报「陈旧」；字典键不得截断生成。")
A("其余：patch 剔末层；`h_ℓ=hidden_states[ℓ+1]`；写入窗=相邻最大增量；`jump/max|effect|≥0.5`；剔实例 token 同剔类别 token；绝对/相对双剂量 `x*·r_ℓ`；固定基报 `overlap`；换口径=新自由度⇒校准臂；cos 与 rel-L2 双报；paired `margin` 负=候选优。")
A("")
A("## 2 N 线主线（P4→P21，关键数）")
A("- **P4–P7** 主轴三段（嵌入=词典/层=开关/**权重绑定定 is-a 落点**）；单头 share_max **3.0%**；读位槽=**G−1=5 维**充分必要；**跨族近正交⇒无通用类别算子**；R1 K_d 降级。")
A("- **P8–P11** 写入端**分布式**（向量预算 MLP 0.472/单头 0.074）；L6 内**无阈值增益**、非线性在其**之后**（S 形 `x*`≈0.6）⇒ **栈=软门**；深端塌陷主因**方向失配**。")
A("- **P12–P16** `rho(xhalf,depth)=−0.783`；**P16 预算否证⇒P12/13/14 深度表述撤回**；新域 `REACH={ℓ:ρ≥0.10}`。")
A("- **P17** `w_ℓ`+`com_V` **26.15/26.70/26.68** ≫ median(REACH)；`MLP_DOMINANT` 0.740/0.975/0.824；`spearman(w,J)` −0.55/−0.79/−0.60。")
A("- **P18** `share_mlp_beh` 0.66/0.96/0.75；`spearman(w,|b|)` **正**（与 P17 **反号⇒P17 P6 对象错配**）。")
A("- **P19–P21** 跨精度：向量侧 5/5；行为+剖面 8/9（P9 FAIL=域歧义，不改判）；`share_v`+权重级 `W` **7/9**，A0_bf16 **逐位复现 P8 锚**、**G1_core 四臂全 True**⇒「分布式搬运」非 nf4 kernel 产物；Δ`share_v(mlp)`≤**0.0234**、ρ≥**0.9920**。**P8 与 P16–P18 精度缺口已闭合**。")
A("")
A("## 3 挂账与限界")
A("- R1/R2：10 条成立；P1 纠错 1（K_d）+降级 4+挂账 5。")
A("- **限界（13 条）**：核心三条——① 否定臂基线不成立；② 留一只覆盖「未见实例」，「未见类别」仍崩（水果 0.04/0.05）；③ **激活级干预 ≠ 权重级证明**。④–⑬（基旋转扣除／P13 只 3 对／P14-15 n≤18／P15 家族×规模混杂／`com_V`≠`com_layer`／P16 exec 错记／P18 `b` 不可加／P19 只向量侧／P20 不升格 `com_layer`／P21 argmax 并列带内翻转+地板不可比）见 MEMO 与技能教训 26–38。")
A("- **外部方案裁决**：「骨架/实验卡/三集/竞争性解释/预测未见现象」全采纳；「每族一组特征+拼图还原+统一阈值+通用算子」全改写。")
A("")
A("## 4 G 线（gpt5）")
A("3151 k3_only；3152 k1_not_triggered（k* 定律）；3153 fingerprint_consistent_coverage_partial；3105–3150：真值=记录级一阶矩广播、写入头组 L20–28 主写/L32 擦除；**判决符号=rev-3151b（负=优）**；**3154 已预注册**；**⚠ K1 判定层问题见 §9**。")
A("")
A("## 5 本机缺陷（Windows）")
A("- bash shim 劣化：`ls`/`rm`/`tail`/`dirname`/`cd` 坏、内联 `python -c` stdout 常丢、**反引号被吃** ⇒ Python 文件化+写 `.txt` 再 Read。")
A("- **`Edit` 幻影**+IME 吞汉字 ⇒ 大段中文走 Python 补丁（**禁凭记忆写匹配串，先 dump 真实磁盘**）；调用 `cd <root> && .venv/Scripts/python.exe tests/...`。关键写入后 Grep 复核；GPU 逐模型防 OOM。")
A("- **`Read` 视图可能陈旧**（大文件）⇒「改了没有」一律用 Grep/Python 判定。")
A("- **逐 Phase 教训**在 skill `rdc-phase-closeout` 教训 26–38（P21：identity-probe 取权重 / argmax 并列内翻转 / 跨模型地板不可比 / 复核脚本 `sha8`-vs-64hex 错配⇒修脚本不动产物）。**Q09/Q12 新增**：合取式死线 ⇒ 两条从未被测量；`R\\d\\d` 非唯一命名空间 ⇒ 纯 id grep 必假阳性（改走文本指纹）。")
A("")
A("## 6 G 线外挂账与下一步（死线优先级）")
A("- **Phase 22** = P8 `share_v` 与 P16/P17 逐层 `w_ℓ` 在**同一精度**下逐位对接。并列：邻域 ±2 敏感性；P9 判据补域；P17 `P6` 的 MEMO 改判。")
A("- **其他挂账**：N2h1-α-1 权重级；N2h1-β 水果类崩塌；N3-β→ε；R1 补强；K4；**P3–P7 补登 Ledger**。")
A("- **A 闸门进度**：**Q01/Q02/Q09/Q12 已完成**（§9，零 GPU）；Q03–Q07 为 KPI 装置（需 GPU）；Q08/Q10/Q11 待 seal 或需 GPU。")
A("- **⚠ 待用户 seal**：Q01 更正表 **C1–C6**；Q08（K1 改判，两种读法已备）；I1/I9。")
A("")
A("## 7 技能")
A("`rdc-main-axis-probe`（15 臂 + **63 坑**）、`rdc-phase-closeout`（**38 教训**/十四次链）、`rdc-dual-arm-phase-template`。")
A("")
A("## 8 元层循环诊断与 A 闸门（2026-10-02/03，非 Phase）")
A("产物：`LOOP_DIAGNOSIS_AND_EXIT_v1.md`（`013d08a9`）+ `RDC_RESEARCH_CONSTITUTION_v1.md`（`01df6398`）+ `atlas/phase_queue_v1.json`（`675836fd`，Q01–Q30）；数据 `loop_stats_r3.json`；页 r3/r4/r5 三个 HTML（`q09q12_gate_r5.html` 最新）。")
A("- **诊断核心**：gpt5 MEMO 1.90 MB/14,901 行/**402 Phase**；判决 **685** 次但**唯一标签 48（复用率 1.10）⇒ 不可累积**；A 级 5/62(8%)；「接续」**380** 处。**改判主张**：`k1_not_triggered…operator_line_kept` **应改判**——判据挂 k*（≈7.5%）⇒ above5=**1/3** 未触发；但**行为读出层 0.3316/0.3986/0.3898 = 5% 门的 6.6×**；3153 读出层（交互格 0.0–0.3%、加性残差 34–38%、模板内高秩散布 56.8–60.4%）见诊断。I1–I11 见宪法。")
A("- **教训**：**397 Phase 仅 26（6.55%）报告共享指标变化**（窄词表 18）⇒ 93.45% 是「目录条目」。")
A("- **Q01 单一真源**：五处不自洽全定位（原 3+新 2：MEMO 记账本哈希 `add57ba7`≠实际 `9a3c6ff4`；`measurements` 两套 schema、`evidence_level` **304/304 恒定**）。真源 62 条按**分量**复现 A5/B34/C14/D21/E20（和 94）；**TESTPLAN B6/C10 五口径全不可复现**；「55%/63%」= **量纲混淆**（同量纲唯一值 41.5%=39/94）。哈希自指结构性失效。产物 `META_SINGLE_SOURCE_Q01.md`。**C1–C6 待 seal**。")
A("- **Q02 KPI 冻结**：`metric_dict.json` v1→v2（`03887e51`）；E_read=`b4_rel_readout_mean3seed` **0.33162/0.39860/0.38984（0/3 过门）**；E_ar/C_steer 冻结未测；兼容 P3150 断言。")
A("- **Q09 死线双轨（I3）**：K1/K2/K3 触发**全为全称量词合取** ⇒ **K2/K3 从未被测量**（`phase3154` 不存在；无 top-50 覆盖率量）。K1 双轨：**k* `model_specific`**（轨 A 通过、qwen3-4b 否决）、**读出层 `fired_all_models`**（池化 margin **+0.9409** > −2·MDE 0.1933、E_read 池化 **0.3734**、3/3 否决）⇒ 与旧「未触发、算子代数线保住」**相反**。层位属 Q08。产物 `DEADLINE_DUAL_TRACK_Q09.md`。")
A("- **Q12 D/E 引用审计（I4）**：**新推理链（MEMO@3104+）id 命中仅 1 处**（Phase 3113→R55 复合等级）+ **文本指纹 0 命中** ⇒ **无实质违规**。两制度缺陷：**RULE-UNDECIDABLE**（D∪E **38/62=61.29%**，**26 复合**、纯 E 仅 3 ⇒「E 禁入」不可机械判定）、**NS-COLLISION**（`R\\d\\d` 非唯一命名空间，纯 id grep 必假阳性）。产物 `PROP_CITATION_AUDIT_Q12.md`。复核 **76/0**。")

new = "\n".join(L)

ANCHORS = [
    "AGI_DEEPSEEK_MEMO.md", "AGI_GPT5_MEMO.md", "atlas_ledger.json", "bbda63df",
    "rdc-main-axis-probe", "rdc-phase-closeout", "rdc-dual-arm-phase-template",
    "LOOP_DIAGNOSIS_AND_EXIT_v1.md", "RDC_RESEARCH_CONSTITUTION_v1.md", "phase_queue_v1.json",
    "DEADLINE_DUAL_TRACK_Q09.md", "PROP_CITATION_AUDIT_Q12.md", "META_SINGLE_SOURCE_Q01.md",
    "metric_dict.json", "model_specific", "fired_all_models",
    "RULE-UNDECIDABLE", "NS-COLLISION", "4ba5e22f", "9a3c6ff4",
]
miss = [a for a in ANCHORS if a not in new]
assert not miss, "ANCHORS MISSING: %r" % miss

TARGET = 5050
assert len(new) < TARGET, "MEMORY 仍过长: %d (target<%d)" % (len(new), TARGET)

nl = "\r\n" if crlf else "\n"
payload = new.replace("\n", nl).encode("utf-8")
if has_bom:
    payload = b"\xef\xbb\xbf" + payload
open(P, "wb").write(payload)

back = open(P, "rb").read()
btxt = back.decode("utf-8-sig")
back_sha8 = hashlib.sha256(back).hexdigest()[:8]

rep = []
rep.append("PATH=%s" % P)
rep.append("BOM=%s CRLF=%s" % (has_bom, crlf))
rep.append("OLD chars=%d sha8=%s" % (old_chars, old_sha8))
rep.append("NEW chars=%d bytes=%d lines=%d sha8=%s" % (len(btxt), len(back), btxt.count("\n") + 1, back_sha8))
rep.append("ANCHORS ok=%d/%d" % (len(ANCHORS), len(ANCHORS)))
rep.append("readback_equal=%s" % (btxt == new))
open(REP, "w", encoding="utf-8").write("\n".join(rep) + "\n")
print("\n".join(rep))
