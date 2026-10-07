# -*- coding: utf-8 -*-
"""R3：压缩重写工作区 MEMORY.md（备份 + 校验 + 落盘复核）。"""
import hashlib
import os
import shutil

ROOT = r"D:\AI2050\Ai2050-OpenOne"
MEM = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
BAK = os.path.join(ROOT, "tests", "deepseek_temp", "_review", "_memory_backup_pre_compact_r3.md")
REP = os.path.join(ROOT, "tests", "deepseek_temp", "_review", "_memory_compact_r3.txt")

NEW = """# RDC/LPF 项目纪律（工作区长期记忆）
> 权威记录：`research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`（N 线）、`research\\gpt5\\docs\\AGI_GPT5_MEMO.md`（G 线）。本文件仅跨轮索引，细节一律回查 MEMO / 技能。

## 0 约定与落点
- deepseek 线只写 deepseek 备忘录（append-only、UTF-8+**BOM**+**CRLF**、`bare_lf 0`）。
- 产物 v2：脚本→`tests\\deepseek\\Phase{N}\\`；报告/seal/`verify_*`/memo·wlog 节源码→`tests\\deepseek_temp\\Phase{N}\\`；非 Phase→`_review\\`/`_infra\\`。**禁堆根下**。
- 收尾链（**十四次 P8–P21**）：探针→seal→amend→exec→SMOKE→正式→判决→Ledger→MEMO→基线→wlog→MEMORY→技能→独立复核→present。
- 标题 `## Phase {N}: 短标题（线-P）[hh:mm]`；纠错 append 不回改（**唯一例外：交付前生成件缺陷⇒逐字节回滚基线重跑链**）。
- Ledger n=**304**（P8–P21 各 1；**N 线 P3–P7 待补**）。基线 `post-append-phase21`=**497,406 B/4,774 行/21 标题**（`sha8 4ba5e22f`）。

## 1 装置铁律（26 条全文在 skill `rdc-main-axis-probe` 坑集）
最易违反：(a) 份额只用**精确可加向量预算**。(b) SMOKE 必做**必看数字**。(o) 同消息多 Edit 静默丢⇒Python 补丁+`assert count==1`+回读。(ad) 实现与 seal **逐字一致**。(ae) 数字**一律 result 现场渲染**。(ac) 插值读数不设单一硬门⇒分层。(af) **冻结锚重验**、如实报「陈旧」；**字典键不得截断生成**。
其余：patch 剔末层；`h_ℓ=hidden_states[ℓ+1]`；写入窗=相邻最大增量；`jump/max|effect|≥0.5`；剔实例 token 同剔类别 token；绝对/相对双剂量 `x*·r_ℓ`；固定基报 `overlap`；GQA 禁 `hidden/n_heads`；换口径=新自由度⇒校准臂；cos 与 rel-L2 双报；平均秩；paired `margin` 负=候选优。

## 2 N 线主线（P4→P21，关键数）
- **P4–P7** 主轴三段（嵌入=词典/层=开关/**权重绑定定 is-a 落点**）；单头 share_max **3.0%**；读位槽=**G−1=5 维**充分必要（随机 170–6000×）；**跨族近正交⇒无通用类别算子**；R1 K_d 降级。
- **P8–P11** 写入端**分布式**（向量预算 MLP 0.472/单头 0.074）；L6 内**无阈值增益**、非线性在其**之后**（S 形 `x*`≈0.6）⇒ **栈=软门**；深端塌陷主因**方向失配**。
- **P12–P16** 集中度 `rho(xhalf,depth)=−0.783`；**P16 预算否证 ⇒ P12/13/14 物理深度表述整体撤回**；P16 新域 `REACH={ℓ:ρ≥0.10}`。
- **P17** 逐层 `w_ℓ`+质心 `com_V` **26.15/26.70/26.68** ≫ median(REACH)；`MLP_DOMINANT` 0.740/0.975/0.824；`spearman(w,J)` −0.55/−0.79/−0.60。
- **P18** `share_mlp_beh` 0.66/0.96/0.75；`spearman(w,|b|)` **正**（与 P17 **反号⇒P17 P6 对象错配**）。
- **P19/P20** 向量侧跨精度 5/5；行为+剖面 9 预测 8 PASS；**P9 FAIL=域歧义**（as-coded 0.2645/0.0338 vs 容差 0.05，但容差只 ℓ≥6 标定 ⇒ 冻结域 0.0062/0.0037 皆 PASS），不改判。
- **P21** `share_v`+权重级 `W` 跨精度四臂 **7/9**；A0_bf16 **逐位复现 P8 锚**；**G1_core 四臂全 True** ⇒「分布式搬运」**非 nf4 kernel 产物**；Δ`share_v(mlp)` ≤**0.0234**、ρ≥**0.9920**。**P8 与 P16–P18 的精度缺口已闭合**。

## 3 挂账与限界
- R1/R2：10 条成立；P1 纠错 1（K_d）+降级 4+挂账 5。
- **限界**：① 否定臂基线不成立；② 留一只覆盖「未见实例」，「未见类别」仍崩（水果 0.04/0.05）；③ **激活级干预 ≠ 权重级证明**；④ 跨深度固定基衰减须扣基旋转；⑤ P13 确认集只 3 对；⑥ P14/P15 n≤18 无区分力；⑦ P15 家族×规模混杂；⑧ P17 `com_V`≠`com_layer`；⑨ P16 exec `bootstrap.seeds` 错记；⑩ P18 `b` 不可加；⑪ P19 只覆盖向量侧；⑫ **P20** 只两模型+offload、**不**升格 P16 `com_layer`；⑬ **P21** argmax 可在**并列带内**跨精度翻转（前三差<0.004）；**跨模型地板不可比**。
- **外部方案裁决**：三图谱骨架/统一实验卡/三集划分/竞争性解释/「预测未见现象」→ 全采纳；「每族一组特征+拼图还原+统一阈值+通用算子」→ 全改写。

## 4 G 线（gpt5）
3151 k3_only；3152 k1_not_triggered（k* 定律）；3153 fingerprint_consistent_coverage_partial；3105–3150：真值=记录级一阶矩广播、写入头组 L20–28 主写/L32 擦除；**判决符号=rev-3151b（负=优）**。**3154 已预注册**。**⚠ K1 判定层问题见 §9；元层账本三处不一致须先修。**

## 5 本机缺陷（Windows）
- bash shim 劣化：`ls`/`rm`/`tail` 坏、内联 `python -c` stdout 常丢、**反引号被吃** ⇒ Python 文件化+写 `.txt` 再 Read。
- **`Edit` 幻影**+IME 吞汉字 ⇒ 大段中文走 Python 补丁；调用 `cd <root> && .venv/Scripts/python.exe tests/...`。关键写入后 Grep 复核；GPU 逐模型防 OOM。
- **`Read` 视图可能陈旧**（大文件）⇒「改了没有」一律用 Grep/Python 判定。
- **P16–P21 逐 Phase 教训**：全文在 skill `rdc-phase-closeout` 教训 26–35（P21 新增：identity-probe 取模块权重 / argmax 并列内翻转 / 跨模型地板不可比 / 复核脚本 `sha8`-vs-64hex 错配 ⇒ 修脚本、不动产物）。

## 6 工作方式
「好的，继续」= AI 主导不停、深度自主续研；结构化输出；**关键发现重复 3 次**。

## 7 下一步（死线优先级）
- **Phase 22 = P8 `share_v` 与 P16/P17 逐层 `w_ℓ` 在同一精度下逐位对接**，消除跨 Phase 精度不确定性。
- **并列**=邻域 ±2 敏感性；P9 判据补域；P17 `P6` 的 MEMO 改判。
- **其他挂账**：N2h1-α-1 权重级；N2h1-β 水果类崩塌；N3-β→ε；R1 补强；K4；**P3–P7 补登 Ledger**。
- **⚠ 待用户裁定（§9）**：是否按 I2 正式改判 K1；是否采纳 I1（唯一全局 KPI）/ I9（30-Phase 固定队列）。

## 8 技能
`rdc-main-axis-probe`（15 臂+**63 坑**）、`rdc-phase-closeout`（**35 教训**/十四次链）、`rdc-dual-arm-phase-template`。

## 9 元层循环诊断（2026-10-02，R3 复核轮，非 Phase）
产物：`research/gpt5/docs/LOOP_DIAGNOSIS_AND_EXIT_v1.md` + `tests/deepseek_temp/_review/loop_diagnosis_r3.html`（数据 `loop_stats_r3.json`；生成器 `tests/deepseek/_review/gen_loop_diagnosis_r3.py`）。
- **规模**：gpt5 MEMO 1.90 MB / 14,901 行 / **402 Phase**（sha8 2a84776b）；deepseek 497 KB / 21 Phase。
- **判决语言失衡**：判决 **685** / 预注册 580 / 否证 86 / 降级 51 / 撤回 12；**判决串 47 次 → 48 个唯一标签（复用率 1.10）= 不可累积**；A 级命题 5/62（8%）；「接续」字段 **380** 处（模板写死「与总目标一致 ⇒ 自动进入下一 Phase」）。
- **核心改判主张**：`k1_not_triggered…operator_line_kept` **应改判**——K1 判据被挂在「候选声称作用层位 k*」（≈7.5% 深度）⇒ above5=**1/3** 未触发；但**行为读出层误差 0.3316/0.3986/0.3898 = 5% 门的 6.6×**。3153 读出层：交互格 **0.0–0.3%**、加性残差 34–38%、**模板内高秩散布 56.8–60.4%**（跨模型谱相关 0.968–0.991，402 Phase 一直当噪声）。
- **循环四机制**：M1 验收函数错位（奖励「找到过门局部结构」，高维里几乎恒真）；M2 死线免疫（K1/K2/K3 全是跨模型**合取**，而项目自述「跨模型从不一致」⇒ 永不触发）；M3 自催化议程（议程由残差自动派生，全局目标不投票；N 线 α-1…14 共 14 个连续 Phase 同一主题）；M4 外部证伪被吸收（两篇外部长文均判「无需范式修正」，三处方全延后）。
- **元层三处对不上（须先修）**：① 账本分级 **B=34**（MEMO 3103）vs **B=6**（TESTPLAN），C=14 vs 10；② 「约 55%（A+B）」vs 计数 39/62=**63%**；③ `atlas_ledger` 自声明 `ledger_sha256_8`=**41d65a13** vs 实际 **bbda63df**。
- **出路**：I1 唯一全局 KPI（`E_read` 现 0.3316/0.3986/0.3898、`E_ar(k)` 缺失、`C_steer` 未测）三数每 Phase 必报，未降低者标 `catalog`；I2 判决层由**行为**定义；I3 死线双轨+**单模型否决权**（`model_specific` 不得升为机制）；I5 把 57–60%「模板内散布」当**未识别变量**做因子恢复（循环出口候选）；I6 可识别性门（每发现须报 ≥2 个互不兼容解释 + held-out 分歧，据 ICLR'25）；I9 冻结 30-Phase 队列、删「残差自动派生下一 Phase」。
"""

old = open(MEM, "rb").read()
shutil.copy(MEM, BAK)
oldc = old.decode("utf-8")
newb = NEW.encode("utf-8")

log = []
log.append("backup -> %s (%d B, sha8 %s)" % (os.path.basename(BAK), len(old), hashlib.sha256(old).hexdigest()[:8]))

REQ = ["## 0 约定与落点", "## 1 装置铁律", "## 2 N 线主线", "## 3 挂账与限界", "## 4 G 线",
       "## 5 本机缺陷", "## 6 工作方式", "## 7 下一步", "## 8 技能", "## 9 元层循环诊断",
       "Ledger n=**304**", "share_v`+权重级 `W` 跨精度四臂 **7/9**", "**63 坑**", "**35 教训**",
       "4ba5e22f", "0.3316/0.3986/0.3898", "复用率 1.10", "41d65a13"]
newt = newb.decode("utf-8")
miss = [k for k in REQ if k not in newt]
assert not miss, "missing required anchors: %r" % miss
assert newb[:3] != b"\xef\xbb\xbf", "must not add BOM"
assert b"\r\n" not in newb, "must stay LF-only"

with open(MEM, "wb") as f:
    f.write(newb)

after = open(MEM, "rb").read()
assert after == newb, "write verify failed"
at = after.decode("utf-8")
log.append("MEMORY compacted: %d B / %d chars -> %d B / %d chars (sha8 %s)"
           % (len(old), len(oldc), len(after), len(at), hashlib.sha256(after).hexdigest()[:8]))
log.append("all %d required anchors present; LF-only, no BOM" % len(REQ))
log.append("old markers gone: " + str(all(x not in at for x in ["P21 教训", "## 5 本机缺陷（Windows）\n- bash"])))

open(REP, "w", encoding="utf-8").write("\n".join(log))
print("\n".join(log))
