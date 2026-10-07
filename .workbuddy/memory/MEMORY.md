# RDC/LPF 项目纪律（工作区长期记忆）
> 权威记录 `research\deepseek\docs\AGI_DEEPSEEK_MEMO.md`（**唯一研究日志**，Phase 1–39）。本文件仅跨轮索引，细节回查 MEMO / 技能。

## 0 约定（v4，2026-10-03，**仅本对话遵守**）
- 日志唯一落点 = `AGI_DEEPSEEK_MEMO.md`（append-only、BOM+CRLF、标题 `## Phase {N}: 短标题 [hh:mm]`）；**不新建其他 `.md`**。`AGI_GPT5_MEMO.md` 属他线，**不读不改不追加**。**另有并发写者**（P36 非本对话）⇒ 只 append 不碰前文；文件 free bare_lf = **0**（他线于 R11 期间把 P36 裸 LF 规范化；R11 前为 45）。
- 三桶：测试脚本 `tests\deepseek\`；临时脚本 `tests\deepseek_temp\`；测试结果 `tests\deepseek\result\`。历史 `Phase1..21\` 不动。
- MEMO：P1–21 N 线；P22 = A 闸门 R3–R5b；P23–30 = 8 件并入；P31 = R6 路由；P32–34 = 归属整理；**P35 = A 闸门 seal（R8）**；**P36 = E4/E4b（他线）**；**P37 = Q03 E_read 基线（R9）**；**P38 = Q04 E_ar(k) 装置（R10）**；**P39 = Q05 E_ar(k) 正式测量（R11）**。现 `d04d2f09`（718342 B）。
- Ledger `atlas_ledger.json` n=**304** `bbda63df`（**N 线 P3–P7 待补**）。「好的，继续」= AI 主导续研；**关键发现重复 3 次**。

## 1 装置铁律（63 坑全文见 skill `rdc-main-axis-probe`）
(a) 份额只用**精确可加向量预算**；(b) SMOKE 必做**必看数字**；(o) 多 Edit 静默丢 ⇒ Python 补丁 + `assert count==1` + 回读；(ad) 实现与 seal **逐字一致**；(ae) 数字**一律 result 现场渲染**；(af) **冻结锚重验**。
其余：patch 剔末层；写入窗=相邻最大增量；`jump/max|effect|≥0.5`；双剂量 `x*·r_ℓ`；固定基报 `overlap`；paired `margin` 负=候选优。

## 2 N 线主线（P4→P21）
- P4–P7 主轴三段（嵌入=词典/层=开关/**权重绑定定 is-a 落点**）；单头 share_max **3.0%**；读位槽 **G−1=5 维**；跨族近正交 ⇒ 无通用类别算子。
- P8–P11 写入端**分布式**（向量预算 MLP 0.472）；L6 内**无阈值增益** ⇒ **栈=软门**；深端塌陷主因**方向失配**。
- P12–P16 `rho(xhalf,depth)=−0.783`；**P16 否证 ⇒ P12/13/14 深度表述撤回**；域 `REACH={ℓ:ρ≥0.10}`。
- P17–P21 `w_ℓ`+`com_V` ≈26 ≫ median(REACH)；`spearman(w,|b|)` 与 P17 **反号 ⇒ P17 P6 对象错配**；跨精度 7/9、A0_bf16 **逐位复现 P8 锚**。

## 3 挂账与限界
- **核心三条限界**：① 否定臂基线不成立；② 「未见类别」仍崩（水果 0.04/0.05）；③ **激活级干预 ≠ 权重级证明**（其余见 MEMO / 技能 26–43）。
- R1/R2：10 条成立；P1 纠错 1+降级 4+挂账 5。G 线索引：3151 k3_only / 3152 k1_not_triggered；判决符号 rev-3151b（负=优）。

## 4 本机缺陷（Windows）
- bash shim `ls/rm/tail/dirname/cd` 坏、内联 `python -c` stdout 常丢、**反引号被吃** ⇒ Python 文件化 + 写 `.txt` 再 Read。
- **`Edit` 幻影（R9 复现 2 次）**、脚本日志亦可能幻影 ⇒ 中文大段走 Python 补丁（**禁凭记忆写匹配串，先 dump 真实磁盘**）；**落盘证据只能靠独立进程 re-hash**。`Read` 大文件视图陈旧 ⇒ 改没改用 Grep/Python。
- **`%` 格式化陷阱**：`% (...)` 行里字面 `%`（如 `5% 门`）须写 `%%`；**无格式化**的行写 `%%` 会原样输出。两者都要查。
- **`AGI_DEEPSEEK_MEMO.md` 未被 git 跟踪** ⇒ 改动前落快照；**外部 +2 B 写入**已两次（R6→R7、R8→P36，非本线）。

## 5 下一步 / 死线
- **✅ A 闸门已关闭（R8）**：seal `Q08=甲 | C=全接受 | I1=确认 | I9=确认`。K1 判定层=**行为读出层** ⇒ **K1 触发**（3/3 否决、池化 **0.3734** = 门 7.5×）；k* `model_specific`。「条件齿轮组=算子代数」**降级 descriptive**。C1–C6 全接受（C4/C6 为跨线账本 ⇒ 只落补丁 `ledger_corrections_v1.json`，未施加）。产物 `24c60160`；复核 38/0。
- **✅ Q03 E_read 基线锁定（R9）**：三模型 0.331615/0.398601/0.389835，池化 **0.373350**，**5% 门 0/3**（最小 6.63×）；三份 `collect.npz` 锚**逐位复现 drift 0.00e+00**；三模型**同一 held-out 指纹**；队列 6/30 sealed `d0d2e208`；复核 42/0 + 28/0。
- **✅ Q04 `E_ar(k)` 装置建成（R10）**：design_sha `33ddf69d`（幂等）／res_sha8 `04ad1af3`（三跑恒等）；SMOKE qwen3-4b（180 cells / 900 fwd）**装置门 4/4**；独立复核 **14/0**。口径登记 `metric_dict` **v2→v3**（content `5ce974ee`→`72a3d2c4`），E_ar.status=`device_built`。队列 `d0d2e208`→`631c1ba3`（Q04=`device_built`，sealed_items 仍 6）。**E_ar 与 E_read 量纲不同**（logit L1 vs 归一化 MSE）⇒ 只可并排读、以 `rel` 作桥。
- **✅ Q05 `E_ar(k)` 正式测量完成（R11）**：四臂 738×K16；精度策略=4b **bf16**、14B/9B **4-bit NF4**（bf16 offload 因 RAM 不足不可行）；**D4 桥** max|Δrel|=0.0489（门 0.05）**PASS**；形状 4b=flat/14b=saturating/9b=flat；S_rel min=0.6858/0.6939/0.7392；独立复核 TOTAL  PASS=47  FAIL=0。口径登记 `metric_dict` v3→v4（E_ar.status=**measured**）；队列 Q05 sealed `1acb1e78`。**下一步 = Q06 `C_steer` 基座测量**（承重轴 + 端口替换：steered 成功率 + 附带损伤）。挂账不变：N 线 P3–P7 补 Ledger；跨线账本补丁施加确认；N2h1-α-1 权重级；水果类；K4。
- **✅ Q06 `C_steer` 基座测量完成（Phase 40，2026-10-07）**：v1 承重轴（qwen3-4b L29 WR 主 PC 同构移植）x 端口替换，441 held-out cells x 10 配置 **C_steer=0.0000**（Wilson 上界 1.0%）；rand 同 0；collateral 干净（frac0 0.933）；identity 逐位恒等；复核 16/0；res `5f88ed7e`；队列 sealed；Ledger n=306 catalog。结论：承重轴=生成稳定性轴，非类身份杠杆（破坏易/定向控制难）。
- **下一步 = Q07 KPI 曲线 v0 汇总**（zero GPU）；Q17/Q24 复用 Q06 装置；Q20 扩 target 族。挂账不变：N 线 P3–P7 补 Ledger；跨线账本补丁施加确认；N2h1-α-1 权重级；水果类；K4。

## 6 元层诊断与 A 闸门（MEMO Phase 22–34）
- **诊断**：gpt5 MEMO 402 Phase/1.90 MB；判决 685 次但**唯一标签 48（复用率 1.10）⇒ 不可累积**；397 Phase 仅 26（6.55%）报告共享指标变化。
- **K1 双轨（Q09）**：k* `model_specific`；读出层 `fired_all_models`（margin +0.9409）。**K2/K3 从未被测量**。
- **Q01/Q02**：真源 62 条按**分量**复现 A5/B34/C14/D21/E20（和 94）；「55%/63%」= **量纲混淆**（41.5%=39/94）；`metric_dict` v2 `03887e51`。
- **Q12**：id 命中 1 + 指纹 0 ⇒ 无实质违规；缺陷 **RULE-UNDECIDABLE**、**NS-COLLISION**。
- **归档（R6+R7）**：11 件 `.md` → MEMO P23–34（备份 `_archive_r6|_r7\gpt5_docs\`）；`metric_dict/phase_queue` → `research\deepseek\atlas\`；复核 44/0、44/0。
- **归属判据（R7）**：**谁的备忘录登记它，就是谁的线**。gpt5 余 20 件判为他线未动；`atlas_ledger.json` 跨线共享不动。

## 7 技能
`rdc-main-axis-probe`（15 臂 + 63 坑）、`rdc-phase-closeout`（**46 教训**）、`rdc-dual-arm-phase-template`。

## G 线（AGI_GPT5_MEMO）3154 状态（2026-10-01）
- **G1-P4 多因素混杂分解闭环**：fingerprint_consistent（fpmin 0.996，KOUT/KSTAR 双层 3 对全≥0.8）；KOUT 份额 [L 5.6, S 19.4, C 57.6, G 1.16, D 0.31, R 15.9]%；**3153 模板内散布 57–60% → 15–17%**；逻辑轴 p<0.0001×3、held-out 新主题 1.00/1.00/0.94；G argmax 中层 k18/k22/k15；ledger n=**305**。权威日志=`research\gpt5\docs\AGI_GPT5_MEMO.md`（本文件 deepseek 节未动）。下一步 3155=**G2-P1 多关系族 K2 死线**（φ_ℓ(c) 与 W_ℓ 可分离性 + held-out 关系泛化门）。
