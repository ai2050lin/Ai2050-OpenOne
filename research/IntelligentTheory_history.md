

## 七，研究阶段历史记录

### 项目全局状态总览（2026-07-15）

| 指标 | GLM5路线 | GPT5路线 |
|------|---------|---------|
| 总阶段数 | ~944 | ~429 |
| MainAnalysis 文件 | 69 | - |
| 最新日期 | 2026-07-05 | 2026-07-14 |
| 核心模型 | Qwen3, GLM4, DS7B | Qwen3, GLM4, DS7B |
| 当前状态 | 语义方向→MLP通道齿轮→边界移动桥接 | 行为资格门+双路线行为合同 |
| 跨模型闭合 | 未完整报告 | 0/72 |
| 精确守恒验证 | 1248/1248 | 1248/1248 |
| 关系签名可读/可搬 | - | 27/27 vs 0/9 |
| 最强正结果 | qwen3 MLP通道齿轮 gap+3.70 | 行为资格门四分离 |
| strict-clean 自然生成 | 0 | 0 |

关键数字总结：
- 已排除候选对象：单个神经元、单个注意力头、单个通道、固定语义方向、组件范数、总层更新、单条动态路线、单个查询位置状态、局部关系签名、宽事件区间、峰值层流水线、完整状态续跑、线性方向注入、静态向量闭合、角色主导假设
- 正面发现：计算图可守恒性、内容状态可搬运、关系签名稳定可读、多位置动态事件链、MLP通道齿轮因果桥接、EOS-vs-a边界齿轮
- 核心瓶颈：跨模型闭合=0、自然门控缺失、strict-clean自然生成=0、行为稳定性脆弱

全项目最重要的科学结论：
语言编码机制不是单个概念向量、单个神经元或单条路径，
而是"因果状态等价类 + 条件化状态转移算子 + 动态路由 + 共享终端接口"组成的系统。
破解顺序：因果状态发现 → 算子识别 → 组合规律 → 全局功能图谱 → 物理实现映射。

### GLM路线
		阶段一：基础方向分解与操作符机制（Phase 301-312）
			核心任务：
				分解 identity、role、frame、construction、operator、scope、norm、position。
			重要进展：
				1，早期 R/F 分解发现伪影，后来改为 full factorial 与 construction/position/norm 分解。
				2，DS7B 出现高范数共享主方向 + 微小差分读取，Qwen3/GLM4 更接近正交子空间。
				3，否定 O(not) 有稳定因果效应，但与反义词替换不同。
				4，O/R/C/S 子空间不能只看 cosine，必须看输出映射和单位因果效力。
			核心成果：
				提出两类编码架构：正交子空间编码 vs 共享主方向 + 差分读取。

		阶段二：关系网络、上下文激活与否定修正（Phase 313-321）
			核心任务：
				分析 W_U 读出、关系保持、属性/功能/否定关系、随机基线和模板控制。
			重要进展：
				1，W_U 本身不解释所有差分放大，中间层存在复杂放大。
				2，模型保留人类语义关系网络，但属性/功能关系需要上下文或模板支持。
				3，Phase 317 修正早期过强结论：属性上下文激活只有 modest 效应。
				4，否定是语义极性偏移，不是严格逻辑取反。
			核心成果：
				语言关系不是静态表，而是条件激活的相对关系网络，但条件门控弱于最初预期。

		阶段三：绑定机制、层归因与 MLP 主导路径（Phase 322-365）
			核心任务：
				研究对象-属性绑定、slot、temperature、interaction、层归因、attention vs MLP。
			重要进展：
				1，binding 由 transformer 层形成，embedding 几乎不贡献。
				2，层归因给出数学精确分解，输出替换法被证明过粗。
				3，MLP 在多个模型中承担 binding 的主要计算，但 attention 也参与路由和中介。
				4，interaction term、value prior、context-gated binding 被反复校正。
			核心成果：
				绑定不是单点存储，而是跨层计算产生的对象-关系-值兼容性梯度。

		阶段四：范数、类别中心、连续属性与规则调制（Phase 366-414）
			核心任务：
				分析范数掩蔽、RMSNorm、类别中心、连续属性、规则强度、候选词读出。
			重要进展：
				1，范数可以掩蔽方向，RMSNorm 会重映射信息。
				2，纯 category centroid 太粗，cross-fit 和 damage_ratio 成为判断方向干净程度的关键。
				3，连续属性表现为静态知识锚定 + 规则调制 + 候选词读出。
				4，规则可以改写几何，但存在非对称反转。
				5，W_U 范数不能解释非对称反转，根源更可能在内部锚定。
			核心成果：
				提出“对象知识锚定 × 上下文调制 × 候选词读出”的连续属性框架。

		阶段五：自然运输、关系槽位、类别边界和读出接口（Phase 415-499）
			核心任务：
				研究虚构对象、自然运输方向、属性中介、类别边界、shared/private 通道、RMSNorm 读出接口。
			重要进展：
				1，属性与类别之间有中介关系，但不是全部由类别解释。
				2，关系槽位会私有化绑定态，对象解锁门控影响 shared/private 分离。
				3，类别边界是跨层累积，不是单层写入。
				4，存在支持层、反对层、共享语义抑制、末层释放。
				5，final RMSNorm 和 gain 维度可导致符号翻转和候选重排。
			核心成果：
				类别边界公式被修正为：投影写入 - 正交抑制 + 末层读出。

		阶段六：正交语义场、轨迹价值与生成策略门（Phase 500-567）
			核心任务：
				统一 GLM5/GPT5 路线，研究 orthogonal semantic field、trajectory、path value、generation closure、paraphrase gate、scaffold route。
			重要进展：
				1，语义支持可以存在于读出正交空间，最终通过接口转移到可读出方向。
				2，trajectory/path value 可以影响生成，但受策略门、格式、脚手架强烈控制。
				3，label-forbidden paraphrase 暴露出语义恢复与输出策略分离。
				4，route restore、wrong donor、object binding、prototype generation 显示生成不是简单语义读出。
			核心成果：
				提出“语义状态”和“生成策略门”分离：知道答案不等于能按要求生成答案。

		阶段七：Norm gate、路径瓶颈与格式/echo 机制（Phase 568-574）
			核心任务：
				追踪 pre-layer source、final norm gate、norm weight 维度、path bottleneck、format gate、echo suppression。
			重要进展：
				1，关键语义状态在 L20-L22 左右形成，L28 更像读出接口。
				2，final norm weight 而非 RMS 归一化本身，可造成 margin 符号翻转。
				3，少量 norm weight 维度可主导格式门。
				4，echo suppression 只能减少 echo，不必然提升 clean semantic output。
			核心成果：
				读出瓶颈被拆成：语义状态、格式门、echo 路径、norm gate、clean output policy。

		阶段八：闭合语义微世界与检索-推理闭包（Phase 575-583）
			核心任务：
				构造闭合微世界，隔离 object-category、object-relation-value、规则检索、两跳组合、yes/no 极性读出。
			重要进展：
				1，对象和类别状态可线性解码，但对象/类别子空间不正交，简单类别交换没有输出效果。
				2，三模型存在规则检索注意力头，category-copy 头更具因果性。
				3，单步检索可达高准确，但组合推理存在中间类别错误、关系使用不稳、yes-bias 等问题。
				4，Phase 581 修正 Phase 580 的 prompt bug，组合推理不是 0%，但仍有 gap。
				5，GLM4/DS7B 在参数化 yes/no 负例上有强 yes-bias。
			核心成果：
				明确检索不等于推理；组合瓶颈主要在中间状态选择和极性读出，而不是简单缺少检索。

		阶段九：状态门修复、值候选竞争与条件化变换图谱（Phase 584-594）
			核心任务：
				研究固定网络中是否能通过 state gate、hidden patch、distributed relation-filter、value winner competition、ranking atlas 修复错误。
			重要进展：
				1，polarity-format gate 出现 hidden causal repair，但 relation-filter/value gate 修复困难。
				2，value winner 竞争不是“有正确候选即可”，关键是 correct-specific 是否压过 old top-wrong。
				3，Phase 592 建立 relation-specific ranking factor atlas，但只能标为 projection evidence。
				4，Phase 593 证明 atlas projection node 不能直接通过单点 residual patch 修复 winner。
				5，Phase 594 发现 candidate-specific ranking 可在 layer-to-layer transition 中增强，DS7B rule_value L26 的 MLP update 是当前最强候选生成点。
			核心成果：
				最新理论从“静态方向 patch”升级为“条件化状态变换图谱”：正确值排序更像某些层/位置/组件根据上下文生成的 update，而不是可直接移植的向量。

		阶段十：读出位置纠错、源贡献通道图谱与短语竞争闭环（Phase 595-707）
			核心任务：
				把 value gate 修复失败的问题继续拆开，依次定位：
					answer-start 读出位置；
					target_value / answer_line / self_last 等源词元贡献；
					L23-L27 附近 attention heads；
					source-restricted positive channels；
					完整 value phrase / donor phrase / prose phrase 竞争。
			重要进展：
				1，Phase 604-620 纠正了 prompt_last 与 answer_last 混用的问题，确认正确值读出更接近 answer-start / answer-side path。
				2，Phase 623-631 说明 result-only、format/protocol、short-answer scaffold 等路线状态可以显著改变 value/prose 竞争，但不能直接等同于语义身份代码。
				3，Phase 636-649 将 DS7B 的 token0 prefix / format protocol / value_short_answer 路线定位到更清晰的协议轨迹。
				4，Phase 650-668 把复用机制推进到语言框架差分层面，说明同一套参数可以在不同格式、不同概念、不同答案路线之间复用，但复用依赖上下文协议。
				5，Phase 669-681 把局部机制整理成 graph atlas，并开始用因果图谱而不是单点 patch 解释语言编码。
				6，Phase 698-704 从 head 统计推进到 source contribution 与 channel ensemble，Phase 703 的 holdout 结果说明 source-restricted positive channels 具有跨样本稳定的 route/readout 因果作用。
				7，Phase 705-707 修正 first-token overlap 问题，用完整短语似然证明 unrelated donor 通常不会成为 donor phrase winner；但 prose phrase 经常获胜。
			核心成果：
				当前最可靠的新公式是：
					source_channel_ensemble = G_route + P_format + E_target_context + V_identity_local。
				其中 G_route 和 P_format 证据较强，E_target_context 证据增强，V_identity_local 尚未被最小因果定位。
				这说明已经找到“路线/读出底座”的关键因果结构，但还没有找到完整可迁移的语义身份代码。

		总阶段性结论：
			目前最可靠的完整理论体系是：
				语言智能 = 相对编码网络 + 对象知识锚定 + 关系/规则检索 + 条件化状态变换 + 源贡献路线增益 + 候选短语竞争 + 范数/格式/策略/生成读出门。
			最关键的新瓶颈是：
				从已经出现的 Level 4 source-channel component causal evidence，推进到自然生成闭环 Level 6，并进一步把 channel ensemble 拆到 neuron / MLP / residual trajectory 级别。
			2026-07-15 更新：跨模型闭合为 0/72，所有干预均为人工缩放而非自然门控，继续扩大候选扫描的边际价值接近零。
				必须转向建立正确的数学对象——因果状态等价类与算子代数。

		阶段十一：相对编码—复用差分—条件化机制图谱理论（Phase 708-713）
			核心任务：
				把 Phase 595-707 的局部读出机制整合为全局理论，系统化三段式理论：
					相对编码（Phase 63/64）→ 复用差分（Phase 632）→ 条件化机制图谱（Phase 646/711/712）。
			重要进展：
				1，Phase 708-710 把自然写入机制从 source-channel 推进到 Q/K addressing、V content、o_proj input、post-attn residual、MLP modulation 的因子拆分。
				2，Phase 711 完成机制图谱 v0 初始化，定义图谱节点 schema：G={u_i, r_i, s_i, e_i}，区分 attention_head 和 attention_channel，标注跨模型差异（qwen3=short_value_route_carrier，DS7B=prose/format route carrier，GLM4=unresolved）。
				3，Phase 712 开始 QK-V 因子图谱审计，把寻址结构和值内容搬运分开。
				4，Phase 713 对三段式理论做系统总结，给出统一数学公式，以词嵌入为例走完相对编码→复用差分→条件化图谱的完整计算流程。
			核心成果：
				理论收敛为"相对编码—复用差分—条件化机制图谱理论"：
					深度神经网络的语言能力来自同一参数骨架在不同输入边界和语义条件下生成的状态轨迹；
					这些轨迹不是孤立正确状态，而是 atlas 节点；
					语言生成 = 状态轨迹进入词表竞争后的自回归执行。
			关键硬伤：
				1，线性叠加公式 ΔM ≈ Σ<·, Δh_w> 与非线性 Transformer 不兼容（Phase 633 的 top12 < top1 失败）。
				2，相对编码未真正证伪点编码（Mantel 相关显著只说明距离结构被保留，不能排除绝对向量恰好诱导相似距离结构）。
				3，条件化图谱目前是索引系统非机制理论（多数节点共享同一组自然生成证据，无单元级因果证明）。
				4，图谱的"边"定义不清（是 Δh 内积？patch 依赖图？因果 do-calculus？三处混用）。
				5，跨模型不可比（qwen3 样本稀疏，GLM4 标 unresolved）。

		阶段十二：语义方向—MLP通道齿轮—边界移动的桥接理论（Phase 852-944）
			核心任务：
				在 GLM5 路线上，从抽象语义因子向下逐层定位到具体的 MLP 通道齿轮，
				建立从语义坐标到 boundary movement 的完整因果桥接链。
				同时在 GPT5 路线上建立行为资格门系统和跨模型审计框架。
			重要进展：
				1，Phase 940-944 在 qwen3 color en→en 上完成桥接：
				   共识残差坐标 → hidden 36/channels 2509,16,249 → MLP通道齿轮 → activation gap +3.70, slope gain +0.69。
				2，Phase 918-920 在 GLM4 L39 MLP 中定位 EOS-vs-a 有符号边界子空间，经共识齿轮压缩验证。
				3，GPT5 Phase 428-429 建立行为资格门系统（内容/接口/改口/终止四分离），
				   公式三分类管理（架构恒等式/证伪门/语言假设），
				   证据等级阶梯（行为资格→物理记录→密封预测→因果闭合→神经元定位）。
				4，跨模型审计发现：三模型 72 机制闭合 = 0/72，关系签名可读但不可搬（27/27 vs 0/9）。
				5，精确守恒验证 1248/1248 完全通过，确认架构可完全还原。
			核心成果：
				全项目最重要的科学结论：
				语言编码机制不是单个概念向量、单个神经元或单条路径，
				而是"因果状态等价类 + 条件化状态转移算子 + 动态路由 + 共享终端接口"组成的系统。
				破解这个系统的正确顺序是：因果状态发现 → 算子识别 → 组合规律 → 全局功能图谱 → 物理实现映射。
			关键硬伤：
				1，跨模型闭合 = 0/72：三模型用不同内部实现完成相同语言任务，共同机制语言未找到。
				2，自然门控完全缺失：所有有效干预都是人工缩放，模型自然状态下为什么不启动这些齿轮未知。
				3，strict-clean 自然生成 = 0：协议续写场压倒 EOS，语义答案被推出但模型不停止。
				4，行为稳定性脆弱：Qwen3 仅在无示例合同下通过，交叉示例后正结果消失。
				5，GLM4 通道干预造成普遍边界移动（非特异性），DS7B 坐标集中但不转化为因果边界移动。
			下一阶段方向：
				1，行为资格审计：穷举任务×合同×模型组合，确定真正稳定的行为窗口。
				2，在 GLM5 最强正结果（qwen3 color）上，用 GPT5 的盲化框架独立复现。
				3，转向"因果状态等价类"而非继续扩大候选扫描。

		阶段十三：全历史交叉分析与三条突破路线（Phase 945）
			核心任务：
				基于AGI_GLM5_MEMO_SUMMARY.md（Phase 20-940完整历史摘要），
				系统审计两年多、1300+Phase的所有关键发现，
				用统一评分体系筛选出最值得深入研究的成果，
				并从跨发现交叉分析中提炼突破路线。
			方法论创新：
				四维评分体系：证据等级(1-5) × 可复现性(1-3) × 桥接层级(1-4) × 理论价值(1-5)。
				这不是ranking而是分类——区分"已稳固可推进"和"有趣但脆弱"。
			12个关键发现（按总分排列）：
				第一梯队（总分>=12）：
				  #1 语义→MLP通道→边界桥接（Phase 940-944，13分，唯一三层桥接）
				  #2 GLM4 L39 EOS-vs-a齿轮（Phase 918-920，11分，最干净负控制）
				第二梯队（总分>=9，方法论基石+关键机制）：
				  #3 可解码≠因果（Phase 208-209，14分，双路线验证，改变方法论）
				  #4 Jacobian暗物质（Phase 225，10分，Top5(J)⟂Row(W_U)）
				  #5 CleanCausalEdge框架（Phase 867-874，12分，四要素因果标准）
				  #6 I-R-F-O分解+否定算子稳定性（Phase 294-300，12分，算子代数入口）
				  #7 Attn-MLP契约差异（Phase 285-293，13分，解释跨模型闭合=0）
				  #8 Protocol场压语义场（Phase 899-907，13分，精确定位终止缺口）
				  #9 Blocker子场+公共骨架（Phase 925-936，11分，可分解结构）
				第三梯队（重要拼图）：
				  #10 W_U候选竞争接口（Phase 255-284）
				  #11 小方差大因果（Phase 348-386）
				  #12 target-lift边界迁移（Phase 885-890）
			核心成果：
				三条突破路线：
				  A（最可操作）：跨模型复现Phase 944三层桥接。
				  B（理论最大潜力）：从否定算子扩展建立语言算子代数。
				  C（最直接瓶颈）：攻克Protocol场，让语义场胜出。
				跨发现整合图景：
				  语言编码 = 场博弈系统（语义场＋Protocol场＋候选场＋EOS场），
				  关键研究对象 = 因果状态等价类 + 场间转移算子。
				已系统排除假说清单：静态向量闭合、单神经元机制、可解码=因果、
				  简单线性边界、跨模型组件对应、单一阈值blocker等10类。
			关键硬伤：
				1，分析基于SUMMARY文件而非原始数据，可能有摘要偏差。
				2，操作符代数方向（路线B）是理论推测，在实践中可能完全失败。
				3，Protocol场本质（训练偏见/架构偏向/续写本能）尚不明确。
				4，GLM4/DS7B上三层桥接可能完全无法复现。
			下一阶段（Phase 946-950）：
				Phase 946：路线B起点——三模型否定算子复现 + 扩展到量化/模态算子族
				Phase 947：路线A扩展——GLM4 color语义→MLP通道齿轮桥接
				Phase 948：路线C起点——Protocol场token组成全量审计
				Phase 949：跨方向交叉验证
				Phase 950：根据结果确定主攻方向

### GPT路线
		阶段一：全链路 Trace 与输出闭合拆分（Phase 195-203）
			核心任务：
				从“记录最大神经元”升级为记录完整计算链路：
					prompt -> residual state -> attention/MLP output -> W_U projection -> candidate field -> global vocab competition。
			重要进展：
				1，发现 candidate color closure 不等于 global output closure。
				2，目标候选在颜色集合中赢，不代表在全词表中赢。
				3，解释词、格式词、标点、continue token 可能压过语义候选。
			核心成果：
				语言机制必须从“候选集合内部闭合”升级为“全词表竞争闭合”。

		阶段二：停止执行与协议边界（Phase 204-208）
			核心任务：
				区分输出句号、EOS 候选、stop 执行、post-period continuation。
			重要进展：
				1，句号不等于真正停止。
				2，模型可能输出正确短答后继续解释。
				3，stop / prose / echo / continue 是独立竞争场。
			核心成果：
				闭合必须包含 StopWins 与 ContinueSuppressed，而不是只看 answer_correct。

		阶段三：语言模式 Pattern 动力学（Phase 209-218）
			核心任务：
				把 answer-only、explain、repeat、list、chatty、protocol-follow 等模式从现象提升为模式对象。
			重要进展：
				1，模型内部不是只有答案机制，而是多个语言模式竞争。
				2，单 attention head route candidate 不等于 route cause。
				3，语言模式是多源、多头、多组件、分布式结构。
			核心成果：
				提出 Pattern：

					Pattern =
					[
						Trigger,
						StateVariables,
						FeatureTrajectory,
						PriorityProxy,
						OutputConstraint,
						FailureModes
					]

		阶段四：StateWrite、Readout 与 Competitor 机制拆解（Phase 219-233）
			核心任务：
				定位谁写入成功状态、谁推动读出、谁制造竞争 token。
			重要进展：
				1，success-drift 残差方向具有可干预效应。
				2，MLP 在多个状态写入中比 attention 更接近主写入器。
				3，readout threshold 和 competitor source 成为关键瓶颈。
			核心成果：
				语言模式必须拆成 StateWriteSource、ReadoutSource、CompetitorSource，而不是只看最终 token。

		阶段五：Pattern Family Atlas v1（Phase 234-245）
			核心任务：
				把模式研究从实验日志变成统一数据结构。
			重要进展：
				1，建立 behavior / internal trace / causal evidence / rollout / closure 字段。
				2，开始固定数据契约和前端可视化入口。
				3，识别高价值失败样本与候选 case bank。
			核心成果：
				语言机制公式升级为：

					LanguageMechanism
						=
						Σ_i
						α_i(x,t)
						P_i(x,t)

		阶段六：机制方向库与共享子空间（Phase 246-253）
			核心任务：
				提取自然机制方向，验证方向增强、方向移除、正交干预是否能控制模式。
			重要进展：
				1，机制方向可由 positive-negative 状态差得到。
				2，不同模式之间存在共享子空间。
				3，控制轴到 readout 轴存在耦合，不是简单独立方向。
			核心成果：
				图谱需要记录 subspace、direction bank 和 control-readout coupling。

		阶段七：Done / Stop / Continue 闭合机制（Phase 254-263）
			核心任务：
				区分 semantic done、template done、stop readout、continue readout。
			重要进展：
				1，答对只是闭合的一部分。
				2，真正闭合需要 done state、stop wins、continue suppression 和 rollout stable。
				3，continuation path 是很多失败的主因。
			核心成果：
				闭合硬判据：

					Closure
						=
						SemanticDone
						∧ StopWins
						∧ ContinueSuppressed
						∧ RolloutStable

		阶段八：Pattern Family Atlas v2 物理路径图谱（Phase 264-278）
			核心任务：
				把九大语言模式族、三模型、路径签名、组件归因、因果审计和 rollout 统一为可查询图谱。
			重要进展：
				1，Phase 264-265 固定 family、mode、case、path schema。
				2，Phase 266 建立三模型 behavior / readout baseline。
				3，Phase 267 开始 layerwise physical path trace。
				4，Phase 268 拆分 attention / MLP / residual 对 continue-stop margin 的贡献。
				5，Phase 269 发现 qwen3 / DS7B 支持 MLP 必要性，但 GLM4 暴露补偿路径。
				6，Phase 273 后形成 v2 主表、atlas_scores、case_details 和 client_index。
				7，Phase 274-278 开始用 gap queue、batch fill 和 recalibrated gaps 驱动后续实验。
			核心成果：
				继续路径公式：

					ContinuePath
						=
						B_embed
						⊕ AttentionRoute
						⊕ MLPWriterSet
						⊕ CompensationPath
						⊕ ReadoutCompetition
						⊕ RolloutEffect

		阶段九：Pattern Family Atlas v2.1 修正
			核心任务：
				修正 v2 简单平均、弱闭合、跨模型平均和线性归因风险。
			重要进展：
				1，Score 从简单平均改为 weighted_score + score_cap。
				2，Closure 从软分数改为四条件硬门槛。
				3，跨模型矛盾不再平均，改为 model_specific_mechanism。
				4，新增 claim_registry，把理论主张、证据、反例和下一步测试登记在一起。
				5，新增 nonlinear_coupling_audit，防止把线性 writer 分解误当完整机制。
				6，新增 prediction_validation，要求图谱能预测 heldout 样本。
			核心成果：
				当前 GPT 路线已经从“局部机制发现”升级为“可预测、可验证、可复用的语言模式图谱工程”。

		GPT路线当前总判断：
			1，方向正确：从单点神经元转向 PatternPath。
			2，价值明确：把语言机制拆成可测量、可干预、可复核的数据对象。
			3，最大风险：图谱完成度分数可能被误读为机制闭合。
			4，下一步重点：扩充样本、补全 component_path / causal / closure_quality、做 semantic_eval、做 nonlinear_coupling_audit、做 heldout prediction。
			5，真正突破标准：不是再发现一个强 patch，而是图谱能稳定预测新样本的路径、竞争项、失败类型和闭合结果。

		阶段十：精确守恒与行为资格门（Phase 358-429）
			核心任务：
				建立精确 Q/K 概率重放和守恒树、行为资格门系统、
				标签盲化发现方法、跨模型审计框架。
			重要进展：
				1，Phase 357-358：原生精度守恒验证 1248/1248，所有层更新全部精确恢复。
				2，Phase 372：精确 Q/K 概率重放和守恒树建立，逐神经元 MLP 写入、注意力来源写入全部可追踪。
				3，Phase 397：关系签名观测复现 27/27，但因果关系载体 = 0/9，答案切换 = 0/144。
				   看到的是计算残留的"足迹"，不是推动计算的"发动机"。
				4，Phase 428-429：行为资格门系统——内容/接口/改口/终止四分离门控。
				   公式三分类管理：架构恒等式（预设）→ 证伪门（预设）→ 语言假设（禁止预设）。
				   证据等级阶梯：行为资格 → 物理记录 → 密封预测 → 因果闭合 → 神经元定位。
				5，Phase 429 关键发现：Qwen3 仅在无示例合同下通过行为资格，
				   完全交叉示例后正结果消失。说明当前仅在极窄窗口中有稳定行为。
			核心成果：
				确立了 GPT5 路线的方法论基础：
				不是找更多候选，而是先用行为资格门审计哪些任务/合同组合真正稳定。
				证明了"观察到的模式（关系签名）不等于因果载体"。
			关键硬伤：
				1，跨模型 72 机制闭合 = 0/72——这是整个项目最致命的瓶颈。
				2，物理预测门 0/96：位置坐标选错导致全部预测失败。
				3，跨模型行为合同 0/3：只有 qwen3 部分通过。
				4，GLM4 内容门通过但终止门失败。
			下一阶段：
				在通过行为资格的组合上，建立执行同构合同，
				验证原生精度守恒，审计功能坐标。
				然后用标签盲化脉络发现方法，在盲化条件下寻找跨表面表达不变的子图。
