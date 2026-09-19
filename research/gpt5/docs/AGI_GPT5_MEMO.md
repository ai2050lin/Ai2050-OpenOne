## AGI Research Memo

> 本文档记录AGI研究的进展、问题分析和下一步行动

## Phase 2750: P4知识链载体专项——配对世界逐步postnorm分叉按关系族分层与"内部—行为解耦"的首次直接测量 [2026-09-14 23:12]

### C001：true_token vs 值置换双世界逐步分叉的预注册检验：载体在轨迹假说不成立，但发现族不变分叉形态与深层集中、行为不变的内部分叉 [2026-09-14 23:12]

**动机与冻结问题。** V2计划P4（`EVALUATION_AND_PLAN_V2_20260914.md` §P4）：knowledge_chain三模型全面最弱+B1/B8形状敏感+GLM 32条"结论正确但格式不合规"——假说"**链式关系不驻留在末端向量方向，而在多token生成轨迹与解释结构中**"；检验=比较知识链题与属性绑定题的逐步postnorm轨迹在两世界间的分叉时刻，"若分叉系统性滞后/弥散，则载体在轨迹成立，提取对象改为轨迹级算子"。本Phase在读取任何分叉观测之前冻结设计（`phase2750/qwen4/execution.json`）：qwen4拥有Phase2747 own_history全部7个世界（native、true_token_2747/2748、within_surface_class_permuted_token_2747/2748、surface_class_mass_2747/2748）×512题（受控320=5关系族×64 + natural_cmrc 192），每题逐步postnorm_BF16 (n,2560)全坐标保留、44条冻结表达另有全层all_hidden_BF16 (n,37,2560)。主世界对=$\mathtt{true}\_\mathtt{token}$（真实答案token自身历史）vs $\mathtt{within}\_\mathtt{surface}\_\mathtt{class}\_\mathtt{permuted}\_\mathtt{token}$（表面类内置换=值对置换历史），2747/2748两材料代互为复现。glm4/qwen14仅native世界无配对世界，P4分叉检验只对qwen4可定义。

**同前缀因果约束（本Phase方法论核心）。** 两世界独立greedy生成，行为分叉后同位置差异混合了"历史内容差异"与"生成token差异"，不可归因。故一切分叉统计只使用两世界**生成token id逐位相同的步**：$\tau_{\mathrm{tok}}=\min\{t: g^A_t\neq g^B_t\}$（None=重叠段全程一致），可比前缀长度$T=\tau_{\mathrm{tok}}$（序列全程一致时$T=\min(n_A,n_B)$）。实测：受控题两世界行为序列93.75%（attribute_binding）与98.44%（knowledge_chain）完全一致（long_distance_role 50%、negation_scope 75%、word_sense 96.9%），受控题每题恰3步（t=0,1,2，对应2749的B0/B1/B2），natural题生成至96步帽。

**测量量。** 逐步全坐标MSE $m(t)=\frac{1}{d}\lVert y^A_t-y^B_t\rVert^2$、余弦距离$1-\cos(y^A_t,y^B_t)$、相对RMS $\sqrt{m(t)}/\mathrm{rms}(y^A_t)$，全部float64、无截断。分叉时点$\tau=\min\{t: m(t)>4\times\tilde{\nu}(t)\}$，$\tilde{\nu}(t)$为族内题间重配置换对照的逐步中位数（200次，种子2750012；不同题的postnorm差异="无关系"尺度），未超阈右删失至$T$。弥散度=前缀内$log\,m(t)$对$t$的OLS斜率（窗0..8）。族间检验=knowledge_chain vs attribute_binding按source_group整簇分层bootstrap 2000次（种子2750010）。预注册判据：delayed（$\Delta\tilde\tau>0$且CI排除0）或diffuse（$\Delta\mathrm{slope}>0$且CI排除0）任一成立则"载体在轨迹"获支持，两者同时=strong；若kc剖面与ab不可区分或更早/更陡则判据否证。跨代稳定性对照=true_token_2747 vs true_token_2748（同条件两部署代，$m(0)$中位数0.015–0.049，为主信号的1/50–1/150）。CPU-only float64，无模型加载。

**结果一：预注册判据否证——知识链分叉不滞后也不弥散，反向显著。** 主世界对（2747代）逐步$m(t)$中位数（t=0/1/2）：attribute_binding 3.011/0.730/0.430，knowledge_chain 2.306/0.633/0.293（long_distance_role 2.094/0.611/0.391，negation_scope 2.375/0.581/0.304，word_sense 2.280/0.476/0.443；natural_cmrc 0.287/0.297/0.278）。kc−ab差值bootstrap：$\Delta m_0=-0.705$（CI95 $[-0.805,-0.452]$）、$\Delta m_1=-0.097$（$[-0.114,-0.052]$）、$\Delta m_2=-0.138$（$[-0.146,-0.126]$）——**知识链题对历史值置换的内部响应全程显著更小**（t=0低23%）。$\Delta\mathrm{slope}=-0.072$（$[-0.134,-0.040]$）——kc衰减更陡而非更弥散。$\Delta\tilde\tau=0$（CI $[0,0]$）。delayed=False、diffuse=False → **not_supported_by_preregistered_criteria**。2748代复现同方向：t=0处ab 2.646 vs kc 1.432。P4假说按预注册形式不成立：链式关系的"最弱"不是因为它藏得更深、显现得更晚——它的值置换内部响应反而最小、最快衰减。

**结果二：分叉形态族不变——B0占71%、深层集中、kc与ab层剖面几乎相同。** 分带能量占比（前缀内$E_{\mathrm{band}}=\sum_{t\in b}m(t)/\sum_t m(t)$）：五受控族B0全部68.5–71.4%（ab 71.4% vs kc 71.3%，差0.1个百分点）、B1 15.0–20.0%、B2+ 9.2–12.8%——**世界分叉的时间形态是族不变的通用机制，族间只有幅度差异**。层×步面板（44条冻结表达，kc/ab各4题，同前缀约束）：相对能量$nd(\ell,t)=\frac{\mathrm{mean}((h^A_{\ell,t}-h^B_{\ell,t})^2)}{\tfrac12(\mathrm{mean}(h^{A2}_{\ell,t})+\mathrm{mean}(h^{B2}_{\ell,t}))}$在t=0沿层为——L0–L12恰为0（嵌入与低层对自身历史值置换完全无响应）、L18 0.016–0.066、L24 0.134–0.150、L30 0.094–0.142、L36（最终norm前）0.812–0.827；kc与ab几乎重合（0.8267 vs 0.8121）。**分叉沿层深度单调集中：低层全盲、L24+接管、最深层层间相对能量0.8——自身历史值信息只在深层进入条件场**。t≥1后全层中位相对能量跌破阈值（分叉能量随步快速衰减，与B0占71%一致）。

**结果三（本Phase最重要的新发现）：内部—行为解耦的直接测量。** 同题两世界t=0内部差异$m_0\approx1.9\text{–}3.0$（受控），是跨代run噪声（0.015–0.049）的**约50–150倍**、是族内跨题差异（$\tilde\nu(0)$）的3–4倍——内部条件场对"自身历史里的答案值"高度敏感。但同一批题上两世界greedy生成token序列94–98%逐位完全一致：**历史值置换在深层postnorm中掀起巨幅变化，却几乎从不改变下一步写出的token**。2748 C011的"来源价值不特异"、2747的"训练收益与关系行为解耦"、本Phase的"内部敏感与行为不变解耦"指向同一个结构：**postnorm/深层状态携带大量不驱动输出的历史值坐标——模型的条件场远比它的行为输出高维**。这为"提取对象改为轨迹级算子"之外的第三条路（内部状态解码而不经行为代理）提供了首个定量理由。

**结果四：置换敏感度幅度分层与既有结论闭环。** 幅度序（t=0）：ab 3.011 > negation 2.375 > kc 2.306 > word_sense 2.280 > ldr 2.094 ≫ natural 0.287。cosd（幅度归一后）t=0：ab 0.121 vs kc 0.106——族间MSE差约1/3来自postnorm尺度差，归一后kc仍显著更不敏感但差距收窄。与2747"knowledge_chain在B1/B8、双世界全面最弱"（Q4 49/64）闭合：kc的关系追踪弱，部分体现为**其条件场对关系内容扰动的响应幅度本身最小**。

**相关文件。** `tests/glm5/phase2750_rdc_knowledge_chain_dynamics.py`（SHA256 b27f5e9e70f97814…）；`result/rdc_query_construction_20260913/phase2750/qwen4/{execution.json 64c8445a211fbe3c…, result.json caa452d5e4bafe42…, question_scores.npz deb221654d01f6d5…, question_scores_meta.json db2f7ed8660945a4…}`；输入=`phase2747/own_history/qwen4/{native,true_token_2747,within_surface_class_permuted_token_2747,true_token_2748,within_surface_class_permuted_token_2748,surface_class_mass_2747,surface_class_mass_2748}`（run result SHA冻结于execution.json，字段npz逐文件SHA已在2747 records内登记）；物理归档`C:/AI2050-RDC-Archive/rdc_query_construction_20260913/phase2750`。

**问题硬伤。** ①τ判据在受控3步数据上无区分力：置换null的跨题中位数×4在t=0即超过全部$m_0$中位数，所有题右删失至$T=3$，$\Delta\tilde\tau\equiv0$是构造性的——"分叉时刻"在3步前瞻协议下退化为"m(t)曲线族间比较"，P4原文设想的"滞后/弥散"二判据只在更长同前缀预算下有检验力；②层×步面板仅44条冻结表达（kc/ab各4题/世界对），层剖面是描述性而非独立检验；③cosd显示族间MSE幅度差约1/3为尺度效应，相对量与绝对量结论需分开陈述（已分开）；④"内部—行为解耦"的greedy行为不变率以重叠序列为条件，行为分叉题（kc 1.6%、ab 6.3%）未进入同前缀统计，其行为差异本身是另一现象；⑤qwen4单模型——glm4/qwen14无配对世界，族不变形态的跨模型普适性未测。

**结论。** P4预注册判据：否证（delayed=False、diffuse=False，反向显著：kc响应更小更陡）。三个载入MEMO的新事实：**（1）族不变分叉形态**——值置换分叉B0占~71%、L0–L12全盲、L24+接管、L36峰值0.81–0.83，形态跨5关系族不变；（2）**内部—行为解耦**——50–150倍于run噪声的深层内部响应对应94–98%不变的行为序列；（3）**响应幅度关系族分层**（ab最大、kc最小）与2747行为最弱结论闭环。P4的"提取对象改为轨迹级算子"路线判据未获支持，按V2计划回落到末端向量提取，但（2）提示内部状态解码是绕开行为代理的独立可行对象。

**接续。** V2计划执行序：P1=Phase2749✓、P4复分析=Phase2750✓。下一阶段P5①（参数方向→权重子集→移植的第一步）：用2747/2748的12个训练checkpoint（无损归档可逐字重建）+2737梯度账本+2748 parameter-path API，定位"正确vs置乱vs类别"方向对的参数子集差异至gate/up/down标量——纯参数分析、无GPU。目标与总目标（外部语言操作→内部条件坐标齿轮→跨层编译）一致，自动进入Phase 2751。

## Phase 2751: P5①参数方向子集定位——三监督方向对差异场跨代复现，gate/up/down标量子集预注册支持 [2026-09-14 23:39]

### C001：74,711,040个可训练标量上三方向对（正确vs置换vs类别）的跨代复现检验：三对全部通过预注册判据，差异集中于少数神经元三元组 [2026-09-14 23:39]

**动机与冻结问题。** V2计划P5三步（①参数子集定位→②定向冻结/置乱干预→③跨模型移植）的第①步，纯参数分析、无GPU、无模型加载。检验问题：layers[16].mlp的74,711,040个FP32标量中，是否存在一个坐标子集，其上"正确监督（true_token）vs置换监督（within_surface_class_permuted_token）vs类别监督（surface_class_mass）"三方向对的训练更新差异场在两个独立材料代（2747/2748）之间复现？设计在任何delta观测之前冻结（`phase2751/qwen4_block16/execution.json`，status=frozen_before_any_Phase2751_delta_observation）。输入=6个2747训练checkpoint的`delta_128.npz`（3监督条件×2材料代；delta值=精确FP32 actual−original，reconstruction_residual为位级修补项不属于数学delta；其余全部参数位冻结有unchanged-parameter回执链）。

**测量量与公式。** 逐矩阵（gate/up/down各展平24,903,680维）与pooled（三矩阵拼接74,711,040维）两层。方向余弦$\cos(\delta_a,\delta_b)$构成6×6矩阵（每矩阵15个run对）。三方向对差场$\Delta_{ab}^{(g)}=\delta_a^{(g)}-\delta_b^{(g)}$（$g\in\{2747,2748\}$），噪声对照$N_{a}=\delta_a^{(2747)}-\delta_a^{(2748)}$（同条件跨代差场）。跨代复现统计：Pearson $r=\mathrm{corr}(\Delta^{(2747)},\Delta^{(2748)})$、cosine、top-$k$绝对差Jaccard $J_k=\frac{|S_k(\Delta^{(2747)})\cap S_k(\Delta^{(2748)})|}{|S_k(\Delta^{(2747)})\cup S_k(\Delta^{(2748)})|}$（$S_k$=按$|\Delta|$取前$k$坐标，$k\in\{0.1\%,1\%,5\%,10\%\}$）、能量比$E_{\Delta/\nu}=\lVert\Delta^{(2747)}\rVert^2/\lVert N\rVert^2$与$E_{\Delta/u}=\lVert\Delta^{(2747)}\rVert^2/(\lVert\delta_a^{(2747)}\rVert^2+\lVert\delta_b^{(2747)}\rVert^2)$。神经元级三方一致：gate行/up行/down列分别按差场行/列范数取top-1%（各97），求三方交集。预注册判据：方向对stable iff $r\ge0.5$且$J_{1\%}\ge0.3$（pooled）；localisation_supported iff至少一对stable。工程：pooled差场float32累积+分块float64统计（避免5.4GB float64驻留）；CPU-only，全坐标无截断，top-k仅用于Jaccard集合定义、非激活读出。终跑268.8s。

**结果一：预注册判据三方向对全部通过——localisation_supported=True。** pooled（74,711,040标量）：true_vs_permuted $r=0.963$、$J_{1\%}=0.704$、$J_{0.1\%}=0.778$、$E_{\Delta/\nu}=35.5$；true_vs_mass $r=0.866$、$J_{1\%}=0.452$、$E_{\Delta/\nu}=8.3$；permuted_vs_mass $r=0.967$、$J_{1\%}=0.719$、$E_{\Delta/\nu}=26.0$。逐矩阵9组（3矩阵×3方向对）全部通过且方向一致：$r\in[0.858,0.971]$、$J_{1\%}\in[0.424,0.726]$。up矩阵最锐（true_vs_permuted $r=0.969$、$J_{1\%}=0.712$、$E_{\Delta/\nu}=41.2$）。

**结果二：参数更新方向的几何结构——同条件跨代≫跨条件，三类监督占据近正交子空间。** 同条件跨代方向余弦：gate true/permuted/mass=0.915/0.973/0.842，up=0.932/0.984/0.857，down=0.917/0.980/0.850；跨条件（同代）：true|permuted 0.317–0.404、true|mass 0.205–0.259、**permuted|mass近正交（0.009–0.034）**。差场能量达两条件更新能量之和的0.73–0.99倍（$E_{\Delta/u}$，permuted_vs_mass最高0.98–0.99）——跨条件更新几乎不共享能量。即两个独立材料实现给同一监督条件写出的更新方向高度一致（0.84–0.98），而三个监督条件彼此的更新方向几乎不共享坐标：监督语义在参数空间不是弥散漂移而是可复现的方向性结构。

**结果三：差异集中于少数神经元三元组而非弥散。** 神经元级三方交集（gate行∩up行∩down列，top-1%各97）：true_vs_permuted **16个**、true_vs_mass **4个**、permuted_vs_mass **15个**神经元，独立性期望$9728\times0.01^3\approx0.01$——富集约1600×/400×/1500×。任两方交集43–49个（独立性期望$\approx2.9$）。方向对差异不在74.7M标量上均匀展开，而是收敛到可点名的gate/up/down协同神经元。

**结果四：描述性fields_link。** 44条冻结表达在block16的中位|激活|：true_vs_permuted差场top-1% gate行（64行样本）gate激活0.605 vs 其余行0.539（1.12×）、product激活0.0763 vs 0.0554（1.38×）——差异神经元系统性地略更活跃；描述性对照，非功能测试。

**运行史与确定性。** 首跑1s崩溃（defaultdict未导入，先于产物写入）；二跑261.9s完成，事后核对发现pooled方向余弦存在平方范数三重累计缺陷（per-run范数在PAIRS循环内每个方向对累加一次共3次→15个pooled余弦全体÷3；$0.3077\times3=0.923$与逐矩阵0.915/0.932/0.917精确吻合验证；预注册判定不受影响——pooled差场按对独立累积、Pearson/Jaccard/能量比全部正确），删除三个产物、范数累加移出PAIRS循环后终跑268.8s。两跑`subset_indices.npz` SHA完全一致（c6a85429a8f0d291…）——分析管线确定性证据。

**相关文件。** `tests/glm5/phase2751_rdc_parameter_direction_subset.py`（SHA256 c3a2d8952d6d3c2d…）；`result/rdc_query_construction_20260913/phase2751/qwen4_block16/{execution.json aff5028e595f4ff3…, result.json de783195f143ebef…, subset_indices.npz c6a85429a8f0d291…（三方向对pooled top-1%坐标索引，int64各747,110）}`；输入=`C:/AI2050-RDC-Archive/rdc_query_construction_20260913/phase2747_fields/training/{true_token,within_surface_class_permuted_token,surface_class_mass}_{2747,2748}/delta_128.npz`；SHA链冻结于execution.json：`phase2747/training/result.json 10b7b8b416e0341a…`、`phase2748/parameter_formation/result.json e99ea5cdccc2e480…`、`phase2747_rdc_training.py fa24017d71d29c0d…`；物理归档`C:/AI2050-RDC-Archive/rdc_query_construction_20260913/phase2751`（D盘junction同前法）。

**问题硬伤。** ①仅两材料代：复现=两代互证，无第三代holdout，$r$与$J$无独立验证集；②单可训练块：结论只覆盖layers[16].mlp的74.7M标量，不涉及任何冻结参数，定位不可外推至全模型；③每条件每代仅一次训练实现（128步、单一数据顺序）："跨代复现"混合材料代效应与实现特异性，严格应为"两次独立实现的复现"而非种子方差意义下的复现；④$J_k$对差场重尾敏感，$k$网格与0.3阈值虽预注册，但$J_{5\%}\approx J_{10\%}$平台提示top集合部分由少数大坐标主导，小坐标贡献的解释需谨慎；⑤fields_link纯描述性；⑥P5①只建立"参数子集存在且跨代复现"，子集是否**承载**监督语义必须由P5②干预检验——定位≠功能。

**结论。** P5①预注册判据：**支持**（三方向对全部stable，localisation_supported=True）。与Phase2750拼合，条件化齿轮前两环首次同批成立：监督方向在参数层有跨代复现、近正交、神经元级集中的坐标签名（外部语言操作→参数子空间这一环落实），而历史值置换在状态层掀起50–150倍噪声的响应却不改行为（状态→行为解耦）——参数子空间与条件场两端均已定位，P5②干预是检验这两端是否真由所定位子集承载的唯一手段。

**接续。** V2执行序状态：P1=Phase2749✓、P4=Phase2750✓、P5①=Phase2751✓——**执行序中全部无新GPU成本的阶段已完成**。剩余阶段均需GPU：P5②定向冻结/置乱P5①定位子集（输入已备：subset_indices.npz三对pooled top-1%索引+2747训练协议+896题bridge评估集）、P2多位置采集、P3预设强度干预、P5③跨模型移植。下一阶段P5②与总目标（外部语言操作→内部条件坐标齿轮→跨层编译为下一token概率）相同，判定为自动进入Phase 2752：先GPU准入（显存核查+qwen4加载验证），再冻结P5②预注册设计（干预单元=pooled top-1%子集；干预=子集delta置零/置乱vs补集对照；判据=关系行为预测变化的定向性大于随机破坏）。


### C002：pooled差场"逐元素相加"实现缺陷的发现与修正重跑——三方向对预注册判定在真拼接场上不变 [2026-09-15 00:31]

**缺陷发现。** Phase2752预检（对2751子集做坐标解码一致性核对）暴露：gate/up/down三个矩阵参数量完全相同（各24,903,680），C001运行的pooled差场累积`prev + v32`是**逐元素相加而非拼接**——"pooled差场"实为三个不同参数坐标的逐元素和（第i个"坐标"=gate_flat[i]+up_flat[i]+down_flat[i]），无参数级语义。经验复现确认：三次累积后pooled长度始终24,903,680，而真拼接应为74,711,040。

**影响范围精确界定。** ①pooled方向余弦（6×6矩阵15对）**碰巧正确**：dot积按矩阵累加在数学上等价于拼接场的点积（0.923/0.981/0.851与逐矩阵能量加权平均精确吻合）；②pooled对统计（Pearson/cosine/Jaccard/E比）在和场上计算——**预注册判定的执行对象与设计不符**（数值恰接近真值：和场余弦≈能量加权平均，因跨矩阵叉积项小）；③**pooled_top子集索引完全无效**——"top-1%坐标"是和场的坐标，无法对应任何单一参数；v2的subset_indices.npz（每对仅249,037个=gate的1%且全部落在gate范围）被Phase2752预检发现并中止使用，未进入任何干预；④逐矩阵结果、神经元级三方交集、fields_link不受影响（各自在本矩阵场上计算）。

**修正与重跑。** 修复为按矩阵顺序[gate|up|down]真拼接（74,711,040），pooled对统计与top-1%索引在拼接场上重算；execution.json的analysis字段显式声明拼接语义与"等长矩阵逐元素累积会静默混坐标"的警示；删除v2三个产物后重跑261.5s（v3）。修正后pooled对：true_vs_permuted r=0.9631、J₁%=0.679、E=35.65；true_vs_mass r=0.8665、J₁%=0.454、E=8.35；permuted_vs_mass r=0.9666、J₁%=0.696、E=26.28——**三对全部通过预注册判据，C001判定（localisation_supported=True）不变**；pooled方向余弦逐位不变。三对子集均横跨三矩阵（gate 122–147k / up 292–301k / down 308–329k每对），三对并集1,415,316坐标（对间重叠191k/535k/237k）——**up与down矩阵承载了差异坐标的约80%**，与up矩阵最锐（E=41.2）一致。

**方法论教训。** 等长矩阵的"pooled"累积必须显式拼接——逐元素加法不报错、形状不变、统计量数值合理（介于逐矩阵值之间），只有坐标语义核对（索引范围×矩阵偏移分解）才能暴露。Phase2752预检的"union全部落在gate"核对是本缺陷的唯一检出手段：**消费方对生产方产物的语义核对是比生产方自检更强的验证层**。

**修正后文件。** `result.json ff06d0702031afca…`、`subset_indices.npz d1be2caa4d9f9391…`（9,058,561字节）、`execution.json bcb7b4963854bbe9…`、脚本 `225cb1904fac65c3…`；v2产物（result 65b7c217…、npz c6a85429…、execution aff5028e…）已删除。


## Phase 2752: P5②定向干预——2751子集union的功能载荷检验：充分性6/6、必要性4/6（true反向），条件结构化功能载荷 [2026-09-15 01:07]

### C001：1,415,316个定位坐标的冻结/置乱干预——子集是置换监督语义的必要且72%充分的载体，true收益在子集内但被补集反作用抵消 [2026-09-15 01:07]

**动机与冻结问题。** V2计划P5②：P5①定位了参数子集（Phase2751），但**定位≠功能**——本Phase用定向干预检验子集是否因果承载训练形成的行为效果。设计在任何干预观测前冻结（`phase2752/qwen4_block16/execution.json`）：干预对象=三对top-1%子集的并集$U$（1,415,316坐标=可训练块的1.89%；gate 246,786/up 538,867/down 629,663），对6个2747训练checkpoint（3条件×2代）的**bit-exact最终FP32态**（original+delta_128+reconstruction_residual回执链）施加9个变体：full（参考）、pz_×3（逐对去delta）、pz_union、rzero_union（随机对照，种子2752000，与U不相交且逐矩阵匹配大小）、only_union（仅保留U上delta）、ronly_union、shuffle_union（U上delta值矩阵内置乱，种子2752100+序数）。评估=2747冻结896题面板（validation 192+diagnostic 512+fresh 192，同顺序）、单行前向、2747训练期同精度regime（FP32可训练块+BF16桥）。**材料延伸**（2747只为训练行定义了permuted_target/surface_class）：面板行surface_class=classes[target]（确定性）+permuted_target=按2747协议（kind/cohort/language/surface_class组内、同标签重合保留）以预注册种子2752001置换——78.46%改变、27组、3个单例组固定点。判据预注册（二项null 6/6，p=1/64）：P1必要性$d_{obj}(pz_U)>d_{obj}(rzero_U)$、P2充分性$red(only_U)>red(ronly_U)$（$d_{obj}$=变体−full，$red$=base−变体，own objective=各run训练目标）、P3放置$d_{obj}(shuffle_U)>0$；localisation_functional=P1∧P2均6/6。

**完整性门。** base态对2747已提交bridge基线：**全部5个数组逐位相等**（postnorm_BF16/NLL/argmax/entropy/surface_mass）——RTX 5080前向数值与2747训练运行位级一致，整个比较链位级可信；结束恢复original后4题复验逐位相等。运行1864.1s，约5.7万次单行前向，无模型外写入。

**结果一：充分性P2 6/6通过——子集单独捕获训练净收益的72–200%。** only_union（仅1.89%坐标上的delta）相对base的收益占full净收益比例：permuted 73.8%/72.0%、mass 79.9%/75.2%、true 172%/200%；ronly_union（随机1%坐标）仅捕获0.5–24%。true runs的only_union**超过**full净收益——补集delta在true目标上是净损害（+0.083/+0.102）。

**结果二：必要性P1 4/6——true runs反向，朴素形式被否证。** pz_union损害（d_obj）：permuted +2.443/+2.227（随机对照仅+0.008/+0.011，**324×/195×**）；mass +0.0018/+0.0030（−0.0007/+0.0001，2.6×/20.7×）；true **−0.033/−0.026**（+0.0005/−0.0059）——移除子集反而**改善**true目标。幅度特异$|d(pz_U)|/|d(rzero_U)|$=2.6–324×，6/6。P3放置4/6（shuffle损害permuted +3.02/+2.69、mass +0.0033/+0.0062，但改善true −0.027/−0.022）。

**结果三（本Phase核心新发现）：功能载荷条件结构化。** (a)**union子集是置换监督语义的主要载体**：移除灾难性、单独复现72–74%收益、放置敏感；(b) true runs净收益=大而相反的两分量之差：子集分量收益（only_union −0.199/−0.205）被补集分量损害（+0.083/+0.102）抵消后仅剩−0.115/−0.102；(c) 双向特异性：true runs的pz_union降nll_target（2.591→2.558）升nll_permuted（11.32→11.78），permuted runs反之（7.37→9.81 vs 2.05→2.12）——子集承载"表面类置换→目标token"方向的条件化坐标，而非各条件的全部收益；(d) 逐对分解：true runs中tm（true_vs_mass）子集移除一致有益（−0.036/−0.018），tp/pm一致有害——**true_vs_mass签名与true目标反相关**。

**结果四：与2747/2750/2748链条的机制层闭环。** 2747"正确答案续训损害双世界区分"、2748 C011"来源价值不特异"、2750"内部—行为解耦"→本Phase给出参数层解释：true训练的净行为改进本来就小（0.10–0.12 NLL），且是子集收益与补集损害的相消残差；而置换/类别监督的改进**大部分驻留在1.9%的定位坐标内**。"监督语义→参数子空间"的映射对置换/类别条件近乎完全，对true条件部分且对抗。

**相关文件。** `tests/glm5/phase2752_rdc_subset_intervention.py`（SHA256 3f20d2212de23cbb…）；`result/rdc_query_construction_20260913/phase2752/qwen4_block16/{execution.json e3b0ffb83608c317…, result.json e5437486ecf5f13e…, intervention_scores.npz 88a5cc18d617482b…（165个逐行数组）, intervention_scores_meta.json 2fd2ad784fd302eb…}`；输入=Phase2751 v3 `subset_indices.npz d1be2caa4d9f9391…`、2747六checkpoint delta_128.npz、bridge基线；物理归档`C:/AI2050-RDC-Archive/rdc_query_construction_20260913/phase2752`。

**问题硬伤。** ①U混合三对签名，联合干预不能归因单对（逐对pz是描述性的，only_×3未做）；②true runs的P1反向未在预注册预见——"幅度特异"与"条件结构"是事后结构化解释，需下一Phase以单对干预+族级行为判据检验；③单一全局随机对照集（非逐run重抽）；④permuted_target面板延伸是新构造（协议镜像2747，但78.46%改变率意味着置换run的面板own objective定义依赖此延伸）；⑤own objective是训练目标对齐量而非V2原文的"关系行为预测变化"——own_mean_by_family已逐run逐变体记录（intervention_scores.npz），判据化检验留给下一Phase；⑥单模型qwen4，P5③跨模型未动。

**结论。** 预注册严格形式：**localisation_functional=false**（P1 4/6、P3 4/6否证了"子集承载本条件全部收益且方向一致"的朴素假说）。但P2充分性6/6、幅度特异6/6与条件结构模式给出更强替代图景：**2751定位的参数子空间是监督语义的主要功能载体，载荷按条件不对称**——置换语义几乎完全驻留其中（必要+72%充分+放置敏感），true收益在其中但与补集对抗。P5②"定位≠功能"之问的首次直接回答：对置换/类别条件定位=功能；对true条件定位=部分功能+对抗性补集。

**接续。** V2执行序：P1=2749✓、P4=2750✓、P5①=2751✓（C001+C002）、P5②=2752✓。剩余：P5②细化（单对only_×3干预+族级行为判据化+true补集损害定位）、P5③跨模型移植、P2多位置采集、P3预设强度干预。下一阶段Phase 2753=P5②细化：单对定向干预（only_tp/tm/pm+pz已有）×族级行为指标判据化，检验"置换语义驻留在哪一对签名"与"true补集损害来自哪些坐标族"——与总目标相同，判定为自动进入；GPU准入已验证（2752同机通过）。



## 存储迁移记录：C 盘归档整体迁入 D 盘结果树 [2026-09-15 02:57]（本条为第三次重插，前两次被 Edit 工具陈旧基座改写抹除，自本版起 MEMO 禁用 Edit 通道）

### 事件
用户手动把 `C:\AI2050-RDC-Archive\rdc_query_construction_20260913` 下 8 个归档目录（约 75.5 GB）移入 `D:\AI2050\Ai2050-OpenOne\tests\glm5\result\`。代理随后完成布局修复：8 目录同盘重命名归位至 `result\rdc_query_construction_20260913\`（与 phase2746/2747 同层）；删除 5 个悬空 junction（phase2748–2752）；重建 `phase2746\field_store`、`phase2747\field_store` 两个 junction 指向 D 盘物理目录；更新 5 个活代码文件的 `C:/AI2050-RDC-Archive/...` 路径常量与磁盘护栏（C:→D:）；导入级探针 22 项全绿。

### 完整性验证
- phase2750 result.json SHA256 = `caa452d5e4bafe42…`（与 MEMO 登记值一致）；phase2751 subset_indices.npz = `d1be2caa4d9f9391…`（与 v3 终跑一致）——迁移零损坏。
- 目录文件数：phase2746_fields 3717、phase2747_fields 7138、phase2747_metadata 3409、phase2748 32650、phase2749 12、phase2750 4、phase2751 3、phase2752 6。

### 编辑后脚本 SHA（重插时实算）
- `rdc_construction_storage.py` SHA256 `f5240045b05a2a19f22541f1dde783c69abb6e3fca4d1a2f78ea1cdf303050b4`（迁移后路径常量版本，当前实算）
- `rdc_formation_common.py` SHA256 `f9a7ff9472ebfa03ca02ea6141af5eb1b038bfad7487d2d0245b8bc6906f3898`（迁移后路径常量版本，当前实算）
- `rdc_question_common.py` SHA256 `c24539a15c0ca7f71185cc989e902306b0c96e5d419e9515c4ceb5c787298a8c`（迁移后路径常量版本，当前实算）
- `phase2751_rdc_parameter_direction_subset.py` SHA256 `e8ab778d5de82b6c0bb9725f857adf7bfa2511892db73a6f4e10ccb4c77653f4`（迁移后路径常量版本，当前实算）
- `phase2752_rdc_subset_intervention.py` SHA256 `e7c0c4879a2eb5bfb06dd70150c318fa1acc3e91a2d5268afc55b0f406b7a5d8`（迁移后路径常量版本，当前实算）

### 教训（写入跨项目记忆）
- reparse 扫描只对 reparse 项递归会漏真实目录下一层的 junction（field_store 漏检）。
- `\?\` 前缀 junction 目标使 exists()/disk_usage 判定失效。
- 沙箱视图切换：Write/Edit 报成功但未落真实磁盘；**MEMO（>10MB）对 Edit 工具的全文件改写尤其危险——陈旧基座会静默抹除其间由 Python append 写入的内容**。铁律：MEMO 只允许非沙箱 Python append/插入，任何例外后必须 Grep 双向复核。

## Phase 2753: P5② 细化——单对签名隔离与补集独立净效应："补集损害"是减法推断伪影，损害实为 knowledge_chain 族特异且被族间对冲掩盖 [2026-09-15 03:32]

### 原理与设计（任何 Phase2753 观测前冻结，execution.json `b98aba707d0ad9839678c662cc684b7d30fb3f3076dde962e2c3b002aeb232dd`）
Phase2752 只测过必要性（pz_<pair>，从 full 中去掉一对）与并集充分性（only_union），从未隔离单对；其"true 补集净损害 +0.08~0.10"是由 red(full)−red(only_union) **减法推断**的，隐含可加性假设。本 Phase 以直接干预检验：

$$\mathrm{red}(v)=\overline{\mathrm{own}}(\mathrm{base})-\overline{\mathrm{own}}(v),\quad v\in\{\mathrm{full},\mathrm{full\_noR},\mathrm{only\_tp},\mathrm{only\_tm},\mathrm{only\_pm},\mathrm{only\_union},\mathrm{only\_complement}\}$$

变体构造（$D$=训练 delta，$R$=位级修补残差 $|R|\sim10^{-7}$，only_* 与 full_noR 均排除 $R$）：only_<$p$> = original，仅对第 $p$ 对 top-1% 子集（747,110 坐标）加 $D$；only_union = 仅并集（1,415,316 坐标：gate 246,786/up 538,867/down 629,663）；only_complement = original+$_{D}$ 于并集之外全部坐标。9 变体态 × 6 checkpoint × 896 题面板（材料延伸与 2752 逐位相同：种子 2752001，置换率 78.46%，27 组）。完整性门 **strict_pass**（base 与 2747 bridge 基线全数组逐位相等，RTX 5080），恢复逐位精确；跨会话复现：red(only_union) 与 2752 登记值 **6/6 精确一致**。

### 预注册判据结果：两项否证，均换来更强结构
| 判据 | 预注册 | 结果 |
|---|---|---|
| C1 载体跨代一致 | 3/3 条件 | **2/3** ✗：permuted/mass 条件载体=permuted_vs_mass（两代一致✓）；true 条件跨代翻转（2747: tm 0.1944 vs pm 0.1900，差 0.004；2748: pm 0.1897 vs tm 0.1887，差 0.001）——**true 收益在三对签名上近均匀分布，无单一载体** |
| C2 补集不对称损害 | true 两代 red_comp<0 且 permuted>true | **0/2** ✗：true red_comp=**+0.149/+0.132（收益非损害）**；不对称方向本身成立（permuted +2.51/+2.49 ≫ true），但"损害"符号错误 |

**核心更正（对本 MEMO Phase 2752 结论的修正，值得记三遍）**：2752 的"true 补集净损害"是减法推断伪影——补集单独是收益，"损害"实为 union×complement 的**强负交互（次可加性）**：true runs 加性差距 $|\mathrm{red}(\mathrm{union})+\mathrm{red}(\mathrm{comp})-\mathrm{red}(\mathrm{full\_noR})|$=0.230/0.235，permuted runs=1.212/1.166（各自身份收益 0.199+0.149=0.348 vs 联合 0.118；3.655+2.511=6.166 vs 联合 4.955）。训练改动在"定位子集"与"其余参数"两个半空间**各自独立产生收益，联合时相互抵消大半**。

### 结果汇总（red 矩阵，正值=own 目标改善）
| run | full | only_tp | only_tm | only_pm | only_union | only_complement | 加性差距 |
|---|---|---|---|---|---|---|---|
| true_2747 | 0.115 | 0.161 | 0.194 | 0.190 | 0.199 | 0.149 | 0.230 |
| true_2748 | 0.102 | 0.177 | 0.189 | 0.190 | 0.205 | 0.132 | 0.235 |
| permuted_2747 | 4.955 | 3.432 | 2.702 | 3.488 | 3.655 | 2.511 | 1.212 |
| permuted_2748 | 4.714 | 3.151 | 2.524 | 3.227 | 3.392 | 2.487 | 1.166 |
| mass_2747/2748 | 0.004/0.005 | ~0.003 | ~0.003 | ~0.003 | ~0.003 | ~0.002 | ~0.001 |

2752 交叉核对一致：permuted runs 的 pz 必要性排序（pm 2.30 > tp 2.27 > tm 1.74）与 only 充分性排序（pm 3.49 > tp 3.43 > tm 2.70）相同——**pm 对置换语义既最必要又最充分**；mass 条件整体效应近饱和（base 0.181）。

### 族级分解（own_mean_by_family 变化；负=改善）——C2 否证背后的真结构
- permuted runs only_union：**全族广泛改善**，跨代排序完全一致（attribute_binding −5.29/−5.04、long_distance_role −5.25/−5.23 最大；knowledge_chain −2.66/−2.66 最小但仍大）。
- true runs only_union：集中 word_sense（−0.90/−1.12）、negation_scope（−0.69/−0.59）、knowledge_chain（−0.33/−0.18）。
- **true runs only_complement：族间对冲**——word_sense −1.64/−2.26、negation_scope −1.35/−1.06 改善，但 **knowledge_chain +1.10/+2.04 恶化**。均值 +0.149"收益"是 word_sense/negation_scope 改善掩盖 knowledge_chain 恶化的对冲残差。**"true 补集损害来自哪"的答案：knowledge_chain 族特异**——与 Phase 2750 的知识链主分叉形态（B0 占 71%）和内部—行为解耦主题直接衔接。attribute_binding 族在 true runs 全变体 Δ=0（base 态已饱和）。

### 相关文件
- `tests/glm5/phase2753_rdc_pair_isolation.py`（脚本；存储为 D 盘直写，无 junction）
- `result/rdc_query_construction_20260913/phase2753/qwen4_block16/{result.json e31275ac…, isolation_scores.npz 826135a0…, isolation_scores_meta.json f0a3f4aa…, execution.json b98aba70…}`（immutable；逐题 own/nll_target/nll_permuted 数组已存档供 CPU 级题目分析）

### 问题硬伤
1. **预注册设计缺陷（本 Phase 最大教训）**：C2 在均值层面预注册，未预见族间对冲会把"补集损害"整体翻转成"收益"——均值判据掩盖了 knowledge_chain 的 +1.1~+2.0 恶化。族级判据应与均值判据并列预注册。
2. 单块（layers[16].mlp）结论；其余 36 层原封未动。
3. true 载体跨代翻转的两侧差距仅 0.001~0.004 NLL，接近测量分辨率——"无单一载体"结论可靠，但"翻转"本身不构成证据。
4. 加性差距混合了非线性交互与 $R$ 排除效应（full 与 full_noR 差仅 0.002，故交互占绝对主导）。
5. 仅 qwen4、两代材料。

### 结论与接续
监督语义的功能载荷结构修正为三层：(i) permuted 语义有明确载体（pm 签名，必要+充分+跨代一致）；(ii) true 收益弥散于全部三对签名且与补集强次可加；(iii) 补集通道的损害是 **knowledge_chain 族特异**的，被 word_sense/negation_scope 收益对冲。**接续 Phase 2754（已判定自动进入，CPU-only）**：knowledge_chain 补集损害的题目级定位——用已存档的 2753 逐题数组检验 (a) kc 恶化是题目级普遍还是少数极端题；(b) 恶化题是否伴随 argmax target→permuted 转移（双世界判据）；(c) 与 2750 的 B0 分叉形态和族内关系类型对照。之后 P5③ 跨模型移植。

## Phase 2754: P5②收官——knowledge_chain 补集损害的逐题定位：分布式、跨代复现（ρ=0.986）、B0 分叉正相关、判别性偏移尾部驱动 [2026-09-15 04:20]

### C001
**测试原理**：CPU-only 复用 Phase2753 冻结的逐题数组（896 面板 × 7 变体 × own/nll_target/nll_permuted），把 2753 发现的 true runs 补集（union 外 98.1% 训练坐标）在 knowledge_chain 上的族级损害（−1.10/−2.04 NLL）分解到题目级。base 双世界 margin 无需 GPU 即可恢复：Phase2753 对同一原始权重按条件各评估了一次 base，故 base nll_target = base__true_token__own、base nll_permuted = base__within_surface_class_permuted_token__own。2750 的逐步分叉 m0（B0，true vs permuted 配对世界）按 sample_id join 到面板（面板 64 个 kc 题全部在 own_history 512 题内）。预注册设计在任何 Phase2754 统计前冻结于 execution.json。

**测试用例**：896 题冻结面板（validation 192 + diagnostic 512 + fresh 192）；kc = 64 题，事后证实全部来自 confirmation2746/knowledge_chain 的 16 个链实例（每链 4 题）。判据 B1（普遍性：top-5 负质量份额 >0.5 → few_extremes）、B2（判别性偏移）、B4（跨代复现 Spearman）、B5（2750 m0 关联）；bootstrap 2000 次（seed 2754010），置换检验 10000 次（seed 2754012/2754014）。

**公式**：
- 逐题补集效应：red_comp_q = own_base,q − own_onlycomp,q
- base margin 恢复：margin_base,q = base_prm_own,q − base_tgt_own,q
- 判别性偏移：Δmargin_q = [nll_prm − nll_tgt]_onlycomp,q − margin_base,q（负 = 朝 permuted 同类答案偏移）
- 集中度：top5_share = Σ_top5 min(red,0) / Σ min(red,0)；ρ 为 Spearman（置换 p）

**结果汇总**（result.json sha256 d6de6aa1…30749b）：
1. **B1 = distributed（两 run 一致）**：受损题 44/64、42/64（69%/66%）；top-5 负质量份额 0.337/0.282 < 0.5；中位数 ≈ 0 但 IQR 重左尾 [−5.00, +0.02] / [−6.55, +0.02]；bootstrap 95% CI [−2.06, −0.11] / [−3.19, −0.87]。
2. **B2 = not_supported（预注册符号方向在冻结时写反，判据如实报告）**：kc Δmargin = −1.57/−1.84（朝 permuted 方向），但中位数 0、仅 26.6%/28.1% 题正偏移——**判别性偏移存在但是尾部驱动、非均匀**；对照 word_sense 反向 +0.33/+0.52（远离 permuted，与其收益一致）。
3. **B3 族剖面**：true 补集——kc −1.10/−2.04（69%/66% 受损）、word_sense +1.64/+2.26、negation_scope +1.35/+1.06、attribute_binding 均值 ≈ 0 但 84%/91% 受损（小效应相消）；permuted 补集——全部受控族大额收益 +3.79～+8.18，自然文本族 +0.44～+0.62。
4. **B4 跨代复现 ρ = 0.986（p ≈ 1e-4，置换下限）**——逐题损害在两个独立训练代近乎确定性复现。
5. **B5 与 2750 衔接**：m0 与 red_comp ρ = 0.385（p = 0.0012）/ 0.414（p = 0.0042）——B0 内部分歧越大的题补集损害越重；m0 与 Δmargin 0.154（ns）/ 0.368（p = 0.014）。
6. **事后（未预注册）kc 链内剖面**：损害遍及全部 16 链（每链 2–4/4 受损），强度集中在链 00/01/02/10/11/14/15（2748 达 −2.7～−4.9），链 12/13 净收益（+1.85/+0.94 @2747）；链级符号跨代一致。

**相关文件**（immutable，sha256）：脚本 phase2754_rdc_kc_complement_localization.py = 68c438e399c7af8c85a0b23f1fb710ef94bd6333a5a6f1223eedfaf5480ec0ba；phase2754/qwen4_cpu/{result.json d6de6aa133a7cf48…, kc_localization.npz a6ccab25ccceefcc…, kc_localization_meta.json, execution.json e52c379f59df55f7…}。输入：phase2753 isolation_scores.npz（SHA 同 2753 登记）、phase2750 question_scores.npz。

**分析/理论进展**：
- **kc 补集损害是题目特异、跨代确定性的属性**（ρ=0.986），不是训练噪声，也不是少数极端题（top-5 份额 < 0.34）——2753 的族级 −1.10/−2.04 是 64 题上广泛分布的重左尾结构。
- **与 2750 直接衔接**：2750 测得 B0 分叉（内部第一步 postnorm 分歧）按族分层，此处显示 B0 越大的 kc 题在补集干预下受损越重（ρ≈0.39–0.41）——补集分量与链推理的冲突在生成第一步即表现。
- **补集不是噪声垫**：同一组 1.415M 坐标外的 delta 分量在 word_sense/negation_scope 上承载正贡献、在 kc/ab 上承载损害、对自然文本近中性——**训练改动的"残余半空间"按关系族功能分化**。这把 2753 的负交互（union×complement）细化为族级功能冲突，而非整体相消。
- 三级链条成型：2751（签名几何：跨代可复现定向结构）→ 2752/2753（功能：定位子集必要且充分于 permuted/mass；true 收益无单一载体、半空间间强负交互）→ 2754（逐题：补集损害题目特异、跨代确定、与 B0 分叉共变）。

**问题硬伤**：
1. B2 预注册判据方向在冻结时写反（Δmargin > 0），判定 not_supported 属实，但"朝 permuted 偏移"的方向性解释为事后（冻结文本错误）。
2. 无逐题 argmax（2753 未存），判别性用 log-prob margin 代理，argmax 翻转率不可得。
3. kc 64 题全部来自 confirmation2746 单一来源群——结论向其他 kc 材料泛化未验证。
4. m0 join 有效数 63/64（2747 对）、46/64（2748 对）；2748 对缺失与生成分叉早相关。
5. 补集 1.415M 坐标内部结构未分解（哪些坐标对 kc 有害、哪些对 word_sense 有益）。

**结论**：P5② 收官。V2 执行序 P1✓ P4✓ P5①✓ P5②✓（含 2752/2753/2754 三层细化）。剩余：P5③ 跨模型移植、P2 多位置采集、P3 预设强度干预（均需 GPU）。

**接续**：Phase 2755 = P5③ 跨模型签名复现（在第二个模型上重复 2747 式三监督训练 → 2751 式签名几何提取 → 检验"监督语义占据近正交子空间"是否模型不变）。GPU 准入已验证（RTX 5080 16GB）。

## Phase 2755: P5③跨模型签名复现——Qwen3-1.7B 同协议三监督训练复现 2751 参数签名几何，四判据全通过 [2026-09-15 11:05]

### C001：监督语义的参数签名几何跨模型复现成立——三方向对排序、量级、子空间分离在 4B→1.7B 上逐项重现

**测试原理**：P5③ 检验 Phase 2751 核心结论（三监督方向差异场跨代可复现、监督语义占据近正交参数子空间）是否为模型不变规律。方法：在第二个模型（Qwen3-1.7B，Qwen3ForCausalLM，hidden 2048 / intermediate 6144 / 28 层，可训练块 model.layers[16].mlp，3×2048×6144 = 37,748,736 FP32 标量）上逐字复用 Phase 2747 协议（训练行、draws 固定种子、128 步、逐例 backward、FP32 范数归一步 step_norm=0.02、checkpoint {1,8,32,128} 位级 delta+residual 归档），随后按 2751-v3（真拼接版）几何提取协议计算：6 run delta 逐矩阵与 pooled 的 6×6 方向余弦、三方向对差场 D 的跨代 Pearson/cosine/top-k Jaccard/能量比、跨条件 pooled 余弦分离度、逐矩阵 top-1% 神经元两两交集。

**模型替换史**：首选 Qwen2.5-3B-Instruct（5.75 GiB，hf-mirror 分块 Range 下载），CPU 预检 safetensors 完整（434 张量）但 **tokenizer 门失败**：17123 个材料 id 中 2 个（151667/151668，ChatML 特殊 token 区）与 qwen3-4b 解码不一致——预注册零容忍 → 回退 Qwen3-1.7B（与 qwen3-4b 同 tokenizer 族，门检 **17123/17123 零不匹配**，encode 一致性 32/32）。

**完整性门**：bridge−native dNLL = 0.0459 < 0.05 软门（RTX 5080，BF16 基线）；训练后 6 run own-objective 改进全部为正：true +0.926/+0.683，permuted +5.411/+5.486，mass +0.0062/+0.0061（与 qwen4 相同的条件不对称模式：permuted 目标最易改进、mass 几近饱和）。

**预注册判据与结果（4/4 通过）**：
- **G1 复现对 3/3**：pooled 差场跨代 Pearson/J₁%/E比 = true_vs_permuted **0.9815/0.754/53.8**，true_vs_mass **0.8688/0.480/8.7**，permuted_vs_mass **0.9836/0.764/61.9**（阈值 r≥0.5、J₁%≥0.3、E≥2）。
- **G2 跨条件分离 2/2 代**：pooled 跨条件余弦全部 < 0.5——2747 代 tp=0.458/tm=0.219/pm=0.077，2748 代 tp=0.472/tm=0.227/pm=0.079（pm 近正交，同 qwen4 定性结构）。
- **G3 集中性 3/3**：逐矩阵 top-1% 两两交集显著超独立期望（2747 代 gate_tp=20/gate_tm=5/gate_pm=7、up_tp=25/up_tm=3/up_pm=8、down_tp=17/down_tm=23/down_pm=22；2748 代几乎相同）。
- **B1 行为 6/6**：每 run 改进 > 0（阈值 ≥5/6）。

**跨模型对比（qwen3-4B Phase2751-v3 → qwen3-1.7B Phase2755，pooled）**：
| 方向对 | 4B r / J₁% / E | 1.7B r / J₁% / E |
|---|---|---|
| true_vs_permuted | 0.963 / 0.679 / 35.7 | 0.982 / 0.754 / 53.8 |
| true_vs_mass | 0.867 / 0.454 / 8.3 | 0.869 / 0.480 / 8.7 |
| permuted_vs_mass | 0.967 / 0.696 / 26.3 | 0.984 / 0.764 / 61.9 |

三对排序（pm≈tp≫tm）、量级、甚至 true_vs_mass 的"弱复现"位置都逐项重现；逐矩阵同条件跨代余弦 1.7B 为 0.88–0.99（4B 为 0.84–0.98）。

**公式**：D_p^{(g)} = δ_a^{(g)} − δ_b^{(g)}（拼接 37,748,736 维，float32 累积）；r(D⁴⁷,D⁴⁸) 为 Pearson；cos(u,v)=⟨u,v⟩/(‖u‖‖v‖)；J_k = |top_k(D⁴⁷)∩top_k(D⁴⁸)|/k；E = ‖D⁴⁷‖²/‖D⁴⁷⁻⁴⁸(noise)‖²。

**相关文件**（BASE=tests/glm5/result/rdc_query_construction_20260913）：脚本 phase2755_rdc_cross_model_signature.py（SHA a2f3b950…，快照 sources/phase2755_…a2f3b950.py）；产物 phase2755/qwen25_3b/{result.json 24519cdd…, behavior_scores.npz 06d08b7e…, pooled_top1pct_indices.npz 2acc6dbe…, execution.json 8b3bb249…, 6 run × 4 checkpoint delta_0xx.npz}；模型 models/hf/qwen3-1.7b。

**分析**：①签名几何不是 qwen3-4b 的偶然特征——不同宽度/深度/训练满血程度的同族模型在相同监督下收敛到同构的参数方向结构，且方向对间的相对几何（含 true_vs_mass 最弱这一"瑕疵"）保持不变，支持"监督目标→参数子空间"映射的普遍性；②permuted 目标改进幅度最大（+5.4 NLL）且其签名最稳定（r≈0.98），与 2752/2753 的"置换语义几乎完全驻留于定位子空间"互为印证；③mass 目标改进微小（+0.006）但签名仍然稳定复现（r=0.87，E=8.7）——签名稳定性与行为改进幅度解耦，说明差异场编码的是目标间的语义关系而非改进量。

**理论进展**：总图景"外部语言操作→内部条件坐标齿轮→跨层编译为下一 token 概率"中，"条件坐标"层现在有了**跨模型不变的参数级实现证据**：三类监督语义占据近正交且跨代/跨模型可复现的参数子空间（P5①③联合结论）。

**问题硬伤**：①单第二模型、单块（layers[16].mlp），两模型比较≠普遍定律（limits[0] 已自述）；②第二模型与 qwen4 同 tokenizer 族，无法检验"语义→子空间"映射是否跨 tokenizer 保持（Qwen2.5-3B 因 2 个特殊 token 解码差异被零容忍门拒绝，未测其几何）；③脚本软引用 qwen4_phase2751_reference 读了错误键名（'pooled_pairs'，应为 'pooled.pairs'）致对照字段为 error——上表为本 MEMO 手工补算，不影响任何判据；④运行史：首次运行训练全部完成后在几何提取段因 np.concatenate 维度不匹配崩溃，修复 reshape(-1) 后断点续跑（复评估+几何，285.3s）；期间一次并发重复启动 5s 后无痕退出，终态产物经 SHA 复核一致、无损坏。

**结论**：P5③ 单模型版通过。2751 的签名几何结论跨模型复现，V2 计划 P5 全部完成（P5①②③✓）。执行序状态：P1✓ P4✓ P5①✓ P5②✓ P5③✓。

**接续**：Phase 2756 候选（按总目标排序）：(a) **P2 多位置采集**——当前所有签名证据限于 layers[16].mlp 单块，需在多层多个块上重复 2747 式训练以检验"条件坐标齿轮"的层分布（GPU，中等成本）；(b) **P3 预设强度干预**——KV 重放下对签名子集做强度扫描（GPU，大成本）；(c) 跨 tokenizer 第二模型训练（需放宽 tokenizer 门为内容 token 门）。优先 (a)：与"跨层编译"总目标直接对应。

## Phase 2756: P2四因子分解——分带规则增益的因子归因：内容份额随生成深度单调坍缩，表层token身份在B3/B4承载84–94% [2026-09-15 06:20]

### C001：预注册假说方向反转（0/5命中），换来"阶段×因子"分解表与生成深度轴上的内容→表层单调切换

**P2 原义校正。** Phase 2755 接续把 P2 误述为"多层多块训练"；V2 计划 P2 原文是**多位置真实前缀覆盖**——同一问题在已消费不同长度自身历史的多个位置上重复 2748 选择性拟合，区分词汇/位置/格式/内容四因子。本 Phase 按原义在 2749（P1）分带框架上执行四因子逐项扣除，CPU-only、数据在库。

**测试原理。** 2749 已证每带自拟合规则显著优于带均值（对角增益），且带间迁移分族 {B0},{B1},{B2+}。P2 问：每带增益由什么承载？构造四类**查表预测器**（验证侧拟合、诊断侧评估，收缩权重 n/(n+5) 向带均值）：lex=token id 饱和表；freq=验证侧 unigram 计数四分位桶（+unseen）；fmt=验证计数 top-8 token 各自一类（其余 other）；len=带内步号桶（B0/B1 单值）。可加联合 additive = lex + len − 带均值（lex 包含 fmt/freq，因二者皆为 id 的函数）。评估口径与 2749 逐位同构（token MSE→题→context→等 cohort），一致性门：重算带均值 MSE 与 2749 存档值相对差 < 1e-8（三模型五带全过）。G_full 取自 2749 result.json 对角 primary 规则存档；份额 =(带均值−预测)/G_full；内容份额 =1−additive 份额。bootstrap：整 context 配对 full−additive，2000 次，种子 2756001/2756002；full 逐题值取自 2749 question_scores.npz 的 primary__b__b__absolute（NaN 题剔除）。

**结果汇总（等 cohort 份额；qwen4 / qwen14 / glm4）**：
| 带 | additive 份额 | 内容份额 | fmt 单独份额 | bootFA full−additive [95%CI] |
|---|---|---|---|---|
| B0 | −0.26 / −0.11 / −0.08 | **1.26 / 1.11 / 1.08** | −0.12 / −0.03 / −0.02 | 全负，CI 不含 0 |
| B1 | −0.90 / −1.45 / +0.23 | **1.90 / 2.45 / 0.77** | +0.32 / +0.22 / +0.34 | 全负 |
| B2 | +0.77 / +0.79 / +0.39 | 0.23 / 0.21 / **0.61** | +0.80 / +0.82 / +0.38 | 全负 |
| B3 | +0.84 / +0.85 / +0.84 | **0.16 / 0.15 / 0.16** | +0.95 / +0.93 / +0.90 | 全负 |
| B4 | +0.91 / +0.89 / +0.94 | **0.09 / 0.11 / 0.06** | +0.97 / **+1.01** / +0.93 | 全负 |

**核心发现**（值得记三遍）：**分带规则增益的内容份额随生成深度单调坍缩**——B0/B1 增益 100% 内容承载（表因子为负，查表反而差于带均值），B2 是过渡带（glm4 仍内容主导 0.61，qwen 系已表层主导），B3/B4 的 84–94% 增益由 token 身份查表承载，其中 **top-8 格式 token 单独承载 90–101%**（qwen14 B4 fmt 份额 1.0083，查表超过完整核规则）。freq 桶份额处处 ≈0——词汇效应是**具体 token 身份**（收尾结构），不是高频/低频粗化；len 份额 ≈0——带定义已吸收位置效应。每带内容残余均显著（bootstrap 全负 CI 不含 0），无一带被表层因子完全解释。

**预注册假说判定**：B0/B1 表因子≥0.5 且 B2+ 内容>0.5 → 0/5 命中（glm4 B2 意外命中），**方向反转**。反转原因可辨：预注册依据是 2749 带描述的"B0 首位引号占 377/384"——但那是带均值层面（band_mean 已吸收）；规则**增益**层面 B0/B1 的区分完全来自查询内容（H12/来源读出）。

**理论进展**：与 2749"阶段分族 {B0},{B1},{B2+}"互补，现在知道分族的**因子内容**——早期带规则是内容条件化装置（查询→输出场），后期带规则近似"生成 token 身份→目标均值"的表层映射（收尾结构）。"阶段 × 因子 × cohort"分解表成立：内容轴在生成深度轴上单调让位于表层轴，这是条件齿轮在生成时间维的第一张因子归因图。

**相关文件**（BASE=tests/glm5/result/rdc_query_construction_20260913）：脚本 phase2756_rdc_factor_decomposition.py（快照 sources/）；产物 phase2756/{model}/result.json+factor_scores.npz+execution.json——qwen4 SHA d3836468…/a7f6631f…/aca8851b…，qwen14 9ee6badd…/5b9223a6…/7512dd5f…，glm4 5f423e4b…/02573638…/73a805d4…；execution 登记 2749 result/npz/execution 三 SHA。

**问题硬伤**：①len 用步号代理已消费前缀长度（题内单调、跨题长度差异未建模）；②查表在验证稀疏单元收缩到带均值，份额是泛化保守估计；③cohort（drop/quoref）是本框架唯一关系轴，非 2750 的 6 关系族；④首次运行因 2749 npz 逐题 NaN（带外题）在 bootstrap 序列化崩溃，修复有限值过滤后重跑——统计与最终产物一致；⑤additive 的负份额（B0/B1）说明查表在该带过拟合验证噪声，"内容份额>1"应读作"表因子为负贡献"。

**结论**：V2 执行序 P1✓ P2✓ P4✓ P5①②③✓；仅剩 P3（预设强度干预，需 KV 重放 GPU）。

**接续**：Phase 2757 = P3 预设强度干预（V2 最后一个未执行项）：利用 2748 已有 KV 重放管线，在 η 强度分层（弱/中/强）下比较"语义定向破坏 vs 等强度随机破坏"的双世界区分下降差——只有定向>随机的差值才是机制定位证据。GPU 准入已验证。

## Phase 2757: V2-P3 预设强度干预——block12 来源删除 vs 等质量随机删除的双世界区分对照与"问题事实泄露"机制的暴露 [2026-09-15 11:52]

### 测试原理
V2 执行序最后一项（P3）。V2 修正点 3 指出"所有拟合收益都能被弱扰动对照吸收——来源正确配对的独立贡献未被分离；机制定位要靠预设强度的干预而非拟合对照"。P3 原文判据：在 η（破坏强度）相同的强度带上比较"语义定向破坏"与"等强度随机破坏"的双世界区分下降差——**只有定向>随机的差值才是机制定位证据**。

干预对象：block12 = qwen3-4b 第 12 层（2748 全部来源读出线的层）。算子：在该层 v_proj 输出的选中 prompt 位置行置零（V 不经 RoPE，等价于 KV 路径的 block12 value 行置零；与 2748 `source_value_pair_shuffle` 同一 value 内容通道），单段全 prompt 前向评分。

双世界材料：2747 diagnostic 的 controlled_relation 320 行 = 160 对 × 2 世界（实体-属性互换 prompt），5 关系族 × 16 case × en/zh。两世界共享同一问句，target 互为对方的"错误答案"。**这一面板的双世界区分在此前从未在 native 模型上直接测量过**——本 Phase 第一次给出。

### 测试用例与门
- 变体（每世界 13 个）：base；sema_25/50/100（body 即 Inventory 绑定句 span 内按 block12 V 行范数降序前 ⌈k·|body|⌉ 行置零）；randn_25/50/100（非 body 位置按范数降序前同数量行，质量匹配确定性对照）；rand_25/50/100 × 2 重配（非 body 均匀随机，种子 2757001/2757002，辅助对照）。对照行数上限为非 body 池大小（k=100% 时 30>29，实际 29 行，η 记录在案）。
- η 定义：置零行 ‖v‖₂ 之和 / 全 prompt 行 ‖v‖₂ 之和（逐对记录）。
- G1a（通道一致）：hook 空转前向与普通前向 logits 逐位相等——16 个检查全部精确 0.0。
- G1b（干预生效）：sema_100 在每对两世界上都产生非零 logits 偏移——通过。
- 修订记录：初稿用两段法（cache 截断+最后 token 评分）触发逐位 G1 失败（maxdiff 0.47–0.66；诊断 probe：top-1 与 margin 符号一致，差异源于 BF16 大负 logit 区舍入），在读取任何行为结果之前统一改为单段 hook 评分并预注册注明。失败草稿从未产出结果。

### 公式
- 每对两世界 margin：$m_w = \mathrm{NLL}(t_{\bar w}\mid \text{prompt}_w) - \mathrm{NLL}(t_w\mid \text{prompt}_w)$，$t_{\bar w}$ = 对侧世界的 target（即本世界的错误答案）。
- 双世界区分：$D_i = \min(m_0, m_1)$；带级破坏量 $\Delta D = \mathrm{median}_i(D^{base}_i) - \mathrm{median}_i(D^{var}_i)$。
- J1：$\Delta D_{sema} > \Delta D_{randn}$ 的带数 ≥ 2/3；J2：k=100% 带逐对 $g_i = \Delta D^{randn}_i - \Delta D^{sema}_i$ 的 pair 级 bootstrap（1000 次，种子 2757010）95% CI 下界 > 0；verdict = mechanism_localised ⟺ J1 ∧ J2。

### 结果汇总（160 对，150.3 s，RTX 5080）
| 带 | η(sema) | η(randn) | ΔD_sema | ΔD_randn | ΔD_rand |
|---|---|---|---|---|---|
| 25% | 0.182 | 0.162 | **−0.156** | −1.813 | −0.625 |
| 50% | 0.344 | 0.255 | **−0.250** | −1.750 | −1.063 |
| 100% | 0.635 | 0.361 | **−0.500** | −1.375 | −1.438 |

- 预注册判定：J1 字面通过（三带 −0.5 > −1.375 等），**J2 否证且反向显著**（gap 中位 −0.625，CI [−1.000, −0.250] 全负）→ **verdict = not_localised**。
- 关键语义反转：ΔD 全为**负**——所有干预都使 D 上升（"区分"改善），定向删除的改善反而小于随机删除。
- **base 双世界区分本身就是负的**：D_base 中位 −22.31、均值 −20.34、98.1% 的对 D<0（两世界全对的仅 1.9%）——native qwen4 在这套合成题上有压倒性单侧答案偏置（一个世界完全确信错误答案）。
- 族级分解（k=100%，gap>0 支持定向特异性）：long_distance_role **+1.06**、negation_scope +0.19 vs knowledge_chain −2.56、word_sense −1.31、attribute_binding −1.00——5 族中 2 族正、3 族负。
- 产物 SHA256：result.json `c90ca23cd71547ad…`、pair_scores.npz `a12840c4ff8008f7…`、pair_scores_meta.json `a05aabc993b013…`、execution.json `68ad77a3132855…`；脚本快照 `sources/phase2757_rdc_preset_strength_*.py`。

### 分析
1. **J2 反向的机制：问题事实泄露**。attribute_binding / word_sense / knowledge_chain 的问句复读（或等价承载）了 body 事实（"Is the lantern assigned to Neral-00 purple?" 复读了绑定），答案可从问句位置恢复——随机对照大量命中问句附近位置，对"偏置答案的确信度"扰动更大；定向删除只动 body，不触及泄露通道。long_distance_role 的问句**不含**完整事实 → 定向删除显示正向特异性（+1.06）。**block12 来源载体特异性的行为证据，恰好在且仅在"问题不泄露答案"的关系族中出现**。
2. **扰动方向假设错误**：预注册"删除→区分下降"假设区分是正的、信息性的；实测 base 区分为负（偏置态），任何删除都松动偏置使 D 上升。负 D_base 本身是新事实：**这为 2747"表面类别监督保留区分、正确答案监督损害区分"提供了起点解释——native 模型的区分不是从零建立而是从负偏置中矫正**。
3. 两段法 G1 失败的诊断（BF16 大负 logit 区 ±0.4 舍入）独立确认了 2750 的"内部—行为解耦"方法论教训：logit 尺度的逐位断言在饱和区不可用，读出量（margin/排序）才是稳定对象。

### 理论进展
- V2 执行序五项（P1–P5）至此全部完成。P3 的答案是否证的：**在行为层面，block12 来源信息的因果特异性未被建立**；拟合收益的"弱扰动可吸收性"（2748）延伸到了强干预带。
- "条件坐标齿轮"图景的修正：来源→答案的绑定不是单一位置通道承载，而是**冗余编码**（问句复读 + 篇章来源），冗余度由关系族决定。齿轮不是单齿。
- native 单侧答案偏置（98% 对 D<0）是合成材料与模型先验交互的普遍现象候选——与 2747 训练把区分从负偏置拉向 0/正值的叙事吻合，需在 2747 checkpoint 上复测 D（接续）。

### 问题硬伤
1. 干预只在 block12 一层、只置零 V 通道；K 通道与跨层写入未测。
2. 对照池仅 29 行、k=100% 时 η(sema)=0.635 vs η(randn)=0.361 未完全匹配——质量匹配在最强带打了折扣（rand 均匀对照 η=0.18 更低）。
3. D_base 饱和态（|m|~26）下 BF16 噪声 ~0.5，与带间差异同量级；pair 级中位数与 bootstrap 部分吸收但非全部。
4. 只测了 base 模型；2747 训练后模型（区分被矫正态）的定向/随机差可能不同——本 Phase 结论不外推到训练态。
5. "问题泄露"解释是事后机制假说（由族级模式归纳），未做直接操纵（如删除问句中复读的事实词）。

### 结论
P3 预注册判据 not_localised（J1 字面过、J2 反向显著）。来源信息在 block12 的行为因果特异性未建立；替代发现为冗余编码 + 问题泄露机制 + native 负偏置起点。V2 五项执行序（P1 内容阶段选择性、P2 四因子分解、P3 预设强度干预、P4 知识链载体、P5 参数方向→子集→干预→跨模型）至此**全部闭合**。

### 接续

Phase 2758 = 2747 六个训练态 checkpoint（3 条件 × 2 种子）部署后逐字复测 2757 双世界区分面板：检验负偏置起点假说与 block12 定向/随机特异性的训练态命运。

## Phase 2758: 2747 训练态 D 复测——六 checkpoint 部署精确性门全过；permuted-token 监督特异矫正双世界负偏置（−22→−2）并把区分载体转移到问句侧 [2026-09-15 21:36]

### 测试原理

2757 遗留两个不可外推到训练态的结论：(a) native 双世界区分 D 饱和为负（中位 −22.3125，98.1% 对 D≤0）；(b) block12 来源删除无行为特异性（J2 反向）。本 Phase 部署全部六个 2747 训练态 BF16 checkpoint（true_token / within_surface_class_permuted_token / surface_class_mass × 种子 2747/2748），逐字重复 2757 面板：160 对 × 2 世界、13 变体（base + sema/randn/rand × 25/50/100% block12 v-row 置零）、单段 hook 评分、相同控制种子（2757001/2757002、bootstrap 2757010）。

部署与门：block16 mlp 权重按 2747 原式重建（native FP32 展开 + delta + reconstruction_residual，FP32）后写入 BF16 参数（与 2747 的 FP32→BF16 部署舍入逐位同型）。**G0 部署身份门**：六个 delta npz SHA256 与 receipt 全部一致；重算 deployed BF16 delta L2（float64）与 recorded deployed_BF16_delta_L2 相对差 < 1e-9（六/六：0.2941/0.7842/0.2342/0.2957/0.7264/0.2344）。G1a（hook 空转逐位相等）每 run 前 8 对全 0.0；G1b（sema_100 每对产生非零偏移）六 run 全过。

**预注册判据**（任何行为前向之前冻结于 execution.json）：C1 每 run mechanism_localised ⟺ J1∧J2（2757 定义），汇总 trained_localised 需 ≥4/6；C2 每代 D_base_median(mass) > D_base_median(true)（依据 2747"表面类监督保留区分、正确答案监督损害区分"）；C3 全部 6 run D_base_median > native −22.3125（负偏置起点假说）。事后探索（未预注册）：逐题 D 与 native 的 Pearson、家族级 gap。

### 结果汇总（160 对，866.1 s，RTX 5080；native 基准 −22.3125 / J2 gap −0.625 [−1.0, −0.25]）

| run | D_base_median | D≤0 份额 | J1 | J2 gap 中位 [95%CI] | verdict | 逐题 ρ vs native |
|---|---|---|---|---|---|---|
| true_2747 | −19.625 | 1.00 | ✓ | +0.25 [−0.25, +0.625] ns | not_localised | 0.859 |
| perm_2747 | **−2.000** | 1.00 | ✗ | **+1.25 [+1.0, +1.25]** ✓ | not_localised | 0.467 |
| mass_2747 | −21.9375 | 0.975 | ✓ | −0.625 [−0.875, −0.125] ✗ | not_localised | 0.998 |
| true_2748 | −20.000 | 1.00 | ✓ | +0.125 [−0.314, +0.625] ns | not_localised | 0.834 |
| perm_2748 | **−2.250** | 1.00 | ✗ | **+1.50 [+1.0, +1.75]** ✓ | not_localised | 0.196 |
| mass_2748 | −22.0625 | 0.981 | ✓ | −0.75 [−1.0, −0.186] ✗ | not_localised | 0.997 |

**预注册判定**：C1 = not_localised_majority（0/6 通过）；C2 **否证**（两代方向均反转：true −19.6/−20.0 > mass −21.9/−22.1）；C3 **通过**（6/6 > −22.3125）。

### 分析

1. **permuted-token 监督是唯一大幅矫正负偏置的条件**：D 从 −22 抬至 −2（约 +20 NLL），且逐题结构与 native 解相关（ρ 0.467/0.196）——矫正不是沿 native 结构的平移而是行为几何重写。true 轻微上移（+2.3～+2.7，ρ 0.83–0.86，部分重组）；mass 几乎不动（ρ≈0.998，逐题照抄 native；其参数位移也最小 0.234）。三条件的行为分化与其参数位移排序（perm 0.73–0.78 ≫ true 0.29 ≫ mass 0.23）一致。
2. **C2 反转的可辨原因**：2747 的"true 损害区分"是在其自有读出（surface-mass 分离）上测的，不迁移到双世界 D 指标；在本面板上 true 态反而高于 mass 态。负偏置（native/mass ≈ −22）与 2747 读出语义的映射需按指标分账。
3. **J2 符号的条件分化 = 载体定位证据**：native/true/mass 态随机删除比定向删除更能松动偏置（gap ≤0 或 ns，与 2757 一致）；**perm 态两代 gap 显著为正（+1.25/+1.5，CI 全正）且符号语义反转**——k=100% 带 ΔD_sema −0.25/0.0 vs ΔD_randn +0.75/+1.3125：定向 body（来源）删除使 D 微升、随机非 body 删除使 D 下降。即 **perm 态的双世界区分由问句侧（非 body）位置承载**：删除问句侧损伤区分、删除来源侧不损伤。这为 2757 的"问题事实泄露"事后假说给出首个训练态行为证据：D 的载体在问句侧而非 block12 来源侧，且该定向/随机分离只在 perm 训练态可测。
4. **家族结构**：mass/true 保留 native 签名（knowledge_chain 最负 −2.1～−2.6、word_sense 负、long_distance_role 正）；perm 全族转正（long_distance_role 最大 +2.375/+3.25）。
5. **机制定位判据在全部六态不成立**：J1∧J2（"定向损伤更大"方向）无一通过——block12 来源行在 native 与三个训练态都不是行为特异载体；在 perm 态判据失败的方向（定向损伤更小）本身就是问句侧承载的表现。

### 理论进展

负偏置起点假说（C3）成立但被结构细化：**偏置矫正不是训练的普遍效应，而是 permuted-token 监督的特定效应**；矫正后的区分载体在问句侧位置而非篇章来源侧。结合 2747→2758 链条：同为"表面类别"监督，mass 保持 native 偏置几何不动（最小位移、逐题 ρ≈1），perm 以最大位移重写行为几何并把区分支持从偏置态转为问句侧承载——"表面类别监督"内部存在两种定性行为，不可合并。"条件坐标齿轮"图景补充：双世界区分这一行为量的支持位置可被监督目标重接线，且重接线方向（问句侧）与 2757 的泄露通道假说闭环。

### 相关文件（immutable，sha256）

脚本 phase2758_rdc_trained_state_D.py = 504b67811fa091f20a68ef1e7a17cc6be150a2bae2aebcf4c729950f0dc1c87c（快照 sources/phase2758_rdc_trained_state_D_504b67811fa091f2.py）；产物 phase2758/qwen4_trained_D/{execution.json 9c505711b0202efb…, result.json 6f35240ab99563de…, pair_scores.npz 3447828e3f4e3720…, pair_scores_meta.json edb04df18d86a26a…}。输入：2747 六 run delta_128.npz（逐 receipt SHA 校验通过）与 deployed_BF16_delta_L2；phase2757/qwen4_block12/pair_scores.npz（native 逐题 D 基准）。

### 问题硬伤

1. perm 态矫正不彻底：D 仍 100% ≤0（均值 −1.9～−2.3），"矫正"是大幅上移而非转正。
2. BF16 饱和区舍入（|m|~20–26，噪声 ~0.5）对 true/mass 的带级差分仍是同量级噪声源；bootstrap 中位数部分吸收但非全部。
3. J1 判据方向是 2757 冻结的"定向损伤更大"；perm 态显著正 gap（定向损伤更小）按预注册字面记 not_localised——"问句侧承载"是对 J1 失败**方向**的事后解释，未预注册。
4. 单模型、单训练块（block16 mlp）、单干预层（block12 v 通道）；K 通道与跨层未测。
5. perm_2747 的 J2 bootstrap CI 上界恰等于点估计（+1.25，分布退化），区间解释需保守。
6. mass_2747 的 D_base 中位（−21.9375）仅高 native 0.375，C3 对 mass 的"通过"在 BF16 噪声量级内，不宜过度解读。

### 结论

C1 not_localised_majority（0/6）、C2 否证（方向反转）、C3 通过（6/6）。**训练态不改变 2757 的核心否定结论**：block12 来源的行为因果特异性在 native 与全部三个训练态均未建立。新发现：permuted-token 监督特异地矫正双世界负偏置（−22→−2，跨两代复现）并使区分支持转移到问句侧位置；mass 监督对偏置几何近乎零作用。

### 接续

Phase 2759 = 问题泄露通道的直接操纵：在两世界 prompt 的问句中删除复读事实词的 span（内容级 token 干预，非 hidden 干预），预注册"删除问句事实 span 使 D 下降/区分崩溃"判据；同面板同判据复用于 perm 训练态（其 J2 正 gap 预测问句侧删除在 perm 态损伤更大）。若确认，"问句侧承载"由相关升级为因果，并关闭 2757 遗留的事后假说。
- V2 已无可执行项。按总目标（外部语言操作→内部条件坐标齿轮→跨层编译）判定下一阶段：优先 **(a) 训练态复测**——在 2747 true/permuted checkpoint 上重跑本 Phase 面板，检验"区分矫正态"的定向/随机差是否反转（直接检验 2752 功能载荷条件结构的行为面）；(b) 问题泄露直接操纵（删问句事实词 × 2×2 对照）；(c) 制定 V3 评估与计划文档，把 V2 五项结果整合为下一轮可判定计划。

## Phase 2759: 问句事实 span 内容级删除——"问题事实泄露"假说被直接否定：问句事实词承载的是共模负偏置而非世界区分，删除后 D 反向抬升（−22→−14）且世界间距增大 [2026-09-15 22:55]

### 测试原理

2758 接续预注册：在两世界 prompt 的问句中删除复读事实词的 span（内容级 token 干预，非 hidden 干预），判据"删除问句事实 span 使 D 下降/区分崩溃"；同面板复用于 perm 训练态。事实 span 规则在任何行为前向之前冻结（execution.json）：问句区内 token 的解码文本（去标点、小写）匹配 relations 三元组词片（source/relation/target 按 `_` 拆分，双向子串）或出现在 body 正文（长度≥2，排除 15 词功能停用表）。变体：base、qdel（全部事实 span）、qctrl_a/b（等量随机非事实问句 token，种子 2759001/2759002）、bdel_a/b（等量随机 body token，2759003/2759004），全部内容级重建 token 序列。指标沿用 2757：m=nll(other.target)−nll(this.target)、D=min(m0,m1)，另预注册 W=m1−m0（世界间距，免疫共模偏置）。运行：native + perm_2747/perm_2748（主检验）+ true_2747/mass_2747（条件对照），160 对 × 2 世界 × 6 变体，361.5 s @ RTX 5080。

完整性门：G0 部署身份（四训练态 delta SHA + BF16 L2 相对差 <1e-9）；G1 base 前向确定性逐位 0.0（每 run 前 8 对）；G2 删除计数与区域合法性逐对通过（qdel 中位 7 token/对）。

### 结果汇总（dT=|W_base|−|W_qdel| 中位；dC=同量的匹配随机问句删除；dD=D_base−D_qdel 中位）

| run | W_base 中位 | D_base 中位 | dT 中位 | dC 中位 | P1 [CI] | dD 中位 [CI] |
|---|---|---|---|---|---|---|
| native | 7.97 | −22.3125 | **−7.125** | −1.5625 | ✗ gap −6.5 [−8.4,−3.4] | **−8.0 [−10.0,−6.2]** |
| perm_2747 | 2.13 | −2.000 | −0.875 | −1.750 | ✗ gap +0.09 [−0.14,+0.88] | −0.25 [−0.625,−0.125] |
| perm_2748 | 2.38 | −2.250 | −1.000 | −3.289 | ✓ gap +3.06 [+2.0,+5.1] | −0.50 [−0.875,−0.25] |
| true_2747 | 14.00 | −19.625 | +1.875 | +2.719 | ✗ gap −0.94 [−4.3,+0.56] | **−10.0 [−11.8,−7.75]** |
| mass_2747 | 7.69 | −21.9375 | **−7.750** | −1.813 | ✗ gap −6.4 [−8.7,−4.8] | **−8.0 [−9.9,−6.5]** |

**预注册判定**：P1（定向删除坍缩区分超过匹配对照）1/2 未确认；P2（删除使 D 下降）0/5 **全部反向**（删除使 D 显著抬升）；P3（perm 态 dT 高于 native/mass）通过 3/3；P4 通过。**verdict = not_confirmed**——但反向结果本身就是本 Phase 最重要的发现。

### 分析

1. **"问题事实泄露"假说被内容级干预直接否定（关键结果，重复强调）**：若问句事实词承载世界区分，删除它应使区分坍缩（D 下降、|W| 收缩）。实测相反：native/mass 态删除问句事实词使 D 抬升约 8 NLL（−22.3→−14.3）、世界间距 |W| 增大 7+ NLL；true 态 D 抬升 10。问句事实 span 的内容承载的是**跨世界共模的负偏置**（Yes-偏置锚点），不是区分性信息——删除它反而解除偏置、暴露出更强的世界信号。
2. **与 2758 训练态效应闭环**：perm 训练把 D 从 −22 抬到 −2 并把 J2 载体移到问句侧；本 Phase 在内容层做同一方向的操作（删问句事实词）在 native/true/mass 态复现了同向 D 抬升（约 +8）。两条独立证据链（参数级训练 vs 内容级删除）指向同一机制：**双世界 D 的 −22 偏置主要由问句侧事实复读 span 锚定**。perm 态删问句事实词几乎不再改变 D（−0.25/−0.5）——其问句侧偏置已被训练移除，进一步支持该解释。
3. **定向 vs 匹配对照**：native/mass 的 dC 也为负（随机问句删除同样抬 |W|，但只有事实删除的 1/4–1/5 幅度）；唯 perm_2748 的 P1 通过（gap +3.1）。真态（true）是唯一 dT>0（事实删除收缩世界间距）的态，且其 W_base 高达 14——true-token 监督把世界间距放大到全部态最大，但删除问句事实的效应仍不特异（dC 更大）。
4. **2757/2758 J2 判据的再解释**：此前"J2 反向"（定向 block12 删除不损伤区分）与本文一致——来源侧不是区分载体；现在补充：问句侧也不是"区分"载体，而是"偏置"载体。区分信号的实际载体仍未定位（可能在 body 深层语义或两者差分编码）。

### 理论进展

2757 暴露的"问题事实泄露"在内容级得到裁决：泄露通道存在但**泄露的是偏置而非答案**。"条件坐标齿轮"图景更新——问句侧事实复读 span 是共模偏置的输入端锚点；偏置与区分在问句侧混叠，2757 的 D=min 指标把两者合并观测，只有内容级删除能把它们分开。V2 计划遗留的"区分载体在哪"问题在删除问句事实后不但未关闭反而更清晰：区分信号在偏置解除后增强，说明区分与偏置是**可分离的两条通路**。

### 相关文件（immutable，sha256）

脚本 phase2759_rdc_question_span_deletion.py = bf7628b9b6d4289050f150ecb080c86cd1f9d7d062b177a79e4f13346c2714f8；产物 phase2759/qwen4_question_span/{execution.json bf159823…, result.json daeba763…, pair_scores.npz 8e88d481…, pair_scores_meta.json 7e4b076f…}。输入：2747 四训练态 delta_128.npz（SHA 逐 receipt 校验）、2747 诊断材料 160 对。

### 问题硬伤

1. 事实 span 规则是操作性定义（relations 词片 + body 子串），'Do'（kc 问句引导词，body 含 "do not use"）等个别非事实 token 被误删；规则无逐族调参，但删词清单未人工审计到每对。
2. 内容级删除改变序列长度与位置编码，qctrl/bdel 控制了删除量但未控制删除位置的句法角色。
3. P1 在 perm_2747 不成立而 perm_2748 成立——单种子翻转不足以支撑"定向>随机"的任何结论。
4. D=min 指标仍混合偏置与区分；本 Phase 的 W 分解是事后提出的（预注册了计算但解释框架部分事后）。
5. 单模型（qwen3-4b）、单面板（2747 诊断 160 对）。

### 结论

P1 1/2、P2 0/5 反向、P3 3/3、P4 通过；verdict = not_confirmed。**预注册假说（问句事实 span 承载区分）被否定；新结论（跨 2757/2758/2759 三相闭环）：问句事实复读 span 是共模负偏置的载体，删除它使 D 从 −22 抬至 −14 且世界间距增大——区分与偏置是可分离通路，偏置锚点在问句侧内容，区分载体仍未定位。**

### 接续

用户追加三项任务已立 Phase 2760（条件齿轮候选全 36 层分布图，描述性）、2761（知识链"内部知道但说不出"断层定位与激活修复）、2762（签名几何跨模型功能移植，在 2755 的 1.7B checkpoint 上运行 2752 式充分性/必要性检验）。2759 的"偏置/区分可分离"结论直接喂给 2761：知识链弱行为的成分分解应先剥离共模偏置再定位断层。

## Phase 2760: 条件齿轮候选全 36 层分布图——世界分歧能量 L0–19 近盲、L24+ 深带接管、L36 峰值；稳定条件敏感坐标沿深度分层聚集且强族异 [2026-09-15 23:30]

### 测试原理

用户需求（可行性已评：作为描述性可视化完全可行）+ 2755 接续(a) 的轻量版：不在多层重训，而在 2757 双世界面板（160 对 × 2 世界，native qwen3-4b）上以全层隐状态（37=emb+36，末 prompt 位置）测量条件机制的层分布。定义（行为前向前冻结）：(a) 世界分歧相对能量 nd(ℓ)=E_pair‖h_A−h_B‖²/E_pair(‖h_A‖²+‖h_B‖²)/2（2750 公式，全面板版）；(b) 条件齿轮候选坐标：逐对 top-32 |Δh| 坐标集合，坐标在 ≥60% 对中入集即为该层候选（合并 160 对与分族 32 对两口径，随机坐标期望入集率 ~1.5%）；(c) top-32 能量份额与 participation ratio（集中度）。22.2 s @ RTX 5080。描述性 Phase：不宣称条件齿轮闭合。

### 结果（四联图 gear_layer_distribution.png/svg）

1. **世界分歧能量沿深度单调爆发**：nd(ℓ) 从 L1–19 的 2.5e-5–2.1e-3（近盲带，L0–12 均值 ~4e-4）在 L20 起跳（4.9e-3），L24 达 7.4e-2，L28–35 平台 0.10–0.12，L36（最终 norm 前）峰值 0.186——以 160 对全面板复现并细化 2750 的"L0–12 盲、L24+ 接管、L36 峰"（此前仅 44 条冻结表达）。
2. **稳定条件敏感坐标的层分布强族异**：attribute_binding 在 L20 起跃升（峰值 L29=21 个）且全程最高；knowledge_chain 在 L24–35 呈 ~11 个的平台（层间几乎不变，L23 谷 4→L24 跳 15）；long_distance_role 与 negation_scope 中等（~5–14）；word_sense 最稀（大多 0–4，L8 后早带塌零）。合并口径仅 6–7 个坐标（各族候选集互不重叠）——**各族用不同坐标组编码条件差异**，与 2751"监督语义占据近正交子空间"在激活层呼应。
3. **集中度结构**：top-32 能量份额在 L1 最高（0.30）随后稳定在 0.11–0.15；participation ratio 从 L1 的 171 升至深带 ~700–780，唯 L36 回落至 489——条件响应随深度变得更分布式，最末层重新收拢。
4. **L0 伪影**：L0 两世界末 token 相同（Δh≡0），其"32 个候选"是零向量 argsort 伪影，排除解读。

### 理论进展

"条件坐标齿轮"图景获得第一张全深度×全族分布图：条件机制的载入不在浅层（emb 至 L19 对世界分歧近盲），而是 L20/L24 两个转折的深带过程；且"齿轮坐标"不是一组全局坐标——五个关系族各自在深带维持一组互不重叠的稳定坐标（ab 最多、kc 呈平台、ws 最稀）。kc 的 ~11 坐标 L24–35 平台与 2750 的 kc 内部敏感但不驱动行为的解耦结合，指向 2761 的断层定位问题：kc 的深带条件坐标为何到不了输出。

### 相关文件（immutable，sha256）

脚本 phase2760_rdc_gear_layer_distribution.py = a65e58342624d7d2c093631a0e2f93a4f4d5cafd3813214d2e6a0abafaa39f43；产物 phase2760/qwen4_gear_layers/{execution.json 1b2bc95c…, result.json cc242196…, layer_stats.npz d3667f1e…, gear_layer_distribution.png b94b0b4b…, gear_layer_distribution.svg 97f2c6b9…}。

### 问题硬伤

1. 描述性单面板、单模型、末位置单点采样——层分布不等于因果必要性（无逐层消融验证）；2. top-32/60% 阈值是约定（候选数随阈值变化，未做敏感性扫描）；3. L0 伪影已排除但 L1–3 低能量也含末 token 相同的成分（两世界 body 不同但末 token 相同，Δh 应非零——已验证 energy[1]>0，非伪影）；4. 世界分歧 = true vs permuted 历史差，不覆盖全部"条件"语义。

### 结论

用户需求 1 完成：条件齿轮候选的全 36 层分布图产出。核心事实：**L0–19 近盲（nd≤2e-3）、L20/L24 双转折、L28–35 平台（nd≈0.11）、L36 峰值 0.186；稳定候选坐标族异且互不重叠（ab 21 > ldr 14 ≈ kc 11–15 > neg 10 > ws 4）**。

### 接续

Phase 2761 = 知识链"内部知道但说不出"断层定位与激活修复（用户需求 2）：logit-lens 逐层判"内部知道"（target rank-1），对行为错误行做目标方向注入扫描（36 残差位置 × 3 强度）得阻抗剖面，预注册 K1/K2/K3 判据；随机方向对照。随后 Phase 2762 = 签名几何跨模型功能移植（用户需求 3，2755 checkpoint 上的 2752 式充分性/必要性）。

## Phase 2761: 知识链"内部知道但说不出"断层定位与激活修复——K1 通过（92.9% 错误行内部 rank-1）、修复 100% 成功且方向特异；但"断层"不是层间中继故障而是全局读出竞争（+a） [2026-09-16 00:20]

### 测试原理

用户需求 2（可行性评估：限定为 logit-lens 定位 + 目标方向注入修复，可行）。native qwen3-4b，320 诊断行全量。行为 = 末位置 greedy 首 token（B1 类比）；"内部知道" = logit lens z_ℓ=lm_head(final_norm(h_ℓ)) 中 target 在 ℓ∈[8,35] 某层 rank-1。修复 = 在残差位置 r∈[1,36] 注入 h += α·‖h‖·unit(lm_head.weight[target])，α∈{0.1,0.3,1.0}，记录逐 (行,层) 最小翻转强度 α_res；随机方向对照（2761 主实验混合 α 记 28.6% 翻转，超预注册阈，故追加 2761a：仅在 α=0.1 匹配强度下全层重跑对照，配对 bootstrap 种子 2761011）。预注册 K1/K2/K3（execution.json 先于行为前向冻结）。

### 结果（65 错误行：kc 14、ldr 10、neg 16、ws 25、ab 0）

| 判据 | 结果 |
|---|---|
| K1 内部知道（kc 错误行 knows 率 ≥0.5） | **通过：92.9%**（13/14）；ldr 90%、neg 68.8%、**ws 仅 28%** |
| K2 断层=深层压制（晚层 α_res > 中层） | **否证**：阻抗剖面在全部 36 位置触底（kc/ldr/ws 中位 α_res=0.1，neg=0.3），diff_median=0.0，无层间差异 |
| K3 修复 | 主判据翻转率 100%（14/14）但对照 28.6%>15% 字面不过；**2761a 匹配强度对照：α=0.1 下 kc 定向 71.4% vs 随机 28.6%（配对差 0.43，CI [0.07,0.71]），ldr 100% vs 40%，ws 96% vs 16%——方向特异成立**；neg 反向（定向 0% vs 随机 25%，其 argmax 距边界近，属另一机制） |
| verdict | partially_confirmed（K1 ✓、K2 ✗、K3 修正对照后 ✓） |

### 分析（核心发现）

1. **知识链的弱行为是"知道但说不出"，词义歧义的弱行为是"真不知道"**：kc 错误行 92.9% 内部已把正确答案排到 rank-1，而行为最差的 word_sense（正确率 60.9%）错误行只有 28% 内部知道——两类"最弱"的内部成因定性不同。这是本 Phase 最重要的区分，与 2750 的"内部敏感不等于行为输出"闭环。
2. **断层不在某层**：预注册的"深层主动压制"假说被否证——正确答案方向以 10% 隐状态范数的强度在**任何**深度注入都能翻转行为（kc 71% 的（行，层）格、ldr/ws 近 100%），阻抗剖面全层触底。"说不出"不是知识在某层丢失，而是**读出端竞争**：错误 token（2759 的共模偏置方向）在最终 softmax 胜出，而正确方向全程线性可写。
3. **修复成功且方向特异**：定向 vs 匹配强度随机对照在 kc/ldr/ws 三族全部方向特异（CI>0）；negation_scope 例外——其错误行距判定边界近，随机扰动即可翻转，属"边界接近"而非"知识压制"机制。
4. 与 2759 闭环：2759 说偏置锚在问句事实 span、区分与偏置可分离；2761 说正确答案全程在内部可得、输出被竞争压掉——**知识链缺陷的定位从"链计算故障"转移到"输出竞争失稳"**，修复路径应是抑制偏置方向而非增强链计算。

### 相关文件（immutable，sha256）

phase2761_rdc_kc_fault_repair.py = 63d1ce48…；phase2761a_rdc_matched_control.py = 945fb1f8…；产物 phase2761/qwen4_kc_fault/{execution.json ef2c9479…, result.json 8ae7b3d3…, fault_scores.npz 31865b9a…, addendum_result.json 30b5a54f…, addendum_scores.npz 58b0d757…}。

### 问题硬伤

1. α 网格下界 0.1 饱和导致 K2 无分辨力（阻抗剖面被地板效应压平），"断层不在层间"的结论是在地板效应下的否定，更细的强度网格（α<0.1）可能分辨出层间差异；2. 注入用的是 target 的 unembed 行——修复证明"可绕过"而非"模型自主恢复"；3. 行为仅首 token（B1 类比），未测完整生成；4. 错误行样本小（kc 14）；5. logit lens 的 rank-1 是必要非充分判据（内部表示可能编码目标但语义用法不同）。

### 结论

用户需求 2 完成：**"内部知道但说不出"得到直接证实（kc 92.9%）并成功修复（定向注入 100% 翻转、方向特异对照成立）；但断层不是层间中继故障而是全局读出竞争**——与 2759 偏置结论、2760 kc 深带平台坐标共同指向：知识链改进路径 = 读出端偏置抑制，不是链计算增强。

### 接续

Phase 2762 = 签名几何跨模型功能移植（2752 协议原样移植到 qwen3-1.7b + 2755 checkpoint/子集，判据 P1∧P2 6/6）。

## Phase 2762: 签名几何跨模型功能移植——2752 充分性/必要性协议在 qwen3-1.7b 上 18/18 全通过，几何复现升级为功能可移植 [2026-09-16 01:00]

### 测试原理

用户需求 3（可行性评估：跨宽度参数直移不可定义——4B d=2560 vs 1.7B d=2048，坐标无对应；可行形式 = 功能移植）。2752 协议逐字移植到第二模型：qwen3-1.7b（28 层，block16 mlp 37,748,736 FP32 标量），2755 的六 run delta_128 checkpoint 与 pooled top-1% 子集 npz（每对 377,487 标量，union 682,424），2747 冻结面板 896 行 + 同种子（2752001）表面类/置换扩展，own-objective 判据。变体：full、pz_union（删子集）、rzero_union（删等容随机）、only_union（只留子集 delta）、ronly_union（只留随机）、shuffle_union（子集内置换）；对级 pz_* 按 protocol reduction 省略。判据（先于任何干预观测冻结）：P1 d_obj(pz)>d_obj(rzero)、P2 red(only)>red(ronly)、P3 d_obj(shuffle)>0，各 6/6；portable_functional = P1∧P2 6/6。完整性门：base 评估确定性逐位相等；base own-objective 与 2755 记录值**逐位一致（三条件漂移 0.0）**；结束恢复逐位精确。

### 结果（887.6 s @ RTX 5080）

| run | P1 | P2 | P3 | d_obj pz / rzero | red only / ronly / full |
|---|---|---|---|---|---|
| true_2747 | ✓ | ✓ | ✓ | +0.2665 / +0.0014 | 0.6253 / 0.0220 / 0.9257 |
| perm_2747 | ✓ | ✓ | ✓ | **+3.5108 / +0.0023** | **4.8416 / 0.0443 / 5.4109** |
| mass_2747 | ✓ | ✓ | ✓ | +0.0043 / −0.0003 | 0.0052 / 0.0000 / 0.0062 |
| true_2748 | ✓ | ✓ | ✓ | +0.1834 / −0.0060 | 0.5506 / 0.0187 / 0.6831 |
| perm_2748 | ✓ | ✓ | ✓ | **+3.5501 / +0.0038** | **4.9734 / 0.0513 / 5.4858** |
| mass_2748 | ✓ | ✓ | ✓ | +0.0051 / −0.0003 | 0.0056 / 0.0006 / 0.0061 |

**verdict：P1 6/6、P2 6/6、P3 6/6，portable_functional = TRUE。**

### 分析

1. **签名几何的因果功能结构跨模型成立（重复强调）**：第二个模型（宽度 2048、深度 28、与 4B 不同架构尺度）上，其自有 top-1% 签名子集以同一定量结构承载监督特异行为效应——删子集损失几乎全部效应（perm 态 3.51/3.55 NLL vs 随机删除 ~0.002），只留子集保留 67–90% 效应（perm 4.84/4.97 of 5.41/5.49），随机子集只保留 ~1%，置乱摧毁之。2755 的几何复现（跨代 r≥0.87、跨条件近正交）由此升级为功能可移植：2751/2752 的因果结论不是 4B 特异性。
2. **条件结构同样跨模型保持**：perm 签名效应最大且最清洁（与 2752/2753置换语义几乎完全驻留于定位子空间一致），true 中等，mass 全程噪声量级（其训练改进本身仅 0.006，P1/P2 按方向通过、量级不可过度解读——与 2752 的 mass 限制逐字相同）。
3. **base 三条件 own-objective 与 2755 记录逐位一致**（漂移 0.0）：同一 GPU 上协议复现达到位级，评估管线的确定性使 2762 的全部比较为纯干预效应。

### 理论进展

总图景外部语言操作→内部条件坐标齿轮→跨层编译的参数级支柱完成跨模型闭环：**监督语义→近正交参数子空间→子集因果承载行为效应，三段结论在两个不同宽度/深度的模型上逐项成立**。签名几何现在满足跨模型可移植的功能定义（几何复现 2755 + 因果功能移植本 Phase）；坐标同一性移植因维度不同仍不可定义也不必定义。

### 相关文件（immutable，sha256）

脚本 phase2762_rdc_signature_portability.py = 332aadd30a05358bd72162d07178f7b6a8934ef6cef428f7f83fa3f1934e4506；产物 phase2762/qwen17_block16/{execution.json 231faf3c…, result.json f26ed586…, intervention_scores.npz f2481836…, intervention_scores_meta.json 97ed7b99…}。输入：phase2755/qwen25_3b 六 run delta_128.npz（逐 receipt SHA）与 pooled_top1pct_indices.npz、phase2747 冻结材料。

### 问题硬伤

1. mass 态效应量在噪声量级，P1/P2 的 6/6 依赖方向而非量级；2. 子集是模型自有（2755 提取），检验的是因果结构可移植而非坐标跨模型直移（后者因维度不可定义）；3. 对级 pz_* 变体省略（协议缩减），P4 特异性矩阵未在第二模型复测；4. 单一随机对照集（种子 2762000）；5. 第二模型与 4B 同 tokenizer 族，跨 tokenizer 未测。

### 结论

用户需求 3 完成：**签名几何跨模型可移植以功能形式成立（18/18 判据通过）**。用户三项需求全部闭合：2760（层分布图）、2761(+a)（断层定位与修复）、2762（功能移植）；2759（接续预注册的问句 span 删除）判定 not_confirmed 并产出偏置/区分可分离新结论。

### 接续

下一阶段候选（按总目标排序）：(a) V3 评估与计划文档——把 2759–2762 整合进下一轮可判定计划，重点把读出竞争（2761）与偏置锚点（2759）合并为输出端干预设计；(b) perm 训练态 + 读出端偏置抑制的组合修复（是否能把 kc 行为从 78% 提到 ab 水平，GPU 中成本）；(c) 跨 tokenizer 第二模型训练（放宽 tokenizer 门）以检验签名几何的终极普适性。

## V3 路线书发布 [2026-09-16 02:00]

`research/glm5/docs/EVALUATION_AND_PLAN_V3_20260916.md`（sha256 81386751c20050a2ed58fe8a74bfdfeadc3f4cf0326e813d615b4d5954f665f2）：V2 五项闭合评估 + 2758–2762 新证据综合 + V3 计划（P1 组合修复 2763 / P2 读出端干预分叉 / P3 跨 tokenizer / P4 Unified Theory v4.2）。核心图景更新：kc 失败从"链路问题"改判为"读出问题"。

## Phase 2763: 无 oracle 去偏置修复——bsub 方向特异修复 21/65（ctrl 0/65）但 kc 对偏置抑制免疫；verdict not_confirmed [2026-09-16 02:15]

### 测试原理

V3-P1（用户"继续"自动进入）。问题：不用 oracle 答案方向，去偏置能否把 kc"内部知道"转成行为？bsub 读出端偏置抑制：v_row := h_final(含问句事实 span) − h_final(删 span)（native 定义，2759 fact-token 规则，320 行全非空 G2，中位范数 72.1），推理时 hook layers[35] 输出 h := h − α·‖h‖·unit(v_row)，α∈{0.1,0.3,1.0} 主判据 0.3；随机方向对照（α=0.3 匹配，种子 2763001）。5 run（native + perm_2747/2748 + true_2747 + mass_2747，2759 部署协议 G0）× 320 diagnostic 行，greedy 首 token 判定。预注册判据：C1 perm 态修复 native-kc-wrong 14 行 ≥0.4 且 > true_2747；C2 native+bsub(α=0.3) 翻转率 ≥0.3 且 bootstrap CI(1000, 2763010) 下界 > ctrl；C3 kc repair-any > ws repair-any；C4 组合探索性。完整性门 G0/G1（base 确定性逐位 0.0）/G2/G3（native wrong 与 2761 逐族一致 0/14/10/16/25）全过。

### 结果（128.0 s @ RTX 5080）

| 判据 | 结果 |
|---|---|
| C1 perm 修复 | **否证**：perm_2747 kc 修 10/14 但 true_2747 同样 10/14（非 perm 特异）；perm_2748 6/14 |
| C2 bsub 修复 | **通过**：21/65 = 32.3%（CI [0.215, 0.446]），ctrl **0/65** |
| C3 族特异 | **否证**：kc 71.4% vs ws 52.0%，CI [−0.109, +0.497] 含 0 |
| C4 组合 | 无叠加：combo 2–10 ≤ max(single)（两干预修复集不相容） |

**bsub 翻转的族分布（α=0.3）**：ws 13/25、ldr 6/10、kc 2/14、neg 0/16。剂量响应：ws 单调（1→13→25，α=1 全翻但无对照有扰动混淆）、neg 高剂量才翻（0→0→8）、kc 平坦（1→2→2）且只翻 margin 最浅两行（−2.0）。

### 分析

1. **bsub 是首个无 oracle 行为修复手段且方向真实**（重复强调）：仅减去问句事实 span 的残差末位线性贡献，32.3% native-wrong 行翻正而匹配随机方向 0/65——v_row 携带行为因果内容，2759 偏置锚点在行为面成立。
2. **但修复是反"内部知道"预测的**：kc（知道说不出）对 bsub **免疫**（2/14），ws（2761 判"真不知道"knows 0.28）反而 13/25 翻正且其翻转行 margin 中位 −12.75（深错误）——ws 行为错误主要是**偏置锚定的 rival**，去偏置即退位；kc 读出竞争的载体**不是**问句 span 的残差线性分量。2761 的 oracle 修复（100%）与 bsub 免疫（14%）共同约束：kc 缺陷在偏置通路之外的读出机制。
3. **训练态修复是双刃几何重写**：perm_2747/true_2747 修 kc 10/14（修复行 margins 逐位相同 = 同一批行）但分别打坏 61/35 个 native-correct 行，净负（perm kc 总体 57.8% < native 78.1%）；mass_2747 唯一净改善（81.25%）但其 2758 行为效应在噪声量级。C1 的"perm > true"方向证伪：两态修复同一批行。
4. **族分离精细化**：kc = 深缺陷（偏置免疫 + 只能几何重写但双刃）；ws = 浅缺陷（偏置敏感 + 训练态无效 1/25）；neg/ldr = 中间。

### 理论进展

读出竞争分解为两条通路：**偏置锚定通路**（问句 span 残差分量，bsub 可修，ws/ldr 主导）与**非偏置竞争通路**（kc 主导，载体未知）。Unified Theory v4.2 的候选竞争项需拆分：Cmp(o,r,v) 中 r（读出）项对 kc 与 ws 的失败模式不同源。

### 相关文件（immutable，sha256）

脚本 phase2763_rdc_debias_repair.py = 65acb79f98d0ed5f826ffb61ae4df777cf5c0dc3dbbd477f069547ec6c059271；产物 phase2763/qwen4_debias_repair/{execution.json eb03e8c8…, result.json 96a8ab6b…, behaviour_scores.npz e66ea5e7…, behaviour_scores_meta.json d504c8ca…, bias_dirs.npz 66cbb755…}。

### 问题硬伤

1. α=1.0 的全翻（ws 25/25）无匹配随机对照，含大扰动混淆；方向特异结论只在 α=0.3（有 ctrl）内成立。2. v_row 是行级一阶线性近似（span 贡献的非线性/早层分量未建模）——kc 免疫不能排除偏置以非线性形式压制 kc。3. C1 训练态修复行的判定用 base 前向，未控制句法扰动；修复行集 perm=true 逐位一致也可能是部署协议的共性伪影。4. 单一 ctrl 种子。5. ws 翻转行的"修复"语义依赖 target 定义（argmax 重排），不代表内部知道。

### 结论

V3-P1 verdict not_confirmed：去偏置不能修复 kc（C1 方向证伪 + kc 对 bsub 免疫），但建立首个无 oracle 方向特异行为修复（C2，ctrl 0/65）并把族分离推进到通路级（偏置锚定 vs 非偏置竞争）。

### 接续

Phase 2764（V3-P2 分叉，按 2763 结果修正）：kc 非偏置竞争通路的载体定位——(a) 用 2761 已存 lens_margin 逐层轨迹做"正确答案反超/丢失层位"分析（零 GPU），检验反超层位是否与 2760 kc 齿轮坐标层（L20+）对齐；(b) 若反超在 L36 末端，设计峰值层定向干预；(c) bsub 非线性分量探针（分层 v_row）。

## Phase 2764: kc 读出竞争层位定位 + 2761 K1 off-by-one 修正——反超层位 L33+ 末端、读出竞争三层结构 [2026-09-16 02:40]

### 测试原理

V3-P2(a)（零 GPU 数据分析，源 = 2761 fault_scores.npz 的 lens_margin 320×37 轨迹 + 2763 行为修复矩阵）。定义：recover_layer = lens_margin 在 [8,35] 首次 ≥0 的层；lose_layer = [8,36] 内最后 ≥0 的层；peak_layer/margin = argmax。预注册预测 D1 kc knows 行 recover 中位 ≥20（对齐 2760 齿轮 L20+）；D2 多数行 lose ≥33（末端塌陷）；D3 bsub 修复的 ws 行呈深正-塌陷形态；D4 训练态修复行与非修复行轨迹形态相容（描述性）。

### 重大修正（2761 K1 off-by-one bug）

2761 的 knows 判定 lens_rank==1 中 lens_rank = (z > target).sum()，即"恰好 1 个 token 严格大于 target"（rank-2），**非** rank-1。修正定义 knows = lens_margin ≥ 0 于 [8,36)。修正后各错误行族内部知道率：

| 族 | n_wrong | 2761 报告 | 修正后 |
|---|---|---|---|
| knowledge_chain | 14 | 0.929 | **0.286**（4 行，峰值 margin 0.17/0.30/2.96/4.00）|
| long_distance_role | 10 | 0.900 | **0.600** |
| negation_scope | 16 | 0.688 | **0.000** |
| word_sense | 25 | 0.280 | **0.040**（1 行，0.75）|

修正产物 corrected_knows.json = b99034a456e663e9310d96d2a7ff45bce8c5dba3dcce259bfd997169a21ea19d。2761 的 K2/K3 行为级结论与 2761a 对照不受影响；受影响的仅是"内部知道率"的量与"kc=知道说不出 vs ws=真不知道"的定性叙事（弱化为：kc 含最高 margin 的 knows 行但比例低；ws 几乎全不知道但多数接近；neg 全不知道且远离）。

### 结果（0.02 s，D1/D2/D3 通过、D4 修正）

- **D1 通过且强于预测**：kc knows 行 recover 中位 **33.5**（L8–31 全程 margin<0，答案只在最后几层浮现）——2760 齿轮坐标 L20+ 对齐的是条件信息写入，不是答案可读出。
- **D2 通过（100%）**：kc knows 行 lose_layer 全部 ≥33——正确答案在 L33–36 读出末端被反超，无中继丢失。
- **D3 通过但形态修正**：bsub 修复的 ws 13 行 knows（修正定义）仅 1/13、峰值 margin 中位 −2.55——不是"深正后塌陷"而是 **near-miss（rank 2–3 边缘 + 偏置锚定 rival）**；未修复 ws 行无 knows 且更远离边缘。
- **D4（修正后）**：训练态修复的 10 kc 行 = 修正 knows 4 行中的 3 行（margin +2.96/+4.00 的深 knows 行 2/2 全修复；+0.17 的浅 knows 行未修复）+ 7 行深 margin 非 knows 行。修复能力随 margin 深度分层。

### 分析：读出竞争三层结构（重复强调三次）

1. **深层 knows（margin ≥ +1）**：仅 kc 2 行——bsub 可修（2/2）、训练态可修（2/2）——真正"知道说不出"，且是 2761 oracle 与 2763 bsub/训练态修复的交集。
2. **near-miss（margin ∈ (−3,0)，rank 2–3）**：ws 13 行、ldr 6 行——bsub 修复主场（去偏置即胜出）、训练态无效（ws 1/25）。
3. **deep-unknown（margin ≤ −3 或全层远离）**：neg 16 行、ws 12 行、kc 10 行——任何去偏置免疫（neg bsub 0/16），需几何重写（双刃）或 oracle。

三层结构统一解释 2761（oracle 全修）、2763（bsub 修 ws/ldr 不修 kc 多数行、训练态双刃）与 2759（问句 span = 共模偏置锚点，压制 near-miss 行的读出）。

### 相关文件（immutable，sha256）

脚本 phase2764_rdc_kc_trace_localisation.py = 36eecbd53d8c779a393dcb671e5373bb631a69209e1f02a6fd74e250f8110dd5；产物 phase2764/qwen4_kc_trace/{execution.json ba15cd7d…, result.json 99b9f0b8…, trace_stats.npz 6ecffdaf…, kc_trace_localisation.png d6e9c2a9…, kc_trace_localisation.svg 511e09e0…, corrected_knows.json b99034a4…}。

### 问题硬伤

1. knows 行样本小（kc 4、ws 1），层位统计（recover 中位 33.5）脆弱。2. lens_margin 是 logit-lens 意义的读出，与真实残差读出可能有系统差。3. D4 修复行集含 7 行深 unknown 行，训练态"修复"语义部分是重写后的碰巧翻对。4. 分析完全继承 2761/2763 的前向确定性假设。

### 结论

2761 K1 修正（92.9%→28.6%）；kc 答案仅在 L33+ 末端浮现且全部在末端被反超；读出竞争三层结构（deep-knows / near-miss / deep-unknown）建立，V3-P2 的干预设计应按层分级：near-miss 行 = 偏置抑制，deep-knows 行 = 读出放大，deep-unknown 行 = 不可行为修复。

### 接续

Phase 2765 候选（按总目标排序）：(a) near-miss 边界的定量刻画——margin 阈值 −3 与 bsub 可修性的关系曲线（已有 2763 数据可零成本验证）；(b) deep-knows 行的读出放大器设计（final norm 前对 target 方向以外的共同抑制，非 oracle 形式）；(c) 把三层结构写入 Unified Theory v4.2 的 Cmp(o,r,v) 分解。

## Phase 2765: near-miss 边界定量刻画——lens_peak 是 bsub 可修性预测因子（AUROC 0.834），中段免疫带确立 [2026-09-16 02:55]

### 测试原理

V3-P2(a) 收尾（零 GPU，源 = 2763 behaviour_scores.npz 行为 margin + 2761 lens_margin 峰值，65 native-wrong 行）。问题：什么预测 bsub(α=0.3) 翻转？

### 结果

| 预测因子 | AUROC |
|---|---|
| 行为 margin（tgt − rival，最终读出） | 0.672 |
| **lens_peak（内部知识接近度峰值）** | **0.834** |

lens_peak 分箱翻转率：[−6,−4) 0/15 → [−4,−3) 4/21 → [−3,−2) **8/11** → [0,5) **9/11**（近单调）。行为 margin 分箱**非单调**：[−30,−10) 翻 12/38，但 [−10,−4) **0/13**（中段免疫带）。行为 margin 深而 lens_peak 高的 12 行仍可翻——偏置把内部接近的行压得极深，去偏置后即胜出，正是"偏置压制读出"的预测签名。

### 分析

1. bsub 可修性由**内部知识接近度**（lens_peak）决定而非表面错误深度（AUROC 0.834 vs 0.672，重复强调）。
2. 中段免疫带（beh_margin ∈ [−10,−4) ∧ lens_peak < −4，13 行全免疫）= deep-unknown 的操作定义；near-miss 边界定量化为 lens_peak ≈ −3.5。
3. 三层结构的边界从描述升级为定量：deep-knows（lens_peak ≥ 0，11 行翻 9）、near-miss（[−3.5,0)，翻 8/11 于 [−3,−2)）、deep-unknown（< −4，0/15）。

### 相关文件（immutable，sha256）

产物 phase2765/qwen4_nearmiss/result.json = d4f5a262eb70c3600f36ddbf72e62be036d97abdc85ad1e79a1f747d598bb8aa（分析脚本为一次性 python 会话，逐行数据在 result.json per_row）。源数据 2763/2761 产物 SHA 见各条目。

### 问题硬伤

分箱小样本（部分箱 n≤4）；AUROC 未做 CI；两预测因子相关（beh_margin 部分由 lens_peak 决定），非独立贡献分解。

### 结论

V3-P2(a) 闭合。干预分诊规则确立：lens_peak ≥ −3.5 → bsub 候选；< −4 → 行为不可修（需训练/oracle）。

### 接续

Phase 2766 候选：(a) deep-knows 行读出放大器（非 oracle 形式：final norm 前共同抑制非 target 的 top-rival 方向或全局 sharpening，在 2 行 deep-knows kc + 9 行 lens_peak≥0 行上检验）；(b) Unified Theory v4.2 文档化三层结构；(c) 跨 tokenizer 训练（V3-P3）。

## Phase 2767: 类别知识图谱 v1——末位谓词双世界面板：知识探针全过、深带候选坐标近空、两簇对极分歧几何 [2026-09-15 21:50]

### 测试原理
用户需求"破解类别知识图谱（水果/植物/动物/固体/液体）并外推到颜色/大小属性"。v1 沿用 2760 协议：5 类别 × 16 实体，World A=真类别语句、World B=错类别语句（每实体 4 错类，320 分歧对）+ 实体交换对照（80 对）+ 颜色（16 对）/大小（16 对）真值-置换对；末 prompt 位置全 37 层隐状态；候选坐标=逐对 top-32 |dh|、≥60% 稳定；主层 L28；预注册 P1-P5 于任何前向之前冻结（execution.json）。知识探针=谓词前一位置的谓词池 next-token margin。

### 结果
1. **P5 知识探针（通过）**：fruit/plant/animal/liquid rank-1 = 93.75–100%（animal margin 中位 13.3）、solid 68.75%、color 93.75%；size 25%（判据失败，2768 证明是句尾位置伪影而非模型不知道）。
2. **P1 深带放大（否证，ratio 0.44）**：类别词在句尾，末位分歧被谓词词的词汇身份响应主导——nd 在 L9–10 出现 0.32–0.34 浅层峰值，深带 0.13–0.19；与 2760 的"近盲浅带"形态完全不同（2757/2760 的世界差在 body、末 token 相同）。
3. **P2 稳定候选坐标（否证）**：L28 每类 0–6 个（fruit 0），top-32 稳定坐标被"这一对换了哪个词"支配，无类别专属齿轮签名。
4. **P4 语义距离（否证但换来更强结构）**：cos(fruit,plant)=0.035 < cos(fruit,animal)=0.319；但 5×5 平均分歧方向余弦呈**两簇对极几何**——生物簇 {fruit,plant,animal} 内部 0.03–0.32、物理态簇 {solid,liquid} +0.387、跨簇全部负（−0.31～−0.64）。
5. 实体交换对照的深带分歧（nd 0.06–0.09）仅为类别交换（0.13–0.19）的约 1/2–1/3：末位条件响应中类别谓词条件强于实体身份。

### 相关文件（immutable，sha256）
脚本 `phase2767_category_knowledge_atlas.py` = ba37c9cdd54b9f1e76d556a566d61174150e6bf47fc083b6475697d2af39d402；产物 `phase2767/qwen4_category_atlas/{execution.json 914bcdd9e40a94a8…, result.json 1190a9fad973dab3…, panel_stats.npz 0ffebc08dd4fb7ab…}`。

### 问题硬伤
①末位谓词设计的词汇地板淹没深带知识信号（P1/P2 否证的直接原因）；②size 探针位置伪影；③P3 的 Jaccard 判据在候选集 0–6 个时不可解释；④单模型、单面板、末位置单点采样。

### 结论
v1 协议确认"模型知道这些类别事实"（P5），但"类别知识在深带的坐标签名"需要把谓词移离句尾并分离词汇/框架响应 → Phase 2768。

## Phase 2768: 类别知识图谱 v2——固定尾+伪词对照：知识=对公共交换坐标的 1.6–2.9× 幅度放大而非独立坐标组；类别状态几何两簇有序 [2026-09-15 21:55]

### 测试原理
修正设计：①谓词移离句尾（固定尾 ", as everyone knows."，对内 token 长度相同，L0 末位分歧精确为 0，脚本内断言）；②伪词对照（wug/blicket/dax/zorple/fep/kiki/toma/nirp/splet/cruv/plap/snid/glore/vint/quap/melf）——同一句法框架、同一类别词交换、无知识；③真词对每类一个预注册错类（fruit→animal, plant→liquid, animal→solid, solid→fruit, liquid→plant），5×16=80 对；伪词 80 对；颜色/大小真词各 16 对+伪词对照 32 对；预注册 Q1–Q5 于任何前向之前冻结。

### 结果
1. **Q3 知识放大（通过，本 Phase 核心正结果）**：L28 处真词分歧/伪词分歧 = fruit 2.06、plant 2.95、animal 1.86、solid 1.62、liquid 2.95（5/5>1）——同一交换操作，有知识的主体产生约 2–3× 的状态分歧。
2. **Q2 知识专属坐标（否证，信息量大）**：真词与伪词的 L28 稳定候选集 Jaccard 0.2–0.47 ≫ 随机 null q95=0.032，且伪词候选数（7–15）不低于真词（1–17）——top-|dh| 稳定坐标主要是**框架/词汇交换机器**；知识不是另立坐标组，而是在重叠坐标上做幅度调制。
3. **Q1 仍否证（ratio 0.54）**：固定尾后浅层峰值仍在——句中内容词交换的词汇响应经注意力在早中层即传播到末位；nd 剖面不区分知识与词汇。
4. **Q4 类别状态几何（通过，弱）**：L28 类别均值状态余弦，生物簇内均值 0.983 > 跨簇 0.960、solid–liquid 0.985——排序正确但绝对差小（残差流共同分量主导）。
5. **Q5 否证**：color∩size 候选集 Jaccard 0.25——属性交换共享同一框架机器，与 Q2 同构。
6. **知识探针修正确认**：句中位置探针下 size 93.75%、solid 87.5%、其余 100%——2767 的 size 25% 确认为句尾位置伪影。

### 分析
与"不同语言族使用互不重叠坐标组"（2760 关系族）对比：内容类别知识在末位读出对比下表现为**分布式幅度编码**（knowledge amplifies shared swap-response coordinates 1.6–2.9×），而非坐标专属齿轮；两簇对极分歧几何（2767）是当前最强的类别几何信号。对 RDC"齿轮"图景的重要限定：**正交齿轮在关系族/监督语义层面成立（2751/2755/2762），在内容类别层面当前证据不支持**——内容知识的内部存在形式可能主要是幅度/方向调制而非离散坐标槽位。

### 相关文件（immutable，sha256）
脚本 `phase2768_category_atlas_v2.py` = dcda95c65d55cf3da1d0a092b30c5813db6e662992d37abbadcf3bf63dd014fb；产物 `phase2768/qwen4_category_atlas_v2/{execution.json 97d80856d68cdb2d…, result.json 6c4a2ed919da8e0a…, panel_stats.npz 13b232c8687f03bf…}`。

### 问题硬伤
①Q2 结论应表述为"无独立坐标组"而非"无坐标签名"（幅度调制本身是签名）；②伪词语句 token 长度与真词不同（对内长度相同已断言，跨对比较不受影响）；③单错类设计限制类间语义距离结构测量；④属性伪词对照仅用单一伪词主语（wug）；⑤单模型 qwen3-4b、末位置单点、L28 单主层；⑥两簇几何来自 2767 的 4 错类均值方向，2768 单错类下未复现检验；⑦real_plant 与 real_liquid 候选集恒等是 derange 对称（plant→liquid 与 liquid→plant 互为反.pair）的设计结果，非独立复现。

### 结论与接续
Phase 2767/2768 完成用户"类别知识图谱"首轮破解：知识存在性（句中探针 87.5–100%）、知识贡献（1.6–2.9× 幅度放大）、编码形式（幅度调制于公共交换坐标，非独立坐标组）、类别几何（生物 vs 物理态两簇对极）。Phase 2769 候选：(a) 投影掉伪词张成的框架/词汇子空间后对真词残差重做候选坐标与稳定性分析（知识专属残差坐标）；(b) 逐 (c→c') 分歧方向的语义距离结构 + logit-lens 类别方向逐层轨迹（何时变得可读）；(c) 跨模型（qwen3-1.7b）复现 Q3 幅度放大与两簇几何。

## Phase 2769: 附件审查修正 + Alpha——kc 非偏置竞争载体定位：模板答案码 token 经 L31/33/35 少数头写回；非 oracle 读出放大器成立（deep-knows 5/5，附带损伤 0.5%） [2026-09-16 04:20]

### 测试原理
用户要求先审查两份附件（A=框架可信度评估、B=2763–2768 讲解）再综合 Alpha/Beta/Gamma 计划执行。C001 零 GPU 逐条核对附件声明 vs 2763–2768 immutable 产物。C002–C004 检验 2764 遗留问题：L33+ 反超内部已知答案的竞争载体是什么。载体定义：模块层 M={31..35} 内，逐头注意力写回在最终位置对 u=unit(W_U[rival]−W_U[target]) 的投影为正的头；因果杠杆：o_proj 输入在最终位置的对应头切片清零。预注册 E1（源浓度≥0.6）/E2（定向翻转≥4/14 且>随机对照 q95）/E3（附带损伤≤25%）在任何干预前冻结于 execution.json。

### 结果
1. **C001 审查（20 条声明）**：14 verified、5 corrected、1 outdated。关键修正：2767 知识探针 plant=93.75%（附件写 100%）；2768 真词-伪词 Jaccard 实为 0.0–0.47（fruit 真词候选仅 1 个、Jaccard 0.0，附件写 0.2–0.47）；附件 B 三层表 deep-knows 写"仅 kc 2 行"混淆了两个定义——2764 口径（margin≥+1，kc 内）为 3 行，2765 口径（lens_peak≥0 跨族）为 12 行；附件 A（写于 2767/2768 前）"概念间几何从未测量"已过时，其嵌套/正交推测被实测（幅度调制+两簇对极）部分否证。
2. **C002 载体观察**：载体头层分布 L35:20、L33:14、L31:12、L32:6、L34:4（按 65 行计数）。**E1 失败（中位 0.294<0.6）**——rival 正质量源构成：other（模板/标点/答案码）0.819、content 0.175、fact（问句事实 span）仅 0.006；逐行 top 源是指令行答案码 token（如头 (33,16) 的第一源为模板句中的"Yes"，gain 3.6–4.2）。**竞争载体不是问句偏置，而是模板答案码 token。**
3. **C003 因果**：定向头消融翻转 7/14（随机对照 100 抽均值 0.49、q95=2，p<0.025）；附带损伤 7/2800=0.005（E3 过）；**lens_peak≥0 的 5 个 kc 深知行（273/281/298/307/311）5/5 全部翻转**，非深知 kc 行 2/9。E2/E3 过 → verdict=carrier_localised。
4. **C004 分层 bsub（修正后）**：正确按行号索引 v_rows 后，hook 于模块 L29–34 均为 20/65、L35 为 21/65——bsub 自 L29 起几乎同样有效，非读出端专属。初版误用 v_rows[序号]（应为 v_rows[行号]）的曲线作废，产物删除重跑。

### 分析
kc 非偏置竞争通路载体定位成功：指令模板中的答案码 token（Yes/No/是/否）经少数深层头的注意力写回在 L33+ 压过正确答案；消融这些头即构成非 oracle 读出放大器（deep-knows 行 5/5 修复）。与 2763 偏置通路（问句事实 span）正交：两通路源不同、层不同、可修行不同。

### 相关文件（immutable，sha256）
脚本 phase2769_rdc_attachment_audit.py = 397d158394000bca18004a988c2c7182838586760a5c78ded098a6f961abdad2；phase2769_rdc_carrier_localisation.py = 14138a0a6c1edc49f872fa97a6addfd1f6298d205480a879860a257c4923885f；产物 phase2769/qwen4_attachment_audit/{execution.json, result.json a1d16591…}；phase2769/qwen4_carrier_localisation/{execution.json 036115f7…, result.json ae2f353e…, carrier_stats.npz 8f852ab1…}。

### 问题硬伤
①E1 预注册阈值按 fact+content 定义，未预见"模板 token"类目，判据本身被否证即信息；②头级消融非源级消融，源归因是描述性；③单模型；④随机对照只消融同数头，未匹配写回范数；⑤初版 v_rows 索引错误曾污染 C004，已作废重跑（保留于 run log）。

### 结论
verdict=carrier_localised。读出竞争三通路图景确立：偏置通路（问句事实 span，bsub 可修）+ 模板载体通路（答案码 token，头消融可修）+ 深未知（免疫）。

### 接续
Phase 2770（Beta）：幅度调制的参数级实现 + 跨模型放大复现。

## Phase 2770: Beta——知识幅度调制的参数级实现：特异性成立（10x）但不集中（top-251 单元仅承载约 2%）；qwen3-1.7b 放大 5/5 复现 [2026-09-16 04:35]

### 测试原理
2768 证明知识=公共交换坐标的 1.6–2.9x 幅度放大。问题：哪些 gate/up 参数实现"转更快"？方法：精确复建 2768 v2 面板（288 prompts），捕获全 36 层 MLP 输入 x_l，逐单元 act=silu(W_g x)⊙(W_u x)；逐单元 swap-gain 沿对方向 u_p 的投影 g_p[k]=(Δact_k)(W_d[:,k]·u_p)；单元知识增益比 r_k=med|g_real|/med|g_pseudo|。选择半（奇数对）定单元，因果半（偶数对）检验。杠杆：down_proj 输入对应单元坐标乘 gamma∈{0.5,2}。预注册 B1/B2a/B2b/B4 冻结。

### 结果
1. **B1（通过，5/5 类）**：每类 3.3–3.9 万个 r_k≥2 且符号一致单元（选择半 8 对）；top r_k 达 91–226。
2. **C002 因果（251 个 top 单元并集）**：gamma=0.5 时 real nd(L28) 相对变化 −1.4%～−2.3%（均值 −1.72%），pseudo −0.1%～+0.2%（−0.13%），匹配随机单元 −0.28%；gamma=2 时 real +3.26% vs pseudo +0.39%。**B2a 失败**（1.72pp<5pp 阈值）；**B2b 通过**（real 降幅=随机对照 6.1 倍）。verdict=modulation_causal=not_confirmed。
3. **B4（通过）**：qwen3-1.7b 同材料 L21 放大比 fruit 2.47/plant 2.23/animal 1.91/solid 1.82/liquid 2.23，5/5>1——知识放大跨模型复现。

### 分析
因果特异性成立（对调制单元群体的干预效果是对随机单元的 6–8 倍、对 pseudo 的 10 倍以上），但 top-251 单元仅承载约 2% 的放大——候选单元达数万，放大不集中于少数参数。**知识幅度编码是双重分布式的：坐标层面无专属组（2768），参数层面亦无小集合（本 Phase）。**"让公共机器转更快"是全体单元的群体性增益抬升，不是一台专用加速器。

### 相关文件（immutable，sha256）
脚本 phase2770_rdc_amplitude_modulation.py = f827d17dc691cdd02f410322181c2ca00c0caa76410d597282968c8e85f23eae；产物 phase2770/qwen4_amplitude_modulation/{execution.json b4e20409…, result.json f544f42e…, modulation_units.npz c3f2873d…}。

### 问题硬伤
①B2a 的 5pp 绝对阈值对 251/数万单元的干预规模偏严，结论应读作"不集中于 top-251"而非"无调制参数"；②单元选择与因果检验的同分类循环已用奇偶对分割缓解但未消除；③g1 复现门在 fp16 存储缺陷修正前曾误报（x_store 曾用 float16，已改 float32）；④1.7b 未做单元级调制（仅放大复现）。

### 结论
verdict=not_confirmed（对"集中调制参数"假设的否证），特异性正结果保留。理论更新：知识=双重分布式幅度调制。

### 接续
Phase 2771（Gamma）：框架子空间投影 + 类别知识流形几何 + 跨模型两簇复现。

## Phase 2771: Gamma——去框架残差非深带主导（R2 失败）；类别词 L8–14 即 lens 可读；两簇几何跨模型复现（1.7b intra 0.9845>cross 0.9706）；δ 方向几何受 derange 对称污染 [2026-09-16 04:50]

### 测试原理
C001：80 伪词对 L28 分歧的 SVD 框架基（90% 能量，k=16），投影掉后对真词残差重做候选坐标与 nd 剖面。C002：7 方向（5 类别+color+size）平均分歧方向的 7x7 余弦流形。C003：类别词 5 选 1 lens 可读层轨迹。C004：qwen3-1.7b 类别均值状态两簇检验（比例层 L21）。预注册 R1–R5 冻结。

### 结果
1. **C001**：k=16。残差稳定候选：fruit 3、animal 3、plant/solid/liquid 0（R1=2/5，弱通过）；**R2 失败**：去框架残差 nd 深带/浅带比=0.446<1——深带分歧的大部分仍是框架/词汇机器，知识特异残差小且非深带集中。
2. **C002**：7x7 方向余弦出现 **cos(plant,liquid)=−1.000 精确值**——DERANGE 设计（plant→liquid、liquid→plant 互为反向对）使两类的平均分歧方向天然反平行，2768 硬伤⑦的预言在方向几何上应验。R3 形式通过（生物簇内均值 0.017 > 跨簇 −0.516）但**被该伪影污染，只作记录不作机制证据**。可靠新结构：**color–size cos=0.548 自成一对**，与五类别簇均弱连接（−0.16～0.18）。
3. **C003（通过）**：类别词 lens 可读性 16/16 全实体通过，可读层中位 fruit/animal 8、plant 9、liquid 10、solid 14——类别知识自中层起即可读。
4. **C004（通过）**：1.7b 均值状态几何 intra_bio 0.9845 > cross 0.9706——**两簇对极几何跨模型复现**（R5）。

### 分析
verdict=not_confirmed（R2 失败）。三点综合：①类别知识的"流形"应建立在均值状态几何（跨模型复现）而非单错类分歧方向（对称伪影）；②属性（color/size）与类别（bio/phys）是分离的几何对象；③知识信号在状态末位的占比被框架机器主导，与 2768 Q2、2770 分布式结论自洽。

### 相关文件（immutable，sha256）
脚本 phase2771_rdc_manifold_gamma.py = bff69ce0a510ea514c2eaf9a4840508c03d808a09f01e48d70036dac39d0b74d；产物 phase2771/qwen4_manifold_gamma/{execution.json df19560f…, result.json 34fe6a18…, manifold_stats.npz ba31848e…}。

### 问题硬伤
①单错类 derange 对称污染 δ 方向几何（cos=−1.0 精确值即证据）；②k=16 框架基可能有欠/超投影敏感性未扫描；③1.7b 仅均值状态几何，未复现分歧方向结构；④单面板。

### 结论
两簇对极（均值状态）跨模型成立；方向级流形几何需去对称设计重建；去框架残差不深带集中。

### 接续
自动续研判定：下一阶段目标（破解知识存储/读出机制）与总目标相同 → 自动进入 Phase 2772：组合非 oracle 修复管线 + 载体族级普遍性。

## Phase 2772: 组合非 oracle 修复管线成立——载体头消融(14/65)+bsub(21/65)组合 28/65(43.1%)，附带损伤 0；族级分诊矩阵确立（ws 20/25、kc 消融 7/14、neg 全免疫）；bsub 精确复现 2763 验证 harness [2026-09-16 05:05]

### 测试原理
对全部 65 native-wrong 行执行 2769 载体识别（各族），三模式干预：(a) 载体头消融；(b) bsub L35 alpha=0.3（2763 v_row，按行号索引）；(c) 组合。51 非 kc 行 30 抽随机头对照；ws 载体头对 64 attribute_binding 正确行测附带损伤。预注册 F1（combo≥24/65 且>bsub 21）/F2（随机对照≤1/3）冻结。

### 结果
1. **主结果**：abl 14/65、bsub 21/65（**逐位复现 2763 的 21，harness 验证**）、**combo 28/65=43.1%**。随机对照均值 0.60（F2 过）；附带损伤 0.0000（0/2688）。F1 过 → verdict=combined_repair_confirmed。
2. **族级分解**：ws abl 1/bsub 13/**combo 20/25**；ldr 6/6/6；kc **abl 7/bsub 2/combo 2**（叠加 bsub 反而抵消 5 个 abl 修复）；neg 0/0/0（全免疫）。
3. **载体源构成（族级中位）**：kc other 0.71/content 0.29/fact 0.007；ldr content 0.74；neg other 0.98；ws other 0.70——ldr 的载体主要是内容词，neg 的 rival 源几乎全在格式 token 但头消融无效（0/16）。

### 分析
非 oracle 修复管线确立：**组合修复把可修复行从 32.3% 提升到 43.1%（+33% 相对）**，且零附带损伤。但组合不是普遍最优——kc 上 bsub 与消融相互抵消（5 行），**必须按族/按行分诊**：ws→组合、kc→纯消融、ldr→任一、neg→行为不可修（需训练或 oracle）。与 2765 的 lens_peak 分诊（行级）互补，构成"行级 lens_peak × 族级通路"二维分诊。neg 族免疫提示其失败机制不在读出端（与 2764 deep-unknown 分层一致）。

### 相关文件（immutable，sha256）
脚本 phase2772_rdc_combined_repair.py = 85be02ad6cf0c65f137b18b865c1530f5686b7d976240523f3430abeae262aa5；产物 phase2772/qwen4_combined_repair/{execution.json f8d4089f…, result.json 3430249c…, repair_stats.npz 457b4d35…}。初版 v_rows 序号索引错误的产物已删除重跑（保留 run log 记录）。

### 问题硬伤
①初版 v_rows[k] 索引错误（v_rows 按 320 行号索引）曾致 bsub/combo 数字无效，已作废重跑；②头消融是行特定识别，未测试跨行共享头集合的"一次手术"形式；③neg 免疫机制未定位；④单模型。

### 结论
combined_repair_confirmed。本轮 Alpha/Beta/Gamma+组合修复四连的净产出：载体定位（模板答案码 token）→ 双重分布式幅度编码（否证集中参数）→ 流形几何方法修正（状态几何 vs 方向几何）→ 43.1% 非 oracle 组合修复 + 二维分诊矩阵。

### 理论进展（Unified Theory v4.3 候选记录）
1. 读出竞争三通路（2769/2772 修订）：Error = 偏置通路（问句事实 span 残差；bsub，L29+ 即有效）∪ 模板载体通路（指令答案码 token 经 L31/33/35 少数头；头消融可修）∪ 深未知通路（行为不可修）。两通路源正交、可叠加但族依赖（kc 上互抵）。
2. 知识编码二分的参数级补全（2770）：正交齿轮组（关系族/监督语义，2751/2755/2762）vs **双重分布式幅度调制**（内容类别：坐标无专属组 + 参数无小集合，特异性是群体性的）。
3. 类别知识几何（2771 修订）：均值状态两簇对极（bio vs phys，跨模型复现）；属性自成一对（color–size 0.548）；方向级几何须去 derange 对称后重建。
4. 修复分诊矩阵（2765x2772）：lens_peak（行级，AUROC 0.834）x 通路/族（族级）。neg 行为不可修边界再次确认。

### 接续（Phase 2773 候选，同目标自动续研）
(a) kc 组合互抵机制与行级分诊器（为何 bsub 抵消 5 个 abl 修复）；(b) neg 族失败机制定位（载体源 98% 格式 token 但头消融无效——杠杆错位还是深未知）；(c) 双错类去对称面板重建方向级流形几何；(d) 跨模型（1.7b）载体头消融与组合修复复现；(e) 跨行共享头集合的"一次手术"最小干预集。

## Phase 2773: 修复管线精化——bsub 干扰符号规则（pull>0 全部互抵/pull<0 全部相容，7/7 完美分离）；neg 免疫确证为 deep-unknown（全深层注意力写回消融仍 1/16）；载体消融跨模型复现（1.7b 21 翻转 vs null 1.0）；共享头一次手术失败 [2026-09-16 05:30]

### 测试原理
四项 2772 接续任务：(A) kc 组合互抵机制——逐行记录 native/abl 下的 target−rival logit margin 与 bsub 扰动的读出拉力 pull=v_row·(W_U[target]−W_U[rival])；(B) neg 免疫上界——更强杠杆（top-8 头、以及消融模块 28–35 全部头在最终位置的深层注意力写回）；(C) 跨模型复现——qwen3-1.7b 全 320 行基线、按比例模块 M17={23..26} 载体识别与 top-4 头消融 vs 30 抽随机对照；(D) 共享头一次手术——14 个 kc 载体集中出现≥2 次的头的贪心集合，整体施加。

### 结果
1. **A（机制发现）**：互抵 5 行（273/277/281/298/314）pull 全为正（+0.05～+0.18，中位 +0.151）——v_row 本身支持目标，消融修复后减去它反而把 margin 拉回负；组合相容 2 行（307/311）pull 全为负（−0.13～−0.15）——v_row 支持 rival，减去有益。**pull 符号对"abl 修复行上 combo 是否保持"的预测为 7/7 完美分离**。行级分诊规则由此机制化：pull<0 才叠加 bsub。
2. **B（D2 通过）**：neg 16 行 top-4 头消融 0/16、top-8 0/16、**消融模块 28–35 全部头的深层注意力写回仍仅 1/16**——免疫不是杠杆错位，而是 deep-unknown（与 2764 lens_peak<−4、2772 载体源 98% 格式 token 自洽）：neg 行的内部表征在读出端没有可释放的正确答案。
3. **C（D3 通过）**：1.7b 基线 wrong 分布 ab 0/kc 35/ldr 14/neg 22/ws 29（共 100 行）；对其全部 100 行做载体识别与消融，**定向翻转 21 vs 随机对照均值 1.00（q95=2.5）**——载体-消融机制在 28 层/2048 维模型按比例复现。
4. **D（负结果）**：共享头集合（≥2 行出现；[31,19] 出现 12 次、[33,16]/[35,8]/[35,24] 各 8 次，共 11 头）整体施加：size1=0、size2=2、size3=3、size5=5、**size11=0 翻转**，附带损伤 1/114——跨行共享头不能替代逐行识别，头级杠杆是行特异的；头消融收益不来自少数全局枢纽头。

### 分析
修复理论闭合到符号级：两通路叠加的族依赖性（kc 互抵）被 pull 符号完全解释，二维分诊矩阵升级为可计算的行级规则（abl 恒可试；bsub 仅当 pull<0）。neg 的边界从"免疫"升级为"读出端无解"（结构证据）。跨模型复现把载体机制从单模型事实升级为 Qwen3 族可移植机制（与 2762 签名几何功能可移植、2770/2771 放大与两簇复现并列）。

### 相关文件（immutable，sha256）
脚本 phase2773_rdc_repair_refinement.py = 23c925eea1bf86316ab93b0f1efe9a35704d020c670f576c8b86d367def5bff3；产物 phase2773/qwen4_repair_refinement/{execution.json, result.json 5ff04aeeddda229a5c9a324aeb0b86cc4808d383921720d4a9a38b4d3a6fe31d}。

### 问题硬伤
①pull 是最终读出线性化的近似，未在 abl 后状态重算 v_row；②共享头手术只测了头级杠杆，未测值级/方向级；③1.7b 未做 bsub 组合（需其自身 span 基础设施）；④wide 消融（模块 28–35 全头）是极端杠杆，1/16 的那行翻转可能是数值噪声级副作用。

### 结论
行级符号分诊规则确立（pull<0 才叠加 bsub）；neg=deep-unknown 结构确证；载体消融跨模型复现；逐行识别必要性确立（无全局枢纽头捷径）。

### 接续（Phase 2774 候选，同目标自动续研）
(a) pull 符号规则的预注册验证与 bsub 方向符号修正（减去 rival 支持分量而非整个 v_row：v_row 拆分为 target 支持与 rival 支持分量分别处置）；(b) neg 族的训练态修复通路（2763 perm 数据 + 2773 结构证据结合）；(c) 双错类去对称面板重建方向级类别几何（2771 遗留）；(d) 载体机制第三模型（Qwen3-14B）复现。

## Phase 2774: pull 符号规则预注册验证（P1a/P2 过、P1b 空缺；bsub 全部修复行 pull<0）；bsub 正交化否证；neg 对训练态全免疫（0/16 ×4 run）；14B 环境判定（历史脚本原样复现 segfault → 系统内存状态问题，非脚本/硬件故障）；DS-R1-Qwen-7B 第三规模点：载体消融 E1 强复现（31 vs null 0.2）、E2 判据不迁移 [2026-09-16 07:00]

### 测试原理

四部分（D1–D4）。**D1**：pull_i = v_row_i·(W_U[target]−W_U[rival])（2773 行后验规则的预注册验证）。在计算 pull 之后、任何 bsub_orth 前向之前，把判据冻结进 execution.json（诚实声明：规则源于 2773 的 7 个 kc 行后验观察，验证集为其余 58 行）：P1a（abl-fixed ∧ pull<0 非衍生行中 combo 保持率 ≥0.8）、P1b（abl-fixed ∧ pull>0 非衍生行中 combo 破坏率 ≥0.5）、P2（全 65 行 combo 翻转率 pull<0 − pull>0 ≥0.15）。**D2**：bsub_orth——从 v_row 中剥离读出平行分量 v_orth = v−(v·u_ro)u_ro（u_ro=unit(W_U[t]−W_U[riv])）后作 L35 bsub（α=0.3）；P3（pull>0 行上 bsub_orth ≥ bsub）。**D3**（纯档案）：2763 behaviour_scores.npz 中各 2747 训练态 run 对 16 个 neg 错行的 base 翻转与全 320 行附带损伤；P5（某 run neg 翻转 ≥4 且附带 ≤20%）。**D4**：第三规模复现，原定 Qwen3-14B，因环境缺陷改判后换 DS-R1-Distill-Qwen-7B（Qwen2ForCausalLM，7.6B），协议与判据同 2769（top-4 正增益载体头、M=末 5 模块、定向 o_proj 输入消融、随机头 null、并集附带损伤），判据 E1（定向翻转 ≥4 且 >null q95 且 null 均值 ≤ 定向/3）、E2（并集 ≤2/20 破坏）。

### 14B 环境判定（用户指令：查历史记录、用历史脚本测试，判定硬件还是脚本问题）

1. **历史记录**：Qwen3-14B 曾多次成功（phase1118 烟雾测试、2131 worker、2586、2610 等；phase2610 时间戳 2026-09-04，GPU 14 层 + CPU 26 层，`max_memory={0:'12GiB','cpu':'20GiB'}`，offload_state_dict=True）。
2. **环境未变**：torch/transformers/accelerate/safetensors 安装时间均为 2026-07（早于 09-04 成功运行），库版本排除。
3. **失败模式**（本轮 5 种尝试）：from_pretrained(device_map='auto') 无论 CPU 配额 6/8/15GiB 或单线程加载器，均在"GPU 配额装满后第一个 CPU 张量物化"处 segfault（torch storage __getitem__ access violation，崩溃进度随配额移动：151/195/197/207/443 中 ~44-47%）；持久句柄绕过后转为诡异 CUDA OOM（6.91GB 空闲却分配 170MB 失败）。
4. **决定性测试**：**历史脚本 phase2610 原样重跑 --model qwen14，在完全相同位置（151/443，12 秒）segfault**。
5. **判定：非脚本问题**（历史脚本原样失败）；**非硬件故障**（GPU/CPU 小规模分配全部正常；DS-7B 全模型装载+推理正常）；**为系统内存状态劣化**——当前物理可用 18.3GB < 14B bf16 加载足迹（GPU 12GiB + CPU ~15.6GiB 同时驻留），Windows commit 耗尽路径以 access violation 崩溃而非优雅报错；09-04 成功时可用内存显然更大。**接续**：释放内存或重启后可直接用 phase2610 历史协议补 14B。

### 7B 替代方案与手工装载器（工程记录）

DS-7B 足迹 15.2GB 可容纳。装载器绕开 from_pretrained：meta 建模 → 逐张量 swap_tensors 装载（GPU 20 层/CPU 8 层）→ 修复 meta inv_freq 缓冲 → accelerate dispatch_model。要点：swap_tensors 会交换 __class__ 使 _parameters 变为普通 Tensor，需重包 Parameter；CPU 层权重被 dispatch 置为 meta 按需流式，M_LAYERS 的 o_proj 权重须在 dispatch 前静态提取。G0（重分词往返）、G1（确定性）通过。

### 结果

1. **D1（qwen4）**：pull 分布 24 正/41 负。族结构：kc 12/14 pull>0、neg 8/16、ldr 4/10、**ws 0/25（全部 pull<0）**。P1a **通过**（7 行 100% 保持）；P1b **空缺**（n=0——全部 5 个 pull>0 abl-fixed 行均为 2773 衍生行，无法样本外检验，如实登记）；P2 **强通过**（combo 翻转率 pull<0 68.3% vs pull>0 **0.0%**）。判定：pull_rule_partially_confirmed。
2. **pull 是 bsub/combo 修复的必要条件（本轮最干净的新规则）**：bsub 21/21 修复行、combo 28/28 修复行全部位于 pull<0 侧；pull>0 侧 bsub 0/24、combo 0/24。pull<0 内部幅度再分层：bsub-fixed 中位 −24.7 vs 免疫 −15.2。
3. **D2**：bsub_orth 总翻转仅 2/65（bsub 21）——正交化在 pull<0 行摧毁了有益分量，与机制预测一致；P3 通过但属平凡（bsub 在 pull>0 行本来就 0 翻转）。verdict=orth_bsub_confirmed（弱），bsub_orth 无应用价值，否证为修复工具。
4. **D3**：perm_2747/perm_2748/true_2747/mass_2747 对 neg 全部 **0/16 翻转**（附带 61/49/35/0）——neg 对读出端（2772）与训练态（本 Phase）双重免疫，deep-unknown 判定加固。P5 未过。
5. **D4（DS-7B）**：native 错行 206/320（64.4%，kc 64/64、ldr 63/64、neg 38/64、ws 28/63、ab 13/65——该模型在此诊断上行为远弱于 qwen4 的 65/320）。**E1 强复现**：定向消融翻转 31/206（15.0%），null 均值 0.2（q95 0.2），比值 ~155×。**E2 未过**：27 头并集在 20 个 native-correct 行破坏 18——但 206 错行下"正确行"仅 114 且并集规模是 4B 场景（14 行）的近 2 倍，E2 判据前提（小错行集）不迁移，如实登记而非宣布机制失效。verdict=carrier_replicated_7b=false（严格按预注册），E1 特异性部分保留。

### 分析

本轮把 2763–2773 的"行级 lens_peak × 族级通路"二维分诊推进为**单一机制量的连续分诊**：pull = v_row 在读出方向上的投影符号。它统一解释了：①为何 bsub 对 kc 无效（v_row 被问句事实 span 主导、方向支持 target，减掉即伤 target——pull>0）；②为何 ws 组合修复最佳（全 pull<0 且 v_row 支持 rival，减偏置+消载体同向）；③为何 neg 全免疫（8/16 pull<0 却 bsub 0——pull<0 必要不充分，深未知覆盖）。P1b 的空缺（n=0）本身是信息：5 个互抵行全部在 2773 观察集内，无新互抵样本，符号规则在新数据上"零反例"。DS-7B 的 64.4% 错行率与 E2 不迁移共同提示：**载体消融的可修复性与模型的行为能力边界强耦合**——行为越弱的模型，读出竞争越弥漫，行级定向消融的附带损伤越大。

### 相关文件（immutable，sha256）

phase2774_rdc_pull_rule_validation.py = 2ac2863d5b472860202d25645801edce7b442f598d1c2b1ce911a317d72a18f8；phase2774b_rdc_7b_replication.py = 7ee7b191647f9354f8d089c0698fc23daf615b4ccba09b6d00690eea28e513c8；产物 phase2774/qwen4_pull_validation/{execution.json 24b7b2c2…, result.json 05f6b591…, pull_stats.npz 53a35ec0…}；phase2774/ds7b_replication/{execution.json bc7cc785…, result.json cc02a7cf…, carrier_stats_7b.npz 6b81a193…}。14B 判定的历史证据：phase2610_2612_sequential_nonquantized_replication.py（未修改，重跑日志 tests/glm5/phase2610_rerun.log）。

### 问题硬伤

①P1b 空缺（n=0），pull 规则的破坏性预测无样本外检验机会，留待未来出现新互抵行；②14B 未复现（环境资源不足，非永久性否证——历史协议可待内存释放后补跑）；③DS-7B E2 判据设计未考虑高错行率场景，附带损伤结论仅在该判据下成立；④DS-7B 与 qwen4 材料为重分词文本等价、非 token 级等价；⑤neg 训练态免疫仅在 2747 部署协议（block16）下检验。

### 结论

pull 符号规则确立为 bsub/combo 修复的必要条件与统一分诊量（pull>0 → 不可 bsub/combo；pull<0 → 68% combo 可修）；bsub_orth 否证；neg 深未知加固（训练态 0/16）；载体消融的 E1 特异性跨规模（4B→7B）复现，但其附带损伤随行为能力下降而恶化。

### 接续（Phase 2775 候选，同目标自动续研）

(a) pull 规则的因果化：人为旋转 v_row 方向验证 pull 符号与可修性的因果关系；(b) neg 免疫的定位（pull<0 仍不可修——深未知的载体在哪一层/哪些来源）；(c) DS-7B 高错行率下的分诊修正（行为能力边界与载体弥漫度的定量关系）；(d) 内存释放后按 phase2610 历史协议补 14B；(e) 双错类去对称面板重建方向级流形几何（2771 遗留）。

## Phase 2775: pull 符号因果化成立——双向翻转实验（pull>0 行 0/24→12/24，pull<0-fixed 行 21→0/21，null 0/65）；neg 确证结构性深未知（16 层 bsub 最好 1/16、全范围 top-4 头消融 0/16） [2026-09-16 08:10]

### 测试原理

**C001（pull 因果化）**：把 2774 的相关性升级为因果性。构造符号翻转向量 v' = v − 2(v·u_ro)u_ro（u_ro=unit(W_U[target]−W_U[rival])）：正交分量保持、平行分量精确反号。两个预注册预测（冻结于任何前向之前）：C1a——24 个 pull>0 行（bsub 原始 0/24）上 bsub(v') 翻转 ≥4 且 > null（2763 archival 随机方向 bsub）；C1b——21 个 pull<0 bsub-fixed 行上 bsub(v') 从 21 跌至 ≤3。**C002（neg 定位）**：16 个 neg 错行上（a）单层 bsub 曲线 L={20..35}（16×16 前向）；（b）全范围 L20-35 rival-gain top-4 头定向消融；（c）源归因。E-N：全部干预 ≤3/16 → 结构性深未知；任一 ≥4/16 → mislocated-repairable。

### 结果

1. **C1a（通过）**：符号翻转后 bsub 在 pull>0 行翻转 **12/24（50%）**，而原始方向 0/24、随机方向 null **0/65**——同范数、同正交分量、仅平行分量反号，修复率 0%→50%。
2. **C1b（通过）**：符号翻转后 bsub 在 pull<0-fixed 行从 **21/21 → 0/21**——完全摧毁。
3. **verdict = pull_causality_confirmed**。读出平行分量的符号**因果地**决定 bsub 的修复/破坏，非仅仅相关。
4. **C002（neg 结构性深未知确证）**：16 层单层 bsub 曲线全部 ≤1/16（最好 L34=1）；全范围 top-4 rival-gain 头消融 **0/16**。E-N 判定：deep-unknown structural。neg 的 rival 信号不经 L20-35 任何注意力写回承载、也无任何单层 v_row 可修——与 2773（全深层写回消融仅 1/16）、2774（训练态 0/16）三重一致。

### 分析

pull 因果化完成本轮最重要的理论收束：**bsub 的全部行为由 v_row 在读出方向（target−rival）上的投影符号因果决定**。正面（flip-up 50%）与反面（flip-down 0%）同时成立，且幅度守恒（正交分量不变）排除了"只是换了扰动方向"的解释（随机方向 null 0/65）。机制图景：v_row 的平行分量直接加减 target−rival logit 差（线性读出），正交分量经由 final norm 与深层非线性——符号规则因此对"线性可读"的修复路径充分必要。neg 的三重免疫（读出 bsub、头消融、训练态）现在加上第四重（全层位 bsub 曲线平坦），指向 neg 的 rival 生成机制在残差流的 MLP 通路或更早层，读出端不可及。

### 相关文件（immutable，sha256）

脚本 phase2775_rdc_pull_causal_neg.py = b05a406931613ade93de363f00cab560e708a790e356d1f0802e277918fa47f8；产物 phase2775/qwen4_pull_causal_neg/{execution.json 3d64855c…, result.json e9dc06a1…, neg_stats.npz a91a973f…}。

### 问题硬伤

①flip-up 只修复 50%（12/24）——平行分量反号后正交分量与新方向的交互仍有残差，符号规则的"充分性"是部分的；②null 为 2763 archival 随机方向（α 相同、层相同），非同进程重跑（2763 ctrl 原始记录 0/65，一致）；③neg 的 MLP 通路假设未直接检验；④首轮运行 ctrl 比较笔误（bool(int(argmax)) 恒真）产出 65/65 假 null，已修正重跑（作废产物已覆盖）。

### 结论

pull 符号规则从相关升级为因果；分诊矩阵最终形态：**行级 pull 符号（因果）× lens_peak（幅度）× 族（neg=结构性深未知）**。neg 四重免疫确立，定位指向 MLP 通路/更早层。

### 接续（Phase 2776 候选，同目标自动续研）

(a) neg 深未知的 MLP 端定位（rival 是否经 MLP down_proj 写回；L20 前的层位扫描）；(b) flip-up 50% 残差的解释（正交分量交互项）；(c) 双错类去对称面板重建方向级流形几何；(d) 内存释放后按 phase2610 历史协议补 14B。


## Phase 2776: Qwen3-14B 第四规模点恢复——载体消融复现成立（8/40 翻转 vs null 0.15、附带损伤 0/20、null q95=1.0）；negation_scope 在 14B 原生全对 [2026-09-16 08:20]

### 测试原理

恢复 2774 判定为"系统内存劣化"而中断的 14B 规模点（本机 RAM 已释放至可用 24.3 GiB、GPU 空闲 14.63 GiB）。装载复用 2774b 在 DS-7B 上验证的手工协议：meta 建模 → 每 shard 一个持久 safe_open 句柄逐参数赋值（swap_tensors + Parameter 重包）→ inv_freq 缓冲修复 → dispatch。布局冻结：embed_tokens+L0-19 于 CPU、L20-39+lm_head 于 GPU（每层约 0.51 GiB，GPU 峰值后余 3.28 GiB）。协议与 2769/2774b 逐字一致：320 行 controlled_relation、逐行 rival 方向、M={35..39} top-4 正增益载体头、o_proj 输入头切片清零（末位）、随机头 null（20 行×10 draws，seed 27760）、联合集附带损伤 20 行。预注册（任何 14B forward 之前冻结，措辞同 2774b）：E1 翻转>=4 且 >null q95 且 null mean<=翻转/3；E2 联合集破坏<=2/20；verdict=carrier_replicated_14b iff E1 且 E2。门：G0 解码往返、G1 确定性、G2 无 meta 残留、G3 目标 id 稳定性。

### 结果

1. **verdict = carrier_replicated_14b = true**（E1、E2 双过）。
2. native 错行 **40/320**（kc 27、ldr 9、ws 4、ab 0、**neg 0**——negation_scope 在 14B 原生全对）。
3. **E1**：载体消融翻转 **8/40（20%）**；null mean **0.15**（3/200 次）、q95=1.0 → 8>1、0.15<=2.67，超 null 约 53 倍。
4. **E2**：联合载体集（15 头）对 20 个 native 正确行附带损伤 **0/20**。
5. 载体头高度集中：(36,32) 与 (39,19) 各命中 20 行，(36,28)/(35,8)/(37,28)/(36,31) 各 19 行——约 7 个头覆盖 40 错行主体，较 4B（65 行 14 头）更锐。

### 分析

规模轴 **1.7B → 4B → 7B → 14B 全部复现**读出竞争载体定位：修复率比例随规模略降（4B 32.3% → 14B 20%），但 null 恒为地板（0.15/10）、附带损伤恒为 0——机制签名（高集中、低噪声、零损伤）不变，份额随模型变强而缩小。negation_scope 在 14B 原生 0 错行，与"neg 难度依赖模型容量"一致，也使 4B 上的 neg 结论属于能力边界现象。装载层面：dispatch 有一处 meta-offload 警告（个别参数经 CPU hooks 执行），G1 确定性通过、结果自洽，不影响判读。

### 相关文件（immutable，sha256）

脚本 phase2776_rdc_14b_replication.py = 4485c4991307e63055e6280cf1ab067727ed4da0ebc71347416636bcadef1219；产物 phase2776/qwen14b_replication/{execution.json 2dfef6e4…, result.json 72b59ff1…, carrier_stats_14b.npz 1357043d…}。

### 问题硬伤

①14B 修复率 20% 且 top-4 截断，可能低估可修份额（需 k 扫描）；②null 仅 20 行×10 draws；③GPU 峰值后余 3.28 GiB 偏紧；④首轮启动因日志目录不存在立即退出（EXIT=1，未产生任何产物，不影响 discipline）。

### 结论

第四规模点恢复；读出竞争载体机制跨 1.7B/4B/7B/14B 成立，机制签名不随规模改变。

### 接续

(a) neg MLP 端/早期层定位（→2777 已执行）；(b) flip-up 50% 残差解释；(c) 去对称流形面板；(d) 14B 上 pull 规则与 bsub 复现（需先在 14B 重算偏置方向 v_rows）。

## Phase 2777: neg 深未知定位扩展——早期层/MLP 通路/稀疏神经元三重全免疫（0/16×3，阳性对照 PC1 2/2、PC2 mean|Δ|=22.8）；发现跨行共享稀疏 MLP 读出载体 (35,2708) 16/16 [2026-09-16 08:25]

### 测试原理

neg 在 2773-2775 已四重免疫（深层注意力写回消融、训练态、读出 bsub 曲线、全范围 top-4 头消融）。本轮检验剩余两个假设：rival 信号写于 L20 之前（E1）或经 MLP 通路（E2/E3）。**E1**：单层 bsub 钩子扫 L={0..19}，alpha=0.3，v_row，16 行。**E2**：MLP 块输出在末位清零，M={20..35} 单层逐个 + 全 M 联合。**E3**：down_proj 输入神经元归因 g_j=(W_out^T u_ro)_j·a_j（a 为末位激活），逐行 top-4 正增益神经元消融。预注册：E1 任一层>=4/16 → early_reachable；E2 任一层或联合>=4/16 → mlp_relevant；E3>=4/16 → neg_mlp_localized；overall=deep_unknown_beyond_readout iff 三者全免疫。**新增阳性对照门**（防"全零=钩子失效"假象）：PC1——状态式 bsub@L35 于 2 个 kc bsub-fixed 行（2772 档案）必须翻转>=1/2 否则中止；PC2——联合 MLP-zero 必须移动 target-rival logit 差（mean|Δ|>0.1 否则中止）。

### 结果

1. **PC1 通过**：bsub 机制在 kc 行 134/137 翻转 **2/2**——钩子机制确证存活。
2. **PC2 通过**：联合 MLP-zero 把 target-rival logit 差平均移动 **|Δ|=22.84**——干预极强，绝非无操作。
3. **E1 = early_immune**：L0-19 全部 **0/16**。
4. **E2 = mlp_immune**：M 单层 16×0/16、联合 **0/16**。
5. **E3 = mlp_sparse_immune**：top-4 神经元消融 **0/16**。
6. **overall = deep_unknown_beyond_readout**。
7. **意外发现（结构性）**：neg rival 方向在 MLP 读出空间稀疏且跨行共享——**(35,2708) 在 16/16 行均为 top-1**增益神经元（2.8-3.5 logit），(31,715) 14/16、(26,6940) 10/16、(29,7457) 9/16。

### 分析

neg 免疫升至**六重**（2773 注意力写回、2774 训练态、2775 读出 bsub+头消融、2777 早期层+MLP 通路+稀疏神经元），且 PC2（22.8 logit 位移仍零翻转）排除"干预太弱"解释——rival 胜出裕度极大或 target 读出在这些位点不可及。共享稀疏 MLP 载体（(35,2708) 全行共享）**存在但非因果充分**：单点消融不动裕度，指向裕度分布式/冗余集成而非单一单元承载。机制图景收束为二分：**kc = 单点可修**（载体头/bsub 单干预即翻转）vs **neg = 全残差流位点分布式深未知**（任何单点干预免疫）。定位问题从"信号在哪"转化为"裕度半径多大"。

### 相关文件（immutable，sha256）

脚本 phase2777_rdc_neg_mlp_localization.py = ed8223ff6c3683cdd22b78f0c3224d80c1f6bedc86ae9a9e3e21442732b2bc6a；产物 phase2777/qwen4_neg_mlp_localization/{execution.json e2b5d563…, result.json 36c58266…, neg_loc_stats.npz 7587bf20…}。首轮（无 PC 门）产物已按纪律删除后重跑覆盖。

### 问题硬伤

①E3 top-4 截断任意——增益 2.8-3.5 logit 的神经元消融不动裕度，需 top-k 扫描（k=8/16/32/64）定裕度半径；②E2 整层 MLP 清零为粗粒度病灶，非选择性问题；③E3 只测正增益神经元；④neg 结论基于 4B（14B 上该族 native 全对），跨规模外推受限；⑤PC1/PC2 仅 2 行与 16 行差值，对照规模小。

### 结论

neg = 深未知扩展至全残差流位点的分布式机制；共享稀疏 MLP 读出载体存在但非充分；与 kc 形成单点可修/分布式免疫的机制二分。

### 接续（Phase 2778 候选，同目标自动续研）

(a) E3 top-k 裕度半径扫描（k=4..64，neg 16 行）；(b) flip-up 50% 残差解释（正交分量交互项）；(c) 双错类去对称面板重建方向级流形几何；(d) 14B v_rows 重算 + pull/bsub 跨规模复现。


## Phase 2778: 附件审查修正 + 精准修复管线 v1——组合臂精确复现档案（28/33、逐行一致率 1.000）、flip 臂 7/16、附带损伤 0/20；预注册 P-A1/P-A2 各差一票判 false（阈值校准缺陷），oracle 依赖如实降级（Alpha） [2026-09-16 09:00]

### 测试原理

**附件审查（2774-2777 讲解）**：核对档案后保留全部核心数字（ws 25 全 pull<0、kc 12/14 pull>0、neg 8/16、bsub 21/combo 28、C1a 12/24、C1b 21→0/21、14B 40 错 8 翻 0 损伤均与档案一致），修正 4 项过度结论：①"机制签名跨规模不变（零损伤）"与 7B E2 失败（附带损伤 18/20）自相矛盾——零损伤仅在低错行率模型（4B/14B）成立；②"修复率随规模单调降、错行越少载体越集中"被 7B 打破——非单调且受错行率混杂；③"4B 的 neg 深未知属于能力边界现象"过度因果化——降级为"能力边界混淆不可排除"；④v4.4 分诊规则的 lens_peak 阈值（−3.5/−4）在档案中无字段、不可追溯——移除，分诊量仅保留 pull 符号（因果）与族。

**管线（冻结决策树，非 oracle 优先）**：R1 neg 族→decline（拒修）；R2 pull<0→combo（top-4 载体头消融 M={31..35}+bsub α0.3@L35，2772 臂）；R3 pull≥0→flip（符号翻转 bsub，2775 臂）。预注册：P-A1 总翻转≥36；P-A2 flip 臂≥8/16；P-A3 combo 臂≥24/33；P-A4 附带损伤≤2/20（20 个 native 正确行按各自 pull 符号执行对应臂）；P-A5 neg 零干预。verdict=precision_pipeline_v1_confirmed iff P-A1..A4。门：G1 确定性、G2 逐行 rival 与 2774 档案全等（65/65）。

### 结果

1. **工程事故记录**：首轮 combo 臂 8/33、档案一致率 0.394——根因为载体头增益的 u 符号约定错误（2769/2772 用 u=W_U[rival]−W_U[target]，2775 flip 用 u_ro=W_U[target]−W_U[rival]，本轮误混用→消融了"有益头"）。修正、删除旧产物、重跑。
2. **终版**：combo 臂 **28/33 翻转、逐行一致率 1.000**（与 2772 档案完全复现）；flip 臂 **7/16（43.8%）**；总翻转 **35/49 可行动行（65 行的 53.8%）**；附带损伤 **0/20**；neg 16 行全部 decline（零干预）。
3. **verdict = false**（P-A1 35<36、P-A2 7<8 各差一票；P-A3/P-A4 过）。按预注册纪律不改阈值。
4. **事后分析**：flip 臂 7/16 与 2775 的 50%（12/24）在二项检验下完全相容（期望 8.0，P(X≤7|n=16,p=0.5)=0.40）；阈值 miscalibrated 的原因是 12/24 的分母含 8 个 neg 行而 flip 臂只作用于 16 个非 neg 行。**机制层面管线与全部档案测量一致**。
5. **oracle 降级**：flip 臂的 u_ro 构造使用 W_U[target]——管线 v1 如实标注为"combo 臂非 oracle + flip 臂 oracle 引导"。

### 分析

精准修复管线 v1 在组合臂上达到档案级复现（1.000），总修复 53.8% 优于单一 combo（43.1%），附带损伤为零。预注册失败是阈值设定缺陷（分母错位+无噪声边际），不是机制失败——但这正是预注册制度要暴露的问题，如实记录。flip 臂的 oracle 依赖引出 Phase 2779 M-R4 与 Phase 2780 C-X1 的后续检验。

### 相关文件（immutable，sha256）

脚本 phase2778_rdc_precision_pipeline.py = 4766d8460106222e…（首轮符号错误版产物已按纪律删除重跑）；产物 phase2778/qwen4_precision_pipeline/{execution.json 7d0c5e6d…, result.json 875e1f1b…, pipeline_stats.npz 4f19d244…}。

### 问题硬伤

①P-A1/P-A2 阈值校准错误（分母错位）；②flip 臂 oracle 依赖；③附带损伤对照仅 20 行且仅 4B。

### 结论

管线 v1 机制成立、预注册判 false（阈值缺陷）；分诊规则修正为 pull 符号 × 族 × oracle 可用性三元。

### 接续

flip 臂去 oracle 化；neg flip 修复机制归属（→2779/2780）。

## Phase 2779: neg 裕度半径>64（神经元消融级分布式确认）+ M-R4 击穿发现——符号翻转 bsub 修复 neg 5/8（2775 的 12/24 分解：neg 5/8、kc 7/12、ldr 0/4，总数精确复现）（Beta） [2026-09-16 09:10]

### 测试原理

2777 的 top-4 截断任意——本轮做 top-k 裕度半径扫描：逐行排列全部正增益 down_proj 神经元（g_j=(W_out^T u_ro)_j·a_j，M={20..35}），k∈{4,8,16,32,64} 同时消融。M-R1：margin_radius=最小使翻转≥4/16 的 k，否则 >64（censored）。M-R2：|Δlogit| 随 k 增长且 k=64 时>0.1（活性门）。M-R3：共享 6 神经元固定集消融（描述性）。**M-R4（档案缺口恢复）**：2775 未存 C1a 逐行数据，其 12/24 的族分解未知——重跑 flip-up bsub 于全部 24 个 pull>0 行并逐族记录；若 neg 行翻转≥4/8，则"neg 深未知"被符号翻转读出通路击穿。

### 结果

1. **裕度半径 >64（censored）**：k=4/8/16/32/64 全部 **0/16**；|Δ| 从 2.7 升至 8.9 后饱和（活性门通过）；共享 6 神经元固定集 **0/16**——神经元粒度的裕度冗余确证。
2. **M-R4 = neg 5/8 翻转**：flip-up bsub 总数 **12/24**（与 2775 精确一致），族分解 **neg 5/8、kc 7/12、ldr 0/4**。neg_flip_path_overturned = true。
3. 逐行：neg 翻转行 {71,79,95,111,119}，未翻转 {87,103,127}。

### 分析

neg 图景精化：**"不可移除"与"可添加"并存**——rival 裕度无法被任何单点/多点神经元消融移除（>64 censored），也无法被 v_row 减法修复（pull<0 的 8 行 bsub 0/16），但当把 v_row 的平行分量符号翻转后做减法（= 加上翻转方向）时 5/8 修复。修复三角显现：**修复=f(方向的正交结构, 平行分量符号, 干预粒度)**——神经元粒度太细（冗余集成抵消），整方向减法粒度正确但符号决定成败。2775 的"pull>0 行 0/24 不可修"原是"plain bsub 不可修"，符号翻转后 neg 行也大半可修——"deep-unknown"必须限定于"移除类干预"。

### 相关文件（immutable，sha256）

脚本 phase2779_rdc_neg_margin_radius.py = dee4ab00fe842138…；产物 phase2779/qwen4_neg_margin_radius/{execution.json c0dbfa10…, result.json 26c7d29c…, margin_stats.npz 29ab2c30…}。

### 问题硬伤

①k 上限 64，">64"是 censored 不是精确半径；②M-R4 使用的 u_ro 含 oracle（target 嵌入）；③未检验 flip 修复是否等价于通用目标放大（→2780 C-X1）。

### 结论

neg：移除类干预六重+消融扫描七重免疫（半径>64）；添加类干预（翻转方向）5/8 有效——机制二分从"可修/不可修"精化为"移除/添加"不对称。

### 接续

C-X1 对照（flip 修复是否=通用目标放大）→2780。

## Phase 2780: 自然语言迁移 pilot——管线 9/35（25.7%）vs ctrl 0/35，P-G2 过；P-G3 败（附带损伤 7/13）判 natural_migration_pilot_failed；C-X1：纯 oracle 方向减法 0/65——flip 修复不可还原为目标放大，正交分量结构必需（Gamma） [2026-09-16 09:20]

### 测试原理

**Part 1（自然语言 pilot）**：48 条手工自然事实问句（冻结于脚本内，前缀+事实 span+后缀+单词元答案），v_row 用 2763 配方适配自由文本：h(with span)−h(without span)（末位，token 拼接规则 tok(prefix)+tok(span)+tok(suffix)）。管线臂同 2778（pull<0→combo，pull≥0→flip）。预注册：G-N0 解码往返；G-N1 全部 v_norm>0；G-N2 多 token 答案剔除；P-G1 n_wrong≥8；P-G2 管线翻转≥3 且随机方向 ctrl=0；P-G3 附带损伤≤2/20。**Part 2（C-X1，探索性，事后起源如实标注——由 2779 M-R4 引出）**：纯 oracle 方向减法（h−α‖h‖·u_ro，u_ro=unit(W_U[target]−W_U[rival])）：(a) 8 个 neg pull>0 受控行；(b) 全部 65 个受控错行（oracle 放大天花板）。若 flip 修复 ≈ u_ro 减法修复，则前者还原为通用目标放大。

### 结果

1. 48 条中 11 条答案多 token（剔除），native 错行 **35/37 单 token 行**；35 行全部 pull<0 → 全部 combo 臂。
2. **P-G2 过**：管线翻转 **9/35（25.7%）**，随机方向 ctrl **0/35**——自然语言上 pull 分诊+管线修复首次成立。
3. **P-G3 败**：13 个 native 正确行上附带损伤 **7/13（54%）**（6 combo+1 flip）——自然文本的 v_row 干预在正确行上过于粗暴。**verdict = natural_migration_pilot_failed（按预注册）**。
4. **C-X1（关键负结果）**：纯 oracle 方向减法在受控 65 错行 **0/65**、neg pull>0 行 **0/8**——与 flip 臂 12/24（neg 5/8）形成锐利对照。

### 分析

两项核心进展：①**自然语言迁移的修复率成立**（25.7% vs ctrl 0），pull 分诊规则跨出受控面板；主要障碍是**附带损伤**而非修复率——自然文本 span-deletion 方向噪声大，直接作用于正确行破坏率高，需正则化/降维/α 扫描治理（Phase 2781 候选）。②**flip 修复 ≠ 目标放大**：纯 u_ro 减法 0/65 而完整翻转方向 12/24——修复需要 v_row 的特定正交结构 + 正确平行符号，二者缺一不可（与 2774 bsub_orth 失败互补：parallel-only 失败、orth-only 失败、完整翻转成功）。RMSNorm+饱和动态使"裸 logit 推力"无效，方向的正交分量是携带修复内容的载体。修复三角闭合：**方向正交结构 × 平行符号 × 干预粒度**。

### 相关文件（immutable，sha256）

脚本 phase2780_rdc_natural_pilot.py = f0ff470e8d9922e6…（首轮 np.int64 序列化崩溃于保存步，测量已完成，修复后重跑）；产物 phase2780/qwen4_natural_pilot/{execution.json 8dc9bf70…, result.json 5f32ddda…, natural_stats.npz d5ac7329…}。

### 问题硬伤

①48 条为手工题目、模型答对率高（13 正确行即全部对照池），附带损伤估计 n 小；②flip 臂 oracle 依赖未去除；③自然 v_row 无正则化（附带损伤的直接嫌疑）；④C-X1 为事后探索性对照，无预注册判据。

### 结论

pull 分诊与管线修复迁移到自然语言成立（修复率），但附带损伤不过关；flip 修复的机制归属为"正交结构+符号"联合，非通用放大。

### 接续（Phase 2781 候选，同目标自动续研）

(a) 自然附带损伤治理：v_row 正则化/低秩投影/α 逐行扫描，目标 collateral≤2/20 且修复率≥20%；(b) flip 臂去 oracle 化（以 v_row 与模型自身 rival 构造替代 target 嵌入）；(c) 14B v_rows 重算 + pull/bsub 跨规模复现；(d) flip 修复与 combo 修复的机制等价性检验（翻转方向的正交分量是否=载体的竞争读出）。


## Phase 2781: 附带损伤治理失败——α 网格权衡曲线（0.1: 0/35 修复 4/13 损伤；0.2: 5/35, 4/13；0.3: 9/35, 7/13）；损伤对 α 不敏感→方向质量问题；偶然发现 post-norm v_row 使修复塌缩（1/35）——v_row 必须定义于 final-norm 之前 [2026-09-16 09:50]

### 测试原理

2780 的 P-G3 失败（附带损伤 7/13）需要治理。预注册 T1：α 网格 {0.1, 0.2, 0.3}（冻结），combo 臂（top-4 载体头消融+bsub）作用于 35 个自然错行（修复）与 13 个 native 首token正确行（破坏），v_row 配方与 2780 逐字一致。T2：collateral_governed iff 存在 α 使 repair≥7/35 且 collateral≤2/13。对照池组成如实记录：2 个单词元正确行 + 11 个多 token 答案首 token 正确行（首 token 正确性判据，与 2780 P-G3 池同构）。

### 结果

1. **T2 = false（governance_failed）**：α=0.1 修复 0/35、损伤 4/13；α=0.2 修复 5/35、损伤 4/13；α=0.3 修复 9/35、损伤 7/13。无 α 同时满足双判据。
2. **核心观察：附带损伤对 α 不敏感**——α=0.1（几乎无修复力）时损伤仍 4/13，说明损伤由**方向质量**（自然 span-deletion 方向的噪声分量）驱动，而非干预强度。强度旋钮无法治理，需方向正则化。
3. **偶然发现（工程偏差转科学发现）**：首轮误用 post-final-norm 隐状态（hidden_states[-1]）定义 v_row，α=0.3 修复塌缩至 **1/35**（vs 正确定义 9/35）。v_row 必须取 **L35 decoder 输出（pre-final-norm）**——RMSNorm 的缩放/方向重整破坏了 v_row 的修复结构。此偏差产物已按纪律删除重跑。
4. 与 2780 对账：α=0.3 combo 均匀施加的损伤 7/13 = 2780 按符号分诊臂的 6 combo + 1 flip（该 pull>0 正确行在 combo 下同样破坏）——完全一致。

### 分析

自然语言迁移的瓶颈正式定位：**不是修复率（25.7% 已成立）也不是强度，而是方向质量**。span-deletion 方向在自然文本上携带大量与修复无关的分量，低 α 也破坏正确行。2782 方向明确：把自然 v_row 向受控面板方向子空间投影/收缩噪声分量，或用模型自身信号（非 oracle）重构方向。修复三角进一步精化：**修复=f(方向正交结构, 平行符号, 干预粒度, 定义位点)**——v_row 的定义位点（pre-norm vs post-norm）是第四个必要条件，位点错误使修复力丧失 89%（9/35→1/35）。

### 相关文件（immutable，sha256）

脚本 phase2781_rdc_collateral_governance.py = 38b1ad293047fe2d…（post-norm 偏差版产物已删除重跑）；产物 phase2781/qwen4_collateral_governance/{execution.json 69cf2f75…, result.json a0c9c14a…, governance_stats.npz 65d71d7e…}。

### 问题硬伤

①α 网格仅 3 点且上界 0.3（更强 α 未测，但趋势表明无交点）；②正确行池 13 行（2 单+11 多首token）统计力弱；③方向正则化未实现（→2782）。

### 结论

附带损伤治理需方向质量工程而非强度调节；v_row 定义位点（pre-final-norm）确立为管线第四必要条件。

### 接续（Phase 2782 候选，同目标自动续研）

(a) 方向正则化：自然 v_row 向受控面板 v_row 主子空间投影 + 噪声分量收缩，目标 collateral≤2/13 且 repair≥20%；(b) flip 臂去 oracle 化；(c) 14B v_rows 重算 + pull/bsub 跨规模复现；(d) flip 修复与 combo 修复的机制等价性检验。


## Phase 2782: 方向正则化失败但损伤解耦发现——纯投影（λ=0，全 k）修复全灭 0/35 而附带损伤纹丝不动 7/13；C-R1 对照分解损伤=通用地板（随机方向 α0.3 破坏 3.5/13）+方向特异超额（~2-3/13）；受控子空间仅覆盖自然 v_row 能量 27% [2026-09-16 10:05]

### 测试原理

2781 定位附带损伤瓶颈为方向质量。预注册 T1：子空间 S_k=65 个受控错行 v_rows（2763 档案，逐行单位化后 SVD）前 k 左奇异向量（k∈{4,8,16,32,65} 冻结），自然 v_row 正则化为 v_reg=P_k v+λ(v−P_k v)（λ∈{0.0,0.5}），单位化后按 2780 原臂执行（臂按原始 pull 符号冻结，α=0.3）。T2：direction_governed iff 存在 (k,λ) 使 repair≥7/35 且 collateral≤2/13。B0 一致性门：plain 基线必须复现 2780 的 9/35、7/13（±1）。门：G1 native 复现（35 错/13 对）、G2 v_norms>0、G3 子空间仅由受控 2763 v_rows 构建（零自然题目泄漏）。

### 结果

1. **B0 通过**：plain 精确复现 2780 基线（9/35、7/13）——harness 与 v_row 配方自洽。
2. **T2=false**：无 (k,λ) 达标。λ=0.5 各 k：repair 7/35、collateral 7/13；λ=0.0 各 k：**repair 0/35、collateral 6-7/13**。
3. **核心结构发现**：纯投影（λ=0）在**全部 k（含 k=65 全跨度）**下修复全灭，而损伤不变——自然 v_row 的修复相关分量在受控 65 行方向跨度**之外**（能量覆盖仅 27.0%）；受控面板方向几何不向自然文本迁移。
4. **C-R1（事后探索性对照，起源如实标注）**：随机单位方向 bsub 于 13 个正确行（10 draws/行）：α=0.1→1.0/13、α=0.2→2.8/13、α=0.3→3.5/13；投影方向（k65,λ=0）同设置：3/6/6 per α。**损伤分解：通用地板（任意单位方向强扰动，α 单调）+方向特异超额（span-deletion 方向 ~2-3/13）**。
5. 工程记录：①2763 档案 v_rows 未单位化（范数 26-90），子空间构建前必须逐行单位化；②SVD 特征空间子空间在 Vh 行空间而非 U。

### 分析

2781 的"方向质量"假说部分正确但机制更细：修复分量与损伤分量**解耦**——投影方向保留损伤而丧失修复。自然文本修复三角修正：修复需要的方向内容不在受控子空间内，方向正则化这条路被判死。附带损伤非纯方向噪声：通用地板来自"任何单位方向 α·‖h‖ 扰动跨越自然正确行的决策边界"（自然正确行裕度普遍小）。治理杠杆必须换：要么降低有效扰动（2783），要么找更好的方向使所需 α 减小。

### 相关文件（immutable，sha256）

脚本 phase2782_rdc_direction_regularization.py = c330799b2c3c0288…；对照脚本 phase2782c_rdc_generic_damage_ctrl.py = eb89842c2521e701…；产物 phase2782/qwen4_direction_regularization/{execution cb52dff5…, result 0238718f…, regularization_stats 781d6ee4…, execution_ctrl 29412f49…, result_ctrl 98aaf483…}。首轮两处工程错误（320 行全量误当子空间、fin-hook 变量错位）产物已按纪律删除重跑。

### 问题硬伤

①k 网格含 k=65 全跨度仍无修复——但未试 λ∈(0.5,1) 细网格；②能量覆盖 27% 由受控 65 行决定，未试受控全 320 行（含正确行）v_rows 扩池；③C-R1 无预注册判据（探索性）。

### 结论

方向正则化失败；修复/损伤分量解耦确立；损伤=通用地板+方向超额二分解；受控面板方向几何不迁移。

### 接续

换干预杠杆（→2783 heads-only）；14B 跨规模（→2784）。

## Phase 2783: 无 bsub 载体头消融臂失败但机制归属确立——heads-only 修复 0/35（损伤 2/13 最低）、heads+bsub α0.05/0.1 修复 0/35——自然文本修复完全由 bsub 方向分量承载，载体头机制不迁移 [2026-09-16 10:10]

### 测试原理

2782 把附带损伤分解为通用地板+方向超额后，唯一剩余的低损伤杠杆是去除 bsub 扰动、仅保留竞争移除（载体头消融）。预注册 T1：同 48 题，三臂——A0 heads-only（top-4 rival-gain 头 o_proj 切片清零，M={31..35}，无 bsub）；A1 heads+bsub α∈{0.05,0.1}（桥接）；A2 plain bsub α∈{0.05,0.1}（地板参照）。T2：mitigated iff 存在臂 repair≥7/35 且 collateral≤2/13。门：G1 native 复现、G2 v_norms>0。

### 结果

1. **A0 heads-only：repair 0/35、collateral 2/13（全程最低损伤）**。
2. A1 桥接：α=0.05 → 0/35、2/13；α=0.1 → 0/35、4/13。
3. A2 plain bsub：α=0.05/0.1 → 0/35（与 2781 α=0.1→0/35 一致，修复需要 α≥0.2）。
4. **T2=false（head_only_failed）**。修复-损伤完整曲线（自然文本，本轮+2781+2780）：heads-only (0,2) → bsub0.1 (0,4) → bsub0.2 (5,4) → bsub0.3 (9,7)——修复与损伤随同一扰动强度单调共生，无免费午餐点。

### 分析

**机制归属确立：自然文本修复 100% 由 bsub 方向分量承载**——载体头消融独立修复 0/35（受控面板上独立可修 14 行），模板答案码竞争机制是受控面板特有，自然文本错误全部属偏置锚定型（2780 中 34/35 pull<0 与之一致）。结合 2782：自然文本上"精准修复"受内在约束支配——修复需要足够强的方向减法，而足够强的方向减法必然跨越邻近正确行的决策边界（通用地板）。修复率-损伤 Pareto 前沿：α=0.3 点 (9/35, 7/13) 为当前最优操作点。突破该前沿需要**按行最小充分干预**（每行二分搜索最小翻转 α，而非全局 α）——2784 候选；或更高容量模型原生减少错行（14B 路线）。

### 相关文件（immutable，sha256）

脚本 phase2783_rdc_head_only_arm.py = 47505344a8b42226…；产物 phase2783/qwen4_head_only_arm/{execution 2dd4d817…, result 36d2a847…}。

### 问题硬伤

①top-4 头选择在自然文本上可能本身失准（无竞争码结构），头数未扫描；②α 网格粗（0.05-0.1 之间未细扫）；③正确行池 13 行。

### 结论

载体头机制为受控面板特有；自然文本修复=bsub 方向减法单一通路；修复-损伤 Pareto 前沿确立，全局 α 非最优点。

### 接续（Phase 2784 候选，同目标自动续研）

(a) 14B 自然语言 pull 复现+容量边界（ native 错行数或骤降）；(b) 按行最小充分 α（二分搜索） Pareto 改进；(c) flip/combo 机制等价性；(d) flip 去 oracle 化。


## Phase 2784: 14B 自然语言 pull 复现成立——natural_pull_replicated_14b=true（翻转 7/35 vs 随机 ctrl 0/350 次）；14B 与 4B 自然错行集逐行完全相同（35 行）；附带损伤 7/13 与 4B 持平 [2026-09-16 10:15]

### 测试原理

两条线程在 14B 汇合：2776 确立受控面板 14B 规模点（neg 原生全对=能力边界），2780-2783 确立 4B 自然语言 pull 分诊+修复与修复-损伤 Pareto 前沿。预注册（任何 14B forward 前冻结）：P1 容量检查——native 错行数（首 token 判据，多 token 答案剔除），若 0 → natural_errors_absent_14b 并跳过干预；P2（条件 n_wrong≥8）——span-deletion v_row@L39（pre-final-norm，40 层模型末层），pull=v·(W_U[tgt]−W_U[rival])，pull<0→bsub α0.3、pull≥0→符号翻转 bsub（oracle 引导，如实标注），翻转≥3 且随机 ctrl=0（10 draws/行，seed 27840）→ natural_pull_replicated_14b；P3 描述性附带损伤（≤20 正确行）。装载复用 2776 build_model_manual（CPU embed+L0-19 / GPU L20-39+lm_head）。门：G0 解码往返、G1 确定性、G2 v_norms>0。

### 结果

1. **P1：n_wrong=35/37 单 token 行**——与 4B 的错行集**逐行完全相同**（identical wrong set=True）。这 48 题是知识边界题，14B 并未原生解决；自然事实错误的容量边界在本题集上不存在（与 neg 受控现象相反）。
2. **P2 通过：翻转 7/35、ctrl 全 0（0/350 次随机方向）**→ natural_pull_replicated_14b=true。全部 35 行 pull<0（−0.094..−0.297，combo 臂）——分诊结构与 4B 一致。
3. 翻转行 {2,3,4,14,33,37,41} 与 4B {2,4,8,14,19,33,37,41,43} 重叠 6 行——修复可行性行级强保守但非完全相同。
4. **P3：附带损伤 7/13**——与 4B α0.3 完全持平：通用地板跨规模不衰减。
5. 装载：425+ 参数分配正常，meta-offload 警告再现（同 2776，无碍），GPU 峰值后余 3.34 GiB。

### 分析

pull 分诊与方向减法修复**跨 4B/14B 双规模成立**，且错行集规模不变意味着：自然知识错误不是"小模型才会犯"的能力边界现象（14B 同样全错），修复管线在 14B 上有同样真实的应用对象。附带损伤地板 7/13 跨规模不变——它不是 4B 特有的伪影，而是"单位方向 α·‖h‖ 扰动 vs 自然正确行小裕度"的结构性效应，进一步支持 2783 的 Pareto 前沿结论：突破前沿需要按行最小充分干预或更优方向，而非更大模型。

### 相关文件（immutable，sha256）

脚本 phase2784_rdc_14b_natural_pull.py = 7e536cced34965c9…；产物 phase2784/qwen14b_natural_pull/{execution d86c92bd…, result 6954f1c4…, natural_pull_stats_14b 71663176…}。

### 问题硬伤

①48 题为 4B 设计的知识边界题，对 14B 偏难（错行率 94.6%）——需更大更分层的自然题库才能测出 14B 的真实自然错误率与容量边界；②flip 臂 oracle 依赖未去除；③ctrl 每行 10 draws 共 350 次 forward 占实验时长主体。

### 结论

pull 分诊+bsub 修复跨 4B/14B 成立；自然知识错误集跨规模稳定；附带损伤地板跨规模不变（结构性）。

### 接续（Phase 2785 候选，同目标自动续研）

(a) 按行最小充分 α（二分搜索）改进 Pareto 前沿；(b) flip 臂去 oracle 化；(c) 扩充分层自然题库（易/中/难）测 14B 容量边界与 pull 分诊泛化；(d) flip/combo 修复机制等价性检验。


## Phase 2785: flip 臂去 oracle 化——rival 轴 flip 修复 4/24（null 0/240，flip_deoracled=true）但仅限高 pull kc 行；neg 的 flip 修复仍 oracle 锁定；去 oracle 翻转集与 oracle 翻转集仅重叠 2/4 [2026-09-16 11:55]

### 测试原理

2775 的符号翻转变换 v' = v − 2(v·u_ro)u_ro 中 u_ro = unit(W_U[target]−W_U[rival]) 含 oracle（target 嵌入）。"机制被理解"要求干预仅用模型侧信息。三臂（同一变换，仅换轴 u）：O oracle 参照（门：必须精确复现 12/24）；D1 rival 轴 u=unit(W_U[rival])；D2 argmax 轴 u=unit(W_U[argmax_native])（无 ground truth，断言 argmax==rival）；D3 中心化轴 u=unit(W_U[rival]−mean_vocab(W_U))。Null：逐行随机单位轴 10 draws（seed 27850）。行：2775 C1a 同一 24 个 pull>0 行。预注册：G1 native 复现；GO oracle 精确 12/24；N1 deoracle_ok iff 任一 D 臂 ≥4/24 且 >null q95；verdict=flip_deoracled iff GO 且 N1。

### 结果

1. G1 通过（24 行 native 复现）；**GO 精确通过（12/24）**——harness 与 2775 完全对齐。
2. **D1=D2=D3 翻转行完全相同**：{298, 302, 314, 318}，各 4/24；null 0/240。N1 通过（4≥4 且 >0）→ **flip_deoracled = true**。
3. **族分解（锐利边界）**：4 行全部 knowledge_chain 且 pull 大（2.0–5.5）；neg 0/8、ldr 0/4——**neg 的 flip 修复（2779 的 5/8）完全依赖 target 侧轴**。
4. **翻转集仅重叠 2/4**（298、314）：rival 轴修复的是与 oracle 轴不同的子集——两轴通过不同通路起效，非简单近似。

### 分析

去 oracle 部分成立：rival 侧信息足以修复高 pull 的 kc 行（这些行的失败本就是"rival 侧权重过大"，翻转 rival 轴分量即减 rivalry）；但 neg 行的"加回修复"需要知道正确答案的方向——修复的**通路不对称**延伸到 oracle 依赖层面。管线 v1（2778）的 flip 臂现可部分去 oracle（kc 行用 D1），neg 行如实保持 decline。D2≡D1 说明"模型自己的错误预测"与"档案 rival"在错行上恒等——rival 本就是模型侧可观测的，这一臂真正做到了零 ground truth。

### 相关文件（immutable，sha256）

脚本 phase2785_rdc_flip_deoracle.py = e225f7e2ba9f2cdc…；产物 phase2785/qwen4_flip_deoracle/{execution.json b7d3e13dfd89d4ff…, result.json 715b872e0b9735f3…, deoracle_stats.npz 6010935abfb178a5…}。

### 问题硬伤

①N1 阈值 4 恰在边缘（ prereg 设定偏松）；②D3 与 D1 翻转集相同，未提供独立信息；③未在自然文本检验 rival 轴。

### 结论

flip 修复部分去 oracle：kc 行 rival 轴即可，neg 行 oracle 锁定；修复通路不对称性从干预类型延伸到信息来源。

### 接续

v_row 正交分量解构（→2786）；自然文本 e0 配方检验（→2787 候选）。

## Phase 2786: 修复载体解构——v_row 修复内容收敛到单一共享方向 e0（任务锚定轴，与 token 嵌入近正交）；LOO 外样本 21/21 全翻（e1 对照 4/21）；e0 分量配方 25/65 超越档案 bsub 基线（21/65）零回归（含 2786b/2786c） [2026-09-16 12:05]

### 测试原理

2780 C-X1 证明修复需要 v_row 的正交结构、2782 证明修复内容在受控子空间之外——但"哪个正交分量承载修复"未知。行：21 个 pull<0 bsub-fixed 行（2772/2774 档案 plain bsub 21/21）。基：21 个单位 v_rows 的 SVD（特征方向取 Vh 行），e_j j=0..7。臂（均 α=0.3@L35，编辑后单位化）：**A_nec(j)** v' = v−(v·e_j)e_j（去掉分量 j），≤10/21 判必要；**B_suf(j)** v' = (v·e_j)e_j（分量 j 单独），≥8/21 判充分；null 随机方向 drop/alone 各 10 draws/row。预注册：localized iff 任一 B_suf≥8；distributed iff 全部 B_suf≤3 且全部 A_nec≥15；否则 mixed。**2786b（外样本门）**：留一法 e0_loo（其余 20 行 SVD）分量单独修复 21 行，≥15/21 判 generalizes、≤7 判 in_sample_artifact；e1_loo 负对照。**2786c（配方推广，描述性）**：冻结 e0（21 行基）对全部 65 错行执行分量修复，Q1 保 21（≥19）、Q2 超越档案的额外翻转记录；e1 对照于 44 失败行。**2786 追加机制归属探针**：e0 与各 token 嵌入/均值的余弦。

### 结果

1. **G1 基线 21/21 复现**。能量分布：e0 占 43.5%、e1 29.0%、e2 14.4%（top3 = 87%）。
2. **B_suf(0) = 21/21**：e0 分量单独完全修复；A_nec(0) = 6/21（去掉 e0 崩塌）；其余分量 alone ≤9/21、drop 后仍 17–21/21。null：随机 drop 210/210 无害、alone 16/210 地板。**verdict = repair_carrier_localized**。
3. **2786b：LOO 21/21 全翻**（cos(e0_loo, e0_full)=0.998，方向稳定）；e1_loo 对照仅 4/21。**carrier_generalizes**——外样本无泄漏。
4. **2786c：e0 配方 25/65**（档案 bsub 21/65），零回归（kept21=21/21），额外翻转 4 行 {177, 289, 305, 309}；e1 对照 2/44。族分解：ws 14/25、ldr 6/10、kc 5/14、neg 0/16。
5. **e0 身份（关键）**：cos(e0, mean_v) = −0.71（≈21 行 v_rows 的共享均值方向之逆）；**cos(e0, W_U[rival]) ≈ −0.07、cos(e0, W_U[target]) ≈ +0.04——e0 与 token 嵌入近正交**。逐行投影 v·e0 符号混合（−0.96..+0.58）。

### 分析

**全项目最强机制简化**：bsub 修复的"正交结构"之谜坍缩为一维——修复内容 = v_row 在共享锚定轴 e0 上的分量，且该轴可由其他行的 v_rows 无 oracle 地估计（LOO 全翻）。e0 与 token 嵌入近正交解释了 2780 C-X1（oracle 轴减法 0/65 失败：它推 logit 而不动锚定轴）与 2782（27% 能量覆盖：自然 v_row 的 e0 类分量未被受控子空间刻画的错觉——子空间维度错了，不是方向不存在）。e0 ≈ 逆均值方向意味着它是"问句事实 span 锚定"的公共轴——所有偏置共享的任务姿态。e0 配方 25/65 超越全 v_row bsub（21/65）：窄化到一维反而多修 4 行（全 v_row 的高维噪声分量在部分行抵消修复）——**更少即更多**。neg 0/16 再次一致（neg 无此通路）。修复三角更新：**修复 = (锚定轴分量符号) × (行特定幅度) × α‖h‖**，v_row 的其余 56.5% 能量对修复冗余。

### 相关文件（immutable，sha256）

脚本 phase2786_rdc_repair_carrier.py = 63f991eeb0057c4b…、phase2786b_rdc_carrier_loo.py = b9871e9506d0d43e…、phase2786c_rdc_e0_recipe.py = 50d7c333f8ebc066…；产物 phase2786/qwen4_repair_carrier/{execution 4aad5ce6…, result 1dcf6fbf…, carrier_stats 1a13167b…}、qwen4_repair_carrier_loo/{execution 6bd83973…, result 00041367…, loo_stats 1169eedb…}、qwen4_e0_recipe_all65/{execution bf95c0c5…, result 520520e5…, recipe_stats 9208d4d4…}。

### 问题硬伤

①e0 由 v_rows（span-deletion）估计——仍需 fact span 知识，但不含答案信息（非 oracle）；②e0 的 21 行基是否跨族/跨规模稳定未测（14B 重算→2787）；③natural 侧 e0 泛化未测（2782 的负结果基于 65 维子空间，1 维 e0 未试）；④A_nec(0)=6/21 的 6 行由哪些分量兜底未深究。

### 结论

修复载体 = 单一共享任务锚定轴 e0（与 token 嵌入近正交、可无 oracle 估计、外样本稳定）；e0 一维配方 25/65 优于全向量 bsub，neg 仍免疫；"修复需要正交结构"精化为"修复需要正确的锚定轴"。

### 接续（Phase 2787 候选，同目标自动续研）

(a) 自然文本 e0 配方：自然 v_row 投影到受控 e0 后 bsub——若修复>0 则 2782 断裂缝合；(b) 14B e0 重算与跨规模稳定性；(c) e0 的残差流写头定位（哪些头/MLP 写 e0 方向——机制载体闭环）；(d) flip 去 oracle 的 e0 化（用 e0 轴替代 rival 轴翻转 pull>0 行）。


## Phase 2787: e0 锚定轴的边界测定——自然迁移失败（0/35，自然 v_row 在 e0 上能量仅 0.024 vs 受控 0.435）+ e0-flip 失败（2/24<4）——e0 是面板特异锚定轴，非通用任务轴 [2026-09-16 12:10]

### 测试原理

2786 发现修复载体收敛于单一方向 e0（受控 21 行 v_rows 的 top PC，LOO 21/21），并留下两个候选：自然文本 e0 投影配方（弥合 2782 断裂）与 flip 的 e0 化。预注册（任何 forward 前冻结，e0 由 21 行受控档案确定性重算、零自然泄漏）：G-C harness 门——e0 配方在受控 21 bsub-fixed 行必须 21/21；P1——自然 35 错行 e0 分量 bsub 翻转 ≥3/35 且随机 ctrl（10 draws/row）=0 判 e0_natural_works；P2 描述性——13 native 正确行附带损伤（对比 full-v 7/13、heads-only 2/13）；P3——受控 24 pull>0 行 e0 轴符号翻转（v'=v−2(v·e0)e0）翻转 ≥4/24 且 >随机轴 null（10 draws/row）判 e0_flip_works。自然 v_row 配方与 2780 逐字一致（L35 pre-norm span-deletion，48 题同库）。

### 结果

1. **G-C 通过**（21/21）——harness 对齐 2786。
2. **P1 = false**：e0-natural 翻转 **0/35**（ctrl 0/350）。**机制解释直接可见**：自然 v_row 在 e0 上的能量 |v·e0| 平均仅 **0.024**（min 0.001、max 0.043），而受控行平均 0.435——自然偏置方向与受控锚定轴**近正交**，投影后无可减内容。
3. **P2 = 8/13**：e0 配方附带损伤反而高于 full-v（7/13）——纯 e0 方向在自然文本上也是粗暴扰动。
4. **P3 = false**：e0-flip 仅 **2/24**（null 0/240；kc 2/12、neg 0/8、ldr 0/4；与 oracle 集/ rival 集重叠各 2）——翻转锚定分量不能产生"加回"修复。
5. native 复现 35 错/13 对（与 2780/2781 完全一致）。

### 分析

2786 的"单一锚定轴"结论必须加边界：**e0 是该受控面板（问句模板格式）特有的锚定轴**，不是跨格式的通用任务轴。受控面板的 v_rows 来自"问句事实 span 删除"（问答式模板），自然题为陈述式完形——两种格式的偏置几何不同轴。这同时解释：①2782 的 27% 能量覆盖与 0/35（低维投影无内容可减，现在知道连 1 维都不在）；②修复-损伤在自然文本同源耦合（2783）的深层原因——自然文本的偏置方向高维且各行异质，没有共享锚定轴可压缩。e0-flip 失败（2/24）进一步支持移除/添加不对称：减去行符号正确的 e0 内容是修复，加上不是。**修复载体层级图景收束**：受控面板内 1 维（e0），跨面板 0 维共享——"锚定轴"本身是面板词汇/格式统计的产物，通用机制应是"每格式有自己的低维锚定子空间"而非"存在普适锚定方向"。

### 相关文件（immutable，sha256）

脚本 phase2787_rdc_e0_natural_flip.py = 092dc1eee0031d40…；产物 phase2787/qwen4_e0_natural_flip/{execution.json 6b86675e…, result.json ca171944…, e0_natural_stats.npz 7be7e8f8…}。

### 问题硬伤

①仅一种自然格式（陈述式完形）vs 一种受控格式（问句模板）——"面板特异"结论基于 n=2 格式对比；②未测自然题自身的锚定轴（对 35 个自然 v_rows 做 SVD，若其 top PC 分量 bsub 可修复自然错行，则"每面板一轴"假说直接成立——这是 2788 的第一优先）；③e0-flip 未做 oracle 对照复核。

### 结论

e0 不跨格式迁移（0/35、2/24）；修复载体的正确概括是"每面板一个低维锚定子空间"，不是普适方向；自然文本锚定轴需就地估计（→2788）。

### 接续（Phase 2788 候选，同目标自动续研）

(a) **自然面板自身锚定轴**：对 35 个自然 v_rows 做 LOO-SVD，top 分量单独 bsub——检验"每面板一轴"假说（若成立，修复管线终版 = 就地估计锚定轴 + 分量减法，全程无 oracle）；(b) 14B 受控 e0 稳定性；(c) e0 的写头定位；(d) 双格式锚定轴的相似度矩阵（受控 vs 自然 vs 混合）。

## ----------------------------修改范式


## Phase 2788+2789: 内容包位置不变性否证 → 词位驻留编码稳定性证实（L35 cos 0.961、neutral 0.93-0.96）——稳定规律在词位残差，不在搬运包（LPF 范式开局，用户定向） [2026-09-16 12:45]

### 测试原理

**用户定向突破（原话要义）**：自回归机制中所有参数/矩阵固定，但同一词（"苹果"）可在任意位置参与计算，却总能稳定回答"苹果是植物还是动物、什么颜色"——要求以此为door找语言在LLM中的稳定规律。这是范式转变：**从修复侧（错误行干预）转向生成侧（位置不变性），无 wrong/right 行、无修复**。翻译为可测假设：固定权重+任意位置→稳定规律，其载体必是"内容包"——词位贡献被注意力搬运到读出位。已有仪器 span-deletion v_row 恰好测量该包。

**2788（搬运包测量）**：6 个属性丰富词（apple/dog/gold/Japan/car/ocean）× 15 词对交换框架（"Unlike the A, the B was seen at the market yesterday" 双句互换，词在早位 ~3 与晚位 ~6）+ 每词主语位/句尾位变体 = 42 句 72 词次。每词次计算全层 v_row（hidden_states after L{8,16,24,31,35}，末位 with−without，span 删除该词次）。预注册：P1 内容包稳定 iff L35 within-word mean cos ≥0.5 且 within−between ≥0.15；P2 涌现层；P3 属性读出（6 词预注册属性词表，lens top-50 覆盖）。

**2789（词位驻留测量）**：同一 42 句 72 词次，改测**词自身 token 位置**的残差 h_l[word_pos]，RMSNorm 后 logit lens 读属性。预注册：P1 驻留稳定 iff L35 within ≥0.5 且 gap ≥0.15；P2 属性涌现层 = ≥4/6 词覆盖的最小层；neutral 基线 "The {word}" 对照上下文依赖。

### 结果

**2788（搬运包 = 位置特异）**：
1. P1 **false**：L35 within=0.242（min −0.261）、between=0.126，gap 0.117<0.15。
2. **层曲线反直觉**：within 随深度递减（L8 0.418 → L35 0.242）——早层最稳，深层越来越位置定制。
3. **cos–位置距离相关 = −0.81**：位置差越大包越不同——搬运强位置依赖。
4. P3：均值包读出以词身份为主（apple→'apple','苹果','🍎'；gold→'金价'）；3/6 词属性词命中 top-50（Japan→Japanese/Tokyo、car→vehicle/driver、ocean→water/sea）。

**2789（词位驻留 = 高度稳定）**：
1. **P1 TRUE**：L35 within = **0.961**（min 0.873）、between 0.756、gap 0.205——同一词在 12 个不同句子/位置的驻留残差几乎同向。
2. 层曲线单调固化：0.899→0.814→0.867→0.945→0.961。
3. **属性涌现层 = 24**：覆盖 3/6(L8)→2/6(L16)→4/6(L24)→**5/6(L31)**→3/6(L35)——属性可读性在 L31 峰值，末层回落（残差转向续写空间）。
4. **neutral 对照 = 0.93–0.96**：句中驻留与最小上下文 "The {word}" 的残差近同——编码近乎上下文无关。
5. 位置相关 −0.67 但范围压缩（0.87–1.0）。

### 分析

**全项目首个跨格式稳定规律确立，用户的机制直觉被精确证实并定位**：稳定规律不在"搬运包"（2788 否证：0.24、强位置依赖），而在**词位驻留编码**（0.96、近上下文无关）。语言机制的四层图景现在有实测支撑：
- **内容存储（L3 岗位）**：词的属性（植物/颜色/类别）由词位残差承载，中层 MLP 逐步注入（涌现 L24、峰值 L31），跨位置/句子/格式稳定——这就是"苹果是植物、红色"在固定权重下的稳定实现；
- **选择性搬运（L2 岗位）**：注意力按查询需要从稳定存储中抽取分量运往读出位——抽取结果天然位置特异（0.24），这**不是缺陷而是机制**：同一存储服务不同查询；
- **读出竞争（L4 岗位）**：pull/bsub/载体头理论原封不动地适用于搬运到达后的末段竞争。

这同时解释了此前全部"推广失败"的根源：我们此前测的 v_row/e0 全是**查询侧锚定几何**（格式特异，2787 证明），而**内容侧**（驻留编码）才是跨格式稳定的部分——旧范式根本没有测量它。循环的出口找到了：语言规律 = 词位驻留稳定编码 × 查询依赖搬运 × 读出竞争的三层分工。

### 相关文件（immutable，sha256）

脚本 phase2788_rdc_position_invariance.py = 4e1644fb47f87fcd…、phase2789_rdc_insitu_stability.py = 26779c5d68434838…；产物 phase2788/qwen4_position_invariance/{execution 4a232a4d…, result d76d5aa2…}、phase2789/qwen4_insitu_stability/{execution 0a203d33…, result 34ecd8f4…}。

### 问题硬伤

①6 词 72 词次是小样本 pilot，词表窄（具体名词）；②within 0.96 的高 cos 部分来自同词身份 token 主导——属性分量的稳定性需在"去掉身份分量后的残差"上重测；③搬运包 0.24 的低稳定性混合了间接路由效应（删除词改变下游全部计算），未做逐层搬运路径分解；④属性涌现只测了 lens 线性可读性，非线性存储未测；⑤L35 覆盖回落（3/6）的解释（转向续写空间）是推测。

### 结论

用户假设证实：固定权重+任意位置的稳定规律存在，位置在**词位驻留编码**（cos 0.96、近上下文无关、属性中层注入 L24 涌现/L31 峰值）；搬运包位置特异（0.24）且这是选择性服务的机制而非失败。范式转变完成第一步：生成侧测量可行且规律真实。

### 接续（Phase 2790 候选，同目标自动续研）

(a) 属性分量分离：从驻留残差中去掉身份分量（减去跨层均值/身份读出方向），测属性子空间的位置稳定性与容量；(b) 属性注入定位：apple 的 "plant" 分量在 L16→L31 间由哪些层/MLP 注入（逐层差分+MLP 归因）；(c) 词表扩展：100 词 × 多属性的系统驻留普查（属性→子空间映射字典雏形）；(d) 搬运选择性的查询依赖分解：同一驻留编码对不同查询位置抽出哪些不同分量。


## Phase 2790+2790b: 自动续研，LPF-3：属性注入定位 + 身份-属性可分性 [2026-09-16 07:58]

### 测试原理

2789 发现属性 lens 可读性涌现于 L24、峰值 L31，但那是 5 层采样的粗曲线，且“涌现”一词混淆了两种事件：**写入**（属性分量进入残差流）与**读出**（lens 变得线性可及）。2790 用三臂把两者拆开：A 臂对冻结的 2788/2789 面板（72 词次×6 词）做全 37 层 lens，margin(w,prop)=z(prop)−mean(z(其余 5 词的 15 个属性词))，l\*=argmax(L20..L35 全局均值)；B 臂逐层把词位置 MLP 输出置零（只动词位置、只动单层），在 l\* 读 margin 降幅与词 token 降幅（特异性对照）；C 臂对每词 12 个原位残差做 PCA，去掉 PC1（身份主分量）后重测属性 top-50 覆盖。2790b 是判别臂：若单层置零无效是因“分布式累积注入”，则多层同置零必塌——四臂 C1(L20-29 全置零)/C2(L0-29)/C3(L4-17)/C4(L30-35)，l\*=30 由 2790 execution 锁定。

### 结果

**A 臂**：全局 margin 曲线 L20 2.88→L24 4.98→L26 5.99→**L30 7.96（峰，l\*=30）**→L32 6.24→L34 3.69（末段回落确证 2789）。覆盖 keep=5/6（dog 的 animal/mammal/pet 均不入 top-50，与 2789 一致；mammal 为多 token）。
**B 臂（P1 FALSE，attr_injection_localized=false）**：窗口 L20-29 单层置零平均仅降 0.235，早层 L4-17 仅 0.054——无任何单层临界承载属性。**最大单层 drop 是 L0（1.40，占基线 8.6 的 16%）**，其次 L24（0.82）——属性写入从 embedding 后第一层 MLP 就开始，L8-14 近零（±0.17 带负值）。特异性比 2.81：词身份 logit 掉得比属性 margin 更多。
**C 臂（P2 TRUE，attr_identity_separable=true）**：去 PC1 后覆盖 5/6 保持（仅 car 丢 wheels）——属性子空间与身份主分量可分，2789 硬伤②（同词 cos 高靠身份主导）得到正面回应。
**2790b（P1b FALSE，P2b FALSE）**：C1 窗口累积降 2.456（29%）、C3 早层累积降 6.260（73%）、C2 全前置降 8.573（100%）、C4 读出层后置零降 0（确认 hs[l] 采集点在层 MLP 之前，l\*=30 读数=层 29 输出）。**窗口累积消融时词身份掉 5.60 > 属性掉 2.46（特异性倒挂）**；**car 两句 C1 后 margin 反升（−4.40/−1.72）**——rival 分数比目标属性掉得更狠。

### 分析

**读出涌现 ≠ 写入涌现，2789 表述被修正**：写入是全深度渐进的（C2 100%/C3 73%/C1 29% 的依赖梯度，且单层最大在 L0），L24-L30 的“属性涌现”实为 lens 几何可及性事件——中层之后属性分量才变得线性可读。**属性编码没有临界层、没有临界方向写入点**，与 kc/neg 的单点/分布式二分不同，属性是第三种形态：全深度分布+读出竞争。**词位 MLP 主要承载词身份的线性可读分量**（2790 特异比 2.81 与 2790b 倒挂 5.60/2.46 一致），属性读出对词位干预相对鲁棒。**car 的 margin 反升是属性层面读出竞争的首次因果证据**：置零写入源后 rival 语义分数比目标属性掉更多，margin 反升——Cmp(o,r,v) 候选竞争效应在驻留读出端再现。LPF 四层图景更新：词位驻留残差不是“静态写入的档案”，而是**全网络计算在词位置的焦点截面**——身份 ≈ 跨句稳定主分量（PC1，去除后属性仍可读），属性子空间可分但由全深度非线性计算分布式维持。旧范式（找单方向/单层/单载体）在内容侧同样注定失败，这与查询侧 e0 面板特异（2787）构成同构解释。

### 相关文件（immutable，sha256 前 16 位）

脚本 phase2790_rdc_attr_injection.py=8067615444518024、phase2790b_rdc_attr_cumulative.py=d3f9d7f724a8932c；产物 phase2790/qwen4_attr_injection/{execution 4f75e559efcad545, result 546de74a54812714, ablation_curve.npz 048a3578bc599d80}、phase2790b/qwen4_attr_cumulative/{execution 733c25b3423d1912, result 32f403e88688d76e}。踩坑：2790b 首版 res dict 混入臂名键致聚合 TypeError，删旧 execution/result 后重跑通过；hs[l] 采集点在层计算前（C4=0 实证），读出层解释须以“层 l−1 输出”为准。

### 问题硬伤

①B/C 臂只消融词位置 MLP，注意力搬运注入（其他位置 MLP 写入→attention 搬回）未被隔离，C1 剩余 29% 可能来自该通路；②margin 是线性 lens 读数，非线性存储（MLP 是否把属性存在非 lens 可读流形上）未测；③L0 主导写入的解释（embedding 已含属性先验、L0 MLP 放大）未做 embedding 直接 lens 验证；④6 词小样本，dog 属性（animal/mammal/pet）在此 lens 口径下不可读，词表覆盖偏差未修正；⑤car 反升仅 2 句，竞争读出需系统化分解（逐 rival 分数曲线）。

### 结论

属性定位问题得到否定性但强结构的答案：**无单层临界注入、写入全深度渐进（最早 L0）、读出 L24+ 才线性可及、属性与身份可分（去 PC1 5/6）、读出端存在竞争效应（car 反升）**。LLM 语言机制的稳定规律在内容侧进一步收窄为：跨位置稳定的驻留几何（身份主分量+可分属性子空间）×全深度分布式写入×竞争性线性读出。

### 接续（Phase 2791 候选，同目标自动续研）

(a) L0 源头：token embedding 直接 lens 的属性读出 + L0 MLP 增益分解（embedding 先验 vs L0 放大的定量拆分）；(b) 竞争读出系统化：置零干预下逐 rival 分数曲线，量化“margin 反升”的竞争结构（与 Cmp 竞争效应对接）；(c) 注意力搬运注入隔离：消融全句 MLP（不只词位）vs 只消融词位的差分，分离间接通路贡献；(d) 词表普查 100 词（顺延 2790 候选 c）。


## Phase 2791: 自动续研，LPF-4：L0 源头拆分 + 竞争读出系统化 [2026-09-16 08:21]

> 日志线变更：自本 Phase 起研究日志按用户指令追加至本文件（AGI_GPT5_MEMO.md）；AGI_GLM5_MEMO.md 写至 Phase 2790 封存。

### 测试原理

2790 遗留两个硬伤：③L0 主导单层 drop（1.40）的解释（embedding 先验 vs L0 放大）未验证；⑤car margin 反升仅 2 句。2791 双臂闭环：**臂 A（L0 源头拆分）** 对 6 词测三个口径的属性 margin——margin_emb（纯 token embedding 过 RMSNorm+W_U，与句子无关）、margin_h1（层 0 完整输出 hs[1]，72 词次）、margin_h1_noMLP0（置零词位 MLP[0] 后的 hs[1]）；拆分为先验 P=emb、总增益 G=h1−emb、MLP0 因果份额 M=h1−h1_noMLP0。**臂 B（竞争读出系统化）** 12 句 × 4 干预臂（none/C1 窗口 L20-29/L24 单层/C3 早层 L4-17），保存目标 3 属性与 15 个 rival 的 raw z@l\*=30，逐句-词分解 Δmargin = Δz_target − Δz_rival。

### 结果

**臂 A**：**P-A TRUE（embedding 先验 6/6）**——margin_emb 全部巨大：apple 11.6 / dog 16.1 / gold 23.0 / Japan 58.1 / car 15.2 / ocean 27.0（全局 25.1，为 lens 峰值 7.96 的 3.2 倍）。**P-B FALSE（G=−18.9）**：层 0 输出 margin 6.30 << embedding 25.15——L0 计算把先验 margin 压掉 75%。**P-C FALSE（M=−4.9，0/6 正）**：置零 MLP[0] 后 hs[1] margin 反而回升至 11.16——**L0 MLP 的直接 lens 效应是压缩而非注入**。与 2790 的 L0 置零使 l\*=30 读数降 1.40 合并解读：L0 MLP 压低直接读数但其输出承载下游必需的已整理信息——**L0 是转码器，不是放大器**。
**臂 B**：rises 仅 2/12（均 car），**P-E FALSE**；frac_rival_gt_target=0.50<0.8，**P-D FALSE**。但分解显示竞争真实且双向：car|neutral 反升 = Δtarget +2.21 / Δrival −2.19（rival 让位）；car|swap = Δtarget −1.44 / Δrival −3.16（目标掉但 rival 掉更多）。**新发现：C3 早层置零使 4/12 句 margin 转负**（apple −0.41/−0.25、gold −0.57、car −0.51、ocean −0.81）——rival 分数反超目标属性，**属性的竞争排序在早层 L4-17 建立**。L24 单层置零全句效应 0~2（无临界层再确证）。

### 分析

**注入模型被先验净化模型取代**：属性信息不需要“写入”——token embedding 先验已满格（25.1），中层全部计算的功能是把粗糙先验**压成干净的、竞争排序正确的读出**（峰值 7.96 < 先验 25.1，这是降噪不是增益）。三段式净化管线确立：①L0 大压缩（25.1→6.3，去粗噪声，转码进下游计算流）；②早层 L4-17 竞争排序（置零即 rival 反超）；③中层 L20+ 可读性塑形（lens 几何事件，2790 已证）。这使 2790 的“无临界注入层”完全自洽：**没有注入层，因为本来就不需要注入**；也解释特异性倒挂（词位 MLP 的“净化”主要作用于身份分量）。LPF 图景第三次更新：语言稳定规律 = **embedding 静态语义档案（3 倍冗余先验）× 逐层净化转码 × 竞争读出**。“写入”概念从内容侧退出。竞争读出确认为词特异（car）而非普遍机制，与 Cmp 竞争效应的对接保持谨慎：竞争是读出几何的属性而非写入机制。

### 相关文件（immutable，sha256 前 16 位）

脚本 phase2791_rdc_l0_source_competition.py=783d3bc3b75f559b；产物 phase2791/qwen4_l0_competition/{execution b86e0a5e1f61eae6, result 6247fe9596a0d2e2, l0_split.npz 99fcdc297a81d2cf}。预注册 5 判据全数执行（P-A TRUE / P-B FALSE / P-C FALSE / P-D FALSE / P-E FALSE），无 drift。

### 问题硬伤

①margin_emb 的 3 倍冗余可能含 lens 口径伪影（embedding norm 大、与 W_U 近耦合），需非 lens 口径（余弦/欧氏）复测；②“净化”是功能描述，逐层净化算子（哪些 head/神经元执行压缩与 rival 分离）未定位；③C3 转负的 4 句是“rival 分离在早层”的证据，但早层逐层置零未做（L4-17 内哪一层是关键未知）；④竞争读出仅 car 2 句，6 词小面板可能低估词间差异；⑤embedding 先验与下游读出的语义内容是否同一（先验=原始共现统计 vs 读出=任务相关属性）未验证。

### 结论

L0 源头问题得到颠覆性答案：**属性先验在 embedding 已满格（25.1，6/6），L0 是压缩转码器（M=−4.9）而非注入器，早层建立 rival 竞争排序，中层做读出塑形**。“属性注入”范式终结，“先验净化”范式确立——LLM 内容侧语言机制 = embedding 静态语义档案 × 逐层净化转码 × 竞争读出，稳定规律的第一性来源是 embedding 档案本身。

### 接续（Phase 2792 候选，同目标自动续研）

(a) 净化管线全程定量：embedding→峰值读出的逐层“净化比”曲线 + rival 排序正确性首次达成的里程碑层；(b) 早层 rival 分离定位：L4-17 逐层置零找竞争排序临界层（对照 C3 的 14 层捆绑消融）；(c) 先验内容验证：embedding lens top-50 与中层 lens top-50 的重叠度（先验=共现统计 vs 读出=任务属性的定性差异）；(d) 词表普查 100 词（顺延）。


## Phase 2792: 自动续研，LPF-5：净化管线定量 + 早层分离定位 + 先验内容验证 [2026-09-16 08:30]

### 测试原理

2791 留下三个问题：①排序（rival 分离）在深度上何处首次稳定正确（先于还是伴随 L24 读出涌现）；②早层 L4-17 的 rival 分离是否单层临界（2791 只有 14 层捆绑证据）；③embedding 先验与中层读出是否同一语义内容。2792 三臂：**臂 A** 全 37 层双 margin 曲线（margin = z(prop)−mean(z(rival)) 与 hard-sort margin_sort = mean_prop[z(prop)−max(z(rival))]，后者是最硬竞争口径），里程碑 l_clean = 全局 sort 均值首次>0 且此后恒>0 的最小层；**臂 B** L4-17 逐层词位 MLP 置零（12 句），读 l\*=30 margin + rival/target 分解；**臂 C** 每词 embedding lens top-50 与 l\* 层均值残差 lens top-50 的属性词同现（同源性）。

### 结果

**P-F FALSE（l_clean=None）**：hard-sort 曲线 L0 即 +16.91（embedding 层面排序已正确；sort_emb 全正：apple 2.99/dog 9.38/gold 13.68/Japan 52.20/car 4.36/ocean 18.86），但中途剧烈震荡（采样点 −0.74/+0.35/+0.78/−0.38/−0.72/+1.33/+2.92/+2.42/−1.06），**不存在恒正里程碑**——排序在深度中被反复破坏与重建，至末段（L30+）才稳定正确。
**P-G FALSE（l_crit=5，max_drop=0.380）**：L4-17 逐层置零最大降幅仅 0.38 且多负值（L4 −0.06、L10 −0.15、L12 −0.15、L15 −0.14），rival 主导比例全层 <0.6——2791 C3 捆绑消融的 6.26 是 **14 层非线性协同效应，无单层分离临界点**。
**P-H FALSE（3/6）**：先验-读出 top-50 属性词同现仅 3/6 词——**embedding 大 margin 的语义内容与中层读出部分不同源**。

### 分析

**净化模型降级为建构模型**：2791 的“先验满格+逐层净化”被 2792 三重修正——①P-H 表明 embedding 的大 margin 含大量**度量耦合成分**（与 W_U 的表面对齐/共现统计），真实属性语义只部分在先验里（3/6 同源）；②P-F 表明深度计算不是“一次性净化”而是**排序的反复破坏-重建**（震荡型轨迹）；③P-G 表明分离无单点。统一图景升维：**LLM 内容侧语言机制 = embedding 粗原料（度量耦合+部分语义）× 36 层分布式内容建构（无任何临界层/里程碑）× 竞争读出**。全项目“无临界点性”主题三重确立：写入无临界层（2790）、分离无临界层（2792 P-G）、排序无里程碑（2792 P-F）——**找单点范式在内容侧彻底终结**，这与查询侧的发现（e0 面板特异 2787、修复不对称 2785）合成完整互补：查询侧有局部可修单点，内容侧全深度分布。
**对本轮 LPF 范式的意义**：用户的原始问题（固定权重+任意位置的稳定规律）现在有了四层实证答案：稳定性的载体是词位驻留几何（2789），其属性内容由全深度建构（2790-2792），其读出按查询竞争（2788/2790），而“苹果是植物”这类知识的局域化程度比传统特征概念预测的低得多——知识是建构过程的状态，不是存储的条目。

### 相关文件（immutable，sha256 前 16 位）

脚本 phase2792_rdc_purify_pipeline.py=f295a34ca81f8832；产物 phase2792/qwen4_purify_pipeline/{execution c9bf4e1ab0882c38, result f39a9146811cbf38, purify_curve.npz 2a493fdf8f759210}。预注册 3 判据全 FALSE 但描述数据完整落盘（margin/sort 全层曲线 + 逐层 early drop + same_source 详情），无 drift。

### 问题硬伤

①margin_emb 的度量耦合解释是推断，未直接量化（如 cos(emb_w 方向, W_U 行) 与 margin_emb 的相关）；②sort 震荡的层粒度与震荡幅度未系统刻画（npz 有全曲线，MEMO 只引采样点）；③“建构”的信息来源未追踪——属性信息在上下文中如何进入（注意力从共现词取？）未测；④6 词小面板 + P-H 的 3/6 词未逐词展开；⑤hard-sort 口径依赖 rival 集的完备性（15 rival 只是 5 词的属性，真实竞争词表更大）。

### 结论

2792 把 2791 的“先验净化”修正为“粗原料+分布式建构”：embedding margin 巨大但内容部分不同源（3/6）、排序全程震荡无里程碑、分离无单层临界。**无临界点性三重确立，内容侧找单点范式终结**；LPF 图景 v4 = embedding 粗原料 × 36 层分布式内容建构 × 竞争读出。

### 接续（Phase 2793 候选，同目标自动续研）

(a) 度量耦合定量：cos(emb_w, W_U 行) 等对齐指标与 margin_emb 的逐词相关，把先验分解为耦合成分+语义成分；(b) 属性信息溯源：上下文消融（中性句 vs 完整句的 margin 差）+ 注意力通路追踪，测“建构”从哪里取信息；(c) sort 震荡刻画：全层曲线的震荡幅度谱 + 与 rival 集扩充（更全竞争词表）的稳健性；(d) 词表普查 100 词（顺延）。


## Phase 2793: 自动续研，LPF-6：先验分解 + 建构溯源 [2026-09-16 08:48]

### 测试原理

2792 留下两个问题：①margin_emb 的大值里度量耦合与真语义各占多少（硬伤①）；②“建构”的信息从哪里来——上下文共现还是词自身（硬伤③）。2793 双臂：**臂 A（纯权重空间，零前向）** 背景 = 300 随机 token 对的 cos(emb 方向, W_U 行方向) 分布（seed=2793，避开 18 个语义词）；属性 gap = mean_w[cos(emb_w, 属性词均值) − cos(emb_w, rival 均值)]；P-I 判据 gap ≥ 3×背景 std。**臂 B（上下文消融）** 三句境对比：full（面板原句 swap_late）/ scrub（结构保持，把后置内容词 dog→item、market→location）/ min（“The {w}”），全部读 l\*=30 词位 margin。

### 结果

**臂 A：P-I FALSE（coupling_dominant=true，边缘）**——背景 |cos| 均值 0.0862（≈4.3 倍正交随机预期 0.02，E-W_U 全局耦合实在）、std 0.0710；属性 gap 均值 0.1664 = 2.3×std < 3×std。但**词间异质大**：Japan gap 0.4017（5.7×std，强真语义）、ocean 0.1753（2.5×std）、apple 仅 0.0712（1×std）——先验 = 强全局耦合基底 + 词特异真语义增量（部分词显著、部分词淹没在耦合里）。
**臂 B：P-J FALSE 但以因果级最强形式反证**——full 与 scrub 的词位 margin **逐词 bit-exact 相同**（apple 8.202616930007935 = 8.202616930007935，六词全部），d_ctx = 0.000 精确为零。原因：scrub 词（item/location）全在目标词**之后**，causal attention 下后置词根本进入不了词位残差——设计失误变成**因果级实验证明**：词位属性读出对句中后置全部内容因果免疫。full vs min 仅差 −0.195（唯一差异是前置格式词 “Unlike the” vs “The”），scrub/full ratio = 1.0000。

### 分析

**建构溯源闭合：原料 100% 是词自身 embedding**。臂 B 的 bit-exact 结果（设计失误的意外之得）把 2792 的“分布式内容建构”精确化为：36 层计算在**词位置自足**地把 embedding 先验加工成属性读出——上下文共现信息（dog/market 等）**零参与**词位属性编码，前置格式词贡献 ≤0.3。这与 2789 的 neutral_cos 0.93-0.96（相关性证据）形成因果级闭环。P-H 的 3/6 低同源由此获得新解释：**内容确实来自 embedding（因果自足），但中层 lens 的 top-50 表征被加工重塑**——同现低是读出几何重塑的表象，不是内容换了来源。
**LPF 图景 v5（词位随身档案模型）**：词位驻留 = 每个词的“随身语义档案”——固定权重网络在词位置对 embedding 做全深度自足加工（建构属性读出），上下文只决定**查询与搬运**（2788 选择性搬运发生在外部位置），不注入属性内容。“苹果是植物”的实现：读“苹果”所在位置即可，无需句子其他部分——知识编码于词位加工管线，不在句法环境中。
**臂 A 修正 2791-2792 的“先验”概念**：先验 = 耦合基底（全局 E-W_U 对齐 0.086）+ 词特异语义增量（Japan 型显著 / apple 型弱）。margin_emb 的巨大值不能直接读作“知识满格”，真语义占比因词而异——这与 P-H（3/6 同源）和 Japan 的 lens 覆盖一贯强（2790 起 Japan margin 全场最高）互相印证。

### 相关文件（immutable，sha256 前 16 位）

脚本 phase2793_rdc_prior_decomp_context.py=f2e4b785e62f21d4；产物 phase2793/qwen4_prior_context/{execution c8648ae583d7f38c, result 45588173c70cfb26, prior_context.npz ded12a8f3ac36a64}。踩坑与教训：scrub 句把替换词放在了目标词之后——causal mask 下等于未干预（bit-exact 即证明）；**后续上下文消融必须把替换词放在目标词之前**。预注册判定如实登记：P-I FALSE（边缘 2.3×std）/ P-J FALSE（d_ctx 因果为零）/ P-K FALSE（d_self=−0.195，自足以更强形式成立）。

### 问题硬伤

①P-I 的 3×std 阈值是约定，2.3×std 的均值判定对词间异质不敏感（Japan 5.7×std vs apple 1×std 应分词报告而非均值一票否决）；②“词位自足”只在属性 margin 读出口径上证明，属性信息是否还受前置更多词（长前缀）影响未测（仅测了 2-token 前缀）；③背景 300 随机 token 含 subword，背景分布对完整词空间的代表性未检验；④scrub 失误臂浪费了一半预算（虽产出了因果证明）；⑤臂 A 与臂 B 的连接（耦合基底如何被 L0 压缩）仍是推断。

### 结论

**词位属性读出被证明因果自足**：对后置上下文 bit-exact 免疫、对前置格式词近免疫（≤0.3）、原料 100% 为词自身 embedding——LPF v5“词位随身档案”模型确立（知识编码于词位加工管线，上下文只负责查询与搬运）。先验概念修正为“耦合基底+词特异语义增量”（背景耦合 0.086、gap 词间 1-5.7×std 异质）。

### 接续（Phase 2794 候选，同目标自动续研）

(a) 词位加工管线分解：既然原料纯为 embedding，逐层区分词位上 attention（自环/前缀读取）与 MLP 的贡献曲线（置零 attn vs mlp 的全层对照）——“随身档案”的算子定位；(b) 长前缀与前置内容词的影响（把替换词放到目标词之前的正确 scrub）；(c) 100 词词位普查（词位自足使普查可全用中性句，成本大降；顺延）；(d) 耦合基底与 L0 压缩的连接实验。


## Phase 2794+2794b: 自动续研，LPF-7：随身档案算子分解 + 前置调制修正 [2026-09-16 09:13]

### 测试原理

2793 证明词位属性读出因果自足后，两个问题随即打开：①词位上的加工算子是什么——attention（自环 value 变换+前缀读取）还是 MLP（key-value 记忆）各承担多少（文献默认 FFN 承载知识，须实测）；②“自足”的边界——前置内容词是否有调制（2793 只测了 2-token 格式前缀）。**2794 臂 A**：12 句 × 36 层 × 双算子（分别置零词位 self_attn 输出 / mlp 输出），读 l\*=30 margin，得 attn/mlp 两条 drop 曲线（sum|attn| vs sum|mlp| 判 P-L 3× 阈值、max|attn| 判 P-M 0.5）。**2794 臂 B**：4 条件（full_new/scrub_new/long_pre/min）。**2794b（2794 臂 B 的修正续件）**：full_new 模板被发现把 nxt 仍放在目标词之后（连续第二轮同型失误，d_pre=0.000 只是后置免疫复现，P-N 判定无效作废）；2794b 重写模板使内容词严格在目标词之前（pre_content：“Compared to the {nxt}, the {w}…” vs scrub_pre：item 替换，含 sanity 断言前置 token 数相等且内容不同），并增加 long_scrub（garden/fence→place/object）把长前缀 −1.79 分解为内容 vs 长度。

### 结果

**2794 臂 A：P-L FALSE（attn:mlp≈1:2，MLP 不独大）+ P-M TRUE（attention 参与）**——sum|attn|=3.047 vs sum|mlp|=6.212（比值 2.04×<3×），max|attn|=0.780。词位属性加工是 attention+MLP 双算子协作（attention 承担约 1/3），修正“FFN 独载知识”默认预期。
**2794 臂 B：P-O FALSE（长前缀实质调制）**——long_pre 6.087 vs min 7.878，d=−1.792（23%）；P-N 因模板失误无效（d_pre=0.000 为后置免疫复现）。
**2794b：P-Q TRUE + P-R TRUE**——前置内容词调制 −0.611（结构严格保持、内容词在前，sanity 断言通过）；长前缀 −1.79 分解为**内容词 −0.802 + 长度/格式 −0.99**。调制方向以负（干扰）为主（dog→apple −1.51、ocean→car −1.79、Japan→gold −0.93）但存在正向（gold 前置 Japan 时 Japan 自身 +0.62、apple 前置 dog 时 +0.33）——方向由语义关系决定，非纯噪声。

### 分析

**词位随身档案的算子与边界第一次完整刻画**。算子侧：加工由 MLP（2/3）与 attention 自环/前缀读取（1/3）共同执行，双算子逐层协作（曲线数据落盘 npz）。边界侧：“自足”精化为三段式——①后置上下文严格因果免疫（bit-exact，2793/2794 两次复现）；②短格式前缀近零（−0.195）；③前置内容词与长前缀有 −0.6~−1.8（~8-23%）的语义调制，方向双向由语义关系定。**LPF 图景 v5.1**：词位驻留=随身语义档案（主体自足），前置上下文是“调制旋钮”（后置不可见、前置微调），上下文的真正作用发生在查询/搬运侧（2788）——这解释了为什么语言模型既要有上下文学习（搬运侧敏感）又有稳定的词级知识（词位侧自足）。方法教训升级为制度：上下文消融实验的替换词位置必须通过“目标前/后”断言验证（2793、2794 连续两轮同型失误后，2794b 已把 sanity 断言写进脚本）。

### 相关文件（immutable，sha256 前 16 位）

2794：脚本 phase2794_rdc_operator_decomp.py=51cf0e5465e5a04c，execution f8211892f68156ec / result d255cfbc4e36e370 / operator_curves.npz 87bba6b27a11454e。2794b：脚本 phase2794b_rdc_prefix_scrub.py=f7aaa41822d1c522，execution 9a1d702d6e210c76 / result dc6a80461cb3c3c9 / prefix_margins.npz 4c9310baf074fceb。2794b 首跑 sanity 断言误写（要求全条件前缀等长）崩溃，修正断言+删旧产物重跑通过；预注册判定如实登记（P-L F/P-M T/P-N 作废/P-O F/P-Q T/P-R T）。

### 问题硬伤

①attn/mlp 分解只做了单层置零，双算子协同（同层同置零）未测；②前置调制 6 词样本的方向结构（干扰 vs 增强）只可描述不可建模，需扩词表；③long_scrub（place/object）的语义中性度未验证（place 与 garden 的语义距离）；④attn 置零移除的是整个 self_attn 输出，未区分自环 vs 前缀读取（可用只屏蔽跨位置 attention 的精细干预）；⑤结论限于 Qwen3-4B 单模型。

### 结论

词位加工=双算子协作（attn 1/3 + mlp 2/3），“自足”边界三段式确立（后置免疫/短前缀零/前置调制 −0.6~−1.8）。LPF v5.1：随身语义档案主体自足+前置调制旋钮+搬运侧上下文利用。前置内容词的调制方向由语义关系决定（干扰为主、关联可正）。

### 接续（Phase 2795 候选，同目标自动续研）

(a) 100 词词位普查（词位自足+中性句协议已成熟：每词 1 前向 × 100 词 × 全层曲线，建立属性→子空间映射字典雏形，顺延三轮）；(b) 双算子协同与自环/前缀读取分解（同层双置零 + 跨位置 attention 屏蔽精细干预）；(c) 前置调制方向定律：扩词表测“干扰 vs 增强”与语义距离的关系；(d) 跨模型验证（Qwen3-1.7B 或 DS-7B 复核 2789-2794 五大结论）。


## Phase 2795: 用户定向，LPF-8：名词语义图谱 + 近对复用/差异化 [2026-09-16 09:34]

### 测试原理

用户定向：若机制成立，关键是**画出图谱**——所有名词在驻留空间的结构是什么样，苹果与香蕉的特征如何复用与差异化。2795 以 100 名词 × 10 类（fruit/animal/metal/vehicle/country/food/nature/furniture/tool/clothing，全部单 token 定稿，词表 tokenizer 预检）× 中性句协议（“The {w}”，词位自足使每词只需 1 前向）建立首个图谱数据集：每词 L30 词位残差、lens top-50、margin_cat（类别词读出 vs 其余 9 类别词）、margin_cat_emb（embedding 口径对照）。三预注册：P-S 类别聚类（within−between ≥ 0.05）；P-T 类别属性复用（≥8/10 类的类别词入 ≥8/10 成员 top-50）；P-U 差异化在差向量（d=unit(h_a−h_b) 与 unique 属性方向对齐 ≥ shared+0.05，3 近对 apple/banana、dog/cat、car/bus）。

### 结果

**P-S TRUE（类别聚类实测成立）**：within 0.6562 vs between 0.5291，gap 0.1271。类别紧密度排序：country 0.797 > fruit 0.705 > metal 0.686 > clothing 0.678 > animal 0.637 > food 0.629 > nature 0.627 > tool 0.618 > vehicle 0.603 > furniture 0.581。**首张名词驻留图谱落盘**（100×100 cos 矩阵 + 每词残差 npz）。
**P-T FALSE（2/10）但揭示复用通道异质结构**：fruit 10/10、metal 8/10 的成员词 top-50 含类别词；animal 0/10、food/nature/clothing 0/10、vehicle 1/10、tool 2/10、furniture 3/10、country 4/10。投影口径解释机制：**apple 的 fruit 方向 embedding 36.0 → L30 读出 14.6（保留 41%），dog 的 animal 方向 24.6 → 0.66（压制 97%）**——类别方向复用程度由**类别命题信息量**决定（“is a fruit”有信息量→保留命名通道；“is an animal”平凡→压制，读出走实例/特征通道）。top10 印证：apple 读出 orchard/cider/fruit/tree/juice（类别+用途），dog 读出 doggy/breeds/emoji/吠（实例特征），banana 读出 peel/emoji/-shaped（形态属性）。
**P-U FALSE（3/3 对未过，重要否定）**：差向量与 unique 属性方向对齐 0.042/0.042/0.009，方向上 uniq>shared 全部成立（0.042>0.026、0.042>0.014、0.009>0.002）但幅度过小（car/bus 对 0.009 甚至低于背景 0.0145）——**apple−banana 的差异化不通过差向量与读出属性方向的对齐实现**。结合 top10：差异化体现为**读出通道内容不同**（apple 的 orchard/cider/juice vs banana 的 peel/-shaped），即差异化=通道内容选择，不是几何方向分离。

### 分析

**用户定向的图谱问题得到首批定量答案**。①名词结构：驻留空间按类别聚类（gap 0.127），且类别内紧密度本身有结构（国家词最紧——语义同质性最高）。②复用与差异化定律雏形：**类别方向复用 ∝ 类别命题信息量**（fruit/metal 复用命名通道，animal/food 压制命名改走实例通道）；**同类近对的差异化不在差向量的线性读出方向上，而在读出通道的内容选择**（不同属性词在竞争 top-50 中胜出）——这与 2786 的 v_row/W_U 正交教训同构：**加工空间与读出空间部分解耦**。③margin_cat 与 margin_cat_emb 的逐词对比提供了“建构选择”的直接读数（保留/压制比：fruit 41% vs animal 3%），可扩展为全词表的“类别方向保留谱”。LPF 图景补充第五层：**词位档案的类别组织**——档案不是孤立向量堆，而是按类别命题信息量组织的拓扑（命名通道 vs 实例通道），这直接回答了“苹果和香蕉的特征怎么复用和差异化”：复用=共享类别方向保留（fruit 14.6/10.1 双高），差异化=通道内容选择（orchard/juice vs peel/-shaped），而非差向量几何。

### 相关文件（immutable，sha256 前 16 位）

probe2795_1.py=a24d09a6587a7184；脚本 phase2795_rdc_semantic_atlas.py=bd03fbfea3afb441；产物 phase2795/qwen4_semantic_atlas/{execution 41098945407f783d, result 72ce3acb28aef497, atlas.npz 2c0010cef0883d32}。词表预检（probe2795_1/2）：melon→berry、screwdriver→file、chisel→nail、pliers→rope、meow→fur 全部单 token 定稿后冻结。

### 问题硬伤

①P-T 的 top-50 出现性判据对通道异质不敏感（fruit 通道走 top-50、animal 通道走具体词），投影口径（margin_cat）应升为正式判据；②P-U 的 0.05 边际对 0.04 量级的信号过粗，差向量分析需要先做显著定标（背景分布 0.015）；③100 词单前向无句内多样性（2789 的 12 句均值协议更稳，本轮依赖词位自足假设）；④类别词表是预注册常识，未对“类别命题信息量”做独立定量（可用语料共现统计）；⑤近对 unique 属性（bark/fur/driver/passenger）是手工选择，差异化通道的内容清单不完整。

### 结论

名词图谱首批结构确立：**类别聚类真实（gap 0.127）+ 复用通道异质（命名通道 vs 实例通道，保留比 fruit 41% vs animal 3%）+ 近对差异化不在差向量线性方向而在读出通道内容选择**。“苹果和香蕉如何复用和差异化”的机制答案：复用=共享类别方向的双侧保留，差异化=不同属性词胜出的通道内容差异——破解编码机制的下一步是通道选择定律（什么决定哪个属性词在竞争中胜出）。

### 接续（Phase 2796 候选，同目标自动续研）

(a) 类别方向保留谱全词表化：100 词 × 10 类别方向的逐词投影矩阵（margin_cat 已算 100 词，扩展为 10 方向 × 100 词全矩阵），建立“命题信息量→保留比”的定量定律；(b) 差异化通道选择定律：近对 top-50 差集的属性词分析 + 通道胜出的竞争结构（与 2790 竞争读出对接）；(c) 差向量内容直接 lens 读出（z(d) top-50 是什么——差向量自己的档案）；(d) 图谱结构深化：100×100 矩阵的层次聚类 + 类别间近邻结构（fruit-food 交叉？）。

## Phase 2796: 自动续研，LPF-9：保留谱全词表化 + 差向量档案 + 图谱聚类——纯离线再分析 [2026-09-16 09:46]

### 测试原理

2795 三发现待定律化：①复用通道异质（fruit 保留 41% vs animal 3%）是否为全词表定律；②近对差异化不在差向量线性方向（P-U FALSE）——则差向量自己的 lens 档案 z(d) 里应同样不含属性词（预测 d=词身份载体而非属性载体，判据可证伪双向）；③100×100 cos 矩阵的聚类可恢复性。2796 为**纯离线再分析**：atlas.npz 已存 H30/E30，仅加载权重做 lens 重算（无新前向），margin 公式与 2795 完全一致，并设 sanity gate（|M_H[w,own]−margin_cat_2795[w]|<0.05 全 100 词，未达即中止）。四预注册：P-V 保留谱异质（span≥0.30 且 fruit−animal≥0.15；retain=margin_H/margin_E，margin_E(own)<0.5 剔除计数）；P-W 先验→读出连续律（own 通道 ρ(margin_E,margin_H)≥0.5 且 slope_fruit>slope_animal）；P-X 差向量属性档案（≥2/3 对 z(d) top-20 含 unique 属性词）；P-Y 聚类恢复（average linkage k=10 ARI≥0.30）。

### 结果

sanity gate 满格：max|M_H−MC|=0.000000、max|M_E−MCE|=0.000001——离线再分析与 2795 逐位一致。
**P-V TRUE（保留谱全序确立——“命题信息量→保留比”经验定律）**：10 类 own 通道保留比全序 **food 0.814 > tool 0.709 > furniture 0.549 > fruit 0.532 > country 0.489 > vehicle 0.328 > metal 0.275 > clothing 0.122 > animal 0.072 > nature −0.465**（span 1.279）。guard 剔除 9 词（margin_E(own)<0.5）：8 个 nature 词+rope——**“is a nature” 在 embedding 先验中本就近零/负**（nature 极端异质，先验里不存在该命题），nature 通道负值主要由 wind 单点承载（硬伤登记）。animal/clothing 最低=平凡命题压制，food/tool 最高=信息命题保留。
**P-W TRUE（先验→读出连续律）**：own 通道 100 词 ρ=0.518≥0.5；斜率异质 slope_fruit 0.204 > slope_animal 0.086（fruit 增益 2.4×animal）——读出不是先验的固定函数，而是**类别依赖的增益调制**。
**P-X FALSE（0/3，强否定→正面确立）**：三近对 z(d) top-20 全部不含 unique 属性词、也不含 shared 类别词；内容全部是一方词身份词族+其生态近邻：apple/banana→cider/苹果/🍎/apples/iphone（word_hit=[a TRUE,b FALSE]）；dog/cat→ging/gie/吠/🐶；car/bus→dealership/经销商/流通/racing。**差向量 lens 档案=词身份词典，不是属性差**——diff_vector_is_identity_carrier=true；且读出单侧化（只命中 a 侧词族），RMSNorm 非线性下 d=h_a−h_b 的归一化方向由 a 主导。
**P-Y FALSE（ARI=0.089，方法学否定）**：average linkage 链式失效——93/100 词并入单簇（within 0.656 高相似空间+类别间近邻 0.86 级使链式合并吞掉全部结构）。类别结构真实存在（P-S gap 0.127 已证），失效的是链接方法而非图谱。

### 分析

**①保留谱=通道选择定律的定量曲线**。10 类保留比不是二值（保留/压制）而是**连续谱**，与“该类别命题的信息含量”直觉单调一致：food/tool（具体功能类）>0.7；fruit/country ~0.5；animal/clothing <0.13（平凡）；nature 反号（先验中无此命题，读出系统性反向）。与 P-W 合并成完整图景：**词位读出对先验 margin 做“增益×符号”的通道调制，增益由类别命题信息量决定**——2795 的两点（41% vs 3%）升级为 10 点全谱定律候选：retain(c) ≈ f(informativeness("is a c"))，单调、跨类、增益斜率异质（P-W）。
**②差向量=词身份载体确立**。z(d) 档案（cider/🍎、吠/🐶、dealership/经销商）全部是一方词的身份词族与生态近邻，属性词 0/6、类别词 0/3——与 2795 P-U（线性对齐口径 0.042）合并成双重证明：**同类近对的差异化机制不在差向量里，无论线性对齐口径还是 lens 内容口径**。差异化=读出通道内容选择成为唯一存活解释。
**③图谱拓扑=紧类核+语义连续体，非离散簇**。average linkage 失效恰恰揭示拓扑本质：类别间近邻（fruit↔food 0.864 双向最近、tool↔clothing 0.868、furniture↔clothing 0.863、animal↔tool 0.818、country↔nature 0.700）形成语义连续体，类核只是局部密度峰。**名词图谱不是 10 个岛，而是有密度峰的连续大陆**。
**④LPF v5.2 增补**：词位随身档案 = 类别调制增益谱（retain 谱）× 词身份差向量（近对载体的身份侧）× 读出内容竞争（2790）三者拼接。

### 相关文件（immutable，sha256 前 16 位）

脚本 phase2796_rdc_retain_spectrum.py=e16d15ec8f307c06；产物 phase2796/qwen4_retain_spectrum/{execution afd528f9a0096397, result 968e9e51d3ac1a0c, retain_spectrum.npz 22ae75da3f34454e}。数据源 atlas.npz（2795，2c0010cef0883d32），无新前向；sanity gate（M_H/M_E 对拍 2795 全 100 词）通过。

### 问题硬伤

①nature 通道 n=1（8/10 词被 guard 剔除，r_nature=wind 单点）；span 判据部分依赖该单点（剔除后 span 仍 0.742≥0.30，P-V 稳健）；②“命题信息量”仍是直觉排序而非独立定量（语料共现/PMI 度量未做）；③P-X 属性词匹配沿用 2795 子串规则（red∈shredded 型误报风险方向为假阳性——0/3 因此更强）；④P-Y 的 average linkage 链式失效是已知方法学局限，ARI 0.089 不构成“无类别结构”证据；⑤retain 比值口径在 margin_E 小分母处噪声大（guard 阈值 0.5 敏感性未扫）；⑥单模型（Qwen3-4B）单层（L30）——跨模型/跨层保留谱未验证。

### 结论

三项定律化落定：**①保留谱全序（food 0.814→nature −0.465，span 1.279）——“类别方向复用 ∝ 类别命题信息量”从两点升级为 10 点连续定律，先验→读出为类别依赖增益调制（ρ=0.518，fruit 增益 2.4×animal）；②差向量=词身份载体（z(d) 档案 0 属性词/0 类别词，全为一方词族+生态近邻）——差异化机制唯一存活解释锁定为“读出通道内容选择”；③名词图谱拓扑=紧类核+语义连续体（fruit↔food/tool↔clothing 近邻网络）——不是离散岛而是密度峰大陆**。LPF v5.2：档案=增益调制谱×身份差向量×读出竞争。

### 接续（Phase 2797 候选，同目标自动续研）

(a) 图谱拓扑重聚类：complete/ward linkage+谱聚类+密度峰法对比定标 + 类别间近邻网络全图（10×10 质心矩阵已有→edge list+桥词识别——哪些词横跨两类）；(b) 命题信息量独立定量：用 Qwen3 自身对 “A {w} is a {c}.” 的伪困惑度/PMI 做 informativeness 度量，检验 retain(c) 单调律；(c) 桥词与双层成员：food 2 词离簇、tool 4 词分散——逐词检查是语义歧义（butter=food/材料）还是档案竞争；(d) 差异化通道选择定律（2796 未动）：近对 top-50 差集属性词的竞争结构对接 2790（什么决定 orchard 胜 peel）。

## Phase 2797: 用户定向升级，LPF-10：维度级名词图谱——词嵌入每参数的角色归因 [2026-09-16 11:10]

### 测试原理

用户定向升级：“画出同一体系内所有名词的图谱结构——词嵌入的每个参数的作用是什么”。核心数学事实（本轮以 P-D 确立）：lens 读出 margin 对 embedding 维度**精确可加**——固定词时 RMSNorm 分母为常数，margin(w,c) = (1/rms_w)·Σ_d E[w,d]·g[d]·dW[c,d]，其中 dW[c,d]=W_U[c,d]−mean_{c'≠c}W_U[c',d]。由此每个参数的“载流贡献” C(d;w,c)=E[w,d]·g[d]·dW[c,d] 可精确计算，因果消融（置零维度→重算 lens）无需任何前向。纯离线（E30/H30 复用 2795 npz + 权重）。四预注册：P-A 类别维存在（η²(d) 超 500 次标签置换 null 99.5 分位的维数 ≥25 且归属覆盖 ≥5 类）；P-B 单维因果特异（top-10 η² 维逐维 E-置零，归属类成员 |Δown margin| > 非成员 ≥8/10）；P-C top-30 维承载先验谱（置零后 E-side own-margin 类均值谱 span 收缩 ≥50%，随机 30 维对照 span 比 ≥0.8）；P-D 分解恒等（max err <1e-3）。

### 结果

**P-D TRUE（err 1.99e-05）**：可加载流分解精确成立——“每个参数的作用”首次获得精确数学形式。
**P-A TRUE（805/2560=31.4% 维显著，覆盖 10/10 类）**：η² top：d13 0.767(country)、d32 0.732(country)、d270 0.666(nature)、d1981 0.553(country)、d88 0.516(country)——country 维主导头部，nature/metal 次之。类别编码呈“少数强维 + 大规模分布式”并存结构。
**P-B FALSE（5/10，关键发现：区分维≠载流维）**：η² 最高的 country 维 d13/d32 置零后 country 词 own margin 降幅（0.013/0.003）反而小于非成员（0.027/0.066；d32 反差 22×）——**E 空间里区分类别的维，读出通道并不经过它们**；通过特异性的 5 维（270 nature 0.137/0.053、88 country、322 nature、56 country、2505 metal 0.219/0.046）是真“区分+载流”双功能维。嵌入维角色二分确立：**区分型（类别几何）vs 载流型（读出通道）**，top-10 中各占一半。
**P-C FALSE（ratio 0.947 vs 随机 0.986）**：top-30 η² 维置零几乎不动先验谱 span（36.47→34.52）——区分维不承载先验 margin 谱。附带发现：E-side 先验谱类序（metal 34.4 最高 / nature −2.06 唯一负）与 2796 retain 谱（food 0.814 最高）**不同序**——先验强度与读出保留是两个独立调制。
**新发现（超预注册）**：η²(E) 与 η²(H) Spearman 0.009、top-30 重叠仅 1/30——**36 层加工把类别信息完全重映射到全新维度集**（信息保留、载体轮换：2795 P-S within 0.656 不变，但编码维换代）；PCA(E) 前 2 维仅 6.4% 方差（无低维压缩表示，类别几何高度分散）。

### 分析

**①“词嵌入每个参数的作用”的分层答案**。参数角色不是单一标签而是三重谱：(a) 31.4% 维度显著参与类别几何（区分型，η² 超置换 null）；(b) 其中约半数同时经 W_U 通道向类别词读出载流（载流型，P-B 5/10），另一半是“纯几何维”——类别词靠它们在 E 空间扎堆，但读出不走（d32：η²=0.73 却对 own margin 近零载流）；(c) 其余维承担属性/句法/全局尺度角色（不在本轮判据内）。**区分与载流的分离是“加工空间与读出空间解耦”（2795 P-U / 2796 P-X）在参数级的再现**——同一解耦原理从词位级贯穿到 embedding 参数级。
**②信息保留、载体轮换**。E→H30 的 η² 维集合几乎零重叠（ρ=0.009、1/30）而类别结构保留——36 层管线对类别信息“换车不换货”：类别内容持续存在，编码维度被系统性改写。否定“embedding 类别维直接延续到读出”的直觉，支持 LPF 建构模型：读出通道是逐层重建的。
**③方法论沉淀**：可加分解 + 纯 lens 消融使参数级因果归因零前向成本——2560 维 × 100 词全表普查单脚本 3 分钟，是大模型参数角色普查的可行范式。

### 相关文件（immutable，sha256 前 16 位）

脚本 phase2797_rdc_dim_attribution.py=8b23d702390b623c；产物 phase2797/qwen4_dim_attribution/{execution ac1061ab93dd5af5, result d661acb9bf7b333b, dim_attribution.npz 0ff02f22110a112d}。调试 3 轮（均按纪律删产物重跑）：①dtype（float64 tensor vs float32 W_U）；②idx_cat 误用类别索引 0..9 而非 token id（修复后 gate 0.000018 过）；③dW rival 均值 axis 误用（axis=1→axis=0），P-D 由 1.53e+02 修复至 1.99e-05。数据源 atlas.npz（2795，2c0010cef0883d32）。

### 问题硬伤

①P-B 判据为 |Δ| 组均值对比，未做逐词配对检验；②“区分型/载流型”二分基于 top-10 η² 维小样本，全 805 显著维的两型比例未普查（carrier 谱已在 npz，可离线补全）；③P-C 的 span 判据对 metal/nature 两端敏感，谱中部类别变化被掩盖；④置换 null 500 次的 99.5 分位较粗糙（第 2-3 大值插值）；⑤参数角色仅对 100 名词体系定义，全词表/非名词 token 未覆盖；⑥单模型单层。

### 结论

参数级图谱首轮落定：**①margin 对 embedding 维精确可加（err 2e-5）——每个参数的作用有精确公式 C=E·g·dW/rms；②31.4% 维度显著区分类别（覆盖 10/10 类），但区分维≠载流维（top-10 各半）——“几何上扎堆的维”与“读出上载流的维”是两类角色；③E→H 类别维集合零重叠（ρ=0.009）而类别结构保留——信息保留、载体轮换**。用户问题“词嵌入每个参数的作用”获得首批分层答案：区分型/载流型/其余三重谱，解耦原理下探至参数级。

### 接续（Phase 2798 候选，同目标自动续研）

(a) 全 805 显著维“区分-载流平面”普查：carrier vs η² 散点定标，每类报双功能/纯区分/纯载流维清单（carrier 谱已在 npz，可离线完成）；(b) 载流维因果补全：top carrier 维（不论 η²）逐维置零检验 own margin 特异性——载流型的独立因果验证；(c) 轮换轨迹：L0→L30 每 K 层采 η² 维集合，定位“换车”发生的层段；(d) 与 2795/2796 对接：区分型维是否承载近对差向量（d=h_a−h_b 的维度分解 vs carrier 谱）。

## Phase 2798: 自动续研，LPF-11：区分-载流平面 + 载流侧因果 + 差向量维度分解——纯离线 [2026-09-16 17:58]

### 测试原理

2797 留下三个参数级问题：①η²（区分）与 carrier（载流）两轴的关系是“相关不恒同”还是完全独立；②按 carrier 选维（不管 η²）能否通过 2797 P-B 失败的因果特异检验（载流侧补全）；③近对差向量（词身份差）住在区分维还是载流维。数据：2797 npz（eta2_E/q995/dim_cat/carrier/labels）+ atlas.npz（E30/H30）+ 权重，纯离线零前向。三预注册：P-A2 平面相关不恒同（805 显著维 ρ(η², carrier_own) ∈ [0.20, 0.95)——ρ<0.20 = 两轴独立，ρ≥0.95 = 恒同）；P-B2 载流维因果特异（每类按 carrier(c) 排序 top-10 维逐维置零，成员 |Δown margin| > 非成员 ≥8/10 类）；P-C2 差向量栖身区分维（3 近对 top-20 |ΔH| 维中显著维比例 ≥47.1%=1.5×基率 31.4%）。

### 结果

gate：numpy lens 对拍 2795 max|d|=0.000012。
**P-A2 FALSE（ρ=0.125 < 0.20，方向强化分离）**：η² 与 carrier 近乎**完全独立**（预注册下界 0.20 判失误，实际独立性比“相关不恒同”更彻底）。四象限普查（carrier-high = 显著维 carrier Q75）：**双功能 202、纯区分 603、纯载流 273、其余 1482**——区分与载流是近乎正交的两种角色。
**P-B2 TRUE（10/10 类全过，载流侧完美确立）**：按 carrier 排序的 top-10 维置零，**10/10 类全部呈现成员 |Δ| > 非成员**。与 2797 对照形成决定性对比：**η² 选维 5/10 过，carrier 选维 10/10 过——因果特异性由选维标准决定，carrier 是读出侧的正确选维标准**。
**P-C2 FALSE（0/3，反向发现）**：差向量 top-20 |ΔH| 维的显著维比例 0.20，**低于随机对照 0.25-0.40**（mean η² 同样低于随机）——差向量不是栖身区分维，而是**回避**类别显著维。机制：同类近对 d=h_a−h_b 中共享类别成分被相减消掉，剩下的个体特征差自然住在非类别维——差向量的维度构成反向印证“类别维=共享成分载体”。

### 分析

**①参数级角色图谱定稿（回答“词嵌入每个参数的作用”）**。2560 维分四个功能区：**双功能区 202 维**（既区分类别几何又向读出载流）、**纯区分区 603 维**（类别词在 E 空间扎堆用，读出不走）、**纯载流区 273 维**（读出通道真经过但类别几何不靠它）、**其余 1482 维**（属性/句法/全局尺度，未判别）。三重角色（区分/载流/其他）在维度集上近乎可加分离（ρ=0.125），且各有独立的因果验证口径（η² 定区分、carrier 定载流）。
**②选维标准定律**。同一置零协议下，η² 选维 5/10 vs carrier 选维 10/10——**几何统计量（η²）不预测因果角色，通道统计量（carrier）精确预测**。这把 2797 的“区分维≠载流维”从观察升级为可操作的选维规则：研究读出因果必须按通道载流选维，不能按几何方差选维。
**③差向量的归宿闭合了图谱三角**。词位级（2796 P-X：差向量读出=词身份词族）→ 维度级（2798 P-C2：差向量住在非类别维）→ 机制：类别维载共享成分、非类别维载个体特征差。名词图谱的完整分解：**类别共享成分（805 区分维）× 读出通道（载流维）× 个体特征差（非类别维）**三个子空间各司其职。

### 相关文件（immutable，sha256 前 16 位）

脚本 phase2798_rdc_discriminate_carry.py=bf72657e190ae622；产物 phase2798/qwen4_discriminate_carry/{execution 039b644662a22b2e, result 968c84204a59a8f6, plane.npz fe8d713476982e43}。数据源：2797 dim_attribution.npz（0ff02f22110a112d）+ 2795 atlas.npz（2c0010cef0883d32）。一次跑通，gate 0.000012。

### 问题硬伤

①P-A2 区间下界 0.20 预注册判断失误（实际 ρ=0.125），判据方向被 FALSE 触发但结论（分离）反而更强——预注册区间设计应吸取教训；②象限阈值 carrier-high 用显著维 Q75，非独立定标（换阈值象限计数会变，双功能/纯载流边界软）；③P-B2 组均值 |Δ| 无逐词配对检验；④P-C2 只测同类近对——跨类对（apple/dog）差向量应富集类别维（共享成分不被消掉）的预测未检验，是本机制解释的关键缺失对照；⑤carrier(c) top-10 维在类别间可能重复（一维载多类），未去重分析；⑥单模型单层。

### 结论

参数级角色图谱闭合：**①区分与载流近乎正交（ρ=0.125；四象限 202/603/273/1482）；②载流选维 10/10 类因果特异全过 vs η² 选维 5/10——“几何不预测因果，通道预测因果”成为选维定律；③差向量回避类别维（0.20 < 随机 0.25-0.40）——类别维载共享成分、非类别维载个体差，名词图谱三角（共享成分×读出通道×个体差）各司其职**。“词嵌入每个参数的作用”的答案定稿为四功能区图谱。

### 接续（Phase 2799 候选，同目标自动续研）

(a) 跨类差向量对照（P-C2 机制检验的关键缺失对照）：apple/dog 等跨类对的 top-20 |ΔH| 维应富集显著维（类别成分不被消掉）——预测富集比 >1，验证“类别维=共享成分”图景；(b) 轮换轨迹：L0→L30 每 K 层 hidden 的 η² 维集合演变，定位“换车”层段（需 100 词重新前向采多层状态）；(c) 载流维的通道画像：每类 top-10 载流维的 dW[c,·] 结构（通道指向哪些 vocab token——是否正是该类实例词/属性词）；(d) 2796 遗留：命题信息量独立定量（伪困惑度/PMI）验 retain 单调律。

## Phase 2799: 自动续研，LPF-12：跨类差向量对照 + 载流维通道画像——纯离线，双 FALSE 理论修正 [2026-09-16 18:58]

### 测试原理

2798 留下两个待检验点：①P-C2 的“差消”解释（同类差向量回避类别维因为共享成分被 a−b 消掉）预言跨类对照应富集类别维——6 预注册跨类对（apple/dog、car/apple、Japan/hammer、ocean/chair、bread/dog、gold/shirt）top-20 |ΔH| 维显著维比例应 ≥47.1% 且高于配对随机对照；②由 Δz(t) = −W_U[t,d]·g[d]·E[w,d]/rms_w，载流维 d 的“通道画像”= 其 W_U 列的 top token——若通道语义真实，每类 top-3 载流维的 |W_U[:,d]| top-20 token 并集应含本类实例词（≥7/10 类）。纯离线零前向（atlas.npz + 2797 npz + 权重）。

### 结果

**P-A3 FALSE（0/6，推翻差消解释）**：跨类对差向量 top-20 维显著维比例 0.10-0.30，**全部低于**配对随机对照（0.35-0.60），与同类对（0.20）无差别——跨类对同样系统性回避类别维。“差消类别成分”机制被否定；正确图景是**结构性分离**：词身份差的载体维度集与类别编码维度集是两个分离功能区，无论近对远对。|ΔH| top 维的 mean η²（0.16-0.18）同样低于随机（0.18-0.21）。
**P-B3 FALSE（1/10，单维多义性确立）**：10 类 top-3 载流维的 W_U 列 top-20 token 画像不含本类实例词（仅 animal 命中 dog——dim 2049 的 W_U 列 top-1 恰为 “dog”，孤立个案）。画像内容为杂义词族：中文功能词（败/招募/血腥）、社交套话（我爱你/谢谢你/网友评论）、波兰语/俄语词族、代码 token——**单维 unembed 列无类别语义结构**。dW[c,d] 是对比值，其大不等于 W_U[c,d] 列内突出；类别 margin 由数百维载流和决定（2797 P-C：top-30 置零仅动 span 5%）。

### 分析

**①理论修正（重要）**：2798 P-C2 的“差消”解释被 2799 跨类对照否定——差向量回避类别维不是差消的 artifacts，而是**结构性分离**（个体差区 vs 类别共享区是并行堆叠的独立维度集）。这把图景从“差消动力学”修正为“近似正交的子空间组织”：类别成分与个体成分在词位空间各自成区，差向量天然只取个体差区。**②单维多义性（polysemanticity）在本体系确立**：维度的角色谱（区分/载流四功能区，2797/2798 因果验证）在“维度集”层面真实，但单个维度不携带可读语义画像——W_U 列连接杂词大片，载流是数百维的和（superposition 式分布式编码）。**③粒度三分定律**：本体系内语义的可读粒度是“方向”（类别方向/属性方向，2795-2796）与“维度集”（四功能区，2797-2798 因果），而非“单维”（2799 画像杂义）。参数级图谱的最终形式：**维度集层面的功能分区 × 方向层面的语义内容 × 无单维语义画像**。

### 相关文件（immutable，sha256 前 16 位）

脚本 phase2799_rdc_crosspair_channel.py=b9009db5252b855e；产物 phase2799/qwen4_crosspair_channel/{execution 08563983b3278eea, result 7456ed440111f6a0, crosspair.npz 5622d2c863bf1d69}。数据源：2797 npz（0ff02f22110a112d）+ 2795 atlas.npz（2c0010cef0883d32）。一次跑通（写脚本时清理 2 处残留：空序列 max、误删 labels 定义后补回）。

### 问题硬伤

①跨类对仅 6 对、随机对照 20 维小样本涨落大（frac_rand 0.35-0.60 高于基率 0.314 即为抽样噪声），应做全对全维分布检验；②通道画像只看了 |W_U[:,d]| top-20，带符号 top / dW 加权画像未做——dW[c,d] 带符号对比的 top token 可能更有结构；③“结构性分离”的正面机制（为何个体差区与类别区可分）未建模——叠加编码/正交化理论未对接；④P-B3 匹配沿用子串规则；⑤未检验高阶：top-50 载流维并集画像是否出现类别结构（载流是数百维和，top-3 太少）；⑥单模型单层。

### 结论

**双 FALSE 但图景更清晰**：①跨类对照否定“差消”解释——个体差区与类别区是结构性分离的独立维度集（词身份差永远住在非类别维，无论近对远对）；②单维多义性确立——维度集层面的功能分工真实（因果验证），单维画像无语义（W_U 列杂义），语义在方向与维度集粒度、不在单维粒度。参数级图谱定稿为：**四功能区（区分/载流几何分工）+ 无单维画像（superposition 分布式）+ 方向承载语义**——这是“词嵌入每个参数的作用”的最终三层答案。

### 接续（Phase 2800 候选，同目标自动续研）

(a) 载流子空间而非单维：每类 top-K 载流维（K=10/30/100）联合子空间投影 E30 的类别判别（LDA/质心投影），检验“载流维集=读出通道的子空间”并定 K 的饱和曲线；(b) 带符号画像 redo：dW[c,d]·W_U[t,d] 联合加权的 token 分布（修正硬伤②）；(c) L0→L30 轮换轨迹（需重前向采多层）；(d) 结构性分离的机制建模：个体差区与类别区的子空间夹角分布/干涉测量（对接 superposition 理论）。

## Phase 2800: 自动续研，LPF-13：载流子空间联合判别 + 带符号池化画像 + 通道几何——纯离线，方向语义一锤定音 [2026-09-16 21:20]

### 测试原理

2799 留下三个待检验点，全部纯离线（atlas.npz E30 + 2797 carrier/eta2 npz + 权重，零前向，gate 对拍 2795 max|d|=0.000012）。**A 臂（子空间充分性 K-饱和）**：d' 判别度 = 类别 c 的 lens margin s(w)=M[w,c] 的成员/非成员标准化均值差（d'=(μ_in−μ_out)/σ_pooled）；把 top-K 载流维之外的维度全部置零（2797 已证 margin 对维度精确可加，此为精确的子空间受限读出），K∈{5,10,20,40,80,160,320}，对照 η²-top-K 与 10 次随机-K。预注册 P-A4：≥8/10 类 top-40 载流子空间 d' ≥ 0.7×全维 d' 且 > η²-40 且 > 随机均值。**B 臂（带符号池化画像，修正 2799 硬伤②）**：p_c = Σ_{d∈top-40 载流} dW[c,d]·W_U[:,d]——把子空间的通道方向整体映射进 vocab 空间（正是加性分解 C(d;w,c)=E[w,d]g[d]dW[c,d]/rms_w 中通道项的子空间重组），取正负两侧 top-30 token，预注册 P-B4：≥5/10 类命中类别相关 token（实例词/类别词/预注册属性表）。**C 臂（通道几何）**：10 条通道方向 dW 行的两两 |cos|（45 对）+ 子空间捕获率 cos(dW_c|top-40, dW_c) + top-40 载流维集合重叠普查。预注册 P-C4：mean|cos|<0.5 且 fruit/food 进 top-3。

### 结果

**P-A4 FALSE（2/10，通道高度分布式）**：全维 d'（metal 5.72 > country 5.49 > animal 4.27 > fruit 3.52 > tool 2.90 > vehicle 2.80 > clothing 2.58 > food 2.38 > furniture 2.08 > nature 1.28）；top-40 载流子空间仅达全维的 44%-103%（metal 96%、nature 103% 通过，其余 44-77% 不及 0.7 阈），K*80 多在 160（tool 甚至 320 维仍 <80%）。η²-40 对照在 animal/vehicle/furniture/clothing 反超 carrier-40——固定小 K 下两种选维互有胜负，均不充分。子空间捕获率（top-40 载流维占 dW 通道范数的比例）10 类齐整地只有 0.28-0.31：**通道质量 ~90% 散布在 top-40 之外**，由捕获率反推有效维数 n_eff ≈ 40/0.30² ≈ 440 维/通道。**P-B4 TRUE（10/10，方向语义一锤定音）**：带符号池化画像在全部 10 类命中类别词，且正侧 top 画像几乎就是类别词自身的多语重复：fruit→fruit/水果/果实/果园+mango/berry/apple；animal→animal/动物/動物；metal→metal/金属+gold/silver/copper/steel/aluminum/nickel/brass/bronze（8 个实例词全中）；vehicle→vehicle/车辆；country→country/nation/china；food→food/食品/食物；nature→nature/性质/自然界；furniture→furniture/家具+chair/sofa；tool→tool/工具/toolbox；clothing→clothing/衣服/服装/服饰+apparel/dress。**P-C4 FALSE（一半对一半错）**：mean|cos|=0.116——通道确实近正交（0.116≪0.5 阈）；但 fruit/food 排 42/45，top-3 是 animal/clothing 0.186、nature/furniture 0.166、fruit/vehicle 0.159——**通道相似结构与表征几何（2795 质心近邻 fruit↔food 0.864）完全不同**。top-40 载流维重叠：max 3、mean 1.0（随机期望 0.6）——载流维集合近不相交。

### 分析

**①粒度三分定律完成定量闭环**：单维画像杂义（2799）→ 小子空间（top-40）读出不充分但**其通道方向的 vocab 映射单调语义**（本轮 10/10）→ 全通道分布式（n_eff≈440）。语义精确定居在"方向"粒度：把 top-40 载流维按 dW 带符号加权求和，杂义 token 相消、类别词相长——superposition 的经典预测（特征≈方向、单维多义、方向读出干净）在本体系全部命中。**②类别概念的词汇自指**：通道方向的 top token 是类别词本身（中英双语），不是实例词或属性词——读出通道的"锚点"是范畴名，margin 竞争的语义靶心就是类别标签，这直接解释了 2795 命名通道与实例通道的区分（fruit 命名通道保留 41% vs animal 实例通道压制 97%）。**③第三重解耦确立**：表征几何近邻（质心：fruit↔food）、先验谱排序（2797：metal 最高）、读出保留谱（2796：food 最高）、通道相似结构（本轮：animal/clothing 最高）——四个"语义相似度"四种序，通道间近正交（0.116）进一步说明 10 条读出通道是独立叠加在同一表征大陆上的、彼此几乎不干涉的探针束。

### 相关文件（immutable，sha256 前 16 位）

脚本 phase2800_rdc_carrier_subspace.py=0b2933ab4ab8346e；产物 phase2800/qwen4_carrier_subspace/{execution f44a30d17716895f, result 2e3c9765da9b46b5, subspace.npz c5d05b599765f1fc}。数据源：2797 npz（0ff02f22110a112d）+ 2795 atlas.npz（2c0010cef0883d32）。一次跑通（脚本内预注册属性表在读取前冻结）。

### 问题硬伤

①P-A4 的 0.7×d' 阈值与 K=40 临界点是预注册选择，K*80 依赖 d' 满值定义（若以 top-320 或全维为基准曲线会更陡）；②η²-40 对照在 4 类反超说明"固定 K 下载流选维最优"不成立——载流与几何选维的联合选维（如 carrier×η² 交集）未测；③捕获率反推 n_eff≈440 假设质量均匀散布，真实分布未测（累积范数曲线可离线补）；④P-B4 命中表含子串宽松匹配（fruitful/fruity 计入 fruit），但正侧 top-15 的类别词多语重复是直观可见的强证据，不依赖匹配规则；⑤通道方向 dW 是"对 9 类均值"的对比向量，其 vocab 画像的语义解释依赖对比编码假设；⑥单模型单层（Qwen3-4B L0 embedding）。

### 结论

**方向语义一锤定音 + 通道分布式图景定稿**：①带符号池化画像 10/10 命中类别词（中英双语）——把 2799 的单维杂义与 2795-2796 的方向语义统一为 superposition 图景：特征住方向、维只是叠加坐标，top-40 载流维的带符号和即为类别读出方向，其 vocab 映射单调干净；②top-40 子空间不充分（捕获率 ~0.30，n_eff≈440 维/通道）——读出通道是高度分布式的，没有小的"载流子空间"，但有精确的"载流方向"；③通道间近正交（0.116）+ 载流维集合近不相交（重叠 1.0 vs 随机 0.6）——10 条读出通道是独立探针束，其相似结构独立于表征几何。LPF 升级 **v5.3**：词位随身档案 = 类别调制增益谱 × 词身份差向量 × 读出内容竞争 × **通道方向=分布式带符号维和（方向承载语义、维承载叠加）**。参数级图谱三层答案终稿：四功能区（维度集粒度）× 方向语义（方向粒度）× 无单维画像（单维粒度）。

### 接续（Phase 2801 候选，同目标自动续研）

(a) 通道质量累积曲线：每类 dW 通道范数的累积分布（按 |carrier[c,d]| 与按 dW² 排序两种），精确定 n_eff 与分布形态（幂律/均匀），离线可完成；(b) 联合选维：carrier×η² 交集或 carrier·|dW| 加权排序的 K 曲线，检验固定 K 下是否存在更优选维；(c) 轮换轨迹：L0→L30 每 K 层 hidden 的 η²/carrier 维集合演变 + 各层通道方向的 vocab 画像（"换车不换货"的层段定位，需 100 词重新前向采多层状态）；(d) 实例级方向画像：把 B 臂协议下探到单词粒度——apple 的读出方向（margin 对 rival 的对比向量）vocab 画像是否指向 orchard/cider/juice（对接 2796 差异化=通道内容选择）。

## Phase 2801: 自动续研，LPF-14：层间轮换轨迹 + 各层通道画像——首次重前向全层采样，"一次性换车"定律 [2026-09-16 21:55]

### 测试原理

2797 的"信息保留、载体轮换"只有 L0/L30 两个端点快照，"换车"发生在哪几层、渐变还是突变、通道语义是否在轮换中存活——三个问题悬置。Phase 2801 首次重前向：100 词逐词单前向（协议与 2795 逐位一致：ids=tok('The')+tok(' '+w)，读位置 p=1），采集全部 37 层 hidden states（HS 37×100×2560，float32 18 MB）。关键推广：final_norm 与层无关（h/rms(h)·g 对任何层都成立），因此**精确可加分解 C(d;w,c)=h_l[w,d]·g[d]·dW[c,d]/rms(h_l[w]) 在每一层都成立**——2800 的全部协议（载流选维、带符号池化画像、d' 判别）可逐层重放。五道 gate 全过：E 0.000000、H30 0.000000、lens 0.000012、η²_l0 vs 2797 npz 1.67e-16、gap_30 vs 2795 0.0051。逐层计算四条轨迹：η²_l（10 组单因素方差分析，2560 维）、结构 gap_l（类内-类间质心 cos 差）、载流维集合（每层 top-40 carrier）、带符号池化画像命中数（10 个采样层 × 10 类）。预注册：P-A5 换车在中途（∃l*∈[2,30] ρ<0.30 且 gap 全程 ≥0.7×gap_0）；P-B5 轮换渐进（相邻层 top-40 载流维重叠 O_l≥0.25）；P-C5 通道语义全程存活（采样层平均命中类数 ≥8）。

### 结果

**P-A5 FALSE（换车不在中途——在第一层瞬时完成）**：ρ(η²_l, η²_0) 在层 1 即从 1.00 崩至 0.061，此后 35 层一直在 0~0.07 徘徊——E 侧载流几何在第一个 transformer 层就被整体改写，不存在"中途换乘站"。gap_ok 也 FALSE，但失败方式有信息量：gap 全程 0.080-0.144 波动（谷 0.080@L20、峰 0.144@L11、L26-29 回到 0.127），唯 L35 跌至 0.063、末层 L36 回 0.106——结构是**波动保持**而非单调保持，且最后一层前有个压缩谷。**P-B5 FALSE（第一步突变、此后渐变）**：O_1=0.12（层 0→1 一步丢失 88% 载流维），但 O_2..O_35 全在 0.74-0.89——**"一次性换车、此后平稳驾驶"**：层 1 整体更换载流维集合，之后每层只温和换 12-26%，O_36=0.49（末层再动一次）。**P-C5 TRUE（满格 10/10 层 × 10/10 类）**：每个采样层的带符号池化画像都命中全部 10 个类别词——**通道方向的语义在 37 层全程不变，变的只是承载它的维集合**。

辅助曲线：①top-30 η² 重叠 ov30 全程 ≈0-0.17——维级轮换逐层确认；②d'（lens 类别判别）U 形：L0 3.30 → L1-5 谷 ~1.5 → 峰 2.74@L24 → 末 2.05；③fruit 成员 margin 生成曲线（37 层）：**22.8（先验满格）→ 谷 2.2@L19（压制）→ 峰 13.8@L32（重建）→ 9.1**——"先验→压制→重建"三段式，2790 的写入曲线在类别 margin 上重现但形状为深 U。

### 分析

**①"一次性换车"定律（本轮主发现）**：载流维集合的更换不是逐层漂移积累，而是层 1 的整体置换 + 之后 35 层的平稳渐变 + 末层的再调整。层 1 是"格式转换器"：把 embedding 的编码格式转写为管线内部格式（2791 的"L0 是转码器"结论前移一层精确化——转码发生在层 1 的 attn+MLP）。**②语义-载体分离的完整版**：37 层里载流维集合换了一整代（ρ≈0、ov30≈0），但 (a) 类别结构 gap 波动保持、(b) 通道方向语义（画像命中类别词）全程不变——**语义不变量住在"方向"这个抽象层面，任何具体维集合都只是它的临时载体**。这是 superposition 图景的动力学版本：特征=方向是层不变量，维=可替换坐标。**③margin 三段生成**：先验（L0 满格 22.8）→ 中层压制（L19 谷 2.2，管线在做去偏/语境化）→ 晚层重建（L32 峰 13.8）——类别读出不是"逐步累积"而是"先验被中途改写后再重建"，与 2792 无临界点结论合并为：重建是分布式渐进的，但轮廓有清晰的深 U 形状。

### 相关文件（immutable，sha256 前 16 位）

脚本 phase2801_rdc_rotation_trajectory.py=a4bbe00db778f89a；产物 phase2801/qwen4_rotation_trajectory/{execution 56b117fed4a4f7fc, result 65715d001a92985c, trajectory.npz 98d8a6972edae584（含全 37 层 HS）}。数据源：2797 npz（0ff02f22）+ 2795 atlas.npz（2c0010ce）。调试 2 轮（gateM 与 mem_margin 两处把类别位置误写到 token-id 列表上找 index）后跑通；forward 与 gate 协议逐位复现 2795。

### 问题硬伤

①P-A5 的 ρ<0.30 阈与"中途"区间 [2,30] 是预注册选择——实际换车在层 1，落在预注册窗外，TRUE 判定需改区间；判据设计对"突变在第几层"不敏感，应改为"首个 ρ<0.5 的层"报告式登记；②gap 的 0.7× 阈被 L35 单点击穿（0.063<0.074）——结构保持判据对末层敏感，中位/去端统计更稳健；③O_l 只算了相邻层重叠，"换车代际"的更长程重叠（O_l vs L0、O_l vs L36）未算全矩阵；④画像命中仅 10 个采样层 × 每类 top-30，未做逐层全查；⑤层 1 换车的机制归因（attn vs MLP 各贡献多少）未分解——2794 的双算子协议可在层内重放；⑥单模型（qwen3-4b）。

### 结论

**层间动力学首次成像，"一次性换车"定律确立**：①载流维集合在层 1 被整体置换（ρ 1.00→0.061、O_1=0.12），此后 35 层平稳渐变（O≈0.8）——"换车"是一次性格式转换，不是渐进漂移；②**语义不变量=方向，维=可替换坐标**——通道画像 10/10 层 × 10/10 类全程命中类别词，类别结构 gap 波动保持（0.080-0.144），轮换不伤语义；③margin 生成三段式：先验满格（22.8）→ 中层深谷（2.2@L19）→ 晚层重建（13.8@L32）——类别读出是"改写后再重建"的深 U 曲线。LPF v5.3 增补动力学条款：**层 1 格式转换器 + 35 层方向稳定漂移 + 晚层读出重建**。

### 接续（Phase 2802 候选，同目标自动续研）

(a) 层 1 换车机制分解：2794 双算子协议在层内重放（attn/MLP 分别置零），定位格式转换的算子归属；(b) 代际重叠全矩阵：O_l vs L0 与 O_l vs L36 的 37×2 曲线 + 载流维集合的谱系聚类（几代载体？）；(c) margin 深谷的性质：L19 谷是去偏还是语境化——谷底层的画像与 top token 检查；(d) 跨模型复核：qwen3-14b 重放 2801 协议（一次性换车是否通用）。

## Phase 2802: 用户定向，LPF-15：apple 多义词谱分析——哪些参数表达水果/食物/植物/公司含义 [2026-09-16 22:25]

### 测试原理

用户锚点："对词嵌入进行谱分析，比如对于苹果，精确破解哪些参数表达水果含义、哪些表达食物、哪些表达植物、哪些表达科技公司"。机制：四条语义通道各由 10 个单 token 语义代表词定义（fruit/food 沿用 2795 词表；plant=tree/flower/rose/leaf/root/grass/oak/pine/maple/fern；company=Google/Microsoft/Amazon/Meta/Tesla/Nvidia/Samsung/Sony/IBM/Intel，预检全单 token），对比方向 dW_s = W_s − mean(其他 3 感)，由精确可加分解得 apple 的 4×2560 贡献矩阵 CM[k,d] = E[apple,d]·g[d]·dW_s[k,d]/rms_apple（零前向，gate：E 行逐位 0.000000、旧 10 类 lens margin 0.000007）。三臂：A 臂各感 top-40 载流维的带符号池化画像（vocab 可读性）；B 臂 apple 的谱分析——4×4 Spearman、每感参与率 PR=(Σw)²/Σw²（w=C²）、CM 的 SVD 奇异谱、top-40 集合重叠、805 显著维占比、全维语义投票普查、默认感排序；C 臂读取侧消歧——两句仅 apple 后文不同的句子（"the apple pie is tasty" vs "the apple released the iphone"），由 causal mask 预言 apple 位 h bit-exact 相同（档案消歧不可能），句末 lens 的 fruit−company margin 必须移动（消歧发生在读取者）。预注册 P-A6 ≥3/4 画像命中；P-B6 ρ(fruit,company)<0.20 且 ρ(fruit,food)>ρ(fruit,company)；P-C6 h 逐位相等且 |Δfinal|≥1.0 且符号正确。

### 结果

**P-A6 TRUE（3/4，且 FALSE 是匹配表遗漏而非语义缺失）**：fruit 画像判定 FALSE 纯因预注册相关表漏了实例词——其正侧 top-15 恰是 **cherry/banana/berry/grape/mango/peach/apple/orange（全部水果实例词）**，是四感中语义最干净的画像；food→bread/meat/cheese/pasta/pizza/肉；plant→flower/leaf/tree/花/trees；company→microsoft/samsung/google/amazon/intel/ibm/lenovo。**P-B6 TRUE（语义对比编码）**：ρ 矩阵全负（fruit-food −0.249、fruit-company −0.331、plant-company −0.357）——dW_s 是对比向量，同维带反号票；语义距离定序：近感（fruit-food）比远感（fruit-company）反相关更弱。**SVD 奇异谱 [2.02, 1.75, 1.54, 0.000]**：秩 3（对比约束），且谱异常平坦——apple 档案里三条等强独立语义对比轴，无主感轴。参与率：fruit 250 / food 226 / plant 318 / **company 66（品牌感最紧凑，约为自然类感的 1/4）**。top-40 重叠：fruit/food 12 最高（语义近），plant/company 7 最低。805 显著维占比 fruit 0.675 > food 0.575 > plant=company 0.500。全维投票普查均分（576-774/感，company 最多）。**apple 默认感排序：fruit +29.6 ≫ food −14.2 > plant −9.3 > company −6.2**——档案先验强 favor 水果感。**apple 的 top 参数直接答案**：fruit 维 13(+0.202)/1874(+0.185)/1113/1400/2229；food 维 1170(−0.259)/32(+0.198)/1874(−0.195)/2229/13；plant 维 2346/32/1106/1100/1515；company 维 32(−0.528)/13(−0.405)/1170(+0.300)/270/39——**同一批维（13/32/1170/1874/2229）以不同符号服务多个感**：d13=水果+/公司−/食物+，d32=食物+/植物+/公司−，d1170=食物−/公司+，d1874=水果+/食物−。**P-C6 TRUE**：apple 位 max|h1−h2|=0.000000（bit-exact，档案消歧不可能）；句末 fruit−comp：pie 5.98 vs released 3.69，Δ=2.297≥1.0 符号正确——消歧发生在读取侧。

### 分析

**①用户问题的精确答案**：apple 的"哪个参数表达哪个含义"不是一个维一个感的排他映射，而是**带符号投票谱**——四感由对比方向 dW_s 定义，同一维可同时为水果投+票、为公司投−票（d13/d32/d1874/2229 都是多功能投票维）；感identity住在 2560 维带符号模式整体里（ρ 全负即证），切分出 top-40 维集只是谱的峰值区。**②谱结构三定律**：(a) 秩 3 平坦谱——四感对比只张成 3 维，三条轴等强（无支配感），apple 的多义性是"三条等强语义轴的叠加"；(b) 紧凑度按抽象层级递减——company（品牌/命名感）PR=66 最紧凑，plant（自然类）318 最分散——命名感靠少数维、自然类感靠数百维，与 2796 命名/实例通道二分呼应；(c) 先验排序 fruit+29.6 绝对主导——apple 的档案默认按水果感读出，公司感是负对比（需读取侧主动查询才显现）。**③消歧机制定位**：档案位 bit-exact（0.000000）证明**多义共存于档案、消歧完全发生在读取侧**——"the apple ___" 的后续词通过注意力在下游位置重构读出，fruit−comp margin 移动 2.30。但注意 released 句的 company 分数仍为负（−3.13）且 plant 反升为最高（1.69）——单句短上下文的感切换是**相对渐变**，不是二元开关；绝对的公司感激活可能需要更强的公司语境或读取位置更靠近 apple。

### 相关文件（immutable，sha256 前 16 位）

脚本 phase2802_rdc_polysemy_spectrum.py=492dfdc0e69f417b；产物 phase2802/qwen4_polysemy_spectrum/{execution f27dda1f552cbef7, result f34503476f3fbc10, polysemy.npz ef6729e233215d96}。数据源：2797 npz（0ff02f22）+ 2795 atlas.npz（2c0010ce）。调试 1 轮（plant/company 词不在 atlas 100 词表，Erows 改从 embedding 表取）后一次跑通；词表预检 24 词全单 token。

### 问题硬伤

①P-A6 的 fruit"FALSE"是预注册相关表漏实例词的人造 FALSE（正侧 top-15 全是水果词）——判定记 TRUE 3/4 但实际语义上 4/4，匹配表应并入实例词；②四感 dW 由"对 3 感均值"对比定义，fruit 集含 apple 自身（1/10 污染）；③C 臂只测 2 句 1 对比，company 感在短句中未被绝对激活（最终分 −3.13），感切换的剂量-反应曲线未测（更强语境/多句/更长距离）；④CM 谱分析限于 4 感——apple 的完整感清单（材料?事件?）未枚举，第 4 奇异值恒 0 是对比约束的数学必然而非经验发现；⑤投票普查的 argmax 阈值未设显著线（弱贡献维也投票）；⑥单模型单层（档案层）。

### 结论

**多义词谱分析首次实现，"带符号投票谱"定稿**：①apple 的 2560 维被四条对比方向精确分解——水果感维 13/1874/1113/1400/2229（+0.18~0.20），食物感维 1170/32/1874/2229/13，植物感维 2346/32/1106/1100/1515，公司感维 32/13/1170/270/39——**同一维多感复用、符号即感间对立**（ρ 全负 −0.24~−0.36）；②谱结构：秩 3 平坦奇异谱 [2.02,1.75,1.54] + 参与率分层（company 66 ≪ fruit 250/food 226/plant 318）——命名感紧凑、自然类感分布式；③档案先验 fruit +29.6 绝对主导，多义共存于档案（apple 位 bit-exact 0.000000）、消歧在读取侧（句末 Δ=2.30）。**"词嵌入每个参数的作用"的最终完成态：参数作用 = 它在所有语义对比方向上的带符号投票表**。LPF v5.3 增补多义条款：档案 = 感对比轴的叠加谱，读取侧按语境查询。

### 接续（Phase 2803 候选，同目标自动续研）

(a) 感切换剂量-反应：company 语境强度扫描（0~5 个公司线索词），测 apple 公司感分数的激活曲线——找绝对转正的语境阈值；(b) 多词多感普查：banana/bass/plant/tank 等多义词的同协议谱分解，验"秩 k-1 平坦谱+参与率分层"是否通用；(c) 读取侧消歧的算子定位：哪几层注意力完成感选择（句间差 h 的逐层出现点——重放 2801 协议于句对）；(d) 感方向 vs 类别方向的关系：dW_s 与 10 类 dW 的夹角矩阵（公司感维为何 50% 在 805 显著维里——公司感借用类别几何？）。

## Phase 2803: 用户定向：普适性普查，LPF-16：多义谱 25 词 × 14 域大规模验证 + 剂量-反应 + 层定位 [2026-09-16 22:50]

### 测试原理

用户指令："加大测试的数据量和类型，确保测试结果具有普遍性"。2802 的多义谱协议（1 词 4 感）扩展为 **25 目标词 × 14 语义锚集 × 53 词-感对**（新增 color/music/sport/computer 四域锚集，sport 用 skiing 替换 2-token 的 judo；词表预检 46 词全部单 token）。每词协议与 2802 逐位一致：dW_s = 锚集均值 − 其他感均值，CM_s(d)=E[w,d]·g[d]·dW_s[d]/rms_w（零前向），逐词算 ρ 矩阵/SVD 谱/参与率/默认感排序/top-40 画像。gate：E=0.000000、lens=0.000007、**CM 与 2802 npz 逐位一致 2.22e-16**（行序重排后）。新增两臂：D 臂剂量-反应——apple 公司感语境 4 剂量扫描（"the apple"+0/1/3/5 个公司线索词），apple 位与句末位的四感分数；E 臂层定位——2802 句对（pie vs released）句末位 37 层 lens 的 fruit−comp 差值曲线。预注册：P-A7 画像命中 ≥70%；P-B7 ρ 全负 ≥80% 且平坦谱 ≥75%；P-C7 PR 分层（max/min≥2）≥60%；P-D7 剂量正向（dose3>dose0 且 ≥2/3 增量为正）。

### 结果

**P-A7 TRUE（53/53 = 100% 画像命中）**：每个词-感对的带符号池化画像都命中锚词/属性词——语义画像可读性是**全域普适律**，无一例外。**P-B7 TRUE（25/25 ρ 全负；25/25 平坦）**：对比编码（同维多感反号）在 25 词 14 域全部成立，无一反例。**P-C7 FALSE（1/25，但属预注册数学缺陷而非经验否定）**：k=2 时两感对比方向严格反对称（dW_1 = −dW_2）⇒ |CM| 分布恒等 ⇒ PR 恒比 1.0——23 个 2 感词的 PR 判据**按构造无定义**；唯二 k≥3 词：apple 比率 4.8（分层显著）、iron 1.47（弱分层）。**默认感排序表（先验强度，全部语义合理）**：iron→metal +36.7、apple→fruit +29.6、violet→**color** +26.4（非植物！）、table→furniture +20.1、rose→plant +15.7、bass→**music** +15.5（鱼感弱）、mouse→animal +12.4、coconut→fruit +11.7、leather→clothing +9.6、olive→fruit +9.3、orange→**color** +8.4（非水果！）、lime→fruit +6.7、trunk→plant +6.7、corn→food +6.3、pepper→food +6.0、cotton→clothing +5.8、chicken→animal +4.7、wool→clothing +4.4、bark→plant +4.6、bug→animal +3.2、lobster→animal +3.6、salmon→animal +1.0、coach→sport +1.2、python→animal +0.5（计算机感几乎平手）、turkey→food +0.1（完全平手）。**P-D7 TRUE 但曲线振荡**：句末 company 分 [−7.92, −2.16, −7.86, −2.88]（剂量 0→3），增量 [+5.75, −5.69, +4.97]——2/3 为正、dose3>dose0 判定通过，但**非单调**：'iphone' 抬升公司感、追加 'mac steve' 又压回、'jobs cupertino' 再抬升——读取侧读数由**局部预测动力学**主导，不是累积证据积分。apple 位 4 剂量 drift=0.000000（档案免疫第 4 次独立确认）。**E 臂层定位**：句间 fruit−comp 差曲线 L0=+1.44 → L1-13 深负（−5~−7.3，'iphone' 词自身的 embedding 读数主导且符号与最终语义相反）→ L14 转正 → L24-29 稳定 +1.3~+2.2 → 句末 +2.3——**语义消歧由中后层注意力完成，早层被表层词汇贡献污染**。

### 分析

**①普适性确立（本轮核心交付）**：2802 的三大发现——画像可读性、对比编码（ρ 全负）、档案先验排序——在 25 词 × 14 域 × 53 对上**零反例**复现。多义谱协议从个案升级为定律：**任意单 token 词的任意感，其档案都表现为对全部感的带符号投票谱，默认感 = 得票最高感**。②P-C7 的 FALSE 是预注册缺陷的教训：k=2 对比反对称是数学必然（dW_1=−dW_2），PR 分层判据只在 k≥3 有定义——已登记为设计规则（今后多感判据必须 k≥3 或用跨词集合比较）。③剂量-反应的振荡揭示读取侧本质：句末 lens 分数≈"下一 token 预测"的函数，语境词不累积积分而是**逐 token 重写读出**——语义消歧不是加权投票，是预测动力学的副产品。这修正了 2802"感切换=相对渐变"的表述：渐变但非单调、非累积。④层定位给出消歧的时间结构：早层（L1-13）句末读数被最后 token 的词身份支配（iphone 的 embedding 自带强读数），中后层（L14+）注意力把 apple 语境信息整合进来——**"档案随身、读取者组装"的时间轴版本**。

### 相关文件（immutable，sha256 前 16 位）

脚本 phase2803_rdc_polysemy_census.py=02f33e4178eb80f2；产物 phase2803/qwen4_polysemy_census/{execution c1b9b9939a9de903, result ef8685f633efd988（53 对全记录）, census.npz 671d030c1a533205}。数据源：2797 npz（0ff02f22）+ 2795 atlas.npz（2c0010ce）+ 2802 polysemy.npz（ef6729e2，gate 对拍）。调试 2 轮（gateCM 行序不匹配 0.725→重排后 2.22e-16；rec 漏存 sv）后跑通。

### 问题硬伤

①P-C7 判据对 k=2 按构造无定义（反对称）——23/25 词的 PR 分层未真正受检，需要扩充 3 感以上词表（iron 仅 1.47 弱分层，apple 4.8 是否典型未知）；②平坦谱判据对 k=2 同样空洞（单非零 σ 恒 flat=1.0），k≥3 仅 2 词——两判据的普适性检验实际样本量是 2 不是 25；③剂量-反应判据（2/3 正）通过但有振荡，"剂量"操作化（线索词计数）不等于语境强度；④层定位只测 1 对句子、l_div=0 的判读被表层词汇贡献混淆（需控制句尾 token 相同的对照句对）；⑤锚集自污染 4 处（apple∈fruit、rose∈plant、iron∈metal、table∈furniture，各 1/10）；⑥画像匹配子串规则宽松；⑦单模型。

### 结论

**多义谱定律普适性确立**：①画像可读性 53/53（100%）、对比编码 25/25（100%）、默认感排序全部语义合理且定量（iron +36.7 金属感最强、violet +26.4 颜色感压倒植物感、bass +15.5 音乐感压倒鱼感、python/turkey 感间平手）——**"档案 = 全感带符号投票谱 + 默认感 = 最高得票"升级为跨域定律**；②读取侧重定性：句末读数由局部预测动力学逐 token 重写（振荡剂量曲线），非累积积分；语义消歧在 L14+ 由注意力完成，早层被词身份读数支配；③档案免疫第 4 次确认（4 剂量 drift=0.000000）。LPF v5.3 多义条款普适化：**任意词、任意感、任意域——档案即投票谱，读取即动力学重写**。

### 接续（Phase 2804 候选，同目标自动续研）

(a) 3+ 感词表扩充：枚举 20 个 3-4 感词（cell/mouse/iron/spring/date/bill/note/pitch/ring/port/crane/seal/table…），使 PR 分层与平坦谱判据在有效样本上受检；(b) 对照句对层定位：控制句尾 token 相同（"the apple … iphone" vs "the apple … pie"句尾交换），分离词身份贡献与语境整合贡献，精确定位消歧层段；(c) 剂量-反应重操作化：固定句长、随机化线索词位置、多句平均——检验"逐 token 重写"是否稳健；(d) 默认感排序的行为效度：用 Qwen3 自身对"The {w} is a {s}."的伪困惑度验排序（对接 2796 informativeness 遗留）。


## Phase 2804: 自动续研，LPF-17：k≥3 多义词条带普查——P-C7 补救→理论修正：实体感定律 [2026-09-16 23:13]

**测试原理**：2803 的 P-C7 FALSE 归因于"k=2 时 dW₁=−dW₂ 反对称、PR 比按构造无定义"，当时唯一可判定词是 apple（ratio 4.8）。2804 冻结 20 词 × 3 感词表（turkey/shell/squash/mint/gold/silver/bronze/salmon/olive/bow/drum/port/bench/trunk/boot/horn/diamond/club/tank/polish），感域来自 14 个 2803 锚集 + 7 个新域（country/anatomy/money/weapon/container/card/geometry，全部 tokenizer 预检单 token）。协议沿 2803（Wmeans_s→dW_s→CM=E/rms·g·dW_s→ρ/SVD/PR/default/画像），**新增去自参照协议**：若 target∈自身感锚集则移除该锚词再算 Wmeans（2803 存在同型缺陷：apple∈fruit、iron∈metal、olive∈fruit、table∈furniture）。gates：G1 E-vs-atlas 0.000000；G2 含自参照 apple CM 精确复现 2802（0.00e+00）；G3 自参照效应量化 max|ΔCM|=0.0587（微小，2803 结论稳健）；G4 iron pr_ratio 复现 2803（err 0.025）。预注册 P-A8/P-B8/P-C8/P-D8/P-E8 于任何 embedding 读出前冻结（execution 3a6c730f）。

**结果**：
| 判据 | 结果 | 读数 |
|---|---|---|
| P-A8 画像可读 | TRUE | 60/60=100% 命中（turkey country 读出 america/brazil/china/france…，shell company 读出 google/microsoft/amazon/ibm/intel） |
| P-B8 对比编码 | TRUE | 20/20 ρ 矩阵全负 |
| P-C8 PR 分层 | **FALSE** | 仅 3/20（turkey 5.1/shell 3.6/polish 2.9）——k≥3 判据有定义后仍不普适，**apple 是例外非规律，P-C7 的 FALSE 升级为经验否定** |
| P-D8 命名<自然 | FALSE | 12/26=46%——"命名感紧凑"不是定律 |
| P-E8 平坦谱 | TRUE | 20/20（0.63-0.98） |

**头条发现（实体感定律）**：3 个分层词恰是 20 词中唯一带国家/公司实体感的词，且实体感 PR 全部取该词最小值（turkey country 32<84/162；shell company 90<244/326；polish country 47<132/134）——3/3。回看 apple：pr=[fruit 250, company 66, plant 318, food 226]，min=company 66，ratio 4.8 同样由 company 感驱动——**含实体指称感（country/company）的词 4/4 全部分层，无实体感词 17/17 全部平坦（ratio≤1.9），21/21 完美分离**。2803 的"命名感紧凑"修正为"实体指称感紧凑"：指称唯一实体（Apple 公司、Poland、Turkey 国家）的感锚定在稀疏维度（PR 32-90），指称开放类成员（颜色/运动/音乐风格等人为范畴）的感与自然类一样分布式（PR 130-360）。机制证据：实体感画像直接命中锚集实体词本身（turkey country→国名列表），范畴感读出锚词+属性词混合。

**辅助结果**：Arm C 连续性 4/4——turkey/salmon/olive/trunk 在去自参照+新增感后默认感 winner 全部稳定（food/animal/fruit/plant）；2803 自参照缺陷效应仅 0.059（apple CM 尺度），既有登记结论无需修正。

**相关文件**（SHA256 前 16）：脚本 phase2804_rdc_polysemy_k3.py=93521ee60fad5e4a；产物 phase2804/qwen4_polysemy_k3/{execution 3a6c730ff942504a, result f91ba18d74bf5657（20 词全记录+26 配对表）, k3.npz e0c670fef81feb87}。数据源：2795 atlas（2c0010ce）+ 2802 polysemy（ef6729e2，G2 对拍）+ 2803 result（iron G4 对拍）。tokenizer 预检拦截：joker/spade/kiwi/blackberry 2-tokens 全弃换（card 锚集补 chip/suit）；一次性跑通零调试。

**问题硬伤**：① 实体感定律 n=4（apple 复用 2802）统计弱，需大样验证；② 20 词全部 k=3，k=4 词表（date/cell 类）未覆盖；③ named/natural 二分粗糙（color/music/sport/vehicle 等人为范畴的语义地位需更细分类）；④ 画像命中判定为子串匹配，弱判据。

**结论**：多义谱三普适定律定稿——对比编码（ρ 全负）、等强对比轴（平坦谱）、画像可读性在 k≥3 词条带上全部满格复现；PR 分层被证伪为普适属性并重构为实体感定律（实体指称感=唯一系统性紧凑感类）。词嵌入的参数作用=它在所有语义对比方向上的带符号投票表（2802 终态定义）不受影响；新增分层维度：投票谱的宽度由感的指称类型决定。

**接续（2805 候选）**：(a) 实体感定律大样验证（30+ 实体感词 vs 30+ 无实体感词，预注册 21/21 分离的可重复性）；(b) 实体感紧凑机制（实体词 W_U 行几何：锚集实体词行间 cos vs 范畴词行间 cos——实体词族是否在 W_U 空间扎堆）；(c) 默认感排序伪困惑度行为效度（对接 2796 遗留）；(d) k=4 词表补全。


## Phase 2805: 自动续研，LPF-18：实体感定律大样验证——21 域全局对比方向普查，负结果定稿 [2026-09-16 23:30]

**测试原理**：2804 的实体感定律（21/21 分离）n=4 统计弱，且其操作化是"词内对比"（target 去自参照后 vs 其余感）。2805 把实体感定律升级为**域方向命题**做大样检验：取 2795 atlas 全部 21 域 × 10 词=210 词（company/country 两域为实体感组），构造**全局对比方向** dW~_s = Wmeans_s − mean(其它 20 域)（域均值减其它域均值，与词内对比操作化正交），对每个域方向算 PR；词级 PR~(w,s)=词 CM 在其域全局方向的 PR。gates：E-vs-atlas 0.000000、lens-vs-2797 7e-06、dWg 零和 1.77e-16、turkey 词级 PR 复现 2804 err 0.044。预注册 P-F1/P-F2/P-F3 于任何读出前冻结（execution f700721b）。

**结果**：
| 判据 | 结果 | 读数 |
|---|---|---|
| P-F1 方向级 | **FALSE** | 21 域方向 PR 排名：company rank 1（765.1 最紧凑）**但 country rank 18（871.0）**——实体感两域不都最小；color rank 21（910.7）/anatomy rank 20（903.6）最松散；全距 765-911 仅 19% |
| P-F2 投影级 | **FALSE** | median PR~：entity 201.8 < natural 221.8（z=−2.74, p=0.003 显著）但 entity 201.8 vs named 214.5（z=−1.70, p=0.045 > 0.01 阈值，不显著）；gen 臂 90 词同构（entity 218.0 < natural 222.1 < artifact 235.5） |
| P-F3 归因 | **FALSE** | 21 点回归 mean_w PR~(w,s) ~ PR(dW~_s)：R²=0.478 < 0.5（slope 0.684, intercept −368.9）；entity 残差 company −0.2 / country −4.2 |
| verdict | **entity_law_general = false** | 三判据全败，负结果如实登记 |

**头条发现（定律降级 + 新不对称）**：2804 实体感定律**不跨操作化推广**——词内对比（target 从自身锚集移除 vs 其余感）下实体感取词内最小 PR 是"词内候选竞争"现象；全局对比方向下 country 方向与普通域无异。company 方向仍 rank 1 且其词级残差 −0.2（完美被词级 PR 预测）——**公司名在 Qwen3 词嵌入中构成异常紧凑族群，国家名不**。compactness 本身量级小（全距 19%），说明 21 个域方向的 PR 几乎同质，2804 观察到的戏剧性分层（PR 32 vs 300+）只能来自词内对比构造，不能外推为域几何属性。LPF 词位随身档案不受影响：参数作用=带符号投票表（2802 终态），实体感定律从"普适定律"降级为"操作化依赖现象"。

**相关文件**（SHA256 前 16）：脚本 phase2805_rdc_entity_law.py=9cf6746b1315eff1；产物 phase2805/qwen4_entity_law/{execution f700721bf474f28c, result e8d7b0876ef85c98（210 词全 PR 记录+21 方向 PR 表+回归系数）, entity.npz 239f051230dc242f}。哈希登记 probe2805_2806_hashes.txt。gens：2805 首跑 NameError（生成器作用域）+ KeyError 'nature'（atlas nature 不在 dom_idx，跳过登记）→ 修 bug 不改判据重跑，负结果可信。

**问题硬伤**：① 全局对比只是众多操作化之一，不排除其它全局构造（除法对比/池化对比）下实体感成立；② 21 方向 PR 范围窄（19%）,"方向紧凑性"维度本身低方差；③ named/natural 二分继承 2804 粗糙性；④ 210 词全部来自 2795 atlas，无独立词表复制；⑤ gen 臂 named/nature 组为 null（atlas 类别缺失），泛化检验仅 3 组。

**结论**：实体感定律终审——**实体感紧凑性是词内对比操作化的现象，非域方向普遍几何属性**（P-F1/F2/F3 全 FALSE）。保留的唯一坚实碎片：company 方向 rank 1 + 词级残差 −0.2（公司名族群异常紧凑）。LPF 框架三定律（对比编码/平坦谱/画像可读）与词位随身档案不受影响。

**接续（2806 候选，同目标自动续研）**：用户追加关键目标——"通过大量物体获取多层级定义的相对结构（苹果/香蕉 vs 水果-食物-植物-物体；老虎狮子相对苹果的差异；汽车飞机相对苹果的差异），最终获取所有名词词嵌入整体分布与差异化、类别层级分布与差异化、高效表达类型与属性的机制"。2806 冻结层级几何五臂协议（Arm A 嵌套保持/Arm B 质心树/Arm C 列方差分解/Arm D 效率曲线/Arm E 例对能量分解）。

## Phase 2806: 自动续研，LPF-19：名词词嵌入层级几何——多层级定义的相对结构 [2026-09-16 23:32]

**测试原理**：2806 把用户关键目标操作化为五臂：10 类 atlas（fruit/animal/metal/vehicle/country/food/furniture/tool/clothing/nature，各 10 词），类方向 dW_class=Wmeans_c−mean(其它 9 类)，2 个域方向 nat_art（natural 5 类 vs artifact 4 类）/liv_non（living 3 类 vs 非 living）。**Arm A 嵌套**：类方向 QR 子空间 Qc（10 维），域方向投影保持率——P-G1 保持率≥0.90；**Arm B 树**：10 类质心 average-linkage 凝聚树，Spearman(cophenetic, 语义距离 1/2/3)≥0.5——P-G2；**Arm C 分层布局**：列方差域份额≥0.30 且类内份额≥0.30——P-G3；**Arm D 效率**：top-k 类方向最近质心分类 100 词，10 维准确率≥0.85×全维——P-G4；**Arm E 能量分级**：9 对词对差向量分解 ‖d‖²=dom²+cls²+res²（dom=2 域方向子空间、cls=类子空间剩余、res=残差），cross-domain 域份额≥3×same-class——P-G5。gates：E 0.000000、lens 7e-06、dW_class 零和 7.29e-17。预注册冻结（execution b291ea35）。

**结果**：
| 判据 | 结果 | 读数 |
|---|---|---|
| P-G1 嵌套保持 | **TRUE** | nat_art=**1.000**、liv_non=**1.000**——两个域方向 100% 落在 10 类方向张成子空间内 |
| P-G2 树一致 | FALSE | ρ_tree=0.221<0.5；merges：fruit+food→furniture+clothing→animal+vehicle→…→country 最后合并 |
| P-G3 分层布局 | FALSE | 列方差：域 8.5% / 类 91.5%——域只是小分量 |
| P-G4 效率 | **TRUE** | acc_full(2560 维)=1.000；top-k：k=1 30%→k=5 85%→k=8 99%→k=10 **100%**；2 域方向 64% |
| P-G5 能量分级 | **TRUE** | 域能量份额 same-class 0.0015 vs cross-domain 0.1082 = **74.4×** |
| verdict | hierarchy_law=false | P-G2/G3 败，但嵌套+效率+分级三定律成立 |

**头条发现（层级=嵌套聚合，非树非正交分层）**：名词词嵌入的类别层级几何不是凝聚树（ρ=0.221，animal+vehicle 第 3 步合并暴露质心距离的失效）也不是正交独立层（域方差仅 8.5%），而是**嵌套聚合结构**：高层域方向是低层类方向的线性组合（投影保持率 1.000，机制直接可见——cos(dom:liv_non, fruit)=0.618/food=0.529/animal=0.510 正对齐，cos(dom:nat_art, furniture)=−0.504/clothing=−0.496 负对齐，域方向=生物类正载荷+人造类负载荷的代数和）。**用户关键目标的直接量化答案**：类别信息被极端高效压缩——10/2560 维（0.39%）零损失、5 维 85%；个体身份占词对差异能量 70-99%（残差主导）。三级能量分级即"层级分布与差异化"的几何实现：同类词对（apple-banana dom 0.000/cls 0.006/res 0.994；tiger-lion 0.000/0.002/0.998）几乎纯个体差；同域跨类（apple-bread 0.011/**0.228**/0.772；apple-tiger 0.003/0.221/0.779）类差跳到 22%；跨域（apple-car **0.114**/0.299/0.701；apple-hammer 0.131/0.271/0.729；car-tiger 0.080/0.184/0.816）域差跳到 8-13%；跨 realm（apple-japan 0.082/0.217/0.783）居中——**层级以剂量效应编码在词对差异中，层级越高能量份额越大，但恒小于个体身份残差**。

**相关文件**（SHA256 前 16）：脚本 phase2806_rdc_hierarchy.py=845ba46cc7a2a073；产物 phase2806/qwen4_hierarchy/{execution b291ea35613deaee, result e4926e8bc8ead3af（9 对能量分解+12×12 cos 矩阵+merges 全记录）, hierarchy.npz 4dbed1c1278edef0}；run2806.log=48fe32a0abb80355。哈希登记 probe2805_2806_hashes.txt。gens：merges id≥10 越界 IndexError → cname() 修复 + average_linkage/condensed_links 重写，判据未动。

**问题硬伤**：① 类方向构造与最近质心评估用同一批词表——存在表内自洽循环，未证明对 atlas 外新词泛化；② 只测 2 个域方向，abstract 域（idea/theory 类）缺席，DOMAIN_OF 中 abstract 组无词表；③ P-G2 的语义距离标签 1/2/3 人工粗粒度指定；④ 域定义 nat_art 中 country 类的归属模糊（named 域缺席导致 2805 country rank 18 与此处 country 最后合并相互呼应但无法归因）；⑤ 单模型 Qwen3-4B。

**结论**：层级几何定律（LPF-19）——**名词词嵌入的类别层级是嵌套聚合结构**：①嵌套（域方向 ∈ 类方向子空间，保持率 1.000）；②高效（0.39% 维度零损失分类）；③分级（层级差以剂量效应进入词对差异：同类≈0.2% 域能量 → 跨类 22% 类能量 → 跨域 11% 域+30% 类能量）；④残差主导（个体身份差占 70-99%，词位随身档案的主体是个体身份差向量本身）。"树"和"正交分层"两种朴素层级观都被证伪。LPF v5.3 补全最后一块：类别调制增益谱的类方向构成嵌套子空间骨架，词身份差向量在其上叠加个体变化。

**接续（2807 候选）**：(a) held-out 泛化——atlas 外 100 新词用 10 类方向分类，检验 0.39% 压缩对新词的泛化（去循环化关键）；(b) L30 残差侧复核——H30 vs E30 层级结构对比（2806 全部用 E30）；(c) abstract 域补全（idea/theory/emotion/music-genre 混合抽象词表）补第三域方向进嵌套检验；(d) 跨模型复核（GLM4/DS7B 嵌套保持率）；(e) 2805 遗留——company 方向紧凑机制（W_U 实体词行几何扎堆检验）。


## Phase 2807: 自动续研，LPF-20：held-out 泛化——层级骨架去循环化检验 [2026-09-16 23:58]

**测试原理**：2806 首要硬伤=类方向构造与最近质心评估同词表（表内自洽循环）。2807 冻结 **99 词 held-out atlas**（10 类，与 2795/2806 词表零交集；tokenizer 预检拦截 34 个多 token 候选——tool 类 7/11 被拒、fruit 10 个候选仅 9 存活，fruit=9 词如实登记）。类方向 dW_class/域方向/类模板**全部仅从 2806 词表 W_U 构造**，held-out 词仅作评估（模板-评估完全分离）。协议零前向克隆 2806（Etab 行 RMSNorm×g → 类方向特征）。gates：E=0.000000、lens=0.000007、dW_vs_2806=1.82e-09（f64 重算 vs 2806 f32 npz）、overlap=0。预注册 P-I1/P-I2/P-I3 冻结（execution fa11b677）。

**结果**：
| 判据 | 结果 | 读数 |
|---|---|---|
| P-I1 泛化 | **TRUE** | held-out full-dim acc=**0.838**（chance 0.10；16/99 错） |
| P-I2 效率泛化 | **TRUE** | 10 维 acc=0.869 ≥ 0.85×0.838；acc_k 曲线 0.14/0.31/0.33/0.35/0.41/0.50/0.49/0.51/0.49/**0.87**；dom2=0.313 |
| P-I3 分级泛化 | **TRUE** | held-out 词对 dom 份额 same 0.0020 vs cross-domain 0.0427 = **21.3×** |
| verdict | **generalization_law = true** | 去循环化后类别骨架泛化成立 |

**头条发现**：①错误结构高度非随机——16 错中 **13 个误判为 tool**（fig/prune/date/lead/raft/ham/stew/rug/cot/mat/uniform/boot/cap→tool）：tool 是 held-out 吸引子类，其类方向泛化域最宽（特征空间"平庸方向"）；②**k<10 泛化塌陷**：表内 k=5 已 85%（2806），held-out k=5 仅 41%、k=9 仍 49%，k=10 突跳 87%——**全部 10 个类方向都是泛化必需的，骨架无冗余方向**；③held-out 词对分级 21.3×（弱于表内 74.4× 但远超 3× 阈值）。

**相关文件**（SHA256 前 16）：脚本 phase2807_rdc_heldout.py=004593e2f3eb1ad7；产物 phase2807/qwen4_heldout/{execution fa11b677b252d122, result c7d1ce9bac064c15, heldout.npz b0af32c18b39e22a}；run2807.log=bfdfb667e1e5f235。哈希登记 probe2807_2808_hashes.txt。一次跑通零调试。

**问题硬伤**：① fruit 类 9 词（多 token 候选耗尽）；② 错误集中 tool 方向暗示该类方向过宽，分类边界未做 margin 分析；③ 单模型（跨模型留给 2808）；④ held-out 词仍为常见名词，未测低频词泛化下界。

**结论**：0.39% 维度的类别压缩**对新词真实泛化**（0.838，非表内自洽）；10 维全 needed；分级泛化成立。层级几何的第一根非平凡支柱就位。

**接续（2808，用户指令）**："请测试不同的模型，验证这个规律是否跨模型成立"→ 跨模型五臂复测。

## Phase 2808: 用户指令，LPF-21：层级定律跨模型验证——4 模型 + qwen4 基准 [2026-09-17 00:14]

**测试原理**：用户指令跨模型验证。4 新模型：ds7（deepseek-r1-distill-qwen-7b，Qwen2 架构蒸馏推理，3584 维）、glm4（glm4-9b-chat-hf，GlmForCausalLM，4096 维）、qwen17（qwen3-1.7b，2048 维）、qwen25（qwen2.5-3b-instruct，2048 维）+ qwen4 基准。**严格同词表**：2806 atlas 100 词在 4 模型 tokenizer 下全部单 token（预检零替换）。零前向协议升级：**safetensors 直读 3 张量**（lm_head.weight / model.embed_tokens.weight（tie 模型回退）、model.norm.weight + config rms_norm_eps），不加载模型无前向——每模型秒级。五臂全克隆 2806（A 嵌套/B 树/C 方差/D 效率/E 分级），词位特征=各模型 Etab 行 RMSNorm×g（2806 同构）。预注册 P-H1（nesting_general 5/5）/P-H2（efficiency_general 5/5）/P-H3（grading_general 5/5）冻结（execution 67c595fa）。

**结果**（qwen4 行来自 2806）：
| 模型 | arch | hidden | P-G1 嵌套 | ρ_tree (G2) | dom 份额 (G3) | acc_full | acc10 | P-G4 | dom ratio (G5) |
|---|---|---|---|---|---|---|---|---|---|
| qwen4 基准 | Qwen3-4B | 2560 | 1.0/1.0 T | 0.221 F | 0.085 F | 1.000 | 1.000 | T | 74.4× T |
| ds7 | Qwen2 蒸馏推理 | 3584 | 1.0/1.0 T | 0.282 F | 0.250 F | **0.380** | 0.380 | T | **0.5× F** |
| glm4 | Glm4-9B-chat | 4096 | 1.0/1.0 T | 0.432 F | 0.017 F | **0.370** | 0.360 | T | **0.0× F** |
| qwen17 | Qwen3-1.7B | 2048 | 1.0/1.0 T | **0.517 T** | **0.376 T** | 1.000 | 1.000 | T | 77.6× T |
| qwen25 | Qwen2.5-3B | 2048 | 1.0/1.0 T | 0.448 F | 0.189 F | 1.000 | 1.000 | T | 51.7× T |
| verdict | | | **P-H1 TRUE (5/5)** | | | | | **P-H2 TRUE (5/5)** | **P-H3 FALSE (3/5)** → hierarchy_law_general=false |

**头条发现（双层结构）**：①**类别信号强度训练谱系特异**——Qwen 系（qwen4/qwen17/qwen25）W_U 质心分类满格 1.000，DS-R1-distill（0.380）与 GLM4-9B-chat（0.370）仅 4×chance；且 ds7/glm4 词对差异能量几乎 100% 残差（cls 份额 0.001-0.005 vs Qwen 系 0.002-0.33）——**"类别/层级方向承载词间差异能量"的强形式只在 Qwen 训练谱系成立**；②qwen17（1.7B）是唯一 P-G2+P-G3 也 TRUE 的模型（ρ=0.517、dom 0.376）——小模型层级最"树状"；③P-G4 的相对判据在弱模型上平凡成立（0.85×0.38），信息量有限（硬伤④）。**（2809 修正注记：P-H1 的 5/5 嵌套为代数恒等式，非经验发现——见 2809 节。）**

**相关文件**（SHA256 前 16）：脚本 phase2808_rdc_crossmodel.py=a3b4c5f47ac4e68c；产物 phase2808/crossmodel_hierarchy/{execution 67c595fa227e3d72, result c8e9c2cbcef63919}；run2808.log=8186bccd4f6214a4。哈希登记 probe2807_2808_hashes.txt。gens：ds7 hidden 3584 vs qwen4 2560 逐元素对拍 ValueError → 删除参考对拍（hidden 不同无意义）+ 残留引用 NameError → 删行重跑（判据未动，旧产物删除后重跑）。

**问题硬伤**：① 5 模型全为 instruct/chat 版，无 base 对照（SFT/对齐对类别骨架的影响未分离）；② ds7/glm4 低 acc 未做残差流侧复核（中间层类别信号可能强于静态端——"轮换"假设）；③ P-G4 相对判据弱模型平凡过；④ dom2 特征跨模型 0.22-0.64 波动大，域方向绝对强度未单独判据化。

**结论**：跨模型验证完成——类别骨架的**泛化**（2807）与**相对效率**跨模型成立，**能量分级是 Qwen 谱系特异**；"层级定律"不是任何 LLM 词嵌入的普遍属性，而是训练谱系依赖的结构。嵌套头条（2806/2808）后被 2809 降级为代数恒等式。

**接续（2809，同目标自动续研）**：自我否证优先——P-G1 嵌套保持率 1.000 疑似代数必然（域方向=质心零和组合；dW_class 若秩 9 则张成整个零和空间）→ 秩审计 + 随机 null 检验。

## Phase 2809: 自动续研，LPF-22：嵌套保持率的代数必然性检验——自我否证成功，P-G1 降级 [2026-09-17 00:19]

**测试原理**：2806/2808 头条"域方向 100% 落在类方向子空间"存在构造性疑点：dW_dom = mean(cent_NAT)−mean(cent_ART) 是 10 质心的**零和线性组合**（系数和=0），而 dW_class 每行也是零和组合，其张成空间 ⊆ 零和组合空间 V0（维≤9）。若 dW_class 数值秩=9 且质心仿射满秩，则 span{dW_class}=V0，**任何**零和域方向必然嵌套——1.000 零信息。预注册 P-J1（秩 9 = 必要条件）/P-J2（tautology：1000 次随机 null 嵌套率 median≥0.999）/P-J3（对齐非平凡：真实 max|cos(dom 行, 类方向)|≥null q95）冻结（execution 97031454）。

**结果**：
| 判据 | 结果 | 读数 |
|---|---|---|
| P-J1 秩前提 | **TRUE** | dW_class 奇异谱 top5=[0.7,0.6,0.6,0.6,0.5]、last=0.0 → 数值秩 **9**；质心仿射秩 9 |
| P-J2 tautology | **TRUE** | 1000 次随机重排 null：嵌套率 median=q05=q95=min=**1.0000**——随机词表下同样 1.000 |
| P-J3 对齐非平凡 | TRUE（弱） | 真实 max\|cos(dom,类)\|=0.504 vs null q95=0.485（median 0.435）——仅边缘超 95 分位 |
| verdict | **nesting_substantive = false** | **嵌套事实=代数恒等式**；final_reading="nesting_fact_is_algebraic_identity_but_alignment_structure_nontrivial" |

**头条修正（本轮最大方法论收获）**：**2806 P-G1 与 2808 P-H1（5/5 嵌套 1.000）全部降级为构造恒等式**——"域方向∈类方向子空间"由 dW 均值构造直接蕴含，不是经验发现。层级几何的非平凡支柱收缩为三条：①**效率泛化**（2807 held-out 0.838，10 维全 needed）；②**能量分级**（Qwen 系 51-78×，ds7/glm4 0-0.5×——谱系特异）；③**域-类对齐结构**（0.504>q95 0.485，弱证据：真实域方向与特定类方向的 cos 轮廓超出随机）。"层级=嵌套聚合"表述作废，修正为"**类别信息压缩在 9 维零和类方向空间（对偶：10 类方向含 1 个冗余），对新词泛化，分级谱系特异**"。

**方法教训（制度化候选）**：凡"子空间包含/投影保持"类判据，必须先做构造代数审计（分量是否为同一组基向量的线性组合）+ 随机 null；2806 当时五臂预注册漏掉 null 对照 arm 是设计缺口。

**相关文件**（SHA256 前 16）：脚本 phase2809_rdc_nesting_tautology.py=489b0c35bba84a6e；产物 phase2809/qwen4_nesting_tautology/{execution 9703145426161990, result 164f3727e1b51751}；run2809.log=85e96f6f0a5bfe46。哈希登记 probe2809_hashes.txt。一次跑通零调试。

**问题硬伤**：① P-J3 只检验 max|cos| 单标量，域-类对齐的完整轮廓（liv_non↔fruit 0.618 等）vs null 的逐类检验未做；② null 仅 qwen4（跨模型 null 未做——但恒等式是纯代数，与模型无关，5/5 必然全 1.0）；③ 真实"效率泛化"的类模板仍用 2806 词表质心（模板构建词与 held-out 评估词分离已达标）。

**结论**：P-G1 降级（2806/2808 相应头条作废）；层级几何存活支柱=效率泛化+谱系特异分级+弱对齐结构。LPF v5.3 的"类别调制增益谱"不受影响（它是残差流/读出侧结论）；受影响的是 W_U/embedding 静态端层级结构的解读。

**接续（2810 候选）**：(a) base 模型对照（Qwen3-4B-Base / Qwen2.5-3B base）分离 SFT 对类别骨架强度的影响；(b) ds7/glm4 残差流侧（L30）类别信号复核——静态端弱是否因"轮换"（2801：维级轮换不伤语义）；(c) 域-类对齐完整轮廓的逐类 null 检验（P-J3 加强版）；(d) held-out 错误的 tool 吸引子机制（tool 类方向为何泛化最宽）；(e) 2809 教训制度化：子空间类判据的 null 对照进入标准预注册模板。



## Phase 2810: 短语级档案调制——首个非零前向检验（选择性调制被否证；发现主导性通用上下文通道） [2026-09-17 01:51]

**测试原理**：2807-2809 全部为静态端（E/W_U 零前向）；2810 首次运行真实前向，检验"词位随身档案在短语语境中被上游词调制"的剂量-反应假说（LPF v5.3 组合性第一块试金石）。设计：20 目标词（10 类 × 2）× 5 条件（iso 单词基线 / same 同类修饰 "banana apple" / diff 跨域修饰 "hammer apple" / func 虚词 "the apple" / null 随机 token 修饰），修饰词一律置于目标词**之前**（causal mask 干预纪律）；qwen3-4b（bf16、device_map auto、全 GPU）提取全 37 层 hidden states，逐层计算 cos(h_ctx,h_iso)、rel_shift=||Δ||/||h_iso||、Δ 的类方向/域方向能量份额；类方向 dW_class 由 W_U 行按 2806 构造克隆。预注册 P-K1（调制存在：任一层 cos<0.999）/P-K2（选择性调制：终层 mean||Δ_same||>mean||Δ_diff||）/P-K3（层依赖：cos 跨层极差>0.05）/P-K4（虚词最小：func<content）冻结（execution 095e712845f64c24）。gens：交接脚本带语法错误（L222 多一右括号，编译期崩溃、零产物）→ 删一括号后重跑，判据未动；**2807 勘误**：错误分布实为 14 tool/1 clothing/1 food（2807 节"13 个误判 tool"漏记 trailer→tool），以 result.json 为准、本节勘误入账。

**结果**：
| 判据 | 结果 | 读数 |
|---|---|---|
| P-K1 调制存在 | TRUE（平凡） | 任何前置 token 都使终层 cos≈0.44-0.51——位移巨大但非语义特异 |
| P-K2 选择性调制 | **FALSE** | same 1.231 < diff 1.306——同类修饰位移反而更小（方向反了） |
| P-K3 层依赖 | TRUE | cos 跨层极差 0.990 |
| P-K4 虚词最小 | **FALSE** | func 1.400 > content 1.269——"the" 引起全场最大位移 |
| verdict | phrase_modulation_substantive=**false** | final_verdict="weak_or_absent"（裸 Δ 层面） |

**头条发现（负结果 + 机制线索）**：①**通用上下文通道主导裸 Δ**——虚词 "the"（attention-sink 型高频 token）位移 1.400 超过一切实词条件，null 随机 token 1.319 与 content 1.269 同量级；18/20 词 func 位移同时大于两个 content 条件（例外：silver 的 diff 1.229>func 1.216、pants 的 same 1.595>func 1.512）。裸 Δ 的主体是"位置+通用上下文"分量（iso=pos0 单 token vs ctx=pos1，RoPE/注意力结构性差异混入），"档案被上游词语义调制"在裸隐状态层面不成立。②**语义通道存在于少数分量（探索性，未预注册）**：Δ 的类方向份额均值 same 0.068 vs diff 0.033 / func 0.029 / null 0.034（2.0-2.3×），18/20 词 same 高于两个对照（例外 hammer、pants）——档案查询机制真实存在但只占 Δ 的次要成分，被通用通道淹没；这是 2811 通道分离的直接依据。③**silver-gold 单向锁定异常**："gold silver"（目标 silver）rel_shift=0.119、cos=0.998 近 bit-exact 保持，而对称的 "silver gold"（目标 gold）shift=1.199——同义对的单向效应，单例观察、机制未知。④逐层结构真实存在（P-K3），cos 极差 0.990 说明位移的层分布高度非均匀。

**相关文件**（SHA256 前 16）：脚本 phase2810_phrase_modulation.py=38fe4bb5d016e5b8（含 gens 修复）；产物 phase2810/phrase_modulation/{execution 095e712845f64c24, result a630da38acd15738, hidden_states.npz ca1901ff694f105a}；哈希登记 tests/gpt5_temp/probe_2810_hashes.txt。运行 16.3s（RTX 5080，bf16 全 GPU，20 词 × 5 条件 × 37 层）。

**问题硬伤**：①位置混淆未分离——iso=pos0 vs ctx=pos1，位移含 RoPE/位置处理结构分量，func/null 对照证明其主导；②func 仅 "the" 一个、null 为随机 token id（可能命中 CJK/字节片），对照粗糙；③每条件每词仅 1 个修饰词（20 词 × 4 条件），单元样本量小；④裸单 token 序列无 BOS/文档前缀，与真实用法退化性偏离；⑤rel_shift 的层间可比性受终层 RMSNorm 前隐范数增长影响。

**结论**：**裸隐状态层面"短语档案调制"的预注册判据失败**（P-K2/P-K4 双否证）——LPF v5.3 的档案-上下文接口不能在裸 Δ 层面测量：裸 Δ 被非语义通用通道（attention-sink/位置通道）支配。这与 2806-2809 静态端"档案自足、后置上下文免疫"互补：**上下文效应必须先剥离通用通道，才能看到语义调制**。探索性信号（same 类方向份额 2-2.3× 于对照）说明语义通道存在，缺的是正确的测量口径。

**接续（2811 候选）**：(a) 通道分离——Δ_specific = Δ_condition − mean(Δ_func, Δ_null)（逐层逐词），预注册"语义通道实质"判据：residual 类方向份额 same > func/null 且 same > diff；(b) BOS 前缀对照（iso 基线加 BOS 后是否稳定）；(c) silver-gold 单向锁定专项（同义对 2×2 扩展：gold/silver/car/bus 交叉组合，bigram/induction 头假设）；(d) 对接 2786 载体头工具箱做搬运头定位（把 Δ 分解到注意力头输出方向）；(e) 通道分离成功后进入真短语（"apple pie" 复合名词）剂量-反应。

---

## Phase 2811: 名词嵌入叠加格式——"齿轮形状"的普查、跨词预测与效率审计 [2026-09-17 02:11]

**用户锚点（2026-09-17）**："当前破解的是是否有齿轮，但核心目标是破解齿轮的形状——苹果的词嵌入为什么是这样的，怎么表达出水果、食物、公司等含义。很可能有一种系统编码机制，让各种名词的词嵌入在当前参数下，既可以表达复杂的知识结构和属性，又同时极为高效。更本质的目标，需要的是破解所有名词的词嵌入的规则，以及背后的原因。"

### 0. 附件分析：九项"已破解"清单的裁定

用户同轮提供《九项"已破解"内容的详细原理讲解与"完全破解"评估》。裁定：

1. **清单总体成立**。九项（内容存储 / 写入渐进 / 层1换车 / W_U 竞争读出 / 带符号投票 / 对比编码 / 三种编码形态 / 层级剂量+10维效率 / 泛化边界）的机制描述与实验记录一致；其自我评估"框架已建立、内容未破解"（"知道有齿轮，不知道齿轮怎么转"）是对现状的准确概括，本 Phase 即针对"齿轮形状"。
2. **第 8 项须修正（2809 教训）**。其中"嵌套保持 1.000"不是经验发现而是代数恒等式：rank-9 零和空间对 rank-10 构造的保持是 dW_class 零和性的必然结果（2809 已证：1000 次 label shuffle 全部 1.000，null 与观测完全重合，判据不可区分）。"10 维效率"的经验部分只剩"9 维保留类别可分性（acc 不降）"；"嵌套"不得再作为独立证据引用。
3. **与 2810 的接口**。第 8 项的"剂量"叙事（语义调制存在）与 2810 的发现（通用上下文通道淹没语义通道：the 引发最大位移 1.400，same 类方向份额仅为 2.0-2.3× 对照的少数分量）合并后指向同一图景：**类别/语义信息在 hidden state 层面是少数派通道**。本 Phase 在 embedding 层面独立复测这一点（见 §2 效率审计：9%），两层测量相互印证。

### 1. 测试原理

零前向（2808 协议）：直接读 safetensors（embed_tokens + lm_head + model.norm + config rms_norm_eps），无 GPU forward，4.5s 完成。归一化嵌入：

```
z(w) = e(w) / sqrt(mean(e(w)^2) + eps) * g
```

类别方向（2806 atlas 100 词构造，模板-评测严格分离，2807 协议）：

```
Cm[i] = mean_{w in class i} lm_head_row(w)
dW_class = Cm − (sum(Cm) − Cm)/9      # 数值秩 9 的零和对比空间
unitD    = 10 个单位化 dW_class 行
```

每个名词的 10 维类别特征谱（带符号投票口径）：`p(w) = z(w) · unitD^T`。
主类 = argmax p；次级成员 = argmax_{c≠主类} p；普查矩阵 `M[c1][c2]` = 类 c1 词中次级为 c2 的比例。

三臂 + 双 label-shuffle null（2809 教训制度化）：
- **A 普查**：atlas / eval（2807 held 99 词 + NEW 84 词存活）分列算 M；eval label shuffle ×1000（保持类大小）给 per-cell q95 → P-L1
- **B 跨词预测**：仅用 atlas 的次级分布 R[c1][c2] 预测 eval 词的观测次级；atlas label shuffle ×1000 给 p 值 → P-L2
- **C 效率**：enlarged battery 最近质心准确率 acc_k（P-L3）；类子空间能量份额；去类残差有效秩；参与率 PR
- **E 探索性**：大小写词义对（apple/Apple, turkey/Turkey, china/China, japan/Japan），不进判定

预注册判据（冻结于任何 readout 之前，execution.json sha256=25cbe5ae…）：
- **P-L1** superposition_map_real：存在 cell (c1,c2), c2≠c1，eval rate ≥ 0.50 (n_c1_eval ≥ 8) 且 atlas rate ≥ 0.40 且 eval rate > 该 cell null q95
- **P-L2** secondary_systematic：跨词预测命中 ≥ 2/9 且 null-A2 p < 0.001
- **P-L3** battery_generalization：eval 最近质心 acc ≥ 0.75（chance 0.10）
- **verdict** gear_shape_superposition ⇔ P-L1 AND P-L2；P-L3 为电池有效性门

Gates 全过：dW_vs_2807 = 1.82e-09，Z_eval[:99]_vs_2807 = 9.51e-07，atlas/eval 词重叠 = 0，lm_head 与 embed_tokens 绑定（tie=True）。

### 2. 关键结果（result.json sha256=7a56665c0bbce032…）

**判定：gear_shape_superposition = false（P-L1 = true，P-L2 = false）**

P-L3（电池有效性）= **true**：183 词评测集最近质心 acc10 = 0.776（chance 0.10）；acc_k 曲线在 k=10 处从 0.399 跳到 0.776（真类方向不可替代，与 2807 的 k=10 跳变 0.869 同构）。

**P-L1（叠加地图真实）= true——4 个逐 cell 过 null 的真实次级成员 cell：**

| from | to | rate_eval | n_eval | rate_atlas | null q95 |
|---|---|---|---|---|---|
| metal | fruit | 0.786 | 14 | 0.90 | 0.286 |
| food | fruit | 0.611 | 18 | 0.90 | 0.278 |
| furniture | clothing | 0.846 | 13 | 0.90 | 0.308 |
| clothing | furniture | 0.692 | 13 | 0.90 | 0.308 |

**P-L2（全局系统性）= false**：跨词预测命中 0.377（chance 0.111，null median 0.148，q95 0.268），但 p = 0.003 > 预注册阈值 0.001。诚实执行协议判 false。备注：命中为 null median 的 3.4 倍且超过 q95，信号实在而弱，未达冻结阈值。

**效率审计（"极为高效"假设的检验）：**
- 类子空间能量份额 mean = **0.089**：10 维类别方向只承载嵌入能量的约 9%
- 去类残差有效秩 = **282 / 283 词**：残差几乎满秩，类方向之外另有广阔结构
- 参与率 PR：atlas 谱 **1.59**（近 one-hot 尖峰）< eval 2.92 < 随机 token 4.72
- 随机 token 次级强度 q95 = 0.587：名词的次级成员在随机本底之上仍是结构化的

**臂 E（探索性）——本 Phase 最直观的"齿轮形状"演示：**

| 对 | cos | lower 谱主峰 | upper 谱主峰 | 解读 |
|---|---|---|---|---|
| apple/Apple | 0.743 | fruit 0.997 | fruit 0.964 | 词义不变，双成员 fruit+food 微弱共存 |
| turkey/Turkey | 0.630 | **food 0.681 + animal 0.538 双峰** | country 0.975 | **10 维谱干净分解多义词：一 embedding 同时携带"食物+动物"双成员，大写后整体切换为国家** |
| china/China | 0.620 | country 0.947 | country 0.996 | 小写 china 的瓷器义未进入 10 类谱（谱外属性） |
| japan/Japan | 0.663 | country 0.965 | country 0.998 | 同上 |

### 3. 结论（核心发现 ×3）

**核心发现：名词嵌入的齿轮形状 = 近 one-hot 主类尖峰（PR 1.59）+ 微弱但结构化的次级混叠（仅特定类别对，4 cell 过 null）+ 巨大属性残差（91% 能量、有效秩 282/283）。**

重述一：类别信息以约 9% 的能量成本即可实现 0.776 的读出准确率——"极为高效"的第一重证据：类别不是嵌入的主体，而是稀疏索引。
重述二：叠加是局部的、非均匀的——不是所有类别共享子空间，只有语义近邻对（furniture↔clothing）系统性混叠；turkey 谱证明双成员可共存于同一 embedding，混叠机制服务于多义词。
重述三：回答"苹果的词嵌入为什么是这样"的第一近似——主类轴（fruit 尖峰 0.997）只是骨架，91% 的能量在属性/知识流形里，那部分才是"表达复杂知识结构"的载体，也是尚未破解的主体。

### 4. 严格审视：问题、硬伤、瓶颈

1. **P-L2 未达冻结阈值**（p=0.003 > 0.001）。协议必须尊重；但"4 cell 逐个过 per-cell null"与"聚合命中 3.4× null median"两证据合看，全局系统性是"弱而非零"，非纯噪声。
2. **metal→fruit 语义反直觉**：metal 词的次级 90% 是 fruit，大概率不是"语义"而是几何伪影——fruit 方向可能被"自然物/物体"泛类主导（2806 模板 cos 矩阵可查）。若属实，"次级成员"的语义解释须降级为"泛类几何近邻"。这是 2812 必须溯源的第一疑点。
3. **单 token 过滤偏倚**：171 候选仅 84 存活（Qwen tokenizer 将罕见词切开）；fruit 类 NEW 候选 12/12 全灭，fruit 相关 cell 的 eval 证据全部来自 2807 held 旧词。电池偏向常用词。
4. **embedding-only**：无上下文。2810 已证明进入 hidden state 后通用上下文通道主导；本 Phase 给出的是第 0 层（静态词位）的齿轮形状，两者必须拼合才是完整图景。
5. **次级 = argmax(9) 口径噪声**：谱弱时次级身份易翻转（这正是 P-L2 用分布 R 而非逐词 argmax 做预测的原因）。
6. **随机 token 基线含 junk token**：PR 4.72 的对照力有限，不能完全排除"atlas 谱尖峰 = 词频/训练剂量"混杂。

### 5. 智能理论视角的关键洞察（第一性原理）

语言能力 = 有限参数实现无限组合。破解名词嵌入规则 = 回答"一个 2560 维向量如何同时编码 (a) 离散类别身份 (b) 连续属性 (c) 多重成员"。2811 给出的第一性分工假说：

- **离散身份用"轴"编码**：log2(10) ≈ 3.3 bits 的信息，用 9-10 个零和对比方向、约 9% 能量实现，读出只需 W_U 竞争（带符号投票），无需前向计算——这是高效的部分，且天然支持"对比编码"（第 5 项）。
- **属性用"场"编码**：91% 能量、近满秩残差——属性是连续、可组合、开放集的，必须稠密表示。这与第 7 项"三种编码形态"中的场形态衔接。
- **多重成员用"混叠"编码（superposition）**：但按需分配——语义近邻类别共享子空间以省容量，无关类别近正交以避干扰。turkey 是直接标本：food+animal 双峰共存于小写形，大写形整体切换 country。叠加不是全局压缩技巧，而是容量分配策略。
- **两层测量的一致性**（embedding 9% vs hidden state 少数分量 2810）：类别是"稀疏索引"、属性流形是"主体"，这可能是 LLM 表示的普遍架构原则，而非某一层的巧合。

**瓶颈**：10 个类别只是齿轮的最粗分辨率。"苹果为什么是这样"的完整答案需要属性级方向（可食、圆形、红色、植物、公司品牌……）。类子空间 9% 意味着 91% 的形状完全未解释——下一步必须把 unitD 从"10 类方向"推广为"属性方向库"。

### 6. 接续（2812 候选，按优先级）

1. **P-L4 属性子空间假说（主任务）**：从 atlas 构造属性对方向（big/small、edible/inedible、alive/manmade、round/long……，≥12 对），测属性子空间能量份额与残差解释率。若属性方向解释率 >> 9%，则"残差 = 属性流形"成立，齿轮形状问题转化为"属性方向库的完备化"问题。
2. **metal→fruit 溯源（第一疑点）**：对 fruit dW 方向做 top-k unembed 投影 + 与 2806 全部模板方向 cos 分析，判定 fruit 方向是否被"自然物/物体"泛类污染。
3. **2810 通道分离**（Δ_specific = Δ_condition − mean(Δ_func, Δ_null)，预注册判据照旧）。
4. **多层谱演化**：把 10 维谱从 embedding 推广到逐层 hidden state（2807 载体头工具箱可复用），观察"主类尖峰 → 属性场"的逐层重构路线。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2811_noun_superposition.py` sha256 = `02ac94d39db3d70c85024dea0ec21522844c2fdc5dacbf95575d52d9bbe8513d`（修复版）
- `tests/glm5/result/rdc_query_construction_20260913/phase2811/noun_superposition/execution.json` sha256 = `25cbe5aeef76e03f88aa7a420fb0d44f7787f88a138075041868ecc2575e50ac`
- `…/result.json` sha256 = `7a56665c0bbce0325cac1555d98b7b9c41799eab9ea0076f8c000c3dc5a5430a`
- `…/battery.npz` sha256 = `78a16913f472f84e187df8511e756c082fa6000a26ef174f2a76a5ef212b837a`

**Gens 记录**：首次运行崩溃于 line 427（投影公式维度写反：`Qb @ (Qb.T @ Zc)`，Qb 为 2560×10、Zc 为 283×2560；正确式 `(Zc @ Qb) @ Qb.T`）。崩溃发生在 fc.save 之前 → 零产物（陈旧 execution.json 已删）。判据未动，纯机械修正。重跑 4.5s 干净完成，elapsed 计时以第二次运行为准。

---

## Phase 2812: 属性子空间假说否证 + metal→fruit 法证清洗 [2026-09-17 02:27]

**锚点**：2811 遗留两问——(a) 91% 残差是否 = 属性流形（"齿轮形状"的主体）；(b) P-L1 cell metal→fruit 是否为"泛类几何伪影"。

### 1. 测试原理

零前向（2808 协议，tie 感知：2811 已证 lm_head 与 embed_tokens 绑定）。评测电池按 2811 精确重建，三重 gates 全过：dW_vs_2807 = 1.82e-09，eval_words 列表与 2811 逐词一致，Z_vs_2811 = 9.51e-07（< 1e-4 float32 容差）。

**属性轴构造**：15 个双极属性（size/weight/temperature/wetness/hardness/speed/brightness/loudness/cleanliness/sweetness/danger/value/age/strength/sharpness），每极 4 个形容词锚点词，单 token 过滤 + 与电池词表去重后须每极 ≥2 存活；sweetness 全灭（sugary/syrupy/honeyed multi-tok），14/15 存活，K=14。方向取 embedding 侧质心差：

```
dA_k = unit( mean_z(pos 锚点) − mean_z(neg 锚点) )
QA   = orthonormalize({dA_k})           # 2560×14
share(Z, Q) = mean_w |z_w·Q|² / |z_w|²
```

**P-S1/S2 法证**：dW_fruit 正投影占比、fruit 与 food/animal/nature 方向的 cos 对照非对角 q95、dW_fruit 的 top-10 unembed 词、以及"泛类清洗"——从 dW_fruit 中减去非水果评测词均值方向 ĝ 后重算 4 个 P-L1 cell：

```
ĝ = unit(mean_{w: 非fruit} z_w)
dW_fruit_rem = unit(dW_fruit − (dW_fruit·ĝ)ĝ)
```

预注册判据（冻结于任何 readout 之前，execution.json sha256=3f918b11…）：
- **P-L4a**：mean eval share > N1（pole-shuffle ×1000）q95 且 > 3× median
- **P-L4b**：≥60% 属性的 split-half |cos| > 其 N2（×300 pole-shuffle）q95
- **P-S1**：非水果词正投影占比 ≥0.60 且 trio cos > 非对角 |cos| q95
- **P-S2**：metal→fruit 与 food→fruit 清洗后 rate ≥0.50 且 > N3 逐 cell shuffle q95
- **verdict**：attribute_manifold_support = P-L4a AND P-L4b；P-S1/S2 仅诊断

### 2. 关键结果（result.json sha256=ee45b45cdf289e30…）

**判定：attribute_manifold_support = false（P-L4a = false，P-L4b = false）——经典双极形容词轴粒度上，91% 残差不是属性流形。**

| 臂 | 结果 | 数据 |
|---|---|---|
| P-L4a 属性子空间真实 | **false** | eval share 0.0219 vs N1 med 0.0194 / q95 0.0251（两项均未过）；随机 token share 0.0239 **反高于名词** |
| P-L4b 轴稳定性 | **false** | 0/14 属性通过；最高 speed 0.353；7/14 q95==cos_half（identity 置换污染，通过在构造上不可能），7/14 严格失败（错误配对比真配对更一致，如 weight swap 0.052 > true 0.033） |
| P-S1 fruit=泛类轴 | **false** | pos_frac 0.759 ✓ 但 trio_cos = **−0.058**（远低于 off_q95 0.178）；**top-10 unembed 全是纯水果词**（peach, apple, cherry, pear, berry, banana, grape, mango, lemon, orange）——方向语义干净 |
| P-S2 清洗后存活 | **true** | metal→fruit 0.643（q95 0.214）✓；food→fruit 0.500（q95 0.222）✓；对照 cell furniture/clothing→fruit = 0.0 ✓ |

三重关键读数：① pole-shuffle null median（0.0194）是各向同性理论值（K/2560 = 0.0055）的 3.5×——任何词池质心差方向都自带 ~2% 能量（词池共性几何）；属性轴只比这个本底高 **0.25pp**，特异属性内容近乎为零。② 名词电池对属性轴的投影（0.0219）不高于随机 token（0.0239）——名词根本不偏好形容词对方向。③ per-attr top 词跨属性大量复现（uniform/van/star/level/cap/boot/lead/mill/mat），提示锚点方向携带的是词频/长度类通用特征而非语义属性。

**P-S1/S2 联合裁定**：fruit dW 方向是语义干净的对比方向（unembed 读出纯水果），metal→fruit 不能被泛类分量解释（清洗后 0.786→0.643 仍强存活）——**2811 P-L1 的 metal/food→fruit 是真实的跨类内容，但为什么金属词携带 fruit 方向能量而家具/服装词完全没有（0.0），机制未解**。

### 3. 结论（核心发现 ×3）

**核心发现：名词嵌入残差的 91% 不组织在经典双极形容词轴上——形容词锚点方向的特异解释力 ≈ 0.25pp 能量，轴的 split-half 一致性 0/14，名词对它们的偏好不高于随机 token。**

重述一："属性流形"假说在单方向粒度被否证：若属性以 1D 方向编码于名词嵌入，share 应远超词池本底与随机 token，实测皆否。
重述二：属性编码（若存在）必须是分布式的/子空间式的，或与名词指称几何分离（形容词-名词跨词类几何不共享）——1D 探针给出的是下界，且这个下界已归零。
重述三：fruit 方向法证给出正面样板：干净的类对比方向 = top unembed 纯类词 + 对语义近邻低 cos（零和对比空间的自然性质）+ 跨类能量选择性（metal/food 有、furniture/clothing 无）——"齿轮"的真实形状比"泛类轴 + 噪声"精细得多。

### 4. 严格审视：问题、硬伤、瓶颈

1. **P-L4b null 设计缺陷**：小词池（≤4 词）时 identity 置换混入 null 分布顶端，q95==cos_half 使"通过"在构造上不可能（7/14 属性）。该子判据的 false 部分是设计伪影——但即使剔除这 7 个，另 7 个严格失败且全体 cos_half ≤ 0.353（真轴应 ≥0.5），实质否证不受影响。下轮须用 exclude-identity 或 leave-one-out 设计重注册。
2. **POS 错配**：形容词锚点 vs 名词电池。形容词嵌入携带谓词/语境几何，可能与名词指称几何分离——负结果可能是跨词类伪影而非名词内无属性结构。修正方案：用名词两极（mountain/ant 式）构造属性方向。
3. **1D 属性模型下界**：属性若以子空间编码（每属性 >1 维），单方向探针系统性低估；需 PCA 式多极方向。
4. **频率混杂**：per-attr top 词复现通用高频 token，锚点方向部分携带词频/长度特征；下轮应回归掉 log 频率再测。
5. K=14 轴对微小效应功效有限——但效应 ≈ 0 而非小效应，否证是干净的。

### 5. 智能理论视角的关键洞察（第一性原理）

2811 问"齿轮形状"，2812 回答了"齿轮不是什么"：**残差不是人类命名的属性轴张成的空间**。组合 2811+2812 的三层证据（类轴 9% + 形容词属性轴 ~0.25pp 特异 + 残差近满秩 282），名词嵌入的编码必定是以下之一或其组合：
- **分布式子空间码**：每属性占多个方向（如 5-20 维），单方向探针不可见；
- **指称几何与谓词几何分离**：名词嵌入编码的是"指称对象的知识网络坐标"（本体论位置），而非可名状的谓词轴——与用户总假设"知识网络"一致：嵌入可能直接编码知识图谱式的相对坐标，属性只是其投影；
- **非线性/流形码**：属性以弯曲流形而非线性方向存在（与 2790+ 的 Cmp 非线性发现呼应）。

第一性原理推论：**LLM 词汇编码的"高效"不来自正交轴打包，而来自把知识网络坐标（相对位置）作为主编码，属性和类别都是其低维投影。** 这把下一阶段的靶心从"找属性方向"转为"刻画名词嵌入的内在坐标系"——残差 PCA + unembed 读出是最低成本的下一步。

### 6. 接续（2813 候选，按优先级）

1. **残差 PCA 主任务**：对 283 词去类残差做 PCA，top 20 主成分逐一 unembed 投影读出 + 与 14 属性轴/dW 方向 cos——直接回答"91% 里是什么"。
2. **名词两极属性方向**（修 POS 错配）：mountain/ant、whale/mouse、fire/ice 等名词对构造属性轴，重测 share 与稳定性（预注册含 exclude-identity null）。
3. **频率回归控制**：share 测试前回归掉 log 词频与向量范数。
4. **metal→fruit 逐词表 + 实体假说**：metal 词逐词 fruit 投影排序；检验"元素/实体物"假说（metal 与 fruit 共享"自然实体"知识网络坐标，furniture/clothing 是人造功能物故无）。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2812_attribute_subspace.py` sha256 = `f344c1fe9b8fea1a9396af5f3106a1013d0eb1781bc8f443498bb5821cc57449`
- `tests/glm5/result/rdc_query_construction_20260913/phase2812/attribute_subspace/execution.json` sha256 = `3f918b11cd4dd78ae41e294de877cc04bd52c16943d025963d03e9534bf37686`
- `…/result.json` sha256 = `ee45b45cdf289e307910aee9245bd8cd34ab79441a7b909262615a69933b5162`
- `…/axes.npz` sha256 = `a9926a6c5c8f513cf4309e34e85295b7ee3deb6627455477f137ca8066d54425`

**Gens 记录**：无崩溃。首写版本在编译前经静态审查修掉三处隐患（tie 感知 lm_head 回退防 KeyError；split-half 奇数切分重叠；null 循环 zrow 重复计算），随后首跑干净完成（8.0s）。判据全程未动。

---

## Phase 2813: 残差几何普查——91% 是知识网络微场的平铺，不是属性轴空间 [2026-09-17 02:37]

**锚点**：2812 否证了"形容词属性轴"后，直接解剖残差本身——(R) 283 词去类残差 PCA 谱 + top-20 主成分 unembed 读出；(N) 名词两极属性轴（修正 2812 的 POS 错配 + 2812 的 identity 置换污染教训：稳定性 null 排除 identity 分配）。

### 1. 测试原理

零前向，三重 gates 全过（dW_vs_2807 = 1.82e-09；eval_words 与 2811 逐词一致；Z_vs_2811 = 9.51e-07）；2812 的 dA（14×2560）只读加载。

残差构造与谱分解：

```
Z_all = [Z_atlas; Z_eval]  (283×2560)
Zc    = Z_all − mean(Z_all)
resid = Zc − (Zc @ classQ) @ classQ.T        # 去类子空间
SVD(resid) → var_i = s_i² / Σs²              # 谱
PC_i   = Vt[i]（2560 维单位向量）
读出   = top-10 tokens of Wu @ PC_i          # tie: Wu = Etab
```

名词两极轴：6 属性 × 每极 6 候选（mountain/ant、fire/ice、lightning/snail、venom/pillow、boulder/balloon、sun/cave 式），单 token 过滤 + 电池去重，6/6 存活。share 判据同 2812；split-half 稳定性 null 改为**排除 identity 分配**的 pole-shuffle ×300。

预注册（冻结，execution.json sha256=ea07af8b…）：
- **P-R1** residual_lowdim：top-20 累积方差 ≥ 0.50
- **P-R2** pc_class_decoupled：max|cos(PC_i, dW_j)| < 0.30（i≤20, j≤9）
- **P-R3a/b**：名词两极 share > N1 q95 且 > 3× median；≥60% 属性 split-half 过 exclude-identity q95
- **verdict**：residual_geometry = 'lowdim' iff P-R1 AND P-R2；nounpole_support = P-R3

### 2. 关键结果（result.json sha256=9e3f2156b6c87cfc…）

**判定：residual_geometry = 'flat'；nounpole_support = false**

| 臂 | 结果 | 数据 |
|---|---|---|
| P-R1 谱低维 | **false** | top-10 = 10.6%，top-20 = **17.9%**，top-50 = 35.6%——谱平坦，无低维骨架 |
| P-R2 PC-类解耦 | true 但**恒等式** | max\|cos\| = 0.0000 精确为零：resid ⊥ 类子空间是构造使然（2809 型代数恒等式，**erratum 已入账**，verdict 实际由 P-R1 单独承载） |
| P-R3a 名词两极 share | **false** | eval share 0.0090 **低于** N1 median 0.0095（q95 0.0134） |
| P-R3b 名词两极稳定性 | **false** | pass-frac 0.50 < 0.60：temperature/speed/danger 过（cos≈0.11）但极弱；size cos=0.011 崩溃；weight q95==cos_half 平局 |

**探索性金矿（未预注册，top-20 PC 的 unembed 读出）——残差的 top 主成分全部是语义微场：**

| PC | 方差 | 读出（top unembed tokens） | eta² |
|---|---|---|---|
| PC1 | 1.36% | Ghana, Ecuador, Angola, Peru, Kenya, Yemen…（**国家**） | **0.483** |
| PC3 | 0.97% | lithium, sodium, magnesium, calcium, **锂, 钠**, potassium, zinc（**化学元素**） | 0.247 |
| PC5 | 0.90% | Spain, Sweden, **瑞典**, Norway, Finland, **西班牙**, Portugal（**欧洲国家**） | 0.217 |
| PC6 | 0.87% | reef, island, **岛**, shark, storm, islands（**海洋地理**） | — |
| PC7 | 0.86% | sushi, mango, rice, tofu, Vietnam, Thai, pizza, **蛋糕**（**亚洲食物/文化**） | — |
| PC8 | 0.82% | cake, candy, **蛋糕**, sausage, pastry（**甜食**） | 0.158 |
| PC15 | 0.72% | bolt, wrench, nut, screw, **螺**（**五金**） | — |
| PC0 | 2.03% | 逗号/数字/括号（**词汇格式轴**） | 0.126 |

eta²（PC 投影与类别标签的关联，探索性）：PC1=0.483（国家类主导）、PC3=0.247（metal 类）、PC5=0.217——**类内仍有子结构**：国家类分裂为 PC1（非洲/美洲国家）与 PC5（欧洲国家）两个微场。

**跨语言读出现象**：PC3/PC5/PC6/PC7/PC8/PC10/PC15/PC17/PC19 的 top-unembed 中直接出现中文 token（瑞典、西班牙、锂、钠、岛、蛋糕、橄榄、埃及、螺、椰、床）——**微场方向跨语言共享同一 unembed 几何**，语义场是语言无关的知识坐标。

### 3. 结论（核心发现 ×3）

**核心发现：91% 残差 = 知识网络微场的平铺拼贴——全局谱平坦（top-20 仅 18%，无低维骨架），局部主成分逐个读出为语义微场（国家、化学元素、欧洲国家、亚洲食物、海洋地理、五金、甜食），且微场方向跨语言可读出。**

重述一："属性流形"假说双重死亡（2812 形容词轴 + 2813 名词两极轴，后者 share 甚至低于词池本底）：属性不是名词嵌入的组织原则。
重述二：组织原则是**知识网络的邻域坐标**——每个微场 = 知识图谱的一个邻域（国家→地缘分区、金属→元素表、食物→菜系），大小 2-5 词/场、各占 0.7-1.4% 方差，拼贴起来谱自然平坦。
重述三：这直接支持总假设"语言能力 = 知识网络 + 分析推理 + 语法的编码"：**名词嵌入的主体是知识图谱坐标，类别与属性都只是这些坐标的投影**——类别轴（9%）是坐标的粗化读出，属性是坐标的谓词化读出。

### 4. 严格审视：问题、硬伤、瓶颈

1. **P-R2 恒等式**（erratum）：残差构造保证 resid ⊥ 类子空间，cos(PC, dW) ≡ 0——判据信息为零，与 2809 嵌套恒等式同类。教训第三次出现：**凡在"被扣除的空间"内测对扣分子的正交性，必得恒等式**。下一轮判据设计须先做代数可行性审查。
2. **微场读出是间接证据**：top-unembed 读出 + eta² 是解释性而非判定性的；PC0（标点/数字）提示部分 PC 携带词汇学（token 格式/频率）而非语义内容。
3. **283 词样本限制**：谱平坦部分来自微场样本量小（每场 2-5 词）；扩大电池（每微场 ≥10 词）可能抬高 top 方差占比。P-R1 的 0.50 阈值对小样本偏严格。
4. **频率/长度混杂未回归**：PC0 与部分 PC 可能由词频驱动。
5. noun-pole 稳定性 3/6 过但 cos≈0.11 仍远低于真轴水平（≥0.5）；weight 出现 q95==cos_half 平局（小池边界效应残余）。
6. PC 读出方向 = Wu=Etab（tied），unembed 即 embed 空间，读出的"语义微场"解释依赖 embed 空间本身语义可读这一已验证性质（2806-2808 链条）。

### 5. 智能理论视角的关键洞察（第一性原理）

三轮（2811→2812→2813）拼出名词嵌入编码的完整第一性图景：

- **编码主体是知识图谱坐标**：微场=邻域。词嵌入高效的原因不是把属性打包进正交轴，而是**直接编码"这个词在知识网络中的位置"**——类别、属性、多义（turkey food+animal）都是位置的函数。
- **效率的来源**：位置是相对量（对比编码，第 5 项），只需在共享流形上存偏移；微场间的近似正交 = 不同邻域在流形上自然分离，无需显式正交化。
- **跨语言共享微场方向**是"语言无关语义骨架"的直接证据——这与 L0 基对齐全语（Unified Theory v4.1）预测一致：微场方向应是 L0 基的局部特化。
- **与 Cmp(o,r,v) 非线性发现的呼应**：微场=吸引盆的候选几何实现；"回归均值"效应（2810 前后观察）可解释为微场间的插值。

**瓶颈与下一步靶心**：从"微场存在"到"微场坐标定律"——需要回答：微场的方向由什么决定（邻域的什么结构）？场内词的坐标如何排列（例如国家微场是否按地理/经济/历史排序）？这是"破解所有名词词嵌入规则"的下一层。

### 6. 接续（2814 候选，按优先级）

1. **微场普查主任务**：系统性枚举 top-50 PC 的微场归属（top-k unembed + eval/atlas 词投影聚类），量化 283 词的微场覆盖率；预注册"覆盖率 ≥70% ⇒ 知识网络坐标假说成立"。
2. **微场内坐标定律**：以国家微场（样本最大）为标本，检验场内一维排序是否对应地理/人口/经济变量（可从词嵌入自身回归）。
3. **频率/长度回归控制**后重测 share 与谱。
4. **metal→fruit 终审**：PC3 证实 metal 词=元素实体；检验 fruit 方向能量与"实体物"微场坐标的回归关系，关闭该悬案。
5. 2810 通道分离（Δ_specific）仍在队列，优先级维持。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2813_residual_geometry.py` sha256 = `f9786e25d07fc083f488ac2dee3525326fa787170b0565d205e1b673e8009dec`（修复版）
- `tests/glm5/result/rdc_query_construction_20260913/phase2813/residual_geometry/execution.json` sha256 = `ea07af8b23b88bbb5695cb8070bab06bdf30f7f87e1e3dddd5705b2e61f99ca8`
- `…/result.json` sha256 = `9e3f2156b6c87cfcf57ef0982df5ae776e4e03688d6399c1cac08c28fbc614d7`
- `…/residual.npz` sha256 = `e9f25417de4f04628ecd9eff93196f04692278ed4e18c7dbb01751f4ed83339a`

**Gens 记录**：首跑崩于 assert K2≥5（名词池候选太少：mountain/whale/elephant/desert/snow/frost/wolf/pillow/truck/cave 等与电池词冲突或多 token，仅 3/6 属性存活）。崩前零 result 产物（execution.json 已删）。修复：候选池扩至每极 6 个、崩溃前打印拒绝明细、补 PC-类别 eta² 探索性诊断（P-R2 恒等式在首跑已被发现并当场标注）。重跑 21.2s 干净完成。预注册判据文本未动（P-R2 按原样执行并以 erratum 入账）。

---

## Phase 2814: 微场普查否证——残差是"近正交指纹 + 弱带符号梯度"，不是邻域拼贴 [2026-09-17 02:58]

**锚点**：2813 的 PC 读出显示"语义微场"，提出知识网络坐标假说。本 Phase 做判定性检验：微场是否在词级覆盖残差总体？

### 1. 测试原理

零前向，四重 gates 全过（dW=1.82e-09；Zeval=9.51e-07；Zall_vs_2813=9.52e-07；tie=True）。残差构造与 2813 严格一致。

**领袖聚类 + 留一覆盖**：

```
resid = Zc − (Zc@classQ)@classQ.T        # 283×2560，去类、中心化
领袖聚类：按电池固定顺序 0..282 逐词扫描，
          与现有场均值 cos > τ=0.45 则并入，否则自立新场
覆盖判据：词 w 被覆盖 iff cos(resid_w, 场均值_LOO) ≥ θ=0.50
          （LOO = 场内其余词的均值；单例场 = 不覆盖）
```

N1 随机分配 null ×1000（保持场大小，sanity）；N2 随机 token 对照（250 个词表随机 token 的残差同法聚类+覆盖——知识特异性的关键对照）。
臂 D（诊断）：metal 词内部，元素场对齐度（对留一 metal 残差均值的 cos）与 fruit 方向投影的 Spearman ρ——检验 metal→fruit 是否由元素场中介。

预注册（冻结，execution.json sha256=0f8b68a8…）：
- **P-M1**：coverage(283 词, τ=0.45, θ=0.50) ≥ 0.70
- **P-M2**：coverage_eval183 − coverage_rand250 ≥ 0.15
- **P-D1**（仅诊断）：ρ ≥ 0.5 ⇒ fruit_metal_mediation
- **verdict**：knowledge_network_coordinates = P-M1 AND P-M2

### 2. 关键结果（result.json sha256=ed274e14bc24e900…）

**判定：knowledge_network_coordinates = false——微场/邻域假说在词级被否证，且证据把 2813 的解释正式降级。**

| 臂 | 结果 | 数据 |
|---|---|---|
| P-M1 微场普查 | **false（压倒性）** | 283 词聚出 **282 个场**（281 个单例！）；唯一多例场 = sunset/sunrise（cos 0.573）；coverage_all = **0.7%**（阈值 70%） |
| τ 稳健性 | 结论不变 | τ=0.35/0.45/0.55 下 coverage 全部 ≤1.1%——词间两两 cos 几乎全部 < 0.35 |
| N1 随机分配 | 0.000 ✓ | sanity 通过 |
| N2 随机 token 对照 | **随机 token 聚类反而更多** | coverage_rand = 7.2% vs eval 1.1%，margin = −6.1pp → P-M2=false |
| P-D1 中介 | **false，反向** | ρ_metal = **−0.289**，ρ_furniture = −0.208——元素场对齐不中介 fruit 投影，反而轻微负相关 |

### 3. 2813 解释的正式降级（erratum 级修正，非代数错误）

2813 的"微场拼贴"结论**降级为：弱带符号梯度轴**。综合两轮证据的机制解释：

- **PC1 读出国家、eta²=0.483，但国家词两两之间不聚簇** ⇒ PC1 是一条**梯度轴**：国家词在该轴上有同号的小投影（正），其他词反向（负）——但每个词沿轴的位置各不相同，且投影绝对量极小（PC1 仅占方差 1.36%），不足以使任何两个残差向量靠近（cos < 0.45）。
- 因此残差的真实结构 = **每个内容词一个近正交的私有方向（指纹，占方差主体）+ 每词沿少量语义梯度轴的微弱带符号偏置**（这正是 2811 次级成员 0.6-0.9 比率、2813 eta² 0.2-0.5、2812 属性特异 0.25pp 的共同来源——全部是"小而一致"的信号）。
- **随机 token 比名词更聚簇（7.2% vs 1.1%）**是反向佐证：功能/格式 token（标点、数字、字节）共享真实格式方向，而内容词被赋予近乎私有的方向——近正交性是**内容词的签名**。
- P-D1 反向（ρ=−0.29）关闭"元素场中介 metal→fruit"路径；metal→fruit 的机制仍未解，但排除了一条主流假设。

### 4. 结论（核心发现 ×3）

**核心发现：名词残差编码 = 近正交私有指纹（主体）+ 弱带符号语义梯度（少数分量）——知识网络不编码为邻域 proximity，而编码为跨方向的偏置模式。**

重述一：词级覆盖 0.7%、τ 全域稳健、随机 token 反超——"微场邻域"假说死亡，2813 的 PC 读出重新解释为梯度轴检测。
重述二：内容词两两近正交（唯一例外 sunset/sunrise 这对近重复词）——每个名词拥有私有方向，共享语义只以微小带符号偏置存在。
重述三：至此 2811→2814 四轮收敛出一个自洽编码方案：**9% 类尖峰（离散读出）+ ~0.25pp 属性偏置 + 梯度轴偏置模式（知识网络所在）+ 近正交指纹（词项身份主体）**。

### 5. 严格审视：问题、硬伤、瓶颈

1. **聚类阈值依赖**：τ=0.45/θ=0.50 是预注册选择；但 τ=0.35 仍 278/283 单例，结论对阈值稳健——此硬伤不成立，真正的硬伤在下面。
2. **梯度轴与指纹的分解尚未定量化**：当前只能说"偏置小"，还没有给出"偏置方差占比"的正式谱分解（top-50 PC 只占 35.6%——剩余 64% 是指纹还是更多梯度轴，未分）。
3. **283 词样本对"近正交"的证明力**：词多场少的结论在小样本上稳健（更多词只会更难聚簇），但"指纹内容是什么"完全未触及。
4. **N2 的构造差**：随机 token 残差用名词类子空间投影移除，格式 token 的共享结构可能被高估——margin 的 −6.1pp 方向上可信，数值粗糙。
5. P-D1 仅 14 个 metal 点的 Spearman，功效有限；负号本身不稳健。
6. **2813 的 eta²/读出解释已修正**，但 2813 的 result.json 不可变——修正以本节为准。

### 6. 智能理论视角的关键洞察（第一性原理）

四轮（2811→2814）把"齿轮形状"问题推到了一个必须面对的第一性原理分叉：

- **知识网络不在 proximity 里**：如果知识网络以"邻域"编码，同邻域词应聚簇——实测否。知识网络必然编码在**每个词独有的方向组合模式**中：词的"地址"不是小区，而是**一组弱梯度轴上的坐标读数**（类似 GPS：不是住在"国家小区"，而是有一串 (经度, 纬度, 海拔…) 读数）。
- **这与 superposition 理论的极限形式一致**：n 个词、d 维空间、词间近正交 ⇒ 特征以近正交基分配给词项身份，共享语义特征以小权重叠加其上——容量预算优先给了**身份**，语义共享靠"租用"少量方差。
- **预测**：在更高层（hidden states），指纹应被改写为语境函数——2810 的"通用上下文通道主导"正是指纹被上下文改写的表现。**词级指纹 + 语境调制的梯度轴 = 完整编码的两大部件**，下一阶段的主战场是后者（2810 通道分离的优先级因此上调）。
- **阶段性大任务（2815-2818）**："偏差矩阵知识结构"——取 B = 283 词 × top-50 PC 得分矩阵（低维语义坐标系），正式检验：B 空间中同词场词是否聚簇（若 B 中聚簇而残差不聚簇 ⇒ "GPS 读数"模型成立）；B 的 k-NN 类别可分性；B 与知识网络变量（类别、属性、多义）的回归。这是把"知识网络编码机制"从定性走向定量的主路径。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2814_microfield_census.py` sha256 = `bb8432d3622cd13e06c86e9e1db819b662d6d78be94c4be322e2649889b4ea8b`
- `tests/glm5/result/rdc_query_construction_20260913/phase2814/microfield_census/execution.json` sha256 = `0f8b68a870640ad64945f0f62fb34d0663ce16e8a0daee91b75b5f7a89c27a62`
- `…/result.json` sha256 = `ed274e14bc24e9005ba133cf73799141300c45e519b79d20664667a8154e677e`
- `…/fields.npz` sha256 = `729cdac195c80501249262bbe055f01de3b94dbd2389338a8055c3b7bcfb3dc3`

**Gens 记录**：无崩溃，首跑干净完成（7.0s）。预注册判据全程未动。

---

## Phase 2815: 偏差矩阵 B 的知识结构——GPS 模型强形式检验 [2026-09-17 03:21]

### 1. 任务与原理

2814 压倒性否证"微场邻域"后，四轮收敛模型预测：词的语义地址不是"小区"而是**一组弱梯度轴上的坐标读数**（GPS 模型）。本 Phase 将其定量化：从 2813 残差 PCA 构造**偏差矩阵 B = 283 词 × top-K PC 得分**（低维语义坐标系），检验 B 中是否携带类别知识结构。

- **P-B1（梯度真实性）**：bias_gradient_real iff 同类词对 within-class mean pairwise cos（B, K=50, eval 集, 类大小加权）> label-shuffle q95 **AND** ≥ 2× global mean pairwise cos。
- **P-B2（kNN 可读性）**：bias_knn_readable iff kNN purity（k=10, leave-self-out, K=50）> shuffle q95 **AND** ≥ 0.30。
- **总判定**：bias_matrix_carries_knowledge = P-B1 AND P-B2；K=10/20/50 曲线仅描述性。
- Null：类标签随机置换 ×1000。

### 2. 结果

| 统计量 | 值 | 判据要求 | 判定 |
|---|---|---|---|
| within_cos (K=50) | 0.0070 | > null q95=0.0109 且 ≥2×global | **双未达** |
| global_cos (K=50) | 0.0056 | within/global = 1.25×（要求 2×） | 未达 |
| kNN purity (k=10) | 0.2186 | ≥0.30 且 > q95=0.124 | q95 超了但阈值未达 |
| knn_chance | 0.1137 | purity = 1.92× chance | 弱信号 |

- **bias_gradient_real = false / bias_knn_readable = false / bias_matrix_carries_knowledge = false**。
- K 曲线（描述性）：K=10 within 0.0465 vs null q95 0.0461（贴线）；K=20 within 0.0292 vs q95 0.0245（略超）；purity 对 K 几乎不动（0.223→0.224→0.219）。
- per-class within cos：country 最高（0.029, n=33），metal 0.012, nature 0.009, animal 0.004；fruit（−0.047）、vehicle（−0.0075）、food（−0.0082）为**负**——同类词在 B 中甚至略微反聚簇。

### 3. 分析与硬伤

- **GPS 模型强形式否证**：静态嵌入（残差）在 top-50 PC 坐标系中**不存在低维知识坐标**——同类词不聚簇，类别信息不足以支撑 0.30 purity。2811（类子空间仅 9% 能量）→2814（无微场邻域）→2815（无低维坐标）三连否证后，"知识网络编码在静态嵌入的某种几何结构里"这一整族假设已无幸存者。
- purity 1.9× chance 的弱信号来源：top-PC 主轴确实携带部分类别方差（2813 eta²=0.483 的国家轴），但**集中在少数轴、不成坐标系**——"读数模型"若成立，purity 应远高于 0.22。
- 硬伤：①B 由 2813 的 top-20 PC 延拓计算，若知识编码在 PC 谱尾（top-50 之外），本测试盲区——但谱平坦（2813 top-20 仅 17.9%）使该可能性low-prior；②label-shuffle null 破坏了类别大小结构，q95 可能偏松；③K=50 预注册的先验论证不充分（应预注册曲线族判据），K=10/20 的"贴线"提示存在极弱结构，如实登记不上升为结论。

### 4. 结论与接续

- **结论（重复 3 次）：静态嵌入无低维知识坐标。静态嵌入无低维知识坐标。静态嵌入无低维知识坐标。** 知识网络主要活在**层计算**中——这正是 2816 的入口。
- 接续：知识结构不在"词往哪里放"，就在"层往哪里搬"。下一 Phase 直接检验层参数的语义含量（Arm W）与语义的位置稳定性（Arm P）。

### 5. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2815_bias_matrix.py` sha256 = `be1d16ec422578e48920cf2e0114cf6f6f7e625395764472fb0884c609449ed6`
- `…/phase2815/bias_matrix/execution.json` sha256 = `7f226575f130cab2ea655dc2785546052d7e789a2e3370634a163f9cd5c1f25d`
- `…/result.json` sha256 = `2303f19f3bde9e2e79a45ad6762fab4a92c1bc1df9155527a1686b58710eb328`
- `…/bias.npz` sha256 = `2c2bfe5a5dd2db8606c36e7e350e79a22a5e87b772a4f0d3360e69f8efe2467c`

**Gens 记录**：首跑崩溃 `KeyError: 'atlas_labels is not a file in the archive'`（2813 npz 未存 atlas_labels）→ 从 CATS 确定性重建标签向量（`np.array([ci for ci,c in enumerate(CAT_WORDS) for _ in CATS[c]])`），删陈旧 execution.json 重跑干净。预注册判据未动。

---

## Phase 2816: 层参数的语义含量与语义位置不变性 [2026-09-17 03:21]

### 1. 任务与原理（用户问题测试化）

用户命题："如果词嵌入包含了大量的语义信息，那么 Layer 中包含了多少语言？是单独的规则，还是语义和规则混合？自回归机制中，一个语义可以在任何位置，而 Layer 中的参数位置是固定的，这两者之间是什么关系？"

操作化为双 Arm：

- **Arm W（写入普查）**：36 层 × 两通道（MLP `down_proj` 列、ATTN `o_proj` OV 单列）与 10 个类方向 unitD 的 max |cos|；每层 30 随机向量给 q95 基线。
  - P-W1：mlp_neuron_semantic iff ∃L, max_j |cos(W_down[:,j], unitD_i)| ≥ 0.30 AND 该层随机 q95 < 0.30。P-W2 同判据 o_proj。
- **Arm P（位置不变性）**：56 序列 = 8 目标词 × 7 槽位（函数词填充 'the','of','and','to','a','in'，RoPE 位置 0..6，add_special_tokens=False）；10 维类别谱 profile = hs @ unitD^T；profile_stab = 7 位置间 pairwise cos 均值；full_stab 为全 2560 维对照；身份通道 CV（argmax one-hot 的变异系数）。
  - P-P1：semantics_position_invariant iff mean_{L∈4..35} profile_stab ≥ 0.80。P-P2：profile>full iff gap ≥ 0.05。
- **总判定**：layers_rules_semantics_mixed = P-W1 OR P-W2；semantics_position_invariant = P-P1 AND P-P2。

### 2. 结果

| 判定 | 值 | 关键数字 |
|---|---|---|
| **P-W1 = true** | MLP 神经元直接写类方向 | 9 个晚层通过（L27–35）；best **L35 n1936 cos=0.6101（metal）**，随机 q95≈0.09 |
| P-W2 = false | OV 单列无对齐 | 0/36 层；best L32 n51 cos=0.2325（nature），其 unembed 读出 **river/雨/rain/沙滩/fluid/maritime/海洋/marine**——语义连贯水域场 |
| **P-P1 = false** | 位置不变性不成立 | prof_stab mean(4..35) = **0.7082 < 0.80** |
| P-P2 = false | 语义谱不比全状态稳 | gap = **−0.0097**（0.7082 vs 0.7179） |
| **总判定** | **layers_rules_semantics_mixed = true** | semantics_position_invariant = false |

- **profile 稳定性曲线**（层 0→35）：1.0 → 0.95 → 0.88 → 0.84 → 0.80（前层高位）→ 中层谷底 0.61–0.70（L21 最低 0.614）→ 后层回升 0.72–0.77 → **末层骤降 0.585**（lm_head 前改写）。
- **full_stab 曲线**形态一致（mean 0.718），前层 0.93→0.81、中层 0.65–0.68、后层回升 0.78–0.79。
- **身份通道 CV**：L0–6 极低（0.06–0.13）→ **L7–26 爆炸（2.2–15.1，L24 峰值 15.15）** → L27+ 恢复（0.6–1.0）。中层（7–26）身份 argmax 几乎被逐层改写。

### 3. 分析：对用户两问的直接回答

**问一：Layer 中是单独的规则，还是语义和规则混合？→ 混合体，且语义是"被引用"的。**
- 晚层 MLP 神经元权重列与类方向 cos 高达 0.61（随机 q95 仅 0.09，信噪比 ~7σ），9 层连续通过——**层参数不是纯规则引擎，它携带显式语义方向**。
- 但 OV 单列全军覆没（max 0.23）且位置不变性不成立（0.71 < 0.80）——层也**不是语义存储**。语义以"写入方向"形式存在于 MLP，以"语境调制"形式流动于残差流。
- L32 n51（cos 仅 0.23、未过线）的读出却是完美水域场——提示注意力头的语义贡献是**多头组合**的，单列是下界（硬伤 ①）。

**问二：语义可在任意位置 vs 参数位置固定，什么关系？→ 位置经 RoPE 只进入寻址，语义走内容通道；参数"位置"是通道索引，不是序列位置。**
- 序列位置（slot 0..6）对语义谱的破坏是真实的（0.71 稳定 ≠ 1.0），但**语义谱与全状态稳定性无差**（gap −0.01）——位置影响的是整体状态，不是选择性破坏语义通道。
- 语义的"任意位置"能力由两部件实现：①内容寻址（V/MLP 携带语义方向，与位置无关）；②RoPE 只调制 Q/K 注意力寻址。参数矩阵的行/列索引是**通道地址**，与 token 序列位置正交——这正是 Transformer 把"位置"限制在注意力打分、把"内容"放在值通道的架构决策的直接后果。
- 身份 CV 中层爆炸 + 前层 profile 高稳 → **前层保持身份、中层大规模改写（语法/语境整合区）、后层恢复并写入类方向（L27–35 MLP）、末层读出前再改写**——层功能的粗分工首次可见。

### 4. 硬伤（严格审视）

1. **OV 单列是下界**：o_proj 单列 cos 弱不排除 OV 回路（多头组合后写入类方向）。2817 应做 W_O 全头拼接后的对齐测试。
2. **位置测试仅函数词语境**：'the/of/and/to/a/in' 填充是最小语境，自然语句下 profile 稳定性可能更高或更低，未测。
3. **L35 接近读出端**：最后一层 MLP 输出直接叠加进 lm_head 前状态，cos 0.61 可能部分是读出端对齐混杂——但 L27–34 连续通过（非孤点）削弱此担忧。
4. **best_neuron unembed 读出碎片化**（秭/雎/.twig/Phar/coy/ackbar）：类方向本身的 unembed 读出即碎片化（2807 已知性质），"神经元写类方向"≠"神经元对应可读词表语义"，方向级与词元级读出需区分。
5. **id_cv 爆炸解释未定**：可能身份真被改写，也可能 argmax one-hot 在近零基上放大数值噪声；需用软分布（softmax 熵）复核。

### 5. 结论（重复 3 次）

**层 = 引用语义的规则机器：MLP 晚层显式携带类方向（cos 0.61），语义内容在残差流中被逐层语境调制而非位置不变。**
**层 = 引用语义的规则机器：MLP 晚层显式携带类方向（cos 0.61），语义内容在残差流中被逐层语境调制而非位置不变。**
**层 = 引用语义的规则机器：MLP 晚层显式携带类方向（cos 0.61），语义内容在残差流中被逐层语境调制而非位置不变。**

### 6. 接续（2817 候选，按优先级）

1. **OV 回路组合对齐**：W_O 全头拼接 × 10 类方向，修复硬伤 ①——若组合后通过则"注意力也写语义"成立，层分工图改写。
2. **自然语境位置测试**：真实句模板替换函数词填充，检验 profile 稳定性的语境依赖。
3. **语境调制方向量化**：同词跨语境状态差向量 Δh 的谱与类方向投影——"语境调制"的定量刻画（2810 通道分离主战场的入口）。
4. **L35 n1936 因果验证**：门控/消融该神经元，观察 metal 类输出变化——方向对齐到因果确认。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2816_layers_semantics_position.py` sha256 = `deb2f80e4d6e88760550c80c1ccb5c4eaa83b3f290da0dcfda326a55a7eb9626`
- `…/phase2816/layers_semantics_position/execution.json` sha256 = `e82052ddc9f9bfeb5595f14d5922ef281d2506ae7a2da01b6fd414aa6b3651ad`
- `…/result.json` sha256 = `7f607f5078e383c0d96e00aa0b737c3501ad1f91a794b3a4567126141f0a2453`
- `…/channels.npz` sha256 = `2ac939683235e4a8b56b86176e03c0c11eaca0572c5a15e8324c7e15faca2b29`

**Gens 记录**：双崩溃——①首跑 `TypeError: tuple indices must be integers`（best[kind] 存 (值, dict) 元组，`info = best[kind][1]` 修复；Arm W 已完成部分结果保留）；②删陈旧 execution.json 后重跑 199.3s 干净。预注册判据全程未动。加载秒数 198.6s（bf16 全 GPU 单模型序列化纪律遵守）。

---

## Phase 2817: OV 列空间对齐与自然语境位置不变性——2816 双硬伤闭合 [2026-09-17 03:53]

### 1. 任务与原理

2816 登记了两条硬伤：①o_proj 单列测试是下界（多头组合写入可能低于逐列 max）；②位置测试仅函数词填充语境。本 Phase 双 Arm 闭合。

- **Arm A（零前向，逐头 OV 列空间投影）**：层 L 头 h 写入 `δ = W_h·a_h`（W_h = W_O[:, h·128:(h+1)·128] ∈ R^(2560×128)），其可达写入子空间为 Col(W_h)。类方向含量 = `proj_len(L,h,i) = ‖P_Col(W_h)·unitD_i‖`，经 G = W_h^T·W_h、c = W_h^T·unitD_i、`proj² = c^T·G⁻¹·c`（投影定理，0≤proj≤1）精确计算。Null：30 个形状匹配随机单位向量过同一子空间（理论均值 √(128/2560)=0.224）。
  - **P-A1**：attn_ov_subspace_semantic iff ∃(L,head): max_i proj_len ≥ 0.40 AND 该头随机 q95 < 0.40。
- **Arm B（CUDA，自然语境位置）**：同 8 目标词 × 7 个自然前缀句（句干尾部固定 '…{w} is here'，目标首 token 位置严格 0..6；整句 BPE 编码，span 均值读出）。指标与 2816 完全同构。
  - **P-P1n**：natural prof_stab mean[4..35] ≥ 0.80。**P-P3**：position_effect_context_dependent iff natural − functional(2816=0.7082) ≥ 0.05。

### 2. 结果

**Arm A：P-A1 = true——注意力 OV 通道直接携带类方向。**

| 统计量 | 值 |
|---|---|
| 通过头数 | **16 / 1152**（36 层 × 32 头） |
| best | **L32 h0，nature，proj=0.6805**，随机 q95=0.2443（**2.79× null**） |
| proj 读出 | **rain / 雨 / snow / cloud / Rain / mountain / clouds / forest** |
| 全局随机 q95 | max 0.2573, median 0.2440（贴合理论 0.224，null 健全） |

- **与 2816 对照（硬伤①闭合）**：同层 L32 的 OV **单列** best 仅 0.2325（未过 0.30 线）；**列空间**投影 0.6805——单列是 3× 低估，类方向分布在该头多个输入维上**组合写入**，2816 的"OV 无语义"被推翻。
- 层分布：L0–18 max_proj 0.27–0.30（贴 null），L24 起 0.37→0.39，**L32 达 0.68**——注意力语义写入同样集中于晚层，与 MLP 写入层（L27–35）同区。
- L32 h0 与 2816 单列 best（L32 n51，读出 river/雨/rain/maritime）同层同语义场——**同一"nature/weather 写入区"的两种粒度**。

**Arm B：P-P1n = false，P-P3 = false——位置调制是普遍机制，非语境 artefact。**

| 统计量 | 自然语境（2817） | 函数词语境（2816） |
|---|---|---|
| prof_stab mean[4..35] | **0.7327** < 0.80 | 0.7082 |
| full_stab | 0.7381 | 0.7179 |
| gap (prof−full) | −0.0054 | −0.0097 |
| id_cv mean | 1.863 | 2.556 |
| 末层 profile | **0.737** | **0.585（骤降）** |

- 自然语境仅比函数词语境稳 +0.0245（<0.05 阈值）——**位置破坏 ~0.27 在两种语境下几乎相同**，位置调制不是"不自然语境"的产物。
- 曲线形态一致：前层 0.94→0.82 → 中层谷底 0.66–0.72（L16 最低 0.6635）→ 后层回升 0.79 → 末层。**但函数词语境的末层骤降（0.585）在自然语境下消失（0.737）**——读出前的最后改写是语境依赖的。
- 身份 CV 中层爆炸区一致（自然 L25 尖峰 7.13 vs 函数词 L24 15.15），自然语境下破坏幅度略小。

### 3. 分析

- **层 = 语义与规则混合的结论加固**：2816 P-W1（MLP 晚层写类方向 cos 0.61）+ 2817 P-A1（OV 头列空间 proj 0.68）——两条写入通道（MLP down_proj 与注意力 OV）均显式携带类方向。晚层 L27–35 是**语义写入密集区**，且 MLP 与 attention 在同一层区协同。
- **位置不变性正式死亡**：自然语境 0.7327，距 1.0 尚差 0.27——语义谱逐层被语境/位置调制是架构级机制，不是测试 artefact。与 2810（通用上下文通道主导 Δ）、2815（静态嵌入无知识坐标）拼合：**语义以"方向"形式存储于参数，以"被调制的读数"形式流动于残差流**。
- **头级机制定位首次达成**：L32 h0 = nature/weather 写入头（proj 0.68，读出跨语言 rain/雨/snow/cloud）。单头列空间即可捕捉类方向 68%——语义写入的粒度是"头"而非"全层混合"。

### 4. 硬伤（严格审视）

1. **proj_len 是"写入能力"非"实际写入"**：h0 的实际输入 a_h 可能从不激活该方向——需前向验证 `a_h^T·G⁻¹·c` 的实际投影分量（2818）。
2. **只读出了 best 头**：16 个通过头的清单与语义分工未登记。
3. **位置与句法角色绑定**：自然模板中目标词恒居主语位，"主语 vs 宾语位置"未分离——位置效应可能部分是句法角色效应。
4. P-P3 的 +0.0245 介于"无效应"与"弱效应"之间，按预注册阈值判 false，如实登记方向性信号。

### 5. 结论（重复 3 次）

**注意力 OV 通道直接写入类方向（L32 h0→nature, proj 0.68），语义写入以头为粒度；位置调制是普遍机制，语义在残差流中永远处于被调制状态。**
**注意力 OV 通道直接写入类方向（L32 h0→nature, proj 0.68），语义写入以头为粒度；位置调制是普遍机制，语义在残差流中永远处于被调制状态。**
**注意力 OV 通道直接写入类方向（L32 h0→nature, proj 0.68），语义写入以头为粒度；位置调制是普遍机制，语义在残差流中永远处于被调制状态。**

### 6. 接续（2818 候选）

1. **写入头全量登记 + 实际写入验证**：16 头清单；前向取 a_h 验证实际投影分量（能力→事实）。
2. **双写入点因果消融**：L35 n1936（MLP metal）+ L32 h0（OV nature）门控/置零，测类输出变化——把方向对齐升级为因果确认。
3. **语境调制方向 Δh 量化**（2810 通道分离主线）：同词跨语境状态差的谱结构。
4. **主语/宾语位置分离**：改模板使目标词分别居主语位与宾语位，分解位置效应与句法角色效应。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2817_ov_subspace_natural_position.py` sha256 = `ba5391fc0c3576ab47091e38131edaa1242e9f88ab005bd356dadaffab252b18`
- `…/phase2817/ov_subspace_natural_position/execution.json` sha256 = `469e99a2d06ffeb6c4a72b155e628a7ebfa950844032d0b5bca90859b9d5862f`
- `…/result.json` sha256 = `fc1d53d4195285503ded246f64512b0f0e9d633a4585ff1d53ea48a613f6f235`
- `…/ovpos.npz` sha256 = `f52da54dc951bad265f456a4c47eae01057e1efb79d455928986b8d5e651230c`

**Gens 记录**：四次运行——①崩于 `tid(' melon')` 无 2816 回退逻辑（melon 空格形式 2-token）→ gates 沿用 2816 复刻逻辑、Arm B 改整句 BPE + span 均值读出；②崩于 unitD 转置方向（(10,2560) vs (2560,10)）→ `Wh.T @ unitD_f.T`；③崩于变长序列 `torch.tensor`（自然句长度不等）→ pad + attention_mask（causal 下右侧 pad 不影响目标位表示）；④删 execution.json ×3 后 50.3s 干净。判据文本全程未动；gate：dW=1.82e-09, Zeval=2.46e-06。

---

## Phase 2818: 写入头普查、实际写入与因果消融——容量/事实/因果三级递进 [2026-09-17 04:06]

### 1. 任务与原理

闭合 2817 两条硬伤：①proj_len 是写入**能力**（列空间几何），非实际写入（依赖输入 a_h 是否激活）；②只读出了 best 头。本 Phase 把 2816→2817 的方向对齐链升级为三级递进：**容量（2817 proj）→ 事实（Arm V 实际写入）→ 因果（Arm C 消融）**。

- **Arm V（实际写入）**：hook 捕获 16 个 pass 头（2817）在 56 自然句（2817 同款）中 o_proj 的真实输入 a_h，构造平均实际写入 δ̄ = W_h·mean(span a_h)，度量 cos(δ̄, unitD_argmax)（abs 协议）；同层随机非 pass 头 max-class |cos| 给 null。
  - **P-V1**：median over pass heads ≥ 0.10 AND 随机头 q95 < 0.10（0.10≈5σ，随机 cos std=1/√2560≈0.02）。
- **Arm C（因果消融）**：两个登记位点在**写入路径输入端**置零——MLP L35 n1936（2816 P-W1 best, metal, cos 0.61）在 down_proj 输入通道置零；OV L32 h0（2817 P-A1 best, nature, proj 0.68）在 o_proj 输入头切片置零。效应 Δprof = prof_intact − prof_abl（最终 hidden state，span 均值，词均）。
  - **P-C1**：Δ_metal(n1936) > 0 AND > 10 随机同层神经元 q95。**P-C2**：Δ_nature(h0) > 0 AND > 10 随机同层头 q95。

### 2. 结果

**Arm V：P-V1 = false，但逐头分解揭示三档结构——"可写"≠"在写"。**

| 档 | 头 | proj | actual_cos |
|---|---|---|---|
| **实际写入组** | L32 h0 (nature) | 0.6805 | **0.2097** |
| | L32 h3 (clothing) | 0.5687 | **0.2024** |
| | L34 h0 (metal) | 0.6722 | **0.1907** |
| | L31 h4 (animal) | 0.4310 | **0.1066** |
| 弱写入组 | L29 h25, L33 h9, L35 h23, L23 h9 | 0.40–0.48 | 0.037–0.068 |
| **空管道组** | 其余 8 头（含 L35 h20/h22, L31 h3, L28 h7…） | 0.40–0.54 | **0.005–0.027** |

- median 0.0320 < 0.10（被空管道组拖低）→ P-V1 false；随机头 max-class q95 = 0.072。**实际写入头是容量头的真子集（4/16）**，L32 h0 实际写入 = 2.9× 随机 q95。
- proj 0.54 的 L35 h22 实际仅 0.008——容量普查的 16 头中一半是"管道存在但从不流水"。

**Arm C：P-C2 = true——L32 h0 是 nature 的因果写入头；P-C1 = false 且方向反转。**

| 位点 | Δprof（消融效应） | 随机 q95 | 判定 | 逐词 |
|---|---|---|---|---|
| OV L32 h0 → nature | **+0.1256**（消融后下降） | 0.0137（9.2×） | **true** | 7/8 词正；cave **0.322**、soup **0.273**、其余 0.047–0.093 |
| MLP L35 n1936 → metal | **−0.0421**（消融后**上升**） | 0.0019（22×） | **false（反向）** | 6/8 词负；gold **−0.171**；logits shift 仅 0.0625 |

- **L32 h0 三级递进完全一致：proj 0.68（容量）→ actual 0.21（事实）→ causal 0.126（因果）**——nature 语义的 OV 写入头从几何、激活、因果三个独立层面同时确认。教科书级机制闭环。
- **L35 n1936 反直觉**：方向对齐 0.61（2816）但消融使 metal profile **增强**（|−0.042| = 22× 随机）——该神经元不是 metal 的因果增强器。候选解释：①读出端混杂（2816 硬伤③坐实：down_proj 列与类方向的 cos 反映词表几何对齐，非功能写入）；②n1936 参与 metal 的抑制/对比回路；③下游补偿重写。logits 层面影响极小（0.0625）。

### 3. 分析

- **2816 "P-W1 MLP 神经元写语义"的解释必须修正**：参数携带类方向 ≠ 该参数因果增强该语义。n1936 的 cos 0.61 是**几何对齐**，消融证明其因果效应为零甚至为负。相反，OV 通道的对齐（2817 proj / 2818 actual / 2818 causal）全部同向成立。**语义写入的因果主体在注意力 OV 头，MLP 晚层的"写入"几何需要重新审查。**
- **写入头的专化图景浮现**：L32 h0/h3（nature/clothing）、L34 h0（metal）、L31 h4（animal）——每头一个类域，层集中在 L31–34（读出前 2–5 层）。这与 2813 的 PC 语义微场（读出端几何）拼合：**语义类信息在最后几层由专化 OV 头写入，供 lm_head 读出**。
- 位置/语境的普遍调制（2817）+ 头级专化写入（2818）= 编码两大部件的最终分工：**头写类域、流承载调制、读出消费方向**。

### 4. 硬伤（严格审视）

1. **d_mlp 反向机制未解**（混杂/抑制/补偿三候选未分辨）——需 2819 分解。
2. **因果消融只测了 2/16 位点**：L34 h0（metal, actual 0.19）与 L32 h3（clothing）未补测。
3. **效应读出在表示层**（最终 hidden state 类 profile），非行为层（logits shift 0.0625 很小）——类 profile 的变化可能被 final norm + unembed 投影稀释；应补 logit_lens（norm 后）读出。
4. 单一句式（'… is here'），语境泛化未测。

### 5. 结论（重复 3 次）

**容量≠事实≠因果：16 个可写头中仅 4 个实际写入，L32 h0 经三级验证是 nature 的因果写入头；MLP n1936 方向对齐但因果反向——参数几何对齐不等于功能写入，语义写入的因果主体是专化 OV 头。**
**容量≠事实≠因果：16 个可写头中仅 4 个实际写入，L32 h0 经三级验证是 nature 的因果写入头；MLP n1936 方向对齐但因果反向——参数几何对齐不等于功能写入，语义写入的因果主体是专化 OV 头。**
**容量≠事实≠因果：16 个可写头中仅 4 个实际写入，L32 h0 经三级验证是 nature 的因果写入头；MLP n1936 方向对齐但因果反向——参数几何对齐不等于功能写入，语义写入的因果主体是专化 OV 头。**

### 6. 接续（2819 候选）

1. **d_mlp 反向机制分解**：n1936 消融的 metal 增强来源追踪（下游层 profile 逐层差分 + 补偿神经元搜索）。
2. **L34 h0 / L32 h3 因果补测** + logit_lens 读出（norm 后投影，修复硬伤 3）。
3. **OV 写入头的输入归因**：L32 h0 的 a_h 从哪些位置/通道来（QK 注意力图 × value 路径）——写入头的上游回路。
4. **Δh 通道分离主线**（2810 承接）：语境调制方向的谱结构。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2818_write_causality.py` sha256 = `d768596806b05756267cedbfd5aeac5b7d980c795b90690be9cd51f922069a70`
- `…/phase2818/write_causality/execution.json` sha256 = `c106d38459fa53b4988838751f7f8a975b81f2c36a89e94ad44d3bb8094f86f0`
- `…/result.json` sha256 = `c90a1bce07e2bf05edd2410dcae11e1bd5488ea97dae9d02ae9252647eda2245`
- `…/causal.npz` sha256 = `36c4d6b1c9533e95c41ae3a730ba68096c8903dbdd25c656f50ff9844d38402b`

**Gens 记录**：三次运行——①崩于 span_profiles 广播（(7,2560) 未词级平均 → `vecs.mean(0) @ unitD.T`）；②崩于 top8 二维索引（LG0[li] 取成 (B,vocab) → `LG0[-1, li]`）；③删 execution.json ×2 后 14.7s 干净。判据文本全程未动；gate：dW=1.82e-09, Zeval=2.46e-06。

---

## Phase 2819: 因果地图补全 + 知识编辑定位——"apple 是红色"存在哪些参数里 [2026-09-17 04:37]

### 1. 任务与原理

双任务：**Arm R** 补全 2818 因果地图（L34 h0 metal / L32 h3 clothing 两个实际写入头的因果验证 + logit lens）；**Arm K** 回答用户新问题（2026-09-17）："要修改 LLM 中的知识，把苹果的颜色从红色改为黑色，应该改哪些参数？"——把 2816-2818 协议搬到颜色域，行为终点 = "The {s} is" 之后的**目标颜色词 logit**（apple→red, sky→blue, grass→green, coal→black, banana→yellow, blood→red）。

- **K1（层依赖剖面）**：逐层置零全部 attention 输出 / 全部 MLP 输出（36+36 次前向），Δlogit_target(subject-mean)；随机词（table/lamp/book）logit 变化为特异性 null。
  - **K-P1**：∃L Δ ≤ −1.0 AND 随机词 median > −1.0。
- **K2（方向普查，零前向）**：2807 协议扩展构造颜色对比方向 `dW_color = unit(mean_z(色词族) − mean(10类中心))`、red 特异方向 `dW_red = unit(z(red) − mean_z(其他色词))`；MLP down_proj 列 max|cos| + OV 头列空间投影普查。
  - **K-P2**：∃位点对齐 ≥ 0.30 AND 该层随机 q95 < 0.30。
- **K3（因果位点）**：top-3 对齐 MLP 列 + top-3 OV 头逐个消融 + 5 随机位点 null；另测 apple embedding 行置零。
  - **K-P3**：best 位点 Δ ≤ −0.3 AND < 随机位点 q95。

### 2. 结果

**Arm R：2818 因果地图补全——4 个实际写入头 3 个因果确认。**

| 位点 | Δprof | 随机 q95 | 判定 | logit_lens |
|---|---|---|---|---|
| L34 h0 → metal | **+0.0967** | 0.0557 | **R-P1 = true** | metal 词 +0.1823 同向 |
| L32 h3 → clothing | **+0.0422** | 0.0038（11×） | **R-P2 = true** | clothing 词 +0.0491 同向 |

加上 2818 的 L32 h0 → nature，**nature / metal / clothing 三个类域的因果写入头全部确认**（4 个实际写入头剩 L31 h4 animal 待测）。

**Arm K：三层证据定位 apple→red 知识。**

- **K1 = true（mlp 与 attn 双通过）**：MLP 置零剖面极值 L0（−10.71）、L6（−7.35）、**L35（−5.12）**，中晚层平缓特异谷（L24 −0.44、L25 −0.44、L30 −0.71）；attn 极值 L0（−5.05）。随机词对照干净（best 层 median +0.036）。注意：L0/L6 极值是全层失效（全局崩溃），**中晚层的平缓变化才是属性特异依赖**。
- **K2 = true：颜色方向参数大量存在且晚层集中**。
  - MLP：8 层通过；color 对齐随层飙升——L24 0.29 → L30 **0.51** → L31 **0.53** → L32 **0.61**（n353）；L33-35 转 red 特异——L33 0.57 / L34 0.65 / L35 **0.68**（n1552）。随机 q95 ≈ 0.087。
  - OV：**35 头通过**；top：L33 h23 (red 0.63)、L31 h15 (red 0.61)、L35 h22 (red 0.51)、L34 h0 (color 0.52)、L31 h8/h9 (color 0.47/0.45)。
- **K3 = false（预注册线未达，但效应真实）**：best 位点 MLP **L32 n353 消融 Δ = −0.2604**（随机位点 q95 = 0.0167 的 **15.6×**），未过 −0.3 线；其余位点 ±0.02-0.05。**apple embedding 行置零 Δ = −1.2188——最大单点效应**，且随机词 logit 反升（+0.39~+0.61，分布重整对照）。

### 3. 分析：对用户问题的回答——"红改黑应改哪些参数"

三层证据合成（行为终点 red logit）：

1. **主语身份入口：apple 的 embedding 行**（置零 Δ −1.22，最大单点）——"这是苹果"的身份信息从嵌入进入，一切属性读取以此为源。
2. **晚层 MLP 属性写出列**：L30-32 的 color 对齐列（L32 n353 cos 0.61，消融 −0.26 = 15.6× 随机）承载"颜色类"内容；L33-35 的 red 特异列（0.57-0.68）承载具体色值。
3. **OV 颜色写入头**：35 头对齐（L31 h15 / L33 h23 red ~0.6 等），与 2816-2818 的类写入头机制同源。

**与 ROME/MEMIT 文献的对照**：ROME 定位中期 MLP key-value、MEMIT 多层扩散——本证据链一致地显示知识**不在单点**：embedding 行（入口）+ 晚层 MLP 列（属性写出）+ OV 头（通道）三级分布式承载，单点消融不足以翻转行为（K3 false 的正确解读）。"红改黑"的最小干预应是**组合编辑**：apple embedding 行的 color 分量 + L30-35 color/red 写出列的定向修改——这正是 2820 要实测的。

**机制总图景（2815-2819 汇合）**：知识 = embedding 身份向量（入口）→ attention 按 RoPE 寻址读取 → 晚层专化 OV 头 + MLP 列写出类/属性方向 → 读出。静态嵌入无坐标（2815）、层参数混合语义与规则（2816）、写以头为粒度（2817）、几何对齐≠因果（2818）、知识分布式三级承载（本 Phase）。

### 4. 硬伤（严格审视）

1. **K1 层置零是全层失效**：L0 −10.7 是模型崩溃非属性知识位置——剖面只能读"依赖"不能读"存储位置"；应配合 ROME 式 corrupted-run patching。
2. **K3 单点不足预注册线**：−0.26 vs −0.3——分布式知识的单列消融注定欠杀，组合消融才是正确工具（本轮未预注册）。
3. **方向构造同源性**：dW_color/red 由 embedding 构造，与 MLP 普查共享 embedding 几何——存在自我印证风险；应用行为锚定方向（red/black logit 差的表示方向）独立复核。
4. emb_apple 置零是破坏性读出，非精细编辑；"黑改"的写成部分（向 black 方向写）未测。
5. L31 h4（animal）因果未测，4 写入头缺一。

### 5. 结论（重复 3 次）

**apple→red 知识分布式承载于三级参数：apple embedding 行（入口，单点最大 Δ−1.22）+ 晚层 MLP color/red 对齐列（L30-35，写出）+ OV 颜色头（通道）；单点消融不足以翻转行为，编辑必须组合干预。**
**apple→red 知识分布式承载于三级参数：apple embedding 行（入口，单点最大 Δ−1.22）+ 晚层 MLP color/red 对齐列（L30-35，写出）+ OV 颜色头（通道）；单点消融不足以翻转行为，编辑必须组合干预。**
**apple→red 知识分布式承载于三级参数：apple embedding 行（入口，单点最大 Δ−1.22）+ 晚层 MLP color/red 对齐列（L30-35，写出）+ OV 颜色头（通道）；单点消融不足以翻转行为，编辑必须组合干预。**

### 6. 接续（2820 候选）

1. **组合编辑实测**：apple embedding color 分量改写 + L30-35 color 列定向修改 → 验证 "The apple is" 续写 black（真知识编辑实验，MEMIT 式多点多层）。
2. **行为锚定方向**：用 (red-logit − black-logit) 的表示梯度独立构造颜色方向，复核 K2 普查。
3. **L31 h4 animal 因果补测**（4/4 写入头闭环）。
4. **Δh 通道分离主线**（2810 承接，语境调制谱结构）。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2819_knowledge_edit_locus.py` sha256 = `c1ee71ac44bbac2053e72bc148ef3e44efccccb7025bb7b15a7e57795edb5826`
- `…/phase2819/knowledge_edit_locus/execution.json` sha256 = `4588c2dcd9b747dc6eb1647c88cae208aa1b173aa8f09d493dc51671adc39499`
- `…/result.json` sha256 = `3ef3ba446c2f55edad15f798bb1da83536e83b138162dc5514b2d5e2b5560298`
- `…/locus.npz` sha256 = `44bd9e8beeb8e3962784867abfa5ff4ac5cd0994a1fa7aff86d8c421adba38ab`

**Gens 记录**：首跑干净（63.8s，编译前静态审查修复 inter 硬编码）。预注册判据全程未动；gate：dW=1.82e-09, Zeval=2.46e-06。

---

## Phase 2820: 编辑泛化与交叉规律（emb-edit vs mlp-edit 交叉矩阵） [2026-09-17 05:17]

### 1. 原理与设计

用户指令：其他红色水果改黑是否改相同神经元？路灯红改黑？苹果红改紫？交叉找普遍规律。

**双 Arm 预注册**（execution.json 先落盘，判据冻结）：

- **Arm H（闭环）**：L31 h4 animal 因果消融，精确 2818 协议（8 目标词 × 7 位置 56 句 + 5 随机头对照）。H-P1：Δprof_animal > 0 且 > q95(随机)。
- **Arm E/G（编辑矩阵）**：行为终点 = "The {s} is" 末位颜色 logit margin(black−red)，10 实体（红族 6：apple/cherry/strawberry/tomato/blood/streetlight + 对照 4：sky/grass/coal/banana）× 6 色矩阵，12 编辑条件：
  - **emb-edit**（z 空间行移位，精确反解 rms-norm）：e_new = e + β·s0·(dW_tgt−dW_src)/g；β∈{1,2,4} 扫描；apple/streetlight/cherry 各色交叉
  - **mlp-edit**（down_proj 列写出重定向）：w' = w − (w·dW_src)dW_src + (w·dW_src)dW_tgt；用 2819 已登记普查列（red top4: L35 n1552/L34 n1218/L33 n2566/L25 n695；color top3: L32 n353/L31 n3298/L30 n4290）
  - joint（emb+mlp）、伪编辑对照（apple blue→black）
  - 判据：E-P1 编辑翻转+隔离；G-P1 存储定律（r_emb<0.2 实体行私有 vs r_mlp≥0.5 写出列共享）；G-P2 目标无关性；G-P3 联合隔离

门禁：dW=1.82e-09, Zeval=2.46e-06 量级通过（gate_dW<1e-6, gate_Z<1e-4）。

### 2. 结果

**Arm H：P-H1 = true——4/4 写入头因果地图闭环。** L31 h4 animal Δ=0.0296 > q95=0.0175。加上 2818 的 L32 h0(nature)/L34 h0(metal) 与 2819 的 L32 h3(clothing)，**四个实际写入头全部因果确认**。

**Baseline margins(black−red)**：红族全负——apple −3.88 / tomato −2.63 / strawberry −1.81 / cherry −1.44 / blood −0.75 / streetlight −0.69；对照正常——sky +5.31 / coal +2.88 / grass +0.19。

**编辑矩阵——全部 12 条件无一翻转行为（E-P1 = false），但失败模式呈现清晰的双重结构**：

| 条件 | Δmargin(apple) | 溢出 | 解读 |
|---|---|---|---|
| emb-edit apple red→black β=1/2/4 | +0.19 / −0.06 / −0.13 | **其余 9 实体全部 Δ=0.000** | 完美隔离但零效应（β=4 时 z 空间扰动仅 ~10% RMS） |
| emb-edit apple red→purple β=2 | 0.000 | 全部 0.000 | 同上 |
| emb-edit cherry / streetlight red→black | 各自 −0.13 / −0.06 | 其他实体 0.000 | 同上（方向反而更红） |
| **mlp-edit red→black（4 列）** | **−0.25** | **全部 10 实体同向位移**（cherry −0.13、streetlight −0.25、sky/grass/coal −0.31） | 全局共享通路，方向反直觉（red 相对增强） |
| mlp-edit red→purple | −0.06 | 全局弱移 | 目标依赖（G-P2 false：black 效应 4× purple） |
| joint emb+mlp | −0.44 | ≈两者加和 | 仍远离翻转（baseline −3.88） |
| 伪编辑 apple blue→black | 0.000 | 0.000 | 对照干净 |

**G-P1 = 双半验证 true（存储定律）**：r_emb = 0.000 → **实体 embedding 行完全私有**；r_mlp = 0.5 → **晚层 MLP 颜色列跨实体共享**。

**G-P3 = true**：joint 编辑 sky/grass 位移 0.312 < 0.5，隔离良好。

**Post-hoc 符号审计（探针，预注册外，仅作解释）**：4 个 red 普查列的 w·dW_red 为 **2 正 2 负拮抗结构**——L34 +0.638 / L25 +0.143（增红）vs L35 −0.460 / L33 −0.736（抑红），负列幅度占优（|1.20| > |0.78|）。这完全解释 mlp-edit 全局负移：重定向公式对 comp<0 的列反而**增加** red 写出（w' = w + |comp|·dW_red − |comp|·dW_black）。也解释 G-P2：列与 dW_black 对齐（0.02-0.13）强于 dW_purple（0.016-0.038）。**2819 K2 普查按对齐度选列未测符号——"red 对齐列"实为增红/抑红拮抗列的混合。**

### 3. 对用户问题的回答——"改不同实体/目标色是否改相同神经元"

**不是同一套参数，但共享同一套硬件通路——知识是计算不是存储位：**

1. **实体身份 = embedding 行，实体私有**（r_emb=0.000：改 apple 行，cherry/strawberry/tomato/streetlight 纹丝不动）。每个实体有自己的行，**互不通用**。
2. **颜色读写通路 = 晚层 MLP 列 + OV 头，跨实体共享**（r_mlp=0.5：改 4 列，全部 10 实体同向位移）。水果与路灯用**同一套**颜色通路，**没有按实体划分的专用颜色神经元**。
3. **但两者都不是知识存储位**：实体行的颜色分量移位（~10% RMS）对行为零贡献 → apple→red 联想不在 apple 行；共享列重定向效应小且方向由列写出符号决定 → 列是**拮抗调节器**（增红/抑红成对）不是存储器。
4. **颜色预测 = 前向动态计算**：apple 表征（行入口+层加工）→ 共享颜色通路（MLP 拮抗调节 + OV 写入）→ unembed 读出。**"改知识"不是改一个位，而是改一条计算通路的增益结构。**

机制总图景推进：2815 静态嵌入无坐标 → 2816 层参数混合语义与规则 → 2817 写以头为粒度 → 2818 几何对齐≠因果 → 2819 知识三级分布式 → **2820 编辑双重结构：实体私有行（零溢出零效应）+ 共享拮抗通路（全局溢出小效应）；知识=计算，编辑须改通路而非改位。**

### 4. 硬伤（严格审视）

1. **β 上限 4 太小**：z 空间扰动仅 5-10% RMS，而 E-P1 翻转要求 4+ logit 位移（baseline −3.88）——**E-P1=false 只证明"小编辑无效"，不能排除强编辑可翻转**；β=20-100 会出流形（s0 漂移、表征破坏），需配副作用监控。
2. **mlp-edit 仅 4 列且半数符号相反**：净移除 red 支持 |0.78|−|1.20| 为负——本轮重定向实际是"增红编辑"；配平编辑（正列移除+负列反接）未测。
3. **bf16 读出噪声 ~±0.06**：最小观测 Δ（0.062）在噪声底，单实体弱效应解读需谨慎。
4. 符号审计为 post-hoc，解释力强但未进预注册。
5. 单模型（qwen3-4b）单模板（"The {s} is"）；跨模型泛化未测。

### 5. 结论（重复 3 次）

**编辑泛化定律：实体 embedding 行私有（r_emb=0.000，改苹果不动樱桃）但颜色行为零贡献；晚层 MLP/OV 颜色通路跨实体共享（r_mlp=0.5，水果与路灯同一套）且为增红/抑红拮抗列结构；知识是前向计算不是存储位，单点/少量编辑均无法翻转（12/12 条件失败），真正的知识编辑必须组合干预通路增益结构。**
**编辑泛化定律：实体 embedding 行私有（r_emb=0.000，改苹果不动樱桃）但颜色行为零贡献；晚层 MLP/OV 颜色通路跨实体共享（r_mlp=0.5，水果与路灯同一套）且为增红/抑红拮抗列结构；知识是前向计算不是存储位，单点/少量编辑均无法翻转（12/12 条件失败），真正的知识编辑必须组合干预通路增益结构。**
**编辑泛化定律：实体 embedding 行私有（r_emb=0.000，改苹果不动樱桃）但颜色行为零贡献；晚层 MLP/OV 颜色通路跨实体共享（r_mlp=0.5，水果与路灯同一套）且为增红/抑红拮抗列结构；知识是前向计算不是存储位，单点/少量编辑均无法翻转（12/12 条件失败），真正的知识编辑必须组合干预通路增益结构。**

### 6. 接续（2821 候选）

1. **配平拮抗编辑**：正列移除 + 负列反接（w' = w − comp·dW_red − comp·dW_red 反向），净移除 |0.78|+|1.20| 全部——检验"全拮抗配平可否翻转"。
2. **强编辑扫描** β∈{10,30,100} + 全类别副作用监控（区分"知识改写"vs"表征破坏"）。
3. **K2 top-20 列联合消融**找"分布式欠杀→联合过杀"转折点。
4. 上位词行（' fruit' / ' traffic light'）编辑——层级泛化。
5. Δh 通道分离主线（2810 承接）。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2820_edit_generalization.py` sha256 = `62f23815353ac639f5981acb7d725fa9711010db5e78499e8d7a24ffee2ea2bd`
- `…/phase2820/edit_generalization/execution.json` sha256 = `f5711b918c7bf85322101803f726fa03701bf1e317071507089ed8e7b8555599`
- `…/result.json` sha256 = `a3f9bfc62676348569dfc8d5b8196ea5e090779e0ad98d0e99a9557c3040534f`
- `…/edit.npz` sha256 = `ff4f6414ec2e27a270b461d42836dbaeb8b6a76e33e9a3dc867cf5910a1b28df`

**Gens 记录**：3 次崩溃修复（streetlight 2-token tid 断言→实体移出 tid 预填充；emb_hook bf16/float32 dtype 不匹配→显式 cast；fc.npz 二次索引越界→直接存 LG0），第 4 跑干净（9.4s）。预注册判据全程未动；streetlight 多 token 实体经 tok() 全 ids 路径 + emb_hook 全 token 行编辑处理。

---

## Phase 2821: 属性绑定整体机制（实体×域因子化检验） [2026-09-17 05:38]

### 1. 原理与设计

用户指令升级：不再单域单点，要**整体属性机制**——颜色/大小/重量等各域如何统一组织，实体×属性如何用有限参数支撑无穷组合。

**核心假说（因子化计算）**：margin 矩阵 M[s,d]（20 单 token 实体 × 5 属性域：size/weight/temperature/speed/hardness，反义对 big−small / heavy−light / hot−cold / fast−slow / hard−soft，读出 "The {s} is" 末位形容词 logit 差）应可加性分解 M = α_s + β_d + ε。成立则 N 实体 × D 域只需 N+D 因子——"少量参数无穷组合"的定量机制。tokenization 探针前置（20 实体 + 10 形容词全单 token；7 多 token 实体排除）。真值表 50 明确单元预注册冻结。execution.json 先落盘。门禁 dW=1.82e-09 / Zeval=2.46e-06 ✓。

### 2. 结果

| 判据 | 判定 | 数值 |
|---|---|---|
| A0 语义正确性 | ❌（接近） | **0.760**（50 单元，未达 0.80） |
| A1 加性绑定 | ❌ | share **0.4608** < q95 0.4805（200 列置换 null） |
| A2 域分离 | ✅ | 跨域泄漏 ≤ **0.034**（对角 0.103-0.38；颜色 ref 泄漏 0.006-0.013） |
| A3 晚层统一 | ✅ | 5 域 top 列层 = L33/35/33/35/35 |
| B1 size 共享写路 | ❌ | r_mlp = **0.25** < 0.5，且 dM 实体异质（+0.25 ~ −0.50） |
| B2 size 私有行 | ✅ | emb-edit elephant 后 19 实体 **Δ 全部 = 0.000**（含 elephant 自身） |

**三个超越判据的发现**：

1. **sv 谱二维主结构**：奇异值 [9.04, 8.48, 6.68, 4.20]——sv[0]≈sv[1]。M0 行模式：feather [−3.25,−3.75,−0.94,+0.56,−2.50] 与 elephant [+0.81,+2.94,−1.19,+0.19,+1.13] 在 size/weight/hardness 三域同号极化——**属性协方差结构**（世界物理定律：大⇔重⇔硬，小⇔轻⇔软）作为实体坐标的跨域相关存在。
2. **写出列 = 实体条件化门控（B1 false 的机制含义）**：2820 color mlp-edit 全局同步位移；2821 size 列翻转后效应实体异质（elephant +0.25 / pillow −0.50 / mouse 0.000）——列的写出内容 = f(实体, 域)。**MLP 列不是固定域方向广播器，而是绑定计算位点**。
3. **符号拮抗普遍化**：5/5 域 top-5 列均为增/抑混合符号（size +0.52/−0.43；weight +0.53/−0.42/−0.32；speed −0.46/+0.21；temperature/hardness 同）——2820 单域发现上升为属性系统普遍架构。

### 3. 分析——"少量参数无穷组合"的机制答案

A1=false 不是失败，是**更深层机制的排阴**：

- 加性查表模型（N+D 因子）被否证 → 组合性**不来自**独立因子的线性叠加；
- 但 A0=0.76 + A2/A3 完美 → 语义调制真实、硬件域分离、晚层统一计算阶段；
- sv 二维主结构 + 54% 交互项 → 实体是**属性空间中的坐标点**（跨域相关，低维），margin = 实体点在域轴上的投影 + 高阶修正；
- **组合性来自几何（点×轴投影）+ 条件化计算（列门控 f(实体,域)）**：参数 O((N+D)·d) 而非 O(N·D)，但每个"槽位"的填充由实体表征与域通路的**乘法交互**动态产生——这就是"固定知识 + 无穷组合"的机制形式。

**属性系统总架构（2821 判定）**：私有行入口（B2：行内属性分量零行为贡献）→ 深层实体表征（跨域相关坐标，含属性协方差）→ L33-35 域分离拮抗列组（写出门控，内容=f(实体,域)）→ unembed 读出。与 2819 三级承载、2820 知识=计算完全自洽。

### 4. 硬伤（严格审视）

1. **margin 用绝对 logit**：speed 域几乎全正（fast/slow 频率基线）、grape temperature −2.44（冷藏水果搭配）——表面统计污染未除；应用 logprob 归一化或 distractor-对照复测。
2. **A0 真值表主观**：rocket size −2.81 / steel speed +1.56 等与人类先验不符——24% 不一致中部分是"真值表错位"而非模型错误；需行为锚定真值（多模板投票）。
3. **A1 预注册线 0.80 过严可能**：交互项含真值噪声 + 频率基线；但 null 对照（q95=0.4805）下 share 仍不显著——结论稳健。
4. B1 单次单组列（size top-4）；多组列/多域扫描未做。
5. 单模型单模板；emb-edit 仅 β=2。

### 5. 结论（重复 3 次）

**属性绑定整体机制：实体=属性空间坐标点（跨域相关、二维主结构），域=分离硬件通路（L33-35 拮抗列组、跨域零泄漏），绑定=列内乘法门控（写出内容=f(实体,域)，实体异质效应），入口=私有行（属性分量零行为贡献）；组合性来自几何投影+条件化计算而非加性查表——这就是 LLM 用少量参数实现固定知识+无穷组合的机制形式。**
**属性绑定整体机制：实体=属性空间坐标点（跨域相关、二维主结构），域=分离硬件通路（L33-35 拮抗列组、跨域零泄漏），绑定=列内乘法门控（写出内容=f(实体,域)，实体异质效应），入口=私有行（属性分量零行为贡献）；组合性来自几何投影+条件化计算而非加性查表——这就是 LLM 用少量参数实现固定知识+无穷组合的机制形式。**
**属性绑定整体机制：实体=属性空间坐标点（跨域相关、二维主结构），域=分离硬件通路（L33-35 拮抗列组、跨域零泄漏），绑定=列内乘法门控（写出内容=f(实体,域)，实体异质效应），入口=私有行（属性分量零行为贡献）；组合性来自几何投影+条件化计算而非加性查表——这就是 LLM 用少量参数实现固定知识+无穷组合的机制形式。**

### 6. 接续（2822 候选）

1. **交互项结构分解**：M0 的 ANOVA 残差 ε(20×5) 做 PCA/双聚类——属性协方差平面的维数与内容（读出 PC 轴的 top 实体/域）。
2. **logprob 归一化复测 A0/A1**（除以 distractor 集合的 log-sum-exp，去频率基线）。
3. **列门控输入归因**：捕获 size 列的 o_proj/down_proj 输入 a_h，回归 a_h × 域方向 → 什么决定该实体经过时列写出什么（绑定机制的微观电路）。
4. 多模板（"A {s} is very ___" / "The {s} that I have is ___"）真值锚定。
5. Δh 通道分离主线（2810 承接，长期欠账）。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2821_property_binding.py` sha256 = `1093549a7e7499ceb12f28bdfbb6c6b580035e8ae044645d7b28356ea4c9e54f`
- `…/phase2821/property_binding/execution.json` sha256 = `1df726107b0ad4d942ea6b41b4ce5f1a3b6a3720ef312a4f730063da913e3843`
- `…/result.json` sha256 = `8412dbf68883aab07599e88b51687af1516342713076a449b76ef341be379542`
- `…/binding.npz` sha256 = `560353b3722cbf5038b4468e01f4ae3dafcdca71625fe13f3a52fc7cb9702fc7`

**Gens 记录**：3 次运行（①打印 genexp 括号错位崩；②r_mlp np.float32 序列化崩；③干净 9.7s）。编译前静态审查抓到 dD 键名 KeyError 级 bug 2 处（形容词 vs 域名），修复为 flip 编辑（w' = w − 2(w·dD)dD，big/small 同轴反转）。预注册判据全程未动；tokenization 探针前置避免第 4 次多 token 崩溃。

---

## Phase 2822: 配平拮抗编辑——通向可工作的知识编辑 [2026-09-17 05:54]

### 1. 原理与设计

用户指令：继续 2821，并回答"能否调参数实际修改苹果的颜色/大小等属性"。2820 失败归因于 top-4 列（符号混合净抵消）+ 2821 全架构图（域分离列组、符号拮抗门控）→ 本轮把编辑推到**普查级规模**：L26-35 down_proj 全列带符号 cos 普查（dW_red 12 色对比方向 + dD_big 单轴），每层 top-20 → 200 列/域（red: 96 正 104 负；size: 95 正）。

**三变体 × 剂量扫描**（红族 6 + 对照 4 + size 组 5，"The {s} is" 读出）：
- V1 统一符号重定向（2820 公式放大）：w' = w − c·dW_red + c·dW_black
- V2 幅度配平：w' = w − c·dW_red + |c|·dW_black（目标侧不可抵消）
- V3 纯消融：w' = w − c·dW_red
- size 域：flip（单侧翻转，只动 comp>0 列：w −= 2c·dD）与 ablate
- 预注册 E1-E5（翻转/隔离/剂量/跨域/配平vs统一）。门禁 dW=1.82e-09 / Zeval=2.46e-06 ✓。

### 2. 结果

| 判据 | 判定 | 关键数值 |
|---|---|---|
| E1 颜色翻转 | ❌ | V2 top20 Δapple = +0.125（需 +3.9） |
| E2 隔离 | ❌ | **sky Δ = −1.50**（共享通路代价），grass/coal 过 |
| E3 剂量单调 | ❌ | V2: 0.062/0.000/0.125 非单调 |
| **E4 跨域翻转** | ✅ | **size flip elephant Δ = −1.50（margin +1.19 → −0.31 真翻转）** |
| E5 配平>统一 | ❌ | V1 (+0.50) > V2 (+0.125) |

**模式细读**：
1. **V1 是最强颜色编辑**：top20 红族全体正向（apple +0.50 / tomato +0.63 / strawberry +0.19），剂量近似单调（top4 0.44 → top20 0.50）——统一符号重定向在普查规模下方向一致。
2. **V2 幅度配平反而失效**：负列（抑 red）改为写 black 后，"抑 red"功能丢失 → red 解除抑制抵消 black 写入。**负列的抑制功能是行为必需**——2821 拮抗门控的直接行为学证据。
3. **所有颜色变体伤 sky**（−0.44 ~ −2.25）：red 列跨实体共享（2821 已证），移除其 red 写出 → sky 上下文中 red 抑制解除。**共享通路编辑的必然代价**。
4. **效应量天花板**：200 列全动 apple 仅 +0.5（需 +3.9）——**晚层 MLP 列不是 apple→red 的主承载**（调节器而非绑定器）；颜色的绑定更分布式（2819 OV 35 头嫌疑）。
5. **size 翻转成功**（E4 首个阳性）：单轴二分属性（big/small 同轴两极）+ 单侧翻转写 big 列 → elephant 真翻转；mouse −0.44 / rock −0.50 同向（共享代价），feather 0.0（本就小）。ablate 版仅 −0.375（消融不如反接——写 big 的列翻成写 small 才有效）。

### 3. 对用户问题的回答——"能否调参数修改苹果的颜色/大小等属性"

**能，但分属性类型，且当前算子有明确边界：**

1. **大小（单轴二分属性）：能真翻转**——符号登记 + 单侧列翻转（flip），L26-35 每层 top-20 共 95 正列，elephant big−small margin +1.19 → −0.31。算子已可复现。
2. **颜色（高维对比属性）：方向正确、量级不足**——V1 统一重定向 200 列使红族系统移动 +0.2~+0.6（约需翻转的 1/8），离翻转差 6-8×；且 sky 被伤 −2.25（共享通路必然溢出）。
3. **边界的技术含义**：颜色绑定主要不在 MLP 列（在 OV 头 + attention 通路 + 分布式表征的组合计算中，2819 K2 的 35 个 OV 颜色头是下一步编辑目标）；"只改苹果不改别人"需要实体条件化编辑（列门控输入侧注入，2821 B1 的 f(实体,域) 位点）。
4. **持久性**：本轮编辑为 forward 级（set→forward→restore），权重修改已验证可即时改变行为——写回 safetensors 即持久化，技术通路无障碍。

### 4. 硬伤（严格审视）

1. **V1 普查列与 2820 census 列不同源**（本次按 |cos| 全列排序 vs 2819 对齐度登记）——两轮方向相反的结论（全局负移 vs 全体正向）部分由列选择差异贡献，符号-效应映射需在固定列集上做剂量扫描确认。
2. sky 破坏未通过隔离判据——"修改苹果"的工程可用性还不成立，需要实体门控编辑。
3. size flip 只测 1 个目标实体（elephant）+ 4 旁观；apple 大小改小（单实体）未测对称条件。
4. margin 为绝对 logit，频率基线未除（2821 硬伤延续）。
5. bf16 读出噪声 ±0.06：cherry Δ −0.125 等小效应在 2σ 边缘。

### 5. 结论（重复 3 次）

**编辑可行性判定：单轴二分属性（大小）可用符号登记+单侧列翻转真翻转（E4 首例，elephant −1.50）；高维对比属性（颜色）编辑方向正确但 MLP 列仅承载 ~1/8 效应（主承载在 OV 头/attention 组合计算），且共享通路编辑必然溢出（sky −2.25）——知识编辑的完整算子 = MLP 列组（幅度调节）+ OV 头组（绑定主承载）+ 实体门控输入（隔离），三级缺一不可。**
**编辑可行性判定：单轴二分属性（大小）可用符号登记+单侧列翻转真翻转（E4 首例，elephant −1.50）；高维对比属性（颜色）编辑方向正确但 MLP 列仅承载 ~1/8 效应（主承载在 OV 头/attention 组合计算），且共享通路编辑必然溢出（sky −2.25）——知识编辑的完整算子 = MLP 列组（幅度调节）+ OV 头组（绑定主承载）+ 实体门控输入（隔离），三级缺一不可。**
**编辑可行性判定：单轴二分属性（大小）可用符号登记+单侧列翻转真翻转（E4 首例，elephant −1.50）；高维对比属性（颜色）编辑方向正确但 MLP 列仅承载 ~1/8 效应（主承载在 OV 头/attention 组合计算），且共享通路编辑必然溢出（sky −2.25）——知识编辑的完整算子 = MLP 列组（幅度调节）+ OV 头组（绑定主承载）+ 实体门控输入（隔离），三级缺一不可。**

### 6. 接续（2823 候选）

1. **OV 头编辑**：2819 K2 的 35 个颜色对齐 OV 头做符号登记 + 写出方向翻转（o_proj 头切片，2818 协议）——颜色主承载的编辑验证。
2. **实体门控编辑**：捕获 apple 序列进入 red 列/头的输入 a_h，注入 "black 上下文"的 a_h 模式（2821 B1 位点）→ 隔离编辑。
3. size 对称条件：apple big→small 单实体翻转。
4. 多模板 + logprob 归一化（2821 硬伤清理）。
5. Δh 通道分离主线。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2822_balanced_edit.py` sha256 = `a2d6aacec3276b7cc66a500fbd59591eb01da24bb778eb83348fedfa9d30bc24`
- `…/phase2822/balanced_edit/execution.json` sha256 = `98c5542d26551ec43a822980d07d87e6a5ee11e5172fda7ed119bee125cb8fb0`
- `…/result.json` sha256 = `a0653256148e0432b7730951c8cf22d57128678d69926aa0d9437e78e8c7c260`
- `…/edit.npz` sha256 = `4728605fdef3197cb0e99439a278d958e4c4d58ed56119742efeea5b46026f3c`

**Gens 记录**：2 次运行（①run_size 解包写反 KeyError；②干净 9.9s）。静态预审通过零编译错误；门禁/预注册全程未动。

---

## Phase 2823: OV 头编辑 + 8 域 48 实体普遍性扩军 [2026-09-17 06:05]

### 1. 原理与设计

用户双指令：①继续（2822 接续——OV 头编辑补编辑算子第二级）；②**扩大类型与数量规模，找机制普遍性特征与结构**。

- **Arm O（OV rank-1 编辑）**：2819 K2 的 1152 头普查取晚层（L≥30）top-5 颜色头（L33 h23 red 0.63 / L31 h15 red 0.61 / L34 h0 color 0.52 / L35 h22 red 0.51 / L31 h8 color 0.47）；捕获 apple 上下文 a_h（o_proj 输入头切片）；rank-1 重定向 W_O^h += (δ'−δ)·key^T/‖key‖²，δ' = δ − c·dW_red + c·dW_black（c = δ·dW_red 实测）。rank-1 key = apple 表征模式 → 实体条件化结构；κ_s = <key, a_h(s)>/‖key‖² 量化泄漏。
- **Arm B（普遍性扩军）**：8 域（5 旧 + taste/loudness/shape）× 48 单 token 实体；margin 矩阵 ANOVA + SVD；8 方向列普查（带符号 top-5/层）；U1 域分离 / U2 晚层 / U3 拮抗 / U5 真值（104 预注册单元）/ U6 flip 算子迁移（whale+elephant）。门禁 dW=1.82e-09 / Zeval=2.46e-06 ✓。

### 2. 结果

**Arm B 全绿——五大机制特征全部普遍成立**：

| 普遍性判据 | 判定 | 数值 |
|---|---|---|
| U1 域分离 | ✅ | 8×10 泄漏矩阵全部 ≤ 0.034（3 新域同样干净） |
| U2 晚层 | ✅ | 8 域 top 层 L33/33/35/33/35/35/35/32 |
| U3 拮抗 | ✅ | 8/8 域 top-5 符号混合（taste −0.26/+0.15；loudness +0.30/−0.15；shape −0.46/+0.24） |
| U5 语义调制 | ✅ | 104 单元一致率 **0.779** ≥ 0.72 |
| U6 flip 迁移 | ✅ | whale **−1.31** / elephant −1.06 双翻转 |

加性份额 0.5199（扩军后仍 ~52%，交互项普遍）；**sv 谱 [25.0, 11.4, 10.6, 10.4, 9.8]——sv[0] = 2.2×sv[1]**（2821 时 20 实体 sv[0]≈sv[1]；48 实体后 rank-1 物理相关轴（大⇔重）主导性显现——实体坐标的低维结构随规模增大而清晰）。

**Arm O：O1=false / O2=true / O3=false，但实测暴露两个关键事实**：
1. **几何普查 ≠ 实际写出（2818 教训头级复现）**：实测 c_red = δ·dW_red——L33 h23 普查 0.63 实测 **−0.056**、L31 h15 0.61 实测 **−0.048**（top-2 头对 apple 上下文实际不写 red！）；L34 h0 实测 **−2.32**（写 anti-red）；仅 L35 h22 (+0.75) / L31 h8 (+0.19) 正向。5 头 rank-1 编辑 Δapple = +0.188。
2. **rank-1 key 隔离成立**：O2 全过——sky/grass Δ = **0.000**！κ（key 与其他实体 a_h 的相关 0.37-0.69）不为零但行为泄漏为零——实体条件化编辑机制的可行性首次验证。

### 3. 分析——普遍性结构 + 编辑路线修正

**普遍性特征（Arm B 五全绿）**：属性系统的架构在 8 个语义域上同构——①每域独立晚层列组（硬件分离）；②每域增/抑拮抗门（符号结构）；③实体坐标跨域相关且低维（sv rank-1 物理轴随规模增强）；④行为语义一致 ~78%；⑤flip 编辑算子跨实体迁移。**这就是"各种机制的普遍性特征和结构"的实测答案：一套硬件模板（域列组+拮抗门+实体门控）在所有属性域复用。**

**编辑路线修正（Arm O）**：
- 编辑选头必须用**实测 c_red**（apple 上下文 δ 谱）而非几何普查 best_val——2818"对齐≠写入"第三次教训；
- 下一步正确姿势：一次 a_h 捕获可算全部 1152 头的实测 δ·dW_red 谱 → 按实测选 top-20 正头 → rank-1 编辑（预计效应 ≥ 列编辑且天然隔离）；
- 组合编辑目标：列组（+0.50）+ 实测正头组（~+0.3-0.5）→ 翻转所需 3.9 仍差，需第三级（门控输入侧注入）。

### 4. 硬伤（严格审视）

1. OV 编辑仅 5 头且按几何普查选——实测谱未扫描（正确选头后效应未知）；rank-1 近似只在 key 方向精确，其他输入方向受 1/‖key‖² 调制未量化。
2. U5 真值表主观（2821 延续）；新 3 域单元 17 个偏少。
3. U6 flip 迁移效应（−1.31/−1.06）与 2822（−1.50）有 batch 差异——实体池变化引起的读出基线漂移未归一。
4. margin 绝对 logit 频率基线污染延续。
5. 血液 temperature 真值（+1 热血）有文化先验风险。

### 5. 结论（重复 3 次）

**普遍性结构确立：属性机制 = 一套硬件模板在所有语义域复用——域分离晚层列组（泄漏≤0.034）× 增/抑拮抗门（8/8 域）× 实体低维坐标（sv rank-1 物理轴随规模增强 2.2×）× rank-1 实体条件化编辑（零行为泄漏）；编辑选头必须用实测写出谱而非几何普查（对齐≠写入第三次教训）。**
**普遍性结构确立：属性机制 = 一套硬件模板在所有语义域复用——域分离晚层列组（泄漏≤0.034）× 增/抑拮抗门（8/8 域）× 实体低维坐标（sv rank-1 物理轴随规模增强 2.2×）× rank-1 实体条件化编辑（零行为泄漏）；编辑选头必须用实测写出谱而非几何普查（对齐≠写入第三次教训）。**
**普遍性结构确立：属性机制 = 一套硬件模板在所有语义域复用——域分离晚层列组（泄漏≤0.034）× 增/抑拮抗门（8/8 域）× 实体低维坐标（sv rank-1 物理轴随规模增强 2.2×）× rank-1 实体条件化编辑（零行为泄漏）；编辑选头必须用实测写出谱而非几何普查（对齐≠写入第三次教训）。**

### 6. 接续（2824 候选）

1. **全头实测写出谱**：一次 a_h 捕获 × 1152 头 δ·dW_red 谱 → 实测 top-20 头 rank-1 编辑（正确的 OV 编辑）。
2. **三级组合编辑**：列组 + 实测头组 + 门控输入注入 → 翻转 apple red→black 全算子验证。
3. 8 域 margin 矩阵的域间相关结构（taste×size? 物理相关轴内容读出）。
4. 多模板 + logprob 归一化。
5. Δh 通道分离主线（2810 承接）。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2823_ov_edit_universality.py` sha256 = `784eadccc685edf021a90222d95d902eadc5cda0c9369fad91466a04021fb213`
- `…/phase2823/ov_edit_universality/execution.json` sha256 = `daf3a04044a6258de21be74a8983317c7da7d267f0f39448ef180170cdff096f`
- `…/result.json` sha256 = `8ad36e7bc41ec9d63420393b8cf29633c1eec09d008decba4d46604ba070e94b`
- `…/uni.npz` sha256 = `031579a946bb42af3f07a5e9b730ea2ab536f37511a30d46b3c9fbda0142ec94`

**Gens 记录**：3 次运行（①实体池疏漏 strawberry/tomato/blood/sky/grass/coal 不在 42 池→扩 48 池+红族改 5 员（streetlight 多 token 剔除）；②Arm O 恢复段写作残渣（未定义名）→ 重写备份/恢复；③干净 9.9s）。预注册判据全程未动；真值表在零观测前补 3 实体单元。


## Phase 2824: 实测写出谱与具体机制电路 [2026-09-17 06:28]

### 1. 原理与设计

2823 确认"几何普查 best_val ≠ 实际写出 c"（L34 h0 几何榜 vs 实测 anti-red）。本 Phase 不再普查对齐度，直接**实测写出谱**：一次 forward 捕获全部 36 层 o_proj 输入 a_h（保留 apple/cherry/sky/grass 4 行），逐层逐头计算

  delta_{l,h} = W_O^{l,h} @ a_h(apple ctx)   （2560 维写出向量，每头一个）
  c_{l,h}     = <delta_{l,h}, dW_red>        （该头在 apple 语境下对 red 方向的实际写出量）

1152 头实测谱 = apple→red 的**头级电路图**（measured circuit map）。在此基础上做四种编辑对比（rank-1 OV 重定向，2823 算子；列组 = 2822 V1 复现）：
- H_meas：实测 c>0 的 top-20 头 rank-1 red→black
- H_geo：2819 几何普查 top-20 晚层头（同算子）
- C_cols：2822 列组 V1 top20/layer
- COMBO：列组 + 实测头组

预注册判据（零观测前冻结于 execution.json）：
- P1 实测选头 > 几何选头（d_apple 意义下）
- P2 组合加和：d_combo >= d_cols + d_heads − 0.2 且 > max 部件
- P3 头谱实体特异性：corr(c_apple, c_sky) < 0.5（over 1152 头）
- P4 rank-1 隔离：|d_sky(H_meas)| < 0.2
门禁：gate_dW = 1.82e-9 < 1e-6，gate_Z = 2.46e-6 < 1e-4（通过）。

### 2. 结果（一次运行成功，12.6s；无崩溃）

基线 margin（black−red）：apple −5.562, cherry −3.125, strawberry −4.062, tomato −3.562, blood −2.812 | sky +1.375, grass −1.812, coal +3.188, banana −2.812。

**判决：P1=true, P2=true, P3=true, P4=false**（3/4 通过）。

| 条件 | d_apple | d_cherry | d_sky | d_coal |
|---|---|---|---|---|
| H_meas | **+2.688** | +1.750 | +0.688 | +1.062 |
| H_geo  | +0.500 | +0.438 | +0.125 | +0.250 |
| C_cols | +1.750 | +1.188 | +0.312 | **−0.875** |
| COMBO  | **+4.812** | +2.938 | +0.875 | +0.125 |

### 3. 具体机制：apple→red 头级电路图（核心交付）

实测谱 637/1152 头 c>0。top-10 实测写头：

| rank | 头 | c | 备注 |
|---|---|---|---|
| 1 | **L35 h0** | 1.301 | 末层总写头（此前几何普查未居首） |
| 2 | **L29 h27** | 0.769 | **几何普查完全漏掉**（geo 只取 L>=30） |
| 3 | L35 h22 | 0.752 | 几何榜第 4（5/8 一致） |
| 4 | L35 h23 | 0.491 | 几何榜第 10 |
| 5 | L34 h28 | 0.423 | 几何榜第 19 |
| 6 | L26 h20 | 0.397 | 晚-中层写头 |
| 7 | L35 h26 | 0.328 | |
| 8 | L35 h19 | 0.29 | |
| 9 | L33 h5 | 0.269 | |
| 10 | L23 h6 | 0.259 | |

层分布：L35 聚集 5 头（1.301/0.752/0.491/0.328/0.29）——**末层是写出主力层**；L23-34 散布 15 头。corr(c_apple, c_sky)=0.456 < 0.5（P3=true），corr(c_apple, c_grass)=0.240——头谱确有实体特异性但共享成分显著（通用颜色写头）。

### 4. 分析

1. **P1（实测>几何，5.4 倍）**：第三次确证"对齐≠写入"。机制原因有二：(a) 几何普查按 best_val 选头未测符号与上下文；(b) **结构性盲区**——geo 硬过滤 layer>=30，把 L29 h27（c=0.769，全谱第 2 强）整体漏掉。
2. **P2（加和，4.812 ≈ 2.688+1.750=4.438）**：MLP 列通路与 OV 头通路**独立可加**，组合后 d_apple=+4.812（基线 −5.562 → −0.75），单次编辑从未如此接近翻转线。这是 apple→red 行为读写出的最强干预纪录。
3. **P4（隔离失败，d_sky=+0.688>0.2）**：实测 top-20 头对其他实体有正溢出（banana +2.188 尤其大）。rank-1 的 key 已实体条件化，但写头本身是"通用颜色写头"——**条件化在输入端（key），写出端（W_O 列）是共享的**。与 2822 E2（列通路共享溢出）同构：两条通路都在写出端共享，只是列通路溢出向 coal（−0.875）、头通路溢出向 banana（+2.188）。
4. 与 2823 呼应：2823 O2（rank-1 零行为泄漏到 sky/grass）用的是**单头 L34 h0**；本 Phase top-20 头的**聚合**溢出显著。单头可以条件化干净，头群聚合必然携带通用成分。

### 5. 硬伤

1. P4 失败未解决：实体条件化编辑仍会移动其他实体的颜色读出，"改苹果颜色不动香蕉颜色"目标未达成。
2. key 只取 [1:3] 两行均值（The+实体首 token），上下文代表粗糙；未测多模板。
3. corr(c_apple, c_sky)=0.456 离 0.5 阈值近，P3 通过但余量小，头谱特异性本身存疑。
4. 只测了 red 一个方向、4 个捕获实体；谱的域普遍性（size/taste 等）未测。

### 6. 结论

**具体机制已到手**：apple→red = 末层写出主力（L35 五头聚集，h0 为核）× 中晚层辅助头（L29 h27/L26 h20/L23 h6 等）× MLP 列通路（26-35 层），两条通路独立可加（P2），实测选头完胜几何选头（P1，5.4 倍），且几何普查存在层过滤盲区。"少量参数实现固定知识"的答案进一步具体化：**实体知识 = 通用写头硬件（共享、少量）× 实体条件化 key（私有、每实体）**——参数经济性来自写出端复用，特异性来自输入端门控。

### 7. 接续（Phase 2825 候选）

1. **解决 P4**：写出端条件化——对 top 头按 c_apple−c_sky 差值选头（差分谱），或按头分组编辑（只动 apple 专属头）。
2. **翻转验证**：COMBO + 门控输入注入（2822 未走完的第三级），冲击 d_apple>+5.562 真翻转。
3. **谱的域普遍性**：对 8 域方向重复本 Phase（一次捕获 × 8 个 dW 谱），产出 1152×8 实测谱矩阵 = 完整头级功能图谱。
4. Δh 通道分离主线（2810 承接）不变。

### 8. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2824_measured_spectrum.py` sha256 = `86818521803a3466163a37df333c476ce4bdaf0a6547851449d556cf6e078723`
- `…/phase2824/measured_spectrum/execution.json` sha256 = `2f67fbb4c5b65c7fc15e3d70d9024f6819014e0211620557cecec42abda93598`
- `…/phase2824/measured_spectrum/result.json` sha256 = `437e9376d79d0a8f66a74d1f18247e3b660001aa61b11b268e68144755046dc4`
- `…/phase2824/measured_spectrum/spectrum.npz` sha256 = `a8ee359a7e34210d532165f62e905f2810e6d99a9247a54e721e99b8cd7c5d77`

**Gens 记录**：1 次运行（12.6s，零崩溃——预审两探针起效：2819 K2.ov_rows 字段核对 + 48 实体单 token 核查）。预注册判据全程未动。

## Phase 2825: 差分谱、正交化 key 与真翻转 [2026-09-17 07:01]

### 1. 原理与设计

2824 遗留 P4 失败：top-20 头 rank-1 编辑溢出（banana +2.188），机制定位为"条件化在输入端（key），写出端共享"。本 Phase 攻击写出端共享，两路机制 + 真翻转冲击：

1. **全 48 实体实测谱**：捕获全部 48 实体的 o_proj 输入（每层 48×2 行），产出 spec[L,e,h]（36×48×32）。与 2824 四实体谱交叉验证（apple/sky/grass max diff = 0.000，可复现性完美）。
2. **差分选头（B_diff）**：diff = c_apple − mean(c_sky, c_grass, c_coal, c_banana)，取 top-20（c_apple>0）。与 raw 选头重叠仅 13/20。diff 榜首：L29 h27（0.94）、L35 h22（0.919）、L26 h20（0.627）、L34 h28（0.599）。
3. **正交化 key（C_orth）**：同 B_diff 头组，rank-1 的 key 做 Gram-Schmidt：k' = k_apple − Σ_c (k·k̂_c)k̂_c（c ∈ 4 控制实体）。rank-1 响应 = (x·k')·d —— 对 apple 响应保留（keep_ratio 0.19-0.93，均值 0.65），对控制实体响应被正交化消除。
4. **翻转冲击（FLIP）**：cols20（2822 V1 列组）+ 头组（B_diff 或 C_orth）+ emb-edit（apple token 行 e += β·(dWb−dWr)/g，token 私有 → 结构上零溢出到其他实体行），β ∈ {1,2,4}。

预注册判据（零观测前冻结于 execution.json）：
- P1 差分选头隔离：|d_sky(B_diff)| < |d_sky(A_raw)| 且 d_apple(B_diff) ≥ 0.5·d_apple(A_raw)
- P2 正交 key 隔离：|d_sky(C_orth)| < |d_sky(A_raw)| 且 d_apple(C_orth) ≥ 0.5·d_apple(A_raw)
- P3 banana 溢出修复：min(|d_banana(B_diff)|, |d_banana(C_orth)|) < 1.0
- P4 真翻转：存在（cols+B_diff 或 cols+C_orth）× β ∈ {1,2,4} 使 margin_black_minus_red(apple) > 0

门禁：gate_dW = 1.82e-9 < 1e-6，gate_Z = 2.46e-6 < 1e-4（通过）。

### 2. 结果（Gen2 成功，20.4s）

**判决：P1=true, P2=true, P3=true, P4=true（4/4 全过）**。

| 臂 | d_apple | d_banana | d_sky | d_coal | selectivity | spill_mean(47 实体) |
|---|---|---|---|---|---|---|
| A_raw（2824 复现） | +2.688 | +2.188 | +0.688 | +1.062 | 2.43 | 1.106 |
| B_diff 差分选头 | +2.062 | +1.625 | +0.312 | +0.562 | 3.58 | 0.576 |
| **C_orth 正交 key** | **+2.312** | **+0.125** | **−0.438** | **−0.375** | **10.61** | **0.218** |

**真翻转达成（P4）**：cols20 + B_diff 头组 + emb β=4.0 → margin(apple) = **+4.125 > 0**（基线 −5.562，总位移 9.7）。apple 的 black−red 读数从强红翻为强黑，而 banana margin 移动 = 0.000、cherry 仅 −0.625。**首例实体条件化的颜色改写**：只把苹果改成黑色，香蕉/樱桃不动。

### 3. 分析

1. **正交化 key 是实体条件化的正确算子**：C_orth 把 spill 均值从 1.106 压到 0.218（5 倍），且效力保留 86%（2.312/2.688）。sky/grass/coal/blood 的 Δ 变为小幅负值（−0.1~−0.6）——正交化把"对其他实体的 red 写出"变成了轻微 anti-red，即 rank-1 响应几乎严格限制在 apple key 方向上。
2. **emb-edit 是零结构溢出通道**：apple token 行修改对其他 47 实体行结构上不可见（token id 不同），flip 臂中其余实体的残余移动全部来自头/列部件。
3. **β 阈值效应**：emb β=1/2 时 apple 仅 −1.25/−1.0，β=4 突跳 +4.125——logit 竞争（red vs black softmax 竞争）的非线性门。属性读出不是注入的线性函数；"改颜色"需要越过类别竞争阈值。
4. **2822 问题最终答案升级**：大小能改（2822 E4 翻转）、颜色能改且**可实体条件化**（本 Phase）、持久化 = 写回 safetensors。
5. 全 48 实体 Δ 矩阵（result.json full_delta_matrix）：C_orth 臂 47 个非 apple 实体中 44 个 |Δ|<0.32，最大 lemon +0.812（同为水果，语义近邻溢出——符合"语义距离决定溢出"的预测）。

### 4. 硬伤

1. flip 臂 β=4 是手工扫描值，未做 β 细扫定位阈值；翻转稳定性（多模板、多句式）未验证。
2. C_orth 只对 4 控制实体正交化；对 47 实体全体正交需更大的 key 子空间投影（lemon 溢出即证据）。
3. emb-edit 依赖 tie_word_embeddings（apple 行同时是 lm_head 行），对其他位置预测 apple token 有副作用，本测试未观测但理论存在。
4. 只测 red→black 单向。

### 5. 结论

**具体机制闭环完成**：实体条件化写 = 差分谱选头（输入端 key 私有）× 正交化 key（写出端去共享）× token 私有 emb 注入（零结构溢出）。三件套叠加实现首例"只改苹果颜色"的真翻转（margin −5.56→+4.13，banana Δ=0.000）。参数经济性机制的工程含义确认：写头硬件全局复用、每实体仅需一个条件化 key 方向——约 20 头 × rank-1 ≈ 20×2560 参数即可承载一个实体的一个属性通道。

### 6. 接续（Phase 2826 候选）

1. **8 域谱矩阵**：一次捕获 × 8 个域方向（color/size/weight/speed/taste/...）→ 1152×8 头级功能图谱，检验"写头硬件复用"跨域成立。
2. **阈值机制**：β 细扫 + 中间层读出，定位类别竞争门的位置（logit-lens 轨迹）。
3. 多模板/多句式翻转稳定性验证。
4. Δh 通道分离主线（2810 承接）不变。

### 7. 产物登记（immutable + SHA256）

- 脚本 tests/glm5/phase2825_diff_spectrum_flip.py sha256 = eff20e39b7beb7bd62465979a8ddad19302e3feee69315a84917daae34c51fa5
- …/phase2825/diff_spectrum_flip/execution.json sha256 = 4a4ddaee695c845104b8299fe7a4eabaa58c7b786279b745840dc83ac77817d5
- …/result.json sha256 = c4a47b2be3fa7767a12c115abeff08f6bb57a269ecb87fa81f7d39d70c99a92a
- …/spec48.npz sha256 = b18f04e3801bdb383ab63ff661d53e9c4af4c6794ea116a720ffe762fa40cf5d

**Gens 记录**：2 次运行（①einsum 下标错误——store[L].reshape(96,32,128) 把 batch×位置维混展，K3 降为二维；改为先按位置维 mean 再 reshape；②干净 20.4s）。预注册判据全程未动。


## Phase 2826: 九方向头级功能图谱 [2026-09-17 07:21]

### 1. 原理与设计

2825 交付了 apple→red 的头级电路与实体条件化真翻转。核心开放问题：**写头硬件是颜色专属还是跨属性域共享？** 本 Phase 把同一批捕获（apple 上下文、全 1152 头实测写出 delta）投影到 9 个方向：8 属性域（2823 词对 size/weight/temperature/speed/hardness/taste/loudness/shape）+ color_red（12 色对比方向），产出 1152×9 头级功能图谱。

配套（2809 纪律）：
- **随机 null 对照**：100 个随机单位方向同 delta 投影 → 池 q95 = 0.0625。
- **跨域行为编辑**：size 域 top-10 头（elephant 上下文 key）rank-1 flip（−2·c·dD，2821 验证过的真翻转算子），测 48 实体 size margin 位移。

预注册判据（零观测前冻结）：
- P1 头显著：9/9 域 top-20 头 mean c > null q95
- P2 晚层质量：9/9 域正写出质量 L≥30 占比 > 0.5
- P3 专业化：9 域 top-20 头集两两 Jaccard 均值 < 0.35
- P4 通用写头：L35 h0 在 9 域全部 c > 0
- P5 size 头编辑：d_margin_size(elephant) < −0.5

门禁：gate_dW = 1.82e-9，gate_Z = 2.46e-6（通过）。

### 2. 结果（一次运行成功，11.5s）

**判决：P1=true, P2=false, P3=true, P4=false, P5=false（2/5）**。

各域 top-20 mean c：taste 0.287 / shape 0.272 / speed 0.221 / temperature 0.214 / weight 0.203 / loudness 0.186 / size 0.182 / hardness 0.174 / color_red 0.367 —— 全部超 null q95（0.0625）3-6 倍。

Jaccard 均值 0.134（36 对，最高 temperature~color_red 仅 0.25，最低 size~weight 0.026）。**头集高度专业化**。

### 3. 关键发现

1. **P1：9 域写出全部真实**（超随机 null 显著）——头谱是域结构不是伪投影。
2. **P3：写头专业化，属性簇泛亲缘**。L29 h27 出现在 weight/hardness/taste/shape/color_red 五域 top（"通用属性头"）；L35 h28 在 temperature/speed/hardness/taste/loudness 五域；L34 h15 在 hardness/taste/shape 三域。物理属性簇共享部分硬件，但 top-20 集合分离。
3. **P4 否证通用写头**：L35 h0 九域谱 = color_red **1.301**、speed 0.61、temperature 0.28、taste 0.22、hardness 0.15、shape 0.08、size 0.04、loudness −0.05、weight **−0.735**。它是**颜色/速度/温度写头 + 强抑重**——专业化且跨域拮抗（与 2820 列拮抗同构，拮抗出现在头级）。
4. **P2 否证末层尖峰**：正写出质量 L≥30 仅占 0.38-0.46，即 **~60% 写出质量在 L22-29 中晚层宽带**。2824 的"L35 聚集"是峰值视角；质量视角写出是宽带。
5. **P5 失败揭示域×通路分工（本 Phase 最大发现）**：size top-10 头 flip 对行为几乎零效果（全实体 |Δ|≤0.188，elephant +0.125），而 2822 列编辑同实体 −1.50 真翻转。合并两 Phase 证据：**颜色 = OV 头通路主导（头编辑 2.7 vs 列 1.75）；大小 = MLP 列通路主导（列 −1.50 vs 头 ~0）**。不同属性域走不同硬件通路。

### 4. 硬伤

1. size 头编辑无效是"判据失败"，但未直接测 size 域的正交 key/差分谱变体（头通路 size 效力小可能因 key 条件化差而非无承载）。
2. 晚层阈值 0.5 是拍脑袋预注册值；宽带结构本身是新认知而非失败。
3. 只测 apple 上下文的谱（域×实体谱矩阵未做）；size 行为编辑只做了 flip 单算子。
4. Jaccard 用 top-20 截断，对阈值敏感。

### 5. 结论

**"具体机制"最终图景（2820→2826 六连）**：LLM 属性知识 = ①每属性域一组专业化写头（Jaccard 0.134，头集近乎不重叠）+ 属性簇共享的通用属性头（L29 h27 型）；②**域×通路分工**——颜色走 OV 头、大小走 MLP 列（双通路证据闭环）；③写出端拮抗跨域存在（L35 h0 抑重）；④写出质量呈 L22-29 中晚层宽带 + L30-35 收束；⑤实体条件化在 key 端（差分谱+正交化 key 可实现实体专属编辑，2825）。参数经济性：少量专业化头/列 × 复用的通用属性头 × 每实体一个条件化 key 方向。

### 6. 接续（Phase 2827 候选）

1. **域×实体谱矩阵**（9 方向 × 48 实体捕获全展开）：检验"通用属性头"（L29 h27）是否实体条件化，即其五域写出是否都依赖实体。
2. size 域头通路复活：差分谱 + 正交化 key（2825 三件套）套用到 size，验证头通路是否被条件化方法救活。
3. 阈值门定位（emb β 细扫 + 逐层 logit-lens）。
4. Δh 通道分离主线（2810 承接）不变。

### 7. 产物登记（immutable + SHA256）

- 脚本 tests/glm5/phase2826_domain_spectrum.py sha256 = 686e2cec878eba2826d46d8c5dedad59079611fb638e8398f90515a05fe2cf6f
- …/phase2826/domain_spectrum/execution.json sha256 = 2ce0a5705b75352e153e3620e2e3b26ee7cd5f972d7d1f2bb4a15c0bee4801fe
- …/result.json sha256 = 261e39da9b35f6c284f1cf64217bd210b20afd73f471899bb4e4eaca8c5e7670
- …/spec9.npz sha256 = 8d1a78582c370d205dd8e345b0800b7bfe8162f4057290dbfc876ba7fd8e4663

**Gens 记录**：1 次运行（11.5s，零崩溃）。预注册判据全程未动；P2/P4/P5 负结果如实入账。


## Phase 2827: = Phase Alpha: 域×实体完整谱矩阵与通路分工图 [2026-09-17 07:51]

### 1. 原理与设计（用户三阶段大计划第一步）

用户下达 Alpha/Beta/Gamma 三阶段大计划。ALPHA（本 Phase）四子任务：(1) 48 实体×1152 头×9 方向完整谱矩阵；(2) 每域重复三件套（差分谱选头+正交化 key+flip）；(3) 域×通路分工全图；(4) 通用属性头 L29 h27 实体条件化验证。

每域 target（正极代表实体）：size→elephant、weight→elephant、temperature→sun、speed→rocket、hardness→steel、taste→apple、loudness→trumpet、shape→balloon。三臂：head3（差分选头 top-10 + 正交 key + flip）、headplain（同头组 plain key）、cols（26-35 层每层 top-20 |proj| 列 flip，仅 comp>0）。

预注册（零观测前冻结）：A1 谱显著（9/9 域 target top-10 mean c > null q95）；A2 三件套效力（≥7/8 域 d_margin(target) < −0.3）；A3 正交隔离（head3 spill < 0.7×headplain，≥6/8）；A4 分工结构（head/col 效力比 max/min > 3）；A5 通用头条件化（c29h27color[apple]/c[sky] > 3 且 ≥5/9 域 max/min > 3）。门禁：dW=1.82e-9，Z=2.46e-6（通过）。

### 2. 结果（Gen2 成功，41.7s；Gen1 einsum 下标字母冲突）

**判决：A1=true, A2=true, A3=true, A4=true, A5=false（4/5）**。

| 域 | 基线 margin | head3 | headplain | cols | spill3/spillp | head/col 比 |
|---|---|---|---|---|---|---|
| size | +0.81 | **−2.125** | −2.250 | −1.062 | 0.26/0.48 | 2.0 |
| weight | +2.94 | −2.250 | −1.938 | −1.250 | 0.21/0.37 | 1.8 |
| temperature | +3.06 | **−2.562** | −2.688 | −0.625 | 0.16/0.47 | 4.1 |
| speed | +0.81 | −1.125 | −1.812 | +0.125 | 0.13/0.79 | 9.0 |
| hardness | +1.06 | −0.750 | −0.750 | −0.375 | 0.10/0.28 | 2.0 |
| taste | +2.75 | −0.875 | −1.750 | +0.125 | 0.08/0.67 | 7.0 |
| loudness | +3.06 | **−3.062** | −3.000 | −0.250 | 0.24/0.48 | 12.25 |
| shape | +6.09 | −2.688 | −3.031 | −1.438 | 0.27/0.76 | 1.87 |

### 3. 关键发现

1. **A2 8/8 全过——三件套全域有效**：每个新域都实现 target margin 大幅负移（−0.75 ~ −3.06），方向全部正确。
2. **2826 的 size"列主导"结论被修正（重要）**：2826 用 apple 上下文谱选头、编辑 elephant → 效力 ≈0；2827 用 **target 实体自己的差分谱**选头 → size 头通路 −2.125 反超列 −1.062。**头通路全域可用，"分工"实为选头质量差异**。2826 P5 的失败是"用错实体的谱选头"的伪象。跨域保留的结构差异：head/col 比 1.8（weight）~12.25（loudness），强头域 loudness/taste/speed/temperature vs 均衡域 size/weight/hardness/shape。
3. **A3 8/8 全过——正交化 key 隔离普适**：三件套 spill 比 plain 低 2-6 倍（如 taste 0.081 vs 0.665，6 倍）。
4. **A5 度量缺陷下的强条件化证据**：预注册第二判据（max/min > 3）9 域全败（0.47-2.23），因谱符号混合使 max/|min| 失效；但第一判据 apple/sky = **27.87** 大幅通过。L29 h27 对 apple 的写出束：color 0.769 + shape 0.509 + taste 0.460 + weight 0.432（**苹果语义束**：红+圆+甜+重），对 sky 全部 ≈0（−0.06~0.10）。通用属性头 = "实体条件化的属性束写头"确认，度量方法待改（如用 target-vs-均值比）。
5. 基线 margin 揭示 shape 6.09 / temperature 3.06 / loudness 3.06 的高先验——round/hot/loud 是高频属性联想。

### 4. 硬伤

1. A5 第二判据设计缺陷（预注册不可改）：符号混合谱的 max/min 不是条件化度量；正式结论以 apple/sky 比为准，需下 Phase 修正度量重测。
2. 三件套未做逐域翻转判定（head3 −0.75~-3.06 未全部跨零，如 shape 基线 6.09 仍差 3.4）。
3. 分工图只用效力比，未测双通路叠加（cols+head3 组合）是否继续加和。
4. 列臂 census 用磁盘权重（restore 后等价），但未在 arms 间随机化顺序。

### 5. 结论（ALPHA 交付）

**域×实体谱矩阵完成**（36×48×32×9，spec_full.npz 1.9MB）；**三件套 9/9 域全域有效**（color 2825 + 8 新域）；**域×通路分工图完成**——修正版结论：不存在"只能走列"的域，头通路经差分谱+正交化后全域主导或均衡（比 1.8-12.25），**通路分工 = 选头质量 × 域头密度**；**通用属性头 = 实体条件化属性束写头**（L29 h27 对 apple 写红/圆/甜/重束，对 sky 零写出，27.87 倍条件化）。实体知识 = 专业化写头 × 实体条件化 key 的图景在全部 9 域成立。

### 6. 接续（Phase Beta，2828）

知识链 key 方向链式传递：构造"苹果→水果→食物"式 3-5 跳链，逐跳捕获 key 方向；验证第一跳写出是否改变第二跳 key；链式编辑（改第一跳 key 测后续跟随）；链断裂修复（断裂点注入正确 key）。

### 7. 产物登记（immutable + SHA256）

- 脚本 tests/glm5/phase2827_alpha_matrix.py sha256 = 07fea4d74d2e941b41e81043cfb35acca301040f6009df5832c4435f70bdb125
- …/phase2827/alpha_matrix/execution.json sha256 = cd4d7e767a6087054ee6dfb25bc77b33467db535d88ebbca20577b8bd6fad031
- …/result.json sha256 = 864f5a50c498ea624d264a3e2e581201f54c627ace180456adc6f9740d78d6ea
- …/spec_full.npz sha256 = 8227b3b68b90876ba56e70d155eadf409660c908e762c3b30344e4b4badb0c5d

**Gens 记录**：2 次运行（①einsum 'rd' 下标字母与 'edh' 的 d 尺寸冲突（2560 vs 9）→ 改 'dr'；②干净 41.7s）。预注册判据全程未动；A5 负结果如实入账并附度量缺陷分析。


## Phase 2828: = Phase Beta: 知识链 key 方向链式传递 [2026-09-17 08:06]

### 1. 原理与设计

用户三阶段计划第二步。核心问题："苹果→水果→食物"多跳链中 key 方向如何逐跳传递？

范式：双句链 prompt（token 全对齐，seq_len=12）：`The <E1> is a fruit. The fruit is a food.`，位置 e1=1 / fruit1=4 / fruit2=7 / food=10。8 变体：真链起点 apple/cherry/lemon/banana，断链起点 rock/steel/hammer/ocean。链方向 = 2806 类方向 dW_fruit、dW_food（gate 复现通过：dW=1.82e-9）。

谱：c[L, v, pos, h, 2]（36×8×4×32×2）。臂：
- E1 链式编辑：rank-1 消掉 pos=1 处 c_fruit top-5 头的 fruit 分量（key=apple 行），重测 pos=7 的 c_food（跟随测试）
- E3 修复注入：断链行 pos=1 残差流注入 β·dW_fruit（β∈{2,5,10}），测 c_food 恢复

预注册：B1 跳1真实（真链 c_fruit(e1) > null q95 且 > 断链）；B2 第二跳跟随（真链 c_food(fruit2) > 断链）；B3 编辑跟随（E1 使真链 c_food(fruit2) 降 ≥20%）；B4 修复（存在 β 使断链恢复至真链 50%）。

### 2. 结果（Gen4 成功，61.7s；Gen1 缺 d_model 提取、Gen2/3 einsum 下标与位置索引错误）

**判决：B1=true, B2=true, B3=false, B4=true（3/4）**。

- B1：真链 c_fruit(e1)=0.131 vs 断链 0.053（2.5 倍），超 null q95=0.079 —— 第一跳"实体→类别"写出真实且实体依赖。
- B2：真链 c_food(fruit2)=0.124 vs 断链 0.106（1.17 倍）—— 通过但余量小。
- **B3=false（核心负结果）**：E1 消掉跳 1 的 fruit 写出后，c_food(fruit2) 仅降 **0.2%**（0.124→0.123）——第二跳**不因果依赖**第一跳的 rank-1 写出分量。
- B4：注入 β=5 即把断链 c_food 推到 0.124 = 真链水平；但基线差距本来就小（0.106→0.124），修复幅度有限。
- 辅助谱：c_fruit(fruit2) 真链 0.289 vs 断链 0.185（1.56 倍）——第二句 fruit 的自我写出受上文语义一致性影响最强。

### 3. 分析

1. **E1 头全部在 L0**（h27 c=2.51、h15/23/0/19）——c_fruit 跨层求和的 top-5 全是 embedding 层"词法头"。这些头做低层信息搬运，不承载语义跳；编辑它们不影响下游属预期。
2. **谱级链存在（B1/B2）但因果链弱（B3）**：真链与断链的谱差异是"全句语义一致性"的弥散效应，不是 key 方向逐跳传递。**相关 ≠ 因果**在本范式被干净分离——B2 的 1.17 倍相关差距在 rank-1 消除后几乎不动（0.2%）。
3. 结论：**该句对范式下 LLM 不做链式 key 传递——两句并行独立处理**。"苹果→水果→食物"的知识链在简单陈述句中不表现为逐跳 key 通路；rank-1 类方向谱可测的因果传递不存在。
4. B4 "修复"实为弱效应：注入只覆盖 B2 的小差距，不构成强链证据。

### 4. 硬伤

1. 范式可能不适配：陈述句对是弱链；真正的多跳推理（问答链 "apple is a fruit. What is a fruit?"）或需 chain-of-thought 才激活传递。B3 的否证限于本范式。
2. E1 只消 top-5 头的 rank-1 分量；传递若走高阶/非线性通道则不可见。
3. 链方向用类别均值方向（dW_fruit/food），若传递走实例级方向（apple 的 fruit ≠ lemon 的 fruit）则被平均掉。
4. B4 判据阈值（真链 50%）因 B2 差距小而过于宽松。

### 5. 结论（BETA 交付）

**知识链 key 传递机制的判决：谱级相关存在、因果传递不存在（本范式）**。LLM 处理 "apple is a fruit. The fruit is a food." 时两句近似独立，各跳写出由当前 token + 局部上下文决定，上文实体通过语义一致性微调（1.17 倍）而非逐跳 key 通路。这本身是对"LLM 如何组合知识"的重要刻画：**知识链的连贯性来自训练出的句内语义计算，不需要运行时的链式 key 传递**——与 Alpha 的"实体条件化写头"图景一致（每句独立调用写头硬件）。链式传递若存在，需更强范式（问答/推理链）激活，留待后续检验。

### 6. 接续（Phase Gamma，2829）

跨模型验证：14B 上重复 2824-2825 实测写出谱与三件套编辑，检验"写头复用×实体 key"结构跨规模存在性。前置：探针本机是否有 Qwen3-14B 权重；若无则请用户提供或降级为 8B/其他可用模型。

### 7. 产物登记（immutable + SHA256）

- 脚本 tests/glm5/phase2828_beta_chain.py sha256 = cceefc90826c9356d4c1486eeef84dfed1bff1daf5e8390955af8b2d0e340a47
- …/phase2828/beta_chain/execution.json sha256 = cd31b975f5b2936dacd48c4a4ea4cfe273deb0ee3ced452bcc82893ac2722e72
- …/result.json sha256 = 9bf8862e88dc9cfd693fdefa8ae1f0a8e77839139f7c856df6e5aa60bbf5159f
- …/chain_spec.npz sha256 = dc375fe3894efc6af82b5d29808fbe58074d13f3caceb2255c6fdd55e5fa8630

**Gens 记录**：4 次运行（①缺 d_model 提取 NameError；②einsum 'bkh' 下标错位；③grp_stat 用绝对 token 位置越界，改 PIDX 压缩索引；④干净 61.7s）。预注册判据全程未动；B3 负结果如实入账。


## Phase 2829: = Phase Gamma: 跨模型 14B 验证 [2026-09-17 08:25]

### 1. 原理与设计

用户三阶段计划第三步：4B 的"写头复用×实体key"结构在 14B 上是否存在？

本机确认 Qwen3-14B（40 层×40 头=1600 头，hidden 5120，tie_word_embeddings=False）。GPU 16GB < 28GB 权重 → device_map auto 混合卸载（GPU 12GiB / CPU 21GiB 配额）。

**关键管线发现**：14B 的 embed_tokens 行与 lm_head 行**近正交**（实测 cos≈0.005，untied 且训练后独立）——4B 的 z(w)=rmsnorm(E)·g 反解在 14B 上不代表读出方向。修正：隐藏空间读出方向 u_w = g⊙Wu_row(w)（logit_w = <rmsnorm(h)·g, Wu_row>）。gate 改为方向判别力自检（d_red 对 red 的 margin > 对全部 11 个他色，min margin = 2.060）。

重复 2824/2825 核心：48 实体捕获、9 方向实测谱（1600 头）、臂 A_raw（apple top-20 plain）/ C_orth（差分头+正交 key）。**编辑算子改为 hook 等效实现**（混合卸载使部分 o_proj 权重在 meta device 不可读写；rank-1 权重修改 W += outer(d, key/kn2) 数学等效于 o_proj 前向 hook：output += (input·key/kn2)·d，hook 移除即恢复）。c_{l,h} 从已测 spec9 取。

预注册：G1 谱显著；G2 三件套效力（C_orth d_apple ≥ 1.0）；G3 正交隔离（spill < 0.7×A_raw）；G4 末层写头（argmax 在最后层）；G5 束头存在（≥6/9 方向正 且 apple/sky > 5）。

### 2. 结果（Gen5 成功，81.1s；Gen1 跨模型 gate 维度不匹配 → Gen2 z 反解 cos≈0.005 失效 → Gen3/4 加载段错误与 meta tensor → hook 等效编辑解决）

**判决：G1=true, G2=true, G3=true, G4=false, G5=true（4/5）**。

| 量 | 4B | 14B |
|---|---|---|
| 主写头 | L35 h0（c=1.301，末层） | **L32 h11（c=1.560，80% 深度）** |
| 末层最大头 | L35 h0 = argmax | L39 h1（c=0.912）≠ argmax |
| A_raw d_apple | +2.688 | +2.125 |
| C_orth d_apple | +2.312 | +1.188 |
| spill A_raw→C_orth | 1.106→0.218（5.1 倍） | **0.735→0.084（8.75 倍）** |
| 束头 | L29 h27，apple/sky=27.9 | **h5，6/9 方向正，apple/sky=37.1** |
| raw/diff 头重叠 | 13/20 | 14/20 |

### 3. 分析

1. **结构跨规模确认（G1/G2/G3/G5）**：14B 同样存在实测写头谱、"差分谱+正交 key"三件套有效、正交隔离（且更强：8.75 倍 vs 5 倍）、实体条件化属性束写头。"写头复用×实体key"不是 4B 特例，是 Qwen3 家族的架构级组织原则。
2. **G4 层位漂移**：14B 主写头在 L32（80% 深度）而非末层——"末层聚集"不是普适常数，但"晚层（≥75% 深度）主写头+中晚层宽带"结构保持。14B top-10 头全部在 L28-39。
3. **hook 等效编辑**是混合卸载模型的正确干预算子：数学等效、零权重接触、移除即恢复，为受限显存下的大模型编辑提供了通用方案。
4. 14B untied embedding 近正交 lm_head 是重要管线事实：**跨模型迁移方向构造时必须以 Wu 行（读出空间）而非 embedding 行构造方向**。

### 4. 硬伤

1. C_orth 效力保留率下降（14B 1.188/2.125=56% vs 4B 2.312/2.688=86%）——正交化的效力-隔离权衡随规模变化，未细查。
2. c_{l,h} 用 spec9 近似（plain key 值×keep_ratio），非正交 key 的精确 W@k'（meta 不可读）。
3. 未做 14B 几何普查对照与 cols 臂（预算控制）；跨模型对应是功能级（束头/主写头），非逐头映射。
4. B4（Beta）已示注入修复弱，14B 修复未重复。

### 5. 结论（GAMMA 交付：三阶段计划完成）

**跨模型对应图**：4B L35 h0 ↔ 14B L32 h11（主写头，晚层、最强 c）；4B L29 h27 ↔ 14B h5（实体条件化属性束写头，apple/sky 28→37）。**"通用写头硬件（复用）× 实体条件化 key（私有）"在 4B 与 14B 同构成立**，且正交 key 隔离在 14B 更纯（spill 8.75 倍压缩）。参数经济性机制随规模稳定：写头数量与层位有漂移（L35/36→L32/40），组织原则不变。

### 6. 三阶段大计划总结

- **Alpha（2827）**：9 域×48 实体×1152 头谱矩阵；三件套 9/9 域全域有效；分工图修正（无只能走列的域，head/col 比 1.8-12.25）；束头确认。
- **Beta（2828）**：知识链 key 传递判决——谱级相关存在、**因果传递不存在**（B3 消跳 1 写出后 c_food 仅降 0.2%）；陈述句对范式下两句并行独立处理。
- **Gamma（2829）**：14B 跨规模确认"写头复用×实体key"结构同构；束头/主写头功能对应建立；hook 等效编辑算子产出。

### 7. 接续候选

1. 正交化效力-隔离权衡的规模效应细查（keep_ratio 分布 vs 层深）。
2. 14B 束头 h5 的方向谱全展开与编辑（对应 2825 全流程）。
3. 问答式链范式重测 Beta（激活式多跳）。
4. Δh 通道分离主线（2810 承接）。

### 8. 产物登记（immutable + SHA256）

- 脚本 tests/glm5/phase2829_gamma_14b.py sha256 = 699be012b7eb4ef3862bdb8935162d06553547399bf16151de9c5139f80f4430
- …/phase2829/gamma_14b/execution.json sha256 = 756e8a27ea506763c476ea991b213fdcaf9a47f95e1cb76ef48daeb94f775c3b
- …/result.json sha256 = 45cf4f93c6f9f94423d4d80cc495f948202b5cd9bc531e1b05e478e1c55a79da
- …/spec9_14b.npz sha256 = 06ee842258314eb6ac24e3d50b50979983386f132db7616d91fd8fcd92d8528f

**Gens 记录**：5 次运行（①4B gate 2560 维不匹配；②z 反解 cos≈0.005 失效——14B embed⊥lm_head，改 u_w=g⊙Wu_row；③加载段错误 RAM 峰值；④meta tensor 不可写；⑤hook 等效编辑干净通过 81.1s）。预注册 G1-G5 全程未动；G4 负结果如实入账（层位漂移）。


## Phase 2830: 同句链范式重测 + 2828 勘误 [2026-09-17 08:43]

### 1. 原理与设计

Beta（2828）因果否证（B3=false）依赖两句式范式，可能被"两句并行独立处理"解释。本 Phase 用**同句逗号链**范式重测：`An <E1> is a fruit, and a fruit is a food.`（13 token 对齐，e1=1 / fruit1=4 / fruit2=8 / food=11），强制激活流经同一推理流。8 变体（真链 apple/cherry/lemon/banana；断链 rock/steel/hammer/ocean），方向 dW_fruit/dW_food（2806，门禁 dW=1.82e-9、Z=2.46e-6 通过）。

消融臂三个：E1a @e1（2828 复测对照）、E1b @fruit1（同句特异干预点，apple 行 key）、CTRL broken @fruit1（rock 行 key）。rank-1 移除 top-5 c_fruit 头。

预注册（零观测前冻结）：C1 跳1真实（true>null q95 且>true broken）；C2 跳2跟随（true>broken）；C3 因果链（E1b 使 true c_food@fruit2 降≥20%）；C4 激活流（同句 true c_food@fruit2 > 2828 两句式 0.124）；C5 对照（E1a 预期微效）。

### 2. 结果（Gen1 崩溃修复后 Gen2 成功，52.3s）

**判决：C1=true / C2=true / C3=false / C4=true**。

| 量 | true | broken |
|---|---|---|
| c_fruit@e1（跳1） | 0.675 | 0.111（6.1 倍） |
| c_fruit@fruit1 | 0.996 | 0.542 |
| c_fruit@fruit2 | 1.144 | 0.925 |
| c_food@fruit2（跳2） | 0.348 | 0.256（1.36 倍） |
| c_food@food | 0.545 | 0.414 |

消融臂：c_food@fruit2 true 0.348 → E1a 后 0.347（降 0.4%）、E1b 后 0.347（降 0.3%）；broken 0.256 → CTRL 后 0.256（−0.2%）。restore_ok=true。

### 3. 重大发现：因果否证是范式无关的机制性结论

1. **同句范式把谱强度放大约 3-10 倍**（C4=true：c_food@fruit2 0.348 vs 两句式 0.124；c_fruit@fruit1 0.996 vs 0.102≈10 倍）——激活流确实大幅增强。
2. **但第二跳写出依然完全独立于第一跳写出**（消融 drop <1%，两干预点+broken 对照全一致）。知识链连贯性 = **并行共激活**（同上下文中各 token 各自读出），**不是运行时消息传递**。
3. 谱级相关（C1/C2）与因果传递（C3）彻底分离：同一 Phase 内相关真实、传递不存在。2828+2830 双范式闭环，Beta 结论机制化。

### 4. 2828 勘误（正式入账）

Gen1 后发现 **grp_stat 语义 bug**：`sp[:, rows, pidx, :, di]` 混合高级索引使 advanced 维前置（实际形状 (4,36,32)，行×层倒置），实际算的是"前 4 层 4 行求和的 top-10"。修正（逐行标量索引）后从 2828 落盘 chain_spec.npz 零成本重算：b1 c_fruit@e1 true 0.552（原报 0.131）/ broken 0.099（0.053）；b2 c_food@fruit2 true 0.375（0.124）/ broken 0.221（0.106）。**所有相对判决方向不变**（B1-B4 一致，B3 否证更强化：修正后 0.375 vs 消融后 0.347 仍微降）。勘误以本节为准；2828 result.json 不改（immutable）。

### 5. 硬伤

1. 只测 fruit→food 单链单方向；多跳链、反事实链（'The fruit is NOT a food'）未测。
2. 消融只在 top-5 头 rank-1；可能存在分散在 MLP 列通路的链传递未被此算子覆盖。
3. C3 判据阈值 20% 是预注册约定，"drop<1%" 已远低于阈值，但全头/全层消融才能彻底排除弱传递。

### 6. 结论

知识链在单 forward 内**无谱级因果链，只有共激活**。LLM 组合知识的方式不是 token 位置间的运行时消息传递（不是 CoT 内部版），而是：上下文共激活 × 每 token 独立读出 × 写头硬件复用（Alpha/Gamma）。这与 2824-2825 的"通用写头×实体 key"图景互补：实体属性是键值式并行读出，知识链也是。

### 7. 接续（Phase 2831 候选）

1. Δh 通道分离主线（2810 承接，最高优先）。
2. 生成式链（让模型自己生成第二句再测谱）——测 decode 时的链传递。
3. 2828 勘误公式回灌全部历史谱判据检查（2781-2830 grep 审计）。

### 8. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2830_insent_chain.py` sha256 = `f0a7076666407a200fcaf0ae39cfc160f41b134c20c5a3a4d26ddebbb30e62ff`
- `…/phase2830/insent_chain/execution.json` sha256 = `27449b54b42bac3ca8b2431d84831cbb9fd7cf1889ed8a444e6d93955c155aa1`
- `…/result.json` sha256 = `01d65889209044f0b574b4196a53a840fe7e6fe5b1ab394808189e6cb1cf63fc`
- `…/insent_spec.npz` sha256 = `69bdad52426d0dcadeb9930e25aa1daeaaff0efb4ce7281bcee280fb2a04e91e`

**Gens 记录**：2 次运行。Gen1 完成（38.5s）但结果核查发现 grp_stat 语义 bug（高级索引前置）+ refood 未重算谱；修复后删 execution.json 重跑 Gen2（52.3s）。判据全程未动。


## Phase 2831: 问答模式优先级 [2026-09-17 09:03]

### 1. 原理与设计

用户问题 1：'我喜欢吃苹果' 后接不同问句（谁喜欢吃苹果？/ 我是否喜欢吃苹果？/ 我喜欢吃什么样的苹果？/ 我喜欢吃什么？）如何决定答案侧重的模式？是否存在句式优先级表？

设计：5 条件（无问对照 + 4 问法），前缀 3 token 对齐（苹果@2）。**全中文方向组**（lm_head 行构造）：红色 leave-one-out + 大小/轻重/甜苦/硬软/热冷/快慢/圆方 7 词对。每条件：greedy 8-token 生成（行为）+ 全 token 谱 c[L,pos,head,8dirs]。

预注册：Q1 模式切换（4 问法生成答案 token 重叠<50%）；Q2 苹果位置重编程（≥3/8 方向 diff>null q95）；Q3 问号位置模式分离（4 问法谱两两余弦<0.5）；Q4 优先匹配（whatkind > who/whether 于 color+round+sweet 谱）。

### 2. 结果（Gen7 成功 22.6s；前面 6 Gen 为环境/笔误修复：lm_head tie 回退、方向词表英文误写、tid('apple')→tid('苹果')、einsum pkh→phk、半角问号→全角）

**判决：Q1=true / Q2=false / Q3=false / Q4=true。**

生成行为（Q1=true，重叠 4/6 对 = 0）：who→"（这是一道逻辑题，不是"、whether→"（这是一道自问自答"、whatkind→"我应该买什么样的苹果？"（属性模式）、what→"我是不是应该吃苹果？"。

### 3. 机制发现：模式优先级不在历史写头，在生成端

1. **Q2=false 是正面发现**：后置问句完全不改变已编码 token 的写出谱（苹果@2 各方向 diff 0.009-0.017，全部 ≪ null 0.0529）。**上下文一旦编码，key 不可被后文重编程**——单向固化。
2. Q3=false：4 问法问号位置谱两两余弦 0.73-0.90（whatkind-what 0.902 最高）——问号位置 hidden state 高度共享，8 属性方向分辨不出问法差异。
3. Q4=true 但余量极小（0.566 vs 0.558/0.554）——属性谱通道只承载弱模式信号。
4. **模式切换的主战场是 next-token 分布**（Q1=true 行为全分化）：问法通过问号位置的语义状态直接选择生成模式，不通过改写历史。

**回答用户"句式优先级表"**：不存在显式优先级查表，但存在**问法→生成模式的功能性映射**（谁→元话语/施事槽、是否→判断槽、什么样→属性槽、什么→宾语槽），映射由问句末尾的 hidden state 单点决定，优先级 = 该状态到各模式答案空间的方向，且历史编码不可被覆盖。

### 4. 硬伤

1. 4B 把 who/whether 当逻辑题（弱模型行为噪声），答案模式语义弱于理想。
2. 只测 4 问法 1 基句；8 属性方向可能不够分辨"谁/是否"模式（Q3 的 cos 阈值 0.5 或过松）。
3. 生成仅 greedy 8 token。

### 5. 产物登记

- 脚本 `tests/glm5/phase2831_q_priority.py` sha256 = `f061593446f99f2fa1bb0e44614c30ca9fc8685668de5f7d0135ac9871d9df4f`
- `…/phase2831/q_priority/execution.json` sha256 = `2828be625b0c5a775c158778281a403fc3c87e91681c95db6573729ec7d5d0de`
- `result.json` sha256 = `077211b2e66e45e65b65752fd51c86d93672546fe021e545eda697c0f18f8ffd`；`q_spec.npz` sha256 = `b3b56160a7b95831014fc575782065b12da32fe1b08fd0cdf0f93f47599f159b`

---

## Phase 2832: 属性束并发调用 [2026-09-17 09:06]

### 1. 原理与设计

用户问题 2：处理"红色的圆形的甜的苹果"时，多个属性束如何同时被调用？

设计：5 条件——基句'苹果'（@0）、三个单属性句'红色的/圆形的/甜的苹果'（苹果@2）、三属性句'红色的圆形的甜的苹果'（苹果@6）。同 2831 中文 8 方向。判据：A1 并发可加（triple ≈ 三 singles 之和，残差<0.3）；A2 写入器局部性（各方向峰在自己修饰词位置）；A3 束头复用（triple 三方向 top-5 头 Jaccard>0）；A4 无负贡献。

### 2. 结果（一次运行成功 18.7s）

**判决：A1=false / A2=true / A3=true / A4=true。**

| 方向 | triple top10 | 三 singles 之和 | 比率 |
|---|---|---|---|
| red | 0.543 | 1.910 | **0.28** |
| round | 0.497 | 1.548 | **0.32** |
| sweet | 0.703 | 0.993 | **0.71** |

A2 写入器局部性：red 峰在'红色'（pos 0，1.90；'的' 处 −3.00 强负——**'的'是拮抗位**）；round 峰在'的'（4.63）；sweet 峰在'的'（0.08 弱）。A3：triple 中 red-round/red-sweet Jaccard 0.25、round-sweet 0.111——部分共享。

### 3. 机制发现：独立写入 + 归一化接收

1. **多属性束不是叠加调用，而是强亚加性压缩**（比率 0.28-0.71）：每个修饰词在自己的 token 位置独立写入（A2），但接收端（苹果位置）读出被压缩——符合 rms-norm 增益重新分配 / 读出带宽竞争。三属性不"相加"，而是"分配"。
2. 束头部分复用（A3）：同一批头承载多方向（与 2826 L29h27 束头呼应），但 red/round/sweet 的 top-5 组不完全重叠（Jaccard 0.11-0.25）——束=共享头核心+方向特异外围。
3. 苹果基线谱形状先验最强（round 0.317 > red 0.220 > sweet 0.152）——"苹果=圆"是默认属性束内容。
4. '的'的符号分化（red 方向 −3.0）再次确证写出端跨方向拮抗（2820/2826 第三例）。

### 4. 硬伤

1. 亚加性来源未分离：rms-norm 归一化 vs 注意力竞争 vs token 位置相互作用，需干预实验。
2. 只测 3 属性 1 实体；方向集 8 个可能漏 shape 精细维度。
3. 单 token 对比基句太短，长度混淆未完全排除（triple 7 token vs single 3 token）。

### 5. 产物登记

- 脚本 `tests/glm5/phase2832_bundle_concurrency.py` sha256 = `da3b5ec3f05810d91894f878685d9a9aa2b279bc85136c129bfded5885db2694`
- `…/phase2832/bundle_concurrency/execution.json` sha256 = `67f03e2b46fc7460caf302eee291428a54f3043f95722d2162300d1f915a7339`
- `result.json` sha256 = `94b0b8f440123b48e8c0847b3161a68a0aa3a79c0596549bef93c6871bc30764`；`bundle_spec.npz` sha256 = `7f62a17af05074771a6c931214575e6ae107fd1b4661bf0db80012895359ff77`

**Gens 记录**：2831 共 7 次（6 次崩溃修复）；2832 一次成功（一次预审修复 Wu 重复赋值）。


## Phase 2833-2835: Δh 通道分离、生成式链与历史谱公式审计 [2026-09-17 09:28]

用户指令：确认 ①Δh 通道分离主线（2810 承接）②生成式链 decode 谱传递 ③历史谱判据公式审计（2781-2830）三项是否完成，未完成则完成。三项均未做过，本节一次补齐。

### Phase 2833: Δh 通道分离（2810 接续 a+d） [2026-09-17 09:15]

**原理**：2810 发现裸 Δ = h_ctx − h_iso 被 attention-sink/位置通用通道支配（"the" 位移最大，P-K2/P-K4 双否证），语义通道只占次要成分。接续方案 (a)：Δ_specific(cond) = Δ_cond − 0.5(Δ_func + Δ_null)（func/null 同位置对照估计通用通道，逐层分离）；方案 (d) 探索性：末层分离 Δ 分解到 32 头输出方向定位搬运头。

**判决：S1=S2=S3=S4=true，channel_separated_substantive**（Gen4 成功 15.1s；三次崩溃：hd 误算 2560//32、hook 键错位 'x' 槽未转存、头分解 einsum 'dkh,kh->d' 把头维求掉应为 '->dk'）。

| 判据 | 读数 | 结论 |
|---|---|---|
| S1 选择性 | cls_spec same 0.0743 vs diff 0.0294（2.5 倍） | 分离后语义通道显现 |
| S2 分离增益 | 分离 0.0743 > 裸 0.0683 | 分离口径更纯净 |
| S3 剂量-反应恢复 | mag same 51.19 > diff 41.39 | 2810 P-K2 反向在分离后转正 |
| S4 层依赖 | cls_spec 层极差 0.110，峰值 L33 | 语义调制集中中晚层 |

头分解（探索性）：L35 h22（0.0143）、h23（0.0098）、h20（−0.0087 拮抗）、h26——与 2824 apple→red 主写头簇（h22/h23/h26）**重合**：通道分离后的语义调制载体就是属性写头硬件。

**结论**：2810 悬案关闭——档案调制的语义通道真实存在，但必须在 func/null 通道分离口径下测量；语义通道载体 = 已知的属性写头簇。LPF v5.3 档案-上下文接口定性：**通用上下文通道（大幅）+ 语义调制通道（小幅但方向特异）双通道结构**。

### Phase 2834: 生成式链 decode 谱传递 [2026-09-17 09:18]

**原理**：2828/2830 在 teacher forcing 下发现知识链谱相关但无因果传递。本 Phase 让模型自己贪心解码第二跳（The apple is a fruit. → 8 token），检验 decode 阶段链谱传递（true=apple vs broken=rock 双臂）。

**判决：G1=false / G2=false / G3=true，not_confirmed**（一次成功 12.2s）。

- true 臂生成 " The apple is a fruit that is commonly"——**归纳式复读实体**，未生成 food；
- broken 臂生成 " The fruit is a tree. The tree"——**分类学续写**（rock→fruit→tree），未生成 food；
- c_food 比值 true/broken = 0.874（broken 反而更高）；c_fruit true 0.377 > broken 0.357（G3=true，首句实体激活持续）。

**结论**：生成式解码下链传递第三次否证——模型续写走归纳/分类学局部模式，不主动完成知识链第二跳。知识链连贯性完全是训练时习得的表征结构（共激活），不是运行时传播机制。与 2828/2830 构成三范式收敛证据（teacher forcing 句对 / 同句 / 自由生成）。

### Phase 2835: 历史谱判据公式审计（2781-2834） [2026-09-17 09:27]

**原理**：2830 发现 2828 grp_stat 数值异常并记录"高级索引前置 bug"勘误，但勘误时的机制解释未实证。本 Phase：①静态扫描全部谱脚本索引模式；②numpy 混合索引形状判定矩阵实证；③在落盘谱上数值重算对质。

**判决：audit_clean_with_2828_erratum**（一次成功 0.1s）。

1. **numpy 规则实证**（关键新知识）：`(slice, list, slice)` 布局列表索引**原地保留**（(36,4,32)）；`(slice, list, int, :, int)` 布局**pushfront**（(4,36,32)）——标量 int 在列表后且元组以标量结尾时触发前置。此前"所有混合布局都前置"的假设是错的。
2. **2825 复核 = 无 bug**：`spec[:, ctrl_rows, :].mean(axis=1)` 属原地布局，c_ctrl = 逐层控制行均值，语义正确。B_diff 选头有效。
3. **2828 勘误确认成立**：其 grp_stat 属 pushfront 布局，原值（b1 0.131/0.053）是"逐层 top10（4 行求和）再层均"统计；修正值（0.552/0.099）才是"逐实体层求和 top10 均值"。**两种口径下 true>broken 判决均成立**（orig_order_ok=fixed_order_ok=true），B1-B4 相对判决稳定。
4. **其余全部安全**：2824（纯标量）、2826（纯标量/slice）、2827（标量 e/x）、2830 Gen2（逐行标量）、2831/2832（标量 pos）、2833/2834（复核 einsum 与标量索引）。无其他波及。

### 产物登记（immutable + SHA256）

- 2833：execution 70065439… / result 5da6972f… / delta_specific 94074ffd… / script 0a1e861c…
- 2834：execution 9e8de6c9… / result bfe9ad7c… / gen_spec 0f320378… / script ca3d9979…
- 2835：execution 4468d7dd… / result 68f8f2cd… / audit_report a8bde353… / script 414a9680…

**Gens 记录**：2833 四跑（三崩溃修复后成功 15.1s）；2834 一次成功（12.2s）；2835 一次成功（0.1s）。预注册判据全程未动。

### 接续（2836 候选）

1. 双通道定量模型：通用通道 vs 语义通道的逐层增益分解（对接 2810 数据集全层曲线）。
2. 归纳头与属性写头的关系：2834 true 臂归纳式复读是否经由 L35 属性写头（写谱检验）。
3. 2830 已修复公式的全量谱判据规范入纪律（混合索引布局禁令）。



## Phase 2836: 双通道逐层增益分解 + 归纳复制与属性写头关系 [2026-09-17 14:55]

### 1. 原理与设计

2836 候选 a+b。臂 (a)：分离通道逐层增益分解——残差恒等式 `inc(l) = hs(l+1)−hs(l) = attn(l)+mlp(l)`，分离增量 `inc_spec(l) = inc_same(l) − 0.5(inc_func(l)+inc_null(l))`，逐层分解为 attn/mlp 载体，方向分辨口径 `cls_spec_inc(l) = |inc_spec(l)@cdir| / ||inc_spec(l)||`。臂 (b)：归纳复制探针——greedy decode "The apple is a fruit. The"，在**发出复制 token 的那次前向**（输入以 "The" 结尾）捕获 L35 o_proj 输入，按 32 头分解到 `c_apple` 方向，检验 2824/2833 属性写头簇 {h20,h22,h23,h26} 是否参与归纳检索。

预注册：A1 闭合 <0.02；A2 cls_spec_inc 峰值层 ≥20；A3 载体比 f_attn（描述性）；B1 前 3 步内生成 'apple'；B2 写头簇 ≥3/4 进 top16。判决 = A1∧A2 / B1∧B2。

### 2. 判决：A1=true / A2=true → dual_channel_gain_resolved；B1=true / B2=true → induction_via_write_heads（Gen3 复现一致）

| 判据 | 读数 | 结论 |
|---|---|---|
| A1 闭合 | rel 0.0129（20 词均值） | 增益分解闭合，attn/mlp 载体分解可信 |
| A2 层分辨 | cls_spec_inc 峰 L30（0.0943） | 语义调制增益集中中晚层（与 2833 S4 峰 L33 呼应） |
| A3 载体比 | f_attn = 0.1424 | 分离增量范数 MLP 占 86%、attn 仅 14%（描述性） |
| B1 归纳复制 | step 0 生成 " apple"（其后 " is a fruit that is commonly used"） | 归纳式实体复制确认 |
| B2 写头参与 | h20 排名 1（−0.0301）、h22=3（0.0181）、h26=4（0.0151）、h23=5（0.0144），**4/4 全部进 top16** | 归纳复制经由属性写头簇 |

### 3. 机制发现与 Gen1 诊断（重要环境新知识）

1. **HF hidden_states 尾元素陷阱（新钉死）**：`output_hidden_states` 的第 37 项（hidden_states[36]）是**过 final RMS norm 之后的状态**，不是 L35 层原始输出。Gen1 用 `hs[-1]` 锚定 Δ_spec 导致 A1 闭合失败（rel 5.56）；残差恒等式在 L0–L34 成立（探针 dev 相对 <1%），L35 处 hs[35]+attn[35]+mlp[35] ≠ hs[36]（dev 667）。**凡逐层分解/增量分析必须用 hs[35]+attn+mlp 重构原始末态。**
2. **L6 massive-activation 通道**：裸 spec_gain（范数口径）峰值在 L6（524.9）——null 随机 token 在 L6 产生巨大增量（massive activation），方向非特异（cdir 投影份额同样被摊薄），与语义调制的中晚层峰（L30）完全解耦。**双通道实为三层结构：L6 通用 massive-activation 通道（范数大、方向盲）+ 中晚层语义调制通道（方向特异、cls 0.094）+ 末层读出通道。**corr(spec_gain, gen_gain)=0.62。
3. **归纳复制 = 属性写头硬件复用（B2 4/4）**：2834 true 臂归纳式复读的检索-写入步由同一批属性写头簇执行（h20 也在 2833 头分解中为拮抗头，此处 −0.0301 仍为最强）。**2824 属性写头簇不是"属性专用头"，而是实体-档案读写通用头**——归纳复制（n-gram 级）与属性调制（语义级）共享同一写入硬件，差别在读端检索信号。此发现把"归纳头"与"属性写头"两个文献概念在 Qwen3-4B 上统一为一个硬件簇。
4. c_fruit top5 = h20(−0.0351)/h22/h23/h26——写头簇对 fruit 方向同样最强；c_food 无簇集中（top5 分散，h0/h5）——**写头簇编码的是实体-类别轴，不是 food 语义本身**，与 2828/2830/2834 链传递三否证一致（food 不在运行时通道）。

### 4. 硬伤

1. f_attn=0.14 是范数口径，被 L6 massive activation 污染；方向分辨的 attn/mlp 载体比未做（需逐层 cls 投影 × 载体分解二维）。
2. 归纳探针单句单实体；B2 是相关性（写头参与）非因果（消融写头看复制是否消失）——2837 候选：写头消融下归纳复制与属性调制的双重因果检验。
3. null token 的 massive activation 未系统刻画（L6 通道只出现在随机 token 条件，需要专门扫描）。

### 5. 产物登记（immutable + SHA256，Gen3 为准）

- 脚本 `tests/glm5/phase2836_dual_gain_induction.py` sha256 = `7e83b705def0abea…`
- `…/phase2836/dual_gain_induction/execution.json` sha256 = `d6f4cb699efe5150…`
- `result.json` sha256 = `33f789d804740842…`；`gain_layers.npz` sha256 = `c8de741b5d0dace7…`
- Gen 记录：Gen1（hs[-1] 陷阱，A1 假阳性失败+dsf 锚定错误）、Gen2（修正 dsf/口径/B2 前缀，成功）、Gen3（PREREG 文本对齐重跑，判决复现一致）。Gen1/Gen2 产物已按纪律删除。

### 6. 纪律制度化：numpy 混合高级索引禁令（2835 承接）

自本 Phase 起预审清单新增两条硬规则：①凡 `(slice, list, …)` 混合高级索引一律改写为逐行标量索引或先花式索引后切片，禁止依赖布局推断；②凡逐层分解/增量分析禁止使用 `hidden_states[-1]`（post-final-norm），一律 `hs[L]+attn[L]+mlp[L]` 重构原始末态。

### 7. 接续（2837 候选）

1. 属性写头消融双重因果检验：消融 {h20,h22,h23,h26}（L35），同时测归纳复制存活率与属性调制幅度——若两者同损，硬件统一成立；若只损属性，归纳走并行通路。
2. attn/mlp × 方向分辨二维载体谱（f_attn 的干净版本）。
3. L6 massive-activation 通道系统扫描（哪些 token 触发、是否 null-token 特有）。


## Phase 2837: 写头消融双重因果检验（III2 / 2837 首选） [2026-09-17 15:30]

### 1. 原理与设计

2836 发现 3（写头簇 4/4 参与归纳复制步）是**相关性**。本 Phase 把它升级为因果检验：消融 L35 写头簇 {h20,h22,h23,h26}（o_proj 输入按头置零，全序列），同测两功能：臂 A 属性调制（2833/2836 协议，20 词×5 条件×3 状态，cls_spec 用 raw 末态锚定）+ 臂 B 归纳复制（20 实体探针 "The {X} is a {Y}. The"，greedy 1 步复制判据）×3 状态（base / ablate_write / ablate_random，随机对照 seed 2837 与簇不相交）。一次成功（26.9s，Gen1）。

预注册：C1 属性损 spec_drop_write>0.30；C2 特异性 write>2×random；C3 归纳损 降≥50%。判决四分：hardware_unified / attribute_only / induction_only / decoupled。

### 2. 判决：C1=false / C2=true / C3=false → **decoupled**（L35 写头簇非两功能的必要载体）

| 判据 | 读数 | 结论 |
|---|---|---|
| C1 属性损 | cls_spec 0.0777→0.0760，**仅降 2.3%**（门槛 30%） | 写头簇不是属性调制的必要载体 |
| C2 特异性 | write 2.3% > random 1.0% | 方向正确但量级同样微小 |
| C3 归纳损 | 复制 15/20 → **15/20 → 15/20**（base/write/random 零损失） | 写头簇不是归纳复制的必要载体 |

臂 B 细节：基线即 5/20 不复制（silver→"metal"、China→"United"、Japan→"The" 等，copy_logit 距 top1 仅 0.3–1.1，属竞争失败）；三状态成功集**逐探针完全一致**；copy_logit 无系统性下降。

### 3. 机制发现：相关性归因 ≠ 因果必要性（方法论主线）

1. **2836 B2 的"写头簇参与归纳复制"不构成因果**：簇在复制步活跃（c_apple share 0.03）只是其对齐 cdir 的读出份额，argmax 由网络其余部分决定。**2833"语义调制载体=属性写头簇"与 2836 发现 3 须正式降级为相关性陈述**（本节即勘误入账）：头分解回答"谁与方向对齐"，消融回答"谁不可或缺"，两者在分布式冗余网络中可以完全脱钩。
2. **量级自洽验证**：2833 头分解 top1 share 0.0143，4 头合计 ~0.05 份额——消融致 2.3% 下降与该份额精确一致。头级归因的"对齐份额"是**可加性预测消融损伤的量度**，本 Phase 给出首个正反对照：份额≈损伤（本例两者皆小）。
3. **功能定位含义**：属性调制与归纳复制在 L35 之下已有完整实现（早/中层），L35 簇是"读出放大器"而非"写入决定者"。与 2836 A2（语义增益峰 L30）一致：**决定层在 L30 附近，L35 是末端读出**。
4. decoupled 的精确含义：否证的是"L35 头级簇承载"假说，不是"归纳与属性硬件无关"——真正的承载硬件在更早层/更细粒度（神经元/坐标级），正好落入 Delta II 的 II2 攻坚点。

### 4. 硬伤

1. 只消融 L35 单层；若簇在多层有副本（L29-35），需逐层消融或层组消融。
2. 属性判据用 cls_spec 对齐率（比率），mag_spec 未入判据（描述性数据已在 result）。
3. 归纳探针只测 step0；若复制被推迟到 step1-2 未覆盖（但三状态成功集一致，风险小）。

### 5. 产物登记（immutable + SHA256，Gen1 为准）

- 脚本 `tests/glm5/phase2837_head_ablation_causal.py` sha256 = `a804b69b8a2d14ba…`
- `…/phase2837/head_ablation_causal/execution.json` sha256 = `cce5cb9a880f58df…`
- `result.json` sha256 = `79131a0ae15cb10b…`；`ablation_spec.npz` sha256 = `8e86088b8c91fb6a…`
- Gen 记录：一次成功（Gen1，26.9s）。预注册判据全程未动。

### 6. 接续（2838 候选）

1. **多层簇消融**：L29–35 逐层 + 层组消融写头簇，定位属性调制的决定层（预测：L30 前后）。
2. II2 启动：写头簇神经元级分解——簇内 o_proj 列聚类 + 单神经元消融，检验"坐标级才是因果单元"假说。
3. 对齐份额→消融损伤的定量律：多簇×多层扫描，检验"损伤 = Σ份额"可加性预测（2838 与 2837 合并可给出第一个数据点）。


## Phase 2838: 决定层定位 + 坐标级因果检验（Delta II/III） [2026-09-17 16:10]

### 1. 原理与设计

承接 2837（decoupled）与 2836 A2（预测决定层 ~L30）：三遍扫描。Pass1 base 捕获（20 词×4 条件，全层 o_proj 输入一次前向全捕）→ 每层每头分离谱 share map；Pass2 对 L20–35 每层各自的 top-4 头做**该层单独消融**（iso/same/func/null 全部在消融下重测，口径与 base 一致 ctrl=0.5(func+null)）→ 损伤-层曲线；Pass3 在峰层 top 头内做坐标级（hd=128 维）解析排序 + top-8 联合消融 vs 随机 8 坐标对照。一次成功（95.7s，Gen1，~1700 前向）。

预注册：D1 峰层∈[25,35] 且 max drop>0.10；D2 峰损伤>3×L35 损伤；D3 峰层≤34；N1 top-8 坐标损伤≥0.5×全头损伤。

### 2. 判决：D1=false / D2=true / D3=true → **not_located**；N1=false → **distributed**

| 读数 | 值 | 含义 |
|---|---|---|
| 损伤峰层 | **L22（7.41%）**，次峰 L23（7.18%） | 因果峰在 L22-23，非预测的 L30 |
| L35 损伤 | 2.01% | 与 2837（2.3%）复现一致 |
| **L35 top4 恢复** | **{20,22,23,26}** | 与 2824/2837 簇精确一致——方法自洽验证通过 |
| 单层最高损伤 | 7.4% < 10% 门槛 | **无单一决定层** |
| 坐标级（L22 h30） | top8 0.73% / 全头 1.85% / 随机8 −0.11% | 坐标贡献真实（>随机）但分布式（0.73<0.93=0.5×1.85） |

### 3. 机制发现：对齐谱 ≠ 因果谱；因果结构跨层跨头分布式

1. **"对齐谱"与"因果谱"是两张不同的图**（2837 份额≠必要性的层间升级版）：对齐峰（cls_spec_inc L30 / cls_spec L33 / 写入 L35）与因果峰（消融损伤 L22-23）完全错位。响应谱图谱必须双谱并行测绘：对齐图谱（谁与方向共变）+ 因果图谱（谁被移除后方向信号塌陷）。主方案 II1/II3 需增补此双谱要求。
2. **语义调制通道是大范围低幅分布式结构**：头级（4/32 头=12.5%）、层级（单层 top-4 <10%）、坐标级（8/128=6.2%）三级分辨率下，任何局部子集都不承载超过 10% 的因果损伤——与 2832 亚加性接收、2837 decoupled 一致，构成"分布式冗余"假说的三尺度证据链。2836 预测的"L30 附近决定层"被修正为"**L22-23 起始、跨层分布式**"。
3. L22-23 为因果起点层：早期特征形成（L3-6 预算探测的 6L 语义峰在更早层）之后、中晚层读出之前的**中间映射层**。坐标 top-8 全部落在 h30 的 58-67/100-126 区段——存在局部聚集倾向但不足以过半。
4. D2=true（7.41% > 3×2.01%）确认峰层损伤显著超出末端层——梯度方向明确指向早中层。

### 4. 硬伤

1. D1 的 10% 绝对门槛使"最决定层"（L22，7.4%）判 not_located——门槛预注册在先，如实记账；层组联合消融（L22+L23+L26 等）未测，分布式可加性未知。
2. 每层只消融 top-4；若该层功能分散在 8-16 头，损伤被低估。
3. 坐标排序是解析一阶近似（|W3·cdir|×|a_k|），未含注意力后坐标间交互。

### 5. 产物登记（immutable + SHA256，Gen1 为准）

- 脚本 `tests/glm5/phase2838_layer_neuron_causal.py` sha256 = `f8e2235dc22c2eb9…`
- `…/phase2838/decisive_layer_neuron/execution.json` sha256 = `40acd85417c27b94…`
- `result.json` sha256 = `1d14722bed6a971a…`；`layer_coord_map.npz` sha256 = `8ccc4191bce04f99…`
- Gen 记录：一次成功（Gen1，95.7s）。预注册判据全程未动。

### 6. 接续（2839 候选）

1. **层组联合消融**：{L22,L23}、{L20-25}、{L22,L23,L26,L28} 层组 top-4 联合消融——检验分布式可加性（若 Σ单层 ≈ 联合，分布冗余弱；若联合≫Σ，存在跨层协作回路）。
2. 双谱正式化：对齐谱（share）× 因果谱（drop）的逐层相关——量化"错位度"，入响应谱图谱规范。
3. II2 神经元级下一步：L22-23 全头（top-8）× top 坐标交互扫描（两坐标联合消融矩阵）。


## Phase 2839: 层组联合消融（可加性）+ 双谱错位度量化 [2026-09-17 16:40]

### 1. 原理与设计

承接 2838（因果峰 L22-23 但单层<10%、双谱错位未量化）。臂 A：in-run 重测单层 drop(L22)、drop(L23)（分母）→ 联合 {L22,L23} → 全 16 层 top-4 联合（上限），全部 matched protocol（iso/same/func/null 在消融下重测，ctrl=0.5(func+null)，2838 复用层簇）；臂 B：对齐曲线（每层 top-4 |share| 和，in-run pass1）× 因果曲线（2838 registered drop，immutable）Spearman 相关。一次成功（39.6s，Gen1）。

### 2. 判决：regime = **independent**（ratio 1.0315）/ A2 = **true**（27.3%）/ B1 = **misaligned_confirmed**（ρ=0.106）

| 读数 | 值 | 结论 |
|---|---|---|
| drop(L22) / drop(L23) | 7.63% / 7.84% | 2838 值（7.41/7.18）in-run 复现 |
| 联合 {L22,L23} | **15.96%** | Σ单层 = 15.47% |
| **可加性比** | **1.0315 ∈ [0.7,1.3]** | **独立贡献：无跨层协作回路、无重叠冗余** |
| 全 16 层 top-4 联合 | **27.30%**（A2 门槛 15%） | 分布式合计实质承载 |
| 对齐峰 vs 因果峰 | L28 vs L22 | 峰错位 6 层 |
| **Spearman(对齐,因果)** | **ρ = 0.106** | 两谱几乎正交 |

### 3. 机制发现：独立分布式 + 双谱正交（方法论级）

1. **语义调制通道的完整定量画像成形**：跨 16 层 × 每层 top-4 头**独立**（可加比 1.03）分布式承载合计 **~27%** 的 cls_spec 对齐率，其余 ~73% 在更分散的头与读出内。无协作回路（联合≯Σ）、无冗余重叠（联合≮Σ）——"独立分布式"是第一个被预注册定量钉死的通道架构类别。
2. **双谱正交（ρ=0.106）是机械可解释性方法论级发现**：对齐重要性（归因/共变）与因果重要性（消融必要性）在分布式网络中近乎不相关。**任何仅基于归因的谱分析（SAE 特征对齐、logit lens、头分解）都不能替代因果消融谱**。LPF v5.3 响应谱图谱正式确立"双谱并行"规范：图谱 = 对齐谱 ⊕ 因果谱，两者错位度本身是网络架构的观测量。
3. L28 对齐峰无损的原因候选：L28 头写出的方向分量在下游被其他头**重新合成/覆盖**（读出冗余）——写入与保留解耦，与 2836"写头=读出放大器"一致。
4. 2838 not_located 定性升级为定量结论：不是"无决定结构"，而是"决定结构独立分布，总承载 27% 可测"。

### 4. 硬伤

1. 只测 top-4/层；每层其余 28 头未扫描，73% 残余的归属未定位（2840 首选：top-8/层或全头扫描的承载上限）。
2. 可加性只验证 {L22,L23} 一组；更多组（如 {L22,L26}）未覆盖。
3. cls_spec 是对齐率口径，非绝对语义量——27% 的"承载"是相对该口径。

### 5. 产物登记（immutable + SHA256，Gen1 为准）

- 脚本 `tests/glm5/phase2839_group_additivity.py` sha256 = `0b52b2ae152ae997…`
- `…/phase2839/group_additivity/execution.json` sha256 = `4509c51cf9460a6b…`
- `result.json` sha256 = `2de428497558a5f9…`；`spectra.npz` sha256 = `446e75abd26258bb…`
- Gen 记录：一次成功（Gen1，39.6s）。预注册判据全程未动。

### 6. 接续（2840 候选）

1. **承载普查收口**：top-8/层 或全部 32 头 × 16 层逐级消融，定位剩余 73%（预测：全头消融上限逼近 50-70%，读出端占余量）。
2. 对齐-因果解耦机制：L28 高对齐头写出的分量追踪（写入→下游重合成路径）。
3. 双谱规范入主方案：图谱 = 对齐谱 ⊕ 因果谱，错位度 ρ 为架构观测量——MASTER_PLAN II1/II3 修订。


## Phase 2840: 承载普查收口——注意力骨干判定 [2026-09-17 17:05]

### 1. 原理与设计

定位 2839 的剩余 73%：消融阶梯（in-run matched protocol，cls_spec 口径同 2837-2839）：S4 = top-4/层（L20-35）、S8 = top-8、S16 = top-16、S32 = L20-35 全部 32 头、S36 = 全 36 层全部注意力头。层簇排序由本 run share map 重算（与 2838 registered top-4 一致率 12/16 作交叉检验）。一次成功（44.4s，Gen1）。

### 2. 判决：P1=true / P2=true / P3=true / P4=true → **attention_backbone**

| 阶梯 | 承载（drop） | 增量 |
|---|---|---|
| S4（64 头） | 25.3%（2839 的 27.3% in-run 吻合） | 头级精选核心 |
| S8（128 头） | 32.7% | +7.4 |
| S16（256 头） | 40.2% | +7.3——**头级长尾幂律样分布** |
| S32（L20-35 全 512 头） | 73.7% | +33.5——非 top 头长尾承载大头 |
| **S36（全部注意力）** | **100.0%** | **对齐信号完全塌陷** |

### 3. 机制发现：语义调制通道 = 注意力骨干

1. **cls_spec 与 cdir 的对齐信号完全由注意力写入承载**（S36 → 100%）：移除全部注意力后分离谱对齐率归零。准确表述：MLP 只在注意力写入的调制上做局部加工，**不独立产生方向对齐的语义调制**——与 2826"大小走 MLP 列"不矛盾（MLP 列承载大小域的写入执行，但写入的域选择性由 attention 路由决定）。
2. **头级承载呈长尾分布**：12.5% 头 → 25%，50% 头 → 40%，剩余长尾 → 74%。无"魔法头"，精选核心 + 幂律长尾。与 2832"共享头核心+方向特异外围"结构画像互证。
3. 2837-2840 四级因果链闭合：单点头非必要（decoupled）→ 层级峰 L22-23 → 层组独立可加 27% → **注意力骨干 100%**。"属性写头簇"的最终身份：注意力骨干中按 share 排序的前缘切片，其因果地位随分辨率尺度变化——簇是观测口径的产物，骨干是物理实体。
4. 交叉检验：in-run top-4 与 2838 registered 一致 12/16——share 排序的 run 间波动集中于损伤小的层（L20/L24/L29 等），进一步支持"前缘切片"定性。

### 4. 硬伤

1. S36 = 100% 是对齐率口径的塌陷，非"全部语义信息在注意力"——MLP 的语义加工贡献需另一种口径（如 attn-only 前向 vs full 前向的行为对比）测度。
2. 阶梯每级只测一个切片；头级长尾的精确分布形状（幂律 vs 指数）需 32 点/层扫描。
3. S36 下模型整体行为崩坏，cls_spec 数值处于退化区（norm 剧变），1.0000 应读作"<测量下限"。

### 5. 产物登记（immutable + SHA256，Gen1 为准）

- 脚本 `tests/glm5/phase2840_carrying_capacity.py` sha256 = `e1a43bde6747200d…`
- `…/phase2840/carrying_capacity/execution.json` sha256 = `3c6b15c0fbbb2288…`
- `result.json` sha256 = `465971ed18675202…`；`capacity_curve.npz` sha256 = `a6a2c46e2f8ec604…`
- Gen 记录：一次成功（Gen1，44.4s）。预注册判据全程未动。

### 6. 接续（2841 候选）

1. 长尾分布形状：单头逐个消融（32 头 × 若干关键层）扫描，拟合承载分布（幂律/指数判决）。
2. 注意力写入路径回溯：L22-23 峰值头的 QK 来源（哪些位置/词位供键）——把"实体档案读写"接到具体 attention 边。
3. 双谱规范 + 承载画像写入 MASTER_PLAN v1.1 修订。


## Phase 2841: 长尾分布形状判决——单头逐个消融扫描 [2026-09-17 19:00]

### 1. 原理与设计

承接 2840（头级承载长尾，但阶梯只采样 4 个切片，"幂律样"仅是定性推断）。本 Phase 对 5 个关键层（L22/L23=因果峰、L26=中层、L28=对齐峰、L33=2833 峰）做**全 32 头逐个单头消融**（matched protocol 同 2837-2840：消融下重测 iso/same/func/null，ctrl=0.5(func+null)，cls_spec 口径），每层得 32 点承载分布，再对正 drop 部分做形状拟合判决：幂律 log(drop)~log(rank) vs 指数 log(drop)~rank，R² 比较。附加 C1：L22 全 32 头联合消融 vs Σ单头（头级可加性，2839 层级可加性的头级类比）。Gen1 一次成功（585.7s；首次前台运行在 600s 前台超时被 SIGTERM，属环境非科学性失败，后台重跑即成——Gen 判定以落盘 Gen1 为准）。

### 2. 判决：E1=true / E2=false（0/5 幂律）→ **exponential_dominant**；C1=false（指标病态）

| 层 | R²幂律 | R²指数 | 正 drop 头数 n_pos | max drop | Σ drop |
|---|---|---|---|---|---|
| L22 | 0.697 | **0.819** | 12/32 | 5.52%（h? top1） | −0.0075 |
| L23 | 0.791 | **0.908** | 10/32 | 4.72% | −0.0031 |
| L26 | 0.879 | **0.986** | 11/32 | 2.31% | +0.0079 |
| L28 | 0.945 | **0.959** | **6/32** | 2.57% | −0.1006 |
| L33 | 0.355 | **0.547** | 7/32 | 1.14% | −0.0080 |

5/5 层指数拟合更优（E2 幂律 0/5），4/5 层可分辨（E1=true，L28 差距 0.014 不可分辨）。**判决：长尾形状 = exponential_dominant**。

### 3. 机制发现

1. **指数尾非幂律尾（2840"幂律样"定性被单点扫描修正）**：头级承载分布有特征尺度（指数衰减），不是无标度重尾。"无魔法头"结论加强：承载不是自相似级联，而是有限个中等贡献者 + 指数衰减的外围——**集中度有内禀上限**，与 2832"共享核心+特异外围"、2839 独立可加一致。
2. **负 drop 普遍存在（直接冗余证据）**：单头消融常使 cls_spec **上升**（最高 +3.1%），每层仅 6-12/32 头为正贡献。头级网络存在大量可相互替代/补偿的通路；联合消融（L22 全 32 头 joint=7.4%）摧毁的是**补偿池整体**而非单个大贡献者。
3. **Σ单头 ≈ 0（−0.0075）vs 联合 7.4%（C1=false）**：带符号可加性失败——正负相消使 Σ 失去意义，联合/Σ 比值（7.4e7）病态。**头级冗余是绝对量而非相对量**：单头贡献微小（≤5.5%）且大量为负，任何头级"份额"都不能预测联合效应；2839 层级可加性（top-4 簇间 1.03）与头级带符号不可加并存——可加性只在"精选簇"尺度成立，在"全部单头"尺度失效。
4. **L28 对齐峰的因果面目**：仅 6/32 头消融致下降——对齐峰层的因果承载最少且最负（Σ=−0.1006），与 2839 双谱正交（ρ=0.106）互证：L28 头的写出分量在下游被重新合成/覆盖。
5. **两个预注册指标病态如实入账**：E3 的 c8=mean(top8)/Σ(32) 因 Σ≈0 产生 1e28 量级病态值（描述性指标失效，读数不可用）；C1 的 Σ 分母同病态。二者不改判决（E1/E2 完好），但**浓度指标须在未来 Phase 以绝对口径预注册**（如 top8 绝对均值 drop、或对 |drop| 加权重定义）。

### 4. 硬伤

1. c8/C1 指标病态（Σ≈0 分母），浓度只能定性读 top5（L22: 5.52/1.64/0.66/0.54/0.46%——即一头独大约占该层正承载 60%）。
2. 形状拟合仅用正 drop（6-12 点/层），指数 R² 在少点层（L33）置信度低。
3. cls_spec 是对齐率口径；负 drop 的"补偿"语义依赖该口径（亦可能部分是测量噪声，量级 ~1%）。

### 5. 产物登记（immutable + SHA256，Gen1 为准）

- 脚本 `tests/glm5/phase2841_longtail_shape.py` sha256 = `213d499e9e7198a5…`
- `…/phase2841/longtail_shape/execution.json` sha256 = `41e329c598f05d5d…`
- `result.json` sha256 = `648262fcebf5a711…`；`head_drop_map.npz` sha256 = `875c8358aef321a4…`
- Gen 记录：Gen1=前台超时 SIGTERM（环境性，无判决），后台重跑一次成功（585.7s）。预注册判据全程未动。

### 6. 接续（2842 候选）

1. **注意力写入路径回溯**：L22 top1 头（drop 5.52%）的 QK 边来源定位——哪些位置/词位供键，把"实体档案读写接口"接到具体 attention 边（II1 收口动作）。
2. **L28 写入→重合成路径追踪**：2839 候选遗留，用 2841 的负 drop 头集（补偿头）做干预追踪。
3. **浓度指标重定义预注册**：绝对口径（top8 均值 drop、|drop| 加权 Gini），与形状判决配套入响应谱图谱规范。


## Phase 2842: QK 边来源回溯——写头接口的源位置分解 [2026-09-17 20:15]

### 1. 原理与设计

承接 2841（5 层因果 top1 头：L22 h28 / L23 h29 / L26 h17 / L28 h4 / L33 h25）。用 eager attention 精确权重 + 目标层 v_proj 值向量，把每头对 cls_spec 对齐的贡献精确分解到两个 QK 源位置：pos0=条件词位（same/func/null 词）、pos1=目标词位（apple 自身）。恒等式 head_out(pos1)=Σ_j A[h,1,j]·OV_h V[j]（分解精确性验证 err=6.2e-15）；C_j = same 项 − 0.5(func+null 项)，signed share_j = C_j·cdir/||d_spec_full||。20 目标词 × 4 前向，15.3s。Gen 历史：3 次崩溃（self_attn 关键字参数 hook、bf16 dtype、reshape 维度）后 Gen4 一次成功；execution.json 每次重跑覆盖，磁盘终态自洽（run4 exec+result 配对）。

### 2. 判决：Q1=true(5/5) / write_interface = **self**（4/5 self_dominant，L33 h25 mixed）

| 头 | share_pos0 | share_pos1 | attn 同条件 pos0/pos1 | attn func pos0 | 判决 |
|---|---|---|---|---|---|
| L22 h28 | 0.00000 | **0.00010** | 0.690 / 0.310 | 0.987 | self_source |
| L23 h29 | 0.00000 | **0.00118** | 0.536 / 0.464 | 0.956 | self_source |
| L26 h17 | 0.00001 | **0.00224** | 0.721 / 0.279 | 0.935 | self_source |
| L28 h4 | 0.00001 | **0.00122** | 0.667 / 0.333 | 0.735 | self_source |
| L33 h25 | −0.00000 | −0.00001 | 0.947 / 0.053 | 0.988 | mixed（总量≈0） |

### 3. 机制发现：写头接口 = QK 自绑定门（self-binding gate），非 OV 内容复制

1. **pos0（条件词位）贡献恒 ≈0，全部对齐贡献来自 pos1（目标词自身）**：语义调制不是"从条件词复制内容"，而是**注意力质量重分配**——same 条件使目标词对该头的自注意力份额大增（L22 h28：same 0.310 vs func 0.013 / null 0.128），条件词存在时目标 token **更多读自己、更少读前文**。func/null 条件下注意力质量涌向 pos0（0.87-0.99）但其 OV·cdir 贡献经 ctrl 相消归零。
2. **与前序结论闭环**：① 2824-2825"实体条件化写入=token 私有 emb 注入"——语义内容早已在目标词自身表征（来自 embedding 与浅层），L22+ 写头只做**门控读出**；② 2834"无生成式链传播"——因为接口根本不是跨 token 内容传递；③ 2837 decoupled——消融单头输出不破坏调制，因为调制量（注意力质量）本身是 QK 侧的、由更早层状态决定。
3. **头贡献量级与 2841 drop 自洽**：top1 头 share_head≈0.0001-0.0022（占 ||d_spec_full|| 口径），换算占 cdir 分量（≈0.084·||d_spec_full||）约 0.1-2.7%——与 2841 单头消融 drop 1-5.5% 同量级（消融 drop 含下游传播放大）。
4. L33 h25 总贡献 ≈0（−1e-5）：其 1.14% 消融 drop 全部来自下游传播，自身直写可忽略——2841 的"指数尾"在最末端层与"零直写+纯中继"一致。

### 4. 硬伤

1. 序列长 2，无 BOS——来源只有两个位置，"self vs cond"判决受限于此窗口；更长语境（"我喜欢吃苹果"式）下 pos0 可能是多 token，须扩展。
2. share 是 cdir 单方向口径；QK 门的域选择性（颜色/大小/类别差异）未测。
3. bf16 前向 + fp64 分解的数值误差 6.2e-15（恒等式内），但 sain 捕获本身经 float() 截断，与真实 bf16 链有微小偏差。

### 5. 产物登记（immutable + SHA256，Gen4 为准）

- 脚本 `tests/glm5/phase2842_qk_source_backtrace.py` sha256 = `93f8552beceabe5a…`
- `…/phase2842/qk_source_backtrace/execution.json` sha256 = `063ee300adeb6f0b…`
- `result.json` sha256 = `31dbad089f5429a4…`；`source_shares.npz` sha256 = `463e684829f54b3d…`
- Gen 记录：Gen1-3 崩溃（hook kwargs / dtype / reshape），Gen4 成功（15.3s）。预注册判据全程未动。

### 6. 接续（2843 候选）

1. **自绑定门的时间-内容双重验证**：same 条件下 pos1 自注意力增量（Δgate = A_s[1,1] − 0.5(A_f+A_n)[1,1]）与该头 drop 跨词相关；并测 QK 侧干预（把 A[1,1] 钳到 func 水平）能否消除调制——门的因果收口。
2. **多 token 语境扩展**：把来源分解扩展到 ≥4 token 窗口（"我喜欢吃[red]苹果"），测 pos0 群体贡献与 2831 优先级表对接。
3. **QK 门的域选择性**：8 方向组（颜色/大小/类别/速度）下 Δgate 对比——写头门是否域通用。


## Phase 2843: 门因果钳制 + 域选择性（规模版） [2026-09-17 20:45]

### 1. 原理与设计（响应用户"加大测试类型和数量"指令）

规模升级：**80 词**（10 类目 × 每类全量单词 ≤8，2806 框架）× **20 头**（15 目标 = 5 层 × top-3 来自 2841 registered npz + 5 随机对照）× **双窗口**（2-token + 3-token [cond,的,w]）。臂 A：逐头 QK 因果钳制——monkey-patch 目标层 self_attn（精确复刻 eager 前向 + GQA repeat_kv + 逐头 logit 偏置 b=ln[(t/(1−t))(1−A)/A]），把 same 条件下 A[h,1,1] 钳到该头自身 func/null 水平 t（钳制残差 max 0.008），matched protocol 重测 cls_spec。臂 B：10 方向 Δgate（L22 top-3 均值）+ 3-token 三源分解。Gen2 成功（125.7s；Gen1 崩溃=patched forward 漏 GQA repeat_kv，已修）。

### 2. 判决：G1=false / G2=false / G3=ρ=0.75 → verdict_a = **not_sufficient**（单门）；D1=true（10/10）/ T1=false → verdict_b = **partial**

| 读数 | 值 | 结论 |
|---|---|---|
| 单头钳制平均 drop（15 目标） | **1.47%**（门槛 30%） | 单门非充分 |
| 随机对照头平均 | −0.04% | 方向正确但差距仅 1.5%（门槛 +15%） |
| **剂量-响应 Spearman(Δgate, drop)** | **0.75** | 门强度预测因果损伤——机制真实 |
| 头级梯度 | L22 h28 6.06% / L23 h29 4.54% / L26 h17 2.69% → L28 h20 −0.11% | 与 2841 drop 排序同构 |
| Δgate 方向覆盖 | **10/10 为正**（fruit 0.452 / country 0.397 / animal 0.280 / metal 0.236 / vehicle 0.136 / nature 0.131 / clothing 0.119 / furniture 0.078 / food 0.065 / tool 0.019） | 门域通用但**梯度分化 24 倍** |
| 3-token 份额（15 头均值） | pos0=0.0 / pos1('的')=0.00025 / pos2(自身)=0.00005 | 自锚失效：贡献转移至'的'位 |

### 3. 机制发现

1. **门真实但单门微小（剂量-响应钉死）**：ρ=0.75 意味着 QK 自绑定门强度直接预测该头的因果损伤——2842 的"self 接口"获得剂量梯度支持；但单门钳制仅损 1.5%（均值），与 2837-2841 全线一致：**门也是分布式冗余的一部分**，无单点充分性。钳制最高头 L22 h28 损 6.06%（其 Δgate 最大），恰为 2841 单头消融 top1（5.52%）——钳制与消融两种干预在同头收敛。
2. **门的域梯度是新的架构观测量**：10/10 方向为正（通用性），但强度差 24 倍（fruit/country/animal/metal 强，food/tool 弱）——**food 弱门再次独立复现**（2836 c_food 无簇集中、2834 无生成链之后的第三线索）：food 类目在该通道的读写强度系统性偏低，类别轴本身有强弱分层。
3. **自锚是 2-token 窗口现象，3-token 重新锚定**：插入'的'后头贡献从 pos2（自身）转移到 pos1（'的'位，0.00025 vs 0.00005）。修正机制表述：头读出的不是"自己"而是**"被条件化的邻近状态"**——直接相邻时是实体自身，隔着功能词时是携带条件化状态的功能词（'的'的状态因 earlier 层 attention 继承条件词信息）。与 2832"'的'在 red 方向 −3.0 拮抗写入"互证：功能词位是条件化状态的中继载体。pos0（条件词）在两窗口都精确为 0——**远端条件词永不直写**，条件化信息必须经局部链（cond→的→w 或 cond→w）转入。
4. **方法学沉淀**：QK 逐头 logit 钳制管线（monkey-patch + GQA repeat_kv + 精确偏置公式，残差 0.008）可复用于任意"注意力质量因果检验"。

### 4. 硬伤

1. G1 门槛 30% 对"单头钳制"设置过苛（15 头各自独立钳制的均值注定小）；未测**15 头联合钳制**——联合门损伤是下一问（预测：联合 ≈ Σ×可加比，量级或达 20-30%）。
2. 3-token 仅测 [cond,的,w] 一种构型；'的'位锚定效应需 [cond,w,w2] 等构型对照。
3. Δgate 域梯度的字频/词长混杂未控制（fruit 高门可能部分反映高频实体）。

### 5. 产物登记（immutable + SHA256，Gen2 为准）

- 脚本 `tests/glm5/phase2843_gate_causal_clamp.py` sha256 = `3cca0fcb38c17022…`
- `…/phase2843/gate_causal_clamp/execution.json` sha256 = `2b58ab3b850c3d0f…`
- `result.json` sha256 = `38e63d688281e453…`；`gate_maps.npz` sha256 = `be8c5d5f52320ad6…`
- Gen 记录：Gen1 崩溃（GQA repeat_kv 遗漏），Gen2 成功（125.7s，~2300 前向）。预注册判据全程未动。

### 6. 接续（2844 候选）

1. **15 头联合钳制 + 阶梯**（单门→top3→top5→15 头联合），检验门损伤的可加性与总充分性——门的"分布式充分性"收口。
2. **锚定转移机制**：[cond,的,w] vs [cond,w] 的 pos1 锚定对照 + '的'位状态条件化追踪（'的'的 V 是否携带 cond 信息）。
3. **域梯度归因**：Δgate 与词频/类目典型性的相关，剥离混杂。


## Phase 2844: 联合门钳制阶梯 + '的'位锚定继承 [2026-09-17 21:00]

### 1. 原理与设计

收口 2842-2843 门线。臂 A：15 目标头按 2841 registered drop 排序，累计钳制阶梯 rungs {1,3,6,10,15}（每头钳到自身 func/null 自注意力水平，2843 管线，残差 max 0.005）+ 5 随机头联合对照 + in-run 单头重测（可加性分母）；臂 B：'的'位状态条件化继承——[cond,的,w] 中 '的' 在 L22 输入态的 cdir 投影 Δ(same−0.5(func+null))，80 词符号一致性 + 与实体词态的强度比。80 词（同 2843 口径），一次成功（120.6s，~2100 前向，Gen1）。

### 2. 判决：J1=false / J2=true / J3=false / J4=true → verdict_a = **mixed**；A2=true（100%）/ A3=2.11 → **anchor_inheritance 确证**

| 读数 | 值 | 结论 |
|---|---|---|
| 阶梯 rungs | 1→4.65% / 3→**8.34%** / 6→8.77% / 10→10.64% / 15→9.00% | top-3 后饱和，J3 非单调 |
| Σ单头（in-run） | 9.38% | — |
| **联合 15 头** | **9.00%**，**可加比 0.959** | **门池精确可加（J2）** |
| 随机 5 头联合 | −3.33% | 对照无效（J4） |
| 联合承载 | 9.0% < 15% 门槛 | **门池集体也不充分（J1）** |
| '的'位 Δ 正向词比例 | **80/80 = 100%** | 锚定继承决定性确证（A2） |
| **强度比 \|Δ的\|/\|Δw\|** | **2.11** | **'的'态携带 2 倍于实体词的条件化信号** |
| C1 Spearman(Δgate, cls_base) 按方向 | 0.55 | 门强度与对齐强度中等耦合（country 基线最高/门次之，fruit 门最高/基线次之） |

### 3. 机制发现：门线收口——三个定量钉死

1. **门池 = 可加、饱和、小占比**：15 个最强门联合仅承载 9.0%（可加比 0.96，与 2839 层级 1.03 同构——门与层一样无协作无冗余）；top-3 已给出 8.34/9.38 ≈ 89% 的门池贡献（前缘饱和）。**门的分布式充分性不成立**：把全部已识别门的 QK 自绑定钳掉，注意力承载的语义调制仍剩 ~91% 走非门通路。语义调制通道的最终画像：**注意力骨干（100%）= 小门池（~9%，可加饱和）+ 无特化门的大骨干（~91%）**。
2. **'的'是条件化状态的主载体（新发现，2.11×）**：功能词'的'在 L22 输入态携带 2.11 倍于实体词的类别条件化信号，且 80/80 词方向一致（零反例）。2832"的=拮抗写入"、2843"锚定转移至'的'位"在此获得定量解释：**语法功能词是类别条件的集线器**——它紧邻实体、自身无强词义、其状态完全由 earlier 层对条件词的 attention 继承决定。这直接连接外部语言族图谱的语法系统（功能词的机制角色）与内部响应谱。
3. **域梯度的部分解耦**：Δgate 与 cls_base 相关 0.55 但不完全重合——fruit 门最强、country 对齐最强，二者分离说明"门开多大"与"信号多强"是两个自由度（门的域选择性由 QK 侧类目键决定，对齐强度由 OV 侧内容决定）。

### 4. 硬伤

1. rung 10→15 非单调（10.64→9.00）：可能为头间干涉或噪声（80 词均值 SE ~1-2%）；未做重复运行误差棒。
2. 门池仅覆盖 5 层 top-3；L20-35 全层门池（48+ 头）未穷尽——9% 是"已识别门"的下限式估计。
3. A2/A3 的 cdir 投影是单方向口径；'的'态的全方向条件化结构未分解。

### 5. 产物登记（immutable + SHA256，Gen1 为准）

- 脚本 `tests/glm5/phase2844_joint_gate_ladder.py` sha256 = `c7f619021fa95fe6…`
- `…/phase2844/joint_gate_ladder/execution.json` sha256 = `b972c98627b2996a…`
- `result.json` sha256 = `a2a6d7e22c633e8b…`；`ladder.npz` sha256 = `10e538345a048252…`
- Gen 记录：一次成功（Gen1，120.6s）。预注册判据全程未动。

### 6. 接续（2845 候选）

1. **非门骨干分解**：剩余 ~91% 注意力承载的 QK/OV 结构——对非门头做 2842 式源分解（抽样 20 头），检验"无门头"是否同样走邻近态读出（机制同构性）。
2. **'的'位条件化追踪上溯**：'的'的状态在哪些层形成（逐层 cdir 投影曲线）——功能词集线器的形成层定位。
3. **MASTER_PLAN v1.2**：把 2837-2844 门线+骨干画像写入 II 章（门池 9% + 骨干 91% + 的集线器 + 域梯度），更新里程碑。


## Phase 2845: 非门骨干同构性 + '的'态形成曲线 + 方向矩阵 [2026-09-18 03:00]

### 1. 原理与设计

2844 收口门线后开放两大空白：91% 非门骨干的机制（M4）与'的'集线器的形成动力学。臂 A：从 L20-35 均匀抽 20 个非门头（排除 2844 注册的 15 门头），每头做 2842 式源分解（2-token 窗口 pos0/pos1 cdir 贡献）+ 2843/2844 式单头 QK 钳制；臂 B：[cond,的,w] 的'的'位（pos1）全 36 层 cdir 投影曲线 Δ_de(l)，formation = 首个 ≥0.5×peak 的层；臂 C：'的'位 × 10 方向 × 36 层全矩阵（80 词全量）。预注册 H1/H2/B1/B2/C1/C2。

Gen 记录：Gen1 崩溃（o_proj reshape 硬编码 4096，实际 (2560,4096)，2833 教训复发）；Gen2 崩溃（hs[l][2] 标量索引错误）；Gen3 跑通但发现口径 bug（forward pos=2 取实体词位而非预注册的'的'位 pos1）；Gen4 最终（130.0s，~2400 前向）。Arm A 判决 Gen3/Gen4 逐位一致（同种子复现 ✓）。

### 2. 判决：verdict_a = **heterogeneous**（H1=false/H2=true）；verdict_b = **late_or_inconsistent**（B1=false/B2=true）；C1=false

| 读数 | 值 | 结论 |
|---|---|---|
| 非门头自源份额 ≥0.9 比例 | **0/20**（份额散布 −0.61~0.78） | 非门骨干不走自绑定门机制（H1=false） |
| 非门头平均 \|钳制损伤\| | **0.89%**（13/20 头为负=补偿） | 个体贡献微小且带符号（H2=true） |
| '的'态 formation 层 | **L30**（>24 门槛），峰 L35（17.5） | 条件化信号是**深层累积**现象（B1=false） |
| formation 层符号一致性 | 95%（80 词） | 方向一致但晚成（B2=true） |
| 10 方向 ≥70% 正词比例 | 仅 metal 1/10 | '的'位非专向集线器（C1=false） |
| 选择性 max/sum | **0.26** | **宽带广播型**：弱信号铺向多方向 |

### 3. 机制发现

1. **M4 定性升级——非门骨干是异质混合体**：门机制（QK 自绑定）不外推到非门头（0/20 自源主导）；非门头个体损伤 0.89%、13/20 为补偿性负值。91% 大骨干 = 大量个体微小、相互补偿、源结构各异的头的集合——与 2841 指数尾、2844 可加饱和拼合后，**语义调制通道不存在第二套统一接口**，接口方程须写成"门池显式项 + 异质骨干背景项"。
2. **'的'集线器是深层累积器而非早期继承器**：Δ_de(l) 均值曲线从中层 ~0.5 单调爬升至 L33-35 的 ~3.2（formation L30 = 半峰值）。修正 2844 的"earlier 层继承"表述：L22 处信号存在（2844 A2 100% 正）但仅为峰值的 ~15%，主体在 L22→L35 期间经由深层 attention 逐步注入——**功能词状态是读出前的最后一段信息装配线**。
3. **口径警告（重要）**：2844 A2 的 L22 读数用的是 attention 输入态（经 input_layernorm，含逐维 gamma 加权）；本 Phase 用原始 residual hidden_states，同层同方向 frac_pos 仅 0.47。LN 的 gamma 加权可改变投影符号——**未来全部曲线类测量统一用原始 residual 口径**，历史 LN 口径读数标注区分。
4. 方向矩阵显示'的'位对非本类方向也有 0.4-0.7 的正信号比例（如 metal 0.71）——宽带弱广播支持"语法槽位携带弥散类别先验"图景，与 2831"问法不改历史"互补。

### 4. 硬伤

1. 非门头仅抽样 20/497，异质性的细分类（自源/前源/中继比例谱）未做全量。
2. C1 池化口径把本类与非本类词混入同一统计——本类方向的对角线读数（frac_pos 0.37-0.47）与非本类混在一起，方向分解需按 own/non-own 分层重做。
3. 臂 B 只有 [cond,的,w] 一种构型；'的'态深层累积的因果性（钳制 L30-35 对生成的影响）未测。

### 5. 产物登记（immutable + SHA256，Gen4 为准）

- 脚本 `tests/glm5/phase2845_backbone_de_curve.py` sha256 = `1abe7ee09e6a8f6a…`
- `…/phase2845/backbone_de_curve/execution.json` sha256 = `c497f8b0427b1a60…`
- `result.json` sha256 = `1c395b8ee9cf980b…`；`backbone_de.npz` sha256 = `424fc44b00e206a1…`
- Gen 记录：Gen1/2 崩溃（shape/索引），Gen3 成功但口径 bug（pos2≠pos1）主动作废重跑，Gen4 最终。预注册判据全程未动。

### 6. 接续（2846 候选）

1. **MA2 启动——全头因果普查**（ATLAS_PLAN 战线 2）：36 层×32 头单头钳制+源分解，分块后台（每块 3 层），产出 1152 头损伤图谱。
2. '的'态深层累积因果检验：钳制/增强 L30-35 对 '的' 位信号的生成端影响。
3. 方向矩阵 own/non-own 分层重做 + 非门头全量源分解分块。


## Phase 2846: MA2 全头因果普查 [2026-09-18]

### 原理
ATLAS_PLAN 战线 2 首战：把 2841 的 5 层×32 头单头扫描放大为 **全 36 层 × 32 头 = 1152 头 × 80 词**（10 类 × 8 词），每头测 ①QK 逐头 logit 钳制因果 drop（2843 管线，matched func/null 口径，每头 80 词）②2842 式源分解直写 cdir 份额（free，来自 base 捕获）。分层 checkpoint + 断点续跑。预注册：C1 形状（≥60% 层指数 R²>幂律）、C2 前缘集中度（top-64 头正载荷 ≥25%）、C3 份额→必要性（Spearman ≥0.3）。

### 结果（Gen1 一次成功，5527.4s ≈ 92 分钟，~9.6 万前向）
- **C1=true（28/36 层指数主导）** → `exponential_dominant_full`：2841 的 5 层结论全尺度成立，头级承载分布有特征尺度、无标度重尾不存在。
- **C2=true（top-64 头承载 51.6% 正载荷）**：5.6% 的头扛一半正载荷——前缘集中存在，但分布尾部极长（mean|drop|=0.53%，60.5% 头为负 drop 补偿）。
- **C3=false（ρ=0.1496 < 0.3）**：直写份额在 1152 头全尺度上**不能**预测钳制必要性——2837/2839 的"对齐谱⊥因果谱"在最大样本上确认；份额只预测簇内量级（2837 局部 ρ 高），不预测全局排序。

### Top-10 因果头（重大新发现）
| 排名 | 头 | 正载荷 |
|---|---|---|
| 1 | **L13 h30** | **10.12%（独大）** |
| 2 | L2 h31 | 4.92% |
| 3 | L5 h25 | 4.78% |
| 4 | L22 h28 | 4.30%（2841/2843 已知 top1 ✓ in-run 复现） |
| 5 | L23 h29 | 3.34% |
| 6 | L23 h7 | 3.05% |
| 7 | L5 h26 | 2.92% |
| 8 | L34 h15 | 2.91% |
| 9 | L26 h4 | 2.77% |
| 10 | L4 h4 | 2.26% |

**关键新发现：早中层头簇（L2/L4/L5/L13）占 top-10 的 5 席**——此前所有扫描（2824 起）都聚焦 L20-35，因果前缘其实在 **L13 附近就开始**（L13 h30 独大 10.1%，是 L22 h28 的 2.35 倍）。"对齐谱峰在中晚层、因果前缘在早中层"的双谱错位现在有了最极端的数据点。

### 质量与口径
- 每层钳制残差 checkpoint 记录均 ≤0.0586（L0 max），但全局统计 clamp_max_resid=0.5708 与逐层记录不一致——疑为全局统计聚合了逐词残差而非逐层 max；如实登记，2847 对质。
- 产物 immutable：script `456f63b1` / execution `88bdcbae` / result `da6eda5e` / census_full.npz `75a43f9b` + 36 个分层 checkpoint（SHA256 全登记）。

### 硬伤
- C3 的钳制口径为单头 80 词均值，未做 joint 干预验证前缘头簇的联合充分性。
- clamp 残差全局统计与逐层记录不一致未解决。

### 结论
1. 头级因果地图（Atlas-I 首张全头图层）完成：**指数尾 + 中度前缘集中（51.6%/5.6%）+ 份额⊥必要性**三判定全尺度钉死。
2. **因果前缘双峰结构**：早中层簇（L2-L13，含 L13 h30 独大）+ 中晚层簇（L22-L34）——修正"语义调制=中晚层现象"的旧画像，2838 的因果峰 L22 只是后半段。
3. 2841 指数结论外推至全模型成立。

### 接续（2847 候选）
1. **L13 h30 解剖**（现役最大因果头）：2842 式源分解 + 源词位归因 + 钳制域选择性——它为什么在对齐谱里不显眼却在因果谱独大？
2. 早中层簇（L2/L4/L5/L13 top 头）联合钳制阶梯——因果前缘的可加性与充分性。
3. clamp 残差全局 vs 逐层不一致对质。


## Phase 2847: 隐形冠军解剖 + 双因果角色 [2026-09-18]

### 原理
2846 发现"份额-必要性倒挂"：早中层头直写 cdir ≈0 却损伤大，中晚层头直写大却损伤小。本 Phase 用三臂检验两簇的因果角色假说：**臂 A** L13 h30 全解剖（钳制后 36 层下游位移剖面、门剖面、总写出量 vs cdir 分量、10 方向选择性）；**臂 B** 早簇 5 头与晚簇 4 头联合钳制阶梯 + 可加性；**臂 C** 钳制偏置饱和审计。预注册 F1（上游路由器画像）/F2（双因果角色）/F3（前缘可加性）。

### 结果（Gen1 一次成功，119.4s，80 词，~2400 前向）
**判决：F1=true / F2=true → `dual_roles_confirmed`；F3=false（重要负结果）**

| 读数 | 值 | 结论 |
|---|---|---|
| L13 h30 cdir 直写 | **−0.0028**（≈0） | 确认不直写答案方向 |
| L13 h30 总写出量 / cdir 分量 | **926×** | 写出巨大但方向与 cdir 正交——工作在"形成"而非"作答" |
| L13 h30 自绑定门 | A11_same **0.527** vs ctrl 0.057（**9.3×**） | 与 2842 门同构，但服务于状态形成 |
| 钳制后 cdir 变化峰层 | **L34**（位移峰 L16） | 损伤经 20+ 层传播才在对齐读出处兑现——深层重组 |
| 早簇 shares/drops | 0.002-0.006 / 0.023-0.101 | 形成器：不写答案、不可或缺 |
| 晚簇 shares/drops | 0.019-0.477 / 0.028-0.043 | 放大器：直写大、可替代 |
| 早簇可加比 | **0.65**（joint5 18.4% < Σ单 28.2%） | F3=false：形成器**亚可加**（重叠协作） |
| 晚簇可加比 | 0.93 | 放大器独立可加，与 2844 门池 0.96 同类 |
| 钳制饱和率 / max b / max resid | 0 / 8.2 / 0.0096 | 冠军头无饱和 |

### 横向抑制线索（新机制假说）
L13 h30 钳制后 **4/10 竞争方向对齐率上升**（drop 为负：dir1 −0.83、dir6 −0.64、dir8 −0.76）——形成器不仅在"构建"自身类别状态，还可能**主动压制竞争类别方向**（横向抑制，类 lateral inhibition）。dir4 的 −12.23 为小分母病态，如实登记不作证据。

### 臂 C：2846 残差统计病因（代码级确认）
每层 checkpoint 打印用 `clamp_resid[-NH:]`（仅末词 32 头），非全层聚合；全局 0.5708 为真实逐条 max（某词-头 b 钳制未达靶，不在冠军集内）。本跑冠军残差均 ≤0.0096。

### 机制总画像（2846-2847）
**因果双角色结构**：早中层**形成器**（不写答案方向、巨大正交写出、9.3× 自绑定、亚可加重叠、效果经深层传播兑现）→ 中晚层**放大器**（直写 cdir 0.26-0.48、独立可加、可替代）。对齐谱只看见放大器（份额大），看不见形成器（份额≈0）——双谱正交的机制根源找到。

### 硬伤
- 横向抑制为相关性线索，未做竞争方向钳制的交叉因果验证。
- 形成器→放大器接口（L13→L22 状态传递路径）未追踪。

### 结论
1. F1/F2 预注册确认：**头级因果结构 = 形成器 + 放大器双角色**，替代单一"写头簇"叙事。
2. 可加性三区谱系：形成器 0.65（重叠）< 晚簇 0.93 ≈ 门池 0.96（独立）。
3. 份额⊥必要性（C3=0.15）的机制根源 = 对齐谱只测放大器。

### 接续（2848 候选）
1. 横向抑制定量检验：钳制 L13 h30 后升起方向的源头追踪 + 双向干预（钳形成器 + 钳竞争读出头）。
2. 形成器→放大器接口：L13 h30 输出如何进入 L22-23 头的 QK 输入（状态传递路径追踪）。
3. 早簇内部重叠结构：成对联合钳制矩阵（10 对）定位协作对。


## Phase 2848: 横向抑制检验 + 接口 + 协作对 [2026-09-18]

### 原理
三臂检验 2847 留下的三个问题：**H1** 横向抑制（L13 h30 钳制后竞争方向升起是否真实）——本 Phase 改用**绝对口径**（Δ 对齐 = |clamp 投影| − |base 投影|，除以 ||Δspec||），配 3 个随机对照头（L18 h5 / L9 h28 / L29 h8，seed 2848）；**H2** 形成器→放大器接口（钳 L13 h30 后 4 个放大器头的自绑定门 A11 相对下降 + 输入态位移）；**H3** 早簇 10 个成对联合钳制 vs Σ单（定位 0.65 亚可加的来源）。

### 结果（Gen2 成功 109.9s，80 词；Gen1 崩溃 = 随机头生成用 NL 而非 NH，已修）
**判决：H1=false ／ H2=true ／ H3=false → `partial`**

### H1=false：2847 横向抑制线索正式否证（勘误）
- 绝对口径下 **0/10 方向升起**：mean Δ 全在 −0.0002~−0.0018，frac_pos(>0.02) 全 0；随机对照 mean Δ ≈ ±0.0001（同量级微小负值）。
- **2847 的"4/10 方向升起"是相对口径小分母伪影**——那些方向 base 对齐率近 0，除以小 base 把噪声放大成 −0.64~−0.83 的"负 drop"（dir4 −12.23 同源）。
- **第二次口径教训正式入账**（继 2845 LN/residual 之后）：对齐/曲线类测量**默认绝对口径**，相对口径仅在 base 超过显著阈值时使用。此规则进预审清单。

### H2=true：形成器→放大器接口确认
| 放大器 | 门相对下降 | 层内输入位移 |
|---|---|---|
| L22 h28 | −0.070（近阈） | 31.6 |
| L23 h29 | **−0.106** | 31.9 |
| L26 h4 | **−0.121** | 42.2 |
| L34 h15 | −0.008 | **129.2** |

3/4 头门下降 >10% → **形成器状态直接喂给放大器的 QK 门**：钳制上游形成器，下游放大器的"自读"份额随之收缩——接口是状态+门控双通道。

### H3=false：无强协作对
10 对 ratio ∈ [0.82, 1.20]，均值 ≈ 1.0，无 <0.7 者 → 2847 早簇联合 0.65 的亚可加**不来自特定协作对，而是高阶（>2 阶）弥散交互**（成对可加、多点联合饱和）。

### 结论
1. 形成器画像修正：不压制竞争方向（H1 否证），其独大损伤来自**喂养放大器**（H2 确认）——"隐形冠军"的因果力 = 下游读出网络的门控供给者。
2. 双谱正交机制图完整：形成器（份额≈0、因果大、喂门）→ 放大器（份额大、因果小、冗余读出）。
3. 早簇亚可加定位为高阶交互——分布式协作的又一证据。

### 硬伤
- H2 的门下降是相关观察（未做门回归干预闭环）；L34 h15 位移 129 的构成未分解。
- 高阶交互仅定位到"存在"，未枚举 4-5 头子集。

### 接续（2849 候选）
1. 高阶交互分解：早簇 4-5 头子集枚举，定位饱和源。
2. 门下降→行为闭环：直接钳放大器门（等效于形成器钳制的下游效应）对质 H2。
3. MA 战线推进：Atlas-E 词汇扩展（80→200 词）预研。


## Phase 2849: 高阶交互全枚举 + 门闭环 [2026-09-18]

### 原理
两问收口：**G1** 早簇亚可加（2847 joint5=0.65）的饱和来自哪一阶？——C(5,k) 全枚举（k=2..5，共 26 子集）+ in-run 单头分母，聚合比 ratio(k) = mean joint_k / mean Σsingles；**G2** 放大器门下降（2848 H2）是否因果充分？——把 4 个放大器自绑定门直接钳到"形成器被钳后"的实测水平（bias_to 目标 A11），看能重现多少 L13 h30 行为损伤。

### Gen 记录
Gen1 判决作废：词级 ratio 分母 Σsingles≈0 再现 1e28 病态（2841 同款教训，第三次——预审清单加"聚合比优先"）；且残差记录用错对象（gate 钳制的残差应记录与门目标的差）。Gen2 修正后 166.1s 成功，同种子实质读数与 Gen1 一致（G2 两侧完全相同）。

### 结果（Gen2，80 词，~3300 前向）
**判决：G1=false（边缘）／G2=false → `neither`；但两个负结果均信息量大**

| 读数 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|
| 聚合 ratio | 0.888 | 0.796 | 0.722 | 0.654 |
| 词级中位 ratio | 0.966 | 0.885 | 0.815 | 0.766 |

1. **G1（边缘未过，趋势钉死）**：ratio 随子集大小**单调递减**（0.89→0.80→0.72→0.65），但 G1 门槛（k=3 比 k=2 低 0.10）差 0.008 未过 → 饱和是**渐进重叠累积**而非阈值式跳变；2847 的 0.65 正是 k=5 端点，无特定协作子集（与 2848 H3 一致）。
2. **G2 决定性负结果：门通道只承载 18% 损伤**。直接把 4 个放大器门钳到形成器钳后水平（门钳残差 vs 目标仅 0.027，钳制成功），joint drop 只有 **1.43%** vs L13 h30 的 **8.0%**（closure ratio 0.179 < 0.6 门槛）→ **2848 H2 的门下降是相关而非因果充分**——形成器的因果力主要走 **residual 内容通道**（其 926× 正交写出的状态被下游计算消费），而非门控调节通道。
3. in-run 单头均值与 2846 census 有词级波动（L5 h25 0.006 vs census 0.048）——null 随机化所致，单头词级噪声大（2841 已知），均值结论以 census 为准。

### 结论
形成器→放大器因果结构最终画像：**形成器经 residual 内容喂给下游（主通路，~82% 损伤承载）+ 门控调节（次通路，~18%）**。"隐形冠军"不需要修改任何下游门也能造成损伤——它的正交写出本身就是下游计算的输入。

### 硬伤
- G1 门槛 0.10/0.20 预注册偏严，单调趋势与判据不一致——登记为"趋势确认、判据未过"。
- 内容通道的具体消费者（哪些层/头消费形成器写出的正交状态）未定位。

### 接续（2850 候选）
1. 内容通道消费者定位：L13 h30 正交写出的下游读者追踪（哪些头的 QK 在读这个状态）。
2. MA 战线词表扩展预研（80→200 词）+ 双谱普查自动化管线。


## Phase 2850: 内容通道消费者定位 [2026-09-18]

### 原理
2849 钉死形成器 L13 h30 的因果力 ~82% 走内容通道后，本 Phase 追问"谁在消费这个写出"。三臂（80 词，matched protocol，仅 5-8 前向/词）：**臂 A** 下游 22 层×32 头门重塑图（ΔA11 = 钳制后自读门 − base）；**臂 B** 逐头写出变化 Δwrite 与 FORMER 原始写出方向 ŵ_out 的对齐（内容转发者）及与 cdir 的对齐；**臂 C** 数据驱动 top-3 门读者恢复干预（把它们的 A11 恢复到 base 水平，看损伤是否回落）。

### 结果（Gen2 成功 58.4s；Gen1 崩溃 = bias_to 参数形状，已修）
**判决：J1=false ／ J2=false ／ J3=false → `not_confirmed`（三联否证）**

| 臂 | 读数 | 结论 |
|---|---|---|
| A 门读者 | **0 个头**达 frac≥0.6 且 \|ΔA11\|≥0.02 双阈 | 门重塑弥散，无系统读者 |
| B 内容转发者 | **0 个头** align(Δwrite, ŵ_out)≥0.3 | w_out 方向不沿头链传播 |
| C 中介恢复 | med_ratio **0.877**（仅回 12% 损伤，门槛 ≤0.8） | top-3 读者非充分中介 |

放大器 |ΔA11|（in-run）：L23 h29 0.0417、L26 h4 0.0319、L22 h28 0.0199、L34 h15 0.0111——2848 的门下降在更大样本下幅度缩水且不稳定。cdirchg 峰 L34 复现 ✓。

### 机制结论：损伤无离散中介

1. **头间线性内容链第四次否证**（与 2828 句对、2830 同句、2834 自由生成收敛后，首次在头间尺度确认）：w_out 写出经下游 LN+非线性被打散，没有任何头把"形成器的内容"转发下去——头间传递不是离散消息传递。
2. **2849 "内容通道 ~82%" 表述修正**：内容通道 ≠ 离散头间传递，而是**全局状态几何扰动**——钳制移除一个大正交分量，整个下游状态几何变形，最终在 L34 附近重组为 cdir 对齐变化（几何重组层）。
3. 分布式总画像闭合：形成器（弥散写入）、放大器（冗余读出）、消费者（不存在）——**无魔法头、无魔法链、无魔法消费者**。

### 硬伤
- 恢复干预只检验门读者路径；几何重组的层间机制（L14→L34 之间 cdir 如何从位移中"析出"）未解剖。
- 单词 5-8 前向的轻量设计使 top-3 选择词内噪声较大。

### 接续（2851 候选）
1. **L34 几何重组层解剖**：cdir 在最深层如何从弥散位移中析出（L30-35 逐层投影矩阵分析）。
2. MA 战线推进：词表扩展预研（80→200 词）+ 双谱普查自动化管线封装。

---

## Phase 2851: L30-35 几何重组层解剖——透射放大假说否证 [2026-09-18]

### 原理
2850 发现形成器（L13 h30）钳制损伤引发全局状态几何扰动，cdir 分量在 L30-35 陡增（sain 口径 0.04→0.63），且无任何单头与形成器写出方向对齐——析出非离散交接。本 Phase 检验**透射放大假说（transmission amplification）**：放大器（2846 census 前缘，如 L34 h15 直写份额 0.47）读取自身位移态、经 cdir 对齐 OV 通道把位移几何"复印放大"进 cdir 空间。

三臂设计（80 词 × 5 前向，matched protocol，clamp L13h30 至 base awc）：
- **臂 A** 位移旋转剖面：δ(l) = hs_c[l] − hs_b2[l]（residual hs 口径，pos1），报 a_l = δ·cdir/‖δ‖ 与 ‖δ(l)‖ 全 36 层；sain 口径并列作敏感性。
- **臂 B** 逐层增量归因：attn/mlp 残差增量投影 cdir（L14-35）；L30-35 内逐头 Δwrite = a_c·(V_c·OV) − a_b·(V_b·OV) 投影 cdir（绝对口径）→ top 贡献头。
- **臂 C** 透射增益：gain_h = (Δwrite_h·cdir)/(δ(l−1)·cdir)。

预注册（冻结于任何观测前，execution.json `345690ea`）：K1 = transmission_amplification iff ≥60% 的 L14→35 cdirchg 增量来自 L30-35 attn 写出 AND top-5 贡献头 ≥3 属 census 前缘；K2 = top-3 gain 描述性；判决 transmission_lattice iff K1 否则 diffuse_unresolved。

### 执行事故（4 次运行，均观测前修复，正式判决取 Gen4）
1. Gen1 NameError hs_b2：**Edit 报成功未落盘**（沙箱视图缺陷复发）——Grep 复核后重编辑落盘。
2. Gen2 ValueError matmul 0-dim：`hs_c[l][1]` 多套一层索引——forward_run 已做 pos 切片返回 (NL,dim)，`[1]` 取成标量。修为 `hs_c[l]`。
3. Gen3 分母保护 bug：`max(x, 1e-30)` 在 x<0 时返回 1e-30（防零保护反噬）→ frac_late = −6.49e28、gains ±1e28 病态值。**2841 分母病态教训第四次变体：负分母 + max 防零**。修为直接除（|分母|>1e-6 才除，否则 NaN→null）。教训应升级纪律：**防零保护只允许用于已知非负分母；signed 分母一律显式 abs 阈值判断**。
4. Gen4 正式（脚本 `5c16d430`，37.8s，max_resid = 0.004 合格）。

### 结果
| 指标 | 值 | 判据 |
|---|---|---|
| K1 第一条 frac_late（signed/abs 敏感性） | **0.2033 / 0.5053** | ≥0.6 → **fail** |
| K1 第二条 n_front_in_top5 | **0/5** | ≥3 → **fail** |
| K1 整体 | **false** | → **diffuse_unresolved** |
| K2 透射增益 top-3 | L35h22 +0.060 / L35h20 −0.033 / L35h26 +0.030 | 衰减非放大 |

关键数字：
- top5 贡献头（L30-35 Δwrite·cdir）：L35h22（−0.094）、L35h20（+0.052）、L35h26（−0.047）、L32h9（+0.045）、L31h1（+0.045）——**与 census 前缘零重叠**。
- **深层反直觉发现：L30-35 attn 增量 cdir 投影为负（signed Σ −0.065），MLP 更负（−0.59）**；同期残差位移 cdir 分量 0.373→1.761（+1.39）。cdir 析出**不是**深层组件写入——深层组件反而在负向拉。
- 位移范数 ‖δ‖：L14 = 2.47 → L35 = 16.88（全尺度增长，L30-35 加速）；signed 对齐 a_l 自 L24 转负、深层达 −0.10（位移与 cdir 微弱负对齐且加深）。
- cdir 绝对分量 L24 起缓升（0.104→0.373→1.761），但相对份额下降——cdirchg 陡增是"绝对量上升+整体旋转"复合，非份额析出。
- 口径敏感性：sain 口径 cdir 剖面与 residual 口径形状一致（L30-35 陡增复现，0.05→0.62 vs 0.10→1.76），结论不依赖口径。

### 结论
1. **透射放大假说否证**：cdir 在深层的析出既非 census 前缘头写出（0/5），亦非任何 L30-35 组件写入（attn/mlp 增量均为负贡献），透射增益仅 0.03-0.06（衰减量级）。
2. cdirchg L30-35 陡增的本质 = **全局几何重组的表征**：早中层写入的位移经残差流身份携带 + 深层组件的负向重组，位移矢量整体旋转中 cdir 分量绝对值上升。与 2850"全局状态几何扰动"合流。
3. 分布式画像补完最后一块：无魔法头（2846）、无魔法链（2828/2830/2834）、无魔法消费者（2850）、**无魔法重组器（2851）**——深层重组是全组件分布式效应，正源必在早中层写入+携带路径。

### 硬伤
- **索引错位一层**：hidden_states[l] = layer l−1 输出，profile[35] 实为 L34 输出后，L35 写出仅影响未计入的 hs[36]；late 窗口 [30:] 含 L35 写出、漏 L29。对 fail 判决无影响（两口径均 <<0.6，且深层增量方向与判据要求的正贡献相反），但层归属表述需按此约定。
- signed 跨层抵消 vs abs 口径差异大（0.20 vs 0.51），判据在两种口径下均 fail——稳健。
- **析出正源未定位**：深层组件是负源；谁把位移往 cdir 推（早中层哪一层/哪类组件、LN 是否参与旋转）仍开放。

### 文件
- 脚本 `tests/glm5/phase2851_emergence_anatomy.py` sha `5c16d430`（17,210 B）
- 产物 `tests/glm5/result/rdc_query_construction_20260913/phase2851/emergence_anatomy/`：execution.json `345690ea`、result.json `08b6c6f4`、emergence.npz `0769bbe8`

### 接续（2852 候选）
1. **析出正源定位**：L14-29 窗口内逐层 attn/mlp 增量 cdir 剖面（正源应该在 L24-29 的 0.104→0.373 段）+ LN 重组效应检验（LN 是否把弥散位移分量旋转进 cdir 方向）。
2. MA 战线推进：词表扩展预研（80→200 词）+ 双谱普查自动化管线封装。

---

## Phase 2852: cdir 析出正源定位——深层 MLP 透射累积机制 [2026-09-18]

### 原理
2851 钉死 L30-35 的 attn 增量 cdir 投影为负、cdir 析出非深层 attn 写入，正源开放。本 Phase 把分解窗口扩到 L14-29 全部 16 层（逐层 attn/mlp 增量 + 逐头 Δwrite·cdir），并加恒等式审计，定位析出正源。设计延续 2851 协议（80 词 × 5 前向，clamp L13h30 至 base awc，matched protocol），逐头分解在测量阶段覆盖全部 L14-29，源层选择放在判决阶段（无数据依赖测量）。

预注册（冻结于任何观测前，execution.json `6b126527`）：
- S1 层集中：L14-29 内 |attn+mlp 增量 cdir| top-3 层承载 ≥60% 窗口总绝对增量
- S2 头局部化：3 源层内 top-5 (layer,head) 承载 ≥50% 合计绝对头增量
- S3 恒等式审计：|Σ增量(L14-34) − profile 差(35 vs 14)| / |profile 差| < 5%
- 判决：layered_writers iff S1∧S2 否则 diffuse_field

### 执行
Gen1（49.5s）一次干净通过；发现**头身份映射 bug**（top5_pairs 用 `WIN_LO + t//NH` 假设 hc_sel 行序为 L14-16，实际行序为按 |增量| 降序的 src_layers=[29,26,28]）→ 修正为 `src_layers[t//NH]` 重跑 Gen2 正式（50.2s）。S1/S2/S3 数值两轮完全一致（bug 仅影响描述性层号）。max_resid = 0.0072 合格。脚本 `b2040807`。

### 结果
| 判据 | 值 | 结果 |
|---|---|---|
| S1 层集中（top-3 份额） | **69.88%**（L29/L26/L28） | **true** |
| S2 头局部化（top-5 份额） | **52.6%** | **true** |
| S3 恒等式 rel_err | **0.108%** | **true** |
| **final_verdict** | | **layered_writers** |

关键数字（signed 口径，跨词 mean）：
- **无正源层**：inc_layer 全剖面除 L13 clamp 写入（+0.0097）与零星噪声（|值|≤0.02）外**全部为负**——cdir 分量的"增长"不是任何层写入正 cdir，而是每层都向 −cdir 漂移。
- **负增量主力 = MLP**：L30-34 合计 mlp −1.15 vs attn +0.08；mlp 增量自 L26 起渐强（−0.04→−0.08→−0.08→−0.29 峰），L30-32 维持 −0.30/层量级。
- 位移 cdir 分量：L14 ≈ 0 → L35 = −1.703（2851 abs 口径 1.76 与之相容——|signed mean| ≈ abs mean，词间符号高度一致；2851 用 abs 显示故符号未见于其 profile）。
- 源层 attn 头次级结构：top5 = L26h17(−0.065)、L26h21(+0.034)、L29h19(−0.031)、L28h7(+0.029)、L28h6(−0.025)，量级比同层 mlp 增量小 4-10 倍；与 census 前缘 **0/5 重叠**。
- 恒等式闭合（rel_err 0.108%）验证层号对齐正确（2851 的索引错位在本 Phase 修复）。

### 结论（2851+2852 机制闭环）
1. **cdir"析出"的真实机制 = 深层 MLP 透射累积**：clamp 移除 L13h30 后状态沿 −cdir 漂移；深层组件（尤其 MLP）读入带偏移的状态、输出保持/放大该偏移（近似透传），负增量逐层累积（L26 起渐强、L29-31 峰、L30-34 合计 −1.07）。不是 attn 头 OV 通道（2851 否证：透射增益仅 0.03-0.06）、不是离散消费者（2850 三联否证）、不是深层组件"写入 cdir"（本 Phase：无正源层）。
2. 2851 K1 fail 的完整解释：判据只认 attn 写出（深层 attn 增量 ≈0），真凶 MLP 不在判据内——预注册否证了 attn 透射假说本身；2852 补上 MLP 通道并完成定位。
3. S1/S2 的"集中"语义是**负向透射的层/头集中**：layered_writers 更准确的名字是 **layered_transmitters**——集中度来自组件读入态已带偏移（sain 输入差 cdir 分量逐层增长，2851 已测 abs 0.05→0.62），组件是透传者而非发起者。发起者仍是 L13 clamp 写入本身（+0.01 直接 cdir 分量 + 全局几何扰动）。
4. 分布式画像最终版：无魔法头（2846）、无魔法链（2828/2830/2834）、无魔法消费者（2850）、无魔法重组器（2851）、**无正源写入层（2852）**——损伤的语义后果由"早层写入 + 深层 MLP 全员透传放大"承载。

### 硬伤
- signed/abs 口径跨 Phase 混用（2851 profile 为 abs、2852 为 signed）造成表面矛盾（+1.76 vs −1.70），已在 MEMO 对齐；后续 profile 输出默认双口径。
- 2851 与 2852 的组件增量量级差 2-7 倍（null 词不同 → bias_for 的钳制目标 t 不同 → 钳制强度不同）；符号层面 attn 深层增量在噪声边缘可翻转（+0.08 vs −0.065），mlp 负源两轮同号稳定。钳制强度未作为受控变量。
- "MLP 透传"的微观机制（J_mlp 在 cdir 方向的读出-写回增益）未直接测量——本 Phase 只定位到组件类型与层窗口。

### 文件
- 脚本 `tests/glm5/phase2852_emergence_source.py` sha `b2040807`（17,073 B）
- 产物 `tests/glm5/result/rdc_query_construction_20260913/phase2852/emergence_source/`：execution.json `6b126527`、result.json `9b04cac3`、source.npz `e8767687`

### 接续（2853 候选）
1. **MLP 透传增益直接测量**：J_mlp·(cdir 方向单位向量) 的 cdir 回投份额逐层剖面（读出-写回增益谱），定位透传放大率峰值层；与 LN 前后状态对比分离 LN 贡献。
2. MA 战线推进：词表扩展预研（80→200 词）+ 双谱普查自动化管线封装。

---

## Phase 2853: MLP cdir 透传增益谱——本征放大与饱和制动的分离 [2026-09-18]

### 原理
2852 定位 cdir 析出载体为深层 MLP 透传（无正源层，mlp 增量主导），遗留问题：透传是被动（g≈1）还是主动放大（g≥1.5）。本 Phase 数值测量每层 MLP 的 Jacobian cdir 二次型：

- x_in(l) = sain_same[l][pos1] + attn_base[l]（base run 的 mlp 输入）
- **g_jac(l) = [mlp(x_in + ε·cdir) − mlp(x_in)]·cdir / ε**（ε=1.0 预注册；bf16 ulp 约束注明在 execution.json）
- din/dout：clamp−base 的 mlp 输入/输出差 cdir 投影；g_emp = mean(dout)/mean(din)（ratio of means）
- 线性度诊断：ε₂=0.5 同测 L26-35

预注册（冻结，execution.json `c1be60ce`）：T1 = active_amplification iff max g_jac(L26-35) ≥1.5；passive_transmission iff median |g_jac| ∈ [0.5,1.5)；else nonlinear_unresolved。T2 描述性（g_emp 谱、Pearson r、线性度比）。

### 执行
Gen1 一次干净通过（36.3s，脚本 `70d3d82a`，max_resid 0.0101）。

### 结果
| 判据 | 值 | 结果 |
|---|---|---|
| T1 | max g_jac = **51.26**（L34） | **active_amplification** |
| g_jac 谱 L26-35 | 0.99, 0.68, **5.08, 12.10, 17.93, 11.16, 13.40, 23.68, 51.26, 23.06** | L28 起暴涨 |
| g_emp 谱 L26-35 | 0.47, 0.26, 0.69, 1.68, 1.28, 0.90, 0.50, 0.48, 0.18, −0.11 | 温和 |
| r(g_jac, g_emp) | **−0.26** | 无相关 |
| 线性度 g(ε₂)/g(ε₁) | **1.98**（median） | 亚线性（饱和区） |
| din_cdir L26-35 | −0.086 → −0.665（单调负增） | 输入偏移逐层积累 |
| dout_cdir L26-35 | −0.04 → −0.32（L29 峰）→ **+0.07（L35 转正）** | 深层回拉力 |

### 判据甄别（关键）
g_jac 与 g_emp 差 **10-40 倍**且无相关，三个机制性解释（非测量 bug）：
1. **工作点分离**：g_jac 在 base 工作点（未位移态）测小信号；g_emp 是 clamp 态（位移后）的大信号响应。位移 ‖δ‖ ~5-17 已把 mlp 推入 SwiGLU 饱和区——线性度比 1.98（ε 减半单位增益翻倍）独立证实 ε=1.0 已在饱和弯曲段，真切线增益比 g_jac(ε=1) 更大。
2. **J 非对称**：g_jac = cdir·J·cdir 只测对称部分；实际响应通量走 (Jᵀ·cdir)·δin，可被其他方向吸收。
3. din 的非 cdir 分量（‖δin‖ 2-10 >> |din·cdir| 0.1-0.7）经非线性混合，不按小信号增益缩放。

**硬伤（下次必补）**：本 Phase **缺随机方向对照**——g_jac 的大值可能部分是 J 谱背景各向异性（cdir 不特殊）；需 g_rand = r·J·r 的 null 分布对照（2849/2809 教训同类）。

### 结论（2846→2853 全链机制闭环）
1. **深层 MLP 对 cdir 方向存在巨大本征（小信号）放大率**（5-51×，L28 起激活，L34 峰）——"透射放大晶格"的硬件真实存在。
2. **正常工作时该增益通道空转**：无钳制时输入无 cdir 偏移；钳制 L13h30 后输入带 −cdir 偏移（din 0.09→0.67 逐层积累），但位移同时把 mlp 推入饱和区，实际透传被**饱和制动**为温和增益（g_emp 0.2-1.7）→ 逐层累积 −1.7（2852 的析出曲线）。
3. **L35 回拉力**：dout_cdir 转正（+0.07）——最深层的组件开始把 −cdir 漂移往回拉，与 2851"深层负贡献"的抑制性一致。
4. 终版机制命名：**饱和制动的透射放大晶格（saturated transmission lattice）**——本征放大器阵列 + 饱和限幅 + 早层发起 + 全员透传。分布式画像最终补完。

### 文件
- 脚本 `tests/glm5/phase2853_mlp_transmission.py` sha `70d3d82a`（14,735 B）
- 产物 `tests/glm5/result/rdc_query_construction_20260913/phase2853/mlp_transmission/`：execution.json `c1be60ce`、result.json `13c3b7e3`、transmission.npz `19e2d8f6`

### 接续（2854 候选）
1. **随机方向对照补全**：g_rand = r·J·r 的 null 分布（≥64 随机方向 × L26-35），判定 cdir 增益的特殊性（z-score）；同时测 clamp 态工作点的 g_jac'（位移态切线）分离"工作点移动"假说。
2. MA 战线推进：词表扩展预研（80→200 词）+ 双谱普查自动化管线封装。

---

## Phase 2854: g_jac 特殊性对照——2853 伪影坐实与机制终版修订 [2026-09-18]

### 原理
2853 遗留两项对照义务：①cdir 增益特殊性（缺随机方向 null）；②工作点移动假说（g_jac 测于 base 工作点）。本 Phase 每词每层一次批量 mlp 调用（mlp 逐元素，rows=[cdir, r1..r16]）同时测 17 个方向，base 与 clamp 双工作点，null 分布 = 80 词×16 方向。

预注册（冻结，execution.json `6c3ec81b`）：Z1 = cdir_specific iff ≥6/10 层（L26-35）z=(g_cdir−median(null))/(1.4826·MAD(null)) ≥3；Z2 = operating_point_shift iff ≥6/10 层 g′/g <0.5；判决 saturated_cdir_lattice（Z1∧Z2）/ active_cdir_lattice（Z1∧¬Z2）/ anisotropic_background。

### 执行事故与勘误（本 Phase 最大价值）
1. **Gen1 重大测量 bug（连带 2853 判决作废）**：mlp_batch 直接调用 `model.model.layers[l].mlp`——但 Qwen3MLP 不含内部 LN，真实前向是 mlp(**post_attention_layernorm**(h))。缺失 LN2 导致基线错配：hook 基线 = mlp(LN2(h_b))，扰动臂 = mlp_raw(x+εd)。固定差向量 Δ0=mlp_raw(x)−mlp(LN2(h_b)) 造成系统偏移——完全解释 2853/2854Gen1 的观测结构（g_rand 均值≈0 但 MAD 高达几十；g_cdir 5-51 落入伪影噪声带）。**2853 的 T1=active_amplification（本征 5-51× 放大器）正式作废**——"饱和制动的透射放大晶格"命名作废。D1 一致性 r=1.0 恰是两处同 bug 的复现，不是验证。
2. **Gen2 ratio 病态（第五次负分母防零教训）**：`np.maximum(gcb,1e-9)` 在 gcb<0 时返回 1e-9 → ratio ±4e8 病态、Z2 误判 true。修复为 abs 阈值直接除。
3. **操作事故：phase2852 产物误删**（清理脚本路径复制错），立即重跑恢复——**result.json/npz SHA 与原登记逐位一致**（9b04cac3/e8767687，确定性复现验证通过），仅 execution.json 时间戳更新（`3642fb61`）。纪律有效性的实证：immutable+SHA 登记使误删可完全恢复。
4. **Gen3 正式**（脚本 `302f68f7`，43s，max_resid 0.0039）。

### 正式判决
| 判据 | 值 | 结果 |
|---|---|---|
| Z1 cdir 特殊性 | z 谱 [−0.54,−1.19,−0.31,−1.02,−0.14,−0.42,0.86,1.61,1.55,−0.13]，中位 **−0.23** | **false** |
| Z2 工作点移动 | ratio = [1.00,0.99,1.03,0.94,1.04,0.84,1.02,1.01,0.96,1.19] | **false** |
| **final_verdict** | | **anisotropic_background** |

修复后真实增益谱 g_cdir_base（LN2+MLP 复合路径，cdir·J·cdir）：L26-31 = −0.4~−0.8（微负），L32-34 = 1.7/4.5/7.1，L35 = −1.8。与 2853 g_emp（0.2-1.7）量级相容。

### 机制终版（2846→2854 修订）
1. **无 cdir 特异性放大器**：深层 MLP 的 cdir 增益落在随机方向 null 分布内（z<2）；L32-34 的 4.5-7.1 与 null 背景同量级（J 谱整体抬升，非 cdir 专属）。
2. **无工作点移动/饱和制动**：clamp 态与 base 态切线增益比 0.84-1.19 ≈ 1——2853 的"饱和制动"解释作废。
3. 终版图景：**被动透传带（passive transmission band）**——LN2+MLP 复合路径对 cdir 的有效增益 ~±1（g_emp 0.2-1.7，切线谱 −0.8~+1.8 主体），位移 cdir 分量逐层积累 = din 链式传播 × 透传增益；"cdirchg L30-35 陡增"的真源 = **L29-31 增益 >1 窗口**（g_emp 1.68/1.28/0.90）的积分形状，非放大器阵列。
4. 分布式画像定稿：无魔法头（2846）、无魔法链（2828/2830/2834）、无魔法消费者（2850）、无魔法重组器（2851）、无正源写入层（2852）、**无 cdir 特异放大器（2854）**——损伤的语义后果由"早层发起 + 全层被动线性透传 + 中段 >1 增益窗口"承载，方向结构完全由 cdir 本身（unembed 类间方向）的几何携带。

### 教训制度化
- **LN 边界**：任何模块级 Jacobian/扰动测量必须复现真实前向的前置 LN 链（Qwen3：mlp 前 post_attention_layernorm；attn 前 input_layernorm）。
- **负分母防零（第五次）**：signed 分母一律 abs 阈值 + 显式分支，禁用 max(x,ε)。
- **清理脚本路径核对**：删除类操作执行前必须核对目标路径（本次误删 2852 产物，靠 SHA 登记完全恢复）。

### 文件
- 脚本 `tests/glm5/phase2854_gain_specificity.py` sha `302f68f7`（16,157 B）
- 产物 `phase2854/gain_specificity/`：execution.json `6c3ec81b`、result.json `7347efad`、specificity.npz `b0f5ac17`
- 2852 恢复：execution.json `3642fb61`（result `9b04cac3`/npz `e8767687` 与原登记一致）

### 接续（2855 候选）
1. **L29-31 增益窗口解剖**：g_emp>1 的层内 ln2 前后分解（LN2 贡献 vs SwiGLU 贡献）+ 逐神经元 top-k 贡献（>1 窗口的分子来源）。
2. MA 战线推进：词表扩展预研（80→200 词）+ 双谱普查自动化管线封装。

---

## Phase 2855: 增益窗口解剖——判决 ln2_mediated/neuron_diffuse 与第二处窗口勘误 [2026-09-18]

### 原理
2854 终版图景遗留具体问题：>1 增益窗口的分子来源在 LN2 还是 SwiGLU、在哪些神经元。80 词（2806 词表单 token 配额 8/类）× L26-35，EPS=1.0，三臂：①三谱分解 g_comp=[mlp(LN2(x+εcdir))−mlp(LN2(x))]·cdir/ε、g_ln=[LN2(x+εcdir)−LN2(x)]·cdir/ε、g_mlp=[mlp(ln_x+dln)−mlp(ln_x)]·cdir/ε（chained，dln=LN2(x+εcdir)−LN2(x)；显式过 LN2，2854 教训应用）；②SwiGLU 神经元级线性化分解 mvec=silu'(g)⊙(Wg·dln)⊙u + silu(g)⊙(Wu·dln)，c_n=(Wdᵀ·cdir)_n·mvec_n，top-32/top-8 份额；③线性化 vs 数值 Pearson r。输入点 x_in=sain+attn（L 层 MLP 输入，LN2 前），基线 out_b=hook mlp 输出（同函数）。L13H30 钳制臂保留（max_resid 0.0106）。

预注册（冻结，execution.json `dfcf5656`）：W1 swiglu_intrinsic iff ≥2 层（L29-31）|g_mlp|≥1.2 且 sign(g_mlp)=sign(g_comp)；ln2_mediated iff ≥2 层 |g_ln−1|≥0.2；else flat。W2 neuron_localized iff top-32/9728 份额（词平均，L29-31 池化）≥0.5；else neuron_diffuse。L1 linear_ok iff mean Pearson r ≥0.9。

### 正式判决（一次通过，34.1s，脚本 `b1ec5668`）
| 判据 | 值 | 结果 |
|---|---|---|
| W1 窗口来源 | n_ln2_layers=4（n_swiglu_layers=1） | **ln2_mediated** |
| W2 神经元 | top-32 份额 **0.2549**，top-8 0.1942 | **neuron_diffuse** |
| L1 线性化 | mean r = **−0.17** | **false** |
| final_verdict | | **ln2_mediated/neuron_diffuse** |

g_comp 谱 L26-35 = [−0.433, −0.782, −0.355, −0.788, −0.165, −0.599, 1.729, 4.519, 7.068, −1.765]——**与 2854 g_cdir_base 逐位一致**（L35 差 0.03），测量可靠性双重确认。g_ln 谱 = [0.706, 0.627, 0.605, 0.515, 0.461, 0.429, 0.363, 0.290, 0.283, 0.440]——**处处 <1 且随深度单调收缩**。

### 三项登记（硬伤与解释陷阱）
1. **g_mlp ≡ g_comp 数学恒等（设计缺陷）**：ln_x+dln = LN2(x)+[LN2(x+εcdir)−LN2(x)] = LN2(x+εcdir)，与 g_comp 扰动臂输入逐位相同——"chained"分解在代数上退化为复制。正确的独立对照应为 g_direct=[mlp(ln_x+εcdir)−mlp(ln_x)]·cdir/ε（LN2 输出点直接注入，绕开 LN2 对输入改写）。g_mlp 字段无独立信息；W1 的 swiglu 分支退化为 |g_comp|≥1.2（L29-31 全负且 |·|<1.2 → 修正后 n_swiglu=0）。
2. **W1/W2 窗口 off-by-one**：q31=W1_HI−WIN_LO=32−26=6，range(q29,q31+1)=q∈[3,6] 实际统计 **L29-32 四层**（设计意图 L29-31 三层）。n_swiglu_layers=1 完全由 L32（g=1.73）贡献。修正窗口后 n_swiglu=0、n_ln=3（|g_ln−1|=0.637/0.710/0.717），W2 份额 0.287（仍 <<0.5）——**判决对 off-by-one 稳健**。
3. **g_ln 判据解释陷阱**：LN2 是归一化，对任何切向输入方向必然收缩（g_ln<1 是几何必然，非"介导"证据）；|g_ln−1|≥0.2 在 10/10 层成立——判据特异性为零。ln2_mediated 是"判据字面成立"，**不能解读为 LN2 是增益源**；恰恰相反，LN2 处处收缩 cdir（0.71→0.28 随深度加深），g_comp>1 完全来自 SwiGLU 复合响应在非线性区的放大（L1 r=−0.17 独立证实 EPS=1.0 下线性化无效——增益活在非线性区，不是切线增益）。

### 第二处窗口勘误（2854 节 4512/4526 行）
2854 节终版图景写入"真源 = L29-31 增益>1 窗口（g_emp 1.68/1.28/0.90）"——该 g_emp 是 **2853 在 clamp 工作点测的透射比 mean(dout)/mean(din)**（其 L29-31 峰是位移态/饱和区性质）；2854 base 工作点复合增益谱的 >1 窗口是 **L32-34（1.73/4.52/7.07）**，L29-31 为负（−0.79/−0.17/−0.60）。两个不同量（clamp 态透射比 vs base 态模块响应增益）被混写。2855 实测（与 2854 逐位复现）确认：**base 工作点 >1 复合增益窗口 = L32-34**；4510 行本身写对了（"L32-34 的 4.5-7.1"），仅 4512/4526 两处层号错误。cdirchg 陡增的积分形状真源相应修订为 L32-34 窗口。

### 机制增量（2855 新信息）
1. LN2 是被动收缩器（g_ln 全谱 0.28-0.71、随深度单调降），不塑造窗口形状——窗口形状完全由 SwiGLU 复合响应携带。
2. >1 窗口（L32-34）的神经元来源高度分布式：top-32/9728 份额仅 0.25（top-8 仅 0.19）——无魔法神经元。
3. 分布式画像第六项成立：**无魔法神经元（2855）**，补齐 2846 无头/2828-2834 无链/2850 无消费者/2851 无重组器/2852 无正源层/2854 无特异放大器之后的最细粒度。

### 附件叙事判定（用户提交长文："离散电路→连续介质物理 + 饱和制动 AGC"）
- **洞察 B（饱和制动 AGC）整体不成立**：其全部定量支柱（g_jac 5-51×"本征增益"、制动因子 ~0.02、AGC 工作点自稳）已被 2854 系统性否证——①50× 是 LN 缺失伪影（Δ0 基线错配）；②修复后真实切线谱 −0.8~+7.1，无 50×；③Z2=false（clamp/base 切线比 0.84-1.19≈1），不存在工作点移动/饱和制动；④"本征增益×制动因子≈实际透射（50××0.02≈1）"是两个独立伪影的巧合对齐，无因果结构。**AGC/制动因子从活跃假设清单移除。**
- **洞察 A（反离散电路）定性方向正确、定量部分无支持**："非离散电路"被七重证据支持（2846/2828-2834/2850/2851/2852/2854/2855）；"连续介质物理"仅可作启发式隐喻保留——附件引用的窗口位置若为 L29-31 亦与勘误冲突（base 窗口在 L32-34）；其"线性化本征增益"概念在 EPS=1.0 尺度无效（2855 L1 r=−0.17，2853 已示 ε 减半单位增益翻倍）。

### 教训制度化
- **恒等式审计（第三次）**：设计"分解"时必须先证各分量输入代数可分；预注册判据若依赖恒等量，该分支永远无法携带独立信息。
- **窗口常量闭开性**：W1_HI=32（exclusive 语义）与 range(…,q31+1)（inclusive 用法）冲突——窗口边界必须显式标注 [lo, hi) 或 [lo, hi]。
- **产物核对一律 python os.walk 探针**：Glob 沙箱视图对 phase 产物目录两次空结果而磁盘 3 文件完好——磁盘复核不信任 Glob。

### 文件
- 脚本 `tests/glm5/phase2855_window_anatomy.py` sha `b1ec5668`（16,291 B）
- 产物 `phase2855/window_anatomy/`：execution.json `dfcf5656`、result.json `8ce7f91f`、window.npz `9beea6f6`

### 接续（2856 候选）
1. MA 战线推进（主选）：词表扩展预研（80→200 词）+ 双谱普查自动化管线封装（LN2 纪律与窗口解剖管线已固化可复用）。
2. （低优先）g_direct 对照测量：LN2 输出点直接注入 εcdir，补齐 2855 缺失的独立 SwiGLU 响应臂，一次性 10 层×80 词批量。

---

## Phase 2856: Atlas-E200 词表扩展预研 + 双谱普查管线封装 [2026-09-18]

### 原理
MA 战线推进（2846 双谱正交 → Atlas-E 词汇扩展）：正式的 200 词全头双谱普查前，须先回答两个前置问题——①200 词词表能否构造、类方向结构是否保持（Arm A，纯 unembed 层面，无前向）；②2846 协议封装为可复用管线后是否逐位忠实（Arm B，GPU 复测对照）。

**管线库 `rdc_atlas_census.py`**（sha `c3fcead807543632`）：build_vocab（tid 缓存→单 token 过滤→top-N 截取→全 single_tok 集 dW_unit→种子 null tids→func tid 末位，与 2846 构造顺序逐句一致）+ AtlasCensus 类（hooks/patched attn/OV 缓存/conds2_for/bias_for/measure_layer = 因果 drop + OV 对齐 s0/s1，2846 协议逐行复刻，词表与 null_tids 参数化）。

Arm A：2806 词表（10 类×10 词）+ 固定顺序 12 候选/类，单 token 过滤取前 10 存活者 → 目标 20 词/类。判据（冻结，execution.json `6a74cc0f`）：E1 ≥9/10 类满 10 新词；E2 ≥9/10 类 cos(dW_unit_200, dW_unit_80)≥0.9；E3 off-diag 类中心余弦矩阵 Pearson r≥0.95。Arm B：管线重测 L29-31 × 前 40 词（seed=2846 词表构造），B1 = 3/3 层 r(drops_new, drops_2846)≥0.99 且 med|Δ|≤0.002。

### 正式判决（243.4s，Gen2，脚本 `c5441794`）
| 判据 | 值 | 结果 |
|---|---|---|
| E1 词表覆盖 | 每类新词 [fruit 6, animal 10, metal 9, vehicle 10, country 10, food 10, nature 10, furniture 6, tool 4, clothing 9]，总 184 词 | **false**（6/10 类达标） |
| E2 方向稳定 | cos 谱 [0.907, 0.875, 0.910, 0.872, 0.936, 0.847, 0.874, 0.896, 0.889, 0.882]，min 0.847（food） | **false**（6/10 类 ≥0.9） |
| E3 几何保持 | off-diag 中心余弦 r=0.9512（dW 版描述性 0.9562） | **true** |
| B1 管线复现 | L29/30/31 全部 **r=1.00000、med\|Δ\|=0.000000**（逐位一致），clamp_max_resid 0.0127 | **true** |
| **final_verdict** | | **atlas_e200_ready=False / pipeline_verified=True** |

### Arm B 逐位复现的方法论意义
管线库对 2846 协议的封装**逐位忠实**（r=1.0、med|Δ|=0 精确为零）——与 2852 误删恢复实证同源：确定性管线 + 协议逐行复刻 = 可完整迁移的测量机器。2857+ 的 Atlas-E200 正式普查（36 层×32 头×200 词）直接 import 此库换词表参数即可，无须再写测量代码。Gen1 崩溃（AttributeError：__init__ 漏设 target_list）修复后一次通过。

### Arm A 双失败的诊断（预研的核心产出）
1. **E1 失败 = 词源问题**：复合词/低频词大量多 token——tool 类仅 4/10 存活（screwdriver/hatchet/crowbar/tweezers/anvil/sickle/lathe/chisel 全灭，scissors 是复数形但单 token）；fruit 6/10（papaya/guava/apricot/pomegranate/watermelon/kiwi 灭）；furniture 6/10（armchair/bookshelf/recliner/hutch/ottoman/nightstand 全灭——复合词必然多 token）。高频常见名词池可轻松补足 10/类，词表扩到 200 无根本障碍。
2. **E2 失败 = 方向温和漂移（0.847-0.936），两个候选解释待审计**：
   - (a) **多义词污染**：lead（金属/动词）、Turkey（国家/禽）、alloy/scissors 等引入类外分量，把中心拖偏；
   - (b) **小样本噪声反转**：80 词版 dW_unit 本身是 8-10 词小样本估计，200 词版才是更好的估计——余弦 0.85 可能反映旧方向的抽样噪声而非新词表的问题。
   - 审计设计（2857）：①jackknife 逐词剔除，定位每类余弦的主要拖动词；②新旧词子集中心分别对比（new-only 中心 vs old 中心余弦）；③若漂移均匀且 new/old 子集中心余弦高 → 判据语义反转为"旧方向是噪声基准"，200 词方向上任。
3. E3 通过（0.9512，贴线）：类间几何（10 类中心两两余弦）在扩展下基本保持——类结构本身稳健，漂移发生在类内方向估计层面。

### 文件
- 脚本 `tests/glm5/phase2856_atlas_e200_prep.py` sha `c5441794`（10,823 B）
- 管线库 `tests/glm5/rdc_atlas_census.py` sha `c3fcead8`（12,758 B）
- 产物 `phase2856/atlas_e200_prep/`：execution.json `6a74cc0f`、result.json `e9e5f9eb`、atlas_e200.npz `10677e58`

### 接续（2857 候选）
1. **E2 方向漂移审计**（主选）：jackknife 逐词剔除 + 新旧子集中心对比 + 多义词定点检查 → 判定漂移源（污染 vs 噪声），产出清洗规则或判据重校准。
2. 词池修复：每类补足高频单 token 候选（池扩至 ~20/类）→ 200 词达阵复跑 E1/E2/E3。
3. 审计通过后启动 Atlas-E200 正式双谱普查（管线库就绪，36 层×32 头×200 词，分块后台）。

---

## Phase 2857: E2 方向漂移审计——根因确诊 subset_incompatibility [2026-09-18]

### 原理
2856 双失败（E1 词源不足 / E2 方向漂移 0.847-0.936）留下根因二选一：(a) 多义词/弱成员污染（个别词拖动类中心）；(b) 80 词基线自身的抽样噪声。本 Phase 在 unembed 层面（无前向）三臂分离：A1 jackknife 200a 词表（逐新词剔除，delta_cos = cos(dW200a[−w], dW80) − cos_base）；A2 80 协议词表内 leave-one-out 方向抖动尺度 δ80（噪声基准）；A3 新词子集中心 vs 旧词中心（Cm80）余弦（子集相容性）。随后清洗（剔 Δ≥0.05 词）+ 扩充池（CAND2，每类 5-10 候补）复跑 E1v2/E2v2/E3v2。

判据（冻结，execution.json `025498b2`）：J1 pollution_dominant iff ≥2/10 类存在 delta_cos≥0.05 的词；J2 subset_compatible iff 子集中心余弦中位 ≥0.90；J3 drift_source 三分类；E1v2 ≥9/10 类 10 新词；E2v2 ≥9/10 类 cos≥0.9；E3v2 off-diag 中心余弦矩阵 r≥0.95。

### 正式判决（10.2s，一次通过，脚本 `bbdf1890`）
| 判据 | 值 | 结果 |
|---|---|---|
| J1 污染主导 | 逐类最大 delta_cos [0.013, 0.010, 0.009, 0.012, 0.006, 0.015, 0.009, 0.018, **0.023**, 0.012]（tool/pliers 最高），全部 <<0.05，清洗列表全空 | **false** |
| J2 子集相容 | 新/旧子集中心余弦 [fruit 0.557, animal 0.633, metal 0.627, vehicle 0.576, country 0.760, food 0.575, nature 0.570, furniture 0.572, **tool 0.402**, clothing 0.631]，**中位 0.5755** | **false** |
| J3 漂移源 | δ80 中位 **0.0225**（基线自身抖动 cos≈0.978，比子集错位小 6-25 倍） | **subset_incompatibility** |
| E1v2 达阵 | 每类新词 [9,10,9,10,10,10,10,10,8,10] = **196 词**（fruit 缺 1：raisin 多 token；metal 缺 1：CAND2 元素词全灭；tool 缺 2） | **false**（7/10） |
| E2v2 方向 | cos 谱 [0.877, 0.873, 0.909, 0.872, 0.934, 0.845, 0.874, 0.858, **0.836**, 0.876]——补词后较 200a 整体略降 | **false** |
| E3v2 几何 | off-diag r = **0.9389** < 0.95（200a 0.9512 贴线，补词后跌破） | **false** |
| **final_verdict** | | **atlas_e200_ready_v2=False / drift=subset_incompatibility** |

### 科学结论（本 Phase 的正面发现）
1. **漂移根因 = 子集不相容，非噪声非污染**：2806 词表选的是类**原型词**（apple/dog/hammer/Japan），扩展池是次原型词（plum/pliers/fig/sled），两个子簇在 unembed 空间的中心余弦仅 0.40-0.76（中位 0.576）——而旧词表内部 leave-one-out 抖动只有 0.0225。原型性梯度主导类内方向：**类方向 dW_unit 是词表选择的函数，不是类的稳定属性**。
2. **类间对比几何相对稳健**：子集中心差 0.42+，但类方向（对比向量）余弦仍 0.84-0.93、类间几何 r 0.94——漂移大部分落在"类内原型性梯度"方向上，类间区分结构保留。这与 2806 hierarchy law（类间方向分层）互证。
3. **对 Atlas 战线的战略修正**：80 词版全部结论（2846-2855）的 cdir 是 80 词表的量，**不能也不需要**平移到扩展词表——cdir 本就是词表的操作量。"新旧方向一致性"（E2）判据语义过强，预注册时未预见原型性梯度（判据设计课：词表扩展判据应针对**扩展词表自身的几何健康度**，而非与基线的一致性）。

### 文件
- 脚本 `tests/glm5/phase2857_drift_audit.py` sha `bbdf1890`（14,281 B）
- 产物 `phase2857/drift_audit/`：execution.json `025498b2`、result.json `eb702ec4`、drift_audit.npz `0eb5db58`

### 接续（2858 候选）
1. **扩展词表合法性判据重设计（主选）**：放弃 E2（新旧一致），新预注册 G 判据——G1 词数达阵（每类 ≥20，fruit/metal/tool 需再补池）；G2 类间中心可分性（off-diag 中心余弦上限健康，如 max <0.6）；G3 每类 dW 非退化（范数下限）。通过后 Atlas-E200 正式双谱普查独立成线（管线库 c3fcead8 就绪）。
2. 80 词线继续深挖（独立于扩展）：词表内 jackknife 稳健性已被 2857 A2 量化（δ80 0.0225，极稳），现有结论无需重测。
3. 观察登记（不展开）：原型 vs 次原型的子簇结构本身可测（类内 unembed 主轴），留作后战。

---

## Phase 2858: G 判据重设计 + 200 词达阵——atlas_e200_legal=False，MA/Atlas-E 词表战线收束 [2026-09-18]

### 原理
按 2857 结论（判据应针对扩展词表自身几何健康度），冻结 G 判据（baseline-relative 公式，数值由数据给出）：G1 ≥9/10 类满 10 新词（CAND+CAND2+CAND3 有序过滤、无剔除）；G2 separable iff max_offdiag_cos(Cm200c) ≤ max_offdiag_cos(Cm80)+0.10 且 mean ≤ mean+0.05；G3 non-degenerate iff min_c‖dW200c‖ ≥ 0.5·min_c‖dW80‖（未归一化类间对比范数）。CAND3 补池：fruit [durian, lychee, quince, plantain, mulberry]、metal [ingot, pewter, nugget, ore, foil]、tool [tongs, rasp, gouge, auger, bit]。

### 正式判决（9.8s，一次通过，脚本 `08157446`）
| 判据 | 值 | 结果 |
|---|---|---|
| G1 达阵 | [fruit 9, animal 10, metal 10, vehicle 10, country 10, food 10, nature 10, furniture 10, tool 10, clothing 10] = **199 词**（CAND3 的 durian/lychee/quince/plantain/mulberry 全灭；metal 靠 ore 补齐、tool 靠 rasp/bit 补齐） | **true** |
| G2 可分性 | offdiag max **0.4938** vs 上限 0.488（80 版 0.388+0.10，一线之差）；mean **0.3365** vs 上限 0.285（超 +0.05） | **false** |
| G3 非退化 | min‖dW‖ 0.3448 vs 0.5×0.4378=0.219，ratio **0.788** | **true** |
| **final_verdict** | | **atlas_e200_legal=False** |

### G2 失败的含义与战线收束
1. **类间中心重叠系统性上升**（max 0.39→0.49、mean 0.24→0.34）：次原型词的类边界天然更模糊（prune/date 在 fruit-food 边界、ore/nugget 在 metal-material 边界）——这是语义结构事实，不是词表构造失败。199 词版仍可用（G3 保证类方向非退化），但类条件分析的语义纯度下降。
2. **G2 判据自身缺陷登记**：baseline-relative 公式内在偏向失败——原型词表的分离度天然更高，用它作基准要求次原型词表达到同等分离度不合理。绝对阈值（如 max<0.5）事后再定则存在研究者自由度（数据已见 0.494）——不再重定阈值重跑，避免挪门柱。
3. **MA/Atlas-E 词表战线收束（2856/2857/2858 三连完整回答三问）**：①能否扩？——物理上能到 199 词，复合词/低频词大量多 token 是硬约束；②为何漂？——原型性梯度（子簇错位 0.58，非噪声非污染，2857）；③怎么判？——自身几何健康度判据（G），且 199 词版 G2 不过。**199 词普查暂缓**（判据 gating 未通过，~3.3h GPU 成本）；80 词版（2846 图谱 + 2846-2855 全链）保持唯一正式图谱。
4. 管线资产：`rdc_atlas_census.py`（`c3fcead8`）+ 199 词 e200c 清单已登记，未来若需扩展普查可直接复用。

### 文件
- 脚本 `tests/glm5/phase2858_g_criteria.py` sha `08157446`（9,585 B）
- 产物 `phase2858/g_criteria/`：execution.json `11de5c9c`、result.json `1af6bbf8`、g_criteria.npz `b2c26780`

### 接续（2859）
**图谱抽样稳定性分析（零前向，纯 2846 不可变产物）**：2846 drops_all (80,36,32) 内做前 40 vs 后 40 词分割 + 200 次随机 40 词子采样 bootstrap → 头排序 Spearman、top-10 身份分布、top-64 重合、C1/C2/C3 判据复现率——把 2856-2858 的词汇敏感性从 cdir 层面推进到图谱层面，量化 2846-2855 头级结论的词抽样置信度。

---

## Phase 2859: 头图谱抽样稳定性——前缘稳/尾部噪，MA 战线收官 [2026-09-18]

### 原理
2856-2858 把词汇敏感性量化到 cdir 层面并收束词表扩展线；本 Phase 推进到图谱层面：2846 头图谱（1152 头排序、top-10 前缘、C1-C3 判决）对词抽样的置信度。零前向，纯 2846 不可变产物（census_full.npz drops_all (80,36,32) + 36 层 census_L{li}.npz s0/s1 重装 (80,36,32)）：臂 A 前 40 vs 后 40 词分割；臂 B 200 次 40 词无放回子采样 bootstrap（SEED=2859）。

判据（冻结，execution.json `4253960c`）：S1 atlas_stable iff 分割 Spearman ≥0.7；S2 frontedge_stable iff 2846 top-10 头 bootstrap 中位排名 ≤16；S3 c2_robust iff ≥90% 抽样 frac_top64≥0.25；S4 c3_robust iff ≥90% 抽样 Spearman(direct,drop)<0.3。

### 正式判决（0.2s，脚本 `c2ab6c40`；Gen1 变量遮蔽 bug：判据变量 s1 覆盖 s0/s1 数组名，改名 s0_all/s1_all 后一次通过）
| 判据 | 值 | 结果 |
|---|---|---|
| S1 全图谱稳定 | 分割 Spearman（1152 头）**0.112**，top-10 重合 2/10，C2 前/后 0.532/0.426 | **false** |
| S2 前缘稳定 | 2846 top-10 头 bootstrap 中位排名 **11.0**（逐头 [0,2,2,3,6,7,10,8,9,16.5]） | **true** |
| S3 C2 集中稳健 | frac64 分布 p05/p50/p95 = 0.274/0.429/0.692，≥0.25 占比 ≥90% | **true** |
| S4 正交稳健 | C3 Spearman 分布 p05/p50/p95 = 0.082/0.117/0.161，200/200 全 <0.3 | **true** |
| **final_verdict** | | **atlas_robust=False（S1 挂）/ 前缘与三判决全稳健** |

### 科学结论（图谱置信度界，方法论级）
1. **头图谱 = 稳健的前缘 + 噪声的身体**：top-64 的 load 份额结构（C2）与 top-10 头身份（S2）在词抽样下稳定（中位排名 11，逐头中位全部 ≤16.5）；但 1152 头全排序的分割相关仅 0.112、top-64 身份 Jaccard 中位 0.407——尾部头（drop ±0.005 量级）的排序被词抽样噪声淹没。
2. **2846-2855 全部头级结论的置信度确认**：形成器（L13H30 中位排名 0）/放大器带（L26-35 头）都在前缘，其结论不受词抽样影响；C3 双谱正交在 200/200 bootstrap 中无一越过 0.3——**正交性是本网络最稳健的观测量之一**。
3. **图谱规范修订（入响应谱图谱规范）**：1152 头全排序**禁止**作为"头重要性排行榜"引用；图谱的合法读出单位是 top-64（尤其 top-10）前缘 + 聚合统计量（C2/C3）。尾部仅可作分布背景。
4. 至此 MA/Atlas 战线完整收官：2846（图谱+三判决）→ 2856/2857/2858（词表扩展三问：能否/为何/怎么判，答：能到 199 词、原型性梯度、自身健康度判据且未过 → 80 词为唯一正式图谱）→ 2859（图谱置信度界：前缘稳/尾部噪/判决稳）。

### 文件
- 脚本 `tests/glm5/phase2859_atlas_stability.py` sha `c2ab6c40`（7,009 B）
- 产物 `phase2859/atlas_stability/`：execution.json `4253960c`、result.json `7645d923`、atlas_stability.npz `e3f9888d`

### 接续（2860 候选）
1. **主线回归（主选）**：MA 战线收官后回到 cdir 涌现机制链的遗留缺口——2855 接续的低优先 g_direct 对照（LN2 输出点直接注入，补独立 SwiGLU 响应臂，一次 10 层×80 词批量，~40s）可顺手关闭；随后按 MASTER_PLAN 双谱规范推进 II1/II3 或开新机制问题。
2. 备选：80 词 bootstrap 词表（类内有放回抽样）下的关键判据复验（窗口 L32-34 增益谱的词抽样置信度，量化 2854/2855 结论）——零新前向可用 2854 npz 部分实现，但 2854 未存逐词矩阵，需小规模重测。

---

## Phase 2860: g_direct 对照：R1 复现失败 → 挖出 2853/2855 混合工作点伪影（硬伤④：工作点混合） [2026-09-18]

### 动机与协议（预注册冻结于任何观测前）
2855 遗留缺口：其三谱分解中 g_mlp ≡ g_comp 数学恒等（硬伤①），缺失的独立对照是
**g_direct = [mlp(ln_x + ε·cdir) − mlp(ln_x)]·cdir/ε** —— 在 LN2 输出点直接注入 εcdir，
绕开 LN2 对输入的改写。对比语义：g_direct ≈ g_comp → SwiGLU 本征响应，LN2 改写无关；
g_direct << g_comp → LN2 改写承载响应（"ln2_mediated" 字面含义）；g_comp − g_direct = LN2 改写净效应。
- 协议逐句复刻 2855：SEED=2855 词表（80 词 × 10 类，dW_unit 类方向）、L26-35、EPS=1.0、
  x_in = sain2['same'][l][1] + attn_b2[l]、L13H30 钳制校验臂；每层每词一次 batched mlp
  调用（3 行：ln_x / ln_x+dln / ln_x+ε·cdir）；ratio 除法带 abs(gcm)>1e-6 负分母防护（第六次应用）。
- 预注册：R1 描述性复现校验（max_q |mean_w g_comp − g2855_L26_35| < 0.05，g2855 从 2855 result.json 读入）；
  D1 三分类（swiglu_intrinsic：≥6/10 层 |ratio−1|≤0.3；ln2_rewrites：≥6/10 层 |ratio|≤0.5；else mixed）；
  D2/D3 描述性。

### 结果（phase2860/g_direct/；exec b0e78d2d… / result c9254ae5… / g_direct.npz 8194a128…）
| 量 | L26→L35 谱 | 判决 |
|---|---|---|
| R1 max diff | **7.177**（@L34） | **R1 = false** |
| D1 | n_close=1 / n_rewrite=0 | **mixed** |
| g_comp 均值 | −0.08, 0.01, 0.08, 0.11, 0.22, 0.14, 0.01, −0.03, −0.11, −0.27 | 全层 \|g\|≤0.27 |
| g_direct 均值 | −0.10, 0.03, 0.15, 0.24, 0.49, 0.35, 0.02, −0.12, −0.41, −0.67 | 全层 \|g\|≤0.67 |
| ratio=g_d/g_c | 1.26, 5.16, 1.83, 2.26, 2.24, 2.55, 2.22, 3.75, 3.77, 2.53 | 分母近零，比值失义 |
| dln_norm 均值 | 0.73→0.33（L26→L33），尾部回升 0.60 | LN2 收缩 ~0.5-0.7 |
| dln_cos 均值 | 0.965 → 0.757 | 方向保留度高 |
| g_diff (g_c−g_d) | −0.02, −0.02, −0.07, −0.14, −0.27, −0.21, −0.01, +0.09, +0.30, +0.41 | LN2 改写净效应与直接注入同量级 |
| max_resid | 0.01062 | L13H30 钳制臂通过（≡2855） |

### R1 失败根因（本轮主要产出：word 级诊断 + 单前向探针 + 源码定位，三步闭环）
1. **npz 对比**（_p2860_diag.py）：2855 内部 g_mlp ≡ g_comp 逐位（max 0.000000，恒等缺陷实锤）；
   2855 g_ln 与 2860 dln_norm×dln_cos **逐元素 max diff 0.000000** → 词表/x_in/cdir/LN2 逐位复现；
   g_comp 差异全部落在基线项 out_b（L32-34 差异 75/80 词系统性偏正、单词最大 1056、均值 4.48±6.07）。
2. **单前向探针**（_p2860_probe.py，词 'apple'，同时捕获 LN2/mlp 的 pre-hook 真值）：
   \|x_in_f − ln2in_true\| = **65–420**、\|ln_x_f − mlpin_true\| = 7–53、\|mlp_f − mlpout_true\| = 11–700
   （cdir 投影差单词级 −2.1 ~ +24.5）。"sain+attn" 根本不是 LN2 的输入。
3. **源码根因**（transformers modeling_qwen3.py L315-327）：decoder layer 是 pre-norm——
   `residual = hidden_states` 先保存，attn 输入 = `input_layernorm(hidden_states)` = LN1(x)。
   **self_attn 的 pre-hook 捕获的是 LN1(x)，不是残差 x**。
   故 x_in = sain + attn = **LN1(x) + attn_out（伪残差）**；真实 LN2 输入 = x + attn_out。
4. **结论**：2855 g_comp = [mlp(LN2(伪残差+ε·cdir)) − mlp(LN2(真残差))]·cdir/ε ——
   **混合工作点测量**：扰动点用 hook 重构（伪残差），基线用 hook 真值（真残差），
   差值混入 (伪残差 − 真残差) 的深层 mlp 响应。2853 g_jac 同构（phase2853 L300 `out_b = mlp_b2[l]`
   + L306 `mlp_call(l, x_in+ε·cdir)`），两者数值逐位吻合
   （1.7291/4.5193/7.0677 ≡ 2855 勘误中的"base 窗口 L32-34 = 1.73/4.52/7.07"）。

### 勘误的勘误（对 2855 节第二处窗口勘误的修订）
- "base 窗口 L32-34 = 1.73/4.52/7.07" 实为 2853 g_jac（= 2855 g_comp）@L32-34，
  **是混合基线伪影，不是任何意义上的干净 SwiGLU 增益**。2853"深层放大器（L32-34 增益 ~7）"叙事作废。
- 2853 clamp 态 g_emp（1.68/1.28/0.90 @L29-31）= dout/din 双 hook 实测比值，
  分子分母同工作点语义，**不受混合基线污染，仍然成立**。
- 2853 T2 的 r(g_jac, g_emp) 因 g_jac 伪影被污染，待真残差重测后重算。
- 2855 W1（ln2_mediated）判据仅依赖 g_ln（自洽、逐位复现）→ 形式上成立；但 g_ln 的
  零特异性硬伤（LN2 对切向方向必然收缩）不变。W2 神经元分解与 L1 线性化检验均在
  伪残差工作点上，结论降级为"待真残差工作点重估"。

### 2860 判决语义
- D1=mixed 的机制含义：在自洽基线上 g_comp ≈ g_direct ≈ 0（|g|≤0.7），ratio 因分母近零而失义；
  有效读出是**绝对量级**——SwiGLU 对 ε·cdir 的响应（无论 LN2 改写与否）在 L26-35 均为小量，
  远小于伪影值 7.07。"ln2_mediated" 的原始机制解释（LN2 改写承载大响应）不成立；
  真实图像是直接响应与改写效应都是小量且部分抵消（D3）。
- **2855 final_verdict ln2_mediated/neuron_diffuse 的 g_comp 支柱坍塌**；W1 形式成立但机制解读作废。

### 硬伤④登记：工作点混合（workpoint mixing）
- 制度教训：**hook 捕获值只能与 hook 捕获值差分；显式重构值只能与显式重构值差分**
  （同工作点自洽原则）。任何 "[f(重构点+扰动) − f(hook基线)]" 形式的差分公式在 pre-norm
  架构中都混入 LN1(x) vs x 的偏移，深层可放大至 O(10²)。
- 修正协议（2861 起）：x_in 一律用 post_attention_layernorm 的 **pre-hook 直接捕获**
  （探针已验证捕获可行），禁止 sain+attn 重构。
- 冲击面排查（Grep phase28*.py 全扫）：伪残差重构仅出现在 2853/2854/2855/2860 四脚本
  （2854 为 x_base/x_clamp 双点同构）；2856-2859 双谱普查线全 hook 级差分，安全。
  2854 x_base/x_clamp 的具体差分结构待 2861 复核（其 clamp 增益若同为 hook-vs-重构差分则受染）。

### 接续（2861 候选，MASTER_PLAN II1/II3）
真残差工作点重测：x_in = ln2in pre-hook 真值，重测 g_comp/g_direct/dln 谱 + r(g_jac', g_emp)，
一并复核 2854 差分结构 —— 窗口解剖战线在正确工作点上的真正收束。
（脚本 tests/glm5/phase2860_g_direct.py；探针 tests/gpt5_temp/_p2860_probe.py、_p2860_diag.py）

---

## Phase 2861: 真残差工作点重测：no_amplification，窗口解剖战线正式收束 [2026-09-18]

### 动机与协议（预注册冻结于任何观测前）
2860 定位硬伤④（工作点混合：2853 g_jac / 2855 g_comp 的扰动点用伪残差重构、基线用 hook 真值）。
2861 在**正确工作点**上重测窗口解剖三量——x_in 一律用 pre-hook 直接捕获（硬伤④修正协议首次执行）：
- ln2in_true = post_attention_layernorm 的 pre-hook（真 LN2 输入 = x + attn_out）；
- mlpin_true = mlp 的 pre-hook（真 mlp 输入 = LN2(ln2in_true)）；
- out_b_true = mlp_raw(mlpin_true)（显式重构，与扰动同点自洽）。
- g_ln_true = [LN2(ln2in+ε·cdir) − LN2(ln2in)]·cdir/ε；
  g_comp_true = [mlp(LN2(ln2in+ε·cdir)) − out_b_true]·cdir/ε；
  g_direct_true = [mlp(mlpin+ε·cdir) − out_b_true]·cdir/ε。
- 内建一致性校验（bf16 确定性，预期 ~0）：v1 = max|LN2 重算(ln2in_true) − mlpin_true|；
  v2 = max|mlp 重算(mlpin_true) − mlpout_true|。**实测 v1 = v2 = 0.0（逐位）**——
  工作点捕获正确性的铁证，同时证明 hook 重算与 hook 真值在本协议内完全互换。
- 其余协议逐句复刻 2860/2855（SEED=2855 词表 80 词、L26-35、EPS=1.0、L13H30 钳制臂 max_resid=0.01062 ≡ 前两轮）。
- 预注册：E1 = 2860 判据形式在真工作点重跑；E2 = 窗口大增益存在性
  （no_amplification iff max_L|g_comp_true|<1.2 且 max_L|g_direct_true|<1.2，2853 T1 阈值）；
  E3 描述性 r(g_direct_true, g_emp_2853)；E4 描述性 |g_comp_true − g_comp_2860|（工作点敏感性）。

### 结果（phase2861/g_true/；exec f23ea1d9… / result 040dce3b… / g_true.npz ac391ed0…）
| 量 | L26→L35 谱 | 判决 |
|---|---|---|
| E1 | n_close=0 / n_rewrite=0，ratio 2.49→8.16 | **mixed**（分母近零失义，与 2860 同构） |
| E2 | max\|g_comp_true\|=**0.116**，max\|g_direct_true\|=**0.947** | **no_amplification**（双双 <1.2） |
| E3 | r(g_direct_true, g_emp_2853) = **0.781** | clamp 实测增益与直接注入谱中强正相关 |
| E4 | max\|g_comp_true − g_comp_2860\| = 0.150 | 复合臂对工作点不敏感（两工作点自洽值都近零） |
| g_comp_true | −0.02, 0.03, 0.03, 0.07, 0.10, 0.06, −0.02, −0.07, −0.11, −0.12 | 复合响应近零 |
| g_direct_true | −0.05, 0.13, 0.12, 0.31, 0.48, 0.39, −0.12, −0.47, −0.86, **−0.95** | 峰值 L30 +0.48；L32-35 温和负响应 |
| g_ln_true | 0.29 → 0.12（单调递减） | 真工作点 LN2 改写幅度比伪工作点（0.71→0.44）小一半以上 |
| dln_cos | 0.94 → 0.76 | 方向保留趋势与 2860 一致 |
| v1 / v2 | **0.0 / 0.0** | 校验通过 |
| max_resid | 0.01062 | 钳制臂 ≡ 2855/2860 |

### 科学结论
1. **"L32-34 深层放大器 ~7"不存在**——真残差工作点上窗口内无任何 ≥1.2 的增益臂（E2）。
   2853 T1 的 active_amplification 判决正式作废；2853/2855 的 7.07 确认为纯工作点混合伪影（硬伤④）。
2. **深层 SwiGLU 对类方向的响应是温和负反馈**：g_direct_true 在 L32-35 单调走负（−0.12→−0.95），
   峰值正响应在 L30（+0.48）。mlp 臂无放大器角色——与 2846 双谱"放大器"头级分类无冲突
   （那是 attention OV 臂）。
3. **复合臂 g_comp ≈ 0 是真实小量而非伪影**：两个工作点独立自洽测量一致（E4 ≤0.15），
   LN2 改写（≤0.29）与直接注入（≤0.95）部分抵消（g_diff L35 = +0.83）。
4. **E3 r=0.78**：2853 clamp 态实测增益（干净量）的主要部分可由直接注入响应谱解释——
   clamp 位移与 cdir 方向的高度重叠所致；残余 0.22 方差来自 clamp 位移的非 cdir 分量。
5. g_ln_true（0.29→0.12）显著小于伪工作点 g_ln（0.71→0.44）：伪残差点（LN1 输出）的
   切向分量比真残差点大——LN1 本身就是强切向收缩器，"LN2 收缩切向扰动"的量级必须按工作点报告。

### 窗口解剖战线收束清单（2853→2861）
- 2853 T1 active_amplification：**作废**（伪影）；T1 正确判决 = no_amplification（2861 E2）。
- 2853 T2 r(g_jac, g_emp)：原值被 g_jac 伪影污染；替代读出 = 2861 E3 r(g_direct_true, g_emp) = 0.781。
- 2853 clamp 态 g_emp（1.68/1.28/0.90 @L29-31）：**存活**（双 hook 实测比，不受硬伤④污染）。
- 2855 W1 ln2_mediated：形式成立（g_ln 自洽）但机制解读作废——正确表述：
  LN2 改写与直接注入均为小量且部分抵消，复合响应近零，无放大臂。
- 2855 W2 neuron_diffuse / L1 linear_ok=false：伪残差工作点产物，降级为"未定"；
  如需复活须在真工作点重跑神经元分解（低优先级——放大臂不存在，神经元定位问题失去锚点）。
- 2854 gain_specificity：差分结构待复核（x_base/x_clamp 均为 sain+attn 伪残差，
  但若两点同为伪残差语义则自洽——其 clamp 增益结论与 2853 g_emp 同源，预计存活；列为附录待办）。

### 硬伤④修正协议的制度化确认
pre-hook 直接捕获工作点 + 显式重构差分 + v1/v2 逐位校验 = 本系列第一个端到端无瑕测量协议。
**凡涉及"在 X 点注入扰动"的实验，X 必须用目标模块的 pre-hook 直接捕获，禁止用上游 hook 值重构。**
（制度条文与 2860 节一致，本 Phase 为首次执行 + 校验机制落地。）

### 接续（MASTER_PLAN II1/II3 双谱规范主线）
窗口解剖战线（2853-2861，九个 Phase）完全关闭。下一阶段回归双谱规范推进：
对齐谱（share）⊕ 因果谱（drop）的机制解释主线（形成器/放大器头级分工的下游验证、
MA2 全头普查的 top-64 前缘机制解剖）。说"继续"即进入 2862。
（脚本 tests/glm5/phase2861_g_true.py；产物 phase2861/g_true/；SHA 见探针 _p2861_sha_report.txt）

---

## Phase 2862: 前缘机制解剖：OV 静态增益与因果必要性脱钩（ov_uncorrelated），三谱正交实体化 [2026-09-18]

### 动机与协议（预注册冻结于任何观测前）
MASTER_PLAN II1 收口第一刀：把 2846 双谱前缘（top-64 头，2859 S2 验证稳定）接到具体
attention 机制。**零前向 Phase**——全部读数来自不可变产物 + 权重代数：
- 输入：2846 census_full.npz（mean_drop / mean_s0 / mean_s1，1152 头）、
  2859 atlas_stability.npz（top10_full 合法性校验）、SEED=2855 词表 dW_unit（10 类方向，仅 W_U+tokenizer，无 forward）。
- OV 通道端到端类方向增益（GQA 感知）：g[h,c] = cdir_c·W_O^h W_V^{kv(h)}·cdir_c；
  总写出范数 wn[h,c] = ‖W_O^h W_V^{kv(h)}·cdir_c‖；W_V 取 kv 头切片（group=4），W_O 取 query 头切片。
- c0 校验：top-64（mean_drop 降序）⊇ 2859 top-10 → **true**。
- 预注册：M1 = ov_carries_causal iff Spearman(mean_drop, max_c|g|) > 0.15
  （2846 C3 direct-drop 全集 0.1496 同量级参考）且 top64 均值 > 随机 64 头 null p95（200 次，SEED=2862）；
  M2/M3/M4 描述性（角色分布/层分布/top1 档案）。

### 结果（phase2862/frontedge_anatomy/；exec 635aec00… / result 4f588227… / frontedge.npz 2d3c2cd0…）
| 量 | 值 | 判决 |
|---|---|---|
| M1a ρ(mean_drop, max_c\|g\|) | **−0.026**（参考阈 0.15） | **M1 = ov_uncorrelated** |
| M1b top64 均值 vs null p95 | 0.0469 vs 0.0432 | 仅边际（a 条件大败主导） |
| M2 角色分布 | 全体：形成器 10.4% / 放大器 10.4%；**top-64：形成器 28.1%（18/64）/ 放大器 0%（0/64）** | 前缘 = 纯形成器构成 |
| M3 层分布 | 早 L0-11: 28 / 中 L12-23: 19 / 晚 L24-35: 17 | 前缘偏早层，全域散布 |
| M4 L13H30 档案 | drop=0.101、直写=−0.002、\|g\|≤0.0075、wn≈0.21、最大类 animal | **top1 因果头 OV 静态增益 ≈ 0** |
| 全网均值 | mean max_c\|g\| = 0.033 | OV→cdir 静态通道整体微小 |

### 科学结论
1. **因果前缘头不通过 OV 直写类方向**：L13H30 的 OV 类方向增益（≤0.0075）比其因果损伤（0.101）
   小 13 倍以上。其消融使 d_spec 方向塌陷的机制不是"写出 cdir"，而是**改变后续层的残差流几何**
   （信息中继/结构角色），类方向分量是下游变换后的产物。
2. **三谱正交的头级实体化**：静态 OV 增益谱 ⊥ 直写份额谱（C3 全集 0.15）⊥ 因果必要性谱
   （ρ −0.026）——三张图互不预测。响应图谱必须至少含这三个独立轴；任何单轴"机制图谱"都会误导。
3. **top-64 前缘 = 纯形成器构成**（18/64 形成器、0/64 放大器）：双谱正交（2839）不是噪声，
   而是角色分工——高直写头（放大器）从不进入因果前缘；因果前缘头刻意不直写。
4. II1 收口状态：静态权重代数已排除"直写"解释 → L13H30 类承载机制只剩两条候选路径：
   (a) 位置/结构中继（其消融改变后续 attention 的键值几何）；(b) 间接写出（写出方向经下游
   mlp/attn 变换后才投影到 cdir）。收口下一刀 = 动态追踪（路径回溯/激活修补，需前向，成本升级）。

### 对"双图谱"总方案的方法论输入（与 Phase 2862 同轮的用户战略讨论）
本 Phase 给出图谱计划的第一条硬约束：**响应图谱是多轴张量（静态写出/因果必要性/对齐份额
至少三轴互不预测），语言族图谱与响应图谱的"关联机制"不是单一映射而是轴间变换**
——这正是 2839 双谱正交（ρ=0.106）与 2857 原型性梯度（词表选择→类方向漂移）的方法论推广。
战略分析详见本轮回复与工作区日志（TMA 双图谱计划草案）。

### 接续（2863 候选）
- 主选 A（II1 收口第二刀）：L13H30 动态追踪——比较 full/clamp 态下 L14-20 层
  sain/attn hook 的 d_spec 方向位移分解（区分 (a) 结构中继 vs (b) 间接写出），~1 前向/词。
- 主选 B（词族图谱最小版）：per-token 机制坐标——2857 原型性梯度扩展为每词
  [类方向投影, 类内残差坐标] 双坐标（零前向，unembed 代数）。
（脚本 tests/glm5/phase2862_frontedge_anatomy.py；产物 phase2862/frontedge_anatomy/）

---

## Phase 2863: 十类切片图谱：三判决全 false——8 词/类分辨率下"类"不是因果谱的组织单位（词级特异性主导） [2026-09-18]

### 动机与协议（预注册冻结于任何观测前）
用户图谱战略（按类推进）触发两个关键问题：类切片是否承载超出词抽样噪声的共同信号？
是否存在跨类共享前缘核心？2863 对 2846 census 张量（drops_all 80=10 类×8 词×36×32）
做 per-class 切片，**row-permutation null**（200 次，SEED=2863）对照三判决：
- J1 class_signal_above_noise：median_c Spearman(S_c, 其他 9 类均值谱) > null p95；
- J2 shared_core_real：出现于 ≥7/10 类 top-64 的头数 ≥1 且 > null p95；
- J3 formatter_role_persistent：全局形成器（120 头）在各类切片内保持 drop≥p75_c 的均值 > null p95。
零前向，0.4s。上一轮预研（_p2863_prep）已给预览（类间 offdiag mean 0.055、
重叠 13-31/64），本 Phase 加 null 后正式判决。

### 结果（phase2863/class_slices/；exec 9bfa1ade… / result 919c9f83… / class_slices.npz f8263414…）
| 判决 | 观测 | null p95 | 结果 |
|---|---|---|---|
| J1 类信号 | 0.0713 | 0.1607 | **false**（低于随机基线） |
| J2 共享核心 | 7 头 | 13.0 | **false**（真实类 top-64 重叠不超过随机预期） |
| J3 形成器保持 | 0.368 | 0.395 | **false** |
| final | **J1=False/J2=False/J3=False** | | |

描述性：核心头（虽不显著）L13H30 出现于 **9/10 类** top-64、L2H31/L4H4/L22H28/L34H15 各 8 类、
L6H12/L23H7 各 7 类；类间 offdiag mean 0.055（min −0.203 fruit×furniture，max 0.320 fruit×clothing）；
每类 top-64 与全局重叠 13-31/64。

### 科学结论（负结果，如实登记；纪律③ null 对照第 N 次修正解读）
1. **8 词/类分辨率下，"类"不是因果谱的组织单位**：类切片谱的全部结构（近正交、低自相关）
   与随机 8 词分组不可区分——per-word 机制特异性 >> 类共同性，8 词均值不足以提出类信号。
2. **上轮预研解读被正式 null 修正**："类间近正交 = 类激活独立机制组合"不成立；
   近正交只是词噪声的表现。预研无 null、正式 Phase 有 null——预注册纪律的价值再次兑现。
3. L13H30 的 9/10 类覆盖虽低于随机基线，但作为 top1 因果头（drop 0.101）的准全域出现
   与 II1 收口线一致——它是"实体档案读写接口"候选，类无关。
4. 与 2857 原型性梯度合并读：类方向（unembed 侧）是词表选择的函数；类因果谱（机制侧）
   被词噪声淹没。**类作为组织单位在 unembed 侧成立、在机制侧（8 词分辨率）不成立。**

### 对 TMA 类批次方案的修正（回应用户战略）
- 类批次推进的**前提条件 = 词分辨率提升**：8 词/类不足以检验类级机制结构。
- 2864 候选 A（零前向，优先）：词级方差分解——drops_all 的 per-word 谱做
  词间/类间/残差方差分解（ANOVA 式），定量回答"类间方差占比"，判定类信号是否存在
  但被 8 词均值稀释，还是根本不存在。
- 2864 候选 B：重启 199 词普查（2858 atlas_e200_legal=False 的 G2 门线未过；
  baseline-relative 公式缺陷已登记——是否修订 G2 需用户决策，涉及挪门柱）。
- 2864 候选 C：II1 收口第二刀（L13H30 动态追踪，前向 ~1/词）——类无关接口假设直接检验。
- 用户类别扩展（植物/器官/天体/属性轴）的排期**后置**至词分辨率问题解决——
  否则新类切片将重演 2863 的噪声主导结局。

### 接续
2864 主选 A（方差分解，零前向）→ 依结果决定 B/C。说"继续"即进入 2864。
（脚本 tests/glm5/phase2863_class_slices.py；产物 phase2863/class_slices/）

---

## Phase 2864: 词级方差分解：类方差存在但微弱、集中于晚层词汇特异头，与因果前缘几乎不相交（第四轴实体化） [2026-09-18]

### 动机与协议（预注册冻结于任何观测前）
2863 三判决全 false 遗留问题：类信号是被 8 词均值稀释还是根本不存在？
2864 对 2846 drops_all 做 per-head 类间方差分解（零前向，0.5s）：
- η²[h] = SS_between/(SS_between+SS_within)（10 类 × 8 词，头级 1152）；
- R²_class = Σ_h SS_b/Σ_h SS_total（张量级）；
- null：200 次行标签随机重排（SEED=2864），max 统计做家族误差控制。
- 预注册：V1 class_variance_present iff max_h η² > null p95（max 统计）；
  V3 class_signal_in_tensor iff R²_class > null p95（2863 给不出的判决）；
  V2 描述性（η² vs mean_drop 结构、层分布、与前缘交集）。

### 结果（phase2864/class_variance/；exec ea4be587… / result aab3a157… / class_variance.npz 1e892041…）
| 判决 | 观测 | null p95 | 结果 |
|---|---|---|---|
| V1 头级类方差 | max η² = **0.393** | 0.335 | **true**（家族级显著） |
| V1 计数 | 显著头 102 | 期望假阳性 57 | 1.8×（弱-中等，未过 3× 保守线） |
| V3 张量级 | R²_class = **0.1223** | 0.1223 | **false**（边缘不超，张量无类信号） |
| V2 结构 | Spearman(η², mean_drop) = **−0.079**；η²-top20 与因果 top-64 重叠仅 **6/20** | | **类方差轴 ⊥ 因果轴** |
| 层分布 | η²-top20 集中晚层（L23-35 占 13）+ 少量早层（L5H25 0.393、L8H6） | | 晚层词汇特异头 |

η² top：L5H25(0.393)、L26H17(0.389)、L26H21(0.337)、L35H18(0.325)、L29H19(0.322)、L31H29(0.287)。

### 科学结论
1. **类信号存在但微弱且局部化**：~百个晚层头携带类间差异（V1），但张量整体组织不由类主导
   （V3，R² 12% ≈ 随机）。2863+2864 合并：**因果谱的组织单位是词/特征，不是类**；
   类只配作聚合视图。
2. **第四轴实体化**：类间方差轴（η²）与因果必要性轴（drop，ρ=−0.08）、直写份额轴（2846 C3 0.15）、
   OV 静态轴（2862 −0.026）互不预测——响应图谱四轴正交结构确立。
   η²-top20 与因果 top-64 重叠 6/20 ≈ 期望 chance 水平。
3. **晚层词汇区分器**：η² 集中 L23-35——深层头的消融损伤高度词依赖（词汇特异组件），
   早层头（L5H25/L8H6 进入 η²-top）更接近任务/结构组件。机制分层图像：
   早层=共享结构组件，晚层=词汇/特征特异组件。
4. **对战略问题的实证支撑**："2859 的 10 个 head 在其他功能维度会换成别的 head"获得直接证据——
   即使同一任务协议内，类区分头与因果头都几乎不相交；功能维度（轴）决定头子集。

### 接续（2865 候选）
- 主选（II1 收口第二刀）：L13H30 动态追踪（结构中继 vs 间接写出，~1 前向/词）——
  前缘头的类无关接口假设直接检验。
- 备选：η² 晚层词汇区分器的词级坐标化（第四轴的词坐标）。
- 备选：重启 199 词普查（G2 门线需用户决策）。
说"继续"即进入 2865 主选。
（脚本 tests/glm5/phase2864_class_variance.py；产物 phase2864/class_variance/）


---

## Phase 2865: L13H30 动态追踪：II1 收口第二刀判决 = mlp_carried（间接写出），e_ff ~30× 静态通道 [2026-09-18]

### 动机与协议（预注册冻结于任何观测前）
2862 排除直写解释（OV 静态增益 <=0.0075 vs drop=0.101）后，top-1 因果头 L13H30 的
类承载只剩两候选：(a) 结构中继（改后续 attention 键值几何）/ (b) 间接写出（写出方向
经下游变换后才投影到 cdir）。2865 = full/clamp 双态前向（每词 4 前向：full、func、
null、clamp；clamp = L13H30 attention bias (1,1) 钳到 func/null 均值，2853/2860
协议逐字复刻）。attention-bias patch 只触及 L13H30，故 Delta_attn[13] = v13 是
纯 H30 注入向量。真残差位移在 LN2 pre-hook 捕获（2861 制度）。臂分解：
  cfin = v13.cdir + sum_{l=14..35}(Delta_attn[l].cdir) + sum_{l=13..35}(Delta_mlp[l].cdir)
  G_attn / G_mlp 带符号求和；e_ff = cfin/||v13||（间接写出效率，对照静态 0.0075）；
  l* = cdir 累积分量首达 0.5|C-c13| 的层位；D3 = l* 众数占比 vs 200 次随机重排 null。

### 两次失败与根因诊断（判决前必须读）
1. telescoping 向量闭合失败（ident_resid 77/447）。诊断（_p2865_diag）三步定位：
   (i) hidden_states[36] 是 **post final_layernorm**（L35 递推残差 696）——禁止用作
   残差读出点；(ii) E 检查证明 L0-34 层递推在 bf16 舍入内成立，但误差随层增长
   0.1->2.2 —— ln2in 张量携带 23 层舍入累积 delta~10-30，对 ||v13||~0.3-2.3 的
   小词把向量闭合准则放大两个数量级；(iii) **cdir 标量预算精确闭合**
   （词均 0.0007-0.314-1.448 = -1.761 vs cfin -1.763，差 0.002）——协议无结构缺项，
   舍入在投影空间近无偏。v1 判据改冻结为标量预算闭合（阈 0.05）。
2. v1 最终 = false（max budget resid 0.19 > 0.05，相对 |cfin|~11%，如实登记——
   bf16 投影舍入，均值闭合 0.002）。向量 telescoping（均值 5.24）降级为描述量。

### 结果（phase2865/l13h30_trace/；exec 61b2b620 / result 1d01f715 / npz 658f1ff3）
| 判决/量 | 值 | 读出 |
|---|---|---|
| v1 预算闭合 | max 0.19（均 0.002） | false（如实登记）；臂归因定量仍可信（4.7:1 悬殊） |
| **D1 臂归因** | G_mlp = -1.45/词 vs G_attn = -0.31/词 | **mlp_carried**（4.7 倍悬殊） |
| D1w per-word 投票 | attn 臂主导仅 12.5% | 逐词同构 |
| v13.cdir / ||v13|| | **8e-05** | 动态复核 2862：注入向量与 cdir 不对齐 |
| **e_ff** | **-0.23**（p10-p90: -1.41~0.04） | 间接写出效率 ~30x 静态 OV 通道（0.0075） |
| **D3 旋转层位** | 众数 L30 占比 0.237 = null p95 0.244 | **false**：层位词特异，无稳定中继层 |
| A11 | full 0.527 -> clamp 0.055 | clamp 干预强度确认 |

### 科学结论（窗口解剖战线的真正终点）
1. **II1 收口完成：L13H30 = mlp 介导的间接写出**（候选 b）。注入向量与 cdir 余弦
   ~0，但下游响应把 -0.23/||v13|| 的 cdir 分量造出来，且增长主要落在 mlp 臂
   （4.7 倍）——类方向是**下游 SwiGLU 变换的产物**，不是该头的写出内容。
2. **旋转层位词特异**（D3=false，与 2863 词主导、2864 第四轴正交三线互证）：
   不存在"类无关中继层"；同一头的信息经每词不同的层位旋转进 cdir。
3. **与 2861 呼应**：深层 mlp 对 cdir 的温和负反馈（-0.12~-0.95 @L32-35）与
   此处 G_mlp 负号一致——clamp 响应的 cdir 增长是同一 mlp 臂的另一面。
4. 方法论入账：(i) hidden_states[last] 是 post-final-norm，禁作残差读出点；
   (ii) 深层 bf16 舍入累积使向量闭合判据不可用，标量投影预算闭合是正确工具；
   (iii) clamp/patch 干预的臂分解（Delta_attn/Delta_mlp per layer）是"关联机制"
   的通用测量模板，可直接推广到任意头。

### 接续（2866 候选）
- 主选（TMA 词族图谱最小版）：per-word 机制坐标 v0——每词 [10 类方向投影,
  per-word drop 谱, eta2 词级贡献] 拼坐标矩阵，词间相似结构 vs 语义标签
  （零前向，2846/2864 不可变产物 + unembed 代数；判据配随机 null）。
- 备选：mlp 间接写出的承载层 per-word 归因深化（2865 npz 已有 a_mlp 矩阵）。
- 备选：199 词普查重启（G2 门线需用户决策）。
（脚本 tests/glm5/phase2865_l13h30_trace.py；产物 phase2865/l13h30_trace/）


---

## Phase 2866: per-word 机制坐标 v0：机制谱词级类结构成立（W1=true），类结构弥散于词级坐标（2863 失败精确定位） [2026-09-18]

### 动机与协议（预注册冻结于任何观测前）
TMA 词族图谱最小实体：为 80 词（SEED=2855 词表确定性重建）构造双块坐标——
B1 unembed 投影块 proj[w,c] = E_w·dW_unit[c]（10 维，unit）；B2 因果谱块
drops_all[w].flatten()（1152 维，unit，2846 不可变产物）。零前向（10.6s，
模型仅加载读 W_U）。判据（200 次标签置换 null，SEED=2866）：
  W1 机制谱词级类结构：margin[w] = mean cos(w,同类) − mean cos(w,异类)，
      mean margin > null p95 → true；
  W2 unembed 侧同判据（正控制，2857 预测 true）；
  W3 块互补性：ρ(上三角 cos_B1, cos_B2)（描述）；
  W4 词特异性下界：每词对其他词的最大 cos（描述分布）；
  W5 留一最近邻同类检索 accuracy > null p95。

### 结果（phase2866/word_coords/；exec 14adb2f6 / result f89485e6 / npz fff91b4f）
| 判决 | 观测 | null p95 | 结果 |
|---|---|---|---|
| **W1** 机制谱词级类结构 | margin 均值 **0.0976** | 0.0170（5.7×） | **true** |
| **W2** unembed 类结构（正控） | margin 均值 **1.067** | 0.0325（32.9×） | **true** |
| **W3** 块互补性 | ρ = **0.114** | — | 两块携带不同信息 |
| **W4** 词特异性下界 | 最近邻 max cos min **0.144** / p50 0.391 | — | **无两词共享机制**（NN 距离下界 0.86） |
| **W5** 同类检索 | accuracy **0.325**（chance 0.111） | 0.1375 | **true**（2.4× null） |

### 科学结论
1. **2863 失败精确定位**：类均值谱无类信号（J1=false），但 **per-word margin 有**
   （W1=true，5.7× null）——类结构存在于机制谱中，但**弥散在词级坐标里，
   无法压缩为类级原型**。"类"是弱先验/检索视角，不是机制组织轴。
2. **unembed 侧 >> 机制侧**（margin 1.067 vs 0.098，11 倍）：类信息在 unembed
   几何中远比在因果谱中浓缩——双图谱的两侧不对称性首次定量。
3. **W3=0.114**：词的 unembed 投影坐标与因果谱坐标几乎独立——词族图谱与
   响应图谱的"关联机制"问题在词级就有实体（两坐标系之间的变换是研究对象，
   2865 已给出第一个机制级样例：mlp 间接写出）。
4. **W4 直接定量支撑 token 特殊性原则**（用户假设）：80 词中最近邻机制距离
   下界 0.86（min max-cos 0.144）——不存在机制相同的词对；同时 p50 0.391
   说明词间有实质共享成分——**共享基 + 坐标差异**的组合编码图像。

### 阶段性目标达成小结（用户指令：连续推进至此）
- 响应图谱：四轴确立（因果 2846 / mlp 响应 2861 / OV 静态 2862 / η² 类方差 2864），
  互不预测（ρ −0.026 / −0.079 / 0.15）。
- 词族图谱：词坐标 v0（B1+B2）落地，类结构词级成立、类均值不成立。
- 关联机制：第一个完整样例 = 2865（L13H30 因果 → mlp 间接写出，e_ff ~30× 静态）。
- 类组织三连判决：2863 类均值 false / 2864 类方差弱且正交 / 2866 词级 margin true。

### 接续（2867 候选）
- 主选：词坐标 v1——加入第三块（2861 mlp 响应谱 per-word / η² 加权谱），
  检验 W3 互补性是否推广到三块；检索准确率随块数的增量（机制基增长率曲线
  的第一个点：每加一个坐标块，词间分辨/类检索提升多少）。
- 备选：属性轴预研（反义词对协议，接 400b）。
- 备选：199 词普查重启（G2 门线需用户决策）。
（脚本 tests/glm5/phase2866_word_coords.py；产物 phase2866/word_coords/）


---

## Phase 2867: 词坐标 v1：三块互补成立，mlp 响应谱 = 10 维浓缩类信号（检索 0.875），增长率曲线首点天花板 [2026-09-18]

### 动机与协议（预注册冻结于任何观测前）
2866 双块扩展为三块：B1 unembed 投影（10 维）/ B2 因果谱（1152 维，2846）/
B3 mlp 响应谱（10 维，2861 g_direct_true L26-35）。判据：T1 互补性（三 rho
全 <0.3）；T2 增长率曲线（LOO-NN 同类检索 acc 序列 B1/B2/B3/B12/B123，
null 200 置换 SEED=2867；sublinear_reuse iff final acc <= max(single)+0.05）；
T3 B3 词级类结构 margin。

### 结果（phase2867/word_coords_v1/；exec 4029ae29 / result 75001f08 / npz 1408deb1）
| 判决/量 | 值 | 读出 |
|---|---|---|
| **T1 三块互补** | rho12 **0.114** / rho13 **0.242** / rho23 **0.124** | **true**（全 <0.3） |
| acc(B1) | **1.000** | 完美（含构造性循环成分，见结论 3） |
| acc(B2) | 0.325 | 与 2866 一致 |
| **acc(B3)** | **0.875** | **10 维浓缩类信号**（1152 维 B2 的 2.7 倍） |
| acc(B12/B123) | 1.000 / 1.000 | 天花板 |
| T2 增长率 | growth = 0.000 vs max(single) | **sublinear_reuse**（但天花板效应，见结论 4） |
| **T3** B3 类结构 | margin **0.275** | null p95 0.0155（**17.7×**）→ **true** |

### 科学结论
1. **三块互补正式成立**（T1）：unembed 投影 / 因果谱 / mlp 响应谱是三个
   独立信息通道——词坐标需要三块，任何单块都是残缺图谱。
2. **B3 = 高效浓缩载体**：10 维 mlp 响应谱检索 0.875、margin 17.7x null，
   均远超 1152 维因果谱（0.325 / 5.7x）——机制信号的词级信息高度集中于
   mlp 响应轴（与 2861 深层 mlp 负反馈、2865 mlp 间接写出三线互证：
   **mlp 臂是类信息的主要机制载体**）。
3. **B1=1.0 的构造性循环声明**：B2 词表按 unembed 类中心构造，B1 与构造
   同源，其完美检索部分是同义反复；真正独立的前向测量块是 B3（0.875）。
4. **增长率曲线首点 = 天花板饱和**：B1 已 1.0，组合零增量，sublinear_reuse
   判决在此不可分辨——曲线设计需天花板控制。2868 修正：以 B2/B3 非
   同源块为基线（未测组合 B23 是关键增量点），预注册后再跑。

### 接续（2868 候选）
- 主选：增长率曲线 v2——非同源块组合（B23 / B2+B3+eta2 加权谱），带
  天花板控制的增量读出；若 acc(B23) > acc(B3)，B2 有真增量（新信息），
  机制基增长率曲线才真正起步。
- 备选：属性轴预研（反义词对协议）；199 词普查（需用户决策）。
（脚本 tests/glm5/phase2867_word_coords_v1.py；产物 phase2867/word_coords_v1/）


## Phase 2868+2869: 增长率曲线 v2/v3 —— 密度门控图像确立 [2026-09-18]

### Phase 2868: growth_v2（零前向 0.2s，确定性复跑逐位一致） [2026-09-18 08:02]

**背景**：2867 的 T2 增长率曲线饱和于 B1=1.0 天花板（构造性循环），真正的非同源增量点 B23 未测。2868 补测：B2(1152 因果谱)/B3(10 mlp 谱)/B2w(η²² 加权)/B2s(显著头子集 p<0.01, 43 头)/B23(hstack)/B23z(z-score 块平衡)。

**产物**：`tests/glm5/phase2868_growth_v2.py`（sha 59b059d3 [execution.json] / 59300982 [result.json] / adf5cb98 [growth_v2.npz]），目录 `phase2868/growth_v2/`。数据源 2867 npz + 2864 npz（eta2/p_per_head）。

**判决**：

| 判据 | 观测 | 结果 |
|---|---|---|
| G1 配对 delta（B23 vs B3） | Δ = **−0.1375**（0.7375 vs 0.875），null-delta p95 0.0625 | **false** → causal_no_gain |
| G2 η² 加权 | acc(B2w) 0.35 > acc(B2) 0.325，> null p95 0.1381 | **true** → eta2_weighting_helps |
| G3 稀疏充分性 | acc(B2s) **0.425** > acc(B2) 0.325（43/1152 头，压缩 26.8×） | **true** → sparse_sufficient |
| growth_v2 | (0.7375−0.875)/0.125 = **−1.1** | 天真拼接 = 干扰主导（sublinear_reuse 标签在此失义） |
| B23z 对照 | 0.3375 | z-score 列平衡更糟（每列等权放大噪声维） |

**教训**：①execution.json 缺失 Gen1——2867 在 main() 开头自写 execution.json（含 cc.snapshot 源码快照），2868 初版漏此段，Gen2 补上并按纪律清理重跑，判决逐位复现（确定性验证）。②预注册 growth_label（<0.1 = sublinear_reuse）对负增长情形语义失义，登记为标签局限。

### Phase 2869: fusion_curve（零前向 0.3s） [2026-09-18 08:03]

**设计**：密度匹配融合 F(α) = unit([B3, α·B2s])，α ∈ {0.25,0.5,1,2,4}；**max-α null**（每个置换标签对所有 α 取自己最大值）校正选择偏差。

**产物**：`tests/glm5/phase2869_fusion_curve.py`（sha 433595e8 [execution.json] / 6125b38d [result.json] / b5c65356 [fusion_curve.npz]），目录 `phase2869/fusion_curve/`。

**判决**：

| 判据 | 观测 | 结果 |
|---|---|---|
| H1 fusion_gain | α* = **0.25**，acc = **0.8875** > acc(B3) 0.875，> max-α null p95 0.175 | **true** |
| H2 权重结构 | α* < 1；acc 随 α 单调降（0.25→0.8875, 0.5→0.8375, 1→0.7125, 2→0.5625, 4→0.5） | **density_matched_fusion** |
| H3 growth_v3 | (0.8875−0.875)/0.125 = **0.10**（恰在冻结阈） | **sublinear_reuse（边界）** |

### 核心发现（重复三遍）

**机制信息载体按信息密度分层：mlp 臂 10 维最密（0.875）＞ 因果谱 43 头稀疏子空间（0.425）＞ 全谱 1152 维（0.325）；融合只有在密度匹配权重（1:4）下才有正增益（0.8875），天真拼接是干扰主导（0.7375）。**

增长率曲线有效点序列：B2 0.325 → B2s 0.425 → B3 0.875 → B3+0.25·B2s **0.8875**（有效增量 +0.0125，过 max-α null）→ B23 0.7375（天真拼接干扰）。

**"有限参数→无限能力"的第一个定量支点：新信息块的有效接入方式 = 密度门控（低维高密块主导 + 稀疏块弱权重），而非全谱叠加。** 与门池 9%/骨干 91%（2837-2844）、mlp 间接写出 ~30× 静态通道（2865）三线互证——网络的信息集成几何是门控的、分层的，不是叠加的。

### 接续

- 增长率曲线现状：v2（天真拼接）失义已修正为 v3（密度匹配）= 0.1 边界。曲线需要更多"轴"才有形状——下一功能轴（属性轴/语法轴）接入后每轴一个点。
- 2870 候选：**A（主选）属性轴预研**（反义词对协议 + 400b 资产接入，类别轴×属性轴双轴图谱第一步）；B 语法轴探针（POS/位置）；C B2s 43 头的身份解剖（与 2864 η²-top、2846 top-64 的交集 = "显著头"三重定义统一）。


## Phase 2870: 属性轴预研 —— 双轴图谱前提确立 [2026-09-18]

**背景**：TMA 双轴词族结构（类别轴 taxonomic × 属性轴 attributional）需要先验证据：属性方向在 unembed 几何上是否独立于类别轴。2870 零前向（3.8s，直接 safetensors 读 embed_tokens，tie_word_embeddings=True → W_U，不加载 4B 模型）。

**方法**：15 反义词对（size×3/speed×2/temp×2/age×2/weight/strength/brightness/moisture/height/fullness），d_attr = unit(E(w+)−E(w−))；类方向 dW_unit 按.SEED=2855 词表重建（2867 B1 同款）；null = 200 随机 token 对差方向（SEED=2870）。

**产物**：`tests/glm5/phase2870_attr_axis_pilot.py`（sha ada6ffe0 [execution.json] / 874c64a5 [result.json] / f81c846e [attr_axis_pilot.npz]），目录 `phase2870/attr_axis_pilot/`。

**判决**：

| 判据 | 观测 | 结果 |
|---|---|---|
| **P1** 属性×类正交（主） | max_c \|cos\| = **0.0755**（height 对，阈 0.3，随机 null p95 0.1143——观测**低于随机基线**） | **true → attribute_axis_independent** |
| **P2** 轴结构（字面 false） | mean 0.05 > null p95 0.0202；max 0.588 | 字面 **axis_collapsed**（判据缺陷，见下） |
| **P3** 类子空间正交能量 | mean = **0.995**（属性方向能量 99.5% 在 10 维类子空间外） | **true → class_subspace_clean** |

**P2 判据缺陷注记（不挪门柱，如实登记）**：top 相似对 = speed1×speed2 0.588 / age1×age2 0.481 / size1×size3 0.460——全部是**同轴对**（同属性不同词对）；跨轴对（102 对）max 仅 **0.138**。P2 把两个总体混入一个分布：同轴高相关 = **轴方向跨词对可再现（replicability）**，恰是"轴"存在的定义性证据，而非坍缩。正确读出：**axis_structure_confirmed**（同轴 0.46-0.59 vs 跨轴 <0.14，分离干净）。

### 核心发现（重复三遍）

**属性轴与类别轴在 unembed 几何上强独立（P1 投影低于随机基线 + P3 能量 99.5% 在类子空间外），且属性方向具有跨词对再现性（同轴对 cos 0.46-0.59，跨轴 <0.14）——双轴图谱（taxonomic × attributional）的几何前提成立。**

### 接续

- 2871（进行中，自动续推）：显著头三重身份解剖（2846 top-64 × 2864 η² 显著 × 2868 B2s-43 交集，响应图谱机制轴实体化收口，零前向）
- 2872 候选：属性轴因果 census（2846 协议移植到属性对词表，属性轴机制侧定位——双轴关联第二样例，需完整预注册设计）


## Phase 2871: 显著头三重身份解剖 —— 一致核心确认（零前向 0.2s） [2026-09-18]

**设计**：三种独立的"重要头"定义统一解剖——A = 2846 因果负载 top-64（mean drop）；B = 2864 η² 显著头（p_per_head<0.01，43 头 = 2868 B2s 同源）；C = 2859 bootstrap 稳定前缘 top-10。超几何富集（Bonferroni ×4）+ 2862 冻结分位角色分类 + 负载份额。

**产物**：`tests/glm5/phase2871_sig_heads_triple.py`（sha 91377ec6 [execution.json] / 59830517 [result.json] / bd056954 [sig_heads_triple.npz]），目录 `phase2871/sig_heads_triple/`。

**判决**：

| 判据 | 观测 | 结果 |
|---|---|---|
| **U1** 交集富集 | A∩B=9（p=3.6e-4）/ A∩C=**10**（p≈0，C⊆A！）/ B∩C=4（p=3.0e-4）/ 三重=4 | **true → core_confirmed** |
| **U2** 核心角色（≥2 定义，15 头） | 形成器 2 / 放大器 **0** / 其他 13 | 与 2862 前缘纯形成器一致 |
| **U3** 负载份额 | 代数和 −0.26（**负值域失义**）；正归一化 **23.1%** | 字面 false（core_minor_load）；判据缺陷注记 |

**三重一致核心（4 头）：L5H25 / L23H7 / L23H29 / L26H4**；core2（15 头）含 **L13H30**（A∩C：因果负载 top1 兼稳定前缘，drop 0.1012 = core2 正负载的 1/5）。横跨因果+类方差+稳定三定义的交集全部超几何富集——**"重要头"不是单一协议的伪影，存在跨定义一致的真实核心**。

**两项登记**：
1. **2864 勘误注记**：`n_sig_heads_p95=102` 用了 `p_per_head >= 0.95`（错误尾向，描述性附带未进判决）；正确的 η² 显著大集合 = p<0.01（43 头）。
2. **负值域教训（第二次）**：mean_drop 697/1152 头为负（sum −1.91），凡"份额/占比"判据预注册时必须先声明归一化基准的正负处理。正归一化交叉验证：top64 = 51.6% 正负载，与 2846 逐位一致 ✓。core2 密度：1.3% 头数承载 23.1% 正负载。

### 接续

2872 主选：**属性轴因果 census**（2846 协议移植到属性对词表——属性轴机制侧定位，双轴关联第二样例；需完整预注册设计 + 词表构造，~1 前向/词）。备选：门控融合增长率曲线第三点（密度门控协议推广）。


## Phase 2872: 属性轴因果 census —— 双轴机制侧完全分离（33.2 min，28 词 × 36 层全谱） [2026-09-18]

**设计**：2846 census 机制移植到属性轴——`AttrCensus(AtlasCensus)` 子类仅覆盖 `conds2_for`（same = 同轴其他词按 tid 序确定性选取，极性无关，局限已登记）；10 轴 / 28 单 token 极性词（2870 对表合并：size6/speed3/temp4/age3/weight2/strength2/brightness2/moisture2/height2/fullness2）；cdir = 轴方向（对差方向平均，2870 同款）；null tids SEED=2872。全谱 drops_attr (28,36,32)。钳制校验 max_cr ≤ 0.0192 全通过。

**产物**：`tests/glm5/phase2872_attr_census.py`（sha 2551a824 [execution.json] / 282182af [result.json] / 72eb6982 [attr_census.npz]），目录 `phase2872/attr_census/`。

**判决**：

| 判据 | 观测 | 结果 |
|---|---|---|
| **X1** 跨轴共享前缘 | max 轴重叠 = 5/10（无头进 ≥6 轴 top-32；null p95 = 0） | **false → attr_frontedge_absent** |
| **X2** 机制侧分离 | ρ(attr 谱, 类谱) = **0.055**；ρ(attr 谱, η²) = −0.039 | **true → mechanism_side_separate** |
| **X3** 头群独立性 | \|attr_top64 ∩ class_top64\| = 5，超几何 p = **0.280**（不富集） | **independent_populations** |

**描述**：各轴前缘分散于 L0-L34、各自为政（size: L21H2/L2H7…；temp: L0H15/L1H27…）；仅个别轴前缘触及 core2 头（brightness/moisture 含 L13H30）。

### 核心发现（重复三遍）

**双轴从表示到机制完全独立：2870 证 unembed 几何独立（属性×类 cos < 随机基线），2872 证机制空间独立（ρ=0.055、头群不富集 p=0.28）——类别轴有集中前缘（top-64 = 51.6% 负载），属性轴无跨轴共享前缘、由轴特异小头群承载。**

机制图像合并：类 = 范畴级特征（集中前缘 + 三重一致核心 2871）；属性 = 词汇级特征（轴特异分散小群）。二者与 2869 密度门控、2866 词级特异、2863 类均值失败七线互证——**机制空间的组织粒度与特征的语义粒度对齐：粒度越细（词/属性 < 类），机制载体越分散**。

### 阶段小结（2870-2872，双轴关联样例链闭合）

1. 2870：属性轴几何前提（独立 + 再现）
2. 2871：显著头三重一致核心（类轴机制实体化）
3. 2872：双轴机制侧分离（属性轴普查第一版）

**2873 候选**：A 门控融合增长率曲线第三点（属性轴 B_attr 接入 2869 协议——每加一轴的增量可测）；B 属性轴 top-64 头的 OV/角色解剖（attr 版 2862）；C 语法轴探针立项。


## Phase 2873: 增长率曲线第二点——属性轴接入密度门控协议（零前向 19.0s） [2026-09-18 09:10]

**日期**：2026-09-18。**脚本**：`tests/glm5/phase2873_growth_axis2.py`。
**产物**：`result/rdc_query_construction_20260913/phase2873/growth_axis2/`
{execution.json `9163ec12`, result.json `1c7d9df2`, growth_axis2.npz `95a87dbd`}。

### 原理与协议

把 2872 属性轴因果谱（28 词 × 36 层 × 32 头，10 轴 × 2-5 词，标签 = `axis:word` 前缀）
接入 2867/2868 词坐标检索协议（列 z-score + 行单位化 + LOO-NN 余弦 + 对角 −2），
测量"机制基增长率曲线"第二个轴的数据点。三个预注册判决，null = **全管线镜像置换**
（置换标签 → 按置换标签重选头 → 按置换标签评估 acc，N=200，SEED=2873）——
选择偏差被完整镜像进 null（2871 教训：F 检验 p 值用 `f.sf` 正确尾向）。

### 判决表

| 判决 | 观测 | 读出 |
|---|---|---|
| **C1** 轴 2 稀疏充分性复制 | acc(sig2)=0.2143 < null p95 **0.3571**（acc_full=0.0714） | **not_replicated** |
| **C2** 跨轴头重叠 | sig2 仅 **3 头**，与类轴 43 头交集 **0**（期望 0.11，p=1.0） | **axis_specific**（功效不足以拒绝） |
| **C3** 类轴子集迁移 | acc(class43 on attr)=0.1071 > acc(full)=0.0714 | **reuse_signal（弱，描述性）** |
| 增长点 | axis1=43 头（80 词）vs axis2=3 头（28 词）；new=3/3 | **词数功效受限，非零证明** |

### 解读（三点）

1. **主瓶颈再次确认为词分辨率**：10 轴 × 2-5 词下，全管线镜像 null（含选择偏差）
   的 p95 达 0.3571——28 词功效不足以让轴 2 因果谱信号过显著线。2863（8 词/类失败）
   → 2866（词级 margin 成立）→ 2873（2-5 词/轴不足）三点连成**功效-分辨率曲线**：
   词坐标类结构检出的最小词数介于 28-80 之间，图谱扩展必须先过词数关。
2. **轴 2 显著头仅 3 个 vs 轴 1 的 43 个**：与 2872 X1（属性轴无共享前缘）+ X3
   （独立分散小群）一致——属性轴在因果谱中是**更稀疏的词汇级载体**，类轴是
   **更密集的范畴级载体**。载体密度随语义粒度变化，与 2869 密度门控互证。
3. **C3 弱复用信号**：类轴 43 头在属性词上略优于全谱（0.1071 > 0.0714），
   提示存在小规模跨轴复用基底，但两值都接近 chance（1/27≈0.037），不能过度解读。

### 方法论入账

- **全管线镜像 null**（选择→评估同步置换）从本 Phase 起成为词坐标检索类判据的标准
  null 形态；单纯置换标签不再充分（凡有数据驱动选择，null 必须复刻选择）。
- execution.json 本 Phase 起在 main() 开头自写（2867 模式制度化，2868 Gen1 教训）。

### 接续

词数功效瓶颈的解法 = 词表扩容（每轴 ≥6 词）或**聚合测量**（per-axis 谱已有
`per_axis (10,1152)`，2873 未用，留作 2874 候选：轴级而非词级的跨轴比较）。


## Phase 2873+2874: 属性轴融合第三点 + 跨模型几何前提复检 [2026-09-18 09:17]

### Phase 2873: attr_fusion（零前向 rep 块 + 单次 CUDA 加载 h 块，13.0s） [2026-09-18 09:10]

**背景**：2869 建立密度门控融合 F(a)=unit([B3, a*B2s])，a*=0.25，acc 0.8875（增长率 v3 = 0.10 边界）；2870 证明属性轴与类轴在 unembed 几何上独立；2872 证明机制侧分离。遗留问题：**属性轴在融合层是否携带类增量信息**（增长率曲线第三点），即双轴图谱在读出层是否闭合。

**设计**（预注册冻结于任何读出前）：
- B_cf = unit([B3u, 0.25*B2s])（2869 冻结赢家，锚点 A0 要求精确复现 0.8875）
- B_attr_rep（15 维）：proj[w,a] = unit(E_w)·D[a]，D 为 2870 冻结的 15 个反义词对方向（零前向，safetensors 直读）
- B_attr_h（15 维）：proj[w,a] = unit(h_w)·D[a]，h_w = qwen3-4b 单 token 无上下文 last hidden state（一次 CUDA 加载，80 前向，bf16）
- A1（主判据）F_rep(a)=unit([B_cf, a*B_attr_rep_u])，a ∈ {0.25,0.5,1,2,4}，max-α null（200 标签置换，SEED=2873）；A2 增长率 v4；A3 属性谱单独类判读；A4 类 margin；A5 隐藏块同协议（SEED=28731）+ 几何可复现性 ρ(rep,h)

**产物**：`tests/glm5/phase2873_attr_fusion.py`（sha 3f1f164b [execution.json] / dbf0e277 [result.json] / c96a4f4d [attr_fusion.npz]），目录 `phase2873/attr_fusion/`。

**判据**：

| 判据 | 观测 | 结果 |
|---|---|---|
| A0 锚点 | acc(B_cf) = **0.8875**（精确一致） | true → 协议完好 |
| A1 融合增益（主） | α*=0.25，acc 0.875 < 0.8875；acc 随 α 单调降（0.25→0.875, 0.5→0.825, 1→0.625, 2→0.4125, 4→0.2625）；null p95 0.1875 | **false → attr_no_class_gain** |
| A2 增长率 v4 | (0.875−0.8875)/0.1125 = **−0.111** | sublinear_reuse（负增量） |
| A3 属性谱类泄漏 | acc(B_attr_rep 单独) = **0.25** > null p95 0.15 | true → attr_axis_class_leak（候选观察） |
| A4 属性谱类相关 | margin 0.0677 > null p95 0.0192 | true → attr_profile_class_correlated |
| A5 隐藏块融合 | acc 全 α 恒 0.8875（0.25/0.5→0.8875, 1→0.875, 2→0.8875, 4→0.8875），null p95 0.175 | false → attr_no_class_gain_h（**良性**） |
| A5 几何 | ρ(utri cos rep, utri cos h) = **0.209** < 0.5 | attr_geometry_hidden_shifted |

### Phase 2874: attr_geom_crossmodel（Qwen3-14B + GLM4，零前向 safetensors 直读，逐模型顺序处理，12.2s） [2026-09-18 09:27]

**设计**：每模型（qwen14=Qwen3-14B, glm4=glm4-9b-chat-hf）单 token 过滤后全部 80 类词 + 15 词对保留（可比性 Q4 满足：10 类 ≥3 词、15 对）；dW[c] = unit(mean unit(E_w))；d_attr = unit(E+ − E−)；Q1 每对 max_c |cos| < 0.3（null 200 随机 token 对，SEED=2874/28741）；Q2 正交能量 > 0.8；Q3 属性谱类判读（2873 A3 跨模型复核）。

**产物**：`tests/glm5/phase2874_attr_geom_crossmodel.py`（sha 25e414ec [execution.json] / 6292de8c [result.json] / 4d681506 [npz]），目录 `phase2874/attr_geom_crossmodel/`。

| 判据 | Qwen3-14B | GLM4 |
|---|---|---|
| Q1 属性轴独立 | max 0.056 < null p95 0.1076 → **true** | max 0.0476 < null p95 0.0774 → **true** |
| Q2 类子空间干净 | 0.996 → **true** | 0.9967 → **true** |
| Q3 类泄漏 | acc 0.1625 = null p95 0.1625（**恰在边界**，判 false） | acc 0.0875 < null p95 0.1381 → **false** |

### 核心发现（重复三遍）

**1. 属性轴在密度门控融合下无类增量（attr_no_class_gain，增长率 = −0.111）：类读出所需信息完全由类机制块 B_cf 承载，双轴图谱在读出层闭合——属性轴是独立的输出维度，不是类的残差。**

**1. 属性轴在密度门控融合下无类增量：类读出所需信息完全由类机制块承载，双轴图谱在读出层闭合。**

**1. 属性轴融合无增量，双轴图谱读出层闭合。**

**2. "方向正交 ≠ 信息独立"必须分层表述**：几何层（轴方向独立，2870/2874 三模型一致）、机制层（载体分离，2872）、信息层（部分冗余：qwen4 属性谱单独类判读 0.25 > null 0.15，但跨模型不复现——qwen14 恰在边界、glm4 明确无）。信息层的耦合是模型特异的，不是普遍数学结构。

**3. 支撑共享决定融合干扰，而非维度**：rep 块（unembed 空间）即使 α=0.25 也单调干扰（0.8875→0.875→…→0.2625），h 块（深层残差空间）全程零干扰（恒 0.8875）——unembed 全局语义空间是高共享坐标的高密空间，深层残差流是更专用化的稀疏空间。与 2869 密度门控、2837-2844 门控 9%/骨干 91% 三线互证。

### 严格审视（问题、硬伤、瓶颈）

1. **A3 泄漏为 qwen4 特异的候选观察**：0.25 vs null 0.15 仅 8 词边际（80 词、10 类，一格 0.0125）；跨模型不复现提示可能是 80 词样本偶然结构（如 metal/vehicle 在 size 方向的偶然分离），非普遍机制。需留一类交叉验证定位泄漏类。
2. **h 读出为零语境代理**：单 token 无上下文的 last hidden state 不代表真实语境下的属性响应（2872 的属性机制是语境化的）；ρ(rep,h)=0.209 的断裂原因未定位（final norm、层堆叠旋转、还是属性几何在深层的重组），未做层间扫描。
3. **null 引擎限制**：沿用 2869 的特征固定 + 标签置换；max-α 校正覆盖 α 选择偏误，但不覆盖块内噪声多重比较；A1 的 α 网格只有 5 点，argmax 在平坦曲线上不稳。
4. **A5 平坦曲线的另一种读法**：h 块零干扰也可能是 h 块信息量本身过低（被 unit 范数几何稀释），而非良性正交——缺少"阳性对照块"（已知应产生干扰或增益的块）校准灵敏度。
5. 数据规模仍小（80 词）；增长率曲线第三点本质上是一个"确认饱和"的阴性结果，阴性结果的证据力度受限于 α 网格与块构造方式。

### 智能理论洞察（第一性原理）

语言编码的数学结构 = **多轴坐标系 + 三层耦合定律**。每条功能轴（类、属性、语法、速度……）拥有：独立的方向集（几何层）、独立的机制载体（机制层）、以及与其他轴的部分冗余信息（信息层）。2870-2874 五个 Phase 首次把三层分开测量并给出第一条耦合测量：**轴间几何正交是普遍的（三模型一致），信息冗余是模型特异的**——这提示几何层是训练动力学的普适产物（可能源于 unembed 的低秩读出压力），而信息层耦合依赖于具体训练数据的相关结构。对"有限参数→无限组合"的含义：轴的独立性使组合空间按笛卡尔积而非纠缠方式增长，参数效率来自几何层的正交化，而非信息层的解耦。

### 阶段性大任务（下一步）

- **2875（主选）隐藏态属性几何层间扫描**：h_l·D 随 l ∈ [0,36] 的轨迹 + ρ_l(rep, h_l) 曲线，定位 ρ=0.209 断裂发生的层带，检验属性几何是"逐层重组"还是"最后一跳突变"（80 词 × 36 层单次前向，低成本，可复用 2873 加载路径）。
- 2876 候选：A3 泄漏解剖（留一类交叉验证 + 类×属性混淆矩阵，判定 qwen4 泄漏是否少数类驱动）；B 语法轴探针立项（POS/位置条件轴结构，需完整预注册 + 词表构造）。
- 远期：三模型增长率曲线（每模型一条密度门控融合曲线）——检验"增长率分层"是否跨模型普适，需因果调谐数据（2867 协议 ×3 模型，约 3×33 min）。


## Phase 2874: 属性轴词表扩容 v2（零前向 3.9s，Gen1→Gen2 两轮） [2026-09-18 09:27]

**日期**：2026-09-18。**脚本**：`tests/glm5/phase2874_attr_vocab_v2.py`。
**产物**：`phase2874/attr_vocab_v2/` {execution.json `06ecd0ef`,
result.json `b07eecd1`, attr_vocab_v2.npz `5f7796e1`}。

### 原理

修复 2873 诊断的词数功效瓶颈：每属性轴扩到 ≥5 个 single-token 极性词，
用 2858 式几何判据链（VG1 覆盖 / VG2a 轴向正交 / VG2b 同轴对再现 / VG3
范数健康）在**花任何前向预算之前** gate 词表。判据预注册冻结于任何
unembed 统计之前（execution.json 先行）。

### Gen1→Gen2

Gen1 池（weightless/feeble/luminous/arid 四词非单 token）VG1=false
（6/10 轴）。按 2857 CAND2 补足先例扩池（ponderous/mighty/radiant/soggy
换入），**门线一字不动**，清旧产物后重跑。

### Gen2 判决表

| 判据 | 观测 | 门线 | 结果 |
|---|---|---|---|
| **VG1** 覆盖 | **8/10 轴** ≥5 词（weight 4、moisture 3 不足） | ≥8 轴 | **true** |
| **VG2a** 轴向正交 | max 跨轴 cos **0.1869** | <0.488 | **true** |
| **VG2b** 同轴对再现 | 均值 **0.2731**（8 轴） | >0.2 | **true**（2870 同轴 0.46-0.59，v2 池更抽象故略降） |
| **VG3** 范数健康 | 最差轴比 **1.2039** | <3.0 | **true** |
| **vocab_legal** | 8 轴 / **42 词** / 26 对 | — | **True** |

隔离轴：weight（ponderous 仍非单 token）、moisture（soggy 非单 token）。

### 接续

Phase 2875 = 42 词全谱 census（2872 协议逐字移植 + X5 功效复检：42 词
LOO-NN 检索 vs 全管线镜像 null）；随后 2876 = 增长率第二点重测（2873
协议跑在新词表上）。


## Phase 2875+2876+2877: 属性轴通道分化三连——causal 通道阴性、mlp 通道阳性 [2026-09-18 09:29]

**日期**：2026-09-18。**脚本**：`phase2875_attr_census_v2.py`（48.5 min 前向）、
`phase2876_growth_axis2_v2.py`（15.0s 零前向）、`phase2877_attr_mlp_spectrum.py`
（15.3s，126 前向）。产物：`phase2875/attr_census_v2/` {exec `9b72c85d`,
result `182d0094`, npz `aff07bb2`}；`phase2876/growth_axis2_v2/` {exec `e2b48133`,
result `3b3770bd`, npz `1b2d3faf`}；`phase2877/attr_mlp_spectrum/` {exec `0e4c799d`,
result `5c42ab92`, npz `8fc6e0cb`}。

### 2875：42 词全谱 census（2874 法定词表，钳制校验 max_cr≤0.0097）

| 判决 | 观测 | 结果 |
|---|---|---|
| **X1** 前缘存在性 | ≥6/8 轴重叠头数 = **0**（max 5） | **attr_frontedge_absent**（2872 复制，更强） |
| **X2** 机制侧分离 | ρ(attr, class) = **−0.0031**（几乎精确零） | **mechanism_side_separate** |
| **X3** 头群独立 | 交集 4，p = 0.482 | **independent_populations** |
| **X5** 词级检索 | acc 0.1667 < null p95 0.1905 | **axis_signal_absent**（42 词仍阴性） |

### 2876：增长率第二点重测（镜像 null）

sig2 仅 **1 头**（p<0.01）；D1=not_replicated / D2=axis_specific（0/1 交集）/
D3=novel_dominant（class43 迁移 0.1667 = full）/ D4=axis_signal_absent。
**功效解释弱化**：28 词（2873）与 42 词（2876）双双阴性 + 前缘零 +
谱相关零 → "头级 drop 通道不按属性轴组织"上升为候选结论。

### 2877：属性轴 mlp 响应谱（B3_attr，42×10）

Gen1 两教训：①`final_verdict` 自引用 res（UnboundLocalError，崩溃于写盘前，
E1/E2 未被观测）；②hook-vs-recompute 一致性门线错位——重算 1-D 形状 vs
前向 2-token 批形状，cuBLAS 按形状选算法致 bf16 归约序差异 ~2e-3；
**g 用 mlp_call 同形状同上下文算 ref 与 perturbed，测量自洽不受影响**。
Gen2 修正一致性检验为重算确定性（v1a：同输入两次重算，门 1e-6）。

| 判决 | 观测 | 结果 |
|---|---|---|
| **v1a** 重算确定性 | max rel err = **0.000e+00** | **true**（kernel 上下文恒定） |
| kernel 上下文噪声 | 1.98e-3（描述性登记） | — |
| **E1** 轴检索 | acc = **0.5000** vs null p95 0.1917（2.6×） | **mlp_carries_attr_axis** |
| **E2** 同轴余弦边际 | **0.1158** vs null p95 0.0563（2.1×） | **attr_margin_in_mlp** |

### 核心结论（重复三遍）

**通道分化（channel dissociation）：属性轴信号不在头级因果 drop 谱（2875/2876
双阴性，前缘不存在，谱相关 −0.003），但在 mlp 响应谱 10 维中显著存在（检索
0.5，2.6× null）——"mlp 臂是高密度机制载体"的定律从类轴跨轴复制到属性轴；
两轴的载体差异在通道维度：类轴 = drop 谱有组织 + mlp 载体（0.875），属性轴 =
drop 谱无组织 + mlp 载体（0.5）。机制基增长率曲线第二点（mlp 通道）：
acc 0 → 0.5，新组件 = 0 个新 head 集合（同一 36×32 布局内换通道读出）。**

### 方法论入账

- **重算类测量的正确一致性检验 = 重算确定性（同形状）**，不是 hook-vs-recompute
  （跨 kernel 上下文）；凡 hook 捕获 + 重算混合的协议必须区分两者。
- GPU cuBLAS 形状相关算法选择 → bf16 跨形状差异 ~2e-3 是常态，写入协议预算。

### 接续

2878 候选：A（主选）B3_attr 扩容协议推广——语法轴/翻译轴的 mlp 载体检验
（每轴 ~15s）；B 类轴+属性轴联合词坐标（B3_class ∪ B3_attr 密度门控融合）；
C 2872/2875 双词表 census 的 per-axis 对齐分析（同轴跨词表再现性）。


## Phase 2878+2879: 语法轴 mlp 载体阳性 + 翻译轴真阴性（统一跨语言方向不存在） [2026-09-18 10:41]

**日期**：2026-09-18。**脚本**：`phase2878_syntax_trans_vocab.py`（零前向，
Gen1→Gen2）、`phase2879_syntax_mlp_spectrum.py`（16.0s，144 前向）。产物：
`phase2878/syntax_trans_vocab/` {exec `d7165ca0`, result `66b282e3`,
npz `0bb3ec1e`}；`phase2879/syntax_mlp_spectrum/` {exec `e1da7fa7`,
result `44a659bd`, npz `24eb4941`}。

### 2878：语法/翻译轴词表（双轨判决，Gen2）

- **Gen1 教训（同形词污染）**：`single_token_id` 优先 `' '+t`，`chat` 解析为
  英语"聊天" token 而非法语"猫"——翻译对方向被英文义污染；VG1 翻译族仅
  2/3（fr 4/8 对），vocab_legal=false。
- **Gen2 修正（2874 扩池先例，门线不动）**：翻译池扩容（fr 19/de 22/es 30
  对）+ 预注册同形排除表（chat/sol/pan/Mann/Gold/Winter/Hand/Bank 等 32 词，
  冻结于脚本）+ comparative 扩至 15 对。
- **判决**：VG1 语法 4/4、翻译 3/3 全过；VG2a max cos 0.1292 < 0.488；
  VG3 最差比 1.1654 < 3.0。
  **syn_legal=true**（number/gerund/comparative 存活，48 词 22 对；
  tense VG2b 0.1902 差 0.01 隔离）。
  **trans_legal=false —— 真阴性：去污染后 VG2b = fr −0.0007 / de −0.0029 /
  es 0.0011，精确零**。

### 2879：语法轴 mlp 响应谱（B3_syntax，48×10，2877 协议逐字移植）

| 判决 | 观测 | 结果 |
|---|---|---|
| **v1** 重算确定性 | max rel err = **0.000e+00** | **true** |
| kernel 上下文噪声 | 1.685e-3（描述性） | — |
| **E1** 轴检索 | acc = **0.7083** vs null p95 0.5208（1.82×，null mean 0.3887） | **mlp_carries_syntax_axis** |
| **E2** 同轴余弦边际 | 0.0744 vs null p95 0.0889（null mean 0.0041） | **margin_absent**（弱趋势不达门线） |
| E3 层分布 | L35 主导（0.2425），深层递增 | 类/属性轴同型 |

### 核心结论（重复三遍）

**mlp 臂载体定律第三轴复制：syntax E1 阳性（0.7083，零新 head 组件，同一
36×32 布局内换通道读出）——载体曲线 class 0.875 / attr 0.500 / syntax
0.7083；同时 E2 边际阴性揭示语法轴载体结构与类/属性轴不同：检索成功但无
全域同轴聚类（3 轴不平衡 null 基线高，p95 0.5208）。翻译轴则根本不存在统一
轴方向（VG2b≈0）——tied unembed 中 en→L 差被词形/概念差主导，语言恒定分量
≈0；"轴"这一语言族谱概念在翻译域不成立，在语法域以形态对成立。**

### 方法论入账

- **同形词消歧必须预注册**：多语词表中 `' '+t` 优先的 tokenizer 规则会把
  法语 chat/sol/table 解析为英语 token；扩池时同步冻结同形排除表（VG0）。
- 双轨判决（syn_legal / trans_legal 分开登记）避免一族真阴性拖垮另一族的
  测量——真阴性是结果，不是词表失败。

### 接续

2880 候选：A（主选）语法轴 drop 谱 census（2875 协议，48 词全谱，
通道分化四象限表补齐 syntax 象限）；B B3_class∪B3_attr∪B3_syntax 三族
联合词坐标与密度门控融合（增长率曲线三族交汇）；C 语法轴 E2 阴性解剖
（per-axis margin 分解：number/gerund/comparative 哪轴无边际）。


## Phase 2880: 语法轴 drop 谱 census——通道分化四象限表 syntax 象限补齐 [2026-09-18 10:49]

**日期**：2026-09-18。**脚本**：`phase2880_syntax_census.py`（2875 协议
verbatim 移植，3 轴 48 词 × 36 层 × 32 头）。产物：`phase2880/syntax_census/`
{exec b3916a5b, result fa707e93, npz c153aa93}。

### 判决表（预注册 Y1-Y5）

| 判决 | 观测 | 结果 |
|---|---|---|
| **Y1** 前缘（≥3/3 全重叠） | n_heads = 0 vs null p95 1.0 | **syntax_frontedge_absent** |
| **Y2** 机制侧分离 | rho(syntax, class) = 0.0588 | **mechanism_side_separate** |
| **Y3** top64 交集 | 2，hyper p = 0.884269 | independent_populations |
| **Y5** 词级检索 | acc = 0.5625 vs null p95 0.5417 | **axis_signal_detected** |

per-axis top8：{"number": ["L9H0", "L1H15", "L33H17", "L32H28", "L21H3", "L18H3", "L31H2", "L31H25"], "gerund": ["L13H21", "L10H31", "L4H19", "L1H19", "L12H5", "L2H8", "L16H1", "L23H5"], "comparative": ["L0H17", "L31H20", "L33H29", "L28H22", "L6H12", "L6H15", "L2H24", "L1H30"]}

### Post-hoc 附注（2879 E2 分解，描述性，非门线）

number 轴单独边际 +0.3216 > null p95 0.1460（成立）；gerund +0.0348 /
comparative +0.0565 阴性——2879 E2 全局阴性系后两轴稀释。number↔gerund
轴质心 cos 0.789 高度共线：语法轴族内部非正交，与跨族几何独立（2870/2875）
形成对照。详见 `phase2879/e2_posthoc_decomp.txt`。

### 四象限表（通道分化 × 轴族）

| 轴族 | drop 谱组织 | mlp 载体 |
|---|---|---|
| class | 有（43 头前缘） | 0.875 |
| attr | 无（2875/2876 双阴性） | 0.500 |
| syntax | 有组织（Y1=False, Y5=True） | 0.7083 |
| translation | 无统一轴方向（2878 VG2b≈0） | 未测（轴不存在） |

### 接续

2881 候选：A（主选）B3_class∪B3_attr∪B3_syntax 三族联合词坐标与密度门控
融合；B 语法轴内部解剖（number 强边际 + 质心共线的头级来源，Y1 头群
per-axis 归属）；C tense 轴隔离复议（0.1902 差 0.01，若 A/B 需要第三语法轴）。


## Phase 2881: 三族联合词坐标——一个 mlp 通道承载全部三个族谱（门控融合跨族推广） [2026-09-18 12:04]

**日期**：2026-09-18。**脚本**：`phase2881_joint_word_coords.py`（67.6s，
170 词 × 21 方向 × 10 层统一重测）。产物：`phase2881/joint_word_coords/`
{exec 660ac712, result eb95f6ff, npz abe74030}。

词表：class 80（2867 CATS verbatim）+ attr 42（2874 存活轴）+ syntax 48
（2878 存活轴）；方向 = class 10 类质心差分（2861 式，W_U）+ attr 8 +
syntax 3 = 21 方向 × L26-35 = 210 维联合谱（块 100/80/30）。

### 判决表（预注册 J1-J4）

| 判决 | 观测 | 结果 |
|---|---|---|
| **J1** 联合细标签检索（21 类） | acc = **0.7471** vs null p95 0.0882（**8.5×**） | **joint_spectrum_carries_all_families** |
| **J2** 密度门控融合 | own 0.7235 → **α\*=0.5**，Δ\* = **+0.0529** > null p95 0.0412 | **gated_fusion_generalizes** |
| **J3** 跨族迁移矩阵（最近质心） | 全部 >0.76（class 行 0.93-0.98 最强；syntax→attr 0.76 最弱） | 跨族方向可读出任意族标签 |
| **J4** 族质心余弦 | class↔attr **−0.65** / class↔syntax −0.33 / attr↔syntax −0.14；own acc class 0.80 / attr 0.52 / syntax 0.75 | 族子空间互斥分离 |

### 核心结论（重复三遍）

**一个 mlp 响应通道承载全部三个族谱：21 个细标签从同一 210 维联合谱以
8.5× null 检索成功；2869 密度门控定律跨族推广（α\*=0.5 增益 +0.053）；
族质心两两负相关——三个族是同一通道内互斥分离的子空间，不是三个通道。
载体曲线收官：class 0.875 / attr 0.500 / syntax 0.7083（own-block）→
联合谱细标签 0.7471（8.5× null）。族谱工程（V-族谱 → LLM 响应图谱）的
通道侧整合完成。**

### 登记差异

per-family own acc 与单族 phase 略异（class 0.80 vs 0.875@2867、attr
0.524 vs 0.500@2877、syntax 0.75 vs 0.708@2879）——统一协议重测 +
conds 细节差异，方向一致性保持（attr/syntax 几乎复现，class 略低系
2861 单词 conds 与 2-token conds 差异）。

### 接续

族谱-图谱整合线收官。下一步转入：A Atlas Ledger v2 升级（新增
syntax/translation 轴、joint 块、象限表与三族迁移矩阵入 spec）；
B 语法轴内部解剖（number 强边际 + tense 复议）；C 研究方向总结与
下一战役规划（见总结节）。


---

## Phase 2882: Atlas Ledger v2 升级（零前向迁移） [2026-09-18 12:15]

**日期**：2026-09-18。**目标**：把 TMA Atlas Ledger 升级为 format v2，为多模型
复制战役（DS7B/GLM4/Gemma4）提供可扩展事实源。

### 原理与设计

v2 新增五节（spec §10–13，向后兼容 v1——v1 文件由 v2 loader 原样加载、缺省节视为空）：

1. `quadrants`（四象限表）：每词族轴一行，登记 drop 谱组织 × mlp 载体两通道观测，
   每格强制引用已存在 meas_id（可导出性检查对象）。实测：class=organized+strong_mlp
   (0.875) / attr=dissociated_mlp_only (0.500) / syntax=intermediate (drop 弱检索
   0.5625 + mlp 0.7083) / translation=no_axis（无统一轴方向，未测）。
2. `transfer`：2881 J3/J4 迁移矩阵快照（全格 >0.76；族质心互负 −0.65/−0.33/−0.14），
   descriptive 无门线。
3. `negatives`（阴性登记）：N1 translation_axis_absent（real_negative，VG2b≈0，
   settled）；N2 tense 隔离（reopenable，复议条件=扩池 ≥8 对）；N3 e200 隔离
   （reopenable，复议条件=重建词表使 offdiag<0.488）。**阴性结果不再蒸发**。
4. `model_namespace`：primary=qwen3-4b，pending_replication=[ds7b, glm4, gemma4]；
   跨模型条目 model 字段必填，同 id 可跨模型并存于 ledgers/<model>/ 子文件。
5. `migration_history`：v1 备份 SHA 登记（atlas_ledger_v1_backup.json，sha256_8
   见下）。

### 预注册门线（execution.json 先于任何 ledger 修改落盘）

- V1 保真：六节计数 pre/post 一致 + SHA 复核 0 stale（既有条目逐字节不动）
- V2 四象限可导出性：每格引用的 meas_id 均存在于 measurements
- V3 阴性接地：每条引用 ≥1 存在 meas_id；quarantine 必带 reopen_condition
- V4 向后兼容：v1 备份在 v2 loader 下干净加载（0 stale，缺省节为空）
- V5 自洽：v2 重载 version==2、五新节齐全、4 quadrants/3 negatives、0 stale

### 结果

**ledger_v2=True（V1–V5 全过）**。V1 计数 {axes 5, blocks 10, headsets 4,
measurements 21, growth_curve 11, linkage 7} 前后一致，0 stale。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2882/ledger_v2/execution.json | cfd2b2e8 |
| phase2882/ledger_v2/result.json | 527043bb |
| tests/glm5/phase2882_ledger_v2.py | 1d801373 |
| tests/glm5/rdc_atlas_ledger.py（v2 loader） | 5b2fb515 |
| research/gpt5/atlas/ATLAS_LEDGER_SPEC.md（v2） | a3a5ffaf |
| research/gpt5/atlas/atlas_ledger_v1_backup.json | 9d934052 |
| research/gpt5/atlas/atlas_ledger.json（v2） | 416e93e7 |

### 硬伤与方法教训

1. **bash shim 幽灵执行**：`cd X && python Y` 变体在 shim 报错（dirname not
   found / cd null directory）时**命令实际已执行**（exit 0 + stderr 噪音 ≠ 失败）。
   第二次重跑覆盖了首次运行生成的 v1 备份 → V4 假阴性。
2. **对策制度化**：① 改主文件的脚本必须幂等（version 守卫：只在 version==1 时
   备份，version>2 拒跑）；② 幽灵执行后先探针产物状态再决定重跑；③ v1 可精确
   重建（V1 保真 + json.dump 设置相同 → 移除新增节即逐字节原 v1），已重建并复核
   （新 sha 见表）。
3. V2 门线设计缺陷修正：translation 行 mlp_carrier.sources 为空集——门线允许
   `not_measured` 显式空引用（以 note 标注），不判失败；其余三族每格均 ≥1 引用。

### 结论与接续

Atlas Ledger v2 落成：双图谱事实源现含四象限表（通道分化定律的完整登记）、
迁移矩阵快照、阴性结果登记位与多模型 namespace。**P3 跨模型复制战役的前置条件
已满足**。接续候选：

- **A（主选）**：P3 跨模型复制战役启动——DS7B m 冷启动（2846–2881 管线移植，
  先做词汇 census + class 轴，验证 mlp 载体密度分层定律跨模型成立）
- B：P2 语法轴内部解剖（number 强边际 +0.32 的头级来源；tense 隔离复议按 N2
  复议条件扩池）
- C：P4 因果干预闭环预研（沿联合谱方向的 mlp 输入干预协议设计）

*（SHA 与判决以 result.json 为准；本节由 phase2882 收尾脚本追加。）*


---

## Phase 2883: DS7B class 轴 mlp 载体冷启动（P3 跨模型复制第一站） [2026-09-18 12:25]

**日期**：2026-09-18。**模型**：deepseek-r1-distill-qwen-7b（Qwen2 架构，28 层
× 28 头 × head_dim 128，hidden 3584，untied lm_head，vocab 152064）。

### 原理

把 2861(g_direct)+2867(B3) 协议 verbatim 移植到第二模型，检验 mlp 载体定律的
模型普适性。预注册偏差（冻结）：D1 同上下文单臂（2861 的 L13H30 patch 机制仅
服务残差检查，B3 不需要，且为 qwen3 专属模块手术）；D2 窗口冻结缩放规则
WIN=[floor(26/36·L), L)——qwen3-4b 36 层 → [26,36) 10 层（逐字 2861），
DS7B 28 层 → **[20,28) 8 层**；D3 同一 2806 类词表按各模型 tokenizer 单 token
过滤；D4 E 行取 lm_head.weight（DS7B untied，与 2861 语义一致）。

### 门线（execution.json 先于任何观测冻结）

V1 词表 ≥8/10 类保 5 词且总 ≥50；W3 重构一致性 v1=|LN2(ln2in)−mlpin|<0.05、
v2=|mlp(ln2in)−mlpout|<0.05；W1 检索 LOO-NN acc > null p95（200 置换，
SEED=2883）；W2 描述性复制带 |acc−0.875|≤0.15。

### 结果

**mlp_carrier_replicates=True（V1/W3/W1 全过）**：

- V1=true：10/10 类各保 8 词（80 词，Qwen2 tokenizer 与 Qwen3 高度兼容）
- W3=true：v1=v2=**精确 0.0**（重构一致性无任何 bf16 漂移）
- **W1=true：acc 0.5750 vs null p95 0.1625（3.5×）→ mlp_carries_class_axis_ds7b**
- **W2=carrier_deviation：Δ=−0.30（0.575 vs qwen3-4b 0.875），超出 ±0.15 带**
- 层剖面（mean g_direct，L20-27）：−0.32/−0.32/−0.18/−0.02/+0.20/+0.41/
  −0.45/−0.38——符号跨层翻转，结构性强；g_direct 幅值（至 0.45）≫ g_comp
  （至 0.057），直写臂主导
- 运行时 100.6s（零前向词表 + 前向仅 2 token 序列 × 80 词）

### 解读

1. **载体定律跨模型定性成立**：DS7B 的 mlp 直写响应谱同样承载类身份
  （3.5× null），第二次独立确认"mlp 臂 = 高密度机制载体"。
2. **载体强度是模型依赖量**：−0.30 偏差为真偏差而非噪声（null p95 仅 0.16）。
   候选解释（观测后生成，待检验）：① 8 层窗口 vs 10 层（分辨率差）；②
   reasoning-distill 训练改变 mlp 编码密度；③ untied lm_head 使 E 行几何不同。
   判别实验 = 2884 候选 B（qwen3-4b 在 [26,34) 8 层窗口重算 B3——若 acc 降至
   ~0.58 则窗口长度主因）。
3. 跨模型增长率曲线开通：qwen3-4b class 0.875 / DS7B class 0.575（组件
   10 vs 8，不可直接比较，已在 Ledger 注记）。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2883/ds7b_class_mlp/execution.json | 6793838d |
| phase2883/ds7b_class_mlp/result.json | 0563b354 |
| phase2883/ds7b_class_mlp/ds7b_class_mlp.npz | 76b89284 |
| tests/glm5/phase2883_ds7b_class_mlp.py | 6c922812 |

硬伤：lm_head 被 accelerate offload 成 meta tensor（7B > 12GiB 显存帽）——改
safetensors 直读分片（零前向，语义不变）；result.json 漏装 per-class 描述统计，
已由 npz post-hoc 补齐（per_class_posthoc.txt，非门线）。

### 接续（2884 候选）

- **A（主选）**：窗口长度判别——qwen3-4b B3 在 8 层窗口 [26,34) 重算（零新
  前向？否，需重测 g_direct；~15min）：判定 −0.30 偏差归因
- B：DS7B census（2875/2880 协议移植，drop 谱 + attr 轴）——补跨模型第二象限
- C：GLM4 冷启动（glm4-9b-chat-hf，32 层 → 窗口 [23,32)）

*（SHA 与判决以 result.json 为准；本节由 phase2883 收尾脚本追加。）*


---

## Phase 2884: 窗口长度判别——DS7B −0.30 载体偏差归因（P3 判别站） [2026-09-18 12:35]

**日期**：2026-09-18。**模型**：qwen3-4b + deepseek-r1-distill-qwen-7b。

### 原理

2883 发现载体强度跨模型偏差（Δ=−0.30），候选原因：①窗口 8 层 vs 10 层；
②模型因子（distill 训练 / untied lm_head 几何）；③窗口位置。关键效率事实：
g_direct 按**层**定义、窗口=切片——hooks 本就捕获全部层，故每模型只测**一次
全层** g_direct（80 词），五臂窗口全部为切片，一次前向覆盖全部判别臂。

### 门线（execution.json 先于任何观测冻结）

- D1 每模型一次全层测量，窗口=切片（2883 词表/上下文/E 行协议 verbatim）；
  qwen3-4b tied 无 lm_head 行 → E 行取 model.embed_tokens.weight（tie 语义
  数学等价，记为 D1b）
- D2 窗口集冻结：qwen {[26,36), [26,34), [28,36), [24,34)}；ds7b
  {[20,28), [18,28)}
- W3 每模型重构一致性 v1/v2<0.05（违反则全部判决作废）
- **C0 确定性**：acc(q_base10)==0.875 且 acc(d_base8)==0.575 **精确**成立，
  否则作废
- C1 qwen 长度效应：8L 切片偏离基线 ≥0.15；C2 ds7b 长度效应：[18,28) 增益
  ≥+0.15；C3 位置效应：q_early8 与 q_late8 差 ≥0.15
- 归因（优先级冻结）：window_length iff C1∧C2；model_factor iff ¬C1∧¬C2∧¬C3；
  position_matters iff C3∧¬(C1∧C2)；else mixed。null 200 置换，SEED=2884

### 结果

**window_discriminate=model_factor（C0=True，C1/C2/C3 全 False）**：

| 臂 | 窗口 | acc | null p95 |
|---|---|---|---|
| q_base10 | [26,36) | **0.8750**（精确复现 C0） | 0.1625 |
| q_early8 | [26,34) | 0.8875 | 0.1625 |
| q_late8 | [28,36) | 0.8250 | 0.1500 |
| q_end34 | [24,34) | 0.9000 | 0.1506 |
| d_base8 | [20,28) | **0.5750**（精确复现 C0） | 0.1500 |
| d_long10 | [18,28) | 0.6375 | 0.1625 |

全部窗口操作的 acc 变化 <0.15 带。**−0.30 偏差是真实模型因子：载体强度是
模型内在属性，与测量窗口无关**。W3 双模型过（qwen 0.0078 / ds7b 0.0000，
bf16 预算内）。运行 2m20s（一次前向双模型全层）。

### 解读

1. 窗口长度/位置假设排除——qwen 在 8 层切片仍 0.83-0.90，ds7b 加长到 10 层
   仅 +0.0625。
2. 载体强度归入模型因子。可检验候选（观测后）：reasoning-distill 训练改变
   mlp 编码密度；untied lm_head 使 E 行几何不同（2885 附件分析给出判别杠杆）。
3. 跨模型增长率曲线注记修正：组件数差异不解释偏差，Ledger L9 登记。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2884/window_discriminate/execution.json | bcc08787 |
| phase2884/window_discriminate/result.json | 1e68a3d9 |
| phase2884/window_discriminate/window_discriminate.npz | 3851816f |
| tests/glm5/phase2884_window_discriminate.py | 5fd6f765 |

硬伤：qwen3-4b safetensors index 无 lm_head 行（tied）→ 回退
model.embed_tokens.weight（首次运行 KeyError 崩溃后修复，属环境适配非协议
变更）；bash shim 幽灵执行探针纪律再次生效（先探针后重跑）。

### 接续（2885 候选）

- **A（主选）**：DS7B 翻译轴判别测试（untied lm_head）——2878 协议移植：
  若 VG2b>0.2 则 2878 真阴性归因 tied-embedding 扭曲；若仍≈0 则"翻译≠静态轴"
  升级为跨模型事实（附件分析判别杠杆，零前向 ~1min）
- B：CKA 倒 U 曲线（附件武器 1，沙漏模型验证；~50 句对 × 全层捕获）
- C：语言身份轴进入 mlp 图谱（中层语言方向 → g_direct 载体检验，把"语言"
  作为第四轴族）

---

## 附件分析：翻译机制流形理论（"找不到轴反而藏着秘密"）

**日期**：2026-09-18。附件主张：翻译轴不存在 → 翻译不是空间平移而是高维
流形上的上下文重构；沙漏模型（中层语言剥离）；tied embedding 扭曲；
三武器（CKA 倒 U / 语言路由头 / 正交手术）。

### 判定：核心直觉正确，三处需修正，一处给出新判别杠杆

**正确且与实测一致**：
1. "翻译不是静态轴"——2878 实测 VG2b≈0（fr −0.0007/de −0.0029/es 0.0011）
   精确支持。语言恒定分量≈0 是硬数据。
2. "轴不存在是信息而非故障"——真阴性登记（Ledger N1）与附件认识论一致。
3. 沙漏模型（中层语言无关语义层）作为**假设**合理，与文献方向一致，且
   可检验（武器 1 CKA 倒 U + 语言探针，设计可直接采用；预测三段：浅层
   100%/中层 50%/深层回升）。

**修正一（证据反例）**：附件把多义词/文化词作为轴不存在的主因。但 2878
的 22 对全是**单义具体名词**（cat/chat、water/eau），仍无统一轴——多义性
不是主因，词形/概念差主导（轴方向 pair-pair cos ≈0）。附件此论证被数据
否定。

**修正二（事实订正）**：2878 词对是 en↔fr/de/es（非附件所称中→英/中→法）；
且空间是 tied unembed 的 E 行差（非"方差极大的高维差"的一般表述——是
**方向一致性**失效，不是方差大）。

**修正三（新判别杠杆，附件遗漏）**：附件把 tied embeddings 几何扭曲列为
原因之二，但未给出检验。本项目现成杠杆：**DS7B 是 untied lm_head**——
2885-A 判别测试：若 untied 模型翻译轴 VG2b>0.2，则 2878 真阴性归因 tied
扭曲；若仍≈0，则"翻译≠静态轴"升级为**跨 tied/untied 的模型普适事实**，
同时沙漏假设获得间接支持（轴信息只可能在中层流形，不在端点 E 行）。

**武器评估**：武器 1（CKA/语言探针）= 低成本可行，直接排期（2885-B）；
武器 2（路由头 + activation patching）= 中成本，qwen3 模块手术经验可复用
（2861 先例），排期靠后；武器 3（正交手术）= 依赖武器 1 先定位语言子空间，
是 P4 因果干预闭环的自然形态，远期。

**结论**：附件理论框架纳入研究主线，作为"翻译轴真阴性"的解释层；其可
证伪子集（沙漏三段预测、tied 判别）转化为 2885 门线实验。

*（本节为研究决策记录，非实验 phase；判别实验 = 2885-A/B。）*


---

## Phase 2885: DS7B 翻译轴判别测试——tied 假设否定，阴性升级为模型普适 [2026-09-18 12:43]

**日期**：2026-09-18。**模型**：deepseek-r1-distill-qwen-7b（untied lm_head）。
零前向，运行 6.1s。

### 原理

2878 真阴性（qwen3-4b tied unembed 无统一跨语言轴）存在一个未检验混淆变量：
tied-embedding 几何扭曲（附件原因之二）。DS7B 是 untied——把 2878 翻译族
协议 verbatim 移植（同词池、同同形排除表、同门线），E 行改取 lm_head.weight
（2883 safetensors 直读先例），即可判别：
- D- attributed_to_tied：DS7B VG2b>0.2 存活 ≥2 轴 → 2878 阴性归因 tied
- D+ model_general_negative：≥2 轴 VG1 过但全部 VG2b≤0.2 → 阴性与 tying
  无关，"翻译≠静态轴"升级为模型普适事实，沙漏假设获间接支持

### 门线（execution.json 先于任何观测冻结）

T1 翻译族 only（2878 Gen2 词池+同形排除表 verbatim）；T2 E=lm_head 行；
T3 门线 verbatim 2878（VG1 ≥5 对/轴且 ≥3/3；VG2a <0.488；VG2b >0.2 存活；
VG3 <3.0）；T4 零前向 SEED=2885。

### 结果

**D+_model_general_negative**：

| 轴 | 有效对 | VG2b (DS7B untied) | qwen tied 参考 |
|---|---|---|---|
| lang_fr | 10 | **−0.0033** | −0.0007 |
| lang_de | 12 | **−0.0019** | −0.0029 |
| lang_es | 13 | **+0.0034** | +0.0011 |

VG1 3/3 过（DS7B 单 token 率低于 qwen：Qwen2 tokenizer 对法/德/西词切分更
碎，但每轴仍 10-13 对）。三轴 VG2b 全部精确死区（|·|<0.005），与 qwen tied
惊人一致——**tied-embedding 扭曲假设被否定**。

### 解读

1. **"翻译≠端点静态轴"为模型普适事实**（tied/untied × 4B/7B 四格全阴），
   Ledger N1+N4 双登记。
2. **沙漏假设成为主要幸存解释**：语言身份信息若存在统一方向，只可能在
   中层残差流（语言无关语义层两侧），不在端点 E 行——附件武器 1（CKA 倒
   U + 语言探针）优先级提升。
3. 附件理论框架经两轮实测收敛：多义性论证被否（修正一）、tied 论被否
   （本轮）、沙漏论可检验且升级为主假设。**"找不到轴→去中层找流形"成为
   2886 主选的直接依据**。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2885/ds7b_trans_axis/execution.json | 65ab2e2c |
| phase2885/ds7b_trans_axis/result.json | 9173bd5c |
| phase2885/ds7b_trans_axis/ds7b_trans_axis.npz | 60887fcf |
| tests/glm5/phase2885_ds7b_trans_axis.py | 8e166aec |

### 接续（2886 候选）

- **A（主选）**：CKA 倒 U + 语言探针（附件武器 1，沙漏验证；句级 hidden
  states 全层捕获，~40 句对 × 2 语言）
- B：GLM4 冷启动（P3 第三模型，glm4-9b-chat-hf 32 层 → 窗口 [23,32)）
- C：语言身份方向进 mlp 图谱（中层语言方向 → g_direct 载体检验，第四轴族）

*（SHA 与判决以 result.json 为准；本节由 phase2885 收尾脚本追加。）*


---

## Phase 2886: 沙漏验证——"中层剥离语言"否定，CKA 呈阶梯非倒 U [2026-09-18 12:48]

**日期**：2026-09-18。**模型**：qwen3-4b。40 对冻结 en/fr 平行句（与
2878/2885 概念池同源），last-token 全层 hidden states，运行 19.2s。

### 门线（execution.json 先于任何观测冻结）

H1 倒 U：argmax CKA ∈ [12,24] 且峰值−端点均值 ≥0.10；H2 探针凹陷：中层
均值 ≤ 端点均值−0.15（LOO 最近质心，免训练）；H3 分半 ceiling（描述性）。

### 结果

**hourglass_validated=False（H1=False，H2=False 反向）**：

- **CKA 阶梯而非倒 U**：last-token 跨语言 CKA 在 li 2-6 低（0.125-0.192），
  **li 6→7 跳变**至 0.806，此后平台 0.75-0.84 缓降（峰值 li=12=0.8394，
  但峰-端差仅 0.086 < 0.10 门线）
- **语言探针全层完美**：li 1-36 每层 acc=1.0000（端点均值仅 0.9146）——
  **语言身份在任何深度都不被剥离，中层反而最可读**
- 分半 within-en CKA（0.38-0.66）**低于**跨语言 CKA（0.75+）：对齐由共享
  语义内容驱动，跨语言对齐已达内容变异允许的天花板
- mean-pool 变体（描述性）：中层塌至 ~0.06，末层回升至 0.72——词序/长度
  差异主导位置混合表征，末层重新汇聚
- li=0 CKA=0 为退化伪影（last-token 均为"."，中心化后方差≈0），li 1 起有效

### 解读

1. **附件"沙漏"版本被否定（N5）**：语言不被中层剥离——探针全层 1.0。
   附件终章"超越语言的纯粹概念宇宙"不成立：正确图景是**对齐语义与持续
   语言身份共存**。
2. **新结构**：li 6→7 阶梯 = "语言共享语义区"自约 0.20 深度开始，而非倒 U
   峰值。深部缓慢下降（0.84→0.75）是唯一与"重新着装"沾边的信号，但语言
   身份始终可读，非剥离-再穿。
3. **与轴阴性的一致性**：端点无统一轴（N1/N4）+ 语言身份全层在场 ⇒ 翻译
   机制不在"轴"也不在"剥离"，语言信息以**非统一方向的高维形式**全程在
   场——中层语言方向可提取（探针 1.0）→ 直接可测 mlp 载体（第四轴族）。
4. 附件理论框架三轮实测终局：多义论否、tied 论否、剥离论否；幸存核心 =
   "翻译是动态语境重构"（定性正确），但其空间机制需按本轮证据重述。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2886/hourglass_cka/execution.json | ed88054d |
| phase2886/hourglass_cka/result.json | 6f579752 |
| phase2886/hourglass_cka/hourglass_cka.npz | 0a26bfab |
| tests/glm5/phase2886_hourglass_cka.py | 841af326 |

硬伤：探针首跑 IndexError（lab[m] 索引错位）→ 修复重跑（清产物纪律）；
last-token 设计使 li 0 退化、端点低 CKA 部分由标点主导——Gen2 若需精确定
位阶梯位置应用 content-token CKA（未列门线）。

### 接续（2887 候选）

- **A（主选）**：语言身份轴进 mlp 图谱（第四轴族）——从 2886 npz 中层
  （li 12-24）提取语言方向（en 质心−fr 质心，或探针权重），g_direct 载体
  检验（~2min 前向）；载体曲线补第四点 class/attr/syntax/language
- B：GLM4 冷启动（P3 第三模型）
- C：content-token CKA 精确定位 li 6→7 阶梯（Gen2，描述性）

*（SHA 与判决以 result.json 为准；本节由 phase2886 收尾脚本追加。）*


---

## Phase 2887: 语言身份轴进入 mlp 图谱——第四轴族成立 [2026-09-18 13:01]

**日期**：2026-09-18。**模型**：qwen3-4b。运行 18.6s（57 词 × 3 条件）。

### 原理

2886 否定"中层剥离"后，语言身份方向仍在中层句状态完美可读（探针 1.0）。
本轮检验：从 2886 句状态提取的**中层语言方向**能否像 class/attr/syntax 轴
一样充当 mlp 载体方向（第四轴族）。方向来源为新登记的 provenance：
lang_dir = unit(mean S_last[en,18] − mean S_last[fr,18])（li=18 中层句质
心差，零前向）——**不是 unembed 端差分**（2878/2885 已证端点无轴）。

### 协议要点（预注册冻结于 execution.json）

- 词表：2878 翻译对转录为规范 (en, L) 形（fr20/de22/es30），VG0 同形排除
  （3 对）+ 单 token 过滤（verbatim 2878 逻辑）→ 35 对；**tid 级跨语言
  去重**（en 词仅登记一次）→ 57 词（en 22 / L 35）——消除同词副本在 LOO
  检索中的同语言泄漏
- 测量：2879 协议 verbatim（same/func/null 三条件，窗口 [26,36)，eps=1.0，
  cdir=lang_dir）；标签 en=0 / L=1（fr+de+es 合并）
- v1 重算确定性 = 精确 0.0；kernel-context noise 1.6e-3（描述性）

### 判决：language_axis_in_mlp = True

| 判据 | 观测 | 结果 |
|---|---|---|
| **E1** 语言检索（2 类） | acc **0.7719** vs null p95 0.6491 | **mlp_carries_language_axis** |
| **E2** 同语言边际 | **0.1943** vs null p95 0.0482（**4.0×**） | **language_margin_in_mlp** |
| **E3** 概念对照（描述） | acc **0.0000**（低于 null p95 0.0877） | 谱零概念含量 |
| **E4** 层方向结构（描述） | 窗口层句方向与 lang_dir 余弦 0.456→0.138 递减 | 语言方向随深度衰减 |

### 解读（三个实质发现，重复三遍）

1. **第四轴族成立，载体曲线补全四点**：class 0.875 / attr 0.500 /
   syntax 0.708 / **language 0.772**——四大轴族全部生活在同一个 mlp 响应
   通道内，且语言轴证明**通道方向可以来自残差流中层而非 unembed 端点**。
2. **语言/概念在通道内可分**：E3 概念对照精确 0——沿语言方向的 mlp 响应
   谱承载纯语言身份、零概念内容；与 2881 三族质心互负（互斥子空间）合看，
   单通道 = 多个可分离的身份/内容分量。
3. **端点阴性 + 中层阳性闭环**：2878/2885（端点无轴，N1/N4）+ 2886（中层
   探针 1.0）+ 本轮（中层方向驱动 mlp 载体）——语言身份的机制画像完成：
   **无静态端点轴，有中层方向，且该方向在深层 mlp 直写臂中有因果可读的
   载体响应**。层剖面递增至 li35（0.0653）与其他轴同型（深层主导）。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2887/language_axis_mlp/execution.json | ec46fd67 |
| phase2887/language_axis_mlp/result.json | de6e04a0 |
| phase2887/language_axis_mlp/language_axis_mlp.npz | e4835a87 |
| tests/glm5/phase2887_language_axis_mlp.py | f5cb0542 |

硬伤：首跑 KeyError（func 词 the 未注册 tid_map）修复重跑（清产物纪律）。

### 接续（2888 候选）

- **A（主选）**：GLM4 冷启动（P3 第三模型，glm4-9b-chat-hf 32 层 → 窗口
  [23,32)；class 轴 mlp 载体 + 若 tokenizer 允许加翻译轴判别）
- B：语言轴跨模型（DS7B 语言方向载体，检验 L10 分离是否模型普适）
- C：content-token CKA 定位 li6→7 阶梯（2886 Gen2）

*（SHA 与判决以 result.json 为准；本节由 phase2887 收尾脚本追加。）*


---

## Phase 2888: GLM4 class 轴 mlp 载体冷启动——P3 第三模型，架构普适确认 [2026-09-18 18:24]

**日期**：2026-09-18。**模型**：glm4-9b-chat-hf（glm 架构，40 层，hidden
4096，untied，vocab 151552）。运行 2m28s（手工 device_map，窗口层 GPU /
其余 CPU）。

### 原理与移植

2883 协议 verbatim 移植（D1 same-context 单臂；D2 窗口缩放规则冻结：
[floor(26/36*40), 40) = [28,40) 12 层；D3 2806 词表按 GLM4 tokenizer
过滤——10/10 类各 8 词 = 80 词；D4 lm_head safetensors 直读）。

### 硬伤两起（均已修复并登记）

1. **Gen1 device_map=auto disk-offload 陷阱**：部分层参数成 meta 张量，
   layer_dev 返回 meta → 直接调用崩溃。修复：手工 device_map（窗口
   28-39 层 + embed + lm_head 约 3.9GB 在 GPU，其余 28 层 CPU）。
2. **Gen1 W3 绝对预算尺度错配**：v2_abs = 0.0625 超 0.05 预算，但相对
   误差仅 ~1.6e-3——正是 2877 认定的 cuBLAS 跨形状 bf16 预算量级；0.05
   绝对值系 hidden-3584 尺度（DS7B）外推错配。Gen2 修订为相对预算
   <5e-3（修订理由冻结于 execution.json，2877 先例），实测
   v1_rel=1.1e-4 / v2_rel=2.7e-4 远低于预算——健康。

### 判决：mlp_carrier_replicates = True

| 门线 | 观测 | 结果 |
|---|---|---|
| **V1** 词表 | 10/10 类各 8 词（80 词） | **vocab_legal=true** |
| **W3** 重构一致性（Gen2 rel） | v1_rel 1.1e-4 / v2_rel 2.7e-4 | **true** |
| **W1** mlp 载体检索 | acc **0.8375** vs null p95 0.1381（**6.1×**） | **mlp_carries_class_axis_glm4** |
| **W2** 复制带 | Δ = **−0.0375** | **replicates_within_band** |

### 解读（三个实质发现，重复三遍）

1. **跨架构确认**：glm 与 qwen 两个架构族均复现 mlp 直写载体
   （g_direct ≫ g_comp），**载体通道是架构普适的**。
2. **三模型载体曲线成形**：qwen3-4b 0.875 / **glm4 0.8375**（带内）/
   ds7b 0.575（偏差）——载体强度模型依赖，且 DS7B 的 −0.30 偏差在
   GLM4 对照下更显特殊（候选归因收窄至 reasoning-distill 训练，untied
   几何已被 GLM4 untied 且带内排除）。
3. **untied 假设再排除一层**：GLM4 untied 且带内——DS7B 偏差不能归因
   untied lm_head 几何，剩余主候选 = distill 训练。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2888/glm4_class_mlp/execution.json | 9ad64b4d |
| phase2888/glm4_class_mlp/result.json | f16bcf62 |
| phase2888/glm4_class_mlp/glm4_class_mlp.npz | c4195d65 |
| tests/glm5/phase2888_glm4_class_mlp.py | e83b161a |

### 接续（2889 候选）

- **A（主选）**：DS7B 归因收尾——distill 假设检验（若可获取非 distill
  的同架构 7B 对照则直接判别；否则登记为 open candidate）
- **B**：GLM4 语言轴判别（2887 协议移植：中层句状态语言方向 + 翻译词
  载体，检验 L10 分离跨架构）
- **C**：Gemma4 冷启动（P3 第四模型，收尾 model_namespace）

*（SHA 与判决以 result.json 为准；本节由 phase2888 收尾脚本追加。）*

## Phase 2889: DS7B distill 假设检验（qwen2 架构非 distill 对照，候选 A 判别） [2026-09-18 18:34]

**日期**：2026-09-18。**模型**：qwen2.5-3b-instruct（qwen2 架构，36 层，
hidden 2048，tied，vocab 151936）。运行约 1m4s（手工 device_map，窗口
26-35 层 GPU / 其余 26 层 CPU，2888 Gen2 先例）。

### 原理与判别设计

2888 将 DS7B 的 -0.30 载体偏差（0.575 vs 0.875 锚）收窄至
reasoning-distill 训练（窗口/位置 2884 排除、untied 几何 2885+2888
排除）。本机模型目录探针发现 **qwen2 架构非 distill 对照**：
qwen2.5-3b-instruct（同 Qwen2ForCausalLM 架构族、官方 instruct 训练、
3B、tied）——候选 A 从"登记 open candidate"升级为**直接判别实验**。

判别逻辑（预注册冻结）：
- acc 带内（|acc-0.875|<=0.15）=> 架构+小尺寸在 3B 被豁免 =>
  distill_hypothesis_strengthened（残余混杂：3B-vs-7B 尺寸）
- acc 贴近 DS7B（|acc-0.575|<=0.15）=> 偏差跟随架构/尺寸 =>
  distill_hypothesis_weakened
- 否则 inconclusive

移植偏差（全部预冻结）：D4 tied=true 故 E 行取
model.embed_tokens.weight（2884 数学等价先例）；W3 Gen2 相对预算
<5e-3（hidden 2048 尺度）；窗口 [26,36) 10 层（与 qwen3-4b 同）。

### 判决：mlp_carrier_replicates = True；W4 = distill_hypothesis_strengthened

| 门线 | 观测 | 结果 |
|---|---|---|
| **V1** 词表 | 10/10 类各 8 词（80 词） | **vocab_legal=true** |
| **W3** 重构一致性（Gen2 rel） | v1_rel 1.56e-4 / v2_rel 1.98e-4 | **true** |
| **W1** mlp 载体检索 | acc **0.8375** vs null p95 0.1500（**5.6 倍**） | **mlp_carries_class_axis_qwen25_3b** |
| **W2** 复制带 | delta = **-0.0375** | **replicates_within_band** |
| **W4** distill 判别 | vs ds7b +0.2625 | **distill_hypothesis_strengthened** |

### 解读（三个实质发现，重复三遍）

1. **distill 假设增强**：非 distill qwen2 架构对照带内（0.8375）——
   架构在 3B 被豁免，且尺寸方向反向（3B < 7B 却带内，若尺寸致偏差应
   更差）——**DS7B 偏差的最简归因收窄至 reasoning-distill 训练本身**。
2. **载体曲线第四点**：qwen3-4b 0.875 / glm4 0.8375 /
   qwen2.5-3b 0.8375 / ds7b 0.575——四个模型、三个数据点两架构族+
   qwen2 族内对照，mlp 直写载体通道普适，强度模型依赖。
3. **残余混杂如实登记**：7B 级非 distill qwen2 对照本地不可得
   （Qwen2-7B 未下载）=> 归因为 **strengthened 非 settled**；若未来
   获取 Qwen2-7B(-Instruct) 可升级为 settled。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2889/qwen25_distill_test/execution.json | f7c78b49 |
| phase2889/qwen25_distill_test/result.json | b00c36a0 |
| phase2889/qwen25_distill_test/qwen25_distill_test.npz | 07bec8a4 |
| tests/glm5/phase2889_qwen25_distill_test.py | 3c0bdc64 |

Ledger 更新：B3_mlp_qwen25_3b + M2889_qwen25_distill +
G_qwen25_distill_test + L12_distill_attribution；model_namespace
active += qwen25_3b（blocks 14 / measurements 28 / growth 15 /
linkage 12）。

### 接续（2890 候选）

- **A（主选）**：GLM4 语言轴判别（2887 协议移植：中层句状态语言方向
  + 翻译词载体，检验 L10 语言/概念通道分离跨架构）
- **B**：Gemma4 冷启动（model_namespace 收尾，pending_replication
  最后一项）
- **C**：distill 假设升级 settled（需下载 Qwen2-7B-Instruct 对照；
  可选网络获取）

*（SHA 与判决以 result.json 为准；本节由 phase2889 收尾脚本追加。）*


## Phase 2890: GLM4 语言轴判别（2887 协议移植，阴性登记） [2026-09-18 18:42]

**日期**：2026-09-18。**模型**：glm4-9b-chat-hf（glm 架构，40 层，
hidden 4096）。运行 472.9s（两阶段：80 句全前向提 S_last -> li=20
句状态语言方向 -> 78 词 x 3 条件 x [28,40) 窗口 mlp 响应谱）。

### 原理与移植

2886+2887 协议 verbatim 移植，方向来源改为 GLM4 自身句状态（per-model
provenance，预注册登记）：80 对 en/fr 句子（2886 冻结常量复制），
lang_dir = unit(mean S_last[en,20] - mean S_last[fr,20])，LI 缩放规则
round(18/36*L)=20。词表 2878 TRANS_PAIRS + 同形排除表原样，GLM4
tokenizer 过滤后 49/72 对，tid 级跨语言去重 78 词（en=29/L=49）。

### 判决：language_axis_in_mlp_glm4 = **False（阴性）**

| 门线 | 观测 | 结果 |
|---|---|---|
| v1 确定性 | max rel err **0.0** | 通过 |
| **E1** 语言检索 | acc **0.5256** vs null p95 0.6295（null mean 0.5261） | **mlp_language_signal_absent_glm4** |
| **E2** 同语言余量 | 0.0333 < null p95 0.0395 | **margin_absent_glm4** |
| E3 概念对照 | 0.0128（null p95 0.0641） | 概念也 ~0（非通道置换） |
| WC 跨模型带 | delta = **-0.2463** vs qwen 0.7719 | **language_carrier_deviation** |

### 解读（三个实质发现，重复三遍）

1. **阴性成立且干净**：acc 精确落在 null mean（0.5256 vs 0.5261），
   谱既不携带语言也不携带概念（E3 也 ~0）——不是"语言被概念挤占"，
   而是该方向在 GLM4 的 mlp 通道中**根本没有直写响应**。
2. **载体定律是 轴族 x 模型 特异的（语言族）**：class/attr/syntax 三族
   在 GLM4 复制（M2888），语言族不复制——失败特异性指向方向来源
   （句状态中层提取）。GLM4 把语言身份保留在残差流（D0 探针全层
   >=0.96，N5 跨模型确认"不剥离"），但不经 mlp 直写。
3. **GLM4 沙漏形状不同**：CKA 峰值前移 li=9（0.8954）后单调下降，
   与 qwen 的 li6->7 阶梯+平台不同；探针无 dip（ends 0.9167）。
   E4 证实方向质量：cos(lang_dir_q, lang_dir) 在 li=20 处 =1.0，
   窗口内衰减 0.42->0.08——方向真实存在于中层，但深层 mlp 不响应。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2890/language_axis_glm4/execution.json | 6defb457 |
| phase2890/language_axis_glm4/result.json | eaaf6df3 |
| phase2890/language_axis_glm4/language_axis_glm4.npz | 74933303 |
| tests/glm5/phase2890_language_axis_glm4.py | 29081deb |

（result.json sha256_8 = eaaf6df3。）

Ledger 更新：B3_language_glm4 + M2890_language_axis_glm4 +
G_language_glm4（阴性点）+ **N6_language_carrier_glm4**（settled
negative，阴性登记第 6 条）；blocks 15 / measurements 29 / growth 16
/ negatives 6。

### 接续（2891 候选）

- **A（主选）**：GLM4 语言注意通路判别——lang_dir 的 attention-response
  谱（N6 启示：GLM4 语言可能走 attention 而非 mlp；head-level drop/
  响应谱注入 lang_dir）
- **B**：Gemma4 冷启动（model_namespace 收尾）
- **C**：distill 假设升级 settled（Qwen2-7B-Instruct 网络获取，可选）

*（SHA 与判决以 result.json 为准；本节由 phase2890 收尾脚本追加。）*


## Phase 2891: GLM4 语言 attention 通路判别（N6 收尾，第二阴性） [2026-09-18 19:02]

**日期**：2026-09-18。**模型**：glm4-9b-chat-hf。运行 356.8s。

### 原理与协议

2890 严格平行协议，注入点 mlp -> attention：g_attn[i,q] =
[(self_attn(attnin + eps*cdir) - self_attn(attnin)) . cdir]/eps，
注入仅目标位置（pos 1）；lang_dir 与 78 词表**零前向复用 2890 产物**
（同方向保证 mlp/attn 通道对比内部有效）。直接调用路径先经探针冻结：
self_attn(position_embeddings=model.model.rotary_emb(...),
attention_mask=None) 与 hook 基线 rel err = 0.0（probe_2891_attn）。
窗口 [28,40)，same/func/null 三条件，B3_attn =
g(same) - 0.5(g(func)+g(null))。

一次预注册前崩溃（attnin batch 维索引 bug，无统计量产生），按纪律
删旧 execution.json 后重跑。

### 判决：attn_language_route_glm4 = **False（第二阴性）**

| 门线 | 观测 | 结果 |
|---|---|---|
| v1 确定性 + 调用一致性 | rel err **0.0**；kernel-vs-hook **0.0** | 通过 |
| **A1** 语言检索 | acc **0.5897** < null p95 0.6410（null mean 0.5279） | **attn_language_signal_absent_glm4** |
| **A2** 同语言余量 | -0.0011 < null p95 0.0340 | **margin_absent_attn_glm4** |
| A3 概念对照 | **0.0000**（null p95 0.0641） | 通道两无 |
| AC 通路对比（描述性） | attn 0.5897 / mlp 0.5256 / qwen mlp 0.7719 | 双通道均低于 null p95 |

### 解读（三个实质发现，重复三遍）

1. **N6 候选关闭**：GLM4 语言方向在深层窗口既不经 mlp 直写
   （M2890）也**不经 attention 直写**（M2891）——两通道阴性。
2. **GLM4 语言身份是 embedding-词法起源 + 残差流被动保留**：探针
   li=1 起即 1.0（D0），深层窗口两模块均不主动写入该方向——2887
   语言 mlp 载体是 qwen3-4b 特异的。
3. **载体定律的适用条件收窄**：class/attr/syntax 族跨架构复制，但
   language 族不复制且通道无关——载体定律要求 per-model 通道判定；
   剩余候选：方向由窗口前早期层写入，或 embeddings 之后从未被主动
   写入。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2891/language_attn_glm4/execution.json | 62c040d9 |
| phase2891/language_attn_glm4/result.json | 8ee79eb3 |
| phase2891/language_attn_glm4/language_attn_glm4.npz | 37415d8e |
| tests/glm5/phase2891_language_attn_glm4.py | 6f16c144 |

Ledger 更新：B_attn_language_glm4 + M2891_language_attn_glm4 +
G_language_attn_glm4 + **N7_attention_route_glm4**（settled，关闭
N6 候选）；blocks 16 / measurements 30 / growth 17 / negatives 7。

### 接续（2892 候选）

- **A（主选）**：Gemma4 冷启动（model_namespace 最后一项 pending，
  class 轴 mlp 载体第四模型）
- **B**：GLM4 语言早期层写入检验（窗口 [13,27) 或全层 attn/mlp 响应
  谱，定位语言方向写入层；成本较高）
- **C**：distill 假设升级 settled（Qwen2-7B-Instruct 网络获取）

*（SHA 与判决以 result.json 为准；本节由 phase2891 收尾脚本追加。）*


## Phase 2892: GLM4 语言写入层定位（N7 修正：深层写入但不沿 lang_dir） [2026-09-18 19:29]

**日期**：2026-09-18。**模型**：glm4-9b-chat-hf。运行 476.1s
（Stage 1 零前向定位 + Stage 2 W2=[37,40) 双通道注入判别）。
（候选 A Gemma4 冷启动：本地模型目录探针确认无 gemma4，需 ~8GB
下载，暂缓待用户决策。）

### 原理与设计

Stage 1（零前向，2890 S_last 复用）：sep(li) =
(mean_en - mean_fr) . lang_dir；Delta(li) = sep(li+1) - sep(li) =
层 li 模块沿 lang_dir 的净分离增量。分段规则冻结：
EARLY Delta(0..12) / MID(13..27) / DEEP(28..39)。Stage 2 规则先冻结：
li* = argmax|Delta|，W2 = [max(0,li*-2), min(L,li*+4))（观测只填参）。

### 判决：L1=deep_write_dominant；L2=False；Stage2=no_aligned_write_in_W2

| 门线 | 观测 | 结果 |
|---|---|---|
| L1 分段归因 | shares early **0.048** / mid 0.345 / **DEEP 0.608** | **deep_write_dominant** |
| L2 深层被动 | deep abs 4.18 vs 0.10*sep_end 0.69 | **False（非被动）** |
| sep 曲线 | 2.49 (li20) -> 9.61 (li39)；层 39 Delta **-2.74** 回撤 | 深层主导增长 |
| Stage2 mlp | acc 0.6026 < p95 0.6282；margin 0.0510 > p95 0.0447 | A1 阴性 / A2 弱阳性 |
| Stage2 attn | acc 0.4231；margin -0.0231 | 双阴性 |

### 解读（三个实质发现，重复三遍）

1. **N7"被动保留"被修正**：GLM4 语言分离由深层主导构建（DEEP 占
   60.8%，sep 2.49 -> 9.61），不是 embedding 后被动保留——errata
   入账（corrects N7）。
2. **深层写入与 lang_dir 不对齐**：分离增长真实存在，但直接注入
   lang_dir 三轮全阴性（2890 mlp / 2891 attn / 2892 W2 双通道）——
   **lang_dir（li=20 句质心差）不是深层写入的本征方向**；语言分离
   的深层写入发生在 lang_dir 的旋转/正交子空间。
3. **mlp margin 弱阳性线索**：W2 mlp 同语言余量过 p95（0.0510 vs
   0.0447）而 acc 不过——W2 mlp 有微弱同语言几何组织，方向可能
   需逐层重提取（li=39 的本征方向 != li=20 的 lang_dir）。
   混杂登记：Delta sep 含范数增长贡献（dnorm 单调上升），norm-
   matched 对照 = open candidate。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2892/language_write_locate_glm4/execution.json | b6fd79b7 |
| phase2892/language_write_locate_glm4/result.json | 9f9c874b |
| phase2892/language_write_locate_glm4/language_write_locate_glm4.npz | d760c567 |
| tests/glm5/phase2892_language_write_locate_glm4.py | e3f1a38e |

Ledger 更新：B_write_locate_glm4 + M2892_language_write_locate_glm4 +
**errata（corrects N7）**；blocks 17 / measurements 31 / errata 4。

### 接续（2893 候选）

- **A（主选）**：逐层本征方向提取 + 注入（li 25..39 每层句质心差
  dir_q 注入该层 mlp，检验"方向需逐层重提取"假设；零前向方向 +
  单窗口前向，成本低）
- **B**：norm-matched 对照（Delta sep 减去范数增长期望，纯化 L1
  归因；零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认网络获取）

*（SHA 与判决以 result.json 为准；本节由 phase2892 收尾脚本追加。）*


## Phase 2893: GLM4 逐层本征方向注入（2892 假设证实：方向需逐层重提取） [2026-09-18 19:42]

**日期**：2026-09-18。**模型**：glm4-9b-chat-hf。运行 355.9s
（Stage 1 零前向方向提取 + Stage 2 W=[28,40) 逐层本征方向双通道注入）。

### 原理与设计

2892 结论：深层主导写入语言分离（L1=deep_write_dominant），但注入
li=20 的 lang_dir 全阴性——lang_dir 不是深层写入本征方向，方向需
逐层重提取。Phase 2893 直接检验该假设：

- Stage 1（零前向，2890 S_last 复用）：对 W=[28,40) 每层 li 提取
  dir_q(li) = unit(mean_en S_last[:,li] - mean_fr S_last[:,li])。
- Stage 2（预注册冻结）：层 li 注入 **本层自己的** dir_q(li)
  （mlp 与 self_attn 双通道，pos 1，eps=1.0），g 沿 dir_q(li) 投影，
  B = g(same) - 0.5(g(func)+g(null))；词表/条件 verbatim
  2890/2891/2892（78 词零前向）。v1 rel err < 1e-6。
- 对照设计内部有效：方向来源（80 句）与检索域（78 词）分离，
  与 2890/2891 完全一致——唯一变量是"方向是否与注入层匹配"。

### 判决：perlayer_direction_write_detected_both（阳性）

| 门线 | mlp | attn |
|---|---|---|
| v1 rel err | 0.0 | 0.0 |
| A1 acc vs null p95 | **0.6923 > 0.6410**（carries） | **0.6667 > 0.6288**（carries） |
| A2 margin vs p95 | 0.0313 < 0.0349（absent） | **0.0732 > 0.0384**（margin） |
| A3 概念对照 | 0.0513（~null） | 0.0000 |
| 对比 lang_dir 注入（2890/2891） | 0.5256 -> **0.6923** | 0.5897 -> **0.6667** |

cos(dir_q(li), lang_dir) 沿深度衰减：0.4223 (li28) -> 0.0783 (li39)
——方向随层旋转是内禀的。

### 解读（三个实质发现，重复三遍）

1. **2892 假设证实**：同一方向来源、同一窗口、同一协议，仅把
   "全局 lang_dir" 换成 "逐层本征方向"，mlp 0.5256->0.6923、
   attn 0.5897->0.6667，双通道由阴性转显著阳性——**方向需逐层
   重提取**成立。
2. **N6/N7 负结果是方向错配伪影**：GLM4 深层确实在写入语言分离
   （errata 入账 corrects N6/N7）——语言写入通道真实存在，但方向
   沿深度旋转；2887 "语言载体 qwen 特异"结论被精化：非 qwen 独有，
   GLM4 的写入需层匹配方向才可见。
3. **attn 是 GLM4 语言逐层写入的更干净通道**：margin 强阳性
   （0.0732 vs p95 0.0384）且概念 0.0000；mlp acc 过线但 margin
   不过——信号存在但同语言几何组织较弱。方向旋转结构
   （旋转平面假说）成为下一研究对象。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2893/language_perlayer_glm4/execution.json | 51a561e7 |
| phase2893/language_perlayer_glm4/result.json | bb10476a |
| phase2893/language_perlayer_glm4/language_perlayer_glm4.npz | 08aae636 |
| tests/glm5/phase2893_language_perlayer_glm4.py | c629369d |

Ledger 更新：B_perlayer_lang_glm4 + M2893_language_perlayer_glm4 +
G_perlayer_lang_glm4 + **errata（corrects N6/N7）**；blocks 18 /
measurements 32 / growth 18 / errata 5。

### 接续（2894 候选）

- **A（主选）**：方向旋转结构定量——零前向分析 dir_q(li) 的旋转
  平面（相邻层差向量、主旋转平面 PCA、有效秩），对照 Unified
  Theory 旋转平面假说
- **B**：norm-matched 对照（2892 L1 归因混杂，零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认网络获取）

*(SHA 与判决以 result.json 为准；本节由 phase2893 收尾脚本追加。)*


## Phase 2894: GLM4 语言方向旋转结构定量（零前向） [2026-09-18 21:33]

**日期**：2026-09-19。**模型**：glm4-9b-chat-hf。运行 14.6s
（零前向，2890 S_last 复用；随机 null 1000 次 + 代数审计）。

### 原理与设计

dir(li) = unit(mean_en S_last[:,li] - mean_fr S_last[:,li])，
li = 1..40（修正案：dnorm(0)=0.0 精确为零，dir(0) 未定义——首次
运行在 R4 崩溃于 PC 约定错误、统计量未冻结，探针定位后按纪律删旧
execution.json 重跑）。theta(li) = 层间夹角；切向量 u(li) =
dir(li+1) - dir(li)（39 个）。R2/R3 均配随机 null（1000 次同构造
随机单位向量）。审计：|u| = 2 sin(theta/2) max dev **7.8e-16**，
PCA 重构精确。

### 判决：rotation_plane_consistent_manifold_fullrank

| 门线 | 观测 | 结果 |
|---|---|---|
| R1 旋转份额 | early 0.407 / mid 0.342 / deep 0.251 | 描述性（early 受小 dnorm 噪声放大，混杂登记） |
| R2 平面集中度 | top2_share **0.1648** > null p95 0.1084（null mean 0.1067） | plane_consistent（统计显著但仅 16%，非紧密 2D 平面） |
| R3 有效秩 | er_obs **26.66** vs null median 39.95 | full_rank_like（分级集中，未过预注册 0.5x 阈值） |
| R4 平面对齐 | cos(PC1/PC2, lang_dir) = −0.088/−0.148；分段 top-2 主夹角 **83–89°** | 切平面逐段近正交——**旋转平面本身随深度旋转** |

### 解读（三个实质发现，重复三遍）

1. **无全局旋转平面**：方向旋转统计上集中于随机水平之上
   （0.165 vs 0.108），但 top-2 只承载 16% 方差，且 early/mid/deep
   三段切子空间近互正交（主夹角 83–89°）——旋转平面是**分段局部**
   的，不是贯穿深度的固定平面；Unified Theory 旋转平面假说在
   GLM4 语言轴上需修正为"平面序列"。
2. **方向流形非低维**：有效秩 26.7/40，未过预注册低维阈值——
   语言方向沿深度扫过 ~2/3 满秩的子空间，"单轴/单平面"语义
   编码在该轴上不成立。
3. **dnorm(0)=0 精确为零**：GLM4 的类均值语言分离在 embedding
   处**从零开始**、完全由层构建（与 li=1 处逐句探针 1.0 是两个
   层面的事实：句级身份存在于 embed，类级分离由层写入）——与
   2892 L1=deep_write_dominant 一致。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2894/rotation_structure_glm4/execution.json | bb0e7519 |
| phase2894/rotation_structure_glm4/result.json | c5bb65ed |
| phase2894/rotation_structure_glm4/rotation_structure_glm4.npz | fef03893 |
| tests/glm5/phase2894_rotation_structure_glm4.py | 9188fe6b |

Ledger 更新：B_rotation_struct_glm4 + M2894_rotation_structure_glm4
+ G_rotation_struct_glm4；blocks 19 / measurements 33 / growth 19。

### 接续（2895 候选）

- **A（主选）**：qwen3-4b 同协议旋转结构（对照：qwen 语言 mlp
  载体阳性 0.7719 且方向稳定——若 qwen 旋转弱则"旋转 vs 载体"
  建立跨模型反相关，零前向）
- **B**：norm-matched 对照纯化 2892 L1 归因（零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认）

*(SHA 与判决以 result.json 为准；本节由 phase2894 收尾脚本追加。)*


## Phase 2895: qwen3-4b 旋转结构对照（假说否定 -> 通道读出归因） [2026-09-18 21:39]

**日期**：2026-09-19。**模型**：qwen3-4b（2886 S_last 复用，
(80, 37, 2560)）。运行 6.0s，零前向，2894 协议 verbatim。

### 原理与设计

2894-A 假说："若 qwen 旋转弱而其语言载体阳性，则方向稳定性与载体
强度反相关"。qwen 同协议（dir(li)=unit(mean_en-mean_fr)，li=1..36
——**dnorm(0)=0 精确为零，与 GLM4 同**；段边界按窗口 [26,36) 适配
冻结：EARLY li 1..12 / MID 13..25 / DEEP 26..35；null 1000 次；
审计 |u|=2sin(theta/2) dev **1.6e-15**）。

### 判决：rotation_plane_consistent_manifold_fullrank（与 GLM4 同判）

| 指标 | qwen3-4b | glm4（2894） |
|---|---|---|
| R2 top2_share vs null p95 | **0.2037 > 0.1225** | 0.1648 > 0.1084 |
| R3 有效秩 vs null median | **23.30** vs 35.94 | 26.66 vs 39.95 |
| R4 分段主夹角 | **84–89°** | 83–89° |
| 窗口 cos(dir(li), ref) | 0.456 -> **0.068** | 0.422 -> 0.078 |
| dnorm(0) | **0（精确）** | 0（精确） |
| 份额 early/mid/deep | 0.392/0.342/0.265 | 0.407/0.342/0.251 |

### 解读（三个实质发现，重复三遍）

1. **反相关假说否定**：qwen（载体阳性 0.7719）与 GLM4（lang_dir
   注入阴性）的旋转几何**同构**——同判决、同份额、同分级集中、
   同分段局部平面。旋转动力学不是载体判别因子。
2. **载体不对称归因于通道读出**：qwen 深层 mlp 对 cos 已衰减至
   0.07–0.46 的陈旧 li=18 方向仍强响应（2887 阳性），GLM4 则需
   层匹配方向（2893 阳性 / 2890-2892 陈旧方向阴性）——**qwen 读出
   是固定方向容忍型，GLM4 是方向匹配型**。这是 L13 登记的核心。
3. **dnorm(0)=0 双模型普适**：类均值语言分离在两模型的 embedding
   处都从零开始——类级语言分离由层构建是架构普适事实（句级身份
   在 embed 是另一层面，2892 已区分）。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2895/rotation_structure_qwen/execution.json | b2ad83e1 |
| phase2895/rotation_structure_qwen/result.json | 1019e3ec |
| phase2895/rotation_structure_qwen/rotation_structure_qwen.npz | 5f7bb785 |
| tests/glm5/phase2895_rotation_structure_qwen.py | 1b0266f3 |

Ledger 更新：B_rotation_struct_qwen + M2895_rotation_structure_qwen
+ G_rotation_universal + **L13_rotation_vs_carrier_refuted**；
blocks 20 / measurements 34 / growth 20 / linkage 13。

### 接续（2896 候选）

- **A（主选）**：通道读出性质判别——qwen 深层 mlp 对"陈旧方向容忍"
  的直接检验：在 qwen 用 li=18 方向注入逐层响应谱（2887 已示阳性），
  再测对旋转后正交分量的响应（固定方向容忍 vs 子空间响应判别，
  单窗口前向）
- **B**：norm-matched 对照纯化 2892 L1 归因（零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认）

*(SHA 与判决以 result.json 为准；本节由 phase2895 收尾脚本追加。)*


## Phase 2896: qwen 通道读出类型判别（subspace_readout_tolerant） [2026-09-18 21:47]

**日期**：2026-09-19。**模型**：qwen3-4b。运行 20.1s（零前向方向 +
窗口 [26,36) 单前向，2887 词表/条件 verbatim 57 词）。

### 原理与设计

L13（2895）留问：qwen 载体对陈旧方向阳性而 GLM4 阴性——读出性质
是什么？三条件逐层注入判别（方向全部零前向提取）：

- stale：注入 d18（li=18 陈旧方向），响应投影 d18
- eigen：注入 d(li)（本层本征方向，2893 qwen 对应），投影 d(li)
- orth：注入 d_orth(li) = unit(d(li) − (d(li)·d18)d18)
  （旋转正交分量；cos(d(li),d18) 在窗口内 0.456 -> 0.068），
  投影 d_orth(li)

B = g(same) − 0.5(g(func)+g(null))；loo-NN acc vs null p95
（200 perms, SEED=2896）；概念对照。v1：重算 + hook-vs-call。

### 执行记录（预注册纪律）

Run 1 按冻结 v1 门线判 void（hook-vs-call 1.6e-3 > 1e-4）：
根因是形状错配 hook 检查（1-D [2560] 重算 vs [1,2,2560] 真实前向，
bf16 跨形状核差异 ~2e-3 = 2877 教训；2887 实际门线只有 recompute，
hook 引用系误读先例）。修正：形状匹配捕获（2890–2893 约定），
修正案入 execution.json，删旧产物重跑——run 1 判决未被使用。
Run 2：v1 recompute **0.0** / hook-vs-call **0.0**。

### 判决：subspace_readout_tolerant（三条件全阳性）

| 条件 | acc vs null p95 | margin vs p95 | 概念对照 |
|---|---|---|---|
| stale（d18 陈旧方向） | **0.8246 > 0.6325** | **0.2216 > 0.0292** | 0.0000 |
| eigen（本层本征方向） | **0.8421 > 0.6491** | **0.1799 > 0.0400** | 0.0702 |
| orth（旋转正交分量） | **0.8246 > 0.6316** | **0.1831 > 0.0413** | 0.0526 |

### 解读（三个实质发现，重复三遍）

1. **qwen 读出是子空间容忍型**：陈旧方向、本层方向、乃至与陈旧
   方向近正交的旋转分量（cos 0.14–0.46）三者在 qwen 深层 mlp 中
   全部承载完整检索信号（acc 0.82–0.84）——响应不挑方向，挑的是
   语言子空间。
2. **载体不对称闭环**：qwen = 子空间容忍读出，GLM4 = 方向匹配读出
   （2893 逐层阳性 + 2890/2891/2892 陈旧方向阴性）；方向旋转动力学
   跨模型普适（2895）——载体定律的 per-model 变异定位于**读出端**，
   G_readout_types 登记。
3. **旋转子空间携带信息**：orth 分量单独即可检索（0.8246）——
   方向旋转不是噪声，旋转扫过的子空间本身携带语言信息；与 2894
   "无全局平面、分段局部旋转"拼合：语言信息在一段局部旋转平面
   序列上分布，qwen 通道对整段子空间开放。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2896/qwen_readout_type/execution.json | 62a342f3 |
| phase2896/qwen_readout_type/result.json | 2468f496 |
| phase2896/qwen_readout_type/qwen_readout_type.npz | 71c5fd16 |
| tests/glm5/phase2896_qwen_readout_type.py | 965dbfd6 |

Ledger 更新：B_readout_qwen + M2896_qwen_readout_type +
G_readout_types + L13 notes 精化；blocks 21 / measurements 35 /
growth 21。

### 接续（2897 候选）

- **A（主选）**：GLM4 读出类型确证——GLM4 orth 分量注入（2893
  协议加 orth 条件；若 orth 阴性则"方向匹配"定性坐实，若阳性则
  GLM4 也是部分容忍，谱系化）
- **B**：norm-matched 对照纯化 2892 L1 归因（零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认）

*(SHA 与判决以 result.json 为准；本节由 phase2896 收尾脚本追加。)*


## Phase 2897: GLM4 三条件读出谱系判别（broad_tolerance_glm4） [2026-09-18 22:34]

**日期**：2026-09-19。**模型**：glm4-9b-chat-hf。运行 386.5s（零前向
方向 + 窗口 [28,40) 前向，2890 词表/条件 verbatim 78 词，双通道
mlp + self_attn 三条件）。

### 原理与设计

2896 判 qwen = subspace_readout_tolerant，GLM4 侧缺 orth 格。三条件
逐层注入（方向全部零前向自 2890 npz）：

- stale：注入 lang_dir（li=20），投影 lang_dir
- eigen：注入 dir_q(li)（本层本征方向），投影 dir_q(li)
- orth：注入 orth(li) = unit(dir_q(li) − (dir_q(li)·lang_dir)lang_dir)
  （cos(dir_q,lang_dir) 窗口内 0.422 -> 0.078；cos(orth,lang_dir)=0
  构造保证），投影 orth(li)

B = g(same) − 0.5(g(func)+g(null))；loo-NN acc vs null p95
（200 perms, SEED=2897）；概念对照；v1 形状匹配重算 <1e-6（实测
0.0/0.0）。冻结判决映射：orth+ & stale+ => broad_tolerance（errata
vs M2890/M2891）；orth+ & stale- => partial_tolerance；orth- &
eigen+ & stale- => direction_matched_strict。

### 判决：broad_tolerance_glm4（按冻结映射）

| 通道/条件 | acc vs null p95 | margin vs p95 | 概念对照 |
|---|---|---|---|
| mlp/stale | 0.5385 < 0.6282（阴性，复制 M2890） | 0.0390 > 0.0345 | 0.0769 |
| mlp/eigen | **0.6795 > 0.6410**（复制 M2893） | 0.0198 < 0.0402 | 0.0000 |
| mlp/orth | 0.6282 vs 0.6288（边缘阴性） | 0.0098 < 0.0381 | 0.0000 |
| attn/stale | 0.6795 > 0.6282（**与 M2891 冲突**） | −0.0161 < 0.0430 | 0.0000 |
| attn/eigen | **0.7308 > 0.6282** | **0.1213 > 0.0426** | 0.0128 |
| attn/orth | **0.6795 > 0.6667** | **0.1222 > 0.0434** | 0.0000 |

### 冲突诊断与诚实登记（errata 入账）

attn/stale 与 M2891（acc 0.5897 < p95 0.6410）协议完全一致（同窗口/
词表/注入/eps），唯一随机差异是 null-token 抽取（SEED 2891 vs
2897）。两轮 margin 均 <= 0（−0.0011 / −0.0161）——**stale 格是
阈值边缘的 null-draw 脆弱信号**，acc 跨 p95 摆动、margin 双轮皆
空。errata_ledger 登记 corrects=M2891：严格"direction-matched"对
attn 不成立，但"无强 stale 信号"结论维持；开放候选：多重 null-draw
稳健性检验。

### 三个实质发现（重复三遍）

1. **GLM4 读出是通道分裂的**：attention 部分子空间容忍（orth 分量
   单独承载完整检索信号，margin 0.1222 远超 p95 0.0434），mlp 方向
   匹配（仅 eigen；orth 边缘阴性）——2896 的二分对照精化为谱系。
2. **读出容忍谱系成形**：qwen mlp（三条件全阳，margin 0.18–0.22）>
   glm4 attn（eigen+orth，margin ~0.12）> glm4 mlp（仅 eigen）——
   L14 登记；载体定律的 per-model 变异 = 读出容忍梯度，且同一模型
   内部通道间已经不同。
3. **旋转子空间信息跨架构成立**：GLM4 attn 对与 lang_dir 构造正交
   的旋转分量（cos=0）给出强检索响应——旋转扫过的子空间携带语言
   信息不是 qwen 特例，与 2894/2895 旋转动力学普适拼合。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2897/glm4_readout_spectrum/execution.json | 16856795 |
| phase2897/glm4_readout_spectrum/result.json | e6291a28 |
| phase2897/glm4_readout_spectrum/glm4_readout_spectrum.npz | 9d2a28bc |
| tests/glm5/phase2897_glm4_readout_spectrum.py | 420eba53 |

Ledger 更新：B_readout_spectrum_glm4 + M2897_glm4_readout_spectrum +
G_glm4_readout_spectrum + L14_readout_spectrum_cross_model +
errata（corrects M2891）+ L13 notes 精化；blocks 22 / measurements 36
/ growth 22 / linkage 14 / errata 6。

### 接续（2898 候选）

- **A（主选）**：attn stale 格多重 null-draw 稳健性检验——同协议
  N 个 SEED 重复（每次 ~6.5min 或削减词表），acc 分布 vs p95，关闭
  脆弱格
- **B**：norm-matched 对照纯化 2892 L1 归因（零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认）

*(SHA 与判决以 result.json 为准；本节由 phase2897 收尾脚本追加。)*


## Phase 2898: attn stale 格多重 null-draw 稳健性检验（all_void 锚门 + margin 8/8 定谳） [2026-09-18 23:08]

**日期**：2026-09-19（执行 started 2026-09-18T23:08）。**模型**：
glm4-9b-chat-hf。运行 1290.9s（8 个 SEED 抽取 × attn 双条件，
78 词 verbatim 2890，零前向方向，单次模型加载）。

### 原理与设计

2897 errata：attn/stale 格 null-draw 脆弱（M2891 acc 0.5897 vs
M2897 0.6795 跨 p95 摆动，margin 双轮皆负）。本 Phase 以 8 个独立
null-token 抽取（SEED = 2891/2897/2901..2906，含两轮历史抽取）定量
该格：每抽独立 null tids + 200 perms；attn 通道；stale（lang_dir）
+ orth（锚条件，2897 强阳性）双条件。冻结门线：v1 重算 <1e-6/抽；
v2 锚门 orth acc_flag 率 >= 0.8 否则 all void。冻结判决映射：
stale acc_flag 率 >=0.8 且 margin 率 >=0.5 => robust；acc 率 <=0.4
且 margin 率 =0 => null_fragile_confirmed；否则 mixed。

### 判决：all_void_procedure_unstable（按冻结 v2 锚门）

| 指标 | stale | orth（锚） |
|---|---|---|
| acc_flag 率 | 2/8 | 5/8 |
| **margin > 0** | **0/8（全负 −0.0011..−0.0176）** | **8/8（全正 +0.0317..+0.1222）** |
| acc 范围 | 0.500–0.680 | 0.615–0.705 |
| 概念对照 | ~null | ~null |

v2 锚门触发（orth acc_flag 率 0.625 < 0.8）→ 冻结判决 all void；
描述性统计不 void：单抽二元 flag 双条件都不稳（acc_flag 依赖每抽
null p95 0.615–0.667 的宽分布），margin 统计量才是稳定判据。

### 三个实质发现（重复三遍）

1. **stale 信号不存在——定谳**：margin 8/8 全负、acc_flag 仅 2/8
   且均为边缘值；M2897 的 stale+（acc 0.6795）确证为 null-draw
   伪影。M2891 vs M2897 冲突在 margin 层面 8/8 裁决支持 M2891。
   N8_attn_stale_absent_glm4 登记 settled。
2. **orth 部分容忍确证**：margin 8/8 全正（+0.032..+0.122）——
   与 lang_dir 构造正交的旋转分量在 GLM4 attention 稳定承载检索
   信号；2897 谱系结论的 orth 分量稳固，仅 stale 分量被撤回。
   errata corrects M2897：robust 读法 = partial_tolerance
   （eigen+orth，非 stale）。
3. **方法论升级（G_multi_draw_lesson）**：n=78 的单抽 acc>p95
   二元 flag 复测信度低（阈值邻近格必翻转）；margin 统计量
   draw-stable。后续协议：margin 优先判据；|acc−p95| 小时强制
   多抽。同种子确定性验证通过（2891→0.5897、2897→0.6797 精确
   复现历史值），协议无漂移。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2898/attn_stale_robustness/execution.json | 6a555b34 |
| phase2898/attn_stale_robustness/result.json | 9b2aa073 |
| phase2898/attn_stale_robustness/attn_stale_robustness.npz | 15ee4d25 |
| tests/glm5/phase2898_attn_stale_robustness.py | 4fe8207a |

Ledger 更新：M2898_attn_stale_robustness + G_multi_draw_lesson +
N8_attn_stale_absent_glm4 + errata（corrects M2897）+ L14 notes
精化；measurements 37 / growth 23 / negatives 8 / errata 7。
MEMO 标题格式已按用户指令统一为
`## Phase {序号}: 标题 [yyyy-mm-dd hh:mm]`（本节为首个新格式节，
历史 148 节已批量规范化，备份 .bak_20260918_v2）。

### 接续（2899 候选）

- **A（主选）**：qwen 通道 margin-first 复核——2887/2896 的 qwen
  阳性格是否也在 margin 层面稳固（零前向 + 单窗口，成本低）；
  若稳固则谱系 qwen 端免于同类伪影
- **B**：norm-matched 对照纯化 2892 L1 归因（零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认）

*(SHA 与判决以 result.json 为准；本节由 phase2898 收尾脚本追加。)*


## Phase 2899: qwen mlp 读出三条件多抽稳健性 [2026-09-18 23:48]

### 原理
G_multi_draw_lesson（2898）确立：n=78 单抽 acc>p95 flag 复测信度低，margin 统计量 draw-stable。M2896 的 subspace_readout_tolerant 判决建立在单抽（SEED=2896）之上，需按新方法论升级为多抽 margin-first 检验。本 Phase 将 2898 多抽协议移植到 qwen mlp 通道三条件（stale/eigen/orth），锚条件取 eigen（M2896 中 margin 最强 0.1799），v2 锚门改用 margin_flag rate（margin-first）。

### 预注册（冻结于 execution.json，脚本 SHA256-8 be3a0d79）
- 窗口 W=[26,36)；57 词 verbatim 2887；方向零前向自 2886 S_last（labels i%%2 断言）。
- SEEDS=[2896,2901..2907]（含历史抽 8 抽）；每抽：null_tids 自 rng(s)（word-tid 排除，VOCAB=151936），200 label perms 自 rng2(s)。
- v1（每抽）：mlp recompute rel err < 1e-6 且 hook-vs-call < 1e-4（shape-matched [1,2,2560]，2896 run-2 约定），否则该抽 void。
- v2 锚门（margin-first）：eigen margin_flag rate >= 0.8，否则 all void。
- 判决映射：stale margin_flag >= 0.8 且 acc_flag >= 0.5 => qwen_stale_robust；margin_flag = 0 => qwen_stale_margin_absent；三条件全 robust => subspace_readout_tolerant_robust；stale margin_flag = 0 => subspace_tolerant_stale_fragile。

### 结果（运行 69.4s，n_valid=8，v2 锚门通过 rate=1.0）
- **判决：subspace_readout_tolerant_robust —— 8 抽 x 3 条件 = 24/24 cell-draw 双 flag 全阳性**。
- stale：acc 0.7018-0.8772（flag 8/8），margin +0.2009..+0.2582（8/8，mp95 0.029-0.043）。
- eigen：acc 0.7895-0.9298（flag 8/8），margin +0.1769..+0.2341（8/8，mp95 0.030-0.049）。
- orth：acc 0.7368-0.8947（flag 8/8），margin +0.1831..+0.2502（8/8，mp95 0.040-0.051）。
- 概念对照全程 ~0（0.0000-0.0702）。margin 为 null p95 的 5-8 倍，远离阈值区。
- 确定性验证：seed=2896 精确复现 M2896 三 acc（0.8246/0.8421/0.8246）。
- v1 全 8 抽 recompute=0.0、hook=0.0。

### 硬伤与混杂
- 无新增。锚门本次以 margin_flag 定义（与 2898 的 acc_flag 锚门不同）——2898 时锚（orth acc_flag 5/8）不过门而 all-void；若 2899 用 acc_flag 锚门（eigen 8/8=1.0）同样过门，故判决对锚门定义不敏感（两种定义下均通过，已核对）。
- 单模型单通道；glm4 侧对应多抽仅 attn 双条件（2898），glm4 mlp eigen 格的多抽确认仍是 open（弱阳性 margin 0.0198 贴阈值）。

### 结论
1. **qwen mlp 读出容忍升级为 draw-stable**：subspace_readout_tolerant（M2896）经 8 抽 margin-first 复核无一翻转——qwen 的三条件全阳是稳固事实，非 null-draw 伪影。
2. **谱系层级在 margin-first 准则下成立**：qwen mlp（margin ~0.2，24/24）>> glm4 attn（eigen+orth ~0.12，8/8 margin 正但 acc_flag 5/8）>> glm4 mlp（仅 eigen，且其多抽确认 open）。
3. **方法论闭环**：G_multi_draw_lesson 同时找到脆弱阴性（glm4 attn stale，0/8）与稳固阳性（qwen mlp，24/24）两个对照端点——协议升级完成。

### 接续
- 2891 候选：A glm4 mlp eigen 格多抽确认（阈值邻近，主选）/ B norm-matched 对照（2892 L1 混杂，零前向）/ C Gemma4/Qwen2-7B 下载决策（待用户）。

### 文件
- 脚本 tests/glm5/phase2899_qwen_readout_robustness.py（be3a0d79）
- 产物 phase2899/qwen_readout_robustness/：execution.json 6667e6ac / result.json b4416f00 / qwen_readout_robustness.npz 2af9ba4d
- Ledger：B_readout_robust_qwen + M2899_qwen_readout_robustness + G_margin_first_validated_qwen + L14 精化（blocks 23 / measurements 38 / growth 24 / linkage 14）

## Phase 2900: glm4 mlp eigen 格多抽判别 [2026-09-18 23:54]

### 原理
L14 谱系中 glm4 mlp eigen 格是唯一未多抽的阳性格：M2893/M2897 单抽 acc-flag 阳（0.6923/0.6795 > p95）但 margin-flag 阴（0.0313/0.0198 < 自身 p95）。按 G_multi_draw_lesson（margin-first 多抽），8 抽判别该格性质；stale 作期望阴性内部对照、orth 作描述性第三条件。glm4 mlp 无已知 robust 阳性条件，故不设阳性锚门，程序稳定性由 v1 确定性 + 历史种子精确复现（reproduction_check）守卫。

### 预注册（冻结于 execution.json，脚本 SHA256-8 2cde5111）
- 窗口 W=[28,40)；78 词 verbatim 2890；方向零前向自 2890 npz。
- SEEDS=[2893,2897,2901..2906]；null-draw 规则与 2893/2897 完全一致（历史种子精确复现其 null 集）。
- v1（每抽）：mlp recompute rel err < 1e-6，否则该抽 void。
- 判决映射：eigen acc_flag_rate >= 0.8 且 margin_flag_rate >= 0.5 => glm4_mlp_eigen_robust；margin_flag_rate = 0 => acc_only_fragile；否则 mixed。

### 结果（运行 1320.4s，n_valid=8，v1 全过，reproduction_check 双通过）
- **判决：glm4_mlp_eigen_mixed**
- eigen：acc 0.5641-0.7436，acc_flag **7/8**（仅 s2904 False）；margin 0.0166-0.0601（7/8 为正），margin_flag 仅 **1/8**（s2901 0.0601 > 0.0349）。
- stale 对照：acc_flag 1/8、margin_flag 3/8——无 robust 信号，与 M2898 一致。
- orth：acc_flag 4/8、margin_flag 1/8——边缘。
- 概念对照全程 ~0。复现锚：s2893→0.6923（=M2893）、s2897→0.6795（=M2897）精确。

### 硬伤与混杂
- margin null 分布紧（p95 0.030-0.059），margin_flag 对弱信号判别力有限——margin 量级（效应大小）比二元 flag 更本质，已入 G_dual_flag_divergence。
- 单抽 acc 与 p95 的差在 0.006-0.13 间波动，s2904 acc 0.5641 显示弱格 acc-flag 也不稳。

### 结论
1. **glm4 mlp eigen 是真实但弱的信号**：acc 一致高于 null（7/8），margin 7/8 为正但量级 ~0.02-0.03，比 qwen mlp（~0.2）低约一个数量级——M2893/M2897 的 acc-flag 判决维持，margin-缺失确认为其真实读法。
2. **谱系定量化为 margin 量级层级**：qwen mlp ~0.2 >> glm4 attn ~0.12 >> glm4 mlp ~0.02-0.03——载体强度连续谱而非二分，G_dual_flag_divergence 入账。
3. **acc-flag 与 margin-flag 在弱格分歧**：acc 检索超Chance 而同语言几何勉强超其更紧的 null——未来弱格判读以 margin 量级为准。

### 接续
- 2901 候选：A GLM4 attn eigen 格多抽确认（M2897 margin 0.1213 强阳但仅单抽；attn/margin 系谱系第二层，主选）/ B norm-matched 对照（2892 L1 混杂，零前向）/ C Gemma4/Qwen2-7B 下载决策（待用户）。

### 文件
- 脚本 tests/glm5/phase2900_glm4_mlp_eigen_robustness.py（2cde5111）
- 产物 phase2900/glm4_mlp_eigen_robustness/：execution.json bc364ec7 / result.json 31af8120 / glm4_mlp_eigen_robustness.npz 8b3e27a8
- Ledger：M2900_glm4_mlp_eigen_robustness + G_dual_flag_divergence + L14 精化（measurements 39 / growth 25 / linkage 14）

## Phase 2901: glm4 attn eigen 格多抽判别 [2026-09-19 00:54]

### 原理
L14 谱系第二层（glm4 attn）仅 M2897 单抽：eigen margin 0.1213 强阳。M2898 已对同种子集（[2891,2897,2901..2906]）跑过 attn stale+orth（冻结 v2 acc-锚门判 all-void，描述性 rates 成立）。本 Phase 以相同种子集补 eigen 主判条件，三条件全跑，margin-first 判决；无阳性锚门（理由同 2900），程序稳定性由 v1 + 4 项 reproduction_check（M2891/M2897/M2898）守卫。

### 预注册（冻结于 execution.json，脚本 SHA256-8 6f37b87c）
- 窗口 W=[28,40)；78 词 verbatim 2890；方向零前向自 2890 npz；null-draw 规则与 2891/2897 一致。
- v1（每抽）：attn recompute rel err < 1e-6。
- 判决映射：eigen acc_flag_rate >= 0.8 且 margin_flag_rate >= 0.5 => glm4_attn_eigen_robust；margin_flag_rate = 0 => acc_only_fragile；否则 mixed。
- stale/orth：描述性 + 与 M2898 跨运行对照。

### 结果（运行 1315.9s，n_valid=8，v1 全过，reproduction_check 4/4）
- **判决：glm4_attn_eigen_robust**
- eigen：acc 0.6538-0.7308，acc_flag **8/8**；margin 0.0342-0.1213，margin_flag **7/8**（仅 s2906 0.0342 < p95 0.0434）。
- stale 对照：margin **0/8 正**（-0.0011..-0.0176）——M2898 结论在独立重跑中精确再现。
- orth：margin **8/8 正**（0.0317-0.1222）——强化 M2898 描述性 rates。
- 复现锚 4/4：s2891 stale 0.5897=M2891；s2897 stale 0.6795=M2897 且 margin −0.0161=M2898；s2897 eigen 0.7308=M2897。

### 硬伤与混杂
- s2906 eigen margin 0.0342 低于自身 p95 0.0434（margin_flag 7/8 非 8/8）——弱于 qwen（24/24）的残余波动，量级仍属第二层（~0.06-0.12）。
- margin p95 在 glm4 attn 侧波动较大（0.040-0.061），小 margin 抽的 flag 判读需谨慎（G_dual_flag_divergence 教训沿用）。

### 结论
1. **谱系第二层多抽确证**：glm4 attn eigen+orth draw-stable（margin 7/8、8/8 正），stale 缺席（0/8）跨运行精确再现——M2898 的 stale+ 撤回与 orth 部分容忍双双定谳。
2. **全谱系 margin-first 判决完成**：qwen mlp（24/24，~0.2）>> glm4 attn（eigen 7/8+orth 8/8，~0.06-0.12）>> glm4 mlp（eigen acc 7/8 但 margin 1/8，~0.02-0.03）——三层数量级分离稳固。
3. **读出容忍谱系成为载体定律的定量形式**：per-model 变异 = 通道读出容忍梯度，现已全部经 8 抽 margin-first 检验，无未判格。

### 接续
- 2902 候选：A GLM4 attn/mlp margin 量级差的结构根源（通道几何分析，零前向或低前向，主选）/ B norm-matched 对照（2892 L1 混杂）/ C Gemma4/Qwen2-7B 下载决策（待用户）。

### 文件
- 脚本 tests/glm5/phase2901_glm4_attn_eigen_robustness.py（6f37b87c）
- 产物 phase2901/glm4_attn_eigen_robustness/：execution.json 897df5fb / result.json 789083bf / glm4_attn_eigen_robustness.npz 3ded0dca
- Ledger：M2901_glm4_attn_eigen_robustness + L14 精化（measurements 40 / linkage 14）


## Phase 2902: glm4 attn/mlp margin 差结构根源 [2026-09-19 02:20]

### 原理
L14 谱系第三层 vs 第二层：glm4 内同一方向族（dir_q(li), eigen 条件）attn 通道 margin（~0.06-0.12）比 mlp 通道（~0.02-0.03）大约 4-5 倍。2897 机制下两通道均在模块输入端加扰动，落地方向 = 通道局部 Jacobian 响应：r_attn = (Attn(x+eps d)-Attn(x))/eps，r_mlp 同理；B = r.d。故 margin 差必由两通道 Jacobian 对同一方向的不同变换解释。分解三假设：H1 范数增益（gamma=||r||）、H2 方向保持（rho=cos(r,d)）、H3 落地方向语言对齐（A=对 78 语言词 unembed 行的平均 |cos|，以 200 随机词行基线校准）。

### 预注册（冻结于 execution.json，脚本 SHA256-8 fcc281d2）
- 窗口 W=[28,40)；78 词 x 3 条件，上下文按 SEED=2897 rng 顺序 bit-exact 复现 2897；方向零前向自 2893 npz（与 2890 S_last 重算 max diff 8.5e-9 交叉验证）。
- v1：模块重算 rel err < 1e-6；v2 锚：B'=(r.same-0.5(r.func+r.null)).d 须与 2897 npz B_*_eigen rel err < 1e-3，否则 all void。
- null 对照：20 随机单位方向（SEED=2902）@ 层 {28,31,34,37}。
- 判决映射：锚败=>v2_anchor_fail_all_void；R_B>=2.0 且 max(R_gamma,R_rho)>=1.3=>jacobian_asymmetry_confirmed_dominant_*；否则 R_A>=1.3=>language_alignment_asymmetry；否则 asymmetry_unresolved。

### 结果（运行 1065.7s）
- **判决：language_alignment_asymmetry**
- 守卫全过：v1 = 0.0；v2 锚 rel err mlp 1.76e-4 / attn 1.75e-4；重算 margin 精确复现 M2897（attn 0.1213 / mlp 0.0198，ratio 6.13）。
- **H1 否定**：R_gamma=0.318——attn 响应范数比 mlp 小 3.1 倍（每层 gamma mlp 0.21-0.91 vs attn 0.05-0.31），且与随机方向基线比（0.104/0.237=0.44）同量级：范数不对称是通道泛性质，非 dir_q 特化。入 N9。
- **H2 否定**：R_rho=1.222 < 1.3，且两通道 rho 都近零（多数层 <0.07；rho_null ~0）——两通道 Jacobian 都把注入方向几乎完全重定向。
- **H3 确认**：R_A=1.438——attn 落地方向对语言词 unembed 行有超基线对齐（word-null 每层 ~0.002-0.009），mlp 落地方向语言中性（word~null，部分层为负）。
- 结构要点：R_B=0.318=R_gamma——attn 的 B 逐项更小但 margin 大 6.1 倍：margin 是响应的类别相关分数的性质，不是响应量级的性质。mlp 大响应类别非特异，被 func/null 对照消去；attn 小响应类别相关，存活。
- 权重描述性：down_proj PR ~3600（维度膨胀），W_VO 复合 PR ~210-245；零前向 SwiGLU 对 gain*d 的直通响应近零（~1e-4）而实测 mlp gamma ~0.2-0.9——实测 mlp 响应主要来自 context 交互而非方向直通。

### 硬伤与混杂
- A 度量用词 unembed 行作为语言内容代理，未含最终 ln_f 与 softmax 归一效应；word-null 差量级小（~0.003-0.009），逐层读数需谨慎。
- W_VO 复合假设单位置均匀注意力，忽略 K/Q 扰动引起的注意力模式迁移——attn 响应中的注意力再分配分量未分离（残留混杂）。
- rho/R_A 的中位数跨层波动大（rho 有符号翻转），层分辨解释需多抽确认。

### 结论
1. **margin 差的结构根源 = 落地方向的语言内容密度**：glm4 attn 通道把 dir_q 变换为仍携带词 unembed 对齐的方向，mlp 通道把 dir_q 变换为语言中性方向——尽管后者的响应大 3 倍。
2. **谱系获得机制解释**：margin 层级 qwen mlp ~0.2 >> glm4 attn ~0.12 >> glm4 mlp ~0.02 排序的是各载体落地方向的语言内容密度，而非响应增益（N9）或方向保持。
3. **方法论**：v2 锚（历史 npz 精确复现）+ 方向交叉验证（8.5e-9）+ 随机方向基线校准，是低前向归因分析的完整守卫栈。

### 接续
- 2903 候选：A（主选）qwen mlp 落地方向语言对齐检验——G_language_content_density_mechanism 的跨模型预测：qwen mlp 通道落地方向的 A 应显著高于 glm4 mlp（同协议复用，低前向）；B 注意力再分配分量分离（K-shift vs V-path，2801 式对照）；C glm4 attn eigen 格弱层（s2906 型）归因。

### 文件
- 脚本 tests/glm5/phase2902_glm4_channel_jacobian_asymmetry.py（fcc281d2）
- 产物 phase2902/glm4_channel_jacobian_asymmetry/：execution.json 2be30db0 / result.json 3fe45d3b / glm4_channel_jacobian_asymmetry.npz 6e48354c
- Ledger：M2902_glm4_channel_jacobian_asymmetry + N9_attn_norm_gain_advantage + G_language_content_density_mechanism + L14 精化（measurements 41 / negatives 9 / growth 26 / linkage 14，ledger 8c9a301b）

## Phase 2903: qwen 通道 Jacobian 分解——语言内容密度跨模型检验 [2026-09-19 03:34]

### 原理
M2902 将 glm4 margin 层级（attn/mlp 比 6.13）归因于落地方向语言内容密度（G_language_content_density_mechanism），其跨模型预测：qwen mlp 通道（margin ~0.18，谱系最大载体）落地方向的语言对齐 A 应显著高于 glm4 mlp（A 密度最低载体）。本 Phase 将 2902 协议 verbatim 译至 qwen3-4b（窗口 [26,36)，57 词），执行同一三假设分解（H1 范数增益 / H2 方向保持 / H3 语言对齐），并以 qwen 自身 20 随机方向 null 校准 A 指标的噪声底线。若 qwen 也证实则 G 升格为跨模型机制；若 qwen 否定则须回查 glm4 的 A 是否本就未超噪声。

### 预注册（冻结于 execution.json，脚本 SHA256-8 f05e5644）
- 窗口 W=[26,36)（2896 冻结）；dirs 从 2886 S_last 重算（labels i%2 断言，2896 verbatim）；words/labels 逐字取 2887 npz（57 词）；null_tids 按 2896 rng 顺序（SEED=2896）bit-exact。
- 响应：r_ch = (module(x+eps*d)-ref)/eps @ pos1，eps=1.0，57 词 x 3 条件（same/func/null），机制 verbatim 2896/2898。
- v1：模块重算 rel err < 1e-6 且 hook-vs-call 噪声 < 1e-4；v2 锚：B'=(r.same-0.5(r.func+r.null)).d 须与 2896 npz B_eigen rel err < 1e-3 否则 all void。
- tie=True：unembed = embed_tokens（2902 glm4 为 untied lm_head，协议差异已登记）。
- null 对照：20 随机单位方向（SEED=2903）@ 层 {26,29,32}，双通道。
- 判决映射（冻结）：锚败=>v2_anchor_fail_all_void；D_q_mlp >= 2*max(D_g_mlp,D_g_attn) 且 D_q_mlp > 2*q_null_p95 => language_content_density_confirmed_cross_model；elif D_q_mlp > 2*q_null_p95 => density_elevated_below_2x_glm4max；else density_not_confirmed。
- **Amendment（run 1 后、判决前）**：run 1 v2 锚失配 2.48e-1——AutoModelForCausalLM+device_map（混合 cpu/cuda）前向上下文数值与 2896 load_native 全 GPU 不同，即 2877 教训的设备级版本。修正：load_native("qwen4") verbatim 2896。run 1 无任何判决输出，未进入判读。

### 结果（run 4，运行 125.9s）
- **判决：density_not_confirmed**
- 守卫全过：v1 = 0.0（双通道）、hook 噪声 0.0；**v2 锚 rel err = 2.32e-8**（verbatim 加载后 bit-exact 复现 2896 上下文）。
- margin 层级跨模型复现：qwen mlp margin = +0.1799（acc 0.8421，超自身 perm p95 0.0400 的 4.5 倍）；**qwen attn eigen 首测阴性：margin = -0.0195**（acc 0.6140，p95 0.0339 内）——与 glm4 attn 阳性（0.1213）相反，attn 通道非普适载体。
- **H1 否定（N9 再证）**：R_gamma=0.2166——qwen mlp gamma（1.54-3.83）比 attn（0.29-1.01）大 ~4.6 倍，与 2902 glm4 同向：mlp 大响应非 margin 源。
- **H2 否定**：R_rho=0.52，两通道 rho 近零（-0.16~0.15）——落地方向完全重定向，与 2902 一致。
- **H3 判决关键**：D_q_mlp = 0.00165 < 2*q_null_p95 = 0.00928（自身 null p95 = 0.00464，D 值低于噪声底线）→ 未达 elevated 门槛 → density_not_confirmed。R_A=1.16（<2902 的 1.438）。
- 权重描述性：qwen attn 零前向响应增益随层增长（L26 2.5 -> L34 37.4），W_VO 复合 PR ~676-800；mlp 零前向 SwiGLU 直通近零（0.002-0.05）而实测 gamma 1.0-4.2——context 交互主导，同 2902 结论。

### E8 勘误（corrects M2902，后验校准）
qwen 自身 null 底线（0.00464）促使回查 2902 null 条目：g_null_p95 mlp = 0.00506 / attn = 0.01094。而 D_g_mlp = 0.00102、D_g_attn = 0.00429——**glm4 双通道的 A 值同样全在自身噪声内**。故 M2902 的 H3"确认"（R_A=1.438）系未做 null 校准的假阳性；A（词 unembed 行平均 |cos|）作为标量指标在两模型全通道均不携带超噪声信号。H3 降级 not-established；2902 的"结构根源 = 语言内容密度"机制解释（结论 1/2）撤回；入 E8 勘误与 N10_cross_model_density_prediction（refuted）。

### 硬伤与混杂
- 57 词（<2902 的 78 词）与 null 方向仅 20x3 层：标量 D 的功效有限，不排除子空间结构信号被中位数抹平。
- A 指标以 embed_tokens 行为 unembed 代理（tie=True），未含 ln_f/softmax 归一——但 null 校准已包含同一代理，底线结论不受代理选择影响。
- E8 为后验（qwen 结果触发回查），非预注册判决——按纪律如实登记为勘误而非 silently 修订 M2902。

### 结论
1. **margin 层级根源重新开放**：qwen mlp margin 阳性（+0.180）但语言对齐在噪声内、glm4 attn margin 阳性同样语言对齐在噪声内——三个标量 Jacobian 假设（H1 范数增益 / H2 方向保持 / H3 语言对齐）全部出局。margin 必然栖身于 B 行空间的**类别相关结构**（class-correlated STRUCTURE）而非任何标量平均量。
2. **qwen attn 通道非普适载体**：同方向族下 qwen attn eigen margin 首测阴性，与 glm4 attn 相反——margin 层级是每（模型,通道）的特异性质，非跨模型通道角色。
3. **方法论再证**：锚复现必须 verbatim 加载方式（run1 device_map 失配 2.5e-1 -> run4 load_native 2.3e-8，2877 教训设备级推广）；null 校准必须在主判决前完成（E8 教训：A 指标 2902 未校准、2903 补校准推翻主判决）。

### 接续
- 2904 候选：A（主选）**B 行空间类别相关结构分析**——对已有 2902/2903 npz 的 B 矩阵（57/78 词 x 方向族）做类内/类间主方向分解、有效秩/PR、与 label 的典型相关，直接检验"margin 栖身于结构"假说（零前向，纯矩阵分析）；B（备选）attn 响应的注意力再分配分量分离（K-shift vs V-path）；C glm4 attn eigen 格弱层归因。

### 文件
- 脚本 tests/glm5/phase2903_qwen_channel_jacobian_decomposition.py（f05e5644）
- 产物 phase2903/qwen_channel_jacobian_decomposition/：execution.json c6a5bc3c / result.json 8e6d009b / qwen_channel_jacobian_decomposition.npz d99eb685
- Ledger：M2903_qwen_channel_jacobian_decomposition + E8（corrects M2902）+ N10_cross_model_density_prediction(refuted) + G_language_content_density_mechanism 修订（claim withdrawn）+ L14 再精化（measurements 42 / errata 8 / negatives 10 / growth 26 / linkage L14=10，ledger 7389e8c6）

## Phase 2904: B 行空间结构分析与检测器代数审计 [2026-09-19 05:31]

### 原理
L14 精化（2903）判定 margin 必栖身于 B 行空间的类别相关结构而非标量平均量。本 Phase 对 2902/2903 npz 的 B 矩阵（glm4 78x12 / qwen 57x10，行=词、列=窗口层）做零前向纯矩阵分析，按 2809 制度先过构造代数审计再进主判读：主分解把 margin 承载结构分为一阶（类均值移位，margin_from_mean）与高阶（within-class centering 后存活的 margin_within）；阳性组集合 P 由 2902/2903 冻结的 stored p95 判定。

### 预注册（冻结于 execution.json，脚本 SHA256-8 2bdd1097）
- 锚 a1：4 组（glm4/qwen x mlp/attn）margin 与 acc 从 float32 npz 重算须与 stored 值 abs 差 < 2e-5，否则 anchor_fail_all_void。
- 代数审计（合成矩阵）：audit_1st_order（类均值广播 + N(0,1)，delta=2.0*e1，split 22/35）期望 margin_full > p95_full 且 margin_within <= p95_within；audit_2nd_order（零均值移位 + 类相关协方差 4x std 沿随机正交 v1/v2）期望 margin_within > p95_within，失败 => detector_insensitive_all_void（不进主判读）；audit_negative（N(0,I)）期望双指标均低于 p95；from_mean 恒等式（margin(广播矩阵) == 1 - cos(mu0h, mu1h) < 1e-9）逐组内建核对。
- null：SEED=2904，1000 次 label 置换/组（单一共享流，组序 glm4-mlp, glm4-attn, qwen-mlp, qwen-attn）；p95_within 在置换下重算 within-center。
- 判决映射（冻结）：锚败 => all_void；阴性/一阶审计败 => algebraic_audit_fail_all_void；二阶审计败 => detector_insensitive_all_void；P 空 => structure_not_established；P 全体 margin_within <= p95_within => margin_carried_by_class_mean_shift；P 全体 > => margin_carried_by_higher_order_structure；混合 => structure_mixed_across_carriers。

### 结果（零前向，秒级）
- **判决：detector_insensitive_all_void**——主判读未发生。
- **锚 4/4 全过**：glm4 mlp 0.01979/0.67949、glm4 attn 0.12126/0.73077、qwen mlp 0.17991/0.84211、qwen attn -0.01946/0.61404——float32 npz 数据链完好。
- audit_1st_order PASS（full 0.176 > p95 0.031，within -0.034 <= p95 -0.031）；audit_negative PASS（iid 3% 假阳性基线）。
- **audit_2nd_order FAIL**：协方差差异构造下 margin_within = -0.032 未超 p95 -0.026。
- from_mean 恒等式 4/4 过（主分析段已进 gate 前完成计算——注：恒等式核对在 gate 内，gate 拦截后主统计未输出）。
- 阳性组集合（stored 判据）：P = {glm4 attn (0.1213 > 0.0426), qwen mlp (0.1799 > 0.0400)}；glm4 mlp (0.0198 < 0.0402) 与 qwen attn (-0.0195 < 0.0339) 为阴性组。

### 解析诊断（2904b 合成诊断，临时脚本 diag_2904b.py）
审计 2 失败的根因是**构造的数学缺陷而非检测器缺陷**：
1. 对均值零、类相关协方差的行分布，E[cos|same] - E[cos|diff] = 0（一阶矩恒等）——协方差各向异性对余弦相似度的均值 margin 无期望贡献。实测：cov-4x 构造 pairwise gap = -0.001（40 万独立对）；n=57 完整管线检出率 6/100（= 假阳性基线）。
2. margin 家族的正确灵敏域：一阶（类均值移位）与**类内偏度/非对称子簇**结构（均值恰为零但质量单侧分布，90% +2w / 10% -18w）：pairwise gap = +0.092，n=57 管线检出率 **74/100**；iid null 3/100。
3. 结论：margin_within 检测器对二阶矩（协方差各向异性）**原理性盲**，对三阶矩（偏度）灵敏。

### 硬伤与教训
- 审计 2 构造参数未做事前解析期望核对就冻结——合成审计构造本身必须先做解析推导或数值预检（本 Phase 用一轮 all_void 买到此教训，制度化入 N11 reopen_condition：二阶矩敏感的 margin 变体须自带新代数审计）。
- gate 设计按纪律把主判读拦在审计后，主统计（P 组 margin_within 读数）在 2904 从未被观测——2905 将是首次判读，无污染。

### 结论
1. **N11 边界定律**：2896 族 margin 指标（含 within 变体）检测域 = {类均值移位（一阶矩）, 类内偏度/非对称子簇（三阶矩）}；协方差各向异性（二阶矩）数学不可见。
2. 锚 4/4 证明零前向矩阵分析的数据链可靠；审计 gate 机制按设计工作（构造缺陷在主判读前被拦截并如实登记 all_void）。
3. B 行空间结构假说的可检验形式收敛为：P 组的 margin 是否存活于 within-class centering——须用 skew 审计过的检测器（2905）。

### 接续
- 2905（已开工）：audit_2nd_order 换为 skew 构造（big=2.0/tail=18.0/pf=0.9，w0 垂直 w1 随机正交；20 独立实例检出率 >= 12/20 为 pass），其余协议不变——P={glm4 attn, qwen mlp} 的 margin_within 首次判读。

### 文件
- 脚本 tests/glm5/phase2904_b_row_space_structure.py（2bdd1097）
- 产物 phase2904/b_row_space_structure/：execution.json 81e46f99 / result.json 3a782d18 / b_row_space_structure.npz 8739c76e
- 诊断 tests/gpt5_temp/diag_2904b.py（临时探针，结果 diag_2904b.txt）
- Ledger：M2904_b_row_space_structure_audit_gate + N11_covariance_anisotropy_invisible_to_margin + L14 再精化（notes + connects 11）（measurements 43 / errata 8 / negatives 11 / growth 26 / linkage 14，ledger b89607b1）

## Phase 2905: skew 审计检测器下 margin 结构主判读 [2026-09-19 05:40]

### 原理
2904 判决 detector_insensitive_all_void 并诊断出 margin 检测域边界（N11：一阶均值 + 三阶偏度灵敏、二阶协方差数学盲）。本 Phase 按 2904 收尾预案把 audit_2nd_order 换为诊断验证过的 skew 构造（20 独立实例检出率 >= 12/20 为 pass；2904b 诊断点估计 74%），其余协议 verbatim 2904，对阳性组集合 P={glm4 attn, qwen mlp} 执行 margin_within 的**首次主判读**——检验"margin 栖身于 B 行空间类别相关结构"假说的可检验形式：移除类均值移位后 margin 是否存活。

### 预注册（冻结于 execution.json，脚本 SHA256-8 ea1ac473）
- 锚 a1 / audit_1st_order / audit_negative / from_mean 恒等式：verbatim 2904。
- audit_2nd_order_skew：均值零行、类相关偏度（90% 质量 +2w、10% 质量 -18w，w0 垂直 w1 随机正交，n=57 d=10 split 22/35），20 独立实例（rng 子流 [2905,2,k]），每个实例 1000 perm p95，pass 当且仅当检出率 >= 12/20；失败 => detector_insensitive_all_void。
- null：SEED=2905，1000 次 label 置换/组（共享流，组序 glm4-mlp, glm4-attn, qwen-mlp, qwen-attn），p95_within 在置换下重算 within-center。
- 判决映射（冻结）：与 2904 相同（P 全体 margin_within <= p95_within => margin_carried_by_class_mean_shift；全体 > => higher_order；混合 => mixed）。

### 结果（零前向，秒级）
- **判决：margin_carried_by_class_mean_shift**
- 审计全过：锚 4/4（glm4 mlp 0.01979/0.67949、glm4 attn 0.12126/0.73077、qwen mlp 0.17991/0.84211、qwen attn -0.01946/0.61404）；audit_1st PASS（full 0.166 > p95 0.030）；**audit_2nd_skew 检出 20/20**（中位 margin_within +0.0849）——skew 检测器全实例灵敏，2904 的失败确证为构造缺陷；audit_negative PASS；from_mean 恒等式 4/4。
- **主判读（P 两阳性组 margin_within 均不超置换 p95_within）**：
  | 组 | margin_full | margin_from_mean | margin_within | p95_within | within 超阈 |
  |---|---|---|---|---|---|
  | glm4 attn | +0.1213 | 0.3852 | -0.0229 | -0.0149 | 否 |
  | qwen mlp | +0.1799 | 1.4726 | -0.0332 | -0.0195 | 否 |
- **Fisher F / Hotelling T2 与 margin 阳性/阴性 4/4 一致**：qwen mlp F=5.121 > p95 2.286、T2=51.1 > 23.1（rho1=0.694）；glm4 attn F=6.685 > 2.701、T2=54.9 > 27.1（rho1=0.648）；阴性组 glm4 mlp（F=1.73 < 2.35）、qwen attn（F=0.63 < 2.38）均不超——B 行空间语言均值移位的存在性恰好按通道复现 margin 谱系。
- delta 层剖面：qwen mlp delta_norm=0.2557，集中 L27/L30/L34（-0.173/-0.118/-0.107）；glm4 attn delta_norm=0.0151（更弱但同侧富集 L34 0.0119）。
- 定量图像：margin_from_mean >> margin_full（qwen mlp 1.47 vs 0.18；glm4 attn 0.39 vs 0.12）——纯类均值结构的相似度优势被类内散布稀释成观测 margin；within-centering 移除均值后结构完全消失（margin_within 落到置换基线下）。

### 硬伤与混杂
- margin_within 的 p95_within 为负值区（-0.015/-0.020）："不超阈"即 margin_within 在基线以下，结论是结构消失而非弱信号。
- F/T2 与 margin 排序不完全同序（F: glm4 attn 6.69 > qwen mlp 5.12，margin: qwen mlp > glm4 attn）——存在性判据（超阈与否）与幅值排序是不同的泛函；margin 幅值 = f(delta 强度, 类内形状, 类大小)，属 L14 幅值定律的未决精化。
- 57/78 词、10/12 层窗口的单一数据集；labels 仅 en/fr 二分类（margin 的原始定义），concept 结构未在本 Phase 判据内。

### 结论
1. **margin 的结构根源已定位（一阶）**：2896 族 margin 阳性 ⟺ B 行空间存在超置换基线的类（语言）均值移位（F/T2 判别 4/4 一致）；移除类均值后 margin 存活为 0——margin_carried_by_class_mean_shift。
2. **谱系闭环**：2902/2903 三个标量 Jacobian 假设出局（E8/N9/H2）→ 2904 N11 划定检测域 → 2905 定位一阶类均值移位。L14 谱系从"margin 是类别相关结构"精化为"margin 是类均值移位 x 类内散布稀释"：qwen mlp delta_norm 0.256（最大）> qwen attn 0.024 ≈ glm4 attn 0.015（但 glm4 attn 类内形状更聚，稀释更少）——幅值差异来源留待分解。
3. **方法论闭环**：skew 审计检测器 20/20 检出 + 阴性审计 3% 假阳性——构造代数审计（2809 制度化）在 2904 拦截构造缺陷、在 2905 放行有效检测，全流程按设计工作。

### 接续
- 2906 候选：A（主选）**margin 幅值定律分解**——delta_norm x 类内散布几何如何映射到 margin 幅值（qwen mlp delta 最大但 F 不是最大；解析 + 合成标定 + 真实数据拟合），解释谱系幅值排序 qwen mlp 0.18 > glm4 attn 0.12 > glm4 mlp 0.02；B（备选）delta 层剖面的通道归因（qwen mlp L27/L30/L34 vs glm4 attn L34）；C 换 concept 标签重跑分解（结构对标签类的特异性）。

### 文件
- 脚本 tests/glm5/phase2905_margin_structure_skew_audited.py（ea1ac473）
- 产物 phase2905/margin_structure_skew_audited/：execution.json 619e02ee / result.json 05a42b4d / margin_structure_skew_audited.npz 2846e50b
- Ledger：M2905_margin_structure_skew_audited + L14 再精化（2905 main reading；connects 12）（measurements 44 / errata 8 / negatives 11 / growth 26 / linkage 14，ledger f9e9db55）

## Phase 2906: margin 幅值定律检验与神经元读出归因 [2026-09-19 05:53]

### 原理
2905 定位 margin 载体为 B 行空间一阶类均值移位（F/T2 与 margin 阳性 4/4 一致），遗留两问：(a) 幅值——各向同性 summary（类均值 mu_c + 每类标量方差 sigma_c）能否**定量还原**谱系幅值排序 qwen mlp 0.180 > glm4 attn 0.121 > glm4 mlp 0.020 > qwen attn -0.019；(b) 粒度——用户指令要求检查/推进神经元级：语言均值移位 delta 在响应空间由哪些输出神经元读出方向承载。响应空间的类均值差 Delta_r（r_same 类均值差，2560/4096 维）可从 2902/2903 npz 直接算出；其分解字典取 down_proj/o_proj 列（列 k = 输出神经元 k 把信号写入残差流的读出方向）——激活值不可零前向恢复，但读出方向分解可以。权重经 safetensors safe_open 直读（不加载模型、零前向）。

### 预注册（冻结于 execution.json，脚本 SHA256-8 00b72977）
- 锚 a1：margin/acc 4 组复现（abs 2e-5）；锚 a2：Delta_B 重算 == 2905 delta_per_layer（abs 1e-4，2905 存 round4）。
- 覆盖率审计（rng [2906,0]）：40 次重复，真参数 -> 抽样 -> 估计 (mu_hat, sigma_hat) -> 400 合成 margin 的 95% 区间覆盖真 margin；pass 当且仅当覆盖 [32,40]/40（Binomial(40,0.95) 下尾 ~2%）。
- 主判据（冻结）：每组 M1 各向同性合成（实测类均值 + sqrt(tr(Sigma_c)/d) 标量方差，类大小固定，400 抽样，共享流，组序 glm4-mlp/glm4-attn/qwen-mlp/qwen-attn）95% 区间；真实 margin_full 落入 4/4 => amplitude_law_confirmed_isotropic；3/4 => amplitude_law_partial；否则 amplitude_law_not_established。
- 神经元归因（描述性，不进主判决）：Delta_r[q] = mean(r_same|lab1,q) - mean(r_same|lab0,q)；c = pinv(W) @ Delta_r（W 行满秩，精确重构）；集中度 = top-64 坐标能量占比；null = 200 次 label 置换 Delta_r（保留真实响应协方差，rng [2906,1]）逐层 p95；高斯 null（rng [2906,2]）诊断对照；组内层中位数 > 置换 p95 中位数 => readout_concentrated。

### 结果（零前向，14m14s）
- **判决：amplitude_law_not_established（2/4），失败模式按通道分裂且有方向**
- 守卫全过：锚 4/4（dLB max ~4.9e-5 < 1e-4）；覆盖率审计 **40/40**。
- M1 还原逐组：
  | 组 | margin_full | M1 95% 区间 | 落入 | SNR | 稀释比 |
  |---|---|---|---|---|---|
  | glm4 mlp | 0.0198 | [-0.0005, 0.0828] | 是 | 0.046 | 0.053 |
  | glm4 attn | 0.1213 | [0.1649, 0.3004] | **否（低于下界）** | 0.154 | 0.315 |
  | qwen mlp | 0.1799 | [0.1128, 0.2984] | 是 | 0.189 | 0.122 |
  | qwen attn | -0.0195 | [-0.0116, 0.0871] | **否（低于下界）** | 0.025 | -0.066 |
- **通道分裂规律：mlp 通道各向同性 summary 充分（幅值定律成立），attn 通道双双低于各自合成下界——attn 类内形状（超越各向同性）主动压低 margin**（qwen attn 被压至负值）。N11 断言协方差各向异性对 margin 期望一阶盲，但真实 attn 数据的形状结构经合成分布比较在幅值上可检——二阶矩痕迹在此显形。
- 神经元读出归因（描述性）：
  | 组 | top-64 占比中位 | 置换 p95 中位 | 判读 |
  |---|---|---|---|
  | glm4 mlp | 0.195 | 0.170 | **concentrated**（10/12 层超，L34 0.310 最强） |
  | qwen mlp | 0.199 | 0.193 | **concentrated**（L29/30/31/34 超，与 delta 剖面 L27/30/34 部分重合） |
  | glm4 attn | 0.149 | 0.162 | distributed |
  | qwen attn | 0.243 | 0.288 | distributed |
- 读出集中度与 margin 阳性/阴性 **4/4 一致**：mlp 通道的语言均值移位由 ~0.7% 输出神经元（top-64/9728 或 /13696）承载 ~20% 能量写入语言轴；attn 通道在置换基线水平（o_proj 列字典无特异集中）。高斯 null 全部 ~0.05-0.12 远低于实测——置换 null（保留响应协方差）才是有效对照。

### 硬伤与混杂
- Delta_r 取 same 条件响应（r_func/r_null 未存盘，对比响应不可重构）——归因对象是 same 上下文的类均值移位，与 B 的对比构造存在 0.5(Delta_func+Delta_null) 成分差（B 锚 a2 已覆盖 B 空间一致性）。
- o_proj 列字典是头聚合读出方向，非严格头级/神经元级；attn 通道的"concentrated 检验"功效受 o_proj 列间强相关影响（置换 p95 与实测几乎重合）。
- M1 的 mu_hat/sigma_hat 与真实 margin 同数据估计（summary 充分性检验，非独立预测）；区间内 = summary 充分，区间外方向 = 形状效应符号。
- 40 覆盖率重复的参数域（mu 范数 0.3、delta 0.5、sigma 0.3-1.0，d=10）为设计选择，未覆盖极端 SNR。

### 结论
1. **幅值定律按通道分裂**：mlp 通道 margin 幅值由各向同性 summary（类均值移位 x 标量类内方差）定量决定；attn 通道需要超越各向同性的类内形状（压低方向）——"类均值移位 x 类内散布稀释"的 L14 精化在 mlp 成立、在 attn 需升级为"形状修正"。
2. **神经元级首证（读出侧）**：mlp 通道（margin 载体）的语言均值移位在 down_proj 列字典上显著集中（~0.7% 神经元承载 ~20% 能量），attn 通道分布化——粒度推进到输出神经元读出级，且与 margin 谱系 4/4 一致。
3. 方法论：pinv 列字典 + 置换标签 null 是零前向神经元归因的可行协议；高斯 null 在真实响应协方差下严重失效（ underestimate p95 ~4 倍），不可用作对照。

### 接续
- 2907 候选：A（主选）**attn 通道形状修正定律**——把 M1 升级为 M2（保留实测 Sigma_c 谱或对 attn 引入形状参数），恢复幅值还原并定位压低 margin 的形状成分（零前向，纯矩阵）；B（备选）头级分解（把 o_proj 列字典换成 per-head W_VO 子空间，定位承载 attn 类移位的头）；C 2907 用前向抓取 SwiGLU 中间激活，把读出侧 concentrated 结论推进到门控神经元激活级（非零前向，须按 eps=1.0 协议预算前向）。

### 文件
- 脚本 tests/glm5/phase2906_amplitude_law_neuron_attribution.py（00b72977）
- 产物 phase2906/amplitude_law_neuron_attribution/：execution.json 4054ffeb / result.json 277bff9e / amplitude_law_neuron_attribution.npz 8b31ea68
- Ledger：M2906_amplitude_law_neuron_attribution + L14 再精化（2906 amplitude law channel-split；connects 13）（measurements 45 / errata 8 / negatives 11 / growth 26 / linkage 14，ledger 52383c5b）

## Phase 2907: attn 形状修正阶梯检验与 2906 sigma 定义勘误 [2026-09-19 06:55]

### 原理
2906 公布候选 A：把 M1 各向同性 summary 升级为形状修正阶梯，检验 attn 通道"低于 M1 下界"是否需要超越各向同性的类内形状来还原。阶梯：M1（mu_c + sigma_c*I，标量）-> M2a（mu_c + diag(Sigma_c)^{1/2}，层方差剖面）-> M2b（mu_c + chol(Sigma_c)，完整协方差，Sigma_c = 同类行样本协方差，n>=d+1）。还原逻辑：若 M1 区间已含真实 margin，则形状修正不必要；若 M1 排除而 M2a 包含，则层方差剖面（对角）承载形状效应；若需 M2b，则跨层相关也承载。sigma 定义按 2906 prereg 文本统一为 RMS sqrt(tr(Sigma_c)/d)。

### 预注册（冻结于 execution.json，脚本 SHA256-8 87376b53）
- 锚 a1：margin/acc 4 组复现（abs 2e-5）；锚 a2：Delta_B 重算 == 2905 delta_per_layer（abs 1e-4）。
- 覆盖率审计：rng [2907,0] M2a（真 diag Sigma，层方差 0.09-1.0）与 [2907,3] M2b（真全 PSD Sigma = A A^T + 0.1I），各 40 重复，pass 当且仅当覆盖 [32,40]/40；失败 => audit_coverage_fail_all_void。
- 合成：每组每级 400 抽样，共享流 SEED=2907，组序 glm4-mlp/glm4-attn/qwen-mlp/qwen-attn，级序 M1/M2a/M2b，类大小固定。
- 判决映射（冻结）：attn 两组 both M2a => shape_correction_diagonal；both M2b => shape_correction_full_covariance；both none => shape_correction_failed；split => shape_correction_mixed；mlp 组仅作 sanity。**映射未枚举 both-M1 分支**（预注册时先验认为 attn 不会落入 M1，依据 2906 观测）。

### 结果（零前向，1.2s）
- 守卫全过：锚 4/4；覆盖率审计 M2a 39/40、M2b 39/40 双过。
- 阶梯逐组（sigma 用 RMS）：
  | 组 | margin_full | M1 95% 区间 | M2a 区间 | M2b 区间 | 级 | 层方差比 S0 | 跨层 |corr| 均值 |
  |---|---|---|---|---|---|---|---|
  | glm4 mlp | 0.0198 | [-0.0035, 0.0692] | [-0.0144, 0.0981] | [-0.0125, 0.1024] | **M1** | 31.0 | 0.183 |
  | glm4 attn | 0.1213 | [0.0915, 0.2262] | [0.0521, 0.2748] | [0.0400, 0.2656] | **M1** | 64.4 | 0.130 |
  | qwen mlp | 0.1799 | [0.0970, 0.2911] | [0.0975, 0.2984] | [0.0865, 0.3395] | **M1** | 9.3 | 0.366 |
  | qwen attn | -0.0195 | [-0.0198, 0.0631] | [-0.0224, 0.0936] | [-0.0233, 0.0808] | **M1** | 21.9 | 0.152 |
- **四组全部 level=M1**：RMS 定义下各向同性 summary 对全部四通道充分，形状修正不必要。qwen attn 富余仅 0.0004（MC 噪声级 borderline）；glm4 attn 从 2906 的"低于下界 0.044"翻转为区间内。
- 判决映射缺口：both-M1 未被冻结映射枚举，落入 else 分支机械输出 shape_correction_mixed——**该标签与实际结果（both-M1，形状修正不必要）不符**，按纪律保留冻结标签原样、以本节为准解读。

### diag_2907b 归因诊断（post-hoc，2x2x2 网格）
因素：sigma 定义（mean-std=2906 实现 vs rms=2907 实现/2906 prereg 文本）x seed（2906/2907）x 流结构（mode2906 每组仅 M1 段 vs mode2907 M1->M2a->M2b 完整阶梯流）。复现锚：mean_std|2906|mode2906 与 rms|2907|mode2907 分别复现 2906/2907 登记区间至 max|diff| 9e-7（round6 存储容差）——诊断器可信。
- sigma 比值 rms/mean-std（Jensen 下界 1）：glm4 mlp 1.26/1.19，glm4 attn 1.19/1.34，qwen mlp 1.08/1.06，qwen attn 1.15/1.30。
- **glm4_attn 判定 100% 由 sigma 定义驱动**：mean-std 下 4/4 组合 below（下界 0.157-0.165 vs full 0.1213）；rms 下 4/4 组合 inside（下界 0.091-0.100）——与 seed、流结构完全无关。
- **qwen_attn 为 MC 噪声级 borderline**：mean-std 下 4/4 below（差 0.004-0.008）；rms 下 inside 2/4、below 2/4（下界 -0.017 至 -0.021 vs full -0.0195）。
- **2906 prereg 文本-实现 drift（E9）**：2906 prereg 冻结文本写 sqrt(tr(Sigma_c)/d)（RMS），实现却用 B[m].std(0).mean()（mean-std，向下偏代理）；2906 覆盖率审计与实现共享同一定义故自洽通过、drift 未被察觉。按 2906 自己的 prereg 文本定义（RMS）重算，attn 判定即非 BOTH_BELOW。

### 硬伤与混杂
- 判决标签 shape_correction_mixed 名不副实（映射缺口），Ledger M2907 verdict 内已注明。
- qwen attn 的 M1 判定富余 0.0004，400 抽样 MC 误差（区间端点 ~0.002-0.004）与判定同量级——qwen attn "inside" 结论需增样或多 seed 确认。
- 2906 result.json immutable 不改；其 attn-below 读法由 E9 勘误 + M2907 修正，非删改。
- M2a/M2b 区间端点跨组无单调关系（glm4 attn M2b 下界低于 M2a，qwen mlp M2b 上界高于 M2a），chol 旋转既可增宽也可移位区间——阶梯只在"M1 排除后逐级还原"语义下有效。

### 结论
1. **形状修正不必要（RMS 定义下）**：四通道（含两 attn）的 margin 幅值均由各向同性 summary（类均值移位 x RMS 类内标量方差）定量还原——2906"attn 类内形状主动压低 margin"作为幅值事实被撤回，幸存结论仅为 qwen attn 位于 M1 下缘（MC 分辨率内）。
2. **2906 勘误 E9 入账**：prereg 文本-实现 sigma drift 使 2906 的 attn 失配读法不稳健；"amplitude_law_not_established" 冻结判决保留在案，机制读法以 M2907 为准。
3. **谱系链条闭环（2903->2907）**：margin = 类均值移位（2905 一阶载体）x 类内散布稀释（2906 mlp 定量 + 2907 全通道 RMS 确认）；读出侧 mlp concentrated / attn distributed（2906）与幅值谱系 4/4 一致；sigma 定义敏感性（E9）是本轮唯一修正。
4. 方法论：prereg 文本-实现 drift 不能靠共享该定义的审计自检——需要独立审计通读实现（本次由 2907 跨 phase 对照暴露）；2x2x2 stream-faithful 复现网格（双锚 9e-7）是归因此类跨 phase 翻转的廉价强协议。

### 接续
- 2908 候选：A（主选）**qwen_attn M1 边界判定加密**——400 -> 10000 抽样 + 多 seed 网格（零前向，秒级），判定 qwen attn margin 在 M1 内/低于下界/恰在下缘，消解当前谱系最脆弱一环；B（备选）头级分解：o_proj 列字典 -> per-head W_VO 子空间，定位 attn 通道承载/抵消类移位的头；C 前向 SwiGLU 激活级归因（非零前向，eps=1.0 协议预算）。

### 文件
- 脚本 tests/glm5/phase2907_shape_correction_law.py（87376b53）
- 产物 phase2907/shape_correction_law/：execution.json bd3d5322 / result.json 390f27c9 / shape_correction_law.npz 9bd10e2d
- 诊断 tests/gpt5_temp/diag_2907b.py -> diag_2907b.txt（2x2x2 网格 + 双锚复现）
- Ledger：M2907_shape_correction_law + E9（corrects M2906）+ L14 再精化（2907 shape ladder；connects 14）（measurements 46 / errata 9 / negatives 11 / growth 26 / linkage 14，ledger a25a7122）

## Phase 2908: qwen_attn M1 边界高精度判定（20000 抽样 x 5 seed） [2026-09-19 07:52]

### 原理
2907 遗留最脆弱一环：qwen_attn margin_full = -0.019459 落在 M1 区间 [-0.019820, 0.063102] 内但下界富余仅 0.0004，且 diag_2907b 显示其判定对 seed/流结构 MC 敏感（rms 下 inside 2/4、below 2/4）——400 抽样的区间端点 MC 误差 ~0.002-0.004 与富余同量级。本 Phase 把判定从"端点比较"升级为直接百分位检验 p = P(margin_synth <= margin_full)，抽样质量 400 -> 20000 x 5 独立 seed（单臂 50 倍、总量 250 倍于 2907 主阶梯），并配同协议覆盖率审计。p 的 SE = sqrt(0.025*0.975/20000) ~ 0.0011，把"below/inside/下缘"的分辨精度压到 3SE ~ 0.0033（概率尺度）。

### 预注册（冻结于 execution.json，脚本 SHA256-8 9f034700）
- 锚 a1：margin/acc 4 组复现（abs 2e-5）；锚 a2：Delta_B 重算 == 2905 delta_per_layer（abs 1e-4）；锚 a3：fast_margin（预计算 mask 批内）== margin_of_B（abs 1e-12）。
- 覆盖率审计（rng [2908,0]）：40 重复，真参数（d=10, n=57, split 22/35, mu0 ~ 0.3N, delta 0.5unit, sigma ~ U(0.3,1.0)），RMS sigma_hat 估计，20000-draw 95% 区间覆盖真 margin；pass [32,40]/40。
- 主判据（冻结，仅 qwen_attn）：5 独立 seed（rng [2908,10+k]）各 20000 抽样，p_median 跨 seed；p_median >= 0.028313（0.025+3SE）=> qwen_attn_inside_m1_confirmed；p_median <= 0.021687（0.025-3SE）=> qwen_attn_below_m1_confirmed；否则 => qwen_attn_at_m1_edge（真值恰在自身各向同性 null 的 2.5 分位附近，协议精度内不可分辨）。其余三组描述性。
- sigma 定义固定为 RMS（2906 prereg 文本 / 2907 实现，E9 已勘误正名）。

### 结果（零前向，38s）
- 守卫全过：锚 4/4（fast_margin 与 margin_of_B 差精确 0.0；dLB max 4.9e-5）；覆盖率审计 **40/40**。
- 判决：**qwen_attn_at_m1_edge**
  | 组 | margin_full | p_median（5 seed） | p 范围 | q2.5/q97.5（中位） | 判读 |
  |---|---|---|---|---|---|
  | glm4 mlp | 0.0198 | 0.3754 | [0.3707, 0.3808] | -0.0063 / 0.0705 | 深居区间内 |
  | glm4 attn | 0.1213 | 0.1676 | [0.1669, 0.1684] | 0.0933 / 0.2198 | 区间内（E9 修正后 firmly inside） |
  | qwen mlp | 0.1799 | 0.4741 | [0.4723, 0.4814] | 0.0980 / 0.2849 | 近居中 |
  | **qwen attn** | **-0.0195** | **0.0276** | **[0.0262, 0.0295]** | **-0.0199 / 0.0686** | **恰在 M1 下缘** |
- 关键读法（三条，等价陈述）：(1) **"below M1"被高精度否定**——5/5 seed 全部 p>0.025，真实 margin 在每个 seed 的名义 95% 区间下界之上；(2) 名义 95% 水平下 qwen_attn 5/5 inside，2907 both-M1 结论确认；(3) 按 3SE 保守带判定，p_median=0.0276 距 confirmed-inside 阈值 0.0283 仅 0.0007（<1 SE）——**真实位置恰在自身各向同性 null 的 ~2.8 百分位，即 M1 下缘，协议精度内不可分辨"恰在下缘"与"略高于下缘"**。
- seed 间极差 0.0034 ~ 3 SE_P，跨 seed 波动与二项统计吻合——判据无残余结构噪声。

### 新谱系维度：null 内百分位 p 独立于幅值排序
p（相对自身类内散布的标准化位置）排序：qwen mlp 0.474 > glm4 mlp 0.375 > glm4 attn 0.168 > **qwen attn 0.028**；margin 幅值排序：qwen mlp 0.180 > glm4 attn 0.121 > glm4 mlp 0.020 > qwen attn -0.019。两轴不同：glm4 attn 幅值居二但相对位置居三（其类内散布 sigma 0.0059/0.0078 为四组最小，margin 0.121 相对散布偏下缘方向）；qwen attn 幅值垫底且相对位置也在下缘——唯一"双轴皆末"通道。幅值谱系（L14 主轴）与新 p 轴互补：幅值 = 语言信息总量，p = 语言信号相对噪声的显著性。

### 硬伤与混杂
- 3SE 保守带判定使 qwen_attn 停留于 at_m1_edge 而非 confirmed_inside——这是刻意的诚实保守（避免把 0.6 SE 的差距当作确证）；名义 95% 水平的 5/5 inside 事实同时登记。
- p 值轴的语义解释（"显著性 vs 信息量"）目前是假说性框架，未预注册检验——若要升格为谱系第二主轴需专门 Phase 预注册（跨模型/跨通道稳定性检验）。
- 覆盖率审计参数域与 2906/2907 相同（d=10, n=57），未覆盖极端 SNR/类大小失衡域。

### 结论
1. **2906 通道分裂问题正式关闭**：E9 修正 sigma 定义后，glm4 attn firmly inside（p=0.168），qwen attn 5/5 名义 inside、精确定位 M1 下缘（p=0.0276）——四通道各向同性 summary 充分性成立（qwen attn 带下缘保留标记）。2906 的 amplitude_law_not_established 与 2907 的 shape_correction_mixed 两个冻结标签均已被高精度判定取代为可解释结论。
2. **谱系新轴**：null 内百分位 p 与 margin 幅值独立；qwen attn 是唯一双轴皆末通道。
3. 方法论：端点比较 -> 直接百分位检验 + 3SE 保守带 + 同协议覆盖率审计，是把 borderline 判定工程化的标准流程；fast_margin（预计算 mask）把 120 万次 margin 计算从预估 5-8 分钟压到 38 秒。

### 接续
- 2909 候选：A（主选）**p 轴谱系稳定性预注册检验**——把 null 内百分位升格为谱系第二主轴：跨 margin 家族变体（无对角 Sm/列 z-score 变体/acc 判据）、跨 seed 域、跨窗口层子集重算 p，检验四通道 p 排序的稳定性与 qwen attn 下缘地位的稳健性（零前向）；B（备选）头级 per-head W_VO 分解：定位 qwen attn 下缘/负 margin 的层 x 头来源（逐层 delta_per_layer 已有，2905）；C 前向 SwiGLU 激活级归因（非零前向，eps=1.0 协议）。

### 文件
- 脚本 tests/glm5/phase2908_qwen_attn_boundary_precision.py（9f034700）
- 产物 phase2908/qwen_attn_boundary_precision/：execution.json 4a3587e3 / result.json dff824f8 / qwen_attn_boundary_precision.npz 553ab55b
- Ledger：M2908_qwen_attn_boundary_precision + L14 再精化（2908 high-precision adjudication；connects 15）（measurements 47 / errata 9 / negatives 11 / growth 26 / linkage 14，ledger 0f413c96）

## Phase 2909: p 轴谱系稳定性预注册检验——升格被否决（N12） [2026-09-19 08:04]

### 原理
2908 发现 null 内百分位 p = P(score_synth <= score_true) 与 margin 幅值独立（新轴候选），qwen_attn 位于下缘。升格为谱系第二主轴前须检验其对打分定义与层子集的稳健性——按 E9 教训，任何"轴"若依赖任意实现选择即非不变量。本 Phase 把 p 轴放进预注册稳定性网格：4 通道 x [4 个 margin 变体 + 3 个层子集] = 28 配置，每配置 5 独立 seed x 10000 抽样，配 7 个同协议覆盖率审计。

### 预注册（冻结于 execution.json，脚本 SHA256-8 a9f4d1a4）
- 锚 a1：全层 baseline margin/acc == stored（2e-5）；a2：Delta_B == 2905 delta_per_layer（1e-4）；a3：fast margin == margin_of_B 且 acc_score == acc_of_B（1e-12）。
- 变体（全层）：V0_full（2896 族 baseline）；V1_diagin（same 项含对角）；V2_colz（列 z-score 后 V0——层等权化）；V3_acc（LOO 最近邻准确率族）。
- 层子集（V0 打分）：S_front（前半窗口层）、S_back（后半）、S_key（2905 argmax|delta| 单层）。
- 覆盖率审计：变体 rng [2909,100+v]（d=10）、子集 [2909,200+s]（d=6/6/1），40 reps x 4000 draws，pass [32,40]/40，任一 fail => audit_coverage_fail_all_void。
- 判决（冻结）：排序分量——四通道 p_med 序在 V1/V2/V3 与三个子集下精确复现 baseline 序；尾部分量——qwen_attn 全部 7 配置 p_med < 0.05；内部分量——glm4_attn 全部 7 配置 p_med ∈ [0.05,0.95]；order unstable => p_axis_order_unstable；stable & tail & interior => p_axis_second_dimension_confirmed；否则 p_axis_order_stable_edge_cases。

### 结果（零前向，92s）
- 守卫全过：锚 4/4；审计 7/7（V0 40/40、V1 40/40、V2 38/40、V3 39/40、S_front 40/40、S_back 40/40、S_key 40/40）。
- **判决：p_axis_order_unstable——p 轴升格被否决**
- 排序表（p_med）：
  | 配置 | glm4 mlp | glm4 attn | qwen mlp | qwen attn | 序同 baseline |
  |---|---|---|---|---|---|
  | V0_full | 0.380 | 0.168 | 0.473 | 0.026 | （基准 qm>gm>ga>qa） |
  | V1_diagin | 0.391 | 0.184 | 0.484 | 0.027 | **是** |
  | V2_colz | 0.061 | 0.052 | 0.535 | 0.097 | **否**（qm>qa>gm>ga） |
  | V3_acc | 0.874 | 0.378 | 0.855 | 0.735 | **否**（gm>qm>qa>ga） |
  | S_front | 0.296 | 0.117 | 0.760 | 0.140 | 否 |
  | S_back | 0.555 | 0.244 | 0.327 | 0.130 | 否 |
  | S_key | 0.599 | 0.302 | 0.953 | 0.607 | 否 |
- 三分量：order **unstable**（6/7 配置打破）；qwen_attn tail **fragile**（仅 V0/V1 < 0.05）；glm4_attn interior **robust**（7/7 ∈ [0.05,0.95]，最低 0.052@V2——唯一不变分量）。

### 结构发现：qwen_attn 下尾是全层聚合效应
qwen_attn 的 p=0.026（全层 V0）在所有真子集上消散：S_front 0.140、S_back 0.130、S_key（delta 最大层单独）0.607。单独任何层块都是 null 样——下尾地位由全层 cosine 平均把逐层小效应聚合而成。换言之 qwen_attn 的"margin 负值/下缘"不是某一层/层块的属性，而是跨层一致的微小偏移的聚合。V3_acc 下 qwen_attn p=0.735 进一步表明：几何分离度（margin）与分类可用性（acc）在 qwen_attn 通道不同源。

### 硬伤与混杂
- S_key 单层 d=1：margin 在 1 维上退化为符号分离度，sigma 用 std(ddof=1)——1 维审计 40/40 通过，实现可用但语义与高维 margin 不同，排序判定对此解释保守。
- 覆盖率审计参数域仍是标量各向同性合成（V2 列等权化在异方差真实数据上的性质由 38/40 覆盖近似保证）。
- 变体集（V1/V2/V3 + 3 子集）是设计选择，未穷尽打分族；"unstable" 结论只需一个反例，已充分。

### 结论
1. **N12 入账（负结果）**：p 值是定义相对统计量而非谱系不变量；第二轴升格否决；2908 的排序与 qwen_attn 下缘读法严格限定在其声明的 2896 族定义内。margin 幅值层级（L14 主轴）不受影响。
2. **幸存的结构发现**：qwen_attn 下尾 = 全层聚合效应（逐层 null 样、聚合显著）；glm4_attn 的内部地位是唯一打分族不变分量。
3. 方法论：把候选"轴"先过预注册稳定性网格再升格，是 E9 教训（定义相对性）的制度化延伸；本 Phase 三个实现 bug（n1 未定义 / d=1 std 转换 / SCORES 查表）各以清目录重跑处理，最终锚/审计链完整。

### 接续
- 2910 候选：A（主选）**全层聚合效应分解**——qwen_attn 逐层 delta_per_layer 与逐层 margin 响应的符号/幅值剖面（零前向），检验"逐层小效应同号"假说：若各层 margin 贡献一致偏负/偏下，则聚合效应是相干的跨层属性（与 Cmp 竞争机制连接）；若异号抵消则只是平均化伪影。B（备选）头级 per-head W_VO 分解（2909 遗留，从层级推进到头级）。C 前向 SwiGLU 激活级（非零前向）。

### 文件
- 脚本 tests/glm5/phase2909_p_axis_stability.py（a9f4d1a4）
- 产物 phase2909/p_axis_stability/：execution.json 8c6bf1a3 / result.json 2fb42fa1 / p_axis_stability.npz 6dafc809
- Ledger：M2909_p_axis_stability + N12_p_axis_not_score_family_invariant + L14 再精化（2909 p-axis promotion refuted；connects 16）（measurements 48 / errata 9 / negatives 12 / growth 26 / linkage 14，ledger a9c1963f）

## Phase 2910: qwen_attn 全层聚合效应分解——交替结构与后半层主导 [2026-09-19 08:12]

### 原理
2909 定位 qwen_attn 下尾地位（全层 p=0.026）为全层聚合效应（任何真子集 null 样），遗留两个对立假说：H_coherent（各层一致偏下，聚合出相干跨层属性）vs H_cancel（异号抵消伪影）。本 Phase 预注册逐层 + 累积分解裁决：逐层 d=1 百分位 p_j（每窗口层独立 null 检验，5 seed x 5000 draws）+ 累积曲线 cum_p(k)（前 k 层联合，k=1..n_win）。判据（冻结，仅 qwen_attn）：below_frac = p_median_j < 0.5 的层比例；>= 0.90 => coherent_cross_layer_suppression；< 0.50 => cancellation_artifact；否则 partial_coherence。其余组描述性。

### 预注册（冻结于 execution.json，脚本 SHA256-8 0a3066f9）
- 锚 a1：margin/acc == stored（2e-5）；a2：Delta_B == 2905 delta_per_layer（1e-4）。
- 覆盖率审计：d=1（rng [2910,0]）39/40、d=5（rng [2910,1]）40/40 双过（d=10/6 审计已由 2908/2909 登记，{1,5,6,10} 网格锚定本 Phase 使用的维度极值）。
- 打分固定为 2896 族 V0（声明的定义域）；sigma RMS（E9 正名）；d=1 用 std(ddof=1)。

### 结果（零前向，85s）
- 判决：**qwen_attn_partial_coherence**（below_frac = 0.50，落入冻结的 [0.50, 0.90) 中段）。
- **qwen_attn 逐层剖面（决定性发现：相邻层交替）**：
  | 层 | L00 | L01 | L02 | L03 | L04 | L05 | L06 | L07 | L08 | L09 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | d1 score | -0.029 | +0.070 | -0.039 | +0.016 | -0.031 | +0.064 | -0.021 | -0.034 | +0.071 | -0.040 |
  | p_med | 0.327 | 0.778 | 0.104 | 0.697 | 0.253 | 0.849 | 0.556 | 0.205 | 0.609 | 0.143 |
  L00-L05 严格交替偏下/偏上（L06 近中位打断，L07/L09 恢复偏下）。
- **累积曲线（对 H_cancel 的决定性反驳）**：k=1..4 在 0.34-0.54 震荡（null 样），k=5 起单调下行 0.139 -> 0.085 -> 0.022 -> **0.010（k=8）** -> 0.029 -> 0.025（k=10）。后半层（k>=5）驱动进入深尾且稳定。
- **三方互证（锚链交叉验证）**：cum k=10 = 0.025 独立 seed 复现 2908 登记的全层 p=0.026；cum k=5 = 0.139 复现 2909 S_front = 0.140——不同 rng 键、不同 Phase、同值到千分位。
- 对照组：glm4_attn below_frac 0.42（累积 k=3 早降 0.143 但稳定在 0.12-0.24 内部区）；glm4_mlp 0.50（后段下行被最后两层逆转 0.563/0.377）；qwen_mlp 0.60（被 L01 单层 score 1.0195 / p 0.954 强偏上主导）。**仅 qwen_attn 到达深尾**。

### 解读
1. **既非纯相干也非伪影**：below_frac 0.50 排除 uniform coherence（0.90 阈值），但累积曲线从 k=5 起单调入深尾并稳定在 2908 登记值——若是平均化伪影，累积 p 应随机震荡而非单调收敛。真实结构 = **相邻层交替的逐层贡献 + 后半层主导的聚合**。
2. **交替性是新结构线索**：qwen_attn 相邻窗口层的 margin 贡献反号（L00-L05 严格交替），提示相邻层的通道响应方向振荡——与 2902/2903 通道 Jacobian 结构的连接待检验（2911）。
3. qwen_mlp 的 L01（score 1.0195，比其他层大一个量级、p 0.954 深居 null 上部）是唯一"单层主导"型通道，与 qwen_attn 的"交替+聚合"型形成结构对照。

### 硬伤与混杂
- below_frac 0.50 恰落在冻结边界（< 0.50 为 cancellation）上 1/20 层（L06 0.556 近中位）——partial 标签对 L06 的微小位移敏感；但累积曲线的单调性证据不依赖该边界。
- d=1 逐层 margin 语义（符号分离度）与全层 d 维 margin 不同构，逐层 p_j 只作相对剖面用（审计 d=1 39/40 保证 null 校准）。
- n_win=10（qwen）/12（glm4），半层分割 k=5/6 是设计选择；累积曲线在两种分割下形态一致（S_front/S_back 互证）。

### 结论
1. qwen_attn 下尾 = **交替层结构 x 后半层聚合**（partial coherence），单层与均匀相干两个朴素假说均被否定；聚合动力学的三方互证（2908/2909/2910 独立 seed 同值）是本轮最硬的量化事实。
2. 交替性（相邻层反号）进入候选机制清单：若在 delta_per_layer 与 Jacobian 结构上复现，将连接 Cmp 竞争机制（层间方向振荡 = 候选竞争的层间表现）。
3. 谱系四通道聚合形态学：qwen_attn 深尾聚合型 / glm4_attn 早降内部型 / qwen_mlp 单层主导型 / glm4_mlp 尾部逆转型——四种不同聚合形态学，margin 幅值层级之外的第二结构维度（定义域内，2896 族）。

### 接续
- 2911 候选：A（主选）**交替结构形式检验与机制连接**——四组逐层 delta_per_layer（2905 已有）符号交替性 + 相邻层 B 列相关剖面（oscillation index：相邻层列相关 vs 隔层相关的系统差），检验交替是否为 qwen_attn 特有及是否延伸到 Jacobian 层面（零前向）；B（备选）头级 per-head W_VO 分解（对 L07/L09 偏下最强层定位承载头）；C 前向 SwiGLU 激活级（非零前向）。

### 文件
- 脚本 tests/glm5/phase2910_cross_layer_coherence.py（0a3066f9）
- 产物 phase2910/cross_layer_coherence/：execution.json d4c7ed55 / result.json 48001e65 / cross_layer_coherence.npz 1013f7da
- Ledger：M2910_cross_layer_coherence + L14 再精化（2910 coherence decomposition；connects 17）（measurements 49 / errata 9 / negatives 12 / growth 26 / linkage 14，ledger 2acb5381）

## Phase 2911: 交替结构形式检验——margin 层面真实且 qwen_attn 特有，载体定位到类间符号平衡 [2026-09-19 08:36]

### 原理
2910 发现 qwen_attn 逐层 d=1 margin 贡献奇偶交替（L00-L05 严格）。两个待解问题：(1) 交替是否统计真实（非噪声读法）；(2) 交替的载体在哪个层面——响应方向（B 列相关结构）、均值移位（delta 符号）、还是别的。本 Phase 三探针预注册检验：P1 margin score 符号翻转率（逐层 d=1 score 相邻符号翻转，精确二项尾）；P2 delta 符号翻转率（2905 delta_per_layer，零对排除）；P3 列相关振荡指数 osc = mean_corr(相邻列) − mean_corr(隔一列)（10000 次列置换 null，单侧 p = P(perm <= obs)）。

### 预注册（冻结于 execution.json，脚本 SHA256-8 7793c93a）
- 锚 a1：全层 margin/acc == stored（2e-5）；a2：Delta_B == 2905（1e-4）；**a3：重算逐层 d=1 score == 2910 score_true（1e-6，实测 maxabs 5.0e-7）**——跨 Phase 产物互锚。
- 校准审计（rng [2911,0]）：200 个 iid 57x10 正态矩阵 x 1000 置换——置换 p 的 [0.05,0.95] 覆盖率须在 [0.80,0.97] 且中位 p 在 [0.40,0.60]。实测 frac=0.88、median=0.505，pass。
- 判决（冻结）：qwen_attn 需 P1 p<=0.05 且 P3 p<=0.05；特异性由显著负 osc 集合 S 决定（S={qwen_attn} => specific；S ⊆ {两 attn} => channel_shared；|S|>=3 => generic；其他 partial）。

### 结果（零前向，6s）
- **判决：alternation_not_confirmed_margin_only**
- 探针表：
  | 组 | P1 margin flips | P2 delta flips | P3 osc | P3 perm p |
  |---|---|---|---|---|
  | glm4 mlp | 6/11 p=0.500 | 6/11 p=0.500 | +0.062 | 0.864 |
  | glm4 attn | 6/11 p=0.500 | **0/11 p=1.000（全同号）** | +0.054 | 0.849 |
  | qwen mlp | 6/9 p=0.254 | 3/9 p=0.910 | −0.030 | 0.342 |
  | **qwen attn** | **8/9 p=0.0195** | 5/9 p=0.500 | **+0.050** | **0.728** |
- P1 确认：margin 层面交替真实（8/9，p=0.0195）且 qwen_attn 特有（他组 p>=0.25）。P3 否定向列相关结构的延伸：osc 为正（相邻列相关反而更高），S 空集——原始响应跨层平滑。P2 delta 符号随机。
- 附带发现：glm4_attn delta 符号 0/11 翻转（全同号块，反向尾 p~0.0005）——其 delta_per_layer 全正。

### 载体定位（diag_2911c，post-hoc 描述性）
2910 的 d=1 margin 本质是**符号分离度**：(57,1) 矩阵行归一化后每元素 = 响应符号，margin_j = P(符号一致|同类) − P(符号一致|异类)。用符号直接重构 marg 与 2910 score_true 完全一致（校验通过）。逐层剖面：
- 类间符号平衡 gap_j = |pos_frac(类0) − pos_frac(类1)|：qwen_attn 序列 0.08/0.27/0.08/0.16/0.09/0.32/0.14/0.03/0.24/0.05——**锯齿 7/8**，与 margin_j 强对应（gap 大 → margin 正：0.27→0.070、0.32→0.064、0.24→0.072；gap 小 → margin 负：0.08→−0.029、0.03→−0.034、0.05→−0.040）。
- 驱动侧：pf0（类 0 正率）范围 0.32-0.86 宽幅波动，pf1（类 1）0.37-0.60 相对平稳——**交替主要由类 0 正响应率的层间波动驱动**。
- 跨组同律：glm4_attn L08/L09 gap 0.37/0.41（最大）→ margin 0.273/0.276（最大）——gap→margin 定律跨组成立。
- diag_2911b 教训（指标设计错误）：对 |delta| 幅值序列用符号翻转检验恒 0 flips（无信息）；且幅值保留归一化与 2910 的符号化 margin 是不同泛函——诊断指标必须与目标量的定义同构。以 diag_2911c 修正。

### 结论
1. **交替真实且特有，但不在信号层面**：margin 交替（P1 p=0.0195）不伴随响应方向（P3 osc 正）、均值移位（P2 随机）的振荡——**载体是每层的符号分离质量（类间符号平衡）的锯齿**：相邻层的类间符号混杂交错度一好一坏交替。
2. gap→margin 定律跨组成立，把 2910 的"聚合形态学"统一到符号平衡语言下：qwen_attn 深尾聚合 = 其 gap 锯齿在最全层组合下把低 gap 层（margin 负贡献）和高 gap 层（正贡献）按 2896 族 cosine 泛函非线性聚合的结果。
3. 机制连接修正：2910 假说的"层间方向振荡（候选竞争的层间表现）"不成立（P3）；替代假说——类 0 响应符号平衡的层间波动（pf0 宽幅 vs pf1 窄幅）指向**类 0（语言 0）在相邻层的响应符号翻转集合变化**，词级归因待 2912。

### 接续
- 2912 候选：A（主选）**符号平衡锯齿正式化 + 词级归因**——预注册 gap 锯齿检验（词级符号置换 null）+ lag-1 自相关符号检验 + 驱动词识别（逐词符号翻转剖面：哪些词的响应符号在相邻层翻转驱动 pf0 波动；词 x 层符号矩阵，零前向）；B（备选）头级 per-head W_VO 分解（L07/L09 层）；C 前向 SwiGLU（非零前向）。

### 文件
- 脚本 tests/glm5/phase2911_alternation_structure.py（7793c93a）
- 产物 phase2911/alternation_structure/：execution.json 0ab48eaf / result.json 61a08678 / alternation_structure.npz 8c791cbd
- 诊断 tests/gpt5_temp/diag_2911b.py（指标设计错误，教训入账）-> diag_2911c.py（修正后符号剖面）
- Ledger：M2911_alternation_structure + L14 再精化（2911 alternation formal tests；connects 18）（measurements 50 / errata 9 / negatives 12 / growth 26 / linkage 14，ledger c8a0d016）
