"""Append-only closeout after real measurements and local-client verification."""
import hashlib,re
from datetime import datetime
import record_selection_core_v3 as R
from record_selection_core_v3 import C

def main():
    r=C.read(C.OUT/'client-report.json');a=C.read(C.OUT/'audit.json');d=C.read(C.OUT/'delivery-audit.json');reg=C.read(C.OUT/'registration.json')
    assert d['passed'] and a['passed'] and r['diagnostic_only'] and all(not e['scope_use_supported'] for e in r['effects'])
    memo=C.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';index=C.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    old=memo.read_bytes();oldindex=index.read_bytes();assert C.RUN.encode() not in old
    phase=max(map(int,re.findall(rb'^## Phase (\d+):',old,re.M)))+1;iphase=max(map(int,re.findall(rb'^## Phase (\d+):',oldindex,re.M)))+1
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    def fmt(v):return f'{v:.6g}' if v is not None else '未检验'
    def ci(v):return '['+', '.join(map(fmt,v['frame_interval']))+']' if v else '未检验'
    history='\n'.join(f"| {x['label']} | {x['n']} | {x['accuracy']:.6%} | {'通过' if x['passed'] else '失败'} |" for x in r['calibration_history'])
    effects='\n'.join(f"| {e['candidate']} | {e['method']} / {e['dose']} | {e['n']} | {fmt(e['mean_margin_loss'])} | {fmt(e['native_group_control_loss'])} | {fmt(e['matched_norm_control_loss'])} | {ci(e['excess_native'])} | {ci(e['excess_norm'])} | {e['first_correct_to_wrong']} / {e['first_wrong_to_correct']} |" for e in r['effects'])
    matched='\n'.join(f"- {e['candidate']}：等实际范数原生方向对照平均优势下降 {fmt(e['control_mean_loss'])}，候选超出对照的句式区间 {ci(e['excess'])}；对照缩放倍数最小/中位/最大 {list(map(fmt,e['scale_min_median_max']))}；实际范数最大相对误差 {e['maximum_norm_error']:.6%}。未缩放对照/候选实际范数之比最小/中位/最大 {list(map(fmt,e['unscaled_realized_norm_ratio_min_median_max']))}。" for e in r['matched_native_controls'])
    observations='\n'.join(f"- {o['id']}：均值 {fmt(o['mean'])}；句式区间 {ci(o['cluster_ci'])}；双侧家族符号置换p={fmt(o['two_sided_family_swap_p'])}；有效在前/在后的头质量差={o['head_positions']}。" for o in r['observations'])
    baselines='\n'.join(f'- {k}：MSE={fmt(v)}。' for k,v in r['baselines']['mse'].items())
    cells='；'.join(f"{k}: n={v['n']}, acc={v['strict_accuracy']:.4%}" for k,v in r['behavior']['cells'].items() if v['strict_accuracy']<.9)
    units=next(c for c in r['candidates'] if c['kind']=='units')['indices']
    entry=rf'''

## Phase {phase}: 有效与过时记录选择——独立行为校准、原生头和MLP组，以及受限作用诊断 [{stamp}]

**任务与结论身份。** 按当前用户指令，承接STATE-FORMATION-20261109-v1，使用本地Qwen3-4B追踪有效记录选择与过时记录排除；完成新样本行为校准、全场采集、候选冻结、留出预测、原模型作用和负对照、客户端同步。当前运行 `{C.RUN}`；模型/tokenizer快照 `{C.SNAP}`。只有4B单模型、CUDA原生BF16、batch1、SDPA显式因果mask；没有运行14B/GLM4或发布远程网页。所有正式晋级为0。**原完整范围行为门失败；后续是明确预注册收窄范围的候选研究与失败诊断，不能把筛选后的结果写成原方案确认成功。** 下述工程验收通过与科学命题成立分别记账。

**校准过程、原始失败和研究方向改变。** 原颜色取值任务v1及明确状态指令v2均有条件格失败；保留两版全部小试输入、输出、模型身份及计划，不继续第四、第五轮提示调优。v3把“选有效来源”与“从来源绑定颜色”拆开，输出只有记录编号1/2。ACTIVE/VOID用于发现，CURRENT/RETIRED用于留出；显式定义有效/作废，均让有效记录在前/后、内容一致/冲突、两种颜色映射交叉。主体记录英文，查询中英文，输出始终数字；不能称为颜色或一般时间推理已解决。

| 版本与范围 | 实际输入数 | 准确率 | 原冻结逐格90%门 |
| --- | ---: | ---: | --- |
{history}

v3小试每格只有4条，0.90点阈值意味着必须4/4；未改变阈值，先扩大为全部发现集1024条行为校准，不采特征、不打开留出。完整发现集仍有15条错误，4个失败格：frame5/有效在前/冲突/中文=13/16；frame7/有效在后/内容一致/中文=13/16；frame1/有效在前/一致/英文=14/16；frame1/有效在前/冲突/英文=9/16。原小试3条错误和完整校准结果分别保留。小试256条属于1024发现条目的子集，不重复计数。v1/v2各256条与v3全4096条合计4608个实际运行输入，干预重跑不是新增独立样本。

**收窄范围的准入与暴露审计。** 在原生采集之前冻结analysis-plan：任一家族8条件中，只要触及一个低于0.90的条件格，就排除整组家族的特征拟合和正面解释，避免只扔掉答错的单条。发现集合格{a['discovery_eligible']}/128家族；前四记录句式拟合原生排名，后四句式选择层和组，发现期可用样本已经暴露，不能再次称盲测。冻结候选、预测器与对照后再打开留出。留出全部384家族、3072条行为结果完整报告，其中合格{a['confirm_eligible']}/384家族，排除{384-a['confirm_eligible']}组。失败组的全场仍作为可复算原始档案保留；不进入展示的特征均值、预测MSE和正面作用汇总，其原模型干预另标失败诊断。**按留出行为筛出的子集是条件诊断，仍存在选择偏差；不是无偏的全域确认。** 全域门、原方案失败、选择过程均不得被覆盖。

留出失败格：{cells or '无；但发现期完整门失败仍保留'}。留出行为错误{len(r['failures'])}条，全部输入与两个数字候选logit在客户端逐页可查。19条错误全部在有效记录在前、过时记录在后；有效在后1536条全对。13条为中文查询、6条英文；12条新旧内容冲突、7条内容一致；其中1条为BF16两候选并列，严格口径判错，但argmax可因索引选中正确词。因此需要保留严格优势和首token top1两套口径。这是事后条件相关，支持继续检验后出现记录偏向，不能据此认定内部错误因果。所有原始字段strict_color_correct/color_tie在v3仅为兼容旧内部键名，实际是数字1/2的选择与并列；客户端与本记录均使用记录编号口径，不能混同v1/v2颜色分数。

**材料独立性和适用域。** v3采用8发现记录句式×16实体家族、16留出记录句式×24实体家族，每家族8个P×M×D条件，共4096条。P为有效记录物理位置，M为两实体颜色映射，D为新旧内容一致/冲突；查询实体q与查询语言L在家族间平衡。新实体标识、新名词、新记录措辞及留出查询句式均冻结；每种语言只有4个查询框架，被16记录句式复用，不能宣称16个独立查询文法。实际文本由作者生成，使用shared/dist_protocol/grammar.py校验，未修改共享协议。功效计划Δ=.06、n384、α=.05/4，工具近似所需310、功效约.889；这是二值独立家族近似，不能直接套用于连续logit、16个相关句式簇或筛选后的样本量。置信区间是4096次家族/记录句式重采样，主文列98.75%句式区间；共享查询框架、固定人工模板及条件筛选会使总体推广不确定性更大。

**测量对象与数学定义。** 每条输入保存原生BF16位模式：hidden[37,4,2560]（embedding与36块后残差）、final_norm[4,2560]，4位置为引导结束、记录1结束、记录2结束、答案前；head_context[36,32,128]保存真实o_proj输入；gate/value/units各[36,9728]保存最后位置MLP门、值和实际down_proj输入。全部36块、全部32头的最后query注意力概率及4来源质量（记录1、记录2、指令、其余）以FP32重算保存；源分解向量source_context只保存B18—26。B29候选有QK质量与整体head输出，但没有归档其逐来源V/O向量，不能伪称这部分已经完全闭合。

原模型自然前向保持SDPA；另由真实归一化Q/K、RoPE和V重算注意力，FP32来源分解与实际BF16头输入的最大相对L2误差为{a['maximum_QKV_relative_error']:.8g}，小于冻结0.02。该误差反映不同精度算术，不能宣称逐位重现SDPA。MLP逐例核验实际A=silu(G)⊙V等于down_proj输入。未来条件尚不可见的引导结束位置，全512家族的八条件hidden均逐位相同；事后CPU另核对{a['prefix_arrays_independently_compared']}组数组。相同token_id=220的答案前位置h0在全部材料相同，是免费词汇身份基线，不能只凭最后token嵌入解释上下文选择。

令最后query对记录i的头质量为m_bhj=Σ_(t属于记录j) softmax(QK^T/√128)_t，路由对照R_bh=m_valid−m_obsolete，分别在有效在前和在后计算。训练句式每块按两位置均值的最小值挑2头，再由验证句式同一最小值挑块；防止总选最后记录的头混入。冻结头组为**B29 H11、H15**。令w=E[1]−E[2]，原生单元激活a_bu=silu(g_bu)v_bu，局部写入读取系数d_bu=w^T W_down[:,u]；按符号s=+1（有效记录1）/−1（有效记录2）在冲突配对中计算E[s a_bu d_bu]，在预设B12—26候选块内选32单元并验证挑层。冻结组为**B26**，索引：{units}。虽然全部36块均采集和展示，MLP候选搜索只覆盖预注册B12—26，不能把B26称为全模型最佳组。

w^T W_o z_h与w^T W_down a是固定位置的局部线性读取，省略后续层和最终RMSNorm，属于定义/代数计算，不等于真实最终因果贡献。两组发现期验证maxT p均为1/1025，门内选择性通过，但行为范围门失败仍不变。负对照不再只用一套可能反号的打乱标签：1024次整家族Rademacher符号交换，绝对自归一化统计量，并对训练期选定的候选组在验证层面取最大统计量校正。测试植入信号能找回、噪声不通过、整体反号结果不变；这些是隔离工程测试，不作为Qwen科学数据。

留出合格配对的候选观测：
{observations}

**同批材料上的简单基线。** 目标是原模型z1−z2，所有方法在同一批合格家族比较。原生模型使用2头各自局部写入、2头物理记录1−2质量、32单元组局部写入和可见语言/冲突条件的7维输入，发现集标准化ridge λ=1拟合；未输入真实有效位置或未来答案，但需要真实原模型激活，尚不是仅从文本预测的图谱算法。32格基线使用外部已解析q/L/P/M/D条件、发现集格均值；全局语言基线用两个均值；hs0基线因为输入相同只预测发现集常数。hs0补记在留出完成前冻结，不改变候选或阈值。

{baselines}

原生组/32格MSE={r['baselines']['native_over_structural']:.6g}；“32格误差−原生误差”句式区间{ci(r['baselines']['native_gain_ci'])}。原生读出对正确编号的准确率{r['baselines']['gold_accuracy']['native_group_readout']:.6%}。始终第一/最后均50%，显式状态解析器100%，解析器是人工给定规则，不是神经机制。只预测二候选logit差，未预测完整词表分布，不可包装成语言编码整体破解进度；也没有与SAE/transcoder公平比较。

**原模型作用设计与结果。** 每个新留出家族预选1个冲突条件及同映射/位置的一致条件；全部384家族都运行，报告分开保留合格条件和失败诊断。仅最后位置将所选头的o_proj输入或单元组down_proj输入乘(1−α)，α=.5/1；不改权重。四个固定同层同规模组作为原生对照，并用随机残差方向匹配候选实际BF16残差变化范数（允许相对误差≤.02）。逐例恢复原前向输出；前8家族另检验0剂量及身份残差重放，均逐位一致。KL为KL(编辑分布||原分布)。作用定义Δm=m_base−m_edit，正值只表示正确编号优势下降。

| 候选 | 操作/剂量 | 合格家族n | Δm | 原同层组Δm | 等范数随机Δm | 超同层组区间 | 超等范数随机区间 | 正确→错/错→对 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
{effects}

**对照补强及其限制。** 在发现期候选冻结后、留出完成与全部因果结果之前，看到同层组写入能量代理仅为候选的约0.25—0.85；因此不能称原同层对照“已经幅度匹配”。另冻结controls-amendment及分析代码：沿每个真实同层对照消融产生的残差方向，缩放并按BF16实际范数迭代，使其扰动量与候选相同，再在原模型下游重放。保持原对照和新对照全部结果，不换候选、不改变主门。该对照回答“相同幅度下，这个方向是否特别”，不是普通原生单元消融，缩放较大时可能外推。四头对照组有成员重复，不是四个独立样本；推断单位仍为家族/句式。

{matched}

头另做最后query到有效记录/过时记录的直接源屏蔽，所有涉及头的注意力重新归一化。来源长度没有匹配，早层与其他位置的信息仍可绕行，所以这些数值只能说明该直接路径的局部敏感性，不能证明整个过时记录被删除或来源必要性。内容一致/冲突配对用于同任务副作用检查；未检测无关数字选择、通用能力、自然自由生成或输出标签重映射。

**严格判读：能留下什么、不能下什么结论。** (1) 保留完整原生索引的候选和量化失败范围，不能把H11/H15或32单元分别命名为“有效性神经元”。(2) 只有作用在等实际范数、真实组方向等强对照后仍有差异，才值得成为下一轮候选；原全域门失败时仍不得晋级本轮正式确认。(3) B26在B29之前，绝不能据两组相关性画出“B29选源→B26执行”的前向链；可能B26准备了编号/状态，B29重新读取来源，也可能两者是不同替代路径，尚需中介干预。(4) 数字1/2读取、物理位置及固定状态词可能解释候选。跨ACTIVE/VOID与CURRENT/RETIRED的观察不等于跨任意时间有效性的机制。(5) 过时记录注意力少不等于被主动抑制，屏蔽过时来源改变输出也可能是重归一化或比较背景受损；必须分开QK选择、V/O写入及下游使用。(6) 按模型行为删整组避免坏样本充当语义正例，但不是免除选择偏差；本轮没有全域盲测机制成功的结论。(7) 不独立的查询框架、单模型单节点、缺输出身份对照、缺全词表预测和中介链，是不能靠继续换层或调门补救的硬瓶颈。

**本轮实际证伪与保留的层级。** B29头组的有效来源质量差在有效位于前/后分别约0.2385/0.2803，响应模式在新材料中复现；但完整移除的优势下降仅0.1142，低于预设0.2，超同层组及超等幅度真实组方向的句式区间都跨0。因此“这对头是主导有效记录选择的组件”未获支持，保留为路由响应候选。B26组完整移除下降0.2399，超过等范数随机方向及真实组方向的差值区间为正，可保留为有限范围内输出优势的作用线索；但全部6种汇总操作在360合格样本中都没有首词正确↔错误翻转，不能称为答案切换开关、必要单元或完整机制。原生读出MSE34.6725，高于32格基线27.5929约25.66%；“基线减原生”的句式区间[-27.2427,10.8007]跨0，未证明结构模型有预测增益。等幅度真实单元对照的最大缩放约17.99倍，虽实际范数匹配合格，方向外推风险很大；其结果只能与未缩放和随机等范数对照一起谨慎解释。至此不再换层、改阈值或改提示补过门，下一阶段先处理输出身份与路径替代的可区分解释。

**理论增量与第一性原理。** 更可操作的问题是条件化来源、关系绑定与输出读取如何衔接。有限参数可共享相同权重，在不同前缀状态下产生不同路由和有效写入；输入嵌入恒等、后续响应不同已经排除“答案前token身份足够”的解释，但并未建立新的数学理论。候选数学对象应是具有输入范围、有效性条件和可组合运算的转移关系，而非把一条激活差分直接当成概念实体。以选源r_i、绑定B(r_i,实体,属性)、输出读出O为分层工作假说，下一步要预测未见条件的状态和输出，并以反例检验；本轮没有证据证明三者独立模块、唯一实现，或由此推出人脑/AGI机制。

**下一大阶段：来源—绑定—输出的跨条件机制检验。** 同一阶段下同时推进三项，冻结共同家族及成本，不按每个小功能另开Phase：

1. **先剥离标签和输出身份混杂。** 固定本轮B29/B26候选，独立交叉状态词同义改写、物理顺序、真正时间更新、无关数字选择、输出1/2↔A/B↔实体名称；一开始冻结完整家族和全域行为门。失败格如实终止其语义主张，不能结果后再换候选/句式来宣布过门。将“能选记录但颜色取值失败”作为独立待解释反例。
2. **再检验组合计算链。** 同时观察不同token位置上的有效状态、实体关系和属性，将B26与B29按真实层序连接为待检假说；定点读取Q/K、源V/O和后续MLP写入，做单独/联合剂量、保持范数的原生方向控制和必要的中介/救援。唯有能预测交互项及替代路径，才从两个组的敏感性提升为链。不得倒置层序或从QK质量跳到完整计算连接。
3. **最后比较可执行图谱预测。** 在从未参与选择的材料上，使用仅预测时可获得的输入建立共享条件算子，比较词汇、位置、n-gram/条件均值、原生组以及预算可比的字典路线；分别报告全场误差、下一token分布KL、实际作用和任务副作用。不能优于强基线、或仅在数字标签内有效时，将其保留为局部读取拼图并停止扩成通用规律，不在同一锁箱上反复调参。新增数据只有带来新的可证伪区分才采集；现有4096条全场可重复分析。

**客户端与工程收尾。** 沿用既有研究报告入口和3D下方常驻工作区，仅增加本轮报告分派；四视图为行为反例、头/MLP原生阵列、简单基线、实际作用。全部1152头与每层9728单元按原生顺序展示，有索引输入、原值及颜色缩放说明；发现/留出、有效位置、原始激活/局部写入可切换。不同阵列按各自绝对最大值着色，色深不可直接跨图比较。两张卡为明确组索引，不用一个残差坐标冒充整个组；只有观察关联边，没有伪造头到MLP因果边。行为失败、排除分母、最弱作用样本、原始输入、冻结和审计链接一并同步。

本地两库逐对象SHA一致，登记{reg['objects']}对象、2卡、2观察边；37坐标产物，共{d['coordinate_values_per_library']}个FP32数值/库，独立解码与分析数组逐值相等；恢复包`{reg['bundle']}`。验证：7项针对采集/匹配/零假设的工程测试、56前端测试、组件lint、生产构建、{d['browser_checks']}项真实浏览器检查（两库身份、真实数值、全部作用、失败与恢复、错误快照阻断、窄屏/键盘）；没有额外触发GPU请求。技术审计核对4096原生文件、384基线重放与恢复、{a['full_vocabulary_arrays_recomputed']}个完整词表输出的独立CPU读出，最大float64/FP32 KL差{a['maximum_CPU64_KL_error']:.8g}。此为同节点技术复算，不是独立科学复现。

完整运行入口及命令均在`tests/glm5/record_selection_*.py`与`verify_record_selection.cjs`；结果在`tests/glm5/result/record_selection_v1/v2/v3`，物理写入C盘已授权研究目录，经D路径Junction访问。逐GPU阶段日志固定批次逐行刷新，WDDM实测峰值专用{d['wddm_peak_dedicated_bytes']/2**30:.3f}GiB、共享{d['wddm_peak_shared_bytes']/2**20:.3f}MiB；共享计数是已分配量，不能证明零数据溢出。v3保留{d['result_retained_bytes']/2**30:.3f}GiB，预算20GiB，收尾C余量{d['C_free_bytes']/2**30:.2f}GiB、D余量{d['D_free_bytes']/2**30:.2f}GiB。没有删除旧证据，已重新核对上阶段冻结源、材料、原报告和恢复包；仅删除本轮可重建前端构建目录。失败小试、失败工程日志及全部原生场继续保留，原生数据保留用于上述跨位置/来源分析，暂不清理。Junction不是第二份备份。未验证范围详见delivery-audit；未上传远程中心、未变更正式科学证据等级。

本地入口：[有效与过时记录实验](http://localhost:5173/workbench?lens=process&station=execution&run={C.RUN}&snapshot={C.SNAP})；选择“研究报告与反例”。报告SHA `{reg['report_sha256']}`；完整交付审计`tests/glm5/result/record_selection_v3/delivery-audit.json`。
'''
    cross=f'\n\n## Phase {iphase}: 有效与过时记录的原生头/单元组检验与受限诊断 [{stamp}]\n\n运行 `{C.RUN}` 已完成本地Qwen3-4B真实采集、独立留出、原模型作用与补强负对照、客户端同步。完整行为门失败，保留全部失败；后续按冻结规则仅报告行为合格配对的条件诊断，0正式晋级。详见 `research/glm5/docs/AGI_GLM5_MEMO.md` Phase {phase} 及 `tests/glm5/result/record_selection_v3/delivery-audit.json`。B29 H11/H15与B26的32单元仅为候选，不能倒置层序拼成因果链。\n'
    assert memo.read_bytes()==old and index.read_bytes()==oldindex,'Concurrent memo edit; rerun against latest prefix'
    with memo.open('ab') as f:f.write(entry.encode('utf-8'))
    with index.open('ab') as f:f.write(cross.encode('utf-8'))
    assert memo.read_bytes()[:len(old)]==old and index.read_bytes()[:len(oldindex)]==oldindex
    C.save('memo-append.json',{'phase':phase,'index_phase':iphase,'time':stamp,'memo':str(memo),'line':old.count(b'\n')+3,
        'old_prefix_sha256':hashlib.sha256(old).hexdigest(),'prefix_preserved':True,'added_bytes':len(entry.encode('utf-8'))})
    print({'phase':phase,'index_phase':iphase,'line':old.count(b'\n')+3},flush=True)

if __name__=='__main__':main()
