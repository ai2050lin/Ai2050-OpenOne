"""Append an evidence-scoped route review; no new model experiment."""
import hashlib,shutil
import b36_semantic14 as S
from b36_semantic14_memo import append
S.OUT=S.ROOT/'tests/glm5/result/goodfire_route_review_20261115'
S.TEMP=S.ROOT/'tests/glm5_temp/goodfire_route_review_20261115'
S.RUN='GOODFIRE-ROUTE-REVIEW-20261115';S.configure(2*2**20)
body='''本轮按用户要求通俗总结当前Goodfire路线；只读核对Phase2916、2919、2920、2923、2924、2928–2934及Goodfire原文，没有新GPU、模型测试、统计重跑或发布。

路线是原生单元观察→提出相关关系解释→新词/句式/领域预测→原模型干预→连接局部计算图。Goodfire《Neural Networks Are Already Clean》是工作假说，启发我们联合查看长前缀、正负端、弱响应及上下游，不代表本项目已证明多数单元可解释；其参数分解VPD也是独立方向，本项目当前主线未复现VPD或完成同预算SAE/transcoder比较。来源：https://already-clean.goodfire.com/；https://www.goodfire.com/research/interpreting-lm-parameters 。

现有去重档案覆盖4B B18–35的175104单元、14B B18–39的382976单元，合计558080；不同执行版本不能直接拼接幅值。档案不是语义破解比例。已核对的计算结构为G=gᵀz、V=uᵀz、A=SiLU(G)V、写入=A d：G/V联合决定状态，负G不能一概视为关闭，固定参数在不同语境下作用可变。这是架构计算定义及观测，不是新数学定律。

阈值、代价、检测标准、形式对象等分支是候选解释；14B两项具体关系解释在Phase2923未通过联合留出门。中段单父自由递推基本退回均值。Phase2924三层自由内部递推0/16通过，干预效应预测0/16通过；但B36固定32组有范围内输出分差作用。Phase2928用完整状态和真实局部算子预测后续根效应14/14通过，ABC换XYZ效应可反向，B39八根恢复近零；不能认定共享语义、必要中介或最小电路。Phase2933 Code读出无首选翻转且高度饱和。Phase2934同缓存回放96/96逐token及首续写logits逐位一致；新96家族只改一个指令token，前两步/前三步联合89/96、42/96，行为门失败，未运行新B36路径干预。3题正确步骤但终值错误，17题缺步骤但终值正确。

主要硬伤：解释标签未稳定转为预测；相关图不能代替因果计算图；完整状态与原权重重算不等于小机制提取；格式、位置、输出阶段和历史条件影响结果；大量覆盖只有少量充分测试；固定快照无法证明训练期递归生长。三图谱目前是广泛内部观察档案、有限外部受控材料及局部有证据关联，尚非通用语言机制图。RDC名称与公式不变，无新AGI结论。

下一大阶段建议围绕“条件选择—内容更新—终态报告”联合推进：先在发现材料正交拆分执行范围、忽略语句和输出项数，再冻结新家族；同时从既有档案选择少量关系候选，建立关键词近反例、新词、新句式和新领域检验，避免只围绕B36/算术循环。对行为可靠且解释可预测的范围，同预算比较原生单元/组、简单基线和稀疏特征；依次预测自然响应、完整输出分布、有限干预效应，加入匹配控制和副作用。只有三者共同通过才连接机制图。训练期递归泛化另需合法检查点或受控训练，不用层间相关代替。计划未执行。

本轮结论：已有可研究的零件档案、条件化响应和局部线路证据，尚未获得语言组合能力的完整工作原理。原测试和失败均保留；仅追加记录与统一索引，不改旧结论。'''
r=append(S.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md','Goodfire研究路线通俗复盘：已有结构、证据硬伤与下一大阶段',body)
c=append(S.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md','Goodfire路线复盘索引',f"详见GLM5 Phase {r['phase']}。只读证据审查，无新模型运行。区分558080单元档案、候选解释、局部作用与未完成的机制闭合；保留Phase2934行为门失败。下一阶段围绕条件选择、内容更新、终态报告联合验证，不单独扩大普查或反复调B36提示。")
S.save('review.json',dict(records=[r,c],source_sha256=S.file_sha(__file__),scope='Review only; no new model results'))
for x in [r,c]:assert hashlib.sha256((S.ROOT/x['path']).read_bytes()[:x['original_bytes']]).hexdigest()==x['original_sha256']
p=S.TEMP.resolve();assert p.is_relative_to(S.ROOT.resolve()) and not S.TEMP.is_junction() and not list(p.iterdir());p.rmdir()
print('Appended GLM5',r['phase'],'Claude',c['phase'],'old prefixes unchanged; empty own scratch removed')
