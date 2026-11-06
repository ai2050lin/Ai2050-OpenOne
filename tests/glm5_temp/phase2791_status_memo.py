"""Documentary progress audit; no inference and no new mechanism evidence."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import re
ROOT=Path(__file__).resolve().parents[2]
glm=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
old=glm.read_text(encoding='utf-8')
marker='整体编码进度审计：P2791-STATUS-v1'
assert marker not in old
phase=max(map(int,re.findall(r'^## Phase (\d+):',old,re.M)))+1
stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
am=json.loads((ROOT/'research/assembly/AM-v1-qwen3-4b/report.json').read_text(encoding='utf-8'))
tc=json.loads((ROOT/'research/transcoder/TC-v1-qwen3-4b-L30/report.json').read_text(encoding='utf-8'))
note=f'''

## Phase {phase}: LLM语言编码整体进度审计与历史拼图、客户端图谱边界澄清 [{stamp}]

{marker}。响应用户“词嵌入、注意力、MLP完成哪些破解、图谱有什么进展”的状态询问。本轮仅审阅既有备忘录、冻结报告、代码和研究库；没有新模型前向、训练、干预或实验数字。主要依据Claude Phase262、264–295，GPT5 Q8/Q9及其勘误，assembly.v1/v2与本机Library/Registry。

**总定位：已有可检验的表示规律和部分因果通道，部分跨架构复现；尚无完整语言编码说明书。没有合理的总破解百分比。** 权重寻址、全部Head/MLP索引显示、可读出、干预有效、留出预测、连续自然生成解释是不同完成层级。

1. 词嵌入：输入查表与真实参数寻址已经实现，不等于语义编码已清楚。Q8静态研究发现词频、词形/共享字、跨语言对应及语义类别的统计结构；频率子空间可预测不意味着前向把它当可控频率旋钮，Q8-D/D3的已测条件未支持后者。实体类别中心、共享属性算子和层级/个体差异主要是中性句式中的上下文化状态，不能全部算作静态embedding结果。EA-v1/v2留出类别预测约.92–.95，共享颜色/大小/重量/含水/可食算子、跨语言相对结构与写入窗口已形成局部拼图；不能据此声称解释百万实体及全部关系。
2. 注意力：在受控颜色知识冲突中已定位上下文信息传到答案位置的后段头/组，范围条件改变注意力选择，存在冗余；早期GLM结果为int8，不能混同后来bf16。代码模式QK干预有Qwen/Gemma/GLM-bf16三架构证据：查询加代码分量使看代码key的质量比分别约4.5/8.9/7.7倍；GLM key端幅度未过阈值，Qwen部分单调判据未过。没有解释全部头，也未证明注意力独占读取或三能力分别对应固定模块；Q9-B已有attention也能写入的反例。
3. MLP：后段少量单元在人工模式冲突中参与压制散文，32单元修补有较大恢复；移除其读取的本层传播方向，Qwen/Gemma/GLM约复现64%/79%/78%的修补效应。这些百分比是特定注入/修补范式效应，不是语言能力解释比例；自然代码并不必需这些冲突单元。Goodfire式两极/读写/steering工具已执行百万位置、2个单元，但两个案例均未全部通过判据，读方向不等于写方向。
4. 字典发现已做过：Claude Phase262在Qwen L30约2M训练位置训练m16384/k64 transcoder。原报告同k下FVU {tc['conditions']['tc-m16384-k64-s0']['fvu']:.3f}优于原生神经元{tc['conditions']['neuron_k64']['fvu']:.3f}，损失恢复却{tc['conditions']['tc-m16384-k64-s0']['loss_recovered_vs_zero']:.3f}低于{tc['conditions']['neuron_k64']['loss_recovered_vs_zero']:.3f}，按预注册未胜出；训练/层/目标均有限，不能泛化字典路线无用。GR-Q前10特征/前10神经元在留出词/句式可读出，但该层必要性门未过，不可升级为完整语法机制。
5. 最新W6自然生成分支：P2790旧十层alpha6操作在四领域等长材料里0/48合格书面→口语变化；两AI独立评分而非人类金标准，8独立家族为探索。它限制这个旧候选的外推，不否定上面的代码冲突通道，更不能把失败直接分成“容量本体”。容量×门控仍是工作假设。

**图谱已有三种不同进展，不能混算：**

- 研究拼图：Claude Phase272与274–291清单编号至27，含支持、部分支持和失败条目；覆盖实体/属性、疑问/否定/真值组合、知识-上下文仲裁、模式状态、槽位条件化融合及代码冲突路径。27不是已破解机制数；跨架构范围各不相同。
- 局部预测：AM-v1确实已用图谱部件预测197个留出实体，14语言×任务格12格胜过身份与类别基线，相对均匀候选分布KL恢复中文{am['pooled']['zh']['recovered_atlas']:.3f}、英文{am['pooled']['en']['recovered_atlas']:.3f}（中文有退化真假格，剔除约.63；英文剔除约.31）。需要另一次中性句式前向取得实体编码，预测目标限定在候选答案。AM-v2迁移恢复约.43–.60，也依赖新事件的基础格模型输出，评分在受限候选集；不是原始任意文本直接经图谱预测全词表、连续生成。事件特有效应预测在AM-v3未成功。上述恢复率不是总破解比例。
- 客户端积累：本次查询Registry为11数据版本、3有范围主张、3双AI评审批次、50历史/定位别名；Library当前41张定位卡跨5运行。卡是层/单元/运行定位入口，10层联合操作仍只算1次，不能说41独立机制。历史研究覆盖比这3条规范化主张更广；报告可归档不等于已完成全部历史结构化迁移。P2790保留37层HiddenState、B18..27全部9728 MLP单元、10996有效生成步，160完整阵列可从客户端回查；这些是响应观测，不是已经理解每个单元。

**澄清上一阶段未完成项的适用范围：** “字典发现待完成”应限定为新中段/多层研究线与全图谱发现，不是全项目从未训练字典；“图谱预测待完成”应限定为无目标信息泄漏的广泛自然文本、全词表与连续生成闭环，不是AM-v1/v2未做。旧报告保留，新阶段不能抹去已有局部成果，也不能以旧局部预测替代自然生成验证。

**第一性问题与下一大阶段：** 关键不是继续把模块贴成知识/逻辑/语法标签，而是建立“外部变量在何条件、何位置、何层、经何真实参数通道影响下一词”的可执行关系。优先把实体属性、同模式QK、后段冲突单元三类已有部件放在相同冻结自然长材料中：A多层字典与全坐标背景；B未见词/句式/领域选择性和hs[0]/位置/词频控制；C来源归因、直接作用与总效应并报、自然生成剂量及保义/能力保持。随后预测未见自然前缀并与身份、位置、n-gram基线比较；独立第二物理节点复算。缺的核心是跨任务/层/时间步的可组合预测及参数级解释，不是客户端能不能展示更多点。

通俗说明：已经找到几种可复现的“读、选、写、竞争”的局部工作方式，并能拼出部分受控答案；还没把它们连成覆盖知识、推理、语法和自然生成的整张可计算地图。本轮没有新理论定律或新测试结果，只校正整体进度叙述。
'''
with glm.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps({'phase':phase,'time':stamp,'new_model_tests':0,'memo':str(glm)},ensure_ascii=False))
