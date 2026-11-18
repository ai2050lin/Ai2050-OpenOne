"""Append the actual completed phase; never replace historical research notes."""
from datetime import datetime
import re
import qwen14_mlp_atlas as C
from qwen14_mlp_provenance import save_result


def main():
    report=C.read(C.OUT/'report.json');api=C.read(C.OUT/'api-acceptance.json');delivery=C.read(C.OUT/'delivery.json')
    assert api['passed'] and delivery['native_archive_verified'] and delivery['browser_verified']
    natural=report['intervention']['natural'];cap=report['intervention']['capability']
    primary=[r for r in natural['rows'] if r['condition']=='native' and r['dose']==0]
    cap_native=[r for r in cap['rows'] if r['condition']=='native']
    zero=sum(r['zero_holdout_keyword_hits'] for r in report['layers'])
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    memo=C.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    old=memo.read_text(encoding='utf-8')
    marker='Qwen3-14B八层全量原生MLP档案、留出失败与精确KV诊断'
    if marker in old:raise RuntimeError('This phase is already appended; do not duplicate')
    phase=max(map(int,re.findall(r'^## Phase (\d+)',old,re.M)))+1
    rows='\n'.join(f"| B{r['block']}/U{r['unit']} | {r['argmax_changes_all']}/{r['n_total']} | {r['correct_to_wrong']} / {r['wrong_to_correct']} | {r['mean_nll_change_all']:.6g} | {r['n_matched']} |" for r in primary)
    text=f'''

## Phase {phase}: {marker} [{stamp}]

### 问题、授权与实际范围

用户要求参考Goodfire，对本地Qwen3-14B的B28–B35所有MLP单元开展分析并接入客户端；补充要求复用以前GPU＋CPU运行方式，按实际效率选择。RUN `{C.RUN}`，BF16快照 `{report['model_snapshot']}`。40层模型的零基B28–B35，每层17,408个，共139,264个；不能误称最后八层。沿用手动部署正式服务器安排。本阶段完成全量档案和有限单元诊断，不声称全部语义破解。

历史记录核对到14B的GPU10GiB＋CPU16GiB＋磁盘方案；材料与本次不同，不能计算公平速度倍数。本轮CPU读原始权重、CUDA逐层加载原生HF块，256篇隐藏状态驻留GPU，前向batch=2，未量化。1024篇经过完整40层前向、采集目标8层，实测{report['collection_seconds']:.2f}秒。显存峰值见collect-r2.log；未测WDDM共享显存，不能宣称共享内存占用为零。

### 材料、方法和数学对象

1024篇中英文1024-token自然文本，512发现、256确认分片、256留出分片。来自已有合法语料，按文档身份选择且保留原划分，不是全项目新盲测。每篇8个固定位置保存G/V/A、MLP归一化输入和输出，全512–1023位置扫描两极。模型、tokenizer、输入token、代码、原生参数及产物哈希保留。

`z_l=RMSNorm(h_l+Attention_l(prefix))`，`G_l=Wgate_l z_l`，`V_l=Wup_l z_l`，`A_l=SiLU(G_l)⊙V_l`，`MLP_l=Wdown_l A_l`。单元j的写出为`A_lj·Wdown_l[:,j]`。这些是模型计算恒等式，不是已发现的语言语义定律。某层写出与下一层gate/up行的余弦只描述参数对齐，遗漏中间注意力、归一化和其他路径，不能直接升级为因果边。

Goodfire启发采用原生神经元、长前缀、G/V/A两极和符号区，不预设一个单元只有一个概念。本轮没有训练VPD、SAE或transcoder，也未完成这些方法的同预算比较。来源：https://already-clean.goodfire.com/ ，https://www.goodfire.com/research/interpreting-lm-parameters 。

响应预测比较相同262维预算的词身份基线与条件化文本特征，发现集拟合ridge，在两个文档留出分片检验；保留文档级置乱对照。条件化特征含128桶当前token、128桶前128token频率以及6个背景变量；未标准化的频率特征和正则化尺度可能限制模型能力。候选门为两个留出分片均较词身份降低至少10%误差且胜过其他基线。

跨层方法在发现集为每个目标单元冻结绝对相关最大的一个源单元和线性系数，与均值、同编号和随机父单元对照；逐步递推从实测B28 A初始化。相邻层全部17,408个关系值从原场按需计算，不落盘数十GB稠密矩阵；不存在Top-K替代全部数值。

### 实际结果与反例

全部139,264个单元的完整参数与档案已保存；32组原生G/V/A及MLP输出计算核验通过。关键词候选在留出前四极值上下文零命中{zero:,}/139,264。G/V/A的简单条件文本模型通过预设筛选的单元均为0；这否定本次算法达到该门槛，不能推出没有上下文信息。已确认语义单元数仍记0。

七个相邻边界的一步留出误差比均值基线约1.07–1.12；部分单元好于基线，但全体平均不成立。连续递推到B35接近均值预测，未形成可用的独立跨层模型。详细全单元误差、同编号/随机父对照及残差保留，不将误差趋近1误称为收敛到真实机制。

发现集预选每层最高A标准差及最高绝对A均值的两个不同单元，共16个。改变最后token的原生列写入倍数0、0.5、2；不等价于权重编辑或重新GEMM精确消融。256篇自然材料检验全词表KL、真实下一词损失和argmax，512道历史选择题检查能力保持。功效用项目工具事前登记：大于等于8个百分点的逐单元配对效应规划n=256，能力变化4个百分点规划n=512；更小效应未获得同等功效。

初始单查询KV回放因BF16计算形状变化未过KL≤0.001，保留失败校准；没有放宽阈值。改为保持原完整query形状、固定前缀KV，4篇发现集校准通过后绑定源码再跑实测。自然回放通过{natural['replay_passed']}/{natural['samples']}，最大KL={natural['max_replay_kl']:.8g}，恢复全部逐位一致={natural['restore_all']}；能力回放通过{cap['replay_passed']}/{cap['samples']}，最大KL={cap['max_replay_kl']:.8g}，恢复全部逐位一致={cap['restore_all']}。K/V张量运行中使用但未逐样本持久化；输入、运行器和最终归一化状态支持重算。

自然等范数有效{natural['norm_controls_valid']}/{natural['norm_controls_total']}，能力有效{cap['norm_controls_valid']}/{cap['norm_controls_total']}。按实际BF16扰动范数检查≤2%误差；全部失败行保留，匹配比较用同一有效样本。零倍原生列结果：

| 单元 | argmax变化 | 原对改错 / 原错改对 | 下一词损失平均变化 | 有效匹配样本 |
| --- | --- | --- | --- | --- |
{rows}

能力保持为每项编辑原对改错比例的单侧95%Clopper–Pearson上界≤2%，原生48项编辑通过{sum(r['capability_gate'] for r in cap_native)}/{len(cap_native)}。这只针对已有选择题、固定位置与剂量；不能证明多语言、编程、多步自由生成普遍保持，更不是单元语义控制成功。

### 严格审视与理论增量

初始plan/pilot/census缺少项目要求的五字段provenance信封；事后admission-note明确登记，不改写原冻结。后续结果通过tools.provenance.write_result保存，仍不追认前序合规、不晋级正式语义或因果机制。一个物理节点、复用材料、单位置干预、浅层哈希预测器、单父递推、没有自然生成语义评分，均是实质边界。

内部图谱增加了模型快照严格隔离的全量原生节点、参数来源、响应分布、未复现解释、剂量和有限副作用记录。外部语言图谱本轮只增加语料/语言/上下文来源标签，没有新确认语言关系。两图谱间的候选词边仍为检索线索；相关与静态对齐未提升为语言机制边。

第一性原理上的可用对象是条件化读入与写出：同一固定列在不同上下文由G与V共同决定系数。有限参数如何组合需要解释这些系数的条件规则以及下游怎样选择读取，不能仅给每个单元取一个词名。这里提出的是后续工作假设；本轮失败未证明多义性、分布式容量或门控唯一性。

下一大阶段：冻结全新长文档家族和实体/关系/句式留出，先校准模型行为与特征尺度；同材料比较原生G/V条件模型、标准化文本基线、跨单元组模型及适用字典方法。把词身份、位置、上游共同输入、G/V乘积和后续注意力分开做消融预测；先独立预测再做组干预/恢复。通过者再建立跨层算子并做多步生成与第二节点复算，未通过者保留明确失败类别，避免只追加关键词卡。

### 客户端、发布与清理

现有菜单和色调保留。全部139,264单元可在专用3D档案中查询；统一空间中B28–B35的MLP选中对象也联动真实14B参数和统计。完整5120参数、长前缀、相邻17408关系、剂量、能力和反例均可回查。3D布局按编号，不能当模型几何。

本地HTTP验收8层全数组、边界单元U0/U17407、参数与原生文件逐值一致、反向关联一致以及错误快照/越界拒绝；浏览器验收记录见delivery.json。协议315项通过，前端62项通过，新增工程7项通过，研究状态9项通过，生产构建通过。新文件lint通过；历史ModelStructure/ResearchSpace的4项lint错误另有日志，没有宣称全仓库lint通过。协议沙盒socketpair阻塞与旧测试扫描真实研究目录已记录；隔离目录重跑通过。

真实全量卡片入库暴露出旧索引一次返回所有对象导致超时。修复为按12条分页，先排序紧凑身份再取完整卡片，定位卡请求与主索引分离，保留输入焦点和mainline_revision一致性校验；上方分布叠加明确仅当前页，专用研究阵列仍包含全部139,264个单元。最后一页4个单元、总数139,264均核验；精简响应约5KB，实测约3秒。早期23秒的分页测量也保留，不能当作最终性能。

中心/网页包版本{delivery['version']}；包与验收清单见`tests/glm5/result/qwen14_mlp_atlas_v1/delivery.json`。正式中心只读检查仍为1.0.29，本轮未自动部署外网。小研究包走已有上传协议，原生大数据单独导入；不能宣称后者已通过客户端断点上传。部署说明`deploy/README_QWEN14_MLP.md`。

本轮清理范围及逐文件释放字节见cleanup-final.json；保留全部原生场、权重、材料、冻结、失败、预测器与重算依据。清理只针对本轮已停止测试的临时副本及与最终结果重复的进度文件，不删除科学负结果。
'''
    with memo.open('a',encoding='utf-8') as f:f.write(text)
    claude=C.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    ctext=claude.read_text(encoding='utf-8');cp=max(map(int,re.findall(r'^## Phase (\d+)',ctext,re.M)))+1
    with claude.open('a',encoding='utf-8') as f:f.write(f'\n\n## Phase {cp}: Qwen3-14B B28–B35全量档案与有限干预交付 [{stamp}]\n\n详见GLM5 Phase {phase}及RUN `{C.RUN}`。139,264个原生单元已建档；关键词/简单条件预测/单父递推未形成合格语义机制，已确认语义数量仍为0。256长文本、16预选单元和512历史能力题完成有限写入诊断，原始失败、范数不合格对照和准入缺陷保留。客户端接入全参数、上下游关联、剂量和反例；本机验收与正式部署区分，正式服务器仍由用户手动更新。此条只作跨路线索引，不替代完整研究记录。\n')
    save_result(C.OUT/'closeout.json',{'phase':phase,'claude_index_phase':cp,'timestamp':stamp,'memo':str(memo),
        'formal_semantic_mechanisms':0,'profiles':139264,'remote_deployed':False})
    print('appended GLM5 Phase',phase,'and Claude Phase',cp,flush=True)

if __name__=='__main__':main()
