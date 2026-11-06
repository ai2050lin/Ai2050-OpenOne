"""Append the required research memo and register deliverable/next-stage metadata."""
import sys,json,re
from pathlib import Path
from datetime import datetime
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials_v2 as P
import mlp_atlas_natural_collect as D
import mlp_atlas_parameter_audit as Q
from mlp_atlas_publish_final import Archive
from shared.dist_protocol.mechanism_trace import file_sha

def main():
    inv=P.read(P.ROOT/'tests/glm5/result/mlp_atlas_inventory_v1/registration-v4.json');A=Archive(P.ROOT/'research/mlp_atlas/inventory_v1',inv['run_id'],inv);a=P.read(P.OUT/'analysis-result.json');dr=P.read(D.OUT/'dictionary-result.json');audit=P.read(Q.OUT/'audit-result.json');accept=P.read(P.OUT/'real-acceptance.json');assert len(audit['layers'])==36 and accept['source_bank_rehashed_files']==254
    assert all(all(v==0 for v in r['projection_max_relative_RMS'].values()) for r in audit['layers'])
    planfile=P.ROOT/'research/mlp_atlas/MLP_REVERSE_RESEARCH_PLAN.md';plan=A.put('system-research-plan',{'schema':'research_plan.v1','status':'planned','path':str(planfile.relative_to(P.ROOT)),'sha256':file_sha(planfile),'markdown':planfile.read_text(encoding='utf-8')})
    proof=A.put('real-client-acceptance',accept);now=datetime.now().astimezone();stamp=now.strftime('%Y-%m-%d %H:%M');memo=P.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';last=max(map(int,re.findall(r'^## Phase\s+(\d+)',memo.read_text(encoding='utf-8'),re.M)));phase=last+1
    text=f'''

## Phase {phase}: 全350208原生MLP单元的参数逆向、答码留出、全层字典负结果与3D证据账本 [{stamp}]

记录时间：{now.isoformat()}。模型：本地CUDA Qwen3-4B，BF16、SDPA，快照 `{inv['model_snapshot']}`，36块×9728原生MLP单元，残差维2560。仅顺序使用本模型；自然采集、字典训练和参数审计分开取得GPU租约，未量化、未编辑模型权重。**完成的是本模型全量索引、范围内数值逆向和一轮全单元功能检验；没有完成全部神经元的语义/因果破解。**

### 研究原理、材料与冻结

1. 全量账本从既有自然普查逐层核验文件哈希，实际计入19,042,014 token；20M只是预算，不写为实测量。为350,208单元建立快照+原生层+down_proj输入边界+单元索引身份，记录自然均值/RMS/极点、原生参数、历史验证与失败，且不把原生单元等同于语义概念。
2. 新答码研究使用416个新Wikipedia背景，来源ID和512token前缀排除旧2048输入；并非未见预训练文本，也并非未见所有源rowgroup。32背景为发现、192confirm、192lockbox；实体nonce、句式按划分分开，代码映射与事实/问法完全配平。每背景24种条件，共9,984条。问题读出位置1023、assistant边界1033；背景前缀至少915token，实际模型输入1034token，不使用过短文本。
3. 第一版Yes/No、是/否反转映射的发现校准384条中16/48格失败；这些失败全量保留。第二版改用A/B、C/D、甲/乙，384条发现校准48/48格过门，再冻结正式采集和分析规则。未因看到留出失败而改变阈值。
4. 全部输入保存两位置的36层完整G/V/A、MLP输入/输出、Head输出、37层hidden及最终全词表logits，共65,704,656,640字节原生库。没有采集逐源attention行，客户端明确缺失，不用Head输出范数代替注意力概率。公式A=BF16(SiLU(G)⊙V)在采集时逐坐标核对。

### 数学对象与判据

设T为所选事实（地点/便携工具），Q为所问关系，M为答码映射，D=1[T=Q]为实验定义的决定。对原生单元激活A，四格对比

`δ_D = 1/2[(A(T0,Q0)-A(T1,Q0))+(A(T1,Q1)-A(T0,Q1))]`。

它消去事实、问法的加性主效应，等于2倍普通二元因子交互系数；不是新发现的智能定律。方向符号仅在发现集A/B、两映射上拟合。每个单元/读出要求3代码×2映射×2留出划分全部同号排序率≥0.8、非零比例≥0.8，单侧精确二项p<0.05/(350208×2)。12个分量取交集，不假设不同代码或同一背景的24变体相互独立；背景间独立近似及源相关是限制。行为门另要求每个留出条件×句式格准确率≥0.9（每格96）。未通过行为门时，响应复现不升级功能解释。

功效澄清在打开真实原场分析之前冻结：当真实排序率0.85、n=192时，单分量功效约0.9715，但12项交集的union lower bound约0.658，不能声称总体≥0.8；真实0.9时该下界约0.99982。样本量和阈值未变。阴性不能排除更弱、条件化或不同编码方式。

真实参数计算为G=xWgᵀ，V=xWvᵀ，A=SiLU(G)⊙V，y=AWdownᵀ。对两观测语境，令S=SiLU(G)，有恒等式

`S1V1-S0V0 = 1/2(S1-S0)(V1+V0)+1/2(V1-V0)(S1+S0)`。

把该对称分摊用于四格对比，分别保存Gate项、Value项、分组均值/RMS与数值残差。S在CPU上以BF16实现后转FP32，残差包括实现与舍入差。恒等式不是唯一因果分解，代码中的Gate/Value也不是已证实的“语义门控/容量”分工。

### 实测结果与反例

- 全350,208单元、9,984输入×2位置的真实权重重投影完成：36块的G、V、A以及记录A的down投影与采集值全部逐值一致，相对RMS误差0。对应每单元19,968次观测，总计6,992,953,344个A值；MLP标量参数槽位共2,689,597,440。这里核验了已知网络算术与数据来源，不把它当成语言理论突破或每个参数语义已解释。
- 完整问题后52,489单元、assistant边界50,704单元通过范围内响应交集；跨读出去重78,673。前选择点全37隐态及问题末embedding的成对哈希相等，打乱标签复现均0。**96个留出行为格6格失败，最差62/96=0.645833；故本轮功能解释资格为0。**不删除失败格，也不因平均准确率较高而掩盖它们。
- 保存普通8项因子系数、每单元失败格、发现均值预测FVU、误差与零方差标记。该预测器可获得操作性输入因子标签，属于描述基线，不是提取出的语言算法，也不是图谱独立next-token预测。
- 引擎A新增全36层自然MLP输入/输出原场：2048篇×16位置=每层32,768位置；训练1024篇FineWeb（16,384独立位置），confirm/lockbox各512篇Wikipedia。语料在旧研究出现过，不声称全局全新。hook与无hook三批相同batch形状的logits逐值一致，模型前后快照一致。
- 每层训练1024特征、TopK=32的transcoder，1000步×256重采样，36,864字典特征。训练不使用语言标签，拟合与方向提名只用发现材料；训练抽样次数不能当成新增独立token。原生对照为|A_i|×||Wdown_i||提名的32项贡献，另有发现均值基线。
- lockbox字典FVU的范围0.592177–0.994683，中位0.877203；原生32项中位0.640662，均值基线中位1.012717。字典优于均值36/36层，优于原生32项0/36层。B35尤其是推广反例：字典FVU从发现0.1961、confirm0.3176到lockbox0.5922，后者劣于原生0.4700。不能只引用有利confirm。原生完整MLP重算输出相对误差0，排除了该对照的实现偏差。
- 保存全部36个9728×1024发现相关矩阵，每个原生单元只用发现集提名一个字典伙伴，再记录同一伙伴在两留出相关。固定描述阈值|ρ|≥0.2且同号的原生单元24,035个；这不是多重校正后的语义发现，也不是必要性/单义性。全部字典权重、训练曲线、死特征、分文档误差和配对t近似区间保留；文档相关/分布变化限制区间解释。

### 客户端与持久化

正式客户端“空间→研究成果3D”新增：全单元证据账本、全量答码研究、全层自然字典、真实参数审计。可访问最后块B35/U9727及字典F1023；保留每层完整9728阵列、真实G/V/A、37层hidden、全部Head输出、完整参数和数值关联。字典特征与原生单元分开编号，字典原片可以读取真实MLP输出与CPU重建；未采集模块不补造。3D距离只表示索引布局，不表示语义几何或因果边。

全部原始输入与token、384条失败校准、正式错误、判据、功效修订、全单元统计和源代码按内容哈希归档。65GB原场、自然库与字典/分析npz建立内容寻址硬链接，不复制巨大张量。客户端离线包包含研究元数据及统计；原场留在采集节点。新增研究的全量远端job_kind调度尚未完成，不能把本机真实脚本说成所有分布式任务均已接入。

验收：261项dist_protocol回归通过，16项全MLP数学/负对照/读取防护及11项轨迹测试通过，69项前端测试通过，生产构建通过；真实API {len(accept['checks'])}项通过，原场254文件及SQLite索引独立重核哈希通过。浏览器确认末单元、末特征、留出原片与3D完整阵列。视觉验收发现参数审计页异步加载期null传入数组组件会崩溃，已修复显式加载态，并以真实canvas组件覆盖加载→全阵列转换。只读API拒绝未提交行、越界、未采集attention与哈希不符文件。API/软件验收不自动晋级科研证据。

### 严格审视、理论进展与下一大任务

现有强项是完整性、可追溯性、参数计算可复算以及失败被保留；弱项是行为定义推广、字典保真度、语义标注、跨层来源、干预充分/必要性与图谱独立预测。全部索引均有记录，个体语义/因果闭合仍为0；这表示未登记完成证据，不表示全部单元都无作用。也不能据一轮低保真字典否定Goodfire式特征路线。

第一性洞察：要提取的是有限描述的条件化非线性关系如何预测新的组合及后续token，不是强行给每个原生单元分配一个固定概念。G/V局部乘积、跨层共享参数、RMS耦合、attention和自回归历史共同参与；低秩门控与分布容量仍是假设而非普适结论。标签可读、数值可重构、语义可预测、因果可控必须分开。

下一阶段围绕“保真提取—外部解释—计算图闭环”组织：在发现集比较字典宽度、训练量、语料覆盖与误差缩放；新材料重新校准行为门，旧lockbox转开发/反例池；用独立新词/句式/组合与长文本自由生成检验原生/稀疏/组合特征；接入真实QKV、跨层归因、预注册干预及新能力非劣效门；最后只用特征与边预测留出激活和next-token分布，与词汇身份/位置/n-gram比较KL和损失恢复。未完成的后续步骤在客户端标“研究计划”，不伪造运行进度。

系统方案：`research/mlp_atlas/MLP_REVERSE_RESEARCH_PLAN.md`。主要产物：`tests/glm5/result/mlp_atlas_code_v2/`、`mlp_atlas_dictionary_v1/`、`mlp_atlas_parameter_audit_v1/`、`mlp_atlas_inventory_v1/`。
'''
    with memo.open('a',encoding='utf-8') as f:f.write(text)
    phase_ref=A.put('phase-summary',{'schema':'research_phase_summary.v1','phase':phase,'timestamp':now.isoformat(),'markdown':text,'scope':'researchrecord; individualsemanticcausalclosure not achieved'})
    doc=A.lib.artifact(inv['report_sha256']);doc.update(research_plan_sha256=plan,phase_summary_sha256=phase_ref,acceptance_sha256=proof,next_stages=[
        {'id':'gate','title':'新材料行为校准','acceptance':'新词、新背景、新句式逐格通过冻结行为门；旧lockbox不复用作新确认'},
        {'id':'fidelity','title':'提取算法与保真缩放','acceptance':'发现集选型，独立留出FVU、实际替换损失恢复及成本共同评估'},
        {'id':'semantics','title':'外部模式与组合预测','acceptance':'AI解释仅为未检验假设；未见语言组合及自然生成预测验证'},
        {'id':'causal','title':'真实路径与能力保持','acceptance':'真实来源边、干预/负控制/救援及新的能力非劣效判据'},
        {'id':'graph','title':'图谱独立预测','acceptance':'冻结特征和边预测留出激活与next-token；KL、恢复比例优于简单基线'}])
    rr=A.put('report-delivery',doc);A.bind(rr,inv['graph_sha256'],{'study_type':'mlp_reverse_inventory','layer_refs':inv['layer_refs'],'all_units':350208,'research_plan_sha256':plan},'350208个原生MLP单元 · 全量计算与研究证据账本',{'inventory':{'status':'done','units':350208},'parameter_audit':{'status':'done'},'controlled_behavior':{'status':'behavior_failed'},'dictionary_fidelity':{'status':'not_supported'},'client_acceptance':{'status':'done','api_checks':len(accept['checks']),'software_tests':357},'semantic_causal_closure':{'status':'not_complete','individual_mechanisms_closed':0}})
    P.save(P.ROOT/'tests/glm5/result/mlp_atlas_inventory_v1/registration-v5.json',{**inv,'report_sha256':rr,'phase':phase,'phase_summary_sha256':phase_ref,'research_plan_sha256':plan,'acceptance_sha256':proof},False)
    claude=P.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md';cp=max(map(int,re.findall(r'^## Phase\s+(\d+)',claude.read_text(encoding='utf-8'),re.M)))+1
    with claude.open('a',encoding='utf-8') as f:f.write(f'\n\n## Phase {cp}: 全MLP索引、真实参数逆向与字典/答码负结果（GLM5 Phase{phase}详记） [{stamp}]\n\n本地Qwen3-4B全部350208原生单元完成9,984长输入×2位置的G/V/A/down逐值计算核验，均与原场一致；响应交集为52,489/50,704单元，但6/96行为格失败，功能升级为0。36层无标注字典lockbox FVU中位0.8772，优于原生32贡献0/36层，完整负结果保留。全部单元、字典方向、自然原片、参数与反例已接入正式客户端3D；图谱独立预测与个体语义/因果闭合仍未完成。详细公式、判据、功效修订、软件/API验收与下一大阶段见 `research/glm5/docs/AGI_GLM5_MEMO.md` Phase{phase} 和 `research/mlp_atlas/MLP_REVERSE_RESEARCH_PLAN.md`。\n')
    print(json.dumps({'phase':phase,'memo':str(memo),'report_sha256':rr,'research_plan_sha256':plan,'claude_phase':cp}),flush=True)
if __name__=='__main__':main()
