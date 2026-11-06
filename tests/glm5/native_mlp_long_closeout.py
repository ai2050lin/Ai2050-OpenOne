"""Append-only scientific record, prospective next task and client acceptance."""
import sys,json,re,argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_long_materials as P
from native_mlp_long_run import status
from shared.dist_protocol.mechanism_trace import file_sha
from deploy.research_library import Library
from tools.power_analysis import check_phase

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--finish',action='store_true');finish=parser.parse_args().finish
    lib=Library(P.ROOT/'distributed_data/research_library');report=P.read(P.PUBLIC/'report.research.json');prov=report['provenance']
    def archive(name,doc):
        d={'kind':'real','run_id':P.RUN,'model_snapshot':report['model_snapshot'],'provenance':prov,**doc};path=P.PUBLIC/(name+'.research.json');P.save(path,d)
        return lib.ingest(d,str(path.relative_to(P.ROOT)),kind='local_report')['sha256']
    inputs=P.ROOT/'tests/glm5/result/phase2773_w6_v2/capability-inputs.json'
    inputref=archive('capability-inputs',{'schema':'research_keyed_capability_inputs.v1','source_sha256':file_sha(inputs),'items':P.read(inputs),'scope':'Previously exposed keyed pool; existing source and eligibility metadata retained'})
    stats=np.load(P.OUT/'population-statistics.npz');norm=np.load(P.OUT/'write-norm.npy');matches=[]
    for (l,u),(rl,ru) in zip(report['selection']['units'],report['selection']['random_units']):
        assert l==rl;rms=np.sqrt(stats['squares'][0,l,2]/stats['count'][0]);ratios=[float(rms[ru]/rms[u]),float(norm[l,ru]/norm[l,u]),float((rms[ru]*norm[l,ru])/(rms[u]*norm[l,u]))]
        assert all(.8<=r<=1.25 for r in ratios);matches.append({'layer':l,'target':u,'random':ru,'A_RMS_ratio':ratios[0],'down_norm_ratio':ratios[1],'product_ratio':ratios[2]})
    matchref=archive('matched-control-audit',{'schema':'research_control_matching.v1','matches':matches,'range':[.8,1.25],'fit_split':'discovery only'})
    power=check_phase(.2,512,p_d=.3,alpha=.05/16);assert power['pass']
    plan={'schema':'research_native_mlp_next_plan.v1','kind':'proposal','title':'从可检索两极走向可预测语义与真实读写子图','status':'proposed_not_executed',
        'previous_negative_results_retained':True,'tasks':[
            {'title':'自然长前缀两极与语义假设','detail':'复用本轮逐token原场，优先在可见前缀≥512处寻找正极、负极、肩部及弱响应，单独保留早位置探索；只用发现集提出8–16个单元假设，按词汇、位置和文档来源控制。AI标签属于未检验假设。'},
            {'title':'功能解释必须预测新例和近似反例','detail':'冻结新文档行组与未见实体/句式；同时含关键词但不满足功能、无关键词但满足功能。先校准模型行为，G、V、A分开预测；重要测试按文档家族分组，正式规模与功效先冻结，不使用本轮确认或锁箱重新选规则。'},
            {'title':'同预算提取方法比较','detail':'比较词汇/位置、前缀n-gram、关系/句式因子、原生读方向，以及同预算SAE/transcoder；统一训练和全新测试材料。以留出激活误差、完整词表KL和未解释残差衡量增量，不以标签数量计进度。'},
            {'title':'跨token读写路径与分布式复算','detail':'沿真实Attention来源、RMS归一化、gate/up/down及下游消费定位；区分当前query与历史位置作用，以剂量、严格能量随机和能力保持复核。先闭合一条小子图，再注册可哈希分片与独立节点复算，最后顺序扩GLM4和Qwen3-14B。'}],
        'prospective_size_example':{'n_independent_document_pairs':512,'max_unit_hypotheses':16,'expected_delta':.2,'power':power,'status':'planning_example_not_frozen_experiment'},
        'budget':'Reuse12GiB local raw arrays. A fresh run requires own immutable material/power/resource contract; no unattended infinite loop.'}
    planref=archive('next-plan',plan);status('native_mlp_plan','proposed',plan_sha256=planref)
    status('prediction','not_supported',random_replicated=report['analysis']['random_replicated'],random_units=432)
    status('intervention','not_supported',supported=report['dose']['supported'],natural_top1_loss=report['dose']['effects']['target_full']['loss']['mean'])
    status('capability',report['capability']['status'],n=report['capability']['n'],net_loss=report['capability']['net_loss'])
    if not finish:
        status('client_acceptance','running',capability_inputs_sha256=inputref,control_matching_sha256=matchref);return
    for log,expected in [('protocol-tests.log','Ran 261 tests'),('software-tests.log','Ran 6 tests'),('trace-tests.log','Ran 11 tests'),('client-tests.log','48 passed'),('frontend-build.log','built in')]:
        assert expected in (P.OUT/log).read_text(encoding='utf-8',errors='replace'),log
    assert P.read(P.OUT/'system-acceptance.json')['passed'] and P.read(P.OUT/'offline-acceptance.json')['passed']
    assert (P.OUT/'client-3d.jpg').is_file() and (P.OUT/'client-report.jpg').is_file()
    names=['server/mechanism_trace_service.py','frontend/src/components/v8/NativeMlpReport.jsx','frontend/src/components/v8/ResearchInterventionReport.jsx',
        'frontend/src/components/v8/ResearchExecution.jsx','frontend/src/components/v8/MechanismTraceExplorer.jsx','tests/glm5/test_native_mlp_long.py','tests/glm5/native_mlp_long_client.test.jsx',str(Path(__file__).relative_to(P.ROOT))]
    source=archive('client-sources',{'schema':'research_source_bundle.v1','files':[{'path':n,'sha256':file_sha(P.ROOT/n),'text':(P.ROOT/n).read_text(encoding='utf8')} for n in names]})
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M');marker='NMLP-LONG-CLOSEOUT-20261104-v1'
    memo=P.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';old=memo.read_text(encoding='utf8');headers=list(re.finditer(r'^## Phase (\d+):',old,re.M))
    phase=int([h for h in headers if h.start()<old.index(marker)][-1].group(1)) if marker in old else max(int(h.group(1)) for h in headers)+1
    a=P.read(P.OUT/'analysis.json');diag=P.read(P.OUT/'diagnostics.json');dose=P.read(P.OUT/'dose-result.json');cap=report['capability'];registration=P.read(P.OUT/'client-registration.json')
    baseline=dose['effects']['base']['accuracy'];full=dose['effects']['target_full']['accuracy'];random=dose['effects']['random_full']['accuracy']
    text=f'''\n\n## Phase {phase}: 原生MLP长语料普查、同token位置留出预测与真实客户端闭环 [{stamp}]

记录标识：{marker}；时区America/Chicago。继续Phase2784的MLP优先路线，完成一次真实Qwen3-4B CUDA/BF16研究及客户端验收。研究本身的负结果完整保留，未修改阈值，不将采集覆盖与图谱节点数包装为破解进度。

### 1. 问题、来源与冻结

问题：原生MLP单元的上下文响应能否在控制当前token和绝对位置后，由仅看前文的简单可执行规则预测？自然极点、参数读入及写出是否可以追溯？参考Goodfire两极/读写思路（https://already-clean.goodfire.com/），本轮并未复现其全部语义解释流程。

本地模型D:/develop/model/Qwen/Qwen3-4B，完整快照{report['model_snapshot']}；36层×9728=350208原生MLP单元。2,048篇自然文档，每篇输入1024token，共2,097,152输入位置。发现集为FineWeb英文/中文各512篇；确认和锁箱各512篇，来自固定版本Wikipedia中旧20M普查未用的第4–6行组。模型预训练是否见过这些文档未知。所有数据版本、原始位置、许可、维基URL/标题、分词和SHA保存，不同语料/划分不混同。

旧同快照约19,042,014有效位置的512token普查仅作历史背景，不冒充本轮长文复现。发现语料曾暴露于旧普查的汇总统计；新留出材料未参与本轮拟合。材料按文档分开，排除旧库完全相同前1024token；16 MinHash/4band候选、5token shingle Jaccard≥0.85检查不能保证穷尽近重复，更不排除语义/翻译重复。

冻结身份：{P.read(P.OUT/'execution-freeze.json')['documents_sha256']}（documents）；四个主要脚本sha随execution-freeze记录，跑后逐项核对未变。432随机单元=各层无放回12个，另32个旧仲裁单元与Goodfire工具历史2单元为34定向样本，分别统计。所有正式材料、代码、结果在tests/glm5和tests/glm5/result/native_mlp_long_v1；公共元数据在research/native_mlp_long/v1。

### 2. 数学对象与判据

原生计算定义：对块l、位置p、单元u，x=RMSNorm(h)，G_u=W_gate[u,:]x，V_u=W_up[u,:]x，A_u=SiLU(G_u)V_u，MLP写出=Σ_u W_down[:,u]A_u。以上为实际架构计算，不是本轮发现的新定律；G、V、A、残差坐标与标量权重分别记录。G/V分区使用>0与非正，零归非正，不能把非正象限全称为负×负。

拟合模型：Ahat=发现集当前token均值+位置分箱残差均值+βᵀφ(前缀)。φ为12种字符类别在16/64/256/512窗口中的频率及前4token类别，共96维；只读取当前位置以前的文本。Ridge λ=100，只用发现集每8位置采样拟合。不输入待预测状态、未来token、答案或来源ID。线性系数是统计代理，不自动解释为单元语义。

每个确认/锁箱检验各256对独立文档；同语种配对，当前tokenID与绝对位置完全相同，p≥512，每篇在各自划分只进入一对。配对只看文本、在激活前冻结。原token/位置预测在对内平局，随机平局决定基线，随机数固定并保留。判据为确认和锁箱同时：排序准确率提升≥0.15；单侧精确McNemar p<0.05/432；真实A不平局比例≥0.8。预期Δ=.25,p_d=.5，功效计算n_required177，n256，power.964。低于该效应规模的规律未被排除。

负对照：在同当前token内打乱发现集残差后重新拟合同一前缀模型。此对照不是完整条件独立证明；它检验该拟合流程是否能从打乱响应中也得到通过信号。

### 3. 实际结果与严格解释

432随机单元确认0通过、锁箱1通过，双留出0/432；打乱负对照双通过0/432。34历史定向单元双通过0/34。随机样本锁箱A非平局比例中位数1.0，不能把失败归结为全部单元没有响应或量化全变平局。随机单元锁箱排序错误合计52,739个单元×配对事件，存在跨单元相关性，不是52739个独立实验；每单元展示前12条，其余可从paired-predictions.npz与冻结配对完整重算。

事后残差诊断未改正式主判：相对当前token/位置基线，确认MSE相对下降中位数0.1479%，锁箱0.2017%。说明这套粗类别规则的增量有限，不能外推为所有自然语义不可预测，也不能用它推翻Goodfire。某些极点中心化输入方向较一致，描述G极点相似与预测A排序是不同问题；高一致度不能直接充当功能解释。

只用确认提升排序选16单元、每层至多1个，锁箱目标值读取前冻结selection。随机对照同层且发现集A_RMS、down列范数、乘积三比值均在[.8,1.25]；实际16对比值另存控制审计{matchref}。不要求入选者已通过正式门，因此它是确认集最佳候选组，不是16个已确认功能单元。

256个锁箱A文档在相同当前query上消融α=.5/1、随机16 α=1。自然实际下一token正确率base={baseline:.8f}，半剂量=.42578125，全剂量={full:.8f}，随机={random:.8f}；全剂量净损失.0078125，即0.78125个百分点，95%配对bootstrap区间[0,1.953125]个百分点。目标相对随机损失.00390625，即.390625个百分点，区间[0,1.171875]个百分点。未达到事前效应门（损失≥10个百分点、特异差≥5个百分点且两区间下界>0）。全词表KL α=.5为.0007953664，α=1为.0007898669，随机.0007457736，不能称为强或单调的语义门控。自然下一token与选择题能力分开计分。

能力保持：冻结旧混合题池重测新16单元操作。2200正式题base=.68545455，edited=.68409091，4道原对改错、1道原错改对；净损失{100*cap['net_loss']:.6f}个百分点，95%同时Clopper-Pearson损失/收益界给出[{100*cap['loss_interval'][0]:.6f},{100*cap['loss_interval'][1]:.6f}]个百分点，上界<2个百分点，非劣效通过。140参考题不入门。它不替代自然生成保义或推理/编程等各域单独检验。

### 4. 数值、真实参数与存储审计

466单元G/V/A逐token原生BF16位模式无损保存；全350208单元保存G/V/A和、平方和、极值、G/V符号分区，各划分排除首32位置，累计2,031,616个有效位置。全部符号分区计数逐层逐单元核对为有效位置数。并未保存全部自然token的全部模块原场，也没有伪造未采集的全场。

实际G/V读取方程在保存的极点输入与原权重上核对：FP32和式对BF16前向的G RMSE最大.02392486，V RMSE最大.01820164；其值包含BF16舍入，非任意精度恒等。A=SiLU(G)×V在全部357888个保存极点单元事件上逐位相同（包含重复位置，不能当独立样本）。256例只读采集hook不改logits；32例α=0复制假操作与无hook logits逐位相同，均通过。

发现一个必须保留的实现差异：采集batch2与剂量batch1所选A最大绝对差.25；主检验统一用batch2原场，所有剂量比较统一batch1，不跨形状直接作干预对比。8条深采使用真实可见前缀，其矩阵形状也不同，不宣称与1024全输入的中途读出逐位相同。未因差异而重新拟合或选择单元。

极值检索范围从位置32开始，所以个别极点实际可见前缀仅33–511token；客户端明确标短前缀探索，不能借后文当长上下文证据。主留出和8条真实深采均p≥512。下一版自然两极语义材料应从可见长前缀专门提取，不能把输入1024token等同于所有读出都已看到1024token。

共保留{report['native']['bytes']/2**30:.6f}GiB原始数组/深采张量，所有原文件SHA登记；8条长前缀全37层隐态、36块Q/K/V与模块读出完整阵列。16条真实参数来源计算保留所有32Head项、2560输入项和9728写出项；固定样本RMS局部贡献是计算来源证据，不自动成为因果必要性。

### 5. 三图谱与客户端增量

外部库新增自然文档身份、严格同token/同位置配对与前缀描述变量；内部库新增466张可回查说明书及16个Attention来源定位，共482张卡、482条类型边。466个单元均只登记观察；16来源卡为parameter_source，未因计数自动升级，不新增“已破解单元”数字。原生参数、两极、统计、误差、留出和能力退化均保留。

正式客户端已增加全36层9728单元G/V/A均值/RMS阵列、随机/定向分开的单元说明书、实际可见自然上下文、G/V/A、排序反例、真实剂量曲线、能力反例和3D联动。自然语料上下文不再走文档/现实仲裁行为门；未分析单元显示缺少说明书，不生成假标签。输入、许可、完整来源与SHA进入CAS；离线附加包保存研究元数据，大张量仍由本机服务查询。

报告CAS {registration['report_sha256']}；图谱CAS {registration['graph_sha256']}。客户端：{registration['client_url']}；3D：{registration['trace_url']}。验收：261协议、6本轮软件、11原生轨迹、48客户端测试通过，生产构建通过，真实API与离线包回查通过。测试fixture不进入研究数字。源码补充CAS {source}；完整能力材料CAS {inputref}。

### 6. 理论边界、第一性原理与下一阶段大任务

本轮成立的是：有可重算的原生门控乘积和参数读写接口，有覆盖长文/弱响应的原场与严格负结果；未找到粗文本类别可以稳定预测的原生语义单元，也未找到16单元当前query强门控。不能由联合小消融无显著效应推导单元无作用、容量只能分布式、门控普遍秩一，或语言机制只住在开关一侧。

第一性原理：需要解释从可见输入条件到G/V读入、乘性选择、down写出、跨token来源和下游使用的条件化转移规则。真正的解释应能在不读取目标状态的情况下预测新词、新句式和新组合，并量化超过词汇/位置/外部规则的收益。计算恒等式、特征标签、统计预测与因果机制必须分别记账。

下一阶段为一个大任务“自然长前缀语义假设→严格反例与新材料→同预算原生/字典→小读写子图”，而非继续叠加展示功能。具体方案CAS {planref}，状态proposed_not_executed：1）发现集在≥512可见位置提两极/肩部假设，保留未知、多用途与弱响应；2）冻结全新文档、未见词/句式、带关键词的反例和无关键词正例，校准行为，G/V/A分别预测；3）同预算比较词汇/位置、n-gram、关系因子、原生方向及SAE/transcoder；4）追踪历史位置、Attention来源、RMS与下游消费，再做剂量、严匹配随机和能力保持，闭合一条可复算小子图后再独立节点与跨模型。规划示例16假设、512独立文档对、Δ=.2,p_d=.3，功效{power['achieved_power']}；正式实验仍须单独冻结材料与资源合同，不能将该方案当已执行。

通俗说明：这轮把MLP做成了可查原始记录、可查正反例、可连到真实参数的研究对象；简单字符统计解释没有过关，拿掉候选16单元也没有明显损伤目标行为。下一步的重点是找到能准确预言新语境响应的功能说明，再验证这些信号怎样被实际使用。
'''
    if marker not in old:
        with memo.open('ab') as f:f.write(text.encode('utf8'))
    status('client_acceptance','done',capability_inputs_sha256=inputref,control_matching_sha256=matchref,source_bundle_sha256=source,visual=True,API=True,offline=True,memo_phase=phase)
    print(json.dumps({'memo_phase':phase,'next_plan_sha256':planref,'input_sha256':inputref,'client_source_sha256':source}))
if __name__=='__main__':main()
