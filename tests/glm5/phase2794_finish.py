"""Append scientific audit and register the actual client closure and next plan."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,re,sys
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
from phase2794_run import status
from shared.dist_protocol.mechanism_trace import file_sha,digest
from deploy.research_library import Library
def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'));report=read(P.PUBLIC/'report.research.json');reg=read(P.OUT/'client-registration.json')
    freeze=report['freeze'];assert all(file_sha(P.ROOT/n)==h for n,h in freeze['dependencies'].items())
    for file,expected in [('software-tests.log','Ran 6 tests'),('trace-tests.log','Ran 11 tests'),('protocol-tests-final.log','Ran 261 tests'),('client-tests-final.log','43 passed'),('frontend-build.log','built in')]:
        assert expected in (P.OUT/file).read_text(encoding='utf-8',errors='replace'),file
    library=Library(P.ROOT/'distributed_data/research_library');a=report['analysis'];cap=report['capability'];prov=report['provenance']
    def archive(name,doc):
        path=P.PUBLIC/(name+'.research.json');P.save(path,doc,True);return library.ingest(doc,str(path.relative_to(P.ROOT)),kind='local_report')['sha256']
    names=['tests/glm5/phase2794_strong_baselines.py','tests/glm5/phase2794_numeric_control.py','tests/glm5/phase2794_figures.py','tests/glm5/phase2794_acceptance.py',
        'tests/glm5/phase2794_verify_offline.py','tests/glm5/test_phase2794.py','tests/glm5/phase2794_client.test.jsx',str(Path(__file__).relative_to(P.ROOT)),
        'server/mechanism_trace_service.py','frontend/src/components/v8/AtomicArbitrationReport.jsx','frontend/src/components/v8/MechanismTraceExplorer.jsx','frontend/src/components/v8/ResearchExecution.jsx']
    source={'schema':'research_source_bundle.v1','kind':'real','run_id':P.RUN,'model_snapshot':report['model_snapshot'],
        'scope':'post-run controls, strong baseline, UI and acceptance; primary execution dependencies unchanged',
        'files':[{'path':n,'sha256':file_sha(P.ROOT/n),'text':(P.ROOT/n).read_bytes().decode('utf-8')} for n in names],'provenance':prov}
    sr=archive('integration-sources-'+digest(source)[:12],source)
    plan={'schema':'research_native_mlp_plan.v1','kind':'proposal','run_id':P.RUN,'model_snapshot':report['model_snapshot'],
        'decision':'Native MLP first, small verified circuits before any completeness claim; do not require solving all MLP before attention.',
        'literature':[{'url':'https://already-clean.goodfire.com/','status':'working hypothesis and model-specific preliminary audits, not established Qwen3-4B completeness'},
            {'path':'research/neurons/goodfire/GF-L30-U9517-local.json','scope':'historical256token two-unit studies; not new long-text confirmation'}],
        'population':{'layers':36,'units_per_layer':9728,'units':350208,'pole_count':700416},
        'sampling':'Stratified432 random units (12/layer), plus current32 candidates andhistorical2; selection and doc splits frozen before explanations or evaluations; counts are planned, not executed.',
        'long_corpus':'512/1024/2048 document contexts, multiple languages/domains; preserve document families and rare strong events. Start with resource-calibrated corpus; grow based on empirical event frequency and independent-document counts, not a universal token quota.',
        'storage_estimates':{'full_GVA_2M_tokens_BF16_bytes':2_000_000*350208*3*2,'candidate432_GVA_2M_tokens_BF16_bytes':2_000_000*432*3*2,
            'rule':'stream population statistics and pole/context IDs; preserve detailed original vectors for pilot units and controlled templates; do not claim all natural-token arrays saved'},
        'stages':report['mlp_plan'],'success':'Heldout activation/pole prediction, near misses, real reader/writer paths, scoped interventions andgraph-only prediction beyond strong baselines; unknown andnegative cards retained.',
        'remaining_closed_book_task':'Factorially compare known-source-free facts with/without conflicting documents andwith/without reference, matched instruction paraphrases/positions; preregistered eligible facts andnew lockbox before long-context mechanism claims.'}
    plan_ref=archive('native-mlp-next-plan',plan)
    audit={'schema':'research_client_acceptance.v1','kind':'real','run_id':P.RUN,'model_snapshot':report['model_snapshot'],
        'report_sha256':reg['report_sha256'],'graph_sha256':reg['graph_sha256'],'sources_sha256':sr,'plan_sha256':plan_ref,
        'tests':{'protocol':261,'new_software':6,'native_trace_regression':11,'client':43,'production_build':'passed'},
        'actual_API':read(P.OUT/'system-acceptance.json'),'actual_offline_pack':read(P.OUT/'offline-acceptance.json'),
        'browser':{'actual_report':True,'scope_limit_visible':True,'all9728_MLP':True,'parameter_connection':True,'exact_locator_URL':True},
        'proofs':[{'path':str(p.relative_to(P.ROOT)),'sha256':file_sha(p)} for p in [P.OUT/'browser-prediction.jpg',P.OUT/'browser-3d.jpg',P.OUT/'research-results.png']],
        'limitations':['single physical node','large native tensors local only','no new Electron installer generated'],'provenance':prov}
    ar=archive('client-acceptance',audit)
    status('calibration',state='done',passed_versions=[4],failed_versions=[1,2,3],scope='reference-assisted only; no closed-book pass claim')
    status('prediction',output_vs_scope_blind=True,output_vs_strong_rule=False,MLP_mse_gain=a['internal_prediction'][0]['relative_reduction'],scope='familiar entities, new combinations; statistical surrogate')
    status('native_audit',files=report['native']['files'],bytes=report['native']['bytes'],manifests=719,parameter_connections=32)
    status('native_mlp_plan',state='proposed',plan_sha256=plan_ref)
    status('client_acceptance',visual_passed=True,offline_metadata_passed=True,artifact_sha256=ar)
    memo=P.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';old=memo.read_text(encoding='utf-8');marker='P2794-REFERENCE-ARBITRATION-AND-NATIVE-MLP-RESEARCH-v1'
    if marker in old:print('Memo already appended');return
    number=max(map(int,re.findall(r'^## Phase (\d+):',old,re.M)))+1;stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    text=f'''

## Phase {number}: 属性标签去歧义、双范围行为校准、真实路径与未见组合预测及MLP优先路线 [{stamp}]

记录标识：{marker}。时区America/Chicago。用户要求先校准属性/文档与现实范围，再用本地Qwen3-4B做路径和未见组合预测，持续积累到客户端；之后分析能否先参考Goodfire破解MLP。此次已执行实际CUDA研究，不能把有参考来源选择包装成内部知识回忆。

### 1. 材料与预注册：解决了什么、没有解决什么

颜色标签易受成熟度、品种、部位影响。此次转为定义唯一的元素—原子序数，参考NIH PubChem官方周期表 {P.SOURCE}，原始JSON字节sha256 {read(P.OUT/'formal-corpus.json')['reference_file_sha256']}。原子序数不因同位素、状态而改变；事实标签与模型能否正确回答分别记录，不把模型预测当真值。

保留四版试跑，全部只用最先16个发现实体、F0/F1、1024实际token，每版64输入。v1四格文档/现实分别为16/16、13/16、12/16、2/16；v2为16/16、13/16、12/16、9/16；v3为16/16、15/16、16/16、11/16。v3改为单实体、等长系统级范围指令后仍未过F1现实门。不能说仅清理标签就解决了无参考长上下文仲裁。

v4在文档和现实两个范围都明确给出独立核对的现实参考卡，同时保留虚构冲突记录，试跑四格均16/16后冻结正式材料。通过只属于“有参考来源仲裁”，不属于闭卷常识检索。系统指令先于实体位置，因此早位置也能响应范围；本轮没有沿用P2793“早位置看不见后置范围线索”的负对照。提示对在实际聊天token上等长，保留相同正文、两来源数值和共同末位token；不同范围的系统措辞仍是混杂，不能直接称抽象语义开关。

运行P2794-ATOMIC-ARBITRATION-v4。64熟悉实体×F0/F1×两范围=256发现输入；同64实体×F2/F3新虚构数值和新句式×两范围=256未见组合输入；32完全独立实体×F2/F3×两范围=128新实体输入，主面板均1024token。另16发现实体×512/2048×两范围=64长度面板，总704。原子序数全部1–96，Qwen数字逐位分词；主行为检查实际最多4新token贪心生成的完整整数，而非仅首数字命中。DLA与图谱输出预测仅针对下一token/首数字，两个判据严格分开。虚构数值与真实数值首位不同，保证首步对比可区分，但不把首步区分等同完整答案正确。

H2在64实体/F2/两范围的128输入上检验联合32单元消融，预期20个百分点效应、翻转率.3、双侧alpha.05，近似配对功效所需59实体，实际64、功效.832。要求行为门通过、完整整数准确率下降≥20个百分点、相对同块/RMS近邻随机32单元下降≥10个百分点且实体簇bootstrap95%下界>0。剂量[0,.25,.5,1]，每个实际生成query均乘(1−alpha)，不修改权重。候选及控制只在发现集确定。连续KL/MSE验证预先冻结阈值但没有宣称已知功效，记为有留出的预测评估。

### 2. 真实采集、路径与能力结果

本地D:/develop/model/Qwen/Qwen3-4B，CUDA bf16/SDPA，RTX5080，快照 {report['model_snapshot']}。采集与干预685.885秒、峰值CUDA分配10,162,129,408字节；前后完整权重哈希一致。全37层HiddenState、36块HeadZ/MLP G、V、A及残差/读写边界，在实体和末位两个读出位置保留全原生坐标。8条深采另存全部前缀token隐态及postnorm/postRoPE Q/K、分组V；另15个真实缓存生成步。26,611文件、14,316,970,144字节=13.3337GiB，全部重新核对sha256；719完整manifest分片归档。最大注意力重算AV对原生SDPA z相对L2 .0019069<冻结.05。

发现两范围各128/128；未见组合文档127/128=99.21875%、现实128/128=100%；新实体文档63/64=98.4375%、现实64/64=100%；512/2048面板两范围各32/32。v4四个留出范围格都达到≥80%门，两个模型原始错误保留。另短源外无参考知识探针93/96正确，校准16实体中15/16正确；Pr/Tm/Cm分别答57/65/98而真实59/69/96。短探针多数会答而长冲突任务失败，提示访问/来源选择问题值得研究，但提示措辞、长度等共同变化，不能据此唯一定位为门控故障。

在发现集以“预期首数字−竞争首数字”的直接logit贡献，冻结实际最终RMS，选择后18–35块的32个MLP原生单元和8个Head。最多单元位于B34（例如U1995/1066/2020/2424等）；这只选择答案支持候选，不是自然语料的神经元功能解释，也不是仲裁专属单元。

MLP32在alpha .25/.5/1时128输入全部仍答对，没有达到H2；其首步间隔衰减分别−.1875、−.2900、+.31543 nats，非简单单调。随机32准确率也不变，间隔衰减约1.91406 nats，大于目标组；候选8头损伤1/128=.78125个百分点，间隔衰减4.15625 nats，随机头准确率不变。不能把DLA排名当作被证实的总因果作用，也不能因零准确率损伤就断言“从未被用到”或“容量必然分散”。准确率有天花板，零事件bootstrap区间退化，不能当作总体零效应的严格置信保证。

随机单元是在同块中，从RMS最近的32个可选单元随机取样，实际匹配比范围.275899–1.33844，不是严格等能量控制，是本轮控制设计的硬伤。虽然H2本就未支持，也必须保留此限制，下一轮按匹配容差与无法匹配分支预注册。

补充的无冲突控制在自身结果前登记：128同实体/范围例将虚构数值替换为一致真实值，保持参考卡，长度可能±1token。原模型、目标32和随机32全为128/128。它没有支持仲裁特异性，也不能证明这些单元都是一般数字模块；两种任务都未找到准确率上的必要性。该额外控制不改变主判。

能力检验对新32单元操作重新跑2340题：2200正式题，原模型68.54545%→消融68.5%，损失2、获益1，净损失.04545个百分点，≥95%同时Clopper-Pearson损失/获益界[-.24198,.32685]个百分点，上界<2个百分点，重复混合题池非劣效通过；140镜像参考题不入门。不是新盲题，不声称所有来源/能力分别保持。所有模型在GPU上顺序加载运行，未并发两模型。

### 3. 预测算法、强基线与理论增量

完整词表下一token代理：p_hat(y|entity,document,scope)=alpha_scope p_copy,firstdigit(y)+(1−alpha_scope)p_reality,entity(y)。实体记忆节点来自该实体发现集的有参考现实输出，copy原型来自发现集同文档首数字；alpha在发现集0–1网格拟合。预测只访问熟悉实体的已存节点、输入中显式出现的来源数值和范围；不调用原模型，不访问待预测状态、测试logits或后续token。测试真实输出仅事后评分。它是条件统计代理，不能直接冒称已有参数/因果全闭合的计算图。

256未见组合：图谱混合器平均完整词表KL .0715481，范围盲混合基线 .692255，降低89.6645%，实体簇KL收益 .620706，95%区间[.478773,.692362]，首token top1一致255/256=99.6094%，通过冻结弱基线判据。它不预测完整多token整数，不包含对新实体知识的图谱泛化主张。

看到主结果后，另加更强的事后基线审查：直接按外部题目规则选择已打印的来源数值，再用发现集首数字分布原型。强规则基线KL .0739810；图谱优势仅 .00243294，实体簇95%区间[−.00002923,.00730259]，没有证实优于强规则基线。这不改原阈值，但足以限制“89.7%是破解进展”的说法：大部分收益可由明确的外部来源规则解释，而非新发现的内部机制。事后强基线不能冒充全新确认测试。

完整状态预测：在发现集选定的B34，对全部9728 MLP原生单元和下一HiddenState的2560坐标，用X的实体×范围、两个已给来源首数字×范围的one-hot拟合Ridge，B=(X^T X+I)^−1 X^T Y，lambda=1事前固定；输入值在原文中可见，不使用目标实际状态/预测或expected答案标签。训练期方差尺度下，MLP标准化MSE .266091，实体×范围均值基线 .535771，降低50.3349%；Hidden .168253 vs .483452，降低65.1976%，家族簇收益区间均>0。相对事后来源数字原型基线，分别降低61.6633%与69.5554%。这是一个固定末位、单块、受控有参考任务的可推广统计响应规律，不等于已经解释其真实计算路径。

已知模型公式仍需逐项追踪：A_j=SiLU(G_j)V_j，y_k=sum_j W_down[k,j]A_j；同一样本固定实际RMS时，Head→unit gate项为(W_gate[j]⊙gamma)^T W_O,h z_h/r，残差项另列。32个单元的完整实际参数贡献另保存32内容寻址产物，含2560输入项、所有32头项和9728输出项，原生BF16值与FP32求和误差分列。大直接贡献、线性可读出、统计可预测和干预有效是不同命题。本轮前两者/部分预测成立，预注册的必要性效应未成立。

### 4. 客户端、证据库与实际验收

报告 {reg['report_sha256']}；图谱 {reg['graph_sha256']}。登记704样本数据版本、三个不同范围的研究主张；44定位卡=32联合操作成员MLP位置+12个样本参数来源Head位置，64类型化边=32任务映射+32局部参数贡献。不是44个已破解机制；单个单元必要性均未检验。719manifest、全部输入、前三版41条校准错误、正式6条失败（2原模型整数错误、1头消融退化、1图谱首token不一致、2能力退化）及逐条原文可查。旧P2793及旧Goodfire结果不覆盖。

正式客户端新增行为门/剂量、预测/强基线、真实3D/完整参数、反例、MLP主线、预注册来源六个渐进页签。原场页面明确有参考条件与系统提示的因果可见性；随实体、范围、读出位置、Head/MLP和生成步联动，保持原有灰Embedding/蓝Attention/橙MLP色调。B34全部9728单元真实G/V/A可读；浏览器验证H28→U1995局部gate贡献约−.428412，原生与重算值并列。块/头/单元变动同步URL；未分析块的统计剖面返回404，不填0，原生数组仍可读取。

最终261协议测试（142.803秒）、新增6项科学软件/防泄漏检查、11项原生轨迹回归、43项客户端测试、Vite生产构建均通过。实际5001接口已核对全维度、完整快照、生成位置、缺失剖面404和图谱对象；未加载GPU即可浏览原场。浏览器已验证正式报告、强基线说明、3D及参数贡献，截图留存。后台实际离线包验证包含704输入、719manifest、32份全项参数贡献、四版校准、能力及控制、6失败与可复算源代码。约13.33GiB原生张量仍在本机，3D需本机服务，不宣称已全部上传公共中心。只有一物理节点，未完成独立复算，也未重打Electron安装器。

### 5. 能否先参考Goodfire破解MLP：系统方案

可以采用“原生MLP优先”的近期主线，交付可验证的单元和小电路说明书；不应以“先把全部MLP破解”作为研究Attention/残差的前提。Goodfire原文 https://already-clean.goodfire.com/ 将原生神经元和上下游控制流作为主要对象，关注两极、弱响应、近似反例和真实电路；Qwen3-32B的可解释性比例是特定模型、阈值和审计口径下的初步估计/工作假设，不是Qwen3-4B全350208单元干净的证明。

优先MLP的理由是计算边界清楚、原生参数可追溯、无需先训练替代字典即可开始；限制是其输入已经混合Attention/残差/位置和归一化，上下游缺失时单看激活不能解释功能。G的读入、V的内容、A的有效输出、W_down的写出分开；正负两极与G/V四象限都需要检查。SiLU负侧并非严格零，双负也会写出正值；不能直接套用ReLU“未激活就没用”的叙述。读写方向不平行不应单独作为无语义解释的证明。现有GF-L30-U9517/U80两例983030计入位置/256token窗口及各3最小对，没有证明全MLP已解释；此次长文本32候选消融失败也不反证原生神经元路线。

下一阶段组织为五项大交付：
1. 全层普查与长语料基座：按独立文档划分自然发现/确认/锁箱，覆盖知识/推理/语法与中文英文代码，512/1024/2048长上下文。全350208单元流式统计G/V/A、两极、弱尾部、范数与上下文索引；原始文档/token、模型/方法哈希、保留/重算范围明确。采样432随机单元（每层12）作人口审计，加当前32和旧2作有目的跟踪，二者结论分开。
2. 给单元写可输的说明书：极值样例提出假设，冻结同词不同义近似反例、无关键词正例、否定/角色/顺序和新文档验证；双AI可提出/交叉审查解释，但不自动升级人类金标或机制证据。要求跨独立文档的两极响应和误差/未知类别，不把Top例漂亮标签算完成。
3. 原生读写与上下游机制：沿gate/up/down与RMS、Attention实际输入和后续单元读入，分别验证“读到什么、通过什么、写出什么、谁读走”；局部乘积与总效应分别记边。预注册剂量、同层严格匹配控制、单元组交互与补充能力/保义门；不再仅靠DLA选高贡献后找准确率大翻转。
4. 小电路预测验收：原生单元与同预算SAE/transcoder并行比较，用强语法/词汇/位置/外部规则基线检验未见组合激活预测、全词表KL/损失恢复和残差。先完成一个能提前预测且可复算的局部电路；若原生解释存在混合或遗漏，字典是比较工具，不是先验禁止项。
5. 分片扩展与闭卷仲裁回归：将自然普查、解释评审、路径/干预和交叉复算按输入/产物schema、资源预算和哈希注册，独立节点复算后顺序扩GLM4-9B、Qwen3-14B。另冻结已知短问事实的“有/无冲突×有/无参考×指令改写/位置”因子试验，真正回答内部知识何时被调用，保留本轮闭卷失败。

资源硬边界：2Mtoken全体350208单元的G/V/A BF16原场约4.2025TB，不能在当前约200GB可用磁盘上盲目完整落盘。全体保留可合并普查统计/上下文索引，432试点单元的全token G/V/A约5.184GB；试点单元极值/肩部的原层输入向量按在线事件堆保存，受控模板保留全坐标原场。明确这是分级保存，不宣称所有自然token全模块均已缓存。根据每极强响应率p规划取得k有效事件所需约k/p token，并检查独立文档数；固定2M或100M都不是普遍充分性保证。

第一性原理增量：同一有限参数在不同来源/任务条件下产生可组合响应，本轮说明部分全坐标响应可用输入中的条件统计预测；但外部规则可预测行为，不代表找到了实现规则的内部电路。要破解的是条件化信息如何读取、选择、写出与被下游复用的规则，而不是给全部单元贴人类名词。MLP是可进入的切口，最终仍须连接Embedding、Attention、残差、归一化及自回归历史。

通俗说明：这次先把题目改得能明确判对错，并把“给参考能做、不给参考仍会混淆”分开；已能预测一部分新组合下的内部信号，但没有找到一组拿掉就失去仲裁能力的MLP开关。下一步可以优先研究MLP，目标是写出能被新例和反例检验的工作说明书，逐步拼成真实计算线路。
'''
    with memo.open('a',encoding='utf-8') as f:f.write(text)
    P.save(P.OUT/'completion.json',{'phase':number,'time':stamp,'client_acceptance_sha256':ar,'plan_sha256':plan_ref,'sources_sha256':sr,'report_sha256':reg['report_sha256']},True)
    print(json.dumps({'phase':number,'time':stamp,'acceptance_sha256':ar,'plan_sha256':plan_ref},ensure_ascii=False))
if __name__=='__main__':main()
