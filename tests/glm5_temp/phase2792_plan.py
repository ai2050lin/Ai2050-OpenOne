"""Prepare a reviewable research/3D implementation plan. No model tests or UI mutation."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.research_library import Library

OUT=ROOT/'research/phase2792';OUT.mkdir(exist_ok=True)
RUN='P2792-CONDITIONAL-COMPUTE-ATLAS-PLAN-v1'
stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
git=lambda *a:subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',*a],cwd=ROOT,text=True).strip()
snapshot=json.loads((ROOT/'tests/glm5/result/phase2790_register/freeze.json').read_text(encoding='utf-8'))['model_snapshot']
pr={'git_sha':git('rev-parse','HEAD'),'git_dirty':bool(git('status','--porcelain')),
    'model_path':'D:/develop/model/Qwen/Qwen3-4B','model_revision':snapshot,'seed':None}

principles=[
 ('统一问题','同一实体属性在不同提问范围、构式和语境中，哪些内部响应与真实计算路径复用，哪些条件改变信息的使用？能否预测未见组合，而不只解释已见输出？'),
 ('以条件化计算通道组织图谱','外部模式与变量→模型特定特征/位置→真实模块路径→输出，记录条件、位置、层、生成步和失效范围。节点布局不是模型几何，物理参数连接不是已验证的语言功能。'),
 ('先复用现有结果再新增采集','先回填实体属性、代码QK和后段冲突单元、L30字典、AM局部预测及W6负结果；新候选只从发现集选择。历史原生场缺什么就标什么，不把未保存的注意力编出来。'),
 ('三维定位，二维核验','3D负责层、模块、索引与局部通道导航；完整坐标热力图、注意力矩阵、逐项乘积、剂量与反例用2D/表格联动，点击任何高亮均可回查实数和哈希。'),
 ('三引擎共同检验','A自然字典或原生全坐标发现；B受控变量与未见实体/句式映射；C来源分解和预注册因果检验。不把均值差、单元标签或删除成功各自当完整机制。'),
 ('完成度以预测界定','执行率、可读出覆盖、范围内干预、留出预测保真度分别展示。已有受控AM预测保留；自然全词表/连续生成是更强目标，不以重跑原模型或注入真实目标残差替代。')]
stages=[
 {'stage':'A · 统一证据与可视化数据契约','status':'方案，未执行','work':'回填3条既有强证据线（属性读写、同模式QK、后段MLP冲突门控）及反例；沿模型完整快照、run、sample、token/span、generation_step、block、boundary统一参数、场、特征和证据身份。先修正实时分析32-token/Top16/attn_last限制与选中位置绑定。',
  'acceptance':'没有缺失值伪装0、没有把末位注意力标成任意token、没有把权重缩略图当原始参数。旧链接可用；原生单元、残差坐标、字典特征和标量参数分别寻址。无需GPU即可完成历史迁移与接口合同测试。'},
 {'stage':'B · 统一长材料与自然前向观察','status':'方案，未执行','work':'主线选择实体属性→文档/现实范围仲裁，配套否定/肯定、语域和代码冲突验证。先16内容家族方法试跑；有效token为512/1024/2048，多句式，按家族留出。先复现原有代码冲突正控制，再在新自然材料观察。',
  'acceptance':'同因子配对的核心事实与其他条件匹配，读出在全部区分信息之后，使用共用token并报告hs[0]、词频/长度/位置与n-gram基线。两AI盲评各语域保义与自然性，建议新版本保义点估计≥.90作为试跑门，正式推广需独立样本与区间；失败格全部记录，修改后升版本。'},
 {'stage':'C · 特征、真实连接与范围内因果验证','status':'方案，未执行','work':'在同一快照冻结材料比较A字典/原生坐标、B条件选择性、C来源分解与干预。完整报告G、V、A、W_down贡献；Q/K含归一化与RoPE、V及W_O；随机方向、同层/同能量单元与读写分离负对照。',
  'acceptance':'所有局部分解对真实前向核对，FP32软件对照与BF16误差分别校准；路径来源、直接logit归因、下游总效应分开。新候选的剂量和阈值在确认前冻结，保义与开放任务能力保持并测。只有范围内干预边可被标为干预，参数连接不自动晋级。'},
 {'stage':'D · 锁箱拼装预测、自然生成与跨节点/模型','status':'方案，未执行','work':'先预测未见实体/构式的属性与仲裁答案，再扩展全词表和连续生成。记录预测可用输入，图谱不能读取目标隐态、logits、未来token或真实重建残差。至少第二物理节点复算，再顺序扩展GLM4-9B、Qwen3-14B。',
  'acceptance':'模型×精度×设计分桶；家族级切分、CI与置换，不把token当独立n。两项二值配对主假设可参考delta=.20,p_d=.5,alpha=.025需119家族、取128；更多主判或连续KL指标须重算/模拟功效。完整KL、相对基线损失恢复及反例保存；不承诺总体已破解百分比。'}]
scenes=[
 {'id':'structure','title':'结构与参数','coordinates':'Y=真实块/hs层索引；层板X/Z=Head或MLP索引布局，非语义距离；token与生成步为独立选择器。',
  'content':'保留全部层、全部Q头与GQA共享KV关系；所选层完整MLP阵列、完整向量与权重矩阵分页读取。Embedding灰、Attention蓝、MLP橙。选中对象才展开统计、原文、参数与上下游。'},
 {'id':'trace','title':'运行与路径（现有实时分析升级）','coordinates':'同一层塔同步选中样本、query token、生成步；实际采集与重算结果有显式来源标记。',
  'content':'点击词/跨度→定位embedding行→查看各层query/key匹配与完整注意力行→各Head输出→MLP gate/value/activation与输出贡献→下一token分布。支持原生/编辑、两句式或两因子的并排对照。缺采集的层/步标未采集。'},
 {'id':'atlas','title':'语言图谱','coordinates':'外部模式/内部特征/实验主张分区，层高度可有意义；布局距离本身无物理含义。',
  'content':'知识、推理/翻译/编程、语法是导航分类，语域/语言/领域/长度是过滤条件。选择苹果/颜色等外部节点，查看多个内部条件化特征与全部有范围主张，点击跳回真实样本路径和参数。'},
 {'id':'comparison','title':'响应对照','coordinates':'完整原生坐标与固定顺序热力图；相关矩阵、跨层/时间曲线、剂量和置信区间。',
  'content':'作为3D的定量核验面板，保留原始值与标准化视图、固定共享色标和弱背景。投影3D点云为可选，必须记录冻结投影与解释方差，不用点云距离证明机制。'}]
contracts={
 'identity':['model_snapshot','precision','tokenizer_sha256','run_id','design_sha','sample_id','source_group_id','split','input_ids_sha256','generation_step','token_position','text_span','block','boundary','method_version'],
 'parameters':'embedding row; attention Q/K/V/O head slices, GQA group, bias, QK norm; MLP gate/up rows and down column; scalar index. CPU exact slices, original weights independent of saved edited responses.',
 'native_fields':'manifest-sha addressed; h0..hL, module norm boundaries, actual MLP G/V/A and output where captured, post-norm/post-RoPE Q/K plus grouped V and actual o_proj input, valid positions and EOS mask. Never infer unrecorded fields.',
 'feature_card':['canonical_feature_id','native_locator_ids','feature_source','dictionary_or_direction_sha','typical_contexts','activation_stats','selectivity_and_baselines','holdout_generalization','readout','upstream_downstream','intervention_effect_CI','dose_curve','min_edit_params_or_null','counterexamples','scope','status','evidence_artifact_sha256'],
 'edge':['source','target','connection_kind','condition','sample_or_scope','metric_name','value','CI','negative_control','reconstruction_error','evidence_level','run_id','artifact_sha256'],
 'connection_kind':['known_parameter_connection','sample_computation_contribution','observed_association','heldout_prediction','scoped_intervention'],
 'rendering':'module hue and activation/effect legends distinct; uncollected=null, observed0=0; default local selected path, full contributor ranges accessible, filtering never changes research statistics; no inferred causal edges from weight similarity alone',
 'query_proposals':['GET /traces/{sha}/manifest','GET /traces/{sha}/field?block&token&step&boundary&range','GET /traces/{sha}/attention-row?block&head&query&step','GET /features/{id}/claims','GET /atlas/subgraph?model&condition&source'],
 'status':'New trace contracts and endpoints above are proposed, not implemented in this planning task.'}
resources={'model':'Qwen3-4B, BF16, 36 blocks, hidden2560, MLP9728,32Q/8KV,head_dim128',
 'analytic_not_measurement':True,'T':2048,'hidden_all37_bf16_GiB':.361328125,'mlp_A_all36_bf16_GiB':1.3359375,
 'dense_attention_all36_bf16_GiB':9.,'dense_attention_all36_float32_GiB':18.,'qkv_grouped_all36_bf16_GiB':.84375,
 'capture_tiers':['discovery panel: all native hidden coordinates at registered readouts across all layers; all units/heads at selected boundaries; every actual generation step where contracted',
 'small deep panel: full-token hidden fields, selected layer-group Q/K/V/O and full MLP G/V/A; exact original order, noTopK truncation',
 'attention row reconstruction from captured Q/K and actual mask/position state, validated against actual saved attention output; label reconstructed and native separately',
 'native fields stay node-local, center stores hashes and metadata; publish needed full arrays by range; budget/checkpoint before expansion'],
 'execution':'One shared GPU lease across live observation and distributed work; no three models loaded concurrently; 9B/14B BF16 need suitable node or explicitly recorded CPU offload; no new quantized experiment.'}
doc={'schema':'research_dataset_atlas_plan.v1','kind':'proposal','run_id':RUN,'title':'条件化计算通道图谱：研究与3D联动执行方案',
 'created_at':stamp,'implementation_status':'proposal_only','no_new_model_tests':True,
 'principles':[{'title':a,'description':b} for a,b in principles],'stages':stages,'scenes':scenes,'contracts':contracts,'resource_plan':resources,
 'client_pages':[{'page':'空间','secondary':x['title'],'behavior':x['content']} for x in scenes],
 'dataset_candidates':[],'source_files':{p:sha(ROOT/p) for p in ['frontend/src/components/v8/SpaceLens.jsx','frontend/src/components/v8/LivePane.jsx','server/live_analysis_service.py','shared/dist_protocol/natural_register_runner.py','frontend/src/components/v8/ResearchFocus.jsx']},
 'provenance':pr}
p=OUT/'mechanism-atlas-plan.research.json';assert not p.exists();p.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
ref=Library(ROOT/'distributed_data/research_library').ingest(doc,str(p.relative_to(ROOT)),kind='local_report')['sha256']
memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';old=memo.read_text(encoding='utf-8');phase=max(map(int,re.findall(r'^## Phase (\d+):',old,re.M)))+1
sections='\n'.join(f'\n### {s["stage"]}（{s["status"]}）\n\n{s["work"]}\n\n验收：{s["acceptance"]}\n' for s in stages)
text=f'''

## Phase {phase}: 条件化计算通道图谱的系统研究与三模块3D联动方案 [{stamp}]

方案标识 `{RUN}`。本轮响应用户“接下来做什么、怎样用3D显示词嵌入/注意力/MLP特性与关联”。只做源码审阅、解析资源估算及方案归档，**没有运行新模型、修改客户端代码、实现新接口或新增机制边**。方案文件research/phase2792/mechanism-atlas-plan.research.json，CAS `{ref}`，kind=proposal。

核心主问题：同一属性与语言关系在不同词汇、句式、提问范围和生成历史中，哪些条件化通道复用，哪些变化；提取出的图谱能否预测未见组合的内部状态及后续输出。主线先以已有实体属性—范围仲裁作锚，代码冲突门控作方法正控制，再加入否定、语域与推理/翻译/程序行为，避免一开始同时全面拆解三种能力。

源码发现的第一优先级缺口：server/live_analysis_service.py的MAX_TOKENS_CAP=32、TOP_MLP=16；返回attn_last=a[:,T-1,:]，LivePane标题却同时引用选中tokIdx。该接口适合短文本教学观察，不能作为512/1024/2048研究轨迹或任意query位置的证据。ParameterExplorer已有完整原生参数与存档9728维响应，P2790已有长文本自然生成场，需统一身份与读取合同，而不是再建一个孤立3D演示。32上限不能只改成2048就调用原eager全attention采集。

解析存储估算（非实测）：Qwen3-4B、2048token、BF16，37层hidden约0.361GiB；36层全部MLP activation约1.336GiB（不含G/V）；36层32头完整T×T attention A约9GiB，若FP32约18GiB；按8KV组保存Q/K/V约0.844GiB。均未计模型、临时张量、生成和磁盘副本。采取全坐标不截维、层/位置/任务分层采集：发现面板按预注册读出采全层/全坐标，小深采面板保留全token并细查层组，生成按实际历史采集；从已保存Q/K/mask按需重算完整query行并验证到实际attention输出，原生/重算明确标注。节点原场按哈希留存，中心仅索引与必要数组，资源预算后扩量。

{sections}

### 三维空间与实际计算关系

复用现有空间四个入口：结构与参数、实时分析升级为运行与路径、语言图谱、响应对照。Y轴是真实层索引；层板内是Head/单元索引布局；token位置和生成步用时间/序列控制器，不拿三根轴硬塞高维语义。Embedding灰、Attention蓝、MLP橙，保留现有白/slate/蓝整体风格；叠加激活/选择性/因果效应时另给明确指标和固定色标，缺失与0分开。每次只展开所选层/位置与局部通道，完整阵列与所有贡献可分页读取，筛选不定义主干、不删除低幅背景。

选择“苹果/颜色”外部节点→证据与适用条件→某个真实样本/读出→embedding行及全维向量→Q/K/V匹配与Head写出→残差接续→MLP完整G/V/A及W_down写出→下一词。点击任一对象右侧先一条结论、一个实例、证据状态，再展开统计、原文、矩阵和反例。参数层同时可遍历全部Head和MLP单元，GQA共享K/V不虚造独立参数；历史存档缺字段则显示未采集，不补模拟连线。

精确局部计算对象：`z_h(t)=sum_s A_h(t,s)V_group(h)(s)`，`o_h=W_O,h z_h`；`a_j=SiLU(G_j)V_j`，`y_k=sum_j W_down[k,j]a_j+b_k`，以及模型实际残差与norm/RoPE边界。保存完整乘积和求和误差，可定位到单参数，但单项非零不等于该参数承载语义概念。经最终norm的直接logit归因需标明近似/冻结尺度；直接贡献、下游总效应及独立干预分开。

连线的来源类型至少包括真实参数连接、该样本计算贡献、统计关联、留出预测、范围内干预；证据等级为另一字段。线型表示来源、宽度表示当前明确指标，正负/未支持不混成一条发光因果边。条件化联合效应必要时用超边/计算块，不强拆成一个神经元一概念。参数相似、PCA点云近、注意力权重大、报告归档均不能自动升级机制。

### 特征卡、预测与科研边界

特征卡绑定完整快照、原生位置/边界、字典/方向版本，保存自然激活、典型语境、hs[0]及干扰基线、未见词/句式、上下游、干预/剂量/反例。min_edit_params无可行上界时为null；代码冲突通道和W6自然语域失败分别留存。样本实际trace与跨样本规律卡不能互相冒充。

验证分层：第一层验证已知局部关系/受控答案；第二层只用预测时可获得的embedding或另一次中性编码检验未见组合，明确调用原模型的边界；第三层才是无需目标前向的全词表/自然连续生成预测。保留原AM-v1/v2进展，但不能把受限候选KL恢复当全模型破解。所有预测器禁止使用目标隐态、未来token、真实目标重建误差或完整原模型复算来伪装提取。损失恢复分母为正才定义，负值不截断；连续KL和多主判需单独功效规划。

理论增量是一个可执行的研究对象与验收方案，不是新定律。最主要瓶颈：长文本采集开销、数据身份/历史拼图迁移、自然保义评分、原生/重算数值一致、冗余与自我修复、非线性交互、尚未完成的第二物理节点复算。第一优先级交付是可靠地“点中一个语言变量，就能回查其条件化样本路径、完整数值和相反证据”；最终交付才是从图谱预测未见语言组合。

通俗说明：3D应像有证据可查的模型显微镜：看清对象在哪、此刻做什么、为何认为它与语言特性有关。研究先把已有强拼图在同一批长材料上连起来，再用新材料检验能否预测；漂亮的动画、更多点和连线都不能代替这一闭环。
'''
with memo.open('a',encoding='utf-8') as f:f.write(text)
print(json.dumps({'proposal':str(p),'sha256':ref,'memo_phase':phase,'time':stamp,'new_model_tests':0,'client_code_changes':0},ensure_ascii=False))
