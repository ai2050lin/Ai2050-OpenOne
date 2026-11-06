"""Source-backed navigation inventory and an unapplied restructuring proposal."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from deploy.research_library import Library

def read(p):return (ROOT/p).read_text(encoding='utf-8')
def entries(path,name):
    text=read(path);m=re.search(r'(?:export\s+)?const '+name+r'\s*=\s*\[(.*?)\n\];',text,re.S)
    if not m:raise ValueError('navigation registry not found: '+name)
    return [{'id':a,'label':b} for a,b in re.findall(r"\{\s*(?:id|k): '([^']+)', (?:label|t): '([^']+)'",m.group(1))]

def main():
    util='frontend/src/components/app/workbenchUtil.js';cfg='frontend/src/components/app/ConfigView.jsx';legacy='frontend/src/components/app/rdc_fusion/LensSpatial.jsx'
    purpose={
      'route:overview':'三方向冻结分片执行进展、模型身份桶、研究库摘要与工作单；执行率不等于破解率。',
      'route:lang':'语言项和模式族覆盖、数据集行为评测、证据等级矩阵及可展开的神经元普查覆盖。',
      'route:rules':'不可变产物、特征卡、图谱边、反例、规则账本及来源登记。',
      'spatial:parameters':'全部层/Head/MLP索引和真实参数CPU切片，研究卡与权重绑定须完整快照一致。',
      'spatial:live':'同一真实样本的Embedding、QK路由、Head写出、MLP G/V/A、生成步和参数贡献；可启动本机长文本采集。',
      'spatial:research':'归档工作包、定位卡及历史产物版本的3D索引，提供报告、反例与真实运行回链；产物架不是语义几何。',
      'spatial:lang':'数据集实测与模式目录两种页内视图；语言项的编码层段、特征和跨模式关系入口。',
      'spatial:response':'模式族分片合并后的响应空间、层间对照和规律统计；区别于单样本实际运行路径。',
      'process:plan':'整体研究路线、计划和工作包阶段；当前同时渲染研究档案和报告。',
      'process:execution':'工作包阶段、报告、真实任务队列以及专门执行工位；与计划页共用ResearchExecution主体。',
      'process:catalog':'研究主张、数据版本、AI/人工评审档案和题目/模式族清单。',
      'config:conn':'中心地址、运行形态、在线/离线模式、连接状态、错误与诊断包。',
      'config:account':'账号、角色、绑定节点、算力贡献、资源预算、托管节点、日志和完成记录；管理员另有账号/数据管理链接。',
      'config:model':'中心登记的被测模型，以及桌面AI助手模型与Key配置；科研模型和评审/助手模型用途不同。'}
    groups={
      'route':entries(util,'ROUTE_VIEWS'),'spatial':entries(util,'SPATIAL_VIEWS'),
      'process':[r for r in entries(util,'STATIONS') if r['id'] in ['plan','catalog','execution']],
      'config':entries(cfg,'CFG_SECTIONS')}
    for group,rows in groups.items():
        for r in rows:r['purpose']=purpose[group+':'+r['id']]
    dedicated={
      'probe':'Goodfire式单MLP单元S1–S9测试；本机GPU；当前面板说明结果只保存在本机。',
      'pipeline':'模板设计→试跑/冻结→分发→审计，同一实验流水线而非四类研究对象。',
      'census':'原生单元普查任务、切片与层组分片、复算及争议处理。',
      'autotest':'登记本机被测模型，选择模式族后顺序自动运行；管理员且本机中心可见。'}
    tools=[dict(r,purpose=dedicated[r['id']]) for r in entries(util,'STATIONS') if r['id'] in dedicated]
    legacy_purpose={'stack':'层结构教学/旧权重浏览，含显式示意激活。','neuron':'种子生成的神经元激活教学示意。',
      'sealed':'历史封存实测激活3D与registry回链。','units':'自然语料原生MLP普查、分布和单元卡。',
      'cloud':'旧语言模板驱动的示意特征点云。','param':'连接时真实权重聚合热图；缺少可用真实数据时可显示标注demo。',
      'fidelity':'历史封存指标的保真度与缺口曲线，需保留原指标和范围。'}
    legacy_modes=[dict(r,purpose=legacy_purpose[r['id']]) for r in entries(legacy,'MODES')]
    issues=[
      {'priority':'P0','finding':'旧本机工具仍暴露数据分析/实时分析以及7种视图，其中包含演示数据。','change':'真实独有能力迁入正式空间，示意迁至独立学习区；旧实时入口转向运行与路径，历史URL保留。'},
      {'priority':'P0','finding':'全局模型条与运行样本、研究成果3D、真实参数的内部模型选择各自持有状态。','change':'统一选中上下文；不可变trace按自身快照固定，跨快照显示不匹配并要求明确切换，不静默替换。'},
      {'priority':'P0','finding':'旧LensSpatial的onGo由Workbench传入空函数；其中部分导航按钮不会完成跳转。','change':'迁移或修复这些入口，并以用户实际到达目标页面作为验收。'},
      {'priority':'P1','finding':'研究计划和执行与审计共用主体，重复工作包、档案和报告。','change':'计划负责问题/假设/预注册，执行负责队列/运行/审计；报告用同一详情组件。'},
      {'priority':'P1','finding':'研究成果3D与语言图谱、进展证据库存在入口交叉；档案节点和机制节点易混淆。','change':'空间设统一图谱浏览，语言/内部/映射作为图层；完整产物版本以证据库列表为主，保留可选3D档案模式。'},
      {'priority':'P1','finding':'神经元普查、跨族关系、本地包、平台信息依赖零散按钮；算力贡献埋在账号页。','change':'原生特征卡并入结构与参数；跨族关系成为图谱图层；导入/同步成为全局动作；算力设常驻快捷入口。'},
      {'priority':'P1','finding':'旧帮助与注释仍把实时运行视为同一层塔/离线只读，实际新增视图有独立实时API。','change':'导航说明和页面帮助由同一注册表维护，注明来源、权限和运行形态。'},
      {'priority':'P2','finding':'材料、任务、研究主张、参数和产物版本混用不同组织维度。','change':'以问题/外部模式/特征组织图谱，以数据集作为证据来源筛选，按实验ID贯通生产与成果。'}]
    proposal=[
      {'id':'route','label':'进展','question':'研究到了哪里，证据有多强？','children':['总览','图谱覆盖','成果与证据']},
      {'id':'spatial','label':'空间','question':'研究对象和实际计算在哪里，如何关联？','children':['图谱浏览','结构与参数','运行与路径','响应对照']},
      {'id':'process','label':'研发','question':'下一项研究怎样设计、执行与复算？','children':['研究计划','材料与数据','实验与队列']},
      {'id':'config','label':'配置','question':'我连接了什么，如何参与和管理？','children':['连接与诊断','账号与权限','本机算力','模型与AI']}]
    migrations=[
      ['空间/研究成果3D','空间/图谱浏览：研究证据图层；版本列表进入进展/成果与证据'],
      ['空间/语言图谱与隐藏relations','空间/图谱浏览：外部模式、内部特征、跨图谱关系'],
      ['空间/神经元普查与特征卡按钮','空间/结构与参数：原生单元、参数、语料统计及特征解释'],
      ['空间/导入本地研究包','全局数据动作：导入、更新、导出；导入后回到对应对象'],
      ['研发/任务与数据','研发/材料与数据；研究主张与AI评审回到同一证据详情'],
      ['研发/执行与审计中的四个专门工位','研发/实验与队列：新建实验时选择类型，详情内进入对应专业面板'],
      ['配置/账号与本机节点','拆成账号与权限、本机算力；顶栏增加贡献算力快捷入口'],
      ['本机工具/真实独有视图','分别并入结构与参数、响应对照、成果与证据，原始范围保持'],
      ['本机工具/演示模式','独立学习/教学区，保留明确演示标识，不进入研究统计']]
    sources=[util,cfg,legacy,'frontend/src/components/app/Workbench.jsx','frontend/src/components/app/WorkbenchHelp.jsx',
      'frontend/src/components/v8/SpaceLens.jsx','frontend/src/components/v8/RouteLens.jsx','frontend/src/components/v8/ResearchSpace.jsx',
      'frontend/src/components/v8/ResearchExecution.jsx','frontend/src/components/v8/ResearchCatalog.jsx','frontend/src/components/v8/MechanismTraceExplorer.jsx',
      'frontend/src/components/app/rdc_fusion/ContributePanel.jsx','frontend/src/components/app/rdc_fusion/GoodfireProbePanel.jsx']
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    d={'schema':'research_navigation_audit.v1','kind':'proposal','run_id':'WB-NAV-AUDIT-20261104-v1','created_at':stamp,'timezone':'America/Chicago',
       'status':'source_audit_and_proposal_not_implemented','primary':entries(util,'LENSES'),'secondary':groups,
       'local_primary':{'label':'本机工具','visibility':'仅platform config.local_tools为真时出现','top_modes':entries(legacy,'TOP_MODES'),'inner_modes':legacy_modes},
       'dedicated_stations':tools,'pipeline_steps':entries(util,'PIPE_STEPS'),
       'hidden_views':[{'id':'neuron','label':'原生神经元普查与特征卡'},{'id':'relations','label':'跨模式族谱图关系'},
          {'id':'local','label':'导入本地研究包'},{'id':'platform','label':'高级平台信息'}, {'id':'route:neuron','label':'旧神经元进展直达链接'}],
       'issues':issues,'proposal':proposal,'migrations':migrations,
       'shared_context':['model_snapshot','precision','run_id','dataset_version','sample_id','token_position','generation_step','block','boundary','head','unit','evidence_sha256'],
       'layout':'顶栏模型/研究对象/来源与同步状态，左侧对象目录，中间图或表，右侧证据卡，下方按需展开原文/数值/反例。',
       'acceptance':['常规用户不再看到含演示数据的本机工具一级入口，真实独有能力有明确新入口。',
          '报告到对应真实轨迹不超过2次点击，轨迹到准确参数位置不超过1次点击；这是目标，不是当前实测。',
          '回退/前进/刷新/深链恢复相同模型快照、样本、位置、层与单元；不遗留跨页面无关查询参数。',
          '负结果、缺失采集、未测、执行失败和演示数据各自有明确状态。',
          '从冻结材料到节点分片、审计、报告和图谱卡共享run/design/artifact身份，不复制出多份可矛盾状态。',
          '不同模型快照/精度不跨桶平均，图谱布局不代表模型几何；全部索引与原始值可查询。',
          '网页/桌面、管理员/成员、在线/离线、本机有/无模型均有测试；网页不能执行的操作说明可去哪里执行。'],
       'source_fingerprints':[{'path':s,'sha256':hashlib.sha256((ROOT/s).read_bytes()).hexdigest()} for s in sources],
       'validation':{'basis':'current source registries and rendered component branches; no user study or browser visual acceptance in this task',
          'primary_count':4,'secondary_visible_count':sum(map(len,groups.values())),'legacy_inner_modes':len(legacy_modes),'application_code_changed':False}}
    assert d['validation']['secondary_visible_count']==14 and len(legacy_modes)==7
    out=ROOT/'research/workbench_navigation/navigation-audit-20261104.research.json';out.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(d,ensure_ascii=False,sort_keys=True,indent=2)
    if out.exists() and out.read_text(encoding='utf-8')!=raw:raise ValueError('audit already exists; create a revision instead')
    out.write_text(raw,encoding='utf-8')
    ref=Library(ROOT/'distributed_data/research_library').ingest(d,str(out.relative_to(ROOT)),kind='proposal')['sha256']
    memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';old=memo.read_text(encoding='utf-8');marker=d['run_id']
    if marker not in old:
        phase=int(re.findall(r'^## Phase (\d+):',old,re.M)[-1])+1
        inv='\n'.join(f'| {next(x["label"] for x in d["primary"] if x["id"]==g)} | {r["label"]} | {r["purpose"]} |' for g,rows in groups.items() for r in rows)
        expert='\n'.join(f'- {r["label"]}：{r["purpose"]}' for r in tools)
        legacy_text='\n'.join(f'- {r["label"]}：{r["purpose"]}' for r in legacy_modes)
        changes='\n'.join(f'- {x["priority"]}：{x["finding"]} 建议：{x["change"]}' for x in issues)
        target='\n'.join(f'- {x["label"]}：'+ ' / '.join(x['children'])+'。'+x['question'] for x in proposal)
        acceptance='\n'.join('- '+x for x in d['acceptance'])
        entry=f'''\n\n## Phase {phase}: 研究工作台全部导航盘点、功能边界与结构调整方案 [{stamp}]\n\n标识{marker}。本轮按用户要求整理页面和分析结构，只做现有源码审计与方案归档，没有修改应用导航、运行模型、变更研究证据或声称完成视觉验收。方案CAS {ref}，kind=proposal。界面判断是源代码层面的可追溯设计分析，不是用户研究或点击次数实测。\n\n### 当前一级、二级页面\n\n实际注册表有4个常驻一级、14个常规二级；local_tools开放时追加本机工具一级。研发上排实际由STATIONS顺序形成研究计划、执行与审计、任务与数据；没有把专业工位全部列为二级。\n\n| 一级 | 二级 | 实际作用 |\n|---|---|---|\n{inv}\n\n执行与审计页的专业入口（属于下钻而不是独立一级）：\n{expert}\n模板流水线再分建模板、冻结、分发、审计四步骤。网页/桌面、成员/管理员和local_tools决定入口可见性，界面隐藏不能取代服务端授权。\n\n本机工具上排数据分析/实时分析，内部7个模式复用：\n{legacy_text}\n旧实时主要扫描权重头/真实权重与逐层参数统计，新运行与路径包含真正前向轨迹；名称相同但数据对象不同。旧组件仍可从一级进入，不能声称已删除。旧按钮由Workbench传入空onGo，部分跳转无效是源码可确认问题；API兼容和浏览器具体表现仍需后续实测。\n\n其他下钻与兼容入口：空间的neuron、relations、local；进展旧neuron深链；配置高级platform。空间语言图谱含数据集实测/模式目录，成果3D含报告/定位卡/选中证据/全部版本。账号管理、数据管理是管理员从账号页打开的独立管理页面，不是所有用户的研究一级或二级。\n\n### 结构问题与建议\n\n{changes}\n\n建议保留四个一级与稳定二级，按用户问题分工：\n{target}\n\n调整后仍是14个常规二级，但空间从5收敛为4，配置从3拆成4以独立显露本机算力；任务流程和图层不再不断增加平行菜单。没有为保持菜单数字而删除科研能力。贡献算力设顶栏常驻快捷入口，直达本机算力，登录后显示本人节点状态；设备/预算/托管授权和日志用现有能力，权限不改变。\n\n空间图谱浏览以外部模式、内部特征及两者映射为核心，知识/分析推理/语法是筛选维度，研究数据集是证据来源筛选。成果档案3D作为可选索引模式保留，完整历史版本的首要载体是证据列表；不把产物架的点当机制图。原生参数和神经元普查用同一对象详情，参数、G/V/A响应、语料两极、选择性、干预、反例在右卡切换。保留完整Head/MLP索引，不以TopK或3D可见点数定义图谱覆盖。\n\n统一研究上下文：模型完整快照+精度、run、design/dataset版本、sample、token/生成步、层/边界、Head/单元及证据哈希。不可变样本有自己的快照，不受全局模型选择静默改写；切换快照时清理不适用索引并说明缺失。参数静态观察允许尚未选样本，但必须明确只有权重、无响应叠加。只在影响当前页面时显示数据筛选，不摆无效旋钮；当前独立模型选择器和旧scope文案需要统一。\n\n默认页面采用一条结论、少量关键指标、一条下一步；点击再看表和图，原始矩阵/JSON/哈希继续展开。执行进度与证据强度分别展示：已冻结任务有分母才显示执行百分比；图谱按观察/留出/干预/预测及反例统计，不造总破解率。\n\n### 执行顺序与验收\n\n第一阶段清理旧入口、无效跳转、模型状态和帮助文本，保留旧URL迁移与浏览历史；第二阶段合并图谱/证据、原生单元详情和实验详情，并拆分本机算力；第三阶段完善多数据源比较、未见组合预测指标、跨节点复算与离线同步。计划/运行/报告共用同一持久化对象与详情渲染器，展示入口可以多个，科学状态只保留一份权威来源。\n\n{acceptance}\n\n本次核对导航数组、条件分支、子组件用途及来源指纹，机器清单断言14常规二级、7旧视图通过；没有改应用，因此不为文档整理重跑GPU或无关完整测试。没有新增数学定律；理论和科研上的增量是减少对象/来源/执行/证据混淆，使每条机制主张都能沿同一身份回查原文、完整数值和反例。当前瓶颈仍是可推广机制与图谱预测，不能把页面收敛当破解进度。\n\n通俗说明：进展看走到哪里，空间看模型如何工作，研发做下一轮实验，配置准备参与条件。保留二级导航来分清对象，把重复页面、旧教学演示和流程步骤各自放到恰当位置。\n'''
        with memo.open('a',encoding='utf-8') as f:f.write(entry)
    print(json.dumps({'artifact':str(out),'sha256':ref,'secondary_count':d['validation']['secondary_visible_count'],'application_code_changed':False},ensure_ascii=False))

if __name__=='__main__':main()
