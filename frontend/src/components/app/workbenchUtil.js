/* 研究工作台（v8）的导航配置与跳转归一（非组件）。URL 即状态：?lens= &view= &station= &rtab= &tm= &ds= &mdl=
   （空间内部的语言项、块、单元等状态由 v8/v8Util.js 记录在同一个地址栏里，键名不重叠）。
   ds / mdl 是全局维度（数据集 / 被测模型）：所有一级界面共用一条上下文条，不再是某个面板的内部状态——
   原来数据集选择困在 LangAtlas 的 useState 里，换一页就丢，导致"换个数据集结论还成立吗"在工作台上无法回答。 */

/* 一级界面：前三个是研究对象（进展 / 空间）与实验生产（研发），同属一组、排在左轨分隔线之上；
   分隔线之下是本机形态相关的界面（配置、本机工具）。 */
export const LENSES = [
  { id: 'route', label: '进展', desc: '图谱进展、研究覆盖与验证证据' },          // id 沿用 route，旧链接 ?lens=route 照常可用
  { id: 'spatial', label: '空间', desc: '真实参数、全部 Head / MLP 索引与语言响应图谱' },
  { id: 'process', label: '研发', desc: '整体研究路线、任务与数据、执行与审计' },
  { id: 'config', label: '配置', desc: '当前连接的中心与运行形态、账号、被测模型与平台信息' },
];

/* 进展视图：与空间/研发同一层级的子导航（?rtab=），默认 overview */
export const ROUTE_VIEWS = [
  { id: 'overview', label: '总览', desc: '外部语言、内部模型与关联验证的研究摘要' },
  { id: 'rules', label: '成果与证据', desc: '查重、负面规则、有效规则与原始证据' },
];

/* 空间四个稳定页面：图谱图层、静态参数、真实运行、多样本对照；各自采用对应结构视图。 */
export const SPATIAL_VIEWS = [
  { id: 'graph', label: '图谱浏览', desc: '外部语言模式、内部特征与两图谱关联；按证据来源回查' },
  { id: 'parameters', label: '结构与参数', desc: '全部层、全部Head、全部MLP单元与真实权重切片' },
  { id: 'live', label: '运行与路径', desc: '真实研究轨迹与长文本采集，联动词嵌入、全部Head、完整MLP及原始参数来源' },
  { id: 'response', label: '响应对照', desc: '模式族全场与点云、规律分析、合并统计' },
  /* 谱图关系（跨模式族共享编码轴 / 概念假设）不再是一级 tab：作为语言图谱里的入口进入（SpaceLens 仍渲染该视图，旧链接 ?view=relations|atlas 照常可用） */
];
export const SPATIAL_MODES = [
  { id: 'graph', label: '结构', desc: '3D 模型空间：全部层、Head 和神经元，以及语言图谱' },
  { id: 'live', label: '运行与路径', desc: '信息流、单句归因图和实际运行响应' },
  { id: 'response', label: '响应对照', desc: '语言到内部的对应、保真看板和公开基准' },
];
export const GRAPH_LAYERS = [
  {id:'structure',label:'模型结构'}, {id:'language',label:'外部语言模式'}, {id:'internal',label:'全部研发成果'}, {id:'relations',label:'共享关系与映射'},
];

/* 研发工位：网页只能浏览；执行测试（神经元测试）与管理都在桌面客户端。
   admin：只给管理员账号（连接本机后端的开发不限）；autotest 另外只在本机中心（local_tools）出现——公开中心没有被测模型。
   2026-10-30 收敛：原「模板工作室 / 试跑与冻结 / 分发与节点 / 审计队列」是同一条闭环的四个阶段，合并为 pipeline 一个工位
   （见 TemplatePipeline.jsx）；「AI 助手」与「贡献算力」是配置与账号属性，迁到「配置」界面，不再占研发工位。
   旧链接 ?station=studio|freeze|dispatch|audit 由 OLD_STATIONS 重定向到 pipeline 的对应步骤。 */
export const STATIONS = [
  { id: 'plan', label: '当前研发', web: true, desc: '选择参与方式，查看研究进度和缺口' },
  { id: 'catalog', label: '材料与数据', web: true, desc: '语料、题目、模式族、数据版本与来源' },
  { id: 'execution', label: '实验与队列', web: true, desc: '实验类型、分片、节点与执行审计' },
  { id: 'probe', label: '神经元测试', desc: '本机 GPU 对一个 MLP 单元跑 S1–S9 找特征流程；结果只存本机' },
  { id: 'pipeline', label: '模板流水线', admin: true, desc: '建模板 → 冻结 → 分发 → 审计：生成式模板的一条闭环' },
  { id: 'census', label: '普查作业', admin: true, desc: '作业进度、逐分片状态、复算不一致队列' },
  { id: 'autotest', label: '自动测试', admin: true, localOnly: true, desc: '登记被测模型 → 勾选模式族 → 一键运行' },
];

/* 流水线的四个步骤（?pstep=），与 TemplatePipeline.PIPE_STEPS 一一对应 */
export const PIPE_STEPS = [
  { id: 'design', label: '建模板' },
  { id: 'freeze', label: '冻结' },
  { id: 'dispatch', label: '分发' },
  { id: 'audit', label: '审计' },
];

/* 已合并/迁移工位的旧链接：旧 station → 新落点（pipeline 的要带步骤） */
const OLD_STATIONS = {
  studio: ['process', 'pipeline', 'design'],
  freeze: ['process', 'pipeline', 'freeze'],
  dispatch: ['process', 'pipeline', 'dispatch'],
  audit: ['process', 'pipeline', 'audit'],
  ai: ['config', '', ''],
  contribute: ['config', '', ''],
};

/* 当前形态可见的工位：live = config.ADMIN_LIVE（桌面客户端或本机开发），admin = 管理员账号或本机开发 */
export function visibleStations({ live, admin, localTools }) {
  return STATIONS.filter(s => (!live ? s.web : (!s.admin || admin) && (!s.localOnly || localTools)));
}

/* 哪些视图真读数据集：只有「语言图谱」两处——
   空间 · 语言图谱（数据集实测：同一批注意力头在不同题库上重跑，看结论复现）
   其余视图读的是自然语料普查（1,900 万 token）、已载入权重或规则账本，与题库无关。
   这条口径写在一处，页顶的数据集方块就按它决定显不显示：留着不工作的方块比不显示更糟，
   因为换一项数字不动，会被读成「做了实验但没差别」。 */
export function dsBound(lens, { view, gview } = {}) {
  return lens === 'spatial' && (view === 'lang' || (view === 'graph'&&gview==='language'));
}

// Canonical routes retain aliases for all previously published research links.
export function normalizeState(input) {
  const st={lens:'route',view:'graph',gview:'structure',ptab:'weights',station:'plan',pstep:'design',csec:'conn',rtab:'overview',opane:'',tm:'',ds:'',mdl:'',...input};
  if(st.lens==='home')st.lens='route';
  if(st.lens==='local'){st.lens='config';st.csec='legacy';}
  const old=OLD_STATIONS[st.station];
  if(old){st.lens=old[0];st.pstep=old[2]||'design';st.station=old[1]||'plan';if(input.station==='contribute')st.csec='node';if(input.station==='ai')st.csec='model';}
  if(st.view==='research'){st.view='graph';st.gview='internal';}
  if(st.view==='lang'){st.view='graph';st.gview='language';}
  if(['relations','atlas'].includes(st.view)){st.view='graph';st.gview='relations';}
  if(['neuron','units'].includes(st.view)){st.view='parameters';st.ptab='neurons';}
  if(st.view==='local'){st.view='graph';st.importing=true;}
  if(!LENSES.some(l=>l.id===st.lens))st.lens='route';
  if(!SPATIAL_VIEWS.some(v=>v.id===st.view))st.view='graph';
  if(!GRAPH_LAYERS.some(v=>v.id===st.gview))st.gview='structure';
  if(st.lens==='route'&&st.rtab==='lang') { st.rtab='overview'; st.opane='language'; }
  if(st.lens!=='route'||st.rtab!=='overview'||!['language','internal','relations'].includes(st.opane)) st.opane='';
  if(st.lens!=='config'||st.csec!=='node'||!['service','runtime','account','device','models','resources'].includes(st.cstep)) st.cstep='';
  return st;
}
export function readState() {
  try {
    const q=new URLSearchParams(window.location.search),input={};
    for(const k of ['lens','view','gview','ptab','station','pstep','csec','cstep','rtab','opane','tm','ds','mdl','mrun'])if(q.has(k))input[k]=q.get(k);
    input.importing=q.get('data')==='import';
    return normalizeState(input);
  }catch{return normalizeState({});}
}
export function writeState(input,{push=false,run,snapshot}={}) {
  try {
    const st=normalizeState(input),url=new URL(window.location.href),q=url.searchParams;
    for(const k of ['lens','view','gview','ptab','station','rtab','opane','tm','pstep','csec','cstep','obj','atlasTab','atlasSel','ds','mdl','data','mrun'])q.delete(k);
    q.set('lens',st.lens);
    if(st.lens==='spatial'){
      q.set('view',st.view);
      if(st.view==='graph')q.set('gview',st.gview);
      if(st.view==='parameters'&&st.ptab!=='weights')q.set('ptab',st.ptab);
    }
    if(st.lens==='process'){q.set('station',st.station);if(st.station==='pipeline'&&st.pstep!=='design')q.set('pstep',st.pstep);}
    if(st.lens==='process'&&st.station==='catalog'&&st.mrun)q.set('mrun',st.mrun);
    if(st.lens==='config'&&st.csec!=='conn')q.set('csec',st.csec);
    if(st.cstep)q.set('cstep',st.cstep);
    if(st.lens==='route'&&st.rtab!=='overview')q.set('rtab',st.rtab);
    if(st.opane)q.set('opane',st.opane);
    for(const k of ['tm','ds','mdl'])if(st[k])q.set(k,st[k]);
    if(st.importing)q.set('data','import');
    const changingRun=run!==undefined&&run!==(q.get('run')||'');
    const changingSnapshot=snapshot&&q.has('snapshot')&&snapshot!==q.get('snapshot');
    if(changingRun){
      if(run)q.set('run',run);else q.delete('run');
      q.delete('snapshot');
    }
    if(changingRun||changingSnapshot)for(const key of ['trace','context_trace','tpos'])q.delete(key);
    if(snapshot){q.set('snapshot',snapshot);}
    // Keep the source run/snapshot, but don't leave active controls from another page.
    const trace=q.get('trace')||q.get('context_trace');
    if(st.lens==='spatial'){
      if(trace)q.set('trace',trace);q.delete('context_trace');
    }else{if(trace)q.set('context_trace',trace);q.delete('trace');}
    if(st.lens!=='spatial')for(const k of ['pmodel','pblock','pkind','phead','punit','prow','pcol','artifact','feature',
      'tblock','thead','tunit','ttab','sblock','smodule','sindex','smode','scoord','strace','straceb','spos','sposb','sposrun','spanel'])q.delete(k);
    if(url.href!==window.location.href)window.history[push?'pushState':'replaceState'](null,'',url);
    window.dispatchEvent(new Event('research-context'));
  }catch{/* Read-only address bars keep component state. */}
}
export function resolveGo(prev,a,b,_c,extra={}) {
  const old=OLD_STATIONS[a];
  if(old)return normalizeState({...prev,station:a});
  if(STATIONS.some(s=>s.id===a))return normalizeState({...prev,lens:'process',station:a,pstep:a==='pipeline'?'design':prev.pstep,...extra});
  if(a==='process')return normalizeState({...prev,lens:'process',station:STATIONS.some(s=>s.id===b)?b:'plan'});
  if(a==='spatial')return normalizeState({...prev,lens:'spatial',view:extra.view||(SPATIAL_VIEWS.some(v=>v.id===b)?b:extra.tm?'response':prev.view),...extra});
  if(['atlas','progress','route','home'].includes(a))return normalizeState({...prev,lens:'route',...extra});
  if(LENSES.some(l=>l.id===a))return normalizeState({...prev,lens:a,...extra});
  return normalizeState(prev);
}
