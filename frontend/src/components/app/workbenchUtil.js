/* 研究工作台（v8）的导航配置与跳转归一（非组件）。URL 即状态：?lens= &view= &station= &rtab= &tm=（空间内部的
   语言项、块、单元等状态由 v8/v8Util.js 记录在同一个地址栏里，键名不重叠）。 */

/* 一级界面：前三个是研究对象（进展 / 空间）与实验生产（研发），同属一组、排在左轨分隔线之上；
   分隔线之下是本机形态相关的界面（配置、本机工具）。 */
export const LENSES = [
  { id: 'route', label: '进展', desc: '神经元逆向分析进展、语言图谱进展与证据门链' },          // id 沿用 route，旧链接 ?lens=route 照常可用
  { id: 'spatial', label: '空间', desc: 'LLM 多层结构：神经元逆向分析结果与语言图谱的有效神经元' },
  { id: 'process', label: '研发', desc: '神经元测试、模板流水线、题库；管理员另有普查作业与自动测试' },
  { id: 'config', label: '配置', desc: '当前连接的中心与运行形态、账号、被测模型与平台信息' },
];

/* 进展视图：与空间/研发同一层级的子导航（?rtab=），默认 neuron */
export const ROUTE_VIEWS = [
  { id: 'neuron', label: '神经元逆向分析', desc: '普查规模、逐层摘要与分型、解释检验、模板选择性（各项分开计数）' },
  { id: 'lang', label: '语言图谱', desc: '大类 × 等级覆盖、语言项 × 模型等级矩阵、研究缺口与证据门链' },
  { id: 'rules', label: '规则账本', desc: '已经测过的规则与结论（含被否证的），先查重再动手，避免重复测试' },
];

/* 空间视图：左侧都常驻 LLM 多层结构（层塔） */
export const SPATIAL_VIEWS = [
  { id: 'live', label: '实时分析', desc: '输入一段文本，逐层观察模型怎么算：每个注意力头与 MLP 单元的实时响应和参数' },
  { id: 'neuron', label: '神经元', desc: '选中块的内部结构、9,728 个 MLP 单元阵列与单元卡' },
  { id: 'lang', label: '语言图谱', desc: '大类 → 语言族 → 语言项：编码层段、点云、有效神经元与证据' },
  { id: 'response', label: '响应点云', desc: '单个模式族：3D 响应空间、规律分析、合并统计' },
  /* 谱图关系（跨模式族共享编码轴 / 概念假设）不再是一级 tab：作为语言图谱里的入口进入（SpaceLens 仍渲染该视图，旧链接 ?view=relations|atlas 照常可用） */
  { id: 'local', label: '本地包', desc: '在浏览器内打开下载的研究包 / 谱图包（不上传）' },
];
const OLD_VIEWS = { atlas: 'relations', units: 'neuron' };   // relations 不在导航里，但视图仍可从语言图谱进入

/* 研发工位：网页只能浏览；执行测试（神经元测试）与管理都在桌面客户端。
   admin：只给管理员账号（连接本机后端的开发不限）；autotest 另外只在本机中心（local_tools）出现——公开中心没有被测模型。
   2026-10-30 收敛：原「模板工作室 / 试跑与冻结 / 分发与节点 / 审计队列」是同一条闭环的四个阶段，合并为 pipeline 一个工位
   （见 TemplatePipeline.jsx）；「AI 助手」与「贡献算力」是配置与账号属性，迁到「配置」界面，不再占研发工位。
   旧链接 ?station=studio|freeze|dispatch|audit 由 OLD_STATIONS 重定向到 pipeline 的对应步骤。 */
export const STATIONS = [
  { id: 'probe', label: '神经元测试', desc: '本机 GPU 对一个 MLP 单元跑 S1–S9 找特征流程；结果只存本机' },
  { id: 'pipeline', label: '模板流水线', admin: true, desc: '建模板 → 冻结 → 分发 → 审计：生成式模板的一条闭环' },
  { id: 'catalog', label: '题库', web: true, desc: '所有用户看到的同一份题目清单' },
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

export function readState() {
  try {
    const q = new URLSearchParams(window.location.search);
    let lens = q.get('lens') === 'home' ? 'route' : q.get('lens') || 'spatial';
    const view = OLD_VIEWS[q.get('view')] || q.get('view') || 'neuron';
    let station = q.get('station') || 'probe';
    let pstep = q.get('pstep') || 'design';
    /* 旧工位链接（studio/freeze/dispatch/audit/ai/contribute）在读取时就地归一，地址栏随后由 writeState 改写 */
    const old = OLD_STATIONS[station];
    if (old) {
      lens = old[0];
      station = old[1] || station;
      pstep = old[2] || 'design';
    }
    return { lens, view, station, rtab: q.get('rtab') || 'neuron', tm: q.get('tm') || '', pstep, csec: q.get('csec') || 'conn' };
  } catch {
    return { lens: 'spatial', view: 'neuron', station: 'probe', rtab: 'neuron', tm: '', pstep: 'design', csec: 'conn' };
  }
}

export function writeState(st) {
  try {
    const url = new URL(window.location.href);
    for (const k of ['lens', 'view', 'station', 'rtab', 'tm', 'pstep', 'csec', 'obj', 'atlasTab', 'atlasSel']) url.searchParams.delete(k);
    url.searchParams.set('lens', st.lens);
    if (st.lens === 'spatial') url.searchParams.set('view', st.view);
    if (st.lens === 'process') url.searchParams.set('station', st.station);
    /* 流水线步骤只在非首步时写进地址栏，避免默认链接变长 */
    if (st.lens === 'process' && st.station === 'pipeline' && st.pstep && st.pstep !== 'design') url.searchParams.set('pstep', st.pstep);
    /* 配置分段同理：首段不进地址栏 */
    if (st.lens === 'config' && st.csec && st.csec !== 'conn') url.searchParams.set('csec', st.csec);
    if (st.lens === 'route' && st.rtab && st.rtab !== 'neuron') url.searchParams.set('rtab', st.rtab);
    if (st.tm) url.searchParams.set('tm', st.tm);
    window.history.replaceState(null, '', url);
  } catch { /* 地址栏不可写时只保留页面状态 */ }
}

/* 各面板的跳转约定统一到 (lens, view/station)：
   onGo('process', 'pipeline') / onGo('audit') / onGo('spatial', null, null, {tm}) / onGo('spatial', null, undefined, {view:'atlas'})
   旧的 'atlas' 透镜（研究账本）不再存在，归到进展（route）；旧工位 id（studio/freeze/dispatch/audit/ai/contribute）走 OLD_STATIONS 别名。 */
export function resolveGo(prev, a, b, _c, extra = {}) {
  const old = OLD_STATIONS[a];
  if (old) return { ...prev, lens: old[0], station: old[1] || prev.station, pstep: old[2] || prev.pstep || 'design' };
  const station = STATIONS.find(s => s.id === a);
  if (station) return { ...prev, lens: 'process', station: a, pstep: a === 'pipeline' ? 'design' : (prev.pstep || 'design') };
  if (a === 'process') return { ...prev, lens: 'process', station: STATIONS.some(s => s.id === b) ? b : prev.station };
  if (a === 'spatial') return { ...prev, lens: 'spatial', view: OLD_VIEWS[extra.view] || extra.view || (extra.tm ? 'response' : prev.view), tm: extra.tm || prev.tm };
  if (a === 'atlas' || a === 'progress' || a === 'route' || a === 'home') return { ...prev, lens: 'route' };
  if (LENSES.some(l => l.id === a)) return { ...prev, lens: a };
  return prev;
}
