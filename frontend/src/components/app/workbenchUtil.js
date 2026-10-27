/* 研究工作台（v8）的导航配置与跳转归一（非组件）。URL 即状态：?lens= &view= &station= &tm=（空间内部的
   语言项、块、单元等状态由 v8/v8Util.js 记录在同一个地址栏里，键名不重叠）。 */

export const LENSES = [
  { id: 'spatial', label: '空间', desc: 'LLM 多层结构：神经元逆向分析结果与语言图谱的有效神经元' },
  { id: 'route', label: '路线', desc: '神经元逆向分析进展、语言图谱进展与证据门链' },
  { id: 'process', label: '研发', desc: '题库、模板工作室、试跑与冻结、分发、审计、普查作业' },
];

/* 空间视图：左侧都常驻 LLM 多层结构（层塔） */
export const SPATIAL_VIEWS = [
  { id: 'neuron', label: '神经元', desc: '选中块的内部结构、9,728 个 MLP 单元阵列与单元卡' },
  { id: 'lang', label: '语言图谱', desc: '大类 → 语言族 → 语言项：编码层段、点云、有效神经元与证据' },
  { id: 'response', label: '响应点云', desc: '单个模式族：3D 响应空间、规律分析、合并统计' },
  { id: 'relations', label: '谱图关系', desc: '跨模式族的特征、共享编码轴、概念假设' },
  { id: 'local', label: '本地包', desc: '在浏览器内打开下载的研究包 / 谱图包（不上传）' },
];
const OLD_VIEWS = { atlas: 'relations', units: 'neuron' };

/* 研发工位：autotest 只在本机中心（local_tools）出现——公开中心没有被测模型 */
export const STATIONS = [
  { id: 'catalog', label: '题库' },
  { id: 'studio', label: '模板工作室' },
  { id: 'freeze', label: '试跑与冻结' },
  { id: 'dispatch', label: '分发与节点' },
  { id: 'audit', label: '审计队列' },
  { id: 'census', label: '普查作业' },
  { id: 'autotest', label: '自动测试', localOnly: true },
];

export function readState() {
  try {
    const q = new URLSearchParams(window.location.search);
    const lens = q.get('lens') === 'home' ? 'route' : q.get('lens') || 'spatial';
    const view = OLD_VIEWS[q.get('view')] || q.get('view') || 'neuron';
    return { lens, view, station: q.get('station') || 'catalog', tm: q.get('tm') || '' };
  } catch {
    return { lens: 'spatial', view: 'neuron', station: 'catalog', tm: '' };
  }
}

export function writeState(st) {
  try {
    const url = new URL(window.location.href);
    for (const k of ['lens', 'view', 'station', 'tm', 'obj', 'atlasTab', 'atlasSel']) url.searchParams.delete(k);
    url.searchParams.set('lens', st.lens);
    if (st.lens === 'spatial') url.searchParams.set('view', st.view);
    if (st.lens === 'process') url.searchParams.set('station', st.station);
    if (st.tm) url.searchParams.set('tm', st.tm);
    window.history.replaceState(null, '', url);
  } catch { /* 地址栏不可写时只保留页面状态 */ }
}

/* 各面板的跳转约定统一到 (lens, view/station)：
   onGo('process', 'studio') / onGo('audit') / onGo('spatial', null, null, {tm}) / onGo('spatial', null, undefined, {view:'atlas'})
   旧的 'atlas' 透镜（研究账本）不再存在，归到路线。 */
export function resolveGo(prev, a, b, _c, extra = {}) {
  const station = STATIONS.find(s => s.id === a);
  if (station) return { ...prev, lens: 'process', station: a };
  if (a === 'process') return { ...prev, lens: 'process', station: STATIONS.some(s => s.id === b) ? b : prev.station };
  if (a === 'spatial') return { ...prev, lens: 'spatial', view: OLD_VIEWS[extra.view] || extra.view || (extra.tm ? 'response' : prev.view), tm: extra.tm || prev.tm };
  if (a === 'atlas' || a === 'progress' || a === 'route' || a === 'home') return { ...prev, lens: 'route' };
  if (LENSES.some(l => l.id === a)) return { ...prev, lens: a };
  return prev;
}
