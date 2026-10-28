/* 研究工作台（v8）的导航配置与跳转归一（非组件）。URL 即状态：?lens= &view= &station= &rtab= &tm=（空间内部的
   语言项、块、单元等状态由 v8/v8Util.js 记录在同一个地址栏里，键名不重叠）。 */

export const LENSES = [
  { id: 'route', label: '进展', desc: '神经元逆向分析进展、语言图谱进展与证据门链' },          // id 沿用 route，旧链接 ?lens=route 照常可用
  { id: 'spatial', label: '空间', desc: 'LLM 多层结构：神经元逆向分析结果与语言图谱的有效神经元' },
  { id: 'process', label: '研发', desc: '题库、贡献算力、模板工作室、试跑与冻结、分发、审计、普查作业' },
];

/* 进展视图：与空间/研发同一层级的子导航（?rtab=），默认 neuron */
export const ROUTE_VIEWS = [
  { id: 'neuron', label: '神经元逆向分析', desc: '普查规模、逐层摘要与分型、解释检验、模板选择性（各项分开计数）' },
  { id: 'lang', label: '语言图谱', desc: '大类 × 等级覆盖、语言项 × 模型等级矩阵、研究缺口与证据门链' },
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

/* 研发工位：autotest 只在本机中心（local_tools）出现——公开中心没有被测模型；
   贡献算力（everywhere）在所有形态出现：桌面客户端控制本机节点，网页版只显示下载入口 */
export const STATIONS = [
  { id: 'catalog', label: '题库' },
  { id: 'contribute', label: '贡献算力', everywhere: true },
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
    return { lens, view, station: q.get('station') || 'catalog', rtab: q.get('rtab') === 'lang' ? 'lang' : 'neuron', tm: q.get('tm') || '' };
  } catch {
    return { lens: 'spatial', view: 'neuron', station: 'catalog', rtab: 'neuron', tm: '' };
  }
}

export function writeState(st) {
  try {
    const url = new URL(window.location.href);
    for (const k of ['lens', 'view', 'station', 'rtab', 'tm', 'obj', 'atlasTab', 'atlasSel']) url.searchParams.delete(k);
    url.searchParams.set('lens', st.lens);
    if (st.lens === 'spatial') url.searchParams.set('view', st.view);
    if (st.lens === 'process') url.searchParams.set('station', st.station);
    if (st.lens === 'route' && st.rtab === 'lang') url.searchParams.set('rtab', 'lang');
    if (st.tm) url.searchParams.set('tm', st.tm);
    window.history.replaceState(null, '', url);
  } catch { /* 地址栏不可写时只保留页面状态 */ }
}

/* 各面板的跳转约定统一到 (lens, view/station)：
   onGo('process', 'studio') / onGo('audit') / onGo('spatial', null, null, {tm}) / onGo('spatial', null, undefined, {view:'atlas'})
   旧的 'atlas' 透镜（研究账本）不再存在，归到进展（route）。 */
export function resolveGo(prev, a, b, _c, extra = {}) {
  const station = STATIONS.find(s => s.id === a);
  if (station) return { ...prev, lens: 'process', station: a };
  if (a === 'process') return { ...prev, lens: 'process', station: STATIONS.some(s => s.id === b) ? b : prev.station };
  if (a === 'spatial') return { ...prev, lens: 'spatial', view: OLD_VIEWS[extra.view] || extra.view || (extra.tm ? 'response' : prev.view), tm: extra.tm || prev.tm };
  if (a === 'atlas' || a === 'progress' || a === 'route' || a === 'home') return { ...prev, lens: 'route' };
  if (LENSES.some(l => l.id === a)) return { ...prev, lens: a };
  return prev;
}
