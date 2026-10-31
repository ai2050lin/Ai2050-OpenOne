/* 研究工作台（v9，以「一句话」为中心）的导航配置与跳转归一（非组件）。设计见 design/attention_neuron_plan_v1.md §4。
   URL 即状态：?lens= &sub= &step= &csec= &case= &cue= &fact= &tm=（空间组件内部的语言项、块、单元等状态由 v8/v8Util.js
   记录在同一个地址栏里，键名不重叠；v8Util 会删除旧开放平台的 tab 参数，所以这里用 sub 而不是 tab）。
   旧链接（v8 的 ?lens=route|spatial|process 与 ?view= ?station= ?rtab= ?pstep=）在读取时就地换算成新页面。 */

/* 一级界面：分隔线之上是五个研究页面；之下是参与与管理（贡献、管理向导）和形态相关的界面（配置、本机工具）。 */
export const LENSES = [
  { id: 'datasets', label: '数据集', desc: '各数据集与类别、模型正确率、行为门；点题目或句子进入「例句」' },
  { id: 'sentence', label: '例句', desc: '一句话：行为 → 注意力路由（直接作用 / 总效应）→ 激活的神经元 → 头电路保真度' },
  { id: 'modes', label: '模式与优先级', desc: '每种冲突的逐层赛跑曲线与优先级表：决出层、负责的头、能翻转胜负的线索' },
  { id: 'neurons', label: '神经元图谱', desc: '按层浏览 MLP 单元；单元卡四轴（可读 / 读出 / 控制 / 连接）与证据等级，按归因优先级排序' },
  { id: 'progress', label: '进展', desc: '只放三类数：各模式保真度、完成的完整案例数、未通过的检验' },
  { id: 'contribute', label: '贡献', desc: '贡献算力与我的上传' },
  { id: 'admin', label: '管理', admin: true, desc: '管理向导：建模板 → 冻结 → 分发 → 审计 → 普查 → 自动测试' },
  { id: 'config', label: '配置', desc: '当前连接的中心与运行形态、账号、被测模型与平台信息' },
];
export const DIVIDER_BEFORE = new Set(['contribute', 'config']);

/* 各页面的子视图（?sub=）；第一个是缺省。desktop：只在能执行测试的形态（ADMIN_LIVE）出现。 */
export const SUBS = {
  datasets: [
    { id: 'overview', label: '数据集总览', desc: '冲突材料、行为广度题库、最小对模板，各自的行为门与正确率' },
    { id: 'lang', label: '语言项', desc: '大类 → 语言族 → 语言项：编码层段、点云、有效神经元与证据' },
    { id: 'response', label: '响应点云', desc: '单个模式族：3D 响应空间、规律分析、合并统计' },
    { id: 'catalog', label: '题库', desc: '所有用户看到的同一份题目清单' },
    { id: 'local', label: '本地包', desc: '在浏览器内打开下载的研究包 / 谱图包（不上传）' },
  ],
  sentence: [
    { id: 'case', label: '案例例句', desc: '机制案例里的句子：行为、注意力路由、神经元、头电路保真度' },
    { id: 'live', label: '输入一句话', desc: '本机模型实时前向：逐层观察每个注意力头与 MLP 单元（只在本机中心可用）' },
  ],
  neurons: [
    { id: 'atlas', label: '按层浏览', desc: '选中块的内部结构、MLP 单元阵列与单元卡' },
    { id: 'priority', label: '归因优先级', desc: '按单元对答案对数几率差的直接作用排序；四轴与证据等级' },
    { id: 'probe', label: '本机测试', desc: '本机 GPU 对一个 MLP 单元跑 S1–S9；结果只存本机', desktop: true },
  ],
};
/* 谱图关系（跨模式族共享轴）不在子导航里：从语言项页的入口进入，旧链接 ?view=relations 照常可用 */
const HIDDEN_SUBS = { datasets: ['relations'] };

/* 管理向导的步骤（?step=）：前四步是模板流水线（TemplatePipeline），后两步是普查与自动测试 */
export const ADMIN_STEPS = [
  { id: 'design', label: '建模板', desc: '句式框架 × 填充词表 × 因子水平' },
  { id: 'freeze', label: '冻结', desc: '先试跑、再正式；冻结后不可修改' },
  { id: 'dispatch', label: '分发', desc: '分片覆盖、节点与租约' },
  { id: 'audit', label: '审计', desc: '自动检查 + 人工裁决' },
  { id: 'census', label: '普查', desc: '作业进度、逐分片状态、复算不一致队列' },
  { id: 'autotest', label: '自动测试', desc: '登记被测模型 → 勾选模式族 → 一键运行', localOnly: true },
];

/* 旧链接换算表：v8 的空间视图、进展、研发工位 → 新页面 [lens, sub, step] */
const OLD_VIEWS = {
  neuron: ['neurons', 'atlas'], units: ['neurons', 'atlas'], live: ['sentence', 'live'],
  lang: ['datasets', 'lang'], response: ['datasets', 'response'], relations: ['datasets', 'relations'], atlas: ['datasets', 'relations'],
  local: ['datasets', 'local'],
};
const OLD_STATIONS = {
  catalog: ['datasets', 'catalog'], probe: ['neurons', 'probe'],
  pipeline: ['admin', '', 'design'], studio: ['admin', '', 'design'], freeze: ['admin', '', 'freeze'],
  dispatch: ['admin', '', 'dispatch'], audit: ['admin', '', 'audit'], census: ['admin', '', 'census'],
  autotest: ['admin', '', 'autotest'], ai: ['config', '', '', 'model'], contribute: ['contribute', ''],
};

export function defaultSub(lens) {
  return SUBS[lens]?.[0]?.id || '';
}

function validSub(lens, sub) {
  return Boolean(SUBS[lens]?.some(s => s.id === sub) || HIDDEN_SUBS[lens]?.includes(sub));
}

/* 从查询参数得到页面状态（纯函数，便于测试）：新参数优先，其余按旧链接换算。 */
export function parseState(search) {
  const q = new URLSearchParams(search || '');
  const st = { lens: 'sentence', sub: '', step: 'design', csec: 'conn', tm: q.get('tm') || '',
               case: q.get('case') || '', cue: q.get('cue') || '', fact: q.get('fact') || '' };
  const lens = q.get('lens') || '';
  if (LENSES.some(l => l.id === lens) || lens === 'local') {
    st.lens = lens;
    st.sub = q.get('sub') || '';
    st.step = q.get('step') || q.get('pstep') || 'design';
    st.csec = q.get('csec') || 'conn';
  } else if (lens === 'route' || lens === 'home' || lens === 'atlas' || lens === 'progress') {
    st.lens = 'progress';
  } else if (lens === 'spatial' || (!lens && q.get('view'))) {
    const v = OLD_VIEWS[q.get('view') || 'neuron'] || OLD_VIEWS.neuron;
    [st.lens, st.sub] = v;
  } else if (lens === 'process') {
    const o = OLD_STATIONS[q.get('station') || 'catalog'] || OLD_STATIONS.catalog;
    [st.lens, st.sub] = o;
    if (o[2]) st.step = o[0] === 'admin' && q.get('pstep') && o[2] === 'design' ? q.get('pstep') : o[2];
    if (o[3]) st.csec = o[3];
  } else if (q.get('station')) {
    const o = OLD_STATIONS[q.get('station')];
    if (o) { [st.lens, st.sub] = o; if (o[2]) st.step = o[2]; if (o[3]) st.csec = o[3]; }
  }
  if (!validSub(st.lens, st.sub)) st.sub = defaultSub(st.lens);
  if (!ADMIN_STEPS.some(s => s.id === st.step)) st.step = 'design';
  return st;
}

export function readState() {
  try { return parseState(window.location.search); } catch { return parseState(''); }
}

const OWN_KEYS = ['lens', 'sub', 'step', 'csec', 'case', 'cue', 'fact', 'tm'];
const OLD_KEYS = ['view', 'station', 'rtab', 'pstep', 'obj', 'atlasTab', 'atlasSel'];

/* 写回地址栏：只写与缺省值不同的状态。 */
export function stateQuery(st, base = '') {
  const url = new URLSearchParams(base);
  for (const k of [...OWN_KEYS, ...OLD_KEYS]) url.delete(k);
  url.set('lens', st.lens);
  if (st.sub && st.sub !== defaultSub(st.lens)) url.set('sub', st.sub);
  if (st.lens === 'admin' && st.step && st.step !== 'design') url.set('step', st.step);
  if (st.lens === 'config' && st.csec && st.csec !== 'conn') url.set('csec', st.csec);
  for (const k of ['case', 'cue', 'fact', 'tm']) if (st[k]) url.set(k, st[k]);
  return url.toString();
}

export function writeState(st) {
  try {
    const url = new URL(window.location.href);
    url.search = stateQuery(st, url.search);
    window.history.replaceState(null, '', url);
  } catch { /* 地址栏不可写时只保留页面状态 */ }
}

/* 当前形态可见的一级界面与子视图：live = config.ADMIN_LIVE，admin = 管理员账号（本机开发不限）。 */
export function visibleLenses({ live, admin, localTools }) {
  const out = LENSES.filter(l => !l.admin || (live && admin));
  return localTools ? [...out, { id: 'local', label: '本机工具', desc: '读取本机模型权重的空间视图（只在本机中心可用）' }] : out;
}
export function visibleSubs(lens, { live }) {
  return (SUBS[lens] || []).filter(s => !s.desktop || live);
}
export function visibleSteps({ localTools }) {
  return ADMIN_STEPS.filter(s => !s.localOnly || localTools);
}

/* 各面板的跳转约定统一到 (lens, sub)：onGo('datasets', 'lang') / onGo('neurons', 'atlas') / onGo('admin', null, 'freeze')；
   旧面板的 onGo('process', 'pipeline') / onGo('audit') / onGo('spatial', null, null, {view, tm}) 经旧链接换算表进入新页面。 */
export function resolveGo(prev, a, b, c, extra = {}) {
  const st = OLD_STATIONS[a];
  if (st) return { ...prev, lens: st[0], sub: st[1] || defaultSub(st[0]), step: st[2] || prev.step, csec: st[3] || prev.csec };
  if (a === 'process') return resolveGo(prev, b || 'catalog', null, null, extra);
  if (a === 'spatial') {
    const v = OLD_VIEWS[extra.view] || (extra.tm ? OLD_VIEWS.response : OLD_VIEWS.neuron);
    return { ...prev, lens: v[0], sub: v[1], tm: extra.tm || prev.tm };
  }
  if (a === 'atlas' || a === 'route' || a === 'home') return { ...prev, lens: 'progress', sub: '' };
  if (LENSES.some(l => l.id === a) || a === 'local') {
    const sub = validSub(a, b) ? b : defaultSub(a);
    return { ...prev, lens: a, sub, step: a === 'admin' && c ? c : prev.step, ...extra };
  }
  return prev;
}
