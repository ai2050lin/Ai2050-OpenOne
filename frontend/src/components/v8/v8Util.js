/* v8 客户端（两条主线：① 语言图谱、② 神经元）的共享函数（非组件）。设计见 design/client_v8_mainline_scheme.md。
   数据：/api/lang/*、/api/model/skeleton（lang_atlas.v1）与 /api/neurons/tower|layer|unit（neuron_layers.v1、unit_card.v1）。 */
import { CATEGORICAL } from '../app/rdc_fusion/Space3D.jsx';

export { CATEGORICAL };

/* 网站顶部导航：首页 · 介绍 · 参与；语言图谱与神经元在研究工作台的「空间」里 */
export const PAGES = [
  { id: 'home', label: '首页', desc: '平台定位、研究工作台入口与参与' },
  { id: 'learn', label: '介绍', desc: '研究方法与概念：模式族、行为门、留出复现、普查与证据等级' },
  { id: 'join', label: '参与', desc: '用你的 GPU 跑一个分片：一键包、贡献榜' },
];
/* 不进主导航的页面：账号管理（管理员，从账号菜单进入） */
export const BRANCHES = [
  { id: 'accounts', label: '账号管理', desc: '管理本中心的全部账号' },
  { id: 'data', label: '数据管理', desc: '服务器上的数据与存储，以及每个账号上传了什么（只读）' },
];
/* 旧链接：开放平台的 ?tab= 与 v8 网站的 ?page=；语言图谱 / 神经元 / 进展等转到研究工作台 */
const OLD_TABS = { home: 'home', learn: 'learn', join: 'join' };
/* 研究工作台地址：网页与 vite dev 用 /workbench；Electron 安装包的 file:// 页面用 index.html?wb=1（main.jsx 据此进入工作台） */
export function workbenchHref(query = '') {
  const file = typeof window !== 'undefined' && window.location?.protocol === 'file:';
  if (file) return `${window.location.pathname}?wb=1${query ? `&${query}` : ''}`;
  return `/workbench${query ? `?${query}` : ''}`;
}
/* 网站页面地址（研究工作台里的链接用）：网页与 vite dev 用 /?page=；Electron 的 file:// 页面用同一文件、不带 wb=1 */
export function siteHref(page = 'home') {
  const file = typeof window !== 'undefined' && window.location?.protocol === 'file:';
  const q = page === 'home' ? '' : `?page=${page}`;
  return file ? `${window.location.pathname}${q}` : `/${q}`;
}
export const WORKBENCH_REDIRECT = {
  atlas: workbenchHref('lens=spatial&view=lang'), lang: workbenchHref('lens=spatial&view=lang'), neuron: workbenchHref('lens=spatial&view=neuron'),
  progress: workbenchHref('lens=route'), response: workbenchHref('lens=spatial&view=response'), relations: workbenchHref('lens=spatial&view=relations'),
};

/* 读出轴等级：符号 + 文字（不只靠颜色）。顺序量用单一色相由浅到深。 */
export const LEVELS = [
  { level: 0, sym: '○', label: '未研究', color: '#e2e8f0', ink: '#475569' },
  { level: 1, sym: '◔', label: '有数据', color: '#cde2fb', ink: '#184f95' },
  { level: 2, sym: '◑', label: '可读出', color: '#6da7ec', ink: '#ffffff' },
  { level: 3, sym: '◕', label: '留出复现', color: '#256abf', ink: '#ffffff' },
  { level: 4, sym: '●', label: '跨模型 / 跨族', color: '#0d366b', ink: '#ffffff' },
];
export const levelMeta = l => LEVELS[Math.max(0, Math.min(4, l || 0))];

export const READOUT_LABEL = { target: '目标词', cue: '线索词', final: '句末', probe_answer: '答案前' };
export const LANG_LABEL = { zh: '中文', en: 'English' };
export const TYPE_COLOR = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'];
export const REGIME_COLOR = ['#2a78d6', '#7a3fbf', '#eb6834', '#1baf7a'];
export const GRADE_COLOR = { 文本可复现: '#184f95', 只在语料中可识别: '#6da7ec', 未解释: '#94a3b8' };
export const EMPTY_CELL = '#eef2f7';

export function shortModel(key) {
  return String(key || '').split('|')[0] || '—';
}

/* 空间各 tab 的选中层分开记录：block = 神经元（块号），lblock / rlayer = 语言图谱 / 响应点云（hs 序号） */
const KEYS = ['page', 'type', 'item', 'model', 'lang', 'piece', 'block', 'unit', 'ov', 'color', 'sort', 'sel', 'ro', 'mod', 'lblock', 'rlayer'];
const NUM = new Set(['block', 'unit', 'lblock', 'rlayer']);

function defaults() {
  return { page: 'home', type: 'GR', item: '', model: '', lang: '', piece: '', block: null, unit: null,
           ov: 'sel', color: 'type', sort: '', sel: '', ro: 'final', mod: 'mlp', lblock: null, rlayer: null };
}

export function readState() {
  const st = defaults();
  try {
    const q = new URLSearchParams(window.location.search);
    if (!q.get('page') && OLD_TABS[q.get('tab')]) st.page = OLD_TABS[q.get('tab')];
    if (!q.get('page') && WORKBENCH_REDIRECT[q.get('tab')]) st.page = q.get('tab');
    for (const k of KEYS) {
      const v = q.get(k);
      if (v == null || v === '') continue;
      st[k] = NUM.has(k) ? (Number.isFinite(Number(v)) ? Number(v) : null) : v;
    }
  } catch { /* 地址栏不可读：用默认状态 */ }
  if (![...PAGES, ...BRANCHES].some(p => p.id === st.page) && !WORKBENCH_REDIRECT[st.page]) st.page = 'home';
  return st;
}

export function writeState(st) {
  try {
    const url = new URL(window.location.href);
    url.searchParams.delete('tab');                    // 旧开放平台参数，已换算成 page
    const d = defaults();
    for (const k of KEYS) {
      const v = st[k];
      if (v != null && v !== '' && v !== d[k]) url.searchParams.set(k, String(v));   // 只写与缺省值不同的状态
      else url.searchParams.delete(k);
    }
    window.history.replaceState(null, '', url);
  } catch { /* 地址栏不可写时只保留页面状态 */ }
}

/* 默认模型：在最多语言项上有拼图、且有普查的模型优先（神经元主线可用）。 */
export function defaultModel(tree, skeletons) {
  const models = tree?.models || [];
  if (!models.length) return '';
  const count = m => (tree.items || []).filter(it => it.status?.by_model?.[m] != null).length;
  const census = m => (skeletons?.[m]?.census_run ? 1 : 0);
  return [...models].sort((a, b) => census(b) - census(a) || count(b) - count(a) || a.localeCompare(b))[0];
}

export function matchLang(item, lang) {
  if (!lang) return true;
  return (item.status?.languages_tested || []).includes(lang) || (item.languages || []).includes(lang);
}

export function itemLevel(item, model) {
  if (!model) return item.status?.read || 0;
  return item.status?.by_model?.[model] ?? 0;
}

export function segText(segments) {
  return (segments || []).map(([a, b]) => (a === b ? `L${a}` : `L${a}–L${b}`)).join('、') || '—';
}

/* 语言项在当前模型上的代表拼图：自己的模式族优先（GR.Q → GR-Q），其次等级、峰值间隔。 */
export function pickPiece(pieces, item, model, wanted) {
  const ps = (pieces || []).filter(p => p.model === model);
  if (wanted) {
    const w = ps.find(p => p.id === wanted);
    if (w) return w;
  }
  const own = p => (p.family_id === String(item).replace('.', '-') ? 1 : 0);
  const peak = p => p.readouts?.[p.readout]?.peak?.sep || 0;
  return [...ps].sort((a, b) => own(b) - own(a) || b.level - a.level || peak(b) - peak(a))[0] || null;
}

export function pieceLabel(p) {
  return `${p.family_id}.${p.factor}`;
}

/* 顺序色阶（与 ui.jsx 的 SEQ_STOPS 一致的单一色相）；v ∈ [0,1]。 */
const SEQ = ['#f3f7fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
function hex(h) { return [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)); }
export function seq(v) {
  const t = Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0)) * (SEQ.length - 1);
  const i = Math.min(SEQ.length - 2, Math.floor(t));
  const a = hex(SEQ[i]), b = hex(SEQ[i + 1]), f = t - i;
  return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * f)).join(',')})`;
}

/* 层阵列的着色：返回 [颜色数组, 排序用的数值数组（越大越靠前）, 图例] */
export function unitColors(layer, mode, selRun, ro) {
  const n = layer.n_units, col = layer.columns;
  const colors = new Array(n).fill(EMPTY_CELL), score = new Array(n).fill(0);
  let legend = [];
  if (mode === 'type') {
    for (let u = 0; u < n; u += 1) { colors[u] = TYPE_COLOR[col.type[u]]; score[u] = 6 - col.type[u]; }
    legend = layer.type_names.map((t, k) => ({ color: TYPE_COLOR[k], label: t }));
  } else if (mode === 'dens') {
    const lg = v => (v > 0 ? Math.max(0, (Math.log10(v) + 4) / 4) : 0);            // 1e-4 … 1 → 0 … 1
    for (let u = 0; u < n; u += 1) { score[u] = col.dens[u]; colors[u] = seq(lg(col.dens[u])); }
    legend = [{ color: seq(0.05), label: '强激活率 ≤ 0.0001' }, { color: seq(0.5), label: '0.01' }, { color: seq(1), label: '1（对数色阶）' }];
  } else if (mode === 'regime') {
    for (let u = 0; u < n; u += 1) { colors[u] = REGIME_COLOR[col.regime[u]]; score[u] = 4 - col.regime[u]; }
    legend = layer.regime_names.map((r, k) => ({ color: REGIME_COLOR[k], label: `${r}（G${r[0]} V${r[1]}）主导` }));
  } else if (mode === 'expl') {
    for (const [u, g] of Object.entries(layer.explained || {})) { colors[u] = GRADE_COLOR[g] || EMPTY_CELL; score[u] = g === '文本可复现' ? 3 : g === '未解释' ? 1 : 2; }
    legend = [...Object.entries(GRADE_COLOR).map(([g, c]) => ({ color: c, label: g })), { color: EMPTY_CELL, label: '未做解释检验' }];
  } else if (mode === 'sel') {
    const s = (layer.selectivity || []).find(x => x.file === selRun) || (layer.selectivity || [])[0];
    for (const [u, d, pos] of (s?.top?.[ro] || [])) {
      colors[u] = pos === s.hi ? CATEGORICAL[0] : CATEGORICAL[1];
      score[u] = Math.abs(d);
    }
    legend = s ? [{ color: CATEGORICAL[0], label: `偏向 ${s.hi}` }, { color: CATEGORICAL[1], label: `偏向 ${s.lo}` },
                  { color: EMPTY_CELL, label: `其他（只标前列；该层共 ${s.n_selected?.[ro] ?? 0} 个入选）` }] : [];
  }
  return [colors, score, legend];
}

/* 层塔叠加量：按块的数值数组；没有数据时为 null（只画结构）。 */
export const OVERLAYS = [['sel', '选择性'], ['expl', '已解释'], ['t1', 'T1 检测器数'], ['dens', '强激活率中位数'], ['none', '无']];

export function towerValues(tower, ov, selRun, ro) {
  const n = tower.n_layers;
  if (ov === 'sel') {
    const s = tower.selectivity.find(x => x.file === selRun) || tower.selectivity[0];
    return s ? Array.from({ length: n }, (_, b) => (s.by_readout?.[ro] || s.by_readout?.[Object.keys(s.by_readout)[0]] || [])[b] || 0) : null;
  }
  if (ov === 'expl') return tower.blocks.map(b => Object.values(b.explained || {}).reduce((s, v) => s + v, 0));
  if (ov === 't1') return tower.blocks.map(b => (b.types ? b.types[0] : 0));
  if (ov === 'dens') return tower.blocks.map(b => b.dens_median || 0);
  return null;
}

/* 当前语言项：URL 里的 item，否则当前大类里第一个有数据的语言项（语言图谱面板与层塔叠加共用） */
export function pickItem(tree, st, model) {
  if (st.item && tree.items.some(i => i.id === st.item)) return st.item;
  return (tree.items.find(i => i.type === st.type && itemLevel(i, model) > 0) || tree.items.find(i => i.type === st.type) || {}).id;
}

/* 层塔的层序号：hs[0] = 嵌入层，块 b 的输出 = hs[b+1] */
export const hsLabel = hs => (hs === 0 ? '嵌入层 hs[0]' : `块 ${hs - 1}（输出 hs[${hs}]）`);
