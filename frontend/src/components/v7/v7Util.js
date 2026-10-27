/* v7 客户端（图谱 / 3D 空间 / 进展）的共享函数（非组件）。数据来自 lang_atlas.v1 端点：
   /api/lang/tree、/api/lang/item/{id}、/api/model/skeleton、/api/space/overlay（设计见 design/client_v7_two_routes_atlas.md）。 */
import { CATEGORICAL } from '../app/rdc_fusion/Space3D.jsx';

export const PAGES = [
  { id: 'atlas', label: '图谱', desc: '按语言浏览：每个语言项在 LLM 内部对应的规则与研究状态' },
  { id: 'space', label: '3D 空间', desc: '在一个模型的多层结构上观察各语言项的编码' },
  { id: 'progress', label: '进展', desc: '研究进度、参与测试与观察已有运行' },
];

export const SPACE_VIEWS = [
  { id: 'layers', label: '层结构', desc: '默认：模型多层结构 + 所选语言项的逐层编码' },
  { id: 'response', label: '响应点云', desc: '单个模式族的样本点云与规律分析（模式族附加包）' },
  { id: 'units', label: '单元卡', desc: '原生 MLP 单元卡（单元卡附加包）' },
  { id: 'relations', label: '谱图关系', desc: '跨模式族的特征谱图与共享编码轴' },
];

/* 读出轴等级：符号 + 文字（不只靠颜色）。顺序量用单一色相由浅到深。 */
export const LEVELS = [
  { level: 0, sym: '○', label: '未研究', color: '#e2e8f0', ink: '#475569' },
  { level: 1, sym: '◔', label: '有数据', color: '#cde2fb', ink: '#184f95' },
  { level: 2, sym: '◑', label: '可读出', color: '#6da7ec', ink: '#ffffff' },
  { level: 3, sym: '◕', label: '留出复现', color: '#256abf', ink: '#ffffff' },
  { level: 4, sym: '●', label: '跨模型 / 跨族', color: '#0d366b', ink: '#ffffff' },
];
export const levelMeta = l => LEVELS[Math.max(0, Math.min(4, l || 0))];

export const MAX_COMPARE = 4;
export const compareColor = i => CATEGORICAL[i % CATEGORICAL.length];

export function shortModel(key) {
  return String(key || '').split('|')[0] || '—';
}

const KEYS = ['page', 'item', 'model', 'lang', 'view', 'cmp', 'type', 'tm'];

export function readState() {
  const st = { page: 'atlas', item: '', model: '', lang: '', view: 'layers', cmp: [], type: '', tm: '' };
  try {
    const q = new URLSearchParams(window.location.search);
    for (const k of KEYS) {
      const v = q.get(k);
      if (v == null) continue;
      st[k] = k === 'cmp' ? v.split(',').filter(Boolean).slice(0, MAX_COMPARE) : v;
    }
  } catch { /* 地址栏不可读：用默认状态 */ }
  if (!PAGES.some(p => p.id === st.page)) st.page = 'atlas';
  if (!SPACE_VIEWS.some(v => v.id === st.view)) st.view = 'layers';
  return st;
}

export function writeState(st) {
  try {
    const url = new URL(window.location.href);
    for (const k of KEYS) {
      const v = k === 'cmp' ? (st.cmp || []).join(',') : st[k];
      if (v) url.searchParams.set(k, v);
      else url.searchParams.delete(k);
    }
    window.history.replaceState(null, '', url);
  } catch { /* 地址栏不可写时只保留页面状态 */ }
}

/* 默认模型：在最多语言项上有 piece 的模型（同数时按名称），保证打开 3D 就有内容。 */
export function defaultModel(tree) {
  const models = tree?.models || [];
  if (!models.length) return '';
  const count = m => (tree.items || []).filter(it => it.status?.by_model?.[m] != null).length;
  return [...models].sort((a, b) => count(b) - count(a) || a.localeCompare(b))[0];
}

/* 语种筛选：选了语种时只留下在该语种上测过、或计划覆盖该语种的语言项。 */
export function matchLang(item, lang) {
  if (!lang) return true;
  return (item.status?.languages_tested || []).includes(lang) || (item.languages || []).includes(lang);
}

export function allLanguages(tree) {
  const s = new Set();
  for (const it of tree?.items || []) for (const l of [...(it.languages || []), ...(it.status?.languages_tested || [])]) s.add(l);
  return [...s].sort();
}

export const LANG_LABEL = { zh: '中文', en: 'English' };

/* 某语言项在某模型上的等级；没选模型时取全部模型中的最高等级。 */
export function itemLevel(item, model) {
  if (!model) return item.status?.read || 0;
  return item.status?.by_model?.[model] ?? 0;
}

export function segText(segments) {
  return (segments || []).map(([a, b]) => (a === b ? `L${a}` : `L${a}–L${b}`)).join('、') || '—';
}
