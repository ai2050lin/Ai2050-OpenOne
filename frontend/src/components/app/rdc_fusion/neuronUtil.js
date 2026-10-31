/* 原生 MLP 神经元视图的共享函数与常量（非组件）。数据全部来自 /api/neurons/*，这里只做格式化。 */

export const REGIMES = ['++', '--', '+-', '-+'];

/* 符号区：G、V 的符号组合。A = silu(G)·V；−×− 的输出为小幅正值（门控关闭时的回声）。 */
export const REGIME_META = {
  '++': { label: '+×+ 门控开·值正', color: '#2f6fb3' },
  '--': { label: '−×− 门控关·值负（小幅正输出）', color: '#9db7d6' },
  '+-': { label: '+×− 门控开·值负', color: '#c0562e' },
  '-+': { label: '−×+ 门控关·值正（小幅负输出）', color: '#e5b29b' },
};

export const AXIS_META = {
  read: '可读',
  readout: '读出',
  control: '控制',
  connect: '连接',
  explain: '解释',
};

/* 单元卡四轴（可读 / 读出 / 控制 / 连接）+ 解释轴：可读、控制、解释来自服务端 evidence_axes（模板选择性、干预与归因、解释检验）；
   读出 = 直接通路读出 token 是否有记录（近似，不含后续层）；连接 = 跨层上下游电路，单元卡里没有，按需在块视图里按权重计算。
   只描述做过哪些检验，不是证据等级。 */
export function unitAxes(card) {
  const ax = card?.evidence_axes || {};
  const ro = card?.census?.readout;
  const hasReadout = Boolean(ro && ((ro.promotes || []).length || (ro.suppresses || []).length));
  return [
    ['read', ax.read || '未检验'],
    ['readout', hasReadout ? '直接通路读出有记录（近似）' : '未检验'],
    ['control', ax.control || '未检验'],
    ['connect', '未检验（块视图里可按权重算上下游电路假设）'],
    ['explain', ax.explain || '未检验'],
  ];
}

export function unitKey(block, unit) {
  return `L${block}.U${unit}`;
}

export function parseUnitKey(text) {
  const m = /^\s*[LB]?(\d+)\s*[.:,\s]\s*U?(\d+)\s*$/i.exec(text || '');
  return m ? { block: Number(m[1]), unit: Number(m[2]) } : null;
}

export function fmtNum(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.01 || a >= 1e4)) return v.toExponential(1);
  return v.toFixed(digits);
}

export function fmtRate(r) {
  if (r == null) return '—';
  if (r === 0) return '0';
  return r < 1e-3 ? r.toExponential(1) : `${(100 * r).toFixed(2)}%`;
}

/* 证据轴状态 → 视觉档：established / partial / none */
export function axisTone(text) {
  if (!text || text.startsWith('未检验')) return 'none';
  if (/未通过|未达|未运行|未解释/.test(text)) return 'partial';
  return 'established';
}

/* 直方图箱：bin_lower 为每箱下边界（负侧为上边界的相反数），返回用于绘制的 symlog 横坐标位置 0..1。 */
export function histColumns(hist) {
  const n = hist.bin_lower.length;
  return hist.bin_lower.map((lo, i) => ({ i, lo, x: i / n, w: 1 / n }));
}

/* 解释检验等级（EXPL-v1 / v2）：顺序量，单一色相由深到浅；未解释用灰色并附原因。 */
export const GRADES = ['文本可复现', '只在语料中可识别', '未解释'];
export const GRADE_META = {
  '文本可复现': { color: '#1d4f86', bg: '#dbe8f6', mark: '●' },
  '只在语料中可识别': { color: '#2f6fb3', bg: '#eaf2fb', mark: '◐' },
  '未解释': { color: '#6b7480', bg: '#f1f3f5', mark: '○' },
};

/* 层空间（layer_space.v1）3D 视图：神经元类型的类别色（参考色板前 6 槽，固定顺序，经 validate_palette 检查：
   CVD 相邻 ΔE ≥ 9.1；三个颜色对浅色底对比 < 3:1，所以图例始终带文字、点击即有单元卡与列表）。 */
export const SPACE_TYPES = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'];
export const SPACE_TYPE_COLOR = { T1: '#2a78d6', T2: '#eb6834', T3: '#1baf7a', T4: '#eda100', T5: '#e87ba4', T6: '#008300' };
export const SPACE_TOPN = [
  { v: 0, label: '全部' }, { v: 2000, label: '因果前 2000' }, { v: 500, label: '因果前 500' }, { v: 100, label: '因果前 100' },
];
