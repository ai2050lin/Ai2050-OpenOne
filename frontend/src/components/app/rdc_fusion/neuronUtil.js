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
  read: '读出',
  control: '控制',
  explain: '解释',
};

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
  if (!text || text === '未检验') return 'none';
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
