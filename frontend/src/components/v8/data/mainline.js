/* Immutable release fallback. Live pages read the shared server snapshot. */
import release from '../../../../../shared/dist_protocol/mainline_release.json';
export const { BENCH, CAPABILITIES, CAP_LEVELS, CLAIMS, CLAIM_LIFE, FINDINGS, LIFECYCLE, LOCKBOXES, PAIRS, RUNS, SCORES, SLICES, SPLITS, STAGES, TESTS, METHOD_RESULTS } = release.state.view;

export const pct = (done, total) => {
  if (!Number.isFinite(done) || !Number.isFinite(total) || total <= 0) return null;
  const value = 100 * done / total;
  return value === 0 || value === 100 ? value.toFixed(0) : value < 1 ? value.toFixed(2) : value.toFixed(0);
};

export function stageProgress(stages = STAGES) {
  const doneStages = stages.filter(stage => stage.done.length === stage.gates.length).length;
  const next = stages.find(stage => stage.done.length < stage.gates.length) || null;
  return { done: doneStages, total: stages.length, next };
}

/* AGENTS 第 13 节功效公式，与 tools/power_analysis.py 一致（McNemar 配对，p_d 缺省 0.10）。 */
function invNorm(p) {
  // Acklam 有理逼近，误差 < 1.2e-9，足够计算所需 n。
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425, hi = 1 - lo;
  if (p < lo) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > hi) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
function normCdf(x) {
  // Abramowitz–Stegun 7.1.26 的误差函数逼近。
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x / 2);
  return x >= 0 ? (1 + y) / 2 : (1 - y) / 2;
}
export function nRequired(delta, pD = 0.1, alpha = 0.05, power = 0.8) {
  if (!(delta > 0 && delta < 1)) return null;
  const z = invNorm(1 - alpha / 2) + invNorm(power);
  return Math.ceil(z * z * pD / (delta * delta));
}
export function achievedPower(n, delta, pD = 0.1, alpha = 0.05) {
  if (!(n > 0) || !(delta > 0 && delta < 1)) return null;
  return normCdf(Math.abs(delta) * Math.sqrt(n / pD) - invNorm(1 - alpha / 2));
}
